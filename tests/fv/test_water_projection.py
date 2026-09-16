"""Momentum-only parcel forcing must not create passive humidity structure."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind import (
    InflowPlane,
    StaggeredVelocity,
    build_pressure_poisson,
    divergence,
    initial_atmospheric_solution,
)
from jaxwind.domain import UniformGrid
from jaxwind.moist_abl import initialize_moisture, transport_moisture
from jaxwind.numerics.poisson import project
from jaxwind.open_boundary import enforce_open_velocity
from jaxwind.physics.moisture import MoistureConfig
from jaxwind.water_parcels import (
    InertialMoistAtmosphericSolution,
    WaterParcelSource,
    build_inertial_water_step,
    initial_water_parcels,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("source_x", [0.0, 0.2])
def test_momentum_feedback_preserves_uniform_passive_vapor_after_projection(source_x):
    config = MoistureConfig()
    grid = UniformGrid(8, 8, 8, 0.4, 0.4, 0.4)
    velocity = StaggeredVelocity(
        jnp.full((8, 8, 9), 3.0), jnp.zeros((8, 9, 8)), jnp.zeros((9, 8, 8))
    )
    flow, _ = initialize_moisture(
        initial_atmospheric_solution(grid, velocity, dtype="float64"),
        300.0,
        0.2,
        config,
    )
    q = 0.005
    flow = flow._replace(moisture=flow.moisture._replace(vapor=jnp.full((8, 8, 8), q)))
    source = WaterParcelSource(
        center=(source_x, 0.2, 0.2),
        radius=0.002,
        speed=22.0,
        temperature=300.0,
        mass_flow=0.2,
        half_angle_degrees=18.0,
        diameter_scale=369e-6,
        diameter_spread=3.67,
        diameter_minimum=74e-6,
        diameter_maximum=518e-6,
        count_per_step=8,
        capacity=32,
        ramp_time=0.0,
    )
    inflow = InflowPlane(*(v[..., 0] for v in velocity), jnp.zeros((8, 8)))
    initial = InertialMoistAtmosphericSolution(
        *flow, initial_water_parcels(source, "float64")
    )
    poisson = build_pressure_poisson(
        grid, backend="gmg", periodic_x=False, periodic_y=False, dtype="float64"
    )

    def passive_step(state, dt, inflow):
        return state._replace(
            moisture=transport_moisture(
                state.moisture, state.velocity, grid, dt, jnp.full((8, 8), q), 0.0
            )
        )

    corrected = build_inertial_water_step(
        passive_step,
        grid,
        source,
        config,
        300.0,
        thermal_exchange=False,
        project_velocity=lambda v, dt, plane: project(
            enforce_open_velocity(v, plane, grid), poisson, dt
        )[0],
    )
    legacy = build_inertial_water_step(
        passive_step,
        grid,
        source,
        config,
        300.0,
        thermal_exchange=False,
    )
    after = jax.jit(corrected)(initial, 0.001, inflow)
    unprojected = jax.jit(legacy)(initial, 0.001, inflow)
    assert float(jnp.max(jnp.abs(divergence(after.velocity, grid)))) < 1e-9
    np.testing.assert_allclose(after.moisture.vapor, q, atol=1e-14, rtol=0)
    assert float(jnp.max(jnp.abs(unprojected.moisture.vapor - q))) > 1e-6
    np.testing.assert_allclose(after.velocity.x[..., 0], 3.0, atol=1e-14)
    assert float(after.parcels.evaporated_mass) == 0.0
    assert float(after.parcels.gas_sensible_energy_loss) == 0.0
    # Projection acts only on the carrier after the otherwise identical exchange.
    np.testing.assert_array_equal(after.parcels.velocity, unprojected.parcels.velocity)
    np.testing.assert_array_equal(after.parcels.mass, unprojected.parcels.mass)
