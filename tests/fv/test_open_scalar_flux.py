"""Physical end cells must retain sources and exchange only boundary fluxes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind import (
    FREE_SLIP,
    OPEN,
    Boundaries,
    FlowModel,
    InflowPlane,
    StaggeredVelocity,
    Wall,
    build_open_atmospheric_step,
    build_pressure_poisson,
    initial_atmospheric_solution,
)
from jaxwind.domain import UniformGrid
from jaxwind.scalar import PassiveScalar, open_scalar_tendency

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("speed", [2.0, -2.0])
@pytest.mark.parametrize("advection", ["upwind", "central"])
def test_open_scalar_integral_equals_independent_boundary_flux(speed, advection):
    grid = UniformGrid(8, 4, 4, 2.0, 1.0, 1.0)
    field = jnp.arange(128, dtype=jnp.float64).reshape(4, 4, 8) / 128
    velocity = StaggeredVelocity(
        jnp.full((4, 4, 9), speed), jnp.zeros((4, 5, 8)), jnp.zeros((5, 4, 8))
    )
    ambient = jnp.full((4, 4), 3.0)
    rhs = open_scalar_tendency(
        field,
        velocity,
        grid,
        PassiveScalar(diffusivity=0.1, advection_scheme=advection),
        ambient,
    )
    left = ambient if speed > 0 else field[..., 0]
    right = field[..., -1] if speed > 0 else ambient
    expected = speed * float(jnp.sum(left - right)) / 16
    np.testing.assert_allclose(jnp.sum(rhs * grid.cell_volumes), expected, atol=1e-13)


@pytest.mark.parametrize("scheme", ["ab2", "fast-rk3", "rk3"])
def test_sources_in_physical_end_cells_survive_carrier_step(scheme):
    grid = UniformGrid(8, 4, 4, 2.0, 1.0, 1.0)
    velocity = StaggeredVelocity(
        jnp.zeros((4, 4, 9)), jnp.zeros((4, 5, 8)), jnp.zeros((5, 4, 8))
    )
    plane = InflowPlane(*(a[..., 0] for a in velocity), jnp.zeros((4, 4)))
    boundaries = Boundaries(
        Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=FREE_SLIP
    )
    poisson = build_pressure_poisson(
        grid, backend="gmg", periodic_x=False, periodic_y=False, dtype="float64"
    )
    initial = initial_atmospheric_solution(grid, velocity, dtype="float64")
    # A previously applied parcel heat sink plus a sustained source. Neither
    # belongs to a ghost cell and neither should vanish on boundary enforcement.
    sink = jnp.zeros((4, 4, 8)).at[..., 0].set(-3.0).at[..., -1].set(-7.0)
    initial = initial._replace(scalar=sink)
    step = build_open_atmospheric_step(
        grid,
        boundaries,
        poisson,
        FlowModel(outlet_backflow="energy" if scheme == "rk3" else "none"),
        PassiveScalar(advection_scheme="upwind"),
        scalar_source=lambda t: sink,
        scheme=scheme,
        scalar_boundary="flux",
    )
    result = jax.jit(step)(initial, 0.01, plane)
    np.testing.assert_allclose(result.scalar, 1.01 * sink, atol=1e-14)
    np.testing.assert_allclose(
        jnp.sum((result.scalar - initial.scalar) * grid.cell_volumes),
        0.01 * jnp.sum(sink * grid.cell_volumes),
        atol=1e-14,
    )


@pytest.mark.parametrize("scheme", ["ab2", "fast-rk3", "rk3"])
def test_external_scalar_transport_is_not_applied_twice_or_from_cached_rhs(scheme):
    grid = UniformGrid(8, 4, 4, 2.0, 1.0, 1.0)
    velocity = StaggeredVelocity(
        jnp.ones((4, 4, 9)), jnp.zeros((4, 5, 8)), jnp.zeros((5, 4, 8))
    )
    plane = InflowPlane(*(a[..., 0] for a in velocity), jnp.zeros((4, 4)))
    boundaries = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP),
                            streamwise=OPEN, spanwise=FREE_SLIP)
    poisson = build_pressure_poisson(grid, backend="gmg", periodic_x=False,
                                     periodic_y=False, dtype="float64")
    initial = initial_atmospheric_solution(grid, velocity, dtype="float64")
    initial = initial._replace(
        scalar=jnp.arange(128, dtype=jnp.float64).reshape(4, 4, 8),
        scalar_tendency=jnp.ones((4, 4, 8)), step=jnp.asarray(1),
    )
    step = build_open_atmospheric_step(
        grid, boundaries, poisson, FlowModel(), PassiveScalar(),
        scheme=scheme, scalar_boundary="flux", transport_scalar=False,
    )
    result = jax.jit(step)(initial, 0.01, plane)
    np.testing.assert_array_equal(result.scalar, initial.scalar)
