"""Independent accuracy, conservation, and positivity checks for scalar transport."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.domain import UniformGrid
from jaxwind.scalar_transport import transport_scalars
from jaxwind.state import StaggeredVelocity

jax.config.update("jax_enable_x64", True)


def periodic_velocity(grid, u=1.0, v=0.0):
    shape = (grid.nz, grid.ny, grid.nx)
    return StaggeredVelocity(
        jnp.full(shape, u),
        jnp.full(shape, v),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx)),
    )


@pytest.mark.parametrize("scheme", ["upwind", "muscl-mc"])
def test_multidimensional_discontinuous_transport_is_positive_bounded_and_conservative(
    scheme,
):
    grid = UniformGrid(20, 16, 4, 1.0, 1.0, 0.2)
    _z, y, x = np.indices((4, 16, 20))
    patch = ((x > 5) & (x < 13) & (y > 3) & (y < 11)).astype(float)
    fields = jnp.asarray(np.stack((patch, 1 - patch, np.zeros_like(patch))))
    velocity = periodic_velocity(grid, 1.0, -0.6)
    ambient = jnp.zeros((3, 4, 16))
    # Many subcycles: dt exceeds the positivity limit, with nonzero diffusion.
    result = jax.jit(
        lambda q: transport_scalars(
            q, velocity, grid, 0.4, ambient, 0.001, scheme=scheme
        )
    )(fields)
    assert float(jnp.min(result)) >= -1e-14
    assert float(jnp.max(result)) <= 1 + 1e-14
    np.testing.assert_allclose(
        jnp.sum(result, axis=(1, 2, 3)),
        jnp.sum(fields, axis=(1, 2, 3)),
        atol=3e-11,
        rtol=0,
    )
    np.testing.assert_allclose(result[0] + result[1], 1, atol=1e-14)
    np.testing.assert_array_equal(result[2], 0)


def test_smooth_translation_is_second_order_and_less_diffusive_than_upwind():
    errors = {}
    for scheme in ("upwind", "muscl-mc"):
        errors[scheme] = []
        for n in (32, 64, 128):
            grid = UniformGrid(n, 2, 2, 1.0, 1.0, 1.0)
            x = (np.arange(n) + 0.5) / n
            initial = 1 + 0.5 * np.sinc(1 / n) * np.sin(2 * np.pi * x)
            exact = 1 + 0.5 * np.sinc(1 / n) * np.sin(2 * np.pi * (x - 0.25))
            field = jnp.broadcast_to(jnp.asarray(initial), (1, 2, 2, n))
            velocity = periodic_velocity(grid)
            result = jax.jit(
                lambda q, velocity=velocity, grid=grid, scheme=scheme: (
                    transport_scalars(
                        q,
                        velocity,
                        grid,
                        0.25,
                        jnp.zeros((1, 2, 2)),
                        0.0,
                        scheme=scheme,
                    )
                )
            )(field)
            errors[scheme].append(
                float(np.mean(np.abs(np.asarray(result[0, 0, 0]) - exact)))
            )
    assert errors["muscl-mc"][1] / errors["muscl-mc"][2] > 3.0
    assert errors["muscl-mc"][-1] < 0.1 * errors["upwind"][-1]


@pytest.mark.parametrize("speed", [-1.2, 1.2])
@pytest.mark.parametrize("scheme", ["upwind", "muscl-mc"])
def test_open_boundary_conservation_matches_ssprk_flux_quadrature(speed, scheme):
    grid = UniformGrid(16, 3, 2, 1.6, 0.6, 0.4)
    velocity = StaggeredVelocity(
        jnp.full((2, 3, 17), speed), jnp.zeros((2, 4, 16)), jnp.zeros((3, 3, 16))
    )
    # Zero slopes over all cells reachable from the outflow within three stages.
    x = np.r_[np.full(5, 0.2), np.linspace(0.2, 0.7, 6), np.full(5, 0.7)]
    q = jnp.broadcast_to(jnp.asarray(x), (1, 2, 3, 16))
    ambient = jnp.full((1, 2, 3), 0.5)
    dt = 0.01
    result = jax.jit(
        lambda q: transport_scalars(q, velocity, grid, dt, ambient, 0.01, scheme=scheme)
    )(q)
    incoming, outgoing = (0.5, 0.7) if speed > 0 else (0.2, 0.5)
    expected = dt * speed * grid.ly * grid.lz * (incoming - outgoing)
    np.testing.assert_allclose(
        jnp.sum((result - q) * grid.cell_volumes), expected, atol=2e-16, rtol=1e-12
    )
    assert float(jnp.min(result)) >= 0.2 - 1e-14
    assert float(jnp.max(result)) <= 0.7 + 1e-14


def test_closed_variable_diffusion_preserves_mass_and_nonnegativity():
    grid = UniformGrid(8, 6, 4, 0.8, 0.6, 0.4)
    rng = np.random.default_rng(92)
    fields = jnp.asarray(rng.random((2, 4, 6, 8)))
    velocity = StaggeredVelocity(
        jnp.zeros((4, 6, 9)), jnp.zeros((4, 7, 8)), jnp.zeros((5, 6, 8))
    )
    diffusivity = jnp.asarray(rng.random((4, 6, 8)) * 0.02)
    result = jax.jit(
        lambda q: transport_scalars(
            q, velocity, grid, 1.0, jnp.zeros((2, 4, 6)), diffusivity
        )
    )(fields)
    assert float(jnp.min(result)) >= 0
    assert float(jnp.max(result)) <= float(jnp.max(fields))
    np.testing.assert_allclose(
        jnp.sum(result, axis=(1, 2, 3)),
        jnp.sum(fields, axis=(1, 2, 3)),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("scheme", ["upwind", "muscl-mc"])
def test_split_moist_transport_conserves_total_water_and_enthalpy(scheme):
    from jaxwind import FREE_SLIP, OPEN, Boundaries, FlowModel, InflowPlane, Wall
    from jaxwind.abl import initial_atmospheric_solution
    from jaxwind.moist_abl import build_moist_atmospheric_step, initialize_moisture
    from jaxwind.physics.moisture import MoistureConfig
    from jaxwind.scalar import PassiveScalar

    grid = UniformGrid(8, 8, 4, 1.0, 1.0, 0.5)
    config = MoistureConfig()
    velocity = StaggeredVelocity(
        jnp.zeros((4, 8, 9)), jnp.full((4, 8, 8), 1.0), jnp.zeros((5, 8, 8))
    )
    flow = initial_atmospheric_solution(grid, velocity, dtype="float64")
    rng = np.random.default_rng(7)
    flow = flow._replace(scalar=jnp.asarray(-2 * rng.random((4, 8, 8))))
    initial, ambient = initialize_moisture(flow, 300.0, 0.2, config)
    boundaries = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN)
    step = build_moist_atmospheric_step(
        lambda flow, dt, inflow, offset: flow,
        grid,
        boundaries,
        FlowModel(),
        PassiveScalar(),
        config,
        300.0,
        300.0,
        ambient,
        scalar_transport_scheme=scheme,
    )
    inflow = InflowPlane(*(v[..., 0] for v in velocity), jnp.zeros((4, 8)))
    result = jax.jit(step)(initial, 0.2, inflow)
    assert np.all(np.asarray(result.moisture) >= 0)
    assert np.all(np.isfinite(result.scalar))
    initial_energy = (
        config.dry_air_heat_capacity * (initial.scalar + 300)
        + config.water_vapor_latent_heat * initial.moisture.vapor
    )
    final_energy = (
        config.dry_air_heat_capacity * (result.scalar + 300)
        + config.water_vapor_latent_heat * result.moisture.vapor
    )
    np.testing.assert_allclose(
        jnp.sum(final_energy), jnp.sum(initial_energy), rtol=2e-14
    )
    np.testing.assert_allclose(
        jnp.sum(jnp.stack(result.moisture[:4])),
        jnp.sum(jnp.stack(initial.moisture[:4])),
        rtol=2e-14,
    )
