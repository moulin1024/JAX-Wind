"""Independent inventory and contact checks for the dual momentum predictor."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.domain import UniformGrid
from jaxwind.numerics.discretization import divergence
from jaxwind.numerics.poisson import build_pressure_poisson
from jaxwind.physics.moisture import MoistureConfig, WaterDropletProperties
from jaxwind.spray_core import WaterCoreBins, advance_water_core
from jaxwind.spray_coupling import GasInventory
from jaxwind.spray_low_mach import build_moist_gas_transport, moist_gas_from_primitive
from jaxwind.spray_momentum import build_momentum_transport
from jaxwind.state import StaggeredVelocity

jax.config.update("jax_enable_x64", True)


def fixture(px=True, py=True):
    grid = UniformGrid(8, 6, 4, 1.0, 0.75, 0.5)
    shape = (grid.nz, grid.ny, grid.nx)
    rng = np.random.default_rng(901)
    rho = jnp.asarray(1 + 0.3 * rng.random(shape))
    velocity = StaggeredVelocity(
        jnp.asarray(rng.normal(size=(*shape[:2], grid.nx if px else grid.nx + 1))),
        jnp.asarray(
            rng.normal(size=(grid.nz, grid.ny if py else grid.ny + 1, grid.nx))
        ),
        jnp.asarray(rng.normal(size=(grid.nz + 1, grid.ny, grid.nx))),
    )
    velocity = velocity._replace(z=velocity.z.at[0].set(0).at[-1].set(0))
    if not py:
        velocity = velocity._replace(y=velocity.y.at[:, 0].set(0).at[:, -1].set(0))
    flow = jax.tree.map(lambda u: 0.2 * u, velocity)
    return grid, rho, velocity, flow, rng


def dual_geometry(grid, component, px, py):
    # Independent NumPy quadrature of half-sized endpoint dual cells.
    widths = [
        np.full(grid.nz, grid.dz),
        np.full(grid.ny, grid.dy),
        np.full(grid.nx, grid.dx),
    ]
    periodic = (False, py, px)
    if not periodic[component]:
        h = widths[component][0]
        widths[component] = np.r_[h / 2, np.full(len(widths[component]) - 1, h), h / 2]
    volumes = (
        widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]
    )
    return widths, volumes


def inventory_average(q, component, periodic):
    if periodic:
        return (q + np.roll(q, 1, axis=component)) / 2
    lo = np.take(q, [0], axis=component)
    hi = np.take(q, [-1], axis=component)
    a = np.take(q, np.arange(q.shape[component] - 1), axis=component)
    b = np.take(q, np.arange(1, q.shape[component]), axis=component)
    return np.concatenate((lo, (a + b) / 2, hi), axis=component)


@pytest.mark.parametrize(
    "px,py", [(True, True), (False, True), (True, False), (False, False)]
)
def test_dual_continuity_vector_momentum_and_wall_ledger(
    px, py, record_testsuite_property
):
    grid, rho, velocity, flow, rng = fixture(px, py)
    dt = 0.005
    dm = jnp.asarray(0.01 * rng.random(rho.shape))
    dp = jnp.asarray(0.02 * rng.normal(size=(3, *rho.shape)))
    new = rho + dm - dt * divergence(flow, grid)
    ambient = jnp.asarray(rng.normal(size=(3, grid.nz, grid.ny)))
    result = jax.jit(build_momentum_transport(grid, periodic_x=px, periodic_y=py))(
        rho, new, velocity, dm, dp, flow, ambient, dt
    )
    assert bool(result.accepted), (result.mass_error, result.outgoing_fraction)
    assert float(result.mass_error) < 1e-14
    periodic = (False, py, px)
    for k, component in enumerate((2, 1, 0)):
        widths, vol = dual_geometry(grid, component, px, py)
        old_p = inventory_average(
            np.asarray(rho), component, periodic[component]
        ) * np.asarray(velocity[k])
        before = np.sum(old_p * vol)
        source = np.sum(np.asarray(dp[k])) * grid.dx * grid.dy * grid.dz
        boundary = 0.0
        if not px:
            fx = np.asarray(result.fluxes[k].x)
            boundary = np.sum(
                (fx[..., -1] - fx[..., 0]) * widths[0][:, None] * widths[1][None, :]
            )
        after = np.sum(np.asarray(result.momentum[k]) * vol)
        wall = np.sum(np.asarray(result.wall_impulse[k]) * vol)
        np.testing.assert_allclose(
            after, before + source - dt * boundary + wall, atol=5e-16, rtol=2e-13
        )
        assert np.min(np.asarray(result.kinetic_defect[k])) >= -1e-13
        rho_star = inventory_average(
            np.asarray(rho + dm), component, periodic[component]
        )
        p_star = old_p + inventory_average(
            np.asarray(dp[k]), component, periodic[component]
        )
        source_updated_ke = np.sum(p_star**2 / (2 * rho_star) * vol)
        final_ke = np.sum(
            np.asarray(result.momentum[k]) ** 2
            / (2 * np.asarray(result.inertia_density[k]))
            * vol
        )
        energy_boundary = 0.0
        if not px:
            kx = np.asarray(result.kinetic_fluxes[k].x)
            energy_boundary = np.sum(
                (kx[..., -1] - kx[..., 0]) * widths[0][:, None] * widths[1][None, :]
            )
        defect = np.sum(np.asarray(result.kinetic_defect[k]) * vol)
        np.testing.assert_allclose(
            final_ke + defect + dt * energy_boundary,
            source_updated_ke,
            atol=5e-16,
            rtol=2e-13,
        )

    if not px:
        assert float(jnp.max(jnp.abs(result.fluxes[0].x))) > 0
    assert float(jnp.max(jnp.abs(result.wall_impulse.z))) > 0
    record_testsuite_property(f"dual_mass_error_{px}_{py}", float(result.mass_error))
    record_testsuite_property(f"dual_cfl_{px}_{py}", float(result.outgoing_fraction))


def test_uniform_motion_preserved_with_equal_velocity_mass_and_momentum_sources():
    grid, rho, velocity, flow, rng = fixture()
    velocity = StaggeredVelocity(
        jnp.full_like(velocity.x, 0.8),
        jnp.full_like(velocity.y, -0.3),
        jnp.zeros_like(velocity.z),
    )
    dm = jnp.asarray(0.01 * rng.normal(size=rho.shape))
    vector = jnp.array([0.8, -0.3, 0.0])
    dp = vector[:, None, None, None] * dm
    dt = 0.005
    new = rho + dm - dt * divergence(flow, grid)
    result = jax.jit(build_momentum_transport(grid, periodic_x=True, periodic_y=True))(
        rho,
        new,
        velocity,
        dm,
        dp,
        flow,
        jnp.broadcast_to(vector[:, None, None], (3, grid.nz, grid.ny)),
        dt,
    )
    assert bool(result.accepted)
    for actual, original in zip(result.velocity, velocity):
        np.testing.assert_allclose(actual, original, atol=5e-16, rtol=1e-14)
    for defect in result.kinetic_defect:
        np.testing.assert_allclose(defect, 0, atol=5e-16)


@pytest.mark.parametrize("kind", ["temperature", "composition"])
def test_thermodynamic_stage_flux_preserves_momentum_contact(kind):
    grid, _, velocity, _, _ = fixture()
    config = MoistureConfig()
    left = jnp.broadcast_to(jnp.arange(grid.nx) < grid.nx // 2, velocity.x.shape)
    gas = moist_gas_from_primitive(
        jnp.where(left, 290.0, 350.0)
        if kind == "temperature"
        else jnp.full(left.shape, 310.0),
        jnp.where(left, 0.005, 0.04)
        if kind == "composition"
        else jnp.full(left.shape, 0.01),
        config,
    )
    velocity = StaggeredVelocity(
        jnp.ones_like(velocity.x),
        jnp.zeros_like(velocity.y),
        jnp.zeros_like(velocity.z),
    )
    rho = gas.dry_density + gas.vapor_density
    ambient = jnp.stack(gas)[..., 0] / rho[..., 0]
    poisson = build_pressure_poisson(grid, backend="fft", dtype="float64")
    carrier = jax.jit(
        build_moist_gas_transport(poisson, config, ambient, tolerance=1e-11)
    )(gas, velocity, jax.tree.map(jnp.zeros_like, gas), 0.25 * grid.dx)
    assert bool(carrier.accepted)
    flow = jax.tree.map(lambda f: f[0] + f[1], carrier.fluxes)
    result = jax.jit(build_momentum_transport(grid, periodic_x=True, periodic_y=True))(
        rho,
        carrier.gas.dry_density + carrier.gas.vapor_density,
        velocity,
        jnp.zeros_like(rho),
        jnp.zeros((3, *rho.shape)),
        flow,
        jnp.zeros((3, grid.nz, grid.ny)),
        0.25 * grid.dx,
    )
    assert bool(result.accepted)
    for actual, original in zip(result.velocity, velocity):
        np.testing.assert_allclose(actual, original, atol=5e-15, rtol=0)


def test_actual_core_full_impulse_balances_liquid_momentum():
    grid, _, velocity, _, _ = fixture()
    config = MoistureConfig()
    gas = moist_gas_from_primitive(
        jnp.full(velocity.x.shape, 310.0), jnp.full(velocity.x.shape, 0.01), config
    )
    rho = gas.dry_density + gas.vapor_density
    velocity = jax.tree.map(jnp.zeros_like, velocity)
    flow = velocity
    index = (1, 2, 3)
    vol = grid.dx * grid.dy * grid.dz
    mass = rho[index] * vol
    core = GasInventory(
        mass,
        jnp.zeros(3),
        gas.enthalpy_density[index] * vol,
        jnp.array([gas.dry_density[index], gas.vapor_density[index]]) * vol,
        jnp.array(0.0),
    )
    drop_mass = config.water_density * jnp.pi * (30e-6) ** 3 / 6
    liquid = WaterCoreBins(
        jnp.array([drop_mass]),
        jnp.array([0.01 * mass / drop_mass]),
        jnp.array([[8.0], [-3.0], [1.0]]),
        jnp.array([295.0]),
    )
    dt = 0.002
    exchange = advance_water_core(
        core, liquid, dt, config, WaterDropletProperties(), drag_heat_fraction=1.0
    )
    assert bool(exchange.accepted) and float(exchange.evaporated_mass) > 0
    dm = jnp.zeros_like(rho).at[index].set((exchange.gas.mass - core.mass) / vol)
    dp = (
        jnp.zeros((3, *rho.shape))
        .at[(slice(None), *index)]
        .set((exchange.gas.momentum - core.momentum) / vol)
    )
    result = jax.jit(build_momentum_transport(grid, periodic_x=True, periodic_y=True))(
        rho,
        rho + dm,
        velocity,
        dm,
        dp,
        flow,
        jnp.zeros((3, grid.nz, grid.ny)),
        dt,
    )
    assert bool(result.accepted)
    gas_impulse = []
    for k, component in enumerate((2, 1, 0)):
        _, volumes = dual_geometry(grid, component, True, True)
        gas_impulse.append(np.sum(np.asarray(result.momentum[k]) * volumes))
    liquid_p = lambda b: np.sum(
        np.asarray(b.velocity * (b.mass * b.multiplicity)[None]), axis=1
    )
    np.testing.assert_allclose(
        gas_impulse,
        liquid_p(liquid) - liquid_p(exchange.liquid),
        rtol=1e-12,
        atol=1e-17,
    )


@pytest.mark.parametrize("failure", ["mass_mismatch", "cfl"])
def test_rejection_preserves_input_momentum_without_committed_flux(failure):
    grid, rho, velocity, flow, _ = fixture()
    dt = 10.0 if failure == "cfl" else 0.005
    # Divergence-free throughflow makes the CFL failure independent of positivity.
    flow = StaggeredVelocity(
        jnp.ones_like(flow.x), jnp.zeros_like(flow.y), jnp.zeros_like(flow.z)
    )
    new = rho + (0.1 if failure == "mass_mismatch" else 0.0)
    result = jax.jit(build_momentum_transport(grid, periodic_x=True, periodic_y=True))(
        rho,
        new,
        velocity,
        jnp.zeros_like(rho),
        jnp.zeros((3, *rho.shape)),
        flow,
        jnp.zeros((3, grid.nz, grid.ny)),
        dt,
    )
    assert not bool(result.accepted)
    for a, b in zip(result.velocity, velocity):
        np.testing.assert_array_equal(a, b)
    for q in jax.tree.leaves(
        (
            result.fluxes,
            result.kinetic_fluxes,
            result.wall_impulse,
            result.kinetic_defect,
        )
    ):
        assert np.all(np.asarray(q) == 0)
