"""Compatible pressure work, dual flux transfer and coupled total-energy budgets."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_spray_carrier import CONFIG, geometry, setup
from test_spray_cell_step import C, budget
from test_spray_injection import coupled, injection_setup

from jaxwind.numerics.discretization import divergence
from jaxwind.numerics.poisson import build_pressure_poisson
from jaxwind.spray_carrier import build_carrier_step
from jaxwind.spray_energy import (
    carrier_energy_terms,
    compatible_energy_terms,
    component_sum,
    kinetic_flux_to_primary,
)
from jaxwind.spray_injection import births_from_mass_flow
from jaxwind.spray_low_mach import moist_gas_eos_density
from jaxwind.spray_momentum import _dual_divergence, dual_mass_flux
from jaxwind.state import StaggeredVelocity


def poisson_for(grid, periodic_x=False, periodic_y=True, open_x_low=True):
    return build_pressure_poisson(
        grid,
        backend="gmg",
        periodic_x=periodic_x,
        periodic_y=periodic_y,
        open_x_low=open_x_low,
        dtype="float64",
    )


@pytest.mark.parametrize(
    "px,py,open_low",
    [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, True),
        (False, True, False),
    ],
)
def test_pressure_product_identity_and_flux_mapping(px, py, open_low):
    grid, gas, _, _, _, _ = setup(periodic=px, nx=6)
    shape = gas.dry_density.shape
    rng = np.random.default_rng(921)
    velocity = StaggeredVelocity(
        jnp.array(rng.normal(size=shape if px else (*shape[:2], shape[2] + 1))),
        jnp.array(rng.normal(size=shape if py else (shape[0], shape[1] + 1, shape[2]))),
        jnp.array(rng.normal(size=(shape[0] + 1, *shape[1:]))),
    )
    velocity = velocity._replace(z=velocity.z.at[0].set(0).at[-1].set(0))
    if not py:
        velocity = velocity._replace(y=velocity.y.at[:, 0].set(0).at[:, -1].set(0))
    periodic = (False, py, px)
    zero = jax.tree.map(jnp.zeros_like, velocity)
    inertia = jax.tree.map(lambda v: jnp.full_like(v, 1.3), velocity)
    momentum = jax.tree.map(lambda u: u * 1.3 * 0.7, velocity)
    fluxes = tuple(
        jax.tree.map(
            lambda f: jnp.array(rng.normal(size=f.shape)),
            dual_mass_flux(zero, c, periodic),
        )
        for c in (2, 1, 0)
    )
    pressure = jnp.array(rng.normal(size=shape))
    terms = jax.jit(
        lambda p, m, u: compatible_energy_terms(
            p,
            m,
            inertia,
            zero,
            zero,
            fluxes,
            u,
            0.003,
            grid,
            periodic_x=px,
            periodic_y=py,
            open_x_low=open_low,
        )
    )(pressure, momentum, velocity)
    scale = max(
        float(jnp.max(jnp.abs(terms.pressure_conversion))),
        float(jnp.max(jnp.abs(0.003 * divergence(terms.pressure_flux, grid)))),
    )
    error = float(jnp.max(jnp.abs(terms.pressure_identity_residual)))
    assert error <= 32 * np.finfo(float).eps * scale
    print("pressure_product", px, py, open_low, error, error / scale)
    primary_div = divergence(kinetic_flux_to_primary(fluxes, periodic), grid)
    dual_div = component_sum(
        StaggeredVelocity(
            *(_dual_divergence(f, grid, c, periodic) for c, f in zip((2, 1, 0), fluxes))
        ),
        periodic,
    )
    np.testing.assert_allclose(primary_div, dual_div, rtol=2e-14, atol=2e-14)


def boundary(flux, grid):
    return float(jnp.sum(flux.x[..., -1] - flux.x[..., 0]) * grid.dy * grid.dz)


def kinetic(gas, velocity, grid):
    # Independent whole-domain half-volume quadrature from the carrier tests.
    from test_spray_carrier import average

    rho = np.asarray(gas.dry_density + gas.vapor_density)
    return sum(
        float(
            np.sum(
                0.5
                * average(rho, c, (False, True, False)[c])
                * np.asarray(u) ** 2
                * geometry(grid, c, False)[1]
            )
        )
        for c, u in zip((2, 1, 0), velocity)
    )


def test_energy_coupled_carrier_closes_budget_and_changes_eos_consistently():
    grid, gas, u, _, increments, dp = setup(periodic=False)
    y = (jnp.arange(grid.ny) + 0.5) * grid.dy
    x = jnp.arange(grid.nx + 1) * grid.dx
    u = u._replace(
        x=u.x
        + 8
        + 4 * jnp.sin(2 * jnp.pi * y[None, :, None] / grid.ly)
        + 2 * jnp.sin(2 * jnp.pi * x[None, None, :] / grid.lx),
        y=u.y + 2,
    )
    poisson = poisson_for(grid)
    ambient = jnp.stack(gas)[..., 0] / (gas.dry_density + gas.vapor_density)[..., 0]
    ambient_u = jnp.stack((u.x[..., 0], u.y[..., 0], jnp.zeros((grid.nz, grid.ny))))
    dt, volume = 0.002, grid.dx * grid.dy * grid.dz
    before = float(jnp.sum(gas.enthalpy_density) * volume) + kinetic(gas, u, grid)
    results = []
    for enabled in [False, True]:
        advance = jax.jit(
            build_carrier_step(
                poisson,
                CONFIG,
                ambient,
                ambient_u,
                energy_coupling=enabled,
                tolerance=1e-10,
                max_iterations=150,
            )
        )
        r = advance(gas, u, increments, dp, dt)
        assert bool(r.accepted), r[15:]
        terms = carrier_energy_terms(r, dt, poisson)
        hflux = jax.tree.map(lambda f: f[2], r.scalar_fluxes)
        iteration = float(
            jnp.sum(component_sum(r.iteration_work, (False, True, False))) * volume
        )
        after = float(jnp.sum(r.gas.enthalpy_density) * volume) + kinetic(
            r.gas, r.velocity, grid
        )
        exports = dt * (
            boundary(hflux, grid)
            + boundary(terms.kinetic_flux, grid)
            + boundary(terms.pressure_flux, grid)
        )
        exports += float(jnp.sum(terms.wall_export) * volume)
        residual = after - before + exports - iteration
        results.append(residual)
        if enabled:
            assert abs(residual) < 8e-11
            assert float(jnp.min(terms.numerical_heat)) > -1e-11
            assert float(jnp.max(jnp.abs(terms.pressure_conversion))) > 1e-6
            np.testing.assert_allclose(
                r.gas.dry_density + r.gas.vapor_density,
                moist_gas_eos_density(r.gas, CONFIG),
                rtol=1.1e-10,
            )
            transported = gas.enthalpy_density - dt * divergence(hflux, grid)
            np.testing.assert_allclose(
                r.gas.enthalpy_density - transported,
                terms.numerical_heat + terms.pressure_conversion,
                rtol=3e-8,
                atol=8e-11,
            )
            print(
                "energy_carrier",
                int(r.iterations),
                float(r.momentum_error),
                residual,
                float(jnp.sum(terms.numerical_heat) * volume),
                float(jnp.sum(terms.pressure_conversion) * volume),
            )
    assert abs(results[0]) > 1e-4
    print("uncoupled_total_energy_defect_J", results[0])


def test_injected_multistep_total_energy_with_outflow_and_gravity():
    grid, gas, u, e, q, p = injection_setup()
    q = q._replace(multiplicity=jnp.zeros_like(q.mass))
    dt, steps = 1e-4, 8
    b = births_from_mass_flow(
        1e-3,
        dt,
        jnp.array([70e-6]),
        jnp.ones(1),
        jnp.array([[18.0], [0.0], [0.0]]),
        jnp.array([300.0]),
        jnp.array([[0.5995], [0.15], [0.15]]),
        jnp.array([0.5]),
        C,
    )
    advance = coupled(grid, gas, energy_coupling=True, tolerance=1e-10)
    poisson = poisson_for(grid)
    volume = grid.dx * grid.dy * grid.dz
    initial = float(budget(grid, gas, u, e, q)[3])
    injected = external = exported = iteration = 0.0
    residuals = []
    for _ in range(steps):
        r = advance(gas, u, e, q, dt, p, b)
        m, c = r.moving, r.moving.phase.carrier
        assert bool(m.phase.accepted), c[15:]
        terms = carrier_energy_terms(c, dt, poisson)
        injected += float(r.injected_enthalpy + r.injected_kinetic_energy)
        external += float(jnp.sum(m.external_work))
        exported += float(jnp.sum(m.exited_energy))
        exported += dt * (
            boundary(jax.tree.map(lambda f: f[2], c.scalar_fluxes), grid)
            + boundary(terms.kinetic_flux, grid)
            + boundary(terms.pressure_flux, grid)
            + boundary(m.phase.unresolved_flux, grid)
        )
        exported += float(jnp.sum(terms.wall_export) * volume)
        iteration += float(
            jnp.sum(component_sum(c.iteration_work, (False, True, False))) * volume
        )
        gas, u, e, q, p = (
            c.gas,
            c.velocity,
            m.phase.unresolved_density,
            m.phase.liquid,
            m.position,
        )
        total = float(budget(grid, gas, u, e, q)[3])
        residuals.append(total - initial + exported - injected - external - iteration)
        assert abs(residuals[-1]) < 5e-11
        np.testing.assert_array_equal(q.multiplicity, 0)
    print(
        "injected_total_energy",
        steps,
        max(abs(v) for v in residuals),
        injected,
        exported,
        iteration,
    )


def test_energy_enabled_failure_rolls_back_all_fields_and_ledgers():
    from test_spray_cell_step import assert_tree_equal

    grid, gas, u, e, q, p = injection_setup()
    from test_spray_injection import births

    r = coupled(grid, gas, energy_coupling=True, max_iterations=1)(
        gas, u, e, q, 1e-4, p, births()
    )
    m = r.moving
    assert not bool(m.phase.accepted)
    assert_tree_equal(
        (
            m.phase.carrier.gas,
            m.phase.carrier.velocity,
            m.phase.unresolved_density,
            m.phase.liquid,
            m.position,
        ),
        (gas, u, e, q, p),
    )
    terms = carrier_energy_terms(m.phase.carrier, 1e-4, poisson_for(grid))
    for value in jax.tree.leaves((terms, r[1:5], m.external_work, m.exited_energy)):
        np.testing.assert_array_equal(value, jnp.zeros_like(value))
