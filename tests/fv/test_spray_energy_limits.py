"""Wall-energy ownership and incompatible fixed-pressure closed heating."""

import jax
import jax.numpy as jnp
import numpy as np
from test_spray_carrier import CONFIG, setup
from test_spray_cell_step import assert_tree_equal

from jaxwind.numerics.poisson import build_pressure_poisson
from jaxwind.spray_carrier import build_carrier_step
from jaxwind.spray_energy import compatible_energy_terms
from jaxwind.spray_momentum import dual_mass_flux


def test_wall_energy_export_is_not_heated_into_gas():
    grid, gas, velocity, _, _, _ = setup(periodic=True)
    rho = 1.3
    inertia = jax.tree.map(lambda q: jnp.full_like(q, rho), velocity)
    impulse = velocity._replace(
        z=velocity.z.at[0].set(-jnp.sqrt(5 * rho)).at[-1].set(-jnp.sqrt(5 * rho))
    )
    loss = velocity._replace(
        x=jnp.full_like(velocity.x, 0.6), z=velocity.z.at[0].set(2.5).at[-1].set(2.5)
    )
    fluxes = tuple(dual_mass_flux(velocity, c, (False, True, True)) for c in (2, 1, 0))
    terms = compatible_energy_terms(
        jnp.zeros_like(gas.dry_density),
        velocity,
        inertia,
        impulse,
        loss,
        fluxes,
        velocity,
        0.01,
        grid,
        periodic_x=True,
        periodic_y=True,
    )
    expected = np.zeros(gas.dry_density.shape)
    expected[0] = expected[-1] = 1.25
    np.testing.assert_allclose(terms.wall_export, expected, atol=1e-15)
    np.testing.assert_allclose(terms.numerical_heat, 0.6, atol=1e-15)
    np.testing.assert_array_equal(terms.pressure_conversion, 0)


def test_closed_fixed_pressure_heating_rejects_without_inventory_repair():
    grid, gas, velocity, _, increments, dp = setup(periodic=True)
    velocity = velocity._replace(
        x=velocity.x
        + 3
        * jnp.sin(2 * jnp.pi * (jnp.arange(grid.ny) + 0.5)[None, :, None] / grid.ny),
        y=velocity.y + 1,
    )
    poisson = build_pressure_poisson(
        grid,
        backend="fft",
        periodic_x=True,
        periodic_y=True,
        open_x_low=False,
        dtype="float64",
    )
    ambient = jnp.stack(gas)[..., 0] / (gas.dry_density + gas.vapor_density)[..., 0]
    ambient_u = jnp.zeros((3, grid.nz, grid.ny))
    default = jax.jit(
        build_carrier_step(poisson, CONFIG, ambient, ambient_u, max_iterations=100)
    )
    coupled = jax.jit(
        build_carrier_step(
            poisson,
            CONFIG,
            ambient,
            ambient_u,
            max_iterations=100,
            energy_coupling=True,
        )
    )
    baseline = default(gas, velocity, increments, dp, 0.005)
    assert bool(baseline.accepted), baseline[15:]
    heated = coupled(gas, velocity, increments, dp, 0.005)
    assert not bool(heated.accepted)
    assert float(heated.eos_error) > 1e-9
    assert_tree_equal((heated.gas, heated.velocity), (gas, velocity))
    for field in jax.tree.leaves(heated[3:15]):
        np.testing.assert_array_equal(field, 0)
    print("closed_energy_rejection", int(heated.iterations), float(heated.eos_error))
