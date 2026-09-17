"""Finite-cell energy/species/momentum checks at the actual LES cell volume."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.fluent_dpm_cell import advance_cell_exchange, cell_temperature
from jaxwind.physics.fluent_dpm import DPMWaterMaterial, gas_properties, liquid_enthalpy
from jaxwind.spray_core import WaterCoreBins
from jaxwind.spray_coupling import GasInventory, mean_kinetic_energy

jax.config.update("jax_enable_x64", True)
P = DPMWaterMaterial()


def fixture(liquid_mass=0.1, volume=16*16*4):
    y, temperature, pressure = 0.006, 310.0, 101325.0
    rho, _ = gas_properties(temperature, y, pressure, P)
    gas_mass = rho*volume
    species = gas_mass*jnp.array([1-y, y])
    enthalpy = ((species[0]*P.dry_air_cp+species[1]*P.vapor_cp)*(temperature-P.reference_temperature)
                + species[1]*P.latent_heat_reference)
    gas = GasInventory(gas_mass, gas_mass*jnp.array([8.0, 0, 0]), enthalpy, species, jnp.asarray(0.0))
    d = jnp.array([50e-6, 200e-6])
    m = jnp.pi/6*P.liquid_density*d**3
    liquid = WaterCoreBins(m, liquid_mass/2/m, jnp.array([[20.0, 15.0], [1.0, -1.0], [0.0, 0.5]]),
                           jnp.array([300.0, 305.0]))
    return gas, liquid, pressure


def totals(g, p):
    mass = p.mass*p.multiplicity
    return (g.mass+jnp.sum(mass), g.momentum+jnp.sum(mass*p.velocity, axis=1),
            g.enthalpy+g.unresolved_energy+mean_kinetic_energy(g)
            + jnp.sum(mass*(liquid_enthalpy(p.temperature, P)+0.5*jnp.sum(p.velocity**2, axis=0))))


@pytest.mark.parametrize("heat_fraction", [0.0, 1.0])
def test_coarse_cell_exchange_conserves_water_momentum_and_total_energy(heat_fraction):
    gas, liquid, pressure = fixture()
    result = jax.jit(lambda g, p: advance_cell_exchange(g, p, pressure, 0.05, P,
        substeps=4, drag_heat_fraction=heat_fraction))(gas, liquid)
    assert bool(result.accepted) and float(result.evaporated_mass) > 0
    for a, b in zip(totals(gas, liquid), totals(result.gas, result.liquid)):
        np.testing.assert_allclose(a, b, rtol=3e-15, atol=1e-11)
    np.testing.assert_allclose(result.gas.species[0], gas.species[0], rtol=0, atol=0)
    np.testing.assert_allclose(result.gas.species[1]-gas.species[1], result.evaporated_mass, atol=2e-15)
    assert float(cell_temperature(result.gas, P)) < float(cell_temperature(gas, P))


def test_finite_cell_substep_convergence():
    gas, liquid, pressure = fixture(liquid_mass=1.0)
    results = []
    for n in (2, 4, 8):
        r = jax.jit(lambda g, p, n=n: advance_cell_exchange(g, p, pressure, 0.01, P,
                    substeps=n, drag_heat_fraction=1.0))(gas, liquid)
        assert bool(r.accepted)
        results.append(float(r.evaporated_mass))
    assert abs(results[2]-results[1]) < abs(results[1]-results[0])


def test_invalid_or_depleted_gas_rolls_back_every_inventory():
    gas, liquid, pressure = fixture()
    bad = gas._replace(species=gas.species.at[0].set(-1))
    r = jax.jit(lambda g, p: advance_cell_exchange(g, p, pressure, 0.1, P,
                    substeps=1, drag_heat_fraction=1.0))(bad, liquid)
    assert not bool(r.accepted) and float(r.evaporated_mass) == 0
    for old, new in zip(jax.tree.leaves((bad, liquid)), jax.tree.leaves((r.gas, r.liquid))):
        np.testing.assert_array_equal(old, new)
