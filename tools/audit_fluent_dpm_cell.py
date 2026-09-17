"""Finite-inventory source audit in a 16 x 16 x 4 m application cell.

A liquid pulse mixes with local gas at fixed source-step pressure. No wake,
spatial transport or carrier expansion is solved. This tests source accounting
and reports a component response, not a Fluent comparison or physical validation.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.fluent_dpm_cell import advance_cell_exchange, cell_temperature
from jaxwind.physics.fluent_dpm import DPMWaterMaterial, gas_properties, liquid_enthalpy
from jaxwind.spray_core import WaterCoreBins
from jaxwind.spray_coupling import GasInventory, mean_kinetic_energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    material = DPMWaterMaterial()
    pressure, temperature, y = 101325.0, 310.0, 0.006
    rho, cp = gas_properties(temperature, y, pressure, material)
    mg = rho*16*16*4
    gas = GasInventory(mg, mg*jnp.array([8.0, 0, 0]),
        mg*(cp*(temperature-material.reference_temperature)+y*material.latent_heat_reference),
        mg*jnp.array([1-y, y]), jnp.asarray(0.0))
    d = jnp.array([50e-6, 200e-6])
    mass = jnp.pi/6*material.liquid_density*d**3
    liquid = WaterCoreBins(mass, 0.05/mass,
        jnp.array([[20.0, 15.0], [1.0, -1.0], [0.0, 0.5]]), jnp.array([300.0, 305.0]))

    def totals(g, p):
        m = p.mass*p.multiplicity
        return (g.mass+jnp.sum(m), g.momentum+jnp.sum(m*p.velocity, axis=1),
            g.enthalpy+g.unresolved_energy+mean_kinetic_energy(g)
            +jnp.sum(m*(liquid_enthalpy(p.temperature, material)+0.5*jnp.sum(p.velocity**2, axis=0))))

    before = totals(gas, liquid)
    cases = []
    for mode in ('diffusion-controlled', 'convection-diffusion-controlled'):
        step = jax.jit(lambda g, p, mode=mode: advance_cell_exchange(g, p, pressure, 0.25, material,
            drag_heat_fraction=1.0, substeps=8, vaporization=mode))
        g, p = gas, liquid
        rows, terminal = [], 0.0
        for index in range(8):
            result = step(g, p)
            if not bool(result.accepted):
                raise RuntimeError(f'rejected {mode} source interval {index}')
            g, p = result.gas, result.liquid
            terminal += float(result.terminal_mass)
            after = totals(g, p)
            errors = [float(abs(after[0]-before[0])), float(jnp.max(abs(after[1]-before[1]))),
                      float(abs(after[2]-before[2]))]
            rows.append({'time_s': (index+1)*0.25, 'gas_temperature_K': float(cell_temperature(g, material)),
                'gas_vapor_mass_fraction': float(g.species[1]/g.mass),
                'liquid_mass_kg': float(jnp.sum(p.mass*p.multiplicity)),
                'diameters_m': np.asarray(jnp.cbrt(6*p.mass/(jnp.pi*material.liquid_density))).tolist(),
                'budget_errors_mass_kg_momentum_kg_m_s_energy_J': errors})
        cases.append({'vaporization': mode, 'terminal_mass_transferred_kg': terminal, 'history': rows})
    result = {'scope': __doc__, 'cell_dimensions_m': [16, 16, 4], 'initial_liquid_mass_kg': 0.1,
        'initial_gas_temperature_K': temperature, 'initial_gas_mass_kg': float(mg),
        'initial_diameters_m': np.asarray(d).tolist(), 'cases': cases,
        'Fluent_solver_comparison': False, 'wake_or_spatial_transport_solved': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({c['vaporization']: c['history'][-1] for c in cases}, indent=2))


if __name__ == '__main__':
    main()
