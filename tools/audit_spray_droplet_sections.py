"""Read-only, instantaneous parcel section diagnostics for sim.pdf Fig. 11.

Slab samples are occupancy-weighted, not crossing events. Flux estimates use
number*positive axial velocity/slab width. Diameter bins use current diameter;
the original parcel diameter is not stored. No direct experimental fit is made.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.run / 'checkpoint.npz') as archive:
        data = {k.removeprefix('state/parcels/'): archive[k] for k in archive.files
                if k.startswith('state/parcels/')}
    import tomllib
    case = tomllib.loads((args.run / 'resolved_case.toml').read_text())
    length = case['mesh']['lengths_m'][0]
    dx = length / case['mesh']['cells'][0]
    # The benchmark uses the shared default liquid density of 997 kg/m3.
    from jaxwind.physics.moisture import MoistureConfig
    density = MoistureConfig().water_density
    diameter = np.cbrt(6 * data['mass'] / (np.pi * density)) * 1e6
    temp = data['temperature'] - 273.15
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained', sharey=True)
    results = []
    for ax, fraction in zip(axes, [0.25, 0.5, 0.75]):
        selected = data['active'] & (abs(data['position'][0] - fraction * length) < dx)
        number = data['multiplicity'] * np.maximum(data['velocity'][0], 0) / (2 * dx)
        rows = []
        for lo, hi in zip([70, 150, 250, 350, 450], [150, 250, 350, 450, 520]):
            m = selected & (diameter >= lo) & (diameter < hi)
            massflux = number[m] * data['mass'][m]
            if not m.any() or massflux.sum() <= 0:
                continue
            rows.append({'diameter_bin_um': [lo, hi], 'parcels': int(m.sum()),
                         'temperature_min_C': float(temp[m].min()),
                         'temperature_max_C': float(temp[m].max()),
                         'mass_flux_weighted_temperature_C': float(np.average(temp[m], weights=massflux)),
                         'mass_flux_kg_s': float(massflux.sum()),
                         'mass_flux_weighted_axial_velocity_m_s': float(np.average(data['velocity'][0, m], weights=massflux)),
                         'mass_flux_weighted_radius_m': float(np.average(np.hypot(data['position'][1, m]-case['mesh']['lengths_m'][1]/2, data['position'][2, m]-case['mesh']['lengths_m'][2]/2), weights=massflux)),
                         'wall_touched_flux_fraction': float(massflux[data['touched_wall'][m]].sum()/massflux.sum())})
        for wall, label, color in [(False, 'Free flight', 'tab:blue'), (True, 'Wall touched', 'tab:orange')]:
            m = selected & (data['touched_wall'] == wall)
            ax.scatter(diameter[m], temp[m], s=5, alpha=.25, label=label, color=color, rasterized=True)
        ax.set(xlabel='Current diameter (µm)', title=f'x/L = {fraction:g}', xlim=(70, 520), ylim=(17, 36))
        ax.grid(alpha=.2)
        results.append({'x_over_L': fraction, 'slab_width_m': 2*dx, 'bins': rows})
    axes[0].set_ylabel('Droplet temperature (°C)')
    axes[-1].legend()
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / 'droplet_sections.png', dpi=180)
    plt.close(fig)
    payload = {'run': str(args.run), 'caveat': __doc__, 'sections': results}
    (args.output / 'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
