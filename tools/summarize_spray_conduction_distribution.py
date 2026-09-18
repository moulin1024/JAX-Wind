"""Approximate mass-distribution weighting of prescribed-path thermal diagnostics.

Five initial sizes cannot establish diameter-quadrature convergence. Report both
nearest-bin integration and linear interpolation in diameter as a sensitivity;
neither includes gas feedback or changed droplet trajectories.
"""
import argparse
import json
from pathlib import Path
import tomllib
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--case', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    case = tomllib.loads(args.case.read_text())
    ref = case['case']['reference']
    scale, spread = ref['rosin_rammler_scale_m'], ref['rosin_rammler_spread']
    lo, hi = ref['minimum_diameter_m'], ref['maximum_diameter_m']
    # Integrate against a normalized truncated mass CDF, not a number CDF.
    cdf = lambda d: 1 - np.exp(-(np.asarray(d) / scale)**spread)
    nodes = np.linspace(lo, hi, 10001)
    mids = .5 * (nodes[:-1] + nodes[1:])
    weights = np.diff(cdf(nodes)) / (cdf(hi) - cdf(lo))
    sizes = np.array(result['selected_diameters_um']) * 1e-6
    assert len(sizes) >= 5 and np.isclose(weights.sum(), 1)
    nearest = np.abs(mids[:, None] - sizes[None, :]).argmin(axis=1)
    reports = {}
    for model in ['lumped', 'k0.6_n32_halfdt']:
        reports[model] = {}
        for station in [.25, .5, .75, .99]:
            rows = [r for r in result['history_runs'][model]['samples'] if np.isclose(r['x_over_L'], station)]
            by_size = []
            for diameter in sizes:
                group = [r for r in rows if np.isclose(r['diameter_um'], diameter * 1e6)]
                assert len(group) == result['rays_per_size']
                values = []
                for row in group:
                    initial_mass = row['mass_kg'] + row['cumulative_evaporated_kg']
                    values.append([row['cumulative_evaporated_kg']/initial_mass,
                                   row['cumulative_heat_J']/initial_mass,
                                   row['mean_temperature_C']])
                by_size.append(np.mean(values, axis=0))
            by_size = np.asarray(by_size)
            entry = {'by_size': by_size.tolist()}
            for method in ['nearest', 'linear']:
                interpolated = by_size[nearest] if method == 'nearest' else np.stack([
                    np.interp(mids, sizes, by_size[:, i]) for i in range(3)], axis=1)
                total = weights @ interpolated
                entry[method] = dict(zip(['evaporated_fraction', 'heat_J_per_kg_injected',
                                         'initial_mass_weighted_bulk_C'], total.tolist()))
            reports[model][str(station)] = entry
    output = {'scope': __doc__, 'diameters_um': (sizes*1e6).tolist(),
              'by_size_columns': ['evaporated_fraction', 'heat_J_per_kg_injected', 'bulk_C'],
              'models': reports, 'differences': {}}
    for station in reports['lumped']:
        output['differences'][station] = {}
        for method in ['nearest', 'linear']:
            base = reports['lumped'][station][method]
            finite = reports['k0.6_n32_halfdt'][station][method]
            output['differences'][station][method] = {
                'evaporation_change_percent': 100*(finite['evaporated_fraction']/base['evaporated_fraction']-1),
                'heat_change_percent': 100*(finite['heat_J_per_kg_injected']/base['heat_J_per_kg_injected']-1),
                'bulk_temperature_shift_K': finite['initial_mass_weighted_bulk_C']-base['initial_mass_weighted_bulk_C']}
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output['differences'], indent=2))


if __name__ == '__main__':
    main()
