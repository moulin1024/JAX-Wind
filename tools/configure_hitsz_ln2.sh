#!/usr/bin/env bash
# Generate a self-contained setup; never run or submit simulations.
# Usage: bash tools/configure_hitsz_ln2.sh [SETUP_DIRECTORY]
set -euo pipefail
(( $# <= 1 )) || { echo "Expected at most one setup-directory argument." >&2; exit 2; }
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON:-python3}" - "${REPO_ROOT}" "${1:-${REPO_ROOT}/outputs/hitsz_ln2_512_setup}" <<'PY'
import copy
import json
from pathlib import Path
import shutil
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
from jaxwind.config.toml import dumps
from jaxwind.workflows.engine import check_workflow
repo, target = map(lambda p: Path(p).resolve(), sys.argv[1:])
if target.exists() and any(target.iterdir()):
    raise SystemExit(f'Refusing to overwrite nonempty setup: {target}; choose a new directory.')
source = repo / 'cases/HITSZWindTunnel'
base = tomllib.loads((source / 'fv_workflow.toml').read_text())
base['case']['name'] = 'hitsz_r9_ln2_comparison_512x128x256'
base['mesh']['cells'] = [512, 128, 256]
base['time'].update(dt_seconds=0.01, steps=180000, cfl=0.6)
base['physics']['turbine']['x_m'] = 6.0
base['numerics']['scalar_advection_scheme'] = 'upwind'
base['physics']['scalar']['buoyancy_acceleration_per_unit'] = 0.0327
base['workflow'].update(warmup_steps=180000, precursor_steps=32000,
    main_steps=32000, precursor_dt_seconds=0.005625,
    main_dt_seconds=0.005625, main_substeps_per_inflow=2,
    main_frame_count=100, evolve_scalar=True,
    output_directory=str(target / 'run'))
base['output']['directory'] = str(target / 'run')
# Keep physical turbine smearing and cooling widths from the earlier case.
cooling = dict(mass_flow_rate_kg_s=0.0125,
    exit_vapor_quality=0.02238146379800278, pre_nozzle_vapor_quality=0.0,
    nozzle_diameter_m=0.005, injection_speed_m_s=4.0,
    cone_half_angle_degrees=0.0, injection_temperature_k=77.34,
    ambient_temperature_k=300.0, liquid_latent_heat_j_kg=199180.0,
    nitrogen_heat_capacity_j_kg_k=1040.0, air_density_kg_m3=1.225,
    air_heat_capacity_j_kg_k=1005.0, thermal_coupling_efficiency=1.0,
    streamwise_offset_m=0.09375, standard_deviation_m=[0.30, 0.09375, 0.1125],
    ramp_time_s=1.0)
warm = dict(case='case.toml', operation='periodic', overrides={
    'diagnostics': {'sample_start_step': 0},
    'workflow': {'evolve_scalar': False}})
precursor = dict(case='case.toml', operation='record-inflow', fixed_dt=True,
    inputs={'checkpoint': '@warmup/checkpoint'}, options={'record_plane': 10},
    overrides={'time': {'steps': 32000, 'dt_seconds': 0.005625, 'frame_count': 0},
               'diagnostics': {'sample_start_step': 0},
               'workflow': {'evolve_scalar': False}})
control = dict(case='case.toml', operation='open-inflow', fixed_dt=True,
    inputs={'checkpoint': '@warmup/checkpoint', 'inflow': '@precursor/inflow'},
    options={'substeps_per_inflow': 2}, overrides={
        'time': {'steps': 64000, 'dt_seconds': 0.0028125, 'chunk_steps': 200, 'frame_count': 100},
        'numerics': {'pressure_backend': 'gmg'},
        'diagnostics': {'sample_start_step': 0}})
main = copy.deepcopy(control)
main['overrides']['physics'] = {'cooling': cooling}
graph = {'schema_version': 1, 'output': {'directory': str(target / 'run')},
         'stages': {'warmup': warm, 'precursor': precursor, 'control': control, 'main': main}}
target.mkdir(parents=True, exist_ok=True)
for name in ('fv_initial_profile.csv', 'fv_reference_results.json'):
    shutil.copy2(source / name, target / name)
(target / 'case.toml').write_text(dumps(base))
(target / 'workflow.toml').write_text(dumps(graph))
checked = check_workflow(target / 'workflow.toml')
(target / 'validation.json').write_text(json.dumps(checked, indent=2) + '\n')
print(f'Configured and validated: {target}/workflow.toml')
print('Durations: warmup 1800 s; precursor 180 s; control 180 s; LN2 main 180 s.')
print('Main cases: dt=0.0028125 s, 100 frames each; source=(6.09375, 3, 0.876) m.')
PY
