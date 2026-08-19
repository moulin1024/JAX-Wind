#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

main_case="cases/DTU10MWPrecursor/config_adbem_dt0p1.toml"
adapted_checkpoint="outputs/dtu10mw_legacy_adapt_1h_128x64x256/checkpoint_final.npz"
chain_output="outputs/dtu10mw_strict_fortran_adbem_tower_nacelle_1h"
python_executable="${JAXWIND_PYTHON:-python}"
openfast_model="${JAXWIND_DTU10MW_FAST:-}"

if [[ -z "$openfast_model" || ! -f "$openfast_model" ]]; then
    echo "set JAXWIND_DTU10MW_FAST to the DTU-10MW OpenFAST .fst deck" >&2
    exit 2
fi

if [[ ! -f "$adapted_checkpoint" ]]; then
    echo "adapted checkpoint is missing: $adapted_checkpoint" >&2
    exit 1
fi

"$python_executable" -u -m applications.windfarm_precursor \
    "$main_case" \
    --restart "$adapted_checkpoint" \
    --output "$chain_output" \
    --precursor-steps 36000 \
    --main-steps 36000 \
    --sample-buffer 8 \
    --read-buffer 8 \
    --compression none \
    --frames 100 \
    --gif-fps 12 \
    --turbine dtu-10mw-ad-bem \
    --openfast-model "$openfast_model" \
    --turbine-x-m 1000 \
    --rotor-speed-rpm 9.6 \
    --blade-pitch-degrees 0.0 \
    --ad-bem-smearing-azimuthal-elements 64 \
    --body-smoothing-width-m 96.0 \
    --overwrite
