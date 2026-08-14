#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

case_file="cases/DTU10MWPrecursor/config.toml"
warmup_dir="outputs/dtu10mw_warmup_128x64x256"
latest_checkpoint="$warmup_dir/checkpoint_latest.npz"
final_checkpoint="$warmup_dir/checkpoint_final.npz"
chain_output="outputs/dtu10mw_precursor_main_1h_128x64x256"
python_executable="${JAXWIND_PYTHON:-python}"

echo "chain_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ ! -f "$final_checkpoint" ]]; then
    if [[ -f "$latest_checkpoint" ]]; then
        "$python_executable" -m applications.pressure_driven_lasd \
            "$case_file" \
            --restart "$latest_checkpoint"
    else
        "$python_executable" -m applications.pressure_driven_lasd \
            "$case_file" \
            --overwrite
    fi
fi

"$python_executable" -m applications.windfarm_precursor \
    "$case_file" \
    --restart "$final_checkpoint" \
    --output "$chain_output" \
    --precursor-steps 18000 \
    --main-steps 18000 \
    --sample-buffer 16 \
    --read-buffer 64 \
    --frames 100 \
    --gif-fps 12 \
    --turbine dtu-10mw-adm \
    --disk-smoothing-width-m 32.0 \
    --overwrite

echo "chain_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
