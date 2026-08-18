#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

case_file="cases/DTU10MWPrecursor/config_optimized.toml"
warmup_dir="outputs/dtu10mw_warmup_optimized_cufft_lasd4_128x64x256"
latest_checkpoint="$warmup_dir/checkpoint_latest.npz"
final_checkpoint="$warmup_dir/checkpoint_final.npz"
chain_output="outputs/dtu10mw_adbem_main_shift31_optimized_cufft_lasd4_1h_x1000_128x64x256"
python_executable="${JAXWIND_PYTHON:-python}"
openfast_model="${JAXWIND_DTU10MW_FAST:-}"

if [[ -z "$openfast_model" || ! -f "$openfast_model" ]]; then
    echo "set JAXWIND_DTU10MW_FAST to the DTU-10MW OpenFAST .fst deck" >&2
    exit 2
fi

echo "optimized_chain_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ ! -f "$final_checkpoint" ]]; then
    if [[ -f "$latest_checkpoint" ]]; then
        "$python_executable" -u -m applications.pressure_driven_lasd \
            "$case_file" \
            --restart "$latest_checkpoint"
    else
        "$python_executable" -u -m applications.pressure_driven_lasd \
            "$case_file" \
            --overwrite
    fi
fi

if [[ ! -f "$final_checkpoint" ]]; then
    echo "warmup did not produce $final_checkpoint" >&2
    exit 1
fi

echo "optimized_precursor_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$python_executable" -u -m applications.windfarm_precursor \
    "$case_file" \
    --restart "$final_checkpoint" \
    --output "$chain_output" \
    --precursor-steps 18000 \
    --main-steps 18000 \
    --fringe-start-fraction 0.75 \
    --fringe-relaxation-seconds 4.0 \
    --section inflow \
    --sample-buffer 128 \
    --read-buffer 128 \
    --spanwise-shift-cells 31 \
    --compression none \
    --frames 100 \
    --gif-fps 12 \
    --turbine dtu-10mw-ad-bem \
    --openfast-model "$openfast_model" \
    --turbine-x-m 1000 \
    --disk-smoothing-width-m 32.0 \
    --overwrite

echo "optimized_chain_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
