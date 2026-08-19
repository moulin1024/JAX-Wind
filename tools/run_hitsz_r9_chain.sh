#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

coarse_case_file="cases/HITSZWindTunnel/config_coarse.toml"
fine_case_file="cases/HITSZWindTunnel/config.toml"
coarse_dir="outputs/hitsz_r9_adbem_benchmark/warmup_coarse"
coarse_latest_checkpoint="$coarse_dir/checkpoint_latest.npz"
coarse_final_checkpoint="$coarse_dir/checkpoint_final.npz"
fine_dir="outputs/hitsz_r9_adbem_benchmark/warmup_fine"
fine_initial_checkpoint="$fine_dir/checkpoint_prolonged_initial.npz"
fine_latest_checkpoint="$fine_dir/checkpoint_latest.npz"
fine_final_checkpoint="$fine_dir/checkpoint_final.npz"
chain_output="outputs/hitsz_r9_adbem_benchmark/precursor_main"
python_executable="${JAXWIND_PYTHON:-python}"

echo "hitsz_chain_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "stage=coarse_warmup grid=128x32x64 duration_s=900 dt_s=0.005"
if [[ ! -f "$coarse_final_checkpoint" ]]; then
    if [[ -f "$coarse_latest_checkpoint" ]]; then
        "$python_executable" -u -m applications.pressure_driven_lasd \
            "$coarse_case_file" \
            --restart "$coarse_latest_checkpoint"
    else
        "$python_executable" -u -m applications.pressure_driven_lasd \
            "$coarse_case_file" \
            --overwrite
    fi
fi

if [[ ! -f "$coarse_final_checkpoint" ]]; then
    echo "HITSZ coarse warmup did not produce $coarse_final_checkpoint" >&2
    exit 1
fi

echo "stage=coarse_to_fine_transfer method=linear_plus_pressure_projection"
if [[ ! -f "$fine_initial_checkpoint" ]]; then
    "$python_executable" -u tools/prolong_pressure_driven_checkpoint.py \
        "$coarse_final_checkpoint" \
        "$fine_case_file" \
        "$fine_initial_checkpoint"
fi

echo "stage=fine_extension grid=256x64x128 duration_s=90 dt_s=0.0025"
if [[ ! -f "$fine_final_checkpoint" ]]; then
    if [[ -f "$fine_latest_checkpoint" ]]; then
        fine_restart="$fine_latest_checkpoint"
    else
        fine_restart="$fine_initial_checkpoint"
    fi
    "$python_executable" -u -m applications.pressure_driven_lasd \
        "$fine_case_file" \
        --restart "$fine_restart"
fi

if [[ ! -f "$fine_final_checkpoint" ]]; then
    echo "HITSZ fine extension did not produce $fine_final_checkpoint" >&2
    exit 1
fi

echo "stage=precursor_main duration_s_each=90 dt_s=0.0025"
"$python_executable" -u -m applications.windfarm_precursor \
    "$fine_case_file" \
    --restart "$fine_final_checkpoint" \
    --output "$chain_output" \
    --precursor-steps 36000 \
    --main-steps 36000 \
    --sample-buffer 8 \
    --read-buffer 8 \
    --compression none \
    --frames 100 \
    --gif-fps 12 \
    --turbine hitsz-r9-ad-bem \
    --turbine-x-m 12.0 \
    --disk-smoothing-width-m 0.09375 \
    --body-smoothing-width-m 0.1875 \
    --ad-bem-smearing-azimuthal-elements 64 \
    --rotor-speed-rpm 480.0 \
    --blade-pitch-degrees 0.0 \
    --overwrite

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/jaxwind-matplotlib}" \
MPLBACKEND=Agg \
"$python_executable" tools/compare_hub_height_gaussian_wake.py \
    "$chain_output/main_xz_frames.npz" \
    --precursor-recording "$chain_output/precursor.h5" \
    --precursor-dt-seconds 0.0025 \
    --output "$chain_output/gaussian_wake_comparison" \
    --turbine-x-m 12.0 \
    --turbine-y-m 3.0 \
    --hub-height-m 0.876 \
    --rotor-diameter-m 1.26 \
    --ct 0.810 \
    --lx-m 24.0 \
    --ly-m 6.0 \
    --lz-m 3.6 \
    --spinup-seconds 15.0 \
    --fit-min-d 3.0 \
    --fit-max-d 8.0

echo "hitsz_chain_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
