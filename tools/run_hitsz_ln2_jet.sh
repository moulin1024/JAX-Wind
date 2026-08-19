#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

duration_seconds="${1:-90}"
checkpoint="${2:-}"
python_executable="${JAXWIND_PYTHON:-python}"
config="${JAXWIND_LN2_CONFIG:-legacy/cases/LiquidNitrogenHubJet/configs/hitsz_256x64x128_full_physics.toml}"
output="${JAXWIND_LN2_OUTPUT:-outputs/hitsz_ln2_hub_jet/full_physics_${duration_seconds}s}"
ln2_x="${JAXWIND_LN2_X:-12.15}"
ln2_y="${JAXWIND_LN2_Y:-3.0}"
ln2_z="${JAXWIND_LN2_Z:-0.876}"
ln2_sigma_x="${JAXWIND_LN2_SIGMA_X:-0.09375}"
# Optional explicit exit speed [m/s]. Unset, the runner derives the pure-liquid
# bulk speed mdot/(rho_LN2*pi*r^2). Set this when the nitrogen flashes in the
# nozzle: even a small vapour fraction cuts the mixture density sharply and
# raises the exit speed (1.2% quality from a 0.015 MPa gauge tank gives
# 0.634 m/s against 0.197 m/s for pure liquid).
ln2_speed="${JAXWIND_LN2_SPEED:-}"
# Optional flash quality in [0,1): the mass fraction already vaporised before
# the nozzle. That share is injected as cold nitrogen gas instead of droplets,
# and it also sets the mixture density used to derive the exit speed.
ln2_quality="${JAXWIND_LN2_QUALITY:-}"

command=(
    "$python_executable" -u
    legacy/cases/ConcurrentPrecursor/run_warmup_diagnostics.py
    --config "$config"
    --output-dir "$output"
    --duration-seconds "$duration_seconds"
    --frames 100
    --flow-height 0.876
    --liquid-nitrogen-nozzle
    --ln2-multiphase
    --ln2-mass-flow-kg-s 0.0125
    --ln2-radius 0.005
    --ln2-x "$ln2_x"
    --ln2-y "$ln2_y"
    --ln2-z "$ln2_z"
    --ln2-sigma-x "$ln2_sigma_x"
    --ln2-sigma-r 0.005
)

if [[ -n "$ln2_quality" ]]; then
    command+=(--ln2-vapor-quality "$ln2_quality")
fi

if [[ -n "$ln2_speed" ]]; then
    command+=(--ln2-injection-speed "$ln2_speed")
fi

if [[ -n "$checkpoint" ]]; then
    command+=(--checkpoint "$checkpoint")
fi

"${command[@]}"
