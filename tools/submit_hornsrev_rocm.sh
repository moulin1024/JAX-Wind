#!/usr/bin/env bash
# Usage: bash tools/submit_hornsrev_rocm.sh prepare|direction|windrose RUNNER_ARGS...
# Optional: PARTITION, WALLTIME, DEPENDENCY=afterok:JOBID, ARRAY_CONCURRENCY.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
case "${1:-}" in
    prepare|direction|windrose) ;;
    *) echo 'Expected prepare, direction, or windrose followed by runner arguments' >&2; exit 2 ;;
esac
partition="${PARTITION:-hx1hdnormal01}"
if ! sinfo --noheader --partition="$partition" --format='%P' | tr -d '*' | awk -v p="$partition" '$1 == p {found=1} END {exit !found}'; then
    echo "ROCm partition $partition is unavailable on this cluster; submit from the ROCm login host." >&2
    exit 2
fi
mkdir -p logs/hornsrev
batch=(--parsable --partition="$partition" --time="${WALLTIME:-48:00:00}"
    --chdir="$REPO_ROOT" --export=ALL --output='logs/hornsrev/%x-%A_%a.out'
    --job-name="hornsrev-$1")
if [[ -n "${DEPENDENCY:-}" ]]; then batch+=(--dependency="$DEPENDENCY"); fi
if [[ "$1" == windrose ]]; then
    batch+=(--array="0-11%${ARRAY_CONCURRENCY:-1}")
fi
export HORNSREV_REPO_ROOT="$REPO_ROOT"
sbatch "${batch[@]}" tools/hornsrev_rocm.sbatch "$@"
