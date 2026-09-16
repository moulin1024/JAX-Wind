#!/usr/bin/env bash
# Usage: bash tools/submit_dtu10mw_rocm.sh prepare|main RUNNER_ARGS...
# Optional: PARTITION, WALLTIME, DEPENDENCY=afterok:JOBID.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
case "${1:-}" in
    prepare|main) ;;
    *) echo 'Expected prepare or main followed by runner arguments' >&2; exit 2 ;;
esac
partition="${PARTITION:-hx1hdnormal01}"
if ! sinfo --noheader --partition="$partition" --format='%P' | tr -d '*' | awk -v p="$partition" '$1 == p {found=1} END {exit !found}'; then
    echo "ROCm partition $partition is unavailable on this cluster; submit from the ROCm login host." >&2
    exit 2
fi
mkdir -p logs/dtu10mw
batch=(--parsable --partition="$partition" --time="${WALLTIME:-08:00:00}"
    --chdir="$REPO_ROOT" --export=ALL --output='logs/dtu10mw/%x-%A_%a.out'
    --job-name="dtu10mw-$1")
if [[ -n "${DEPENDENCY:-}" ]]; then batch+=(--dependency="$DEPENDENCY"); fi
export DTU10MW_REPO_ROOT="$REPO_ROOT"
sbatch "${batch[@]}" tools/dtu10mw_rocm.sbatch "$@"
