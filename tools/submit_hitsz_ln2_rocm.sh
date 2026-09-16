#!/usr/bin/env bash
# Submit the configured four-stage workflow only. sbatch options may follow SETUP.
# Usage: bash tools/submit_hitsz_ln2_rocm.sh [SETUP_DIRECTORY] [SBATCH_OPTIONS...]
# Resume: RESUME=1 bash tools/submit_hitsz_ln2_rocm.sh SETUP_DIRECTORY
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP="${1:-${REPO_ROOT}/outputs/hitsz_ln2_512_setup}"
if (( $# )); then shift; fi
SETUP="$(cd -- "${SETUP}" && pwd)"
[[ -f "${SETUP}/workflow.toml" ]] || { echo 'Run configure_hitsz_ln2.sh first.' >&2; exit 2; }
[[ "${RESUME:-0}" =~ ^[01]$ ]] || { echo 'RESUME must be 0 or 1.' >&2; exit 2; }
# Reuse the repository ROCm launcher: DTK 26.04, OpenMPI 4.1.5,
# conda jax060, MPI preload, JAX_PLATFORMS=rocm, and device preflight.
export REPO_ROOT CONFIG="${SETUP}/workflow.toml" STAGE=all
export RUN_OUTPUT="${SETUP}/run" RESUME="${RESUME:-0}"
export CONDA_ENV="${CONDA_ENV:-jax060}"
# Prevent inherited smoke-test settings from silently shortening this full run.
unset MAX_STEPS
cd -- "${REPO_ROOT}"
exec sbatch --export=ALL --job-name=hitsz-ln2-512 --time="${WALLTIME:-48:00:00}" \
    --output="${SETUP}/slurm-%j.out" --error="${SETUP}/slurm-%j.err" \
    "$@" "${REPO_ROOT}/tools/submit_fv_abl_rocm_single.sh"
