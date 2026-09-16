#!/bin/bash -l
#SBATCH --job-name=backflow-loglaw
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
repo_root="${BACKFLOW_REPO_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$repo_root"
if [[ -n "${ENV_SETUP:-}" ]]; then
    source "$ENV_SETUP"
elif [[ "${BACKEND:-cuda}" == cuda && -n "${CUDA_MODULE-cuda/13.0}" ]]; then
    module load "${CUDA_MODULE-cuda/13.0}"
fi
export JAX_PLATFORMS="${BACKEND:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$repo_root/src:$repo_root/tools${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/backflow-matplotlib-${SLURM_JOB_ID:-local}"
export JAXWIND_V80_FAST="${JAXWIND_V80_FAST:-$repo_root/cases/HornsRev1/turbines/V80/CustomRotor.fst}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --cpu-bind=none \
    "${JAXWIND_PYTHON:?}" -u tools/run_backflow_validation.py "$@"
