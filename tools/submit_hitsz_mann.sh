#!/usr/bin/env bash
#SBATCH --job-name=hitsz-mann-adbem
#SBATCH --account=rzg_gpu
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=hitsz-mann-adbem-%j.out
#SBATCH --error=hitsz-mann-adbem-%j.err
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
JAXWIND_PYTHON="${JAXWIND_PYTHON:-/raven/u/limo/venvs/numba_cuda_waterboa/bin/python}"
CONFIG="${CONFIG:-cases/HITSZWindTunnel/fv_mann_512x128x256_adbem_90s.toml}"
cd "$REPO_ROOT"
module load cuda/13.0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RUN_OUTPUT="${RUN_OUTPUT:-outputs/hitsz_mann/adbem_512x128x256_${SLURM_JOB_ID}}"
if [[ -n "${RESUME_DIRECTORY:-}" ]]; then
    args=(resume "$RESUME_DIRECTORY")
else
    args=(run "$CONFIG" --output "$RUN_OUTPUT")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
    args+=(--max-steps "$MAX_STEPS")
fi
srun --cpu-bind=none "$JAXWIND_PYTHON" -u -m jaxwind "${args[@]}"
