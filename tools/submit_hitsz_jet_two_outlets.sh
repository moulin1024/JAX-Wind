#!/usr/bin/env bash
#SBATCH --job-name=hitsz-jet-1024
#SBATCH --account=rzg_gpu
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=hitsz-jet-1024-%j.out
#SBATCH --error=hitsz-jet-1024-%j.err
set -euo pipefail

# Submit from the repository root. Override site resources with sbatch options.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}}"
JAXWIND_PYTHON="${JAXWIND_PYTHON:-/raven/u/limo/venvs/numba_cuda_waterboa/bin/python}"
CONFIG="${CONFIG:-cases/HITSZWindTunnel/fv_1024x256x512_l24_jet_only_two_outlets_1s.toml}"
cd "$REPO_ROOT"
module load cuda/13.0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

args=(run "$CONFIG")
# A unique output makes repeat submissions non-destructive.
args+=(--output "${RUN_OUTPUT:-outputs/hitsz_ln2_jet/two_outlets_1024x256x512_${SLURM_JOB_ID}}")
if [[ -n "${MAX_STEPS:-}" ]]; then
    args+=(--max-steps "$MAX_STEPS")
fi
srun --cpu-bind=none "$JAXWIND_PYTHON" -u -m jaxwind "${args[@]}"
