#!/bin/bash -l
#SBATCH --job-name=pope-1990-holdout
#SBATCH --partition=gpudev
#SBATCH --account=rzg_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
py="${JAXWIND_PYTHON:-/u/limo/venvs/numba_cuda_waterboa/bin/python}"
solutions="${JAXWIND_KEPSILON_SOLUTIONS:-$PWD/outputs/spray_closure_validation/kepsilon-similarity-30273498}"
out="$PWD/outputs/spray_closure_validation/pope-1990-${SLURM_JOB_ID}"
mkdir -p "$out/tmp" "$out/mpl"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TMPDIR="$out/tmp" MPLCONFIGDIR="$out/mpl"
sha256sum tools/assess_pope_1990_profile.py cases/SprayClosureValidation/pope_1990_profile_protocol.json cases/SprayClosureValidation/reference.json > "$out/source.sha256"
srun "$py" tools/assess_pope_1990_profile.py --solutions "$solutions" --output "$out"
sha256sum --check "$out/source.sha256"
