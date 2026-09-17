#!/bin/bash -l
#SBATCH --job-name=constrained-profile
#SBATCH --partition=gpudev
#SBATCH --account=rzg_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
py="${JAXWIND_PYTHON:-/u/limo/venvs/numba_cuda_waterboa/bin/python}"
solutions="${JAXWIND_CONSTRAINED_SOLUTIONS:-$PWD/outputs/spray_closure_validation/constrained-kepsilon-30273699}"
out="$PWD/outputs/spray_closure_validation/constrained-profile-${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TMPDIR="$out/tmp"
sha256sum tools/assess_constrained_profile.py cases/SprayClosureValidation/constrained_profile_assessment_protocol.json cases/SprayClosureValidation/reference.json > "$out/source.sha256"
srun "$py" tools/assess_constrained_profile.py --solutions "$solutions" --baseline "$PWD/outputs/spray_closure_validation/kepsilon-similarity-30273498" --output "$out"
sha256sum --check "$out/source.sha256"
