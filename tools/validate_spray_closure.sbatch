#!/bin/bash -l
#SBATCH --job-name=spray-closure-validation
#SBATCH --partition=gpudev
#SBATCH --account=rzg_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:05:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load cuda/13.0
py="${JAXWIND_PYTHON:-/u/limo/venvs/numba_cuda_waterboa/bin/python}"
out="$PWD/outputs/spray_closure_validation/job-${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export PYTHONPATH="$PWD/src:$PWD/tools${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TMPDIR="$out/tmp"
srun --cpu-bind=none "$py" -m pytest -q -o 'python_files=test_*.py' tests/physics/test_spray_closure.py --junitxml="$out/tests.xml"
srun --cpu-bind=none "$py" tools/validate_spray_closure.py --output "$out"
