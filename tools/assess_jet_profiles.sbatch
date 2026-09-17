#!/bin/bash -l
#SBATCH --job-name=jet-profile-assessment
#SBATCH --partition=gpudev
#SBATCH --account=rzg_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load cuda/13.0
py="${JAXWIND_PYTHON:-/u/limo/venvs/numba_cuda_waterboa/bin/python}"
out="$PWD/outputs/spray_closure_validation/profile-${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TMPDIR="$out/tmp"
srun --cpu-bind=none "$py" -m pytest -q -o 'python_files=test_*.py' tests/physics/test_jet_profiles.py --junitxml="$out/tests.xml"
srun --cpu-bind=none "$py" tools/assess_jet_profiles.py --output "$out"
