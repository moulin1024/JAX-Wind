#!/bin/bash -l
#SBATCH --job-name=dpm-tracking
#SBATCH --partition=gpudev
#SBATCH --account=rzg_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load cuda/13.0
py="${JAXWIND_PYTHON:-/u/limo/venvs/numba_cuda_waterboa/bin/python}"
out="$PWD/outputs/fluent_dpm_verification/${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export PYTHONPATH="$PWD/src:$PWD/tools:$PWD"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
export TMPDIR="$out/tmp"
sha256sum src/jaxwind/fluent_dpm*.py src/jaxwind/config/fluent_dpm.py src/jaxwind/spray_paths.py tests/fv/test_fluent_dpm_spatial.py tests/fv/test_fluent_dpm_tracking.py tools/verify_fluent_dpm_tracking.sbatch > "$out/source.sha256"
srun --cpu-bind=none "$py" -m pytest -x -q -s -o 'python_files=test_*.py' tests/fv/test_fluent_dpm_tracking.py --junitxml="$out/tests.xml"
