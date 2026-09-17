#!/bin/bash -l
#SBATCH --job-name=racz-gas-source
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
py="${JAXWIND_PYTHON:-/u/limo/venvs/numba_cuda_waterboa/bin/python}"
out="$PWD/outputs/spray_closure_validation/racz-gas-${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export PYTHONPATH="$PWD/src:$PWD/tools:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TMPDIR="$out/tmp"
srun --cpu-bind=none "$py" -m pytest -q -o 'python_files=test_*.py' tests/test_racz_gas_proxy.py tests/test_racz_pda.py --junitxml="$out/tests.xml"
srun --cpu-bind=none "$py" tools/qualify_racz_gas_source.py --data outputs/spray_closure_validation/racz_water_25C_03bar --output "$out"
