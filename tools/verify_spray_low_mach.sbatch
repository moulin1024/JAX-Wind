#!/bin/bash -l
#SBATCH --job-name=spray-low-mach
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
out="$PWD/outputs/spray_closure_validation/low-mach-${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
export TMPDIR="$out/tmp"
sha256sum src/jaxwind/spray_low_mach.py src/jaxwind/low_mach.py src/jaxwind/spray_core.py src/jaxwind/spray_coupling.py src/jaxwind/physics/moisture.py src/jaxwind/numerics/poisson.py tests/fv/test_spray_low_mach.py tests/fv/test_low_mach.py tools/verify_spray_low_mach.sbatch > "$out/source_hashes.txt"
srun --cpu-bind=none "$py" -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'
srun --cpu-bind=none "$py" -m pytest -q -o 'python_files=test_*.py' tests/fv/test_spray_low_mach.py tests/fv/test_low_mach.py --junitxml="$out/tests.xml"
