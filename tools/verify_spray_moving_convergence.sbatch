#!/bin/bash -l
#SBATCH --job-name=spray-motion-dt
#SBATCH --partition=gpudev
#SBATCH --account=rzg_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load cuda/13.0
py="${JAXWIND_PYTHON:-/u/limo/venvs/numba_cuda_waterboa/bin/python}"
out="$PWD/outputs/spray_closure_validation/moving-dt-${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
export TMPDIR="$out/tmp"
sha256sum tests/fv/test_spray_moving_convergence.py src/jaxwind/spray_paths.py src/jaxwind/spray_moving.py tests/fv/test_spray_moving.py src/jaxwind/spray_cell_exchange.py src/jaxwind/spray_cell_step.py src/jaxwind/spray_carrier.py src/jaxwind/spray_pressure.py src/jaxwind/spray_momentum.py src/jaxwind/spray_low_mach.py src/jaxwind/spray_core.py tests/fv/test_spray_cell_step.py tools/verify_spray_moving_convergence.sbatch > "$out/source_hashes.txt"
srun --cpu-bind=none "$py" -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'
srun --cpu-bind=none "$py" -m pytest -q -o 'python_files=test_*.py' tests/fv/test_spray_moving_convergence.py "$@" --junitxml="$out/tests.xml"
