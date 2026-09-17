#!/bin/bash -l
#SBATCH --job-name=dpm-spatial
#SBATCH --partition=gpudev
#SBATCH --account=rzg_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:15:00
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
sha256sum src/jaxwind/fluent_dpm*.py src/jaxwind/config/fluent_dpm.py src/jaxwind/config/moisture.py src/jaxwind/sgs.py src/jaxwind/spray_paths.py src/jaxwind/physics/fluent_dpm.py src/jaxwind/simulation/open_atmospheric.py src/jaxwind/simulation/stages.py tests/fv/test_fluent_dpm_spatial.py tools/verify_fluent_dpm_spatial.sbatch > "$out/source.sha256"
srun --cpu-bind=none "$py" -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'
srun --cpu-bind=none "$py" -m pytest -x -q -s -o 'python_files=test_*.py' tests/fv/test_fluent_dpm_spatial.py tests/test_moisture_config.py tests/fv/test_spray_moving.py tests/physics/test_fluent_dpm.py tests/physics/test_fluent_dpm_dispersion.py tests/physics/test_fluent_dpm_cell.py --junitxml="$out/tests.xml"
