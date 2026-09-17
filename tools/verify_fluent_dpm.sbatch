#!/bin/bash -l
#SBATCH --job-name=fluent-dpm
#SBATCH --partition=gpudev
#SBATCH --account=rzg_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
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
sha256sum src/jaxwind/physics/fluent_dpm.py src/jaxwind/fluent_dpm_cell.py tests/physics/test_fluent_dpm.py tests/physics/test_fluent_dpm_cell.py cases/FluentDPMWater/baseline.json tools/audit_fluent_dpm_cell.py src/jaxwind/spray_core.py src/jaxwind/spray_coupling.py tools/verify_fluent_dpm.sbatch > "$out/source.sha256"
srun --cpu-bind=none "$py" -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'
srun --cpu-bind=none "$py" -m pytest -q -o 'python_files=test_*.py' tests/physics/test_fluent_dpm.py tests/physics/test_fluent_dpm_cell.py --junitxml="$out/tests.xml"
srun --cpu-bind=none "$py" tools/audit_fluent_dpm_cell.py --output "$out/cell_assessment.json"
