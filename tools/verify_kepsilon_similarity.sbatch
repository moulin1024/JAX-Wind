#!/bin/bash -l
#SBATCH --job-name=kepsilon-equations
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
out="$PWD/outputs/spray_closure_validation/kepsilon-equations-${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export PYTHONPATH="$PWD/tools:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TMPDIR="$out/tmp"
sha256sum tools/kepsilon_similarity.py tests/test_kepsilon_similarity.py cases/SprayClosureValidation/kepsilon_profile_protocol.json > "$out/source.sha256"
srun "$py" -m pytest -q tests/test_kepsilon_similarity.py --junitxml="$out/tests.xml"
sha256sum --check "$out/source.sha256"
