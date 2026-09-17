#!/bin/bash -l
#SBATCH --job-name=jet-stress-audit
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
solutions="${JAXWIND_KEPSILON_SOLUTIONS:-$PWD/outputs/spray_closure_validation/kepsilon-similarity-30273498}"
out="$PWD/outputs/spray_closure_validation/jet-stress-${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export PYTHONPATH="$PWD/tools:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TMPDIR="$out/tmp"
sha256sum tools/jet_stress.py tools/audit_jet_stress.py tests/test_jet_stress.py cases/SprayClosureValidation/kepsilon_momentum_audit_protocol.json > "$out/source.sha256"
srun "$py" -m pytest -q tests/test_jet_stress.py --junitxml="$out/tests.xml"
srun "$py" tools/audit_jet_stress.py --solutions "$solutions" --output "$out"
sha256sum --check "$out/source.sha256"
