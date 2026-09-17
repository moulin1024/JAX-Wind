#!/bin/bash -l
#SBATCH --job-name=kepsilon-edge-stress
#SBATCH --partition=gpudev
#SBATCH --account=rzg_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
py="${JAXWIND_PYTHON:-/u/limo/venvs/numba_cuda_waterboa/bin/python}"
solutions="${JAXWIND_KEPSILON_SOLUTIONS:-$PWD/outputs/spray_closure_validation/kepsilon-similarity-30273498}"
out="$PWD/outputs/spray_closure_validation/kepsilon-edge-${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export PYTHONPATH="$PWD/tools:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TMPDIR="$out/tmp"
sha256sum tools/jet_stress.py tools/audit_jet_stress.py tools/audit_kepsilon_edge.py tools/solve_kepsilon_profile.py tools/kepsilon_similarity.py cases/SprayClosureValidation/kepsilon_edge_audit_protocol.json > "$out/source.sha256"
srun "$py" -m pytest -q tests/test_jet_stress.py --junitxml="$out/tests.xml"
srun "$py" tools/audit_kepsilon_edge.py --solutions "$solutions" --output "$out"
sha256sum --check "$out/source.sha256"
