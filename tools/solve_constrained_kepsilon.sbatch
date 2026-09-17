#!/bin/bash -l
#SBATCH --job-name=constrained-kepsilon
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
out="$PWD/outputs/spray_closure_validation/constrained-kepsilon-${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export PYTHONPATH="$PWD/tools:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TMPDIR="$out/tmp"
sha256sum tools/constrained_kepsilon_piecewise.py tools/constrained_kepsilon.py tools/solve_constrained_kepsilon.py tests/test_constrained_kepsilon.py cases/SprayClosureValidation/constrained_kepsilon_protocol.json > "$out/source.sha256"
srun "$py" -m pytest -q tests/test_constrained_kepsilon.py tests/test_jet_realizability.py --junitxml="$out/tests.xml"
srun "$py" tools/solve_constrained_kepsilon.py --edge-solutions "$PWD/outputs/spray_closure_validation/kepsilon-edge-30273563" --output "$out" --split-transition --constrained-seeds "${JAXWIND_CONSTRAINED_SEEDS:-$PWD/outputs/spray_closure_validation/constrained-kepsilon-30273656}"
sha256sum --check "$out/source.sha256"
