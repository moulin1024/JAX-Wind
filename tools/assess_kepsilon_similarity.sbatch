#!/bin/bash -l
#SBATCH --job-name=kepsilon-similarity
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
march="${JAXWIND_KEPSILON_MARCH:-$PWD/outputs/spray_closure_validation/kepsilon-source-30273474}"
out="$PWD/outputs/spray_closure_validation/kepsilon-similarity-${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/tools:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu TMPDIR="$out/tmp"
sha256sum tools/solve_kepsilon_profile.py tools/kepsilon_similarity.py cases/SprayClosureValidation/kepsilon_profile_protocol.json cases/SprayClosureValidation/kepsilon_production_audit_protocol.json cases/SprayClosureValidation/kepsilon_coefficient_audit_protocol.json > "$out/source.sha256"
srun "$py" -m pytest -q tests/test_kepsilon_similarity.py tests/test_jet_streamfunction.py --junitxml="$out/tests.xml"
for pair in pope_printed standard; do
  for correction in 0 .79; do
    srun "$py" tools/solve_kepsilon_profile.py --march-directory "$march" --output "$out" --pair "$pair" --c3 "$correction" > "$out/${pair}-${correction}-shear.log"
    srun "$py" tools/solve_kepsilon_profile.py --march-directory "$march" --output "$out" --pair "$pair" --c3 "$correction" --normal-strain > "$out/${pair}-${correction}-normal.log"
  done
done
sha256sum --check "$out/source.sha256"
