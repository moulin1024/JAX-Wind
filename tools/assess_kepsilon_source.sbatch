#!/bin/bash -l
#SBATCH --job-name=kepsilon-source
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
out="$PWD/outputs/spray_closure_validation/kepsilon-source-${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu TMPDIR="$out/tmp"
sha256sum tools/march_kepsilon_profile.py tools/jet_streamfunction.py cases/SprayClosureValidation/kepsilon_profile_protocol.json > "$out/source.sha256"
for correction in 0 .79; do
  for step in .000375 .0001875; do
    srun "$py" tools/march_kepsilon_profile.py --c3 "$correction" --n 1000 --step "$step" --output "$out" > "$out/run-${correction}-${step}.log"
  done
done
sha256sum --check "$out/source.sha256"
