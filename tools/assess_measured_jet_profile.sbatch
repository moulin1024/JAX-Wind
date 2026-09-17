#!/bin/bash -l
#SBATCH --job-name=measured-profile
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
out="$PWD/outputs/spray_closure_validation/measured-profile-${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
export TMPDIR="$out/tmp"
sha256sum cases/SprayClosureValidation/measured_profile_protocol.json cases/SprayClosureValidation/reference.json tools/fit_measured_jet_profile.py tools/assess_measured_jet_profile.py tools/assess_measured_jet_profile.sbatch src/jaxwind/jet_profiles.py src/jaxwind/spray_closure.py tests/physics/test_jet_profiles.py > "$out/source_hashes.txt"
srun --cpu-bind=none "$py" -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'
srun --cpu-bind=none "$py" -m pytest -q -o 'python_files=test_*.py' tests/physics/test_jet_profiles.py --junitxml="$out/tests.xml"
srun --cpu-bind=none "$py" tools/fit_measured_jet_profile.py --data outputs/spray_closure_validation/shin_2017 --output "$out"
srun --cpu-bind=none "$py" tools/assess_measured_jet_profile.py --calibration "$out/calibration.json" --output "$out"
