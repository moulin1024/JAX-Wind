#!/bin/bash -l
#SBATCH --job-name=v80-spray-fine
#SBATCH --partition=gpudev
#SBATCH --account=rzg_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load cuda/13.0
py="${JAXWIND_PYTHON:-/u/limo/venvs/numba_cuda_waterboa/bin/python}"
out="$PWD/outputs/v80_uniform10_20kgs_200um_290K_refined"
mkdir -p "$out/jobs/${SLURM_JOB_ID}/tmp"
export PYTHONPATH="$PWD/src:$PWD/tools:$PWD"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
export TMPDIR="$out/jobs/${SLURM_JOB_ID}/tmp"
export JAX_COMPILATION_CACHE_DIR="$out/jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
sha256sum src/jaxwind/fluent_dpm*.py src/jaxwind/config/fluent_dpm.py src/jaxwind/sgs.py src/jaxwind/spray_paths.py src/jaxwind/physics/fluent_dpm.py src/jaxwind/simulation/open_atmospheric.py cases/FluentDPMWater/v80_uniform10_20kgs_200um*.toml tools/run_v80_dpm_uniform.py tools/run_v80_dpm_uniform_refined.sbatch > "$out/jobs/${SLURM_JOB_ID}/source.sha256"
srun --cpu-bind=none "$py" -u tools/run_v80_dpm_uniform.py --case cases/FluentDPMWater/v80_uniform10_20kgs_200um_refined.toml --output "$out" --steps 300 --wall-seconds 660
