#!/bin/bash -l
#SBATCH --job-name=v80-fine-plots
#SBATCH --partition=gpudev
#SBATCH --account=rzg_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:05:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
out="$PWD/outputs/v80_uniform10_20kgs_200um_290K_refined"
export MPLCONFIGDIR="$out/matplotlib"
py=/u/limo/venvs/numba_cuda_waterboa/bin/python
srun --cpu-bind=none "$py" tools/plot_v80_dpm_uniform.py "$out"
srun --cpu-bind=none "$py" tools/compare_v80_dpm_grids.py
