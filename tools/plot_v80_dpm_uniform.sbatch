#!/bin/bash -l
#SBATCH --job-name=v80-spray-plots
#SBATCH --partition=gpudev
#SBATCH --account=rzg_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="$PWD/outputs/v80_uniform10_20kgs_200um_290K/matplotlib"
srun --cpu-bind=none /u/limo/venvs/numba_cuda_waterboa/bin/python tools/plot_v80_dpm_uniform.py outputs/v80_uniform10_20kgs_200um_290K
