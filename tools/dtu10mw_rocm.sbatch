#!/usr/bin/env bash
#SBATCH --job-name=dtu10mw
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --time=08:00:00
set -euo pipefail
REPO_ROOT="${DTU10MW_REPO_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$REPO_ROOT"
module purge
module load compiler/dtk/26.04
module load mpi/openmpi/openmpi-4.1.5-gcc9.3.0
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-jax060}"
export LD_LIBRARY_PATH="/public/software/compiler/dtk-26.04/dcc/gcvm/lib:${LD_LIBRARY_PATH:-}"
MPI_LIBDIR="$(mpicc --showme:libdirs | awk '{print $1}')"
export LD_PRELOAD="${MPI_LIBDIR}/libmpi.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
args=("$@")
# Backend is appended deliberately: a submitted job cannot silently select CPU.
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --tres-per-task=gres/dcu:1 --kill-on-bad-exit=1 \
    python -u tools/run_dtu10mw.py "${args[@]}" --backend rocm
