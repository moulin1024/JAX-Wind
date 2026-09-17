#!/bin/bash -l
#SBATCH --job-name=racz-probe
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
py="${JAXWIND_PYTHON:-/u/limo/venvs/numba_cuda_waterboa/bin/python}"
out="$PWD/outputs/spray_closure_validation/racz-probe-${SLURM_JOB_ID}"
mkdir -p "$out/tmp"
export PYTHONPATH="$PWD/src:$PWD/tools:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TMPDIR="$out/tmp"
sha256sum tools/racz_probe.py tools/audit_racz_probe.py tools/racz_pda.py tests/test_racz_probe.py tests/test_racz_pda.py cases/WaterSprayRacz2025/probe_protocol.json cases/WaterSprayRacz2025/protocol.json tools/audit_racz_probe.sbatch > "$out/source_hashes.txt"
srun --cpu-bind=none "$py" -m pytest -q -o 'python_files=test_*.py' tests/test_racz_probe.py tests/test_racz_pda.py --junitxml="$out/tests.xml"
srun --cpu-bind=none "$py" tools/audit_racz_probe.py --data outputs/spray_closure_validation/racz_water_25C_03bar --output "$out"
