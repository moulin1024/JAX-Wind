#!/usr/bin/env bash
# Submit the small wake correction + paired open-domain log-law validation.
# Usage: bash tools/submit_backflow_validation.sh [--dry-run] [RUNNER_OPTIONS]
# See --help. Site overrides: PARTITION, ACCOUNT, GRES, CPUS, MEMORY, WALLTIME,
# BACKEND, JAXWIND_PYTHON, ENV_SETUP, CUDA_MODULE, DEPENDENCY.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src:$repo_root/tools${PYTHONPATH:+:$PYTHONPATH}"
export JAXWIND_PYTHON="${JAXWIND_PYTHON:-$HOME/venvs/numba_cuda_waterboa/bin/python}"
export BACKEND="${BACKEND:-cuda}"
if [[ ! -x "$JAXWIND_PYTHON" ]] && ! command -v "$JAXWIND_PYTHON" >/dev/null; then
    echo 'Set JAXWIND_PYTHON to the Python executable in the JAX environment.' >&2
    exit 2
fi
args=(); dry_run=0; has_output=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        --output|--output=*) has_output=1; args+=("$arg") ;;
        --help|-h)
            "$JAXWIND_PYTHON" tools/run_backflow_validation.py --help
            printf '\nSlurm overrides: PARTITION=gpu1 ACCOUNT=rzg_gpu GRES=gpu:a100:1\n'
            printf 'CPUS=8 MEMORY=24G WALLTIME=02:00:00 BACKEND=cuda JAXWIND_PYTHON=...\n'
            printf 'ENV_SETUP=/path/to/environment.sh overrides CUDA module setup.\n'
            printf '%s\n' '--dry-run validates and prepares cases, then prints sbatch without submitting.'
            exit 0 ;;
        *) args+=("$arg") ;;
    esac
done
if (( ! has_output )); then args+=(--output "outputs/backflow_validation_$(date +%Y%m%d_%H%M%S)_$$"); fi
# This validates paths, checkpoint compatibility, timing, and declarations only.
# No JAX device or numerical simulation is started on the login node.
"$JAXWIND_PYTHON" tools/run_backflow_validation.py "${args[@]}" --configure-only
mkdir -p logs/backflow_validation
batch=(sbatch --parsable --nodes=1 --ntasks=1 --cpus-per-task="${CPUS:-8}"
    --partition="${PARTITION:-gpu1}" --gres="${GRES:-gpu:a100:1}"
    --mem="${MEMORY:-24G}" --time="${WALLTIME:-02:00:00}"
    --job-name=backflow-loglaw --chdir="$repo_root" --export=ALL
    --output='logs/backflow_validation/%x-%j.out')
if [[ -n "${ACCOUNT-rzg_gpu}" ]]; then batch+=(--account="${ACCOUNT-rzg_gpu}"); fi
if [[ -n "${DEPENDENCY:-}" ]]; then batch+=(--dependency="$DEPENDENCY"); fi
export BACKFLOW_REPO_ROOT="$repo_root"
batch+=(tools/backflow_validation.sbatch "${args[@]}")
if (( dry_run )); then printf '%q ' "${batch[@]}"; printf '\n'; else "${batch[@]}"; fi
