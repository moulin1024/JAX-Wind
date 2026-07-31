#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repository_root}"

export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS="cuda"
export JAX_ENABLE_X64="0"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED="1"

exec "${PYTHON:-python}" benchmark/NeutralLogLawAMD/run.py \
  --nx 64 --ny 64 --nz 64 \
  --single --target-cfl 0.5 \
  --linear-solver pcg --krylov-execution jax \
  --pressure-rtol 1e-6 --pressure-max-iterations 20 \
  --projection-method full \
  --sgs lasd --lasd-update-interval 2 \
  --wall-matching-level 2 --wall-filter-width 3 \
  --wall-temporal-filter-gamma 1 \
  --steps 32000 --sample-start-step 16000 \
  --sample-every 10 --log-every 500 --checkpoint-every 500 \
  --flow-frame-count 100 --flow-frame-start-step 1 --flow-gif-fps 10 \
  --output-dir \
    benchmark_results/neutral_loglaw_lasd_64cubed_generalized_32000 \
  "$@"
