# Pressure comparison fixes — 2026-09-08

Both requested comparisons pass on ravg1023 / A100 with JAX 0.10.0.
All saved fields, including pressure, match the freshly captured baseline
exactly (maximum absolute error 0). A second fresh candidate low-Mach run also
matches exactly. The original rtol=2e-5 / atol=2e-6 comparisons remain intact.

Changes are limited to the verification harness and its documentation:

- Preserve baseline frame requests and timestep overrides in the candidate
  smoke inputs, so output requests do not introduce different block boundaries.
- Pin subprocess PYTHONHASHSEED=0 and enable XLA deterministic GPU operations
  for CUDA verification. A controlled seed-5 experiment reproduced the
  candidate initialization difference in default GPU mode; deterministic GPU
  mode removed it. A fixed Python seed alone was insufficient for open flow
  across baseline and candidate modules.
- Record and require matching process seeds and XLA flags in trajectory reports.
  Older baseline captures must be regenerated with the updated runner.

No numerical kernels or tolerances changed. These deterministic smoke timings
are not production performance measurements. The full regression suite was
not rerun; this result does not clear its previously reported failures.
The edited verification scripts pass compilation, Ruff undefined-name checks,
and git diff --check.

See summary.json and the baseline/candidate scenario reports and logs here.
