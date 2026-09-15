# Refactor verification — 2026-09-08

Status: FAILED; numerical acceptance remains open.

Executed on ravg1023, Slurm job 30072440, NVIDIA A100-SXM4-40GB,
Python 3.13 / JAX 0.10.0, platform cuda. Candidate: feat/refactor working
tree (including uncommitted changes). Baseline: f7d83625e481f0fcba04e73bfb3482eff043a603.

The initial baseline run exposed a smoke-input frame-count error. The only
source change made during verification was to tools/verification/scenario.py:
bound precursor/main frame counts to two in both baseline and candidate
six-step workflow inputs. The full baseline and candidate were then run again
in baseline-cuda-v2 and candidate-cuda-v2. Original logs are retained.
Ruff 0.16.6 was installed in the existing verification Python environment.

## Results

- Baseline: all six scenarios captured successfully.
- Candidate: Boussinesq, adaptive, incompressible jet, and low-Mach jet passed.
- Open flow: pressure failed (434/512 elements, maximum absolute error 0.1136474609375).
- Periodic low-Mach: pressure failed (396/512 elements, maximum absolute error 1.1298398021608591e-5).
- All other saved state fields passed. Tolerances: rtol=2e-5, atol=2e-6; integer/boolean fields exact.
- Pytest: 207 passed, 9 failed, 1 skipped, 10 subtests passed, 698.93 seconds.
- Compilation and installed CLI checks passed.
- Ruff undefined-name check failed: dataclass at src/jaxwind/runtime/periodic.py:26.

## Test failures

1. Four AMG tests: jaxamg imports a CPU device, but JAX_PLATFORMS=cuda disables that backend. This is an environment/runner coverage limitation; AMG is not verified.
2. HITSZ workflow duration: expected 90 seconds, configured 180 seconds.
3. Periodic runtime import: undefined dataclass at line 26 (a decorator on run_periodic_blocks).
4. Derived low-Mach mesh: record_plane lies outside the derived mesh.
5. Adaptive runtime/resume: simulation made no finite forward progress.
6. Low-Mach continuation: record_plane lies outside the derived mesh.

The adaptive smoke timing sets performance_review_required. These tiny timings
are not sufficient to conclude a performance regression; repeat measurements
remain pending. Extended tests, separate CPU/ROCm runs, and production-scale
performance measurements were not run.

## Artifacts

- baseline-cuda-v2/summary.json
- candidate-cuda-v2/summary.json
- candidate-cuda-v2/pytest.log
- candidate-cuda-v2/undefined-names.log
- state-comparison.json (all state fields, including failed scenarios)
- Per-scenario state.npz, report.json where successful, and logs in each phase.
