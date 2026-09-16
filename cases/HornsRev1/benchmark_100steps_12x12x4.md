# 100-step warmup timing: 12 x 12 x 4 m

Measured 2026-09-15 on ravg1177, allocation 30244593, one A100-SXM4-40GB,
JAX 0.10.0. Grid: 512 x 512 x 256; domain: 6144 x 6144 x 1024 m.
The benchmark configuration inherits the 12 m case and uses 25-step runtime
blocks so 100 steps contain two startup blocks and two steady timing blocks.
Physics: float32, periodic FFT/Thomas projection, RK3, adaptive CFL 0.9.

| Measurement | Result |
|---|---|
| Steps advanced | 100 |
| Simulated time | 104.302032 s |
| Last 50 steps, advancement only | 9.618145 s |
| Steady wall time per step | 0.192363 s |
| Steady steps per second | 5.198507 |
| All advancement blocks, including first compilation | 33.630091 s |
| End-to-end command wall time | 373.77 s (6 min 14 s) |
| Observed GPU allocation peak, sampled every 10 s | 31,181 MiB (30.45 GiB) |
| Final maximum divergence | 8.0094e-8 s^-1 |
| Final compressed checkpoint | 1.565 GiB |

Steady advancement timing synchronizes device work and excludes startup,
diagnostics, and checkpoint I/O. End-to-end time includes all these costs,
including compressed initial and final checkpoints. It averages 3.7377 s/step
for this short invocation but is not representative of sustained advancement.
The final reported CFL of 5.1069 uses the 6 s timestep cap; the driver selects
each actual timestep from the configured adaptive CFL ceiling of 0.9.

The run exited successfully and is paused at step 100 with its checkpoint
verified. This short benchmark does not establish long-run stability.

## Command

From the repository on the allocated compute node, after loading cuda/13.0:

```bash
/usr/bin/time -p env JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_FLAGS=--xla_gpu_autotune_level=0 PYTHONPATH=src \
  /raven/u/limo/venvs/numba_cuda_waterboa/bin/python -u -m jaxwind workflow \
  cases/HornsRev1/fv_warmup_benchmark_512x512x256_12x12x4.toml \
  --stage warmup \
  --output outputs/hornsrev1_512x512x256_12x12x4_100steps_30244593 \
  --max-steps 100
```

The output already exists. Use a fresh output for a repeat, or replace
`--max-steps 100` with `--resume` to continue the saved warmup.
