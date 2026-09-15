# Single V80 open-boundary diagnostic baseline

Case: `fv_v80_single_open_debug.toml`. This inherits the 300 s Horns Rev
configuration and replaces the layout with exactly one turbine. No solver,
boundary, advection, or controller correction was applied for this baseline.

- Grid: 128 x 64 x 256; domain: 2048 x 1024 x 1024 m (1/32 the farm cells).
- Spacing retained: 16 x 16 x 4 m. Original vertical extent retained.
- Rotor: (512, 512, 70) m; 512 m upstream and 1536 m downstream clearance.
- Uniform 10 m/s inlet, nonperiodic x/y, lateral pressure outlets, GMG.
- Original AMD, rough wall, fast-RK3, fixed pitch, AD-BEM, and lookup controller.
- dt = 0.25 s, 1200 steps, 300 s; 100 frames and 100 history samples.

## Reproduction

From the repository root, with the project Python environment active:

```bash
export JAXWIND_V80_FAST="$PWD/cases/HornsRev1/turbines/V80/CustomRotor.fst"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m jaxwind check cases/HornsRev1/fv_v80_single_open_debug.toml
python -m jaxwind run cases/HornsRev1/fv_v80_single_open_debug.toml
python tools/analyze_open_v80_debug.py outputs/v80_single_open_debug_300s
```

The completed output directory must not be overwritten when running a comparison;
give comparison cases their own names and output directories. The plotting command
must run after simulation export completes.

## Observed results (2026-09-15)

Completed 1200 steps / 300 s. Maximum sampled CFL: 0.340118; final: 0.313061.
Maximum sampled divergence: 8.20e-8 /s. Reported steady advancement throughput:
67.26 steps/s (excludes startup compilation, output overhead, and plotting).

At 300 s:

- Inlet u faces: exactly 10 m/s throughout the full inlet plane.
- Upstream hub-height cells at x < 352 m: 9.94177 to 10.09713 m/s.
- Maximum adjacent raw u-face jump in that upstream region: 0.16038 m/s.
- Outlet hub-height cell minimum: 6.62029 m/s.
- Saved final hub-height frame agrees exactly with checkpoint face averaging.

The raw-face plot shows weak alternating-grid oscillations. Upstream disturbances
are present by 30 s, when the outlet hub-height minimum is still 9.99976 m/s;
the substantial wake deficit reaches the outlet later. This does not support
explaining all upstream disturbance as a wake wrapping around after exiting.
It does not isolate centered-advection dispersion from global pressure response
or boundary effects, and upstream extrema include legitimate induction.

This is a small diagnostic reproducer, not a validated fix. Narrowing the domain
also moves lateral boundaries closer, so its amplitudes should not be interpreted
as a quantitative reproduction of the 80-turbine farm.

Artifacts in `outputs/v80_single_open_debug_300s/`:

- `debug_summary.png`: wake, enhanced velocity departures, upstream time history,
  and raw-face versus cell-averaged upstream profile.
- `debug_metrics.json`: numerical checks and final upstream metrics.
- `v80_single_hub_u.mp4`: 100 frames at 10 fps.
- `history.csv`, `flow_frames.npz`, `checkpoint.npz`, and standard run metadata.

Configuration validation and explicit inheritance checks passed: one turbine,
unchanged numerical/flow/controller settings, original spacing, 300 s duration.
