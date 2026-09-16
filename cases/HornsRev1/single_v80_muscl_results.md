# Single V80: centered versus limited upwind momentum advection

## Implementation

The optional numerical setting is:

```toml
[numerics]
momentum_advection_scheme = "muscl-mc"
```

The default remains `central`. `muscl-mc` is implemented in
`src/jaxwind/numerics/momentum.py` and routed through `FlowModel` and both
atmospheric model builders. It supports uniform Cartesian grids, including
anisotropic spacing. Mapped grids are explicitly rejected.

Transported momentum uses piecewise-linear MUSCL reconstruction and the MC
(monotonized-central) slope limiter. The advecting mass velocity retains the
arithmetic interpolation of the MAC face flux onto momentum control-volume
faces. Shared conservative flux differences are used in all three directions.
Slopes vanish at extrema and nonperiodic endpoints; no opposite-edge stencil
is used unless that direction is explicitly periodic.

This is a production candidate, not a general proof of boundedness: the complete
pressure-coupled fast-RK3 scheme is not asserted to be TVD. Existing boundary
enforcement, normal-face endpoint tendencies, SGS, and pressure treatment are
unchanged. This implementation does not resolve the separate lateral-outflow
audit or make the pressure outlet nonreflecting.

## Controlled comparison

`fv_v80_single_open_debug_muscl.toml` inherits the centered baseline
`fv_v80_single_open_debug.toml`. Only the numerical advection setting, case name,
and output directory differ. Both use one V80, 128 x 64 x 256 cells, 16 x 16 x 4 m
spacing, uniform 10 m/s inflow, open x/y boundaries, GMG, dt = 0.25 s, and 1200
steps (300 s). Both have 100 frames and 100 history samples. Baseline files and
the 80-turbine case were not modified or rerun.

| Diagnostic | Centered | MUSCL-MC |
| --- | ---: | ---: |
| Maximum sampled CFL | 0.340118 | 0.264812 |
| Maximum sampled divergence [1/s] | 8.20e-8 | 7.08e-8 |
| Final upstream maximum adjacent u-face jump [m/s] | 0.160380 | 0.010297 |
| Final upstream second-difference RMS [m/s] | 0.038527 | 0.000320 |
| Final turbine lookup power [MW] | 1.269116 | 1.265442 |
| Reported steady advancement throughput [steps/s] | 67.26 | 55.77 |

Upstream metrics use the hub-height plane and x < 352 m, at least 2 rotor
diameters ahead of the rotor. Adjacent jumps decreased by 93.58%, and the RMS
second difference decreased by 99.17%. These diagnostics also include smooth
physical gradients; neither is an exact separation of physical and numerical
signals. Throughput excludes compilation/output and is not a controlled hardware
benchmark.

The final MUSCL inlet u faces remain exactly 10 m/s. The saved hub-height frame
matches checkpoint face averaging exactly. Upstream cell-centred velocities at
300 s range from 9.94928 to 10.02079 m/s, with a smooth near-axis profile rather
than the centered scheme's grid-to-grid oscillations. The apparent inlet streak
is not visible in the final MUSCL map; the time-history extrema are also smooth.
This strongly implicates the centered momentum discretization in the observed
artifact, without establishing that all remaining boundary effects are absent.

The downstream wake changes as well: in this uniform-inflow case MUSCL suppresses
unsteady grid-scale structure and yields a deeper far wake at 300 s. Removing
oscillations does not by itself validate wake recovery or turbulent mixing. A
grid/time convergence and outlet-placement study is still needed before treating
this as validated full-farm physics.

## Artifacts and reproduction

Results: `outputs/v80_single_open_debug_muscl_300s/`

- `advection_comparison.png` and `.json`: baseline comparison.
- `debug_summary.png` and `debug_metrics.json`: MUSCL-only diagnostics.
- `v80_single_hub_u.mp4`: 100 frames at 10 fps.
- Standard CSV history, saved frames, checkpoint, and run metadata.

Use the same environment exports as `single_v80_open_debug.md`, then:

```bash
python -m jaxwind check cases/HornsRev1/fv_v80_single_open_debug_muscl.toml
python -m jaxwind run cases/HornsRev1/fv_v80_single_open_debug_muscl.toml
python tools/analyze_open_v80_debug.py outputs/v80_single_open_debug_muscl_300s
python tools/compare_open_v80_advection.py outputs/v80_single_open_debug_300s outputs/v80_single_open_debug_muscl_300s
```

Do not overwrite the completed output directory for a new comparison run.

## Verification

16 MUSCL tests passed: positive/negative transport, limiter behavior, smooth-flow
second-order convergence, conservation, shape/uniform preservation, no opposite-
edge coupling, model routing, default preservation, and mapped-grid rejection.
41 existing operator/integration/open-boundary/farm tests (plus 4 subtests) passed.
The completed GPU comparison retained the original CFL safety guard.
