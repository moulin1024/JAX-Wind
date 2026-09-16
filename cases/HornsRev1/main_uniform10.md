# Centered 80-turbine Horns Rev 1 main case

## Current shorter restart

The original dt=0.5 s run stopped on its CFL guard at 1.03569, after the last
logged state at step 420 / 210 s. Its output directory is preserved.

`fv_hornsrev1_80_uniform10_open_gmg_300s.toml` restarts from uniform flow,
retaining the centered 80-turbine geometry and boundary conditions but using
dt=0.25 s, 1200 steps (300 s), and 10 frames at 30, 60, ..., 300 s.
It checkpoints at 150 s and at completion. The new output is
`outputs/hornsrev1_80_v80_uniform10_open_gmg_300s_dt025`; the live log is its
sibling `hornsrev1_80_v80_uniform10_open_gmg_300s_dt025.log`.
The launch chains rendering after successful completion, using 2 fps for a
5-second, 10-frame MP4. The renderer now follows the case's configured frame
count instead of requiring 100 frames. The run is not marked complete here.

The remainder documents the original one-hour configuration and common setup.

Configuration: `fv_hornsrev1_80_uniform10_open_gmg.toml`.

- User-supplied coordinates are preserved in order (T01-T80) and translated
  by (+1337, +2150.5) m. Both arithmetic centroid and bounding-box center are
  (4096, 4096) m. Farm bounds are x=1337-6855 m, y=2150.5-6041.5 m.
- Domain: 8192 x 8192 x 1024 m; grid: 512 x 512 x 256 (16 x 16 x 4 m).
- Direct initialization: uniform u=10 m/s, v=w=0, no precursor or warmup.
  The profile filename retained in the shared schema is not used to initialize
  this direct uniform-inflow builder.
- Inlet x=0: enforced u=10 m/s, v=w=0 at every integration stage.
- Outlet x=Lx and both y sides: zero-normal-gradient velocity extrapolation
  before projection and zero pressure at outlet faces. The projection determines
  final normal flux; lateral boundaries are not periodic or impermeable walls.
  Pressure outlets may permit entrainment/backflow; these are not one-way valves.
- Bottom: impermeable rough-wall surface model, z0=0.0002 m. Top: impermeable
  free slip. Pressure forcing and Coriolis are disabled.
- GMG pressure projection, relative tolerance 1e-5, float32, fast-RK3,
  fixed dt=0.5 s. One hour = 7200 steps. Each turbine uses its own filtered
  1D-upstream rotor-area wind for RPM and lookup electrical power.
- The inherited V80 lookup assumptions remain: approximate figure-read RPM,
  reference power table, fixed aerodynamic pitch. See [V80 lookup](v80_lookup.md).
  Imposed lookup power does not validate AD-BEM torque or above-rated wakes.

## Implementation and verification

The uniform-farm path adds explicit nonperiodic lateral pressure outlets to
GMG and the open atmospheric integrator. Existing boundary defaults remain
unchanged. The new pressure operator is checked for symmetry, positivity,
correct diagonal, and divergence removal while preserving inlet velocity.

To avoid evaluating every rotor on all 67 million cells, AD-BEM is evaluated
on local patches extending six Gaussian smoothing widths beyond the rotor
support. Open-domain kernel distances do not wrap. Patch forces are accumulated
on the global MAC faces. A full-domain comparison checks this truncation on a
small mesh; omitted Gaussian tails are negligible at float32 precision.

Ten new/existing open-boundary tests passed before the full-grid launch.
The full-resolution configuration also passed `python -m jaxwind check`.
The pressure regression suite passed 20 tests and 6 subtests. A final repeat
of the four uniform-farm tests passed after adding the CFL safety guard.
All 25 existing periodic farm/controller tests also passed after the changes.
These checks establish implementation behavior, not a validated Horns Rev
power or wake prediction under uniform, turbulence-free inflow.

### Full-grid startup check

On A100 node `ravg1073`, allocation `30246936`, the actual 80-turbine case
completed 36 steps / 18 simulated seconds and saved a restart checkpoint.
The sampled CFL values were 0.7180, 0.6450 and 0.6844; maximum recorded
divergence was 1.31e-7 1/s. All history diagnostics were finite. GPU memory
observed after the first block was about 17.8 GB. Steady throughput over the
third block was 2.373 steps/s (0.421 s/step), excluding checkpoint I/O.
The final benchmark farm lookup power was 103.07 MW.

The same saved run was then resumed toward 3600 s; this is not a claim that
the one-hour run has finished. Progress is logged in `main_run.log` under
the output directory. The launch command chains MP4 rendering after successful
simulation completion. A nonfinite or >1 CFL at a block boundary stops the
uniform-farm run; it does not silently change dt or boundary conditions.

## Run and animation

On the allocated GPU node, using the project Python environment:

```bash
export JAXWIND_V80_FAST="$PWD/cases/HornsRev1/turbines/V80/CustomRotor.fst"
python -m jaxwind run cases/HornsRev1/fv_hornsrev1_80_uniform10_open_gmg.toml
# Continue a deliberately paused/checkpointed run:
python -m jaxwind resume outputs/hornsrev1_80_v80_uniform10_open_gmg_1h
python tools/render_hornsrev1_farm.py outputs/hornsrev1_80_v80_uniform10_open_gmg_1h
```

The requested 100 hub-height u frames are captured every 36 simulated seconds.
The renderer produces `hornsrev1_hub_u.mp4` (100 frames, 10 fps) and a final-frame
PNG, showing the full domain with all 80 rotor locations. Per-turbine wind,
RPM and lookup power, plus total farm power, are recorded in `history.csv`.
