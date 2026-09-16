# HornsRev1

## Shared precursor and parameterized wind directions

See [directional workflow](directional_workflow.md) for the 10 h warmup + 1 h
shared precursor + 1 h per-direction main setup. Generate a case with
`python tools/create_hornsrev_case.py --wind-direction 270` (or another angle).
All directions reuse the same recording, amplitude-scaled to their wind-rose
mean speed; the main domain is nonperiodic. The selected first direction is 270°.
The [English wind rose](wind_rose.csv) preserves the supplied values.

> Run all commands below on a compute node, including configuration checks.
> This case uses schema version 1. Historical outputs cannot be resumed;
> regenerate inputs in the new format. See [verification](../../doc/verification.md).

Neutral offshore precursor for the Horns Rev 1 wind farm (Vestas V80-2.0 MW,
80 m rotor, 70 m hub height, 7D spacing). The directional workflow above provides
shared warmup/precursor development and independent open-boundary farm mains.
Historical warmup and single-V80 AD-BEM examples are also provided below.

## V80 turbine setup

[`fv_workflow_v80.toml`](fv_workflow_v80.toml) declares the supplied 80 m rotor
using the same `[physics.turbine]` format as the DTU10MW case. Its case-local
OpenFAST-compatible data are under [`turbines/V80`](turbines/V80/README.md).

```bash
export JAXWIND_V80_FAST="$PWD/cases/HornsRev1/turbines/V80/CustomRotor.fst"
jaxwind check cases/HornsRev1/fv_workflow_v80.toml
```

The example uses Horns Rev's 70 m hub height and offshore inflow, with a
single turbine in the 8192 x 8192 x 1024 m domain. It preserves the supplied
blade and polar data. Nacelle/tower drag is disabled because V80 body
dimensions were not supplied. See the turbine README for stage durations,
source-data limits, and running instructions.

### One-hour periodic V80 smoke run

[`fv_v80_periodic_smoke_512x512x256_1h.toml`](fv_v80_periodic_smoke_512x512x256_1h.toml)
uses 512 x 512 x 256 cells in an 8192 x 8192 x 1024 m domain
(16 x 16 x 4 m). The single V80 is active from startup at
(4096, 4096, 70) m, with 16.7 rpm and zero blade pitch. Use `jaxwind run`
for this case, not the turbine-free `workflow --stage warmup` command.

```bash
export JAXWIND_V80_FAST="$PWD/cases/HornsRev1/turbines/V80/CustomRotor.fst"
jaxwind run cases/HornsRev1/fv_v80_periodic_smoke_512x512x256_1h.toml
python tools/render_v80_smoke.py outputs/hornsrev1_v80_periodic_smoke_1h_30244593
```

The adaptive run ends at 3600 simulated seconds and stores 100 hub-height
and centerline frames, at 36-second intervals. The MP4 shows a near-wake
crop of both planes at 10 fps; full-domain planes remain in `flow_frames.npz`.
This starts from the prescribed log profile plus noise, not a fully developed
precursor. Horizontal boundaries are periodic, so the wake can recirculate
during the hour. This is a single-turbine smoke test, not the full farm.

## Independently controlled turbines

The [centered 80-turbine direct main case](main_uniform10.md) uses uniform
10 m/s inflow, lateral/downstream pressure outlets, GMG, and 100 hub-height
u animation frames in the 512 x 512 x 256 domain.

The requested **1D-upstream RPM/power lookup mode** is configured in
[`fv_v80_two_turbine_lookup.toml`](fv_v80_two_turbine_lookup.toml), with a
small [`lookup smoke case`](fv_v80_two_turbine_lookup_smoke.toml).
See [lookup data and limitations](v80_lookup.md): RPM values are approximate
published-figure readings, and power comes from DTU's V80 reference table.

[`fv_v80_two_turbine_tsr.toml`](fv_v80_two_turbine_tsr.toml) adds a two-V80
layout with independent filtered optimal-TSR speed tracking. A small GPU test
is provided in [`fv_v80_two_turbine_tsr_smoke.toml`](fv_v80_two_turbine_tsr_smoke.toml).
See [controller documentation](../../doc/wind-farm-control.md) for parameters,
commands, per-turbine history, and limitations. This is an idealized speed
servo; the example target TSR and response settings are not calibrated V80 data.

## Base warmup configuration

| | |
|---|---|
| Domain | 10240 x 10240 x 1280 m |
| Mesh | 256 x 256 x 64 (dx = dy = 40 m, dz = 20 m), or 256 x 256 x 128 (dz = 10 m) |
| Roughness | `z0 = 2e-4 m` (offshore) |
| Inflow | `U(70 m) = 8 m/s`, neutral, no Coriolis |
| Friction velocity | `u* = kappa U_hub / ln(z_hub/z0) = 0.250672 m/s` |
| Forcing | `dp/dx = u*^2 / lz = 4.9090957826e-05 m/s^2` |
| Scheme | RK3 with adaptive timestep, `time.cfl = 0.9` |
| Warmup | `6000 x 6.0 s = 36000 s` (10 h, ~7 turnovers at `lz/u* = 5106 s`) |

`dt_seconds` is the step **cap**, not a fixed step: with `time.cfl` set, the
solver picks each step from the CFL ceiling and only clips at the cap. The
configured step counts therefore denote the physical schedule
(`duration = steps * dt_seconds`), not the steps actually taken.

## Running

```bash
# dz = 20 m
jaxwind workflow cases/HornsRev1/fv_workflow.toml \
  --stage warmup

# dz = 10 m (same domain, doubled vertical resolution)
jaxwind workflow \
  cases/HornsRev1/fv_workflow_256x256x128.toml --stage warmup
```

Continue an existing warmup by pointing `warmup_restart_checkpoint` in
`[workflow]` at the checkpoint to resume from and sending the run
to a fresh `output_directory`; the stage adds `warmup_steps * dt_seconds` of
physical time to the checkpoint's clock.

At dz = 20 m the field is developed by t = 10 h: continuing to 20 h moved
integrated resolved TKE by +0.04 % and hub-height wind by -0.35 %, and the
resolved stress already matched the exact `1 - z/lz` equilibrium line.

## Initial profiles

`fv_initial_profile_64.csv` matches the 64-level mesh; `fv_initial_profile_128.csv`
matches the 128-level mesh (dz = 10 m). Both follow
the mean-plus-RMS convention shared by the other finite-volume ABL cases:

```
u(z)   = (u*/kappa) ln(z/z0)                    v = w = scalar = 0
u_rms  = v_rms = 0.25 u* sin(pi z / lz)**0.25   (cell centres)
w_rms  = 0.25 u* sin(pi z_upper / lz)           (upper cell faces)
```

A profile must have exactly one row per vertical cell, with `z_m` matching the
grid cell centres.

## Resolution note

512 x 512 x 128 (33.5 M cells) does **not** fit on an 8 GB GPU: the solver needs
roughly 290 B/cell, i.e. ~9.7 GB, and fails to allocate regardless of time
integration scheme (RK3, fast-RK3 and AB2 all exhaust memory). Measured limits
on an RTX 3070 with ~5.6 GB free were 384 x 384 x 128 (18.9 M cells, 5.6 GB
peak) working and 448 x 448 x 128 failing.

## Comparing against the log law

The solver stores **cell averages**, so compare them against the log law
averaged over each cell, not its value at the cell centre. For a log profile the
first cell differs by

```
(u*/kappa)(ln 2 - 1) = -0.192 m/s   at dz = 20 m
```

which is large enough to make a correct first cell look wrong. The correction
falls to -0.012 m/s by the second cell and is negligible above it. Likewise,
when plotting `Phi_m = (kappa z / u*) d<u>/dz`, push the exact log law through
the same finite-difference stencil - the near-wall dip and the bump one cell up
are artifacts of the one-sided difference, not solver error.
