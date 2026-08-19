# HITSZ R9 wind-tunnel-scale AD-BEM case

This active case transfers the strict offline-precursor workflow used by the
DTU 10-MW benchmark to the 1:100 HITSZ wind-tunnel scale. It uses the R9 rotor
speed and reference loads from Yang, Lin & Zhou, *Renewable Energy* 220
(2024), 119625, DOI `10.1016/j.renene.2023.119625`.

The four stages are:

| Stage | Physical duration | Steps | Configuration |
| --- | ---: | ---: | --- |
| Coarse warmup | 900 s | 180,000 | `128 × 32 × 64`, `dt=0.005 s`, pressure-driven neutral LASD |
| Fine extension | 90 s | 36,000 | `256 × 64 × 128`, `dt=0.0025 s`, projected coarse state |
| Precursor | 90 s | 36,000 | pressure-driven LASD with 11-plane HDF5 sampling every 10 steps |
| Main | 90 s | 36,000 | strict inlet overwrite, no main pressure gradient or fringe |

The 900 s and 90 s durations are the DTU benchmark's 10 h and 1 h durations
scaled by the experiment's reported `1:40` time ratio.

The coarse timestep is `0.005 s`; halving all three cell counts permits this
factor-two increase without changing the startup CFL. The production timestep
is `0.0025 s`, the DTU benchmark's `0.1 s` timestep divided by the same 1:40
time scale. Output and checkpoint intervals retain their physical cadence.

The coarse accepted velocity and passive scalar are trilinearly prolonged.
The two horizontal directions are periodic, cell-centred vertical values are
clamped at the walls, and vertical velocity is interpolated on its native
faces. A fine-grid pressure projection restores discrete incompressibility.
AB2 history and LASD trajectory/averaging memory are reset because those are
grid-dependent numerical state; the 90 s fine extension (about three outer
turnover times) rebuilds them before precursor recording begins.

The `256 × 64 × 128` mesh preserves the `24 × 6 × 3.6 m` tunnel and gives
`dx = dy = 0.09375 m` and `dz = 0.028125 m`. Thus `dz/dx = 0.3`; an exact
quarter ratio is incompatible with this explicit mesh and fixed tunnel size.

The main turbine is an azimuthally averaged blade-element actuator disk with
the experimental geometry and rated data:

- rotor diameter: `1.26 m`;
- hub height: `0.876 m`;
- location: `(12, 3) m` in the `24 × 6 × 3.6 m` tunnel;
- measured `C_T = 0.810` and `C_P = 0.459`;
- measured operating speed: `480 RPM`;
- measured thrust and torque: `12.21 N` and `0.61 N m`.

The disk uses 24 radial annuli, three blades, prescribed 480 RPM rotation,
Prandtl root/tip loss, radial thrust and tangential loading, and the legacy
ADMR element-size Gaussian width with 64 virtual azimuthal elements. The
HITSZ001 `E1` lift and `E4` drag curves at `Re = 4.6e4` were digitized from the
supplied raster of paper Fig. 9. Chord and twist markers were digitized from
Fig. 10 and the chord was reduced by the reported 1:100 length scale. These
tables are reproducible raster readings, not the authors' original XFOIL
output; see `reference/hitsz001_polar_digitized.csv` and
`reference/blade_geometry_digitized.csv`.

The paper reports the 40 mm tower diameter. Its nacelle dimensions are not
tabulated; the `0.18 × 0.05 m` nacelle is a documented 1:100 geometric model
assumption and is kept separate from the measured rotor geometry.

This configuration deliberately uses the separately supplied measured inflow
profile, because it follows the same pressure-driven precursor design as the
DTU case. Ordinary least squares of the 20 mean-speed samples over 0.1--2.0 m
against `ln(z)` gives

```text
u* = 0.1229413268 m/s
z0 = 1.6100320416e-5 m
R2 = 0.905575
RMSE = 0.0786173 m/s
Ufit(0.876 m) = 3.3514673 m/s
TSR at 480 RPM = 9.4487731
```

The previous provisional value `u*=0.16140448 m/s` rescaled this curve to
force 4.4 m/s at hub height; it is intentionally not used here. The CSV
heading supplied as `Height_m` contains values 100--2000 and is interpreted as
millimetres, consistent with the experiment and the original tabulation. Full
fit diagnostics are in `reference/inflow_log_fit.json`.

Consequently, this is not the paper's exact R9 uniform-flow condition
(`U=4.4 m/s`, TSR about 7.2). It is the requested fitted-ABL case operated at
the R9 prescribed rotation speed. The R9 `C_T=0.810` Gaussian curve produced
by the runner is therefore a reference overlay, not an acceptance target.

This fitted ABL is not the paper's uniform-flow, below-1%-turbulence baseline.
The versioned profile reports 8.44--10.9% turbulence; the LES develops its own
resolved turbulence during warmup rather than imposing those values directly.

Run the complete workflow:

```bash
tools/run_hitsz_r9_chain.sh
```

The runner independently resumes interrupted coarse and fine warmups, records
the precursor, executes the main turbine run, creates 100 frames, and overlays
the resulting wake with the TI-consistent Gaussian model. The two-section
velocity-plus-scalar recording is expected to use about 10.4 GB (9.7 GiB)
without compression. This case is configured but its wake acceptance envelope
should only be established after the first completed production run.
