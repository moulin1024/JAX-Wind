# Nitrogen-jet models, execution, and HITSZ wake experiment

This guide describes the nitrogen-jet implementations in JAX-Wind and records
how the completed HITSZ turbine-wake injection experiment was configured,
executed, visualized, and compared with its no-jet baseline.

The documentation was prepared on `feat/LN2` on 2026-09-10. The branch tip was
`7589ecb` when the guide was written. The experiments used the shared working
checkout, which also contained uncommitted changes; a branch hash alone is
therefore not a complete source snapshot of those runs. Resolved configurations,
checkpoints, and verification files accompany the local results.

All commands below assume execution from the repository root on a compute node.
Case data live in `cases/`, numerical implementations in `src/jaxwind/`, and
analysis utilities in `tools/`. Historical `applications/fv_ln2_jet/` entry points
are superseded by the installed `jaxwind` CLI.

## 1. Select the intended physical model

These implementations share nitrogen-related parameters but solve different
problems. Their input sections, state fields, and interpretations are not
interchangeable.

| Model | Formulation / configuration | Represented physics | Suitable output interpretation |
|---|---|---|---|
| Cryogenic jet with parcels | `cryogenic-low-mach`, `[physics.jet]`, usually `[physics.source] mode = "volume"` or `"inflow"` | Variable-density carrier, nitrogen transport, liquid parcels, evaporation and carrier exchange; humid-air microphysics where enabled | Carrier temperature, composition, parcel evolution, and continuity residual |
| Fully vaporized embedded jet | `cryogenic-low-mach`, `[physics.source] fully_vaporized_within_source_cell = true` | Direct nitrogen gas mass source and local flash latent-heat sink; parcel exchange is bypassed | Gas-jet mixing and cooling under the prescribed within-cell vaporization assumption |
| Turbine-wake subgrid cooling/jet | `boussinesq`, `[physics.cooling]` in an atmospheric workflow | Prescribed enthalpy-calibrated temperature sink; optional axial momentum; temperature-anomaly buoyancy | Resolved wake velocity and temperature anomaly, not droplet or nitrogen-concentration predictions |

The **180 s turbine experiment in this guide uses the third model**. It does
not simulate a resolved 5 mm nozzle or transport liquid nitrogen parcels.
The detailed standalone cryogenic cases have not been rerun as part of that
experiment, and its successful execution does not validate their predictions.

### Case map

| Purpose | Configuration |
|---|---|
| Standalone humid-air jet, uniform 256³ mesh | [fv_256.toml](fv_256.toml) |
| Resolved inlet geometry on a mapped 256³ mesh | [fv_256_low_mach_rk3_inlet_mapped.toml](fv_256_low_mach_rk3_inlet_mapped.toml) |
| Smaller mapped inlet configuration | [fv_128_low_mach_rk3_inlet_mapped.toml](fv_128_low_mach_rk3_inlet_mapped.toml) |
| Constant-density cryogenic comparison | [fv_256_incompressible_rk3.toml](fv_256_incompressible_rk3.toml) |
| Embedded fully vaporized nitrogen source | [fv_512x128x64_l24_pure_nitrogen_low_mach_1s.toml](fv_512x128x64_l24_pure_nitrogen_low_mach_1s.toml) |
| Jet alone with two streamwise pressure outlets | [512 × 128 × 256 case](../HITSZWindTunnel/fv_512x128x256_l24_jet_only_two_outlets_1s.toml) |
| Adaptive fully vaporized jet | [adaptive CFL 0.6 case](../HITSZWindTunnel/fv_512x128x256_l24_jet_only_two_outlets_adaptive_cfl0p6_1s.toml) |
| Cooling-only turbine-wake example | [fv_far_wake_cooled.toml](../HITSZWindTunnel/fv_far_wake_cooled.toml) |
| Existing non-expanding subgrid jet parameter set | [ALM/round-jet example](../HITSZWindTunnel/fv_uniform_512x128x64_l24_alm_x6_ln2_round_10s.toml) |

The ALM example supplies the jet settings used below; the completed experiment
retains the **AD-BEM** rotor from its HITSZ baseline.

## 2. Standalone cryogenic jet

### 2.1 Geometry and operating point

The canonical [fv_256.toml](fv_256.toml) defines:

| Quantity | Value |
|---|---:|
| Domain | 3 × 3 × 1.8 m |
| Cells | 256 × 256 × 256 |
| Nozzle position | (0.75, 1.5, 0.876) m |
| Injection direction | +x |
| Physical nozzle diameter | 0.010 m |
| Exit speed | 8 m/s |
| Total nitrogen mass flow | 0.0125 kg/s |
| Prescribed exit vapor quality | 0.217596 |
| Injection temperature, loader default | 77.34 K |
| Ambient temperature / pressure | 300 K / 101325 Pa |
| Ambient relative humidity | 0.80 |
| Ambient water-vapor mixing ratio | 0.0179967582 kg/kg dry air |
| Fixed timestep / duration | 0.0005 s / 1 s |
| Pressure backend | GMG, configured tolerance 1e-5 |
| Momentum closure / integration | AMD / RK3 |

The vapor quality partitions the imposed flow into

\[
\dot m_g=\chi\dot m,\qquad \dot m_l=(1-\chi)\dot m.
\]

For this case, the direct vapor component is approximately 2.720 g/s and the
liquid component approximately 9.780 g/s. The prescribed quality is input data;
it should not be silently reused when changing nozzle diameter or exit speed.

The volume-source case starts in still air without a turbine or background
pressure forcing. The x-minus boundary supplies quiescent ambient conditions,
and x-plus is the pressure outlet. Sidewalls and floor use wall-model stress;
the ceiling is impermeable and free-slip. These boundary conditions differ
from the **periodic-y atmospheric turbine workflow** used later in this guide.

### 2.2 Carrier, parcels, and thermodynamics

The low-Mach implementation evaluates carrier density from a constant
thermodynamic-pressure air/nitrogen/water-vapor mixture. Pressure correction
balances density changes and imposed/evaporated gas mass in discrete continuity.
It is not an acoustic or fully compressible nozzle-flow solver.

The implementation couples parcel drag and reaction momentum, sensible/latent
heat exchange, evaporation, nitrogen transport, and thermal/compositional
buoyancy. Humid-air saturation adjustment includes water vapor, liquid water,
and ice. A plotted `liquid_water` field denotes condensed **water**, not liquid
nitrogen; liquid nitrogen belongs to the parcel representation in this mode.

Current parcel defaults include a 150 µm characteristic diameter, a 50–300 µm
range, Rosin–Rammler spread 3, eight parcels injected per step, capacity 16,384,
and four parcel substeps. See [the loader](../../src/jaxwind/config/jet.py)
for the authoritative defaults and constraints. Resolution, parcel capacity,
and these distribution assumptions require assessment for the intended run.

`momentum_closure = "amd"` is selected by the canonical case. The alternate
`"classical-static-smagorinsky"` option uses coefficient 0.16. The explicit
`"rk3"` path projects at each stage; `"fast-rk3"` uses a lagged pressure gradient
and a final-stage projection. These are distinct numerical choices.

### 2.3 Mapped inlet and fully vaporized variants

The mapped inlet case moves the nozzle to x=0 and selects source mode `inflow`.
Tanh clustering follows the inlet and nozzle axis. Its documented local
spacings are approximately 1.28 × 1.29 × 0.76 mm at strength 2.2, so it uses a
50 µs timestep. The local minimum spacing, not the nominal domain-average
spacing, controls explicit-step restrictions.

With `fully_vaporized_within_source_cell = true`, the volume-source path injects
the full nitrogen flow as gas and removes the latent heat associated with the
liquid fraction. It bypasses parcel exchange. This is a prescribed flash
closure, not a spatially resolved evaporation calculation.

`streamwise_boundaries = "outflow-outflow"` supplies pressure outlets at both
ends for an embedded volume source. Ambient backflow is permitted. An imposed
ambient streamwise inflow instead requires `"inflow-outflow"` and a compatible
`physics.ambient.streamwise_velocity_m_s` setting.

Adaptive cryogenic stepping currently requires the fully vaporized volume
source with `time_integration = "rk3"`. In adaptive mode, `dt_seconds` is the
maximum timestep and `dt_seconds * steps` defines the requested physical
horizon. The number of accepted steps can differ. This adaptive jet path checks
projected stage CFL and can reject/retry steps; do not assume the parcel mode
has the same adaptive support.

## 3. Turbine-wake subgrid nitrogen jet

### 3.1 Source selection and units

The atmospheric main-stage assembler reads `[physics.cooling]`. Supplying both
`nozzle_diameter_m` and `injection_speed_m_s` selects the cooling-plus-momentum
`SubgridSpray` path. Omitting both selects the cooling-only `SubgridCooling`
path. Supplying only one is invalid.

The source position is derived from the turbine:

\[
(x_0,y_0,z_0)=(x_T+\Delta x_{\rm source},y_T,z_{\rm hub}).
\]

`streamwise_offset_m` is a physical distance in metres, not a cell count. For
the completed uniform mesh, one cell is 24/256 = **0.09375 m**. Thus a turbine
at (6, 3, 0.876) m has its source at **(6.09375, 3, 0.876) m**. The first receiving
cell center lies downstream of that geometric source location; the physical
nozzle is not snapped to a separately resolved inlet cell.

### 3.2 Cooling power and momentum

The configuration computes positive cooling power as

\[
Q=\eta\dot m\left[(1-\chi)L_v+c_{p,N_2}(T_a-T_j)\right],
\]

where \(\eta\) is the thermal coupling efficiency and \(\chi\) the exit vapor
mass fraction. The injected axial momentum rate is \(F_x=\dot m U_j\).

For the completed turbine experiment:

| Parameter | Value |
|---|---:|
| Total nitrogen flow | 0.0125 kg/s |
| Exit vapor quality | 0.02238146379800278 |
| Pre-nozzle vapor quality | 0 |
| Nozzle diameter / axial speed | 0.005 m / 4 m/s |
| Cone half-angle | 0° |
| Injection / ambient temperature | 77.34 K / 300 K |
| Nitrogen latent heat | 199180 J/kg |
| Nitrogen heat capacity | 1040 J/(kg K) |
| Air density / heat capacity | 1.225 kg/m³ / 1005 J/(kg K) |
| Thermal coupling efficiency | 1 |
| Numerical support widths (x, y, z) | (0.30, 0.09375, 0.1125) m |
| Ramp time | 1 s |
| Resulting cooling power | 5328.60575 W |
| Resulting axial momentum rate | 0.050 N |

These are the existing round-jet example's source parameters, not the 10 mm,
8 m/s standalone parcel-case parameters. The thermodynamic values calibrate
integrated source strength; this Boussinesq model does not compute a local
vapor-quality field or add nitrogen mass to a species equation.

### 3.3 Spatial support and conservation

Let \(s=x-x_0\). For the momentum-jet path, the unnormalized cell kernel is

\[
\widetilde K=\mathbf{1}_{s\ge0}\exp\left[-\frac12\left(
(s/\sigma_x)^2+((y-y_0)/\sigma_y(s))^2+
((z-z_0)/\sigma_z(s))^2\right)\right],
\]

with transverse widths equal to the larger of their configured minimum and
\(d_j/2+\max(s,0)\tan\alpha\). The code normalizes with actual cell volumes:

\[
K_i=\frac{\widetilde K_i}{\sum_j\widetilde K_j V_j},
\qquad \sum_iK_iV_i=1.
\]

The cell sources are

\[
S_T=-\frac{Q}{\rho_a c_{p,a}}r(t)K,
\qquad a_x=\frac{\dot m U_j}{\rho_a}r(t)K.
\]

The ramp is \(r(t)=[1-\cos(\pi\,\mathrm{clip}(t/t_r,0,1))]/2\), or unity
for zero ramp time. Momentum is then interpolated onto x faces with local-width
weighting. The temperature kernel has no cell support upstream of the nozzle;
face interpolation spreads momentum onto the adjacent faces.

Zero cone angle means no downstream expansion of the imposed source support.
The unequal numerical transverse widths remain anisotropic and are not the
physical nozzle diameter. The cooling-only Gaussian does **not** use the same
one-sided mask; selecting the correct source path matters for injection behind
the rotor.

### 3.4 Temperature and buoyancy

The transported scalar is \(T'=T-300\,\mathrm{K}\). The completed run sets
`buoyancy_acceleration_per_unit = 0.0327`, so

\[
b_z=0.0327T'\quad\mathrm{m/s^2}.
\]

A negative anomaly therefore produces downward buoyancy. The actual nozzle
liquid temperature must not be confused with the coarse carrier temperature:
77.34 K is used in source calibration, while the simulated resolved carrier
remains much warmer. Droplet breakup, evaporation history, nitrogen
concentration, fog, and ice are not solved by this wake-source formulation.

## 4. Reproduce the main-only turbine experiment

### 4.1 Required upstream artifacts

The no-jet run completed 1,800 s warmup, 180 s precursor, and 180 s main on a
256 × 64 × 128 mesh in a 24 × 6 × 3.6 m domain. The HITSZ R9 AD-BEM turbine was
at (6, 3, 0.876) m, with D=1.26 m, 480 RPM, and 0° pitch. It uses the fitted
atmospheric inflow profile, not the paper's uniform 4.4 m/s operating point.

The LN₂ run consumes only:

```text
outputs/hitsz_20260910_x6/
  warmup/checkpoint.npz
  precursor/inflow/metadata.json
  precursor/inflow/chunk_*.npz
```

It initializes a new turbine main state from the **warmup** field and replays
the same precursor. It does not restart from the final no-jet main field.
The inlet contains 16,000 samples at 0.01125 s spacing, covering 180 s.

Geometry and fixed inlet cadence must match. Historical HDF5 recordings and
pre-schema-v1 checkpoints are not interchangeable with these artifacts.

### 4.2 Portable configuration example

Save the following as
`cases/HITSZLiquidNitrogenJet/ln2_wake_main.local.toml`. The input directory is
relative to that TOML file; the workflow output below is relative to the
repository root when invoked there. Choose an unused output name for a fresh
run. This example depends on the existing local upstream artifacts.

```toml
schema_version = 1
extends = "../HITSZWindTunnel/fv_workflow.toml"
formulation = "boussinesq"

[case]
name = "hitsz_r9_ln2_one_dx_main"

[numerics]
scalar_advection_scheme = "upwind"

[physics.scalar]
buoyancy_acceleration_per_unit = 0.0327

[physics.turbine]
x_m = 6.0

[physics.cooling]
mass_flow_rate_kg_s = 0.0125
exit_vapor_quality = 0.02238146379800278
pre_nozzle_vapor_quality = 0.0
nozzle_diameter_m = 0.005
injection_speed_m_s = 4.0
cone_half_angle_degrees = 0.0
injection_temperature_k = 77.34
ambient_temperature_k = 300.0
liquid_latent_heat_j_kg = 199180.0
nitrogen_heat_capacity_j_kg_k = 1040.0
air_density_kg_m3 = 1.225
air_heat_capacity_j_kg_k = 1005.0
thermal_coupling_efficiency = 1.0
streamwise_offset_m = 0.09375
standard_deviation_m = [0.30, 0.09375, 0.1125]
ramp_time_s = 1.0

[workflow]
warmup_steps = 90000
precursor_steps = 16000
main_steps = 16000
precursor_dt_seconds = 0.01125
main_dt_seconds = 0.01125
main_substeps_per_inflow = 2
main_frame_count = 100
chunk_steps = 100
input_directory = "../../outputs/hitsz_20260910_x6"
output_directory = "outputs/hitsz_ln2_main_reproduction"
main_pressure_force = false
evolve_scalar = true
```

The nominal `main_steps` count is in inflow intervals. The workflow expands
it to 32,000 solver steps of 0.005625 s, keeping the total duration at 180 s.
Each inlet sample is repeated for two substeps. This smaller step was used for
the evolving upwind temperature calculation. Main remains fixed-step even
though the inherited warmup configuration contains an adaptive CFL setting.

An external `input_directory` removes main's dependencies on newly generated
warmup/precursor stages. **Use `--stage main`**: invoking the whole recipe still
selects its other declared stages.

```bash
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

jaxwind check cases/HITSZLiquidNitrogenJet/ln2_wake_main.local.toml
jaxwind workflow cases/HITSZLiquidNitrogenJet/ln2_wake_main.local.toml \
  --stage main --max-steps 400

# Continue the paused main run; this does not regenerate its inflow.
jaxwind workflow cases/HITSZLiquidNitrogenJet/ln2_wake_main.local.toml \
  --stage main --resume
```

The 400-step startup check advances 2.25 s and crosses the one-second source
ramp. Verify finite velocity/scalar fields and a negative temperature anomaly
before continuing. A paused state is expected when `--max-steps` is used; it
is not a completed 180 s simulation. Configuration changes require a new output
directory rather than reusing a checkpoint with a different fingerprint.

### 4.3 Environment and standalone commands

Use an environment with JAX and the appropriate CUDA plugin, plus the package
installed or `PYTHONPATH=src`. On the Raven allocation used for these runs:

```bash
module load cuda/13.0
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0"
```

The session used `/u/limo/venvs/numba_cuda_waterboa/bin/python`; that is a
machine-local environment, not a portable dependency path. Loading the CUDA
module was necessary for that environment to discover its GPU backend.

For a fresh standalone parcel jet, use a separate output and the ordinary
single-run commands:

```bash
jaxwind check cases/HITSZLiquidNitrogenJet/fv_256.toml
jaxwind run cases/HITSZLiquidNitrogenJet/fv_256.toml \
  --output outputs/ln2_standalone_check --max-steps 2
jaxwind resume outputs/ln2_standalone_check
```

Two steps establish startup and checkpoint compatibility, not plume development
or convergence. The high-resolution two-outlet cases have separate memory
requirements; do not infer their fit on a GPU from a smaller case's success.

## 5. Saved fields and MP4 animation

The completed wake run is in
[outputs/hitsz_20260910_ln2_dx](../../outputs/hitsz_20260910_ln2_dx).
Its main directory contains `checkpoint.npz`, `resolved_case.toml`,
`history.csv`, `summary.json`, `run.json`, and `flow_frames.npz`.

| Frame key | Meaning / array order |
|---|---|
| `u_hub_yx` | Streamwise velocity at hub height, `(frame, y, x)`, m/s |
| `u_center_zx` | Streamwise velocity at turbine y, `(frame, z, x)`, m/s |
| `scalar_hub_yx`, `scalar_center_zx` | Temperature anomaly for this Boussinesq run, K |
| `time_seconds`, `step` | Physical sample time and solver-step index |
| `x_m`, `y_m`, `z_m` | Cell-center coordinates |
| `x_faces_m`, `y_faces_m`, `z_faces_m` | Face coordinates |

Cryogenic jet frames use temperature/composition fields instead of interpreting
the atmospheric `scalar_*` arrays as species. Use the renderer matching the
formulation; a four-panel atmospheric animation is not a droplet visualization.

The wake run saved **100 frames**, one every 320 solver steps, at t=1.8, 3.6,
…, 180 s. The MP4 plays at **10 fps**, giving 10 s playback and 18× physical-time
speed. Horizontal and vertical panels show velocity and temperature anomaly.
The run-specific renderer marks the LN₂ source with a cyan triangle and keeps
color scales fixed; its temperature range includes all sampled anomalies.

The reusable package-side analysis command is:

```bash
python tools/render_fv_wake_gif.py \
  cases/HITSZLiquidNitrogenJet/ln2_wake_main.local.toml \
  --four-panel --fps 10 \
  --output outputs/hitsz_ln2_main_reproduction/ln2_wake.mp4
```

Despite its filename, this tool supports MP4. It requires Matplotlib and either
`imageio-ffmpeg` or system FFmpeg. The original renderer uses percentile-based
color limits; the local experiment's renderer adds the source marker and uses
the full sampled temperature range. No GIF is needed for this workflow.

Completed local artifacts:

- [100-frame MP4](../../outputs/hitsz_20260910_ln2_dx/ln2_wake.mp4)
- [Final-frame preview](../../outputs/hitsz_20260910_ln2_dx/ln2_wake_final.png)
- [Run verification JSON](../../outputs/hitsz_20260910_ln2_dx/verification.json)
- [Resolved experiment configuration](../../outputs/hitsz_20260910_ln2_dx_setup/workflow.toml)
- [Source-marker renderer](../../outputs/hitsz_20260910_ln2_dx_setup/render_animation.py)

These links target **ignored local outputs**, not files distributed by Git.
The numerical settings and results are repeated in this guide so it remains
useful when those artifacts are absent. Archive the output/setup directories
separately when transferring the experiment.

## 6. Centerline deficit against the no-jet case

### 6.1 Definition and sampling

The comparison uses the same reference speed for both cases:

\[
\delta(x)=1-\frac{\overline{u}(x,y_T,z_{\rm hub})}{U_{\rm ref}},
\qquad U_{\rm ref}=3.40825827\ \mathrm{m/s}.
\]

The reference is the no-jet precursor's time-mean hub velocity. The LN₂ run
replays that same precursor. The comparison interpolates each saved center-y
slice to z=0.876 m, matches **50 exact physical sample times** from 91.8–180 s,
and time-averages each case. There is no streamwise smoothing. Distance is
reported as `(x-6)/1.26`, in rotor diameters.

The no-jet run originally saved 800 frames. A dotted curve also uses all 400
late-time no-jet frames to expose sampling sensitivity. Its difference from the
50-frame baseline has normalized RMSE 0.009528 over 4–12D; that is a sensitivity
measure, not a statistical confidence interval.

### 6.2 Observed results

| x/D | No-jet deficit | LN₂ deficit | LN₂ − no jet, percentage points |
|---|---:|---:|---:|
| 2 | 45.69% | 42.82% | −2.87 |
| 4 | 48.38% | 46.13% | −2.25 |
| 6 | 37.04% | 34.10% | −2.94 |
| 8 | 26.52% | 23.30% | −3.21 |
| 10 | 22.05% | 17.34% | −4.70 |
| 12 | 17.36% | 13.07% | −4.29 |

The mean difference over 4–12D is **−3.378 percentage points**: the LN₂ run has
less deficit along the fixed hub centerline in this comparison. This is not a
whole-wake momentum balance or a downstream-turbine power estimate. Buoyant
redistribution can move deficit away from the sampled line.

The existing no-jet run uses dt=0.01125 s; the LN₂ run uses dt=0.005625 s and adds
both momentum and an evolving temperature field. Therefore this experiment
does **not** isolate timestep sensitivity, cooling alone, or injection momentum
alone. A controlled attribution study needs a no-jet run at the same timestep
and, if desired, separate cooling-only and momentum-only cases.

- [Deficit and difference plot](../../outputs/hitsz_20260910_ln2_dx/wake_comparison/centerline_deficit_comparison.png)
- [PDF](../../outputs/hitsz_20260910_ln2_dx/wake_comparison/centerline_deficit_comparison.pdf)
- [CSV profiles](../../outputs/hitsz_20260910_ln2_dx/wake_comparison/centerline_deficit_comparison.csv)
- [Machine-readable comparison](../../outputs/hitsz_20260910_ln2_dx/wake_comparison/summary.json)
- [Matched-time analysis script](../../outputs/hitsz_20260910_ln2_dx_setup/compare_centerline.py)

These results compare the two LES cases directly. They are distinct from the
earlier Gaussian-model fits produced by `tools/postprocess_fv_wake.py`.

## 7. Verification, limitations, and troubleshooting

### Completed checks for the 2026-09-10 wake run

| Check | Result |
|---|---|
| Executed workflow stages | `main` only |
| Final duration / step count | 180 s / 32,000 |
| Source offset | 0.09375 m, exactly one dx |
| Saved frames / decoded MP4 frames | 100 / 100 |
| Frame cadence / encoded playback | 1.8 s / 10 fps |
| Final fields and saved velocity/temperature slices | Finite |
| Maximum recorded main CFL | 0.637230 |
| Final full-field temperature range | 281.399–300 K |
| Lowest sampled hub-plane temperature anomaly | −20.5645 K |

The sampled minimum spans multiple times and is not the same statistic as the
final full-field minimum. Finite values and successful encoding establish
execution and artifact integrity, not grid convergence or experimental
agreement. The subgrid source's widths, coupling efficiency, and Boussinesq
approximation are material modeling choices.

### Common issues

| Symptom | Check or remedy |
|---|---|
| CUDA backend unavailable | Verify the compute allocation, driver/module setup, and CUDA-enabled JAX installation. |
| Warmup starts unexpectedly | Use external `workflow.input_directory` **and** `--stage main`. |
| Input file or metadata missing | Resolve paths relative to the TOML; preserve the inflow manifest and every referenced chunk. |
| Inflow cadence or mesh mismatch | Reuse the matching precursor; do not substitute a differently sized recording. |
| Output directory already exists | Resume the unchanged configuration, or choose a fresh output for a changed case. |
| Source appears upstream | Check whether the cooling-only Gaussian was selected instead of the one-sided momentum-jet path. |
| Missing temperature animation | Enable `evolve_scalar`, use the Boussinesq scalar-anomaly renderer, and confirm nonzero cooling. |
| Unphysical scalar or velocity values | Inspect dt, local grid spacing, source widths, pressure residuals, and scalar advection before interpreting results. |
| A short paused run has few frames | Frames follow the full-run schedule; `--max-steps` does not redefine the requested frame interval. |

For the standalone cryogenic path, also inspect continuity residuals, parcel
inventory/capacity, nitrogen transport, and temperature/species bounds. A
constant-density cryogenic comparison does not demonstrate low-Mach mass
balance. The atmospheric adaptive warmup's logged CFL is evaluated at its
maximum timestep, whereas the separate adaptive cryogenic jet checks projected
stage CFL; do not compare these diagnostics without their definitions.

## 8. Implementation and test references

| Responsibility | Source |
|---|---|
| Cryogenic configuration and validation | [config/jet.py](../../src/jaxwind/config/jet.py) |
| Cryogenic state | [formulations/jet.py](../../src/jaxwind/formulations/jet.py) |
| Carrier/parcel/source assembly | [simulation/jet.py](../../src/jaxwind/simulation/jet.py) |
| Adaptive cryogenic advancement | [simulation/jet_adaptive.py](../../src/jaxwind/simulation/jet_adaptive.py) |
| Parcel exchange | [cryogenic.py](../../src/jaxwind/cryogenic.py) |
| Water/ice thermodynamics | [physics/cryogenic.py](../../src/jaxwind/physics/cryogenic.py) |
| Cooling and jet kernel definitions | [cooling.py](../../src/jaxwind/cooling.py) |
| Cooling parameter lowering | [config/stages.py](../../src/jaxwind/config/stages.py) |
| Turbine/jet/scalar assembly | [simulation/open_atmospheric.py](../../src/jaxwind/simulation/open_atmospheric.py) |
| Stage dependencies and substeps | [workflows/engine.py](../../src/jaxwind/workflows/engine.py) |
| Frame extraction | [runtime/frames.py](../../src/jaxwind/runtime/frames.py) |
| Cryogenic checks | [test_cryogenic.py](../../tests/fv/test_cryogenic.py) |
| Adaptive jet checks | [test_adaptive_jet.py](../../tests/fv/test_adaptive_jet.py) |
| Two-outlet checks | [test_two_outlet_jet.py](../../tests/fv/test_two_outlet_jet.py) |
| Ambient-inflow checks | [test_jet_ambient_inflow.py](../../tests/fv/test_jet_ambient_inflow.py) |

The listed test modules are available verification entry points; this
documentation change does not claim a new full physics-test campaign. See the
repository [verification guide](../../doc/verification.md) for compute-node
execution and schema requirements.

## Three-script ROCm workflow (512 × 128 × 256)

From the repository root, use an environment with JAX-Wind's Python dependencies
(Python 3.11+, or `tomli` for older Python):

```bash
# 1. Generate and validate configuration; no simulation or submission.
bash tools/configure_hitsz_ln2.sh

# 2. Submit simulations only, using the existing single-DCU ROCm launcher.
bash tools/submit_hitsz_ln2_rocm.sh

# 3. After all four stages complete, run CPU analysis separately.
bash tools/postprocess_hitsz_ln2.sh
```

All three scripts accept the same optional setup-directory argument. The default
is `outputs/hitsz_ln2_512_setup`. Configuration refuses to overwrite a nonempty
setup directory. It copies the initial profile and reference data alongside
`case.toml`, `workflow.toml`, and `validation.json`; simulation artifacts go in
`SETUP/run/{warmup,precursor,control,main}`. Generated paths are absolute: configure
on the cluster filesystem where the simulation will run. `PYTHON=/path/to/python`
selects the interpreter for configuration and post-processing.

| Stage | Duration | Timestep | Steps | Saved flow frames |
|---|---:|---:|---:|---:|
| Warmup | 1800 s | Adaptive CFL 0.6, maximum 0.01 s | Adaptive | 0 |
| Precursor | 180 s | 0.005625 s | 32000 | 0 (inlet recorded every step) |
| Control without LN₂ | 180 s | 0.0028125 s | 64000 | 100 |
| Main with LN₂ | 180 s | 0.0028125 s | 64000 | 100 |

The domain is 24 × 6 × 3.6 m, with cells 0.046875 × 0.046875 ×
0.0140625 m. The R9 AD-BEM turbine is at (6, 3, 0.876) m, one quarter
of the streamwise domain. The source is at (6.09375, 3, 0.876) m: its retained
physical offset is now **two x cells**, just downstream of the modeled nacelle's
0.09 m half-length. The subgrid source is spatially smeared; this placement does
not imply that the full source support lies outside the nacelle.

The source retains the earlier 0.0125 kg/s, 4 m/s, 77.34 K settings and physical
Gaussian widths (0.30, 0.09375, 0.1125) m. Physical turbine and body smearing
widths are also retained. This is the Boussinesq cooling/momentum model described
above, not a resolved multiphase nitrogen flow. Both main stages start from the
same warmup checkpoint and consume the same precursor, using two substeps per
inlet sample, the same scalar transport and buoyancy, and identical numerical
settings. Only the LN₂ main stage enables the cooling/momentum source.

Submission reuses `tools/submit_fv_abl_rocm_single.sh`: partition
`hx1hdnormal01`, one `dcu`, eight CPU cores, DTK 26.04, OpenMPI 4.1.5,
conda environment `jax060`, and a ROCm device preflight. It defaults to a
48-hour allocation; runtime and memory at this resolution have not been
benchmarked on ROCm. Override scheduler settings after the setup argument:

```bash
CONDA_ENV=jax060 WALLTIME=72:00:00 bash tools/submit_hitsz_ln2_rocm.sh /path/to/setup
# Continue from checkpoints after a time limit or interruption:
RESUME=1 bash tools/submit_hitsz_ln2_rocm.sh /path/to/setup
# Additional sbatch options:
bash tools/submit_hitsz_ln2_rocm.sh /path/to/setup --account=YOUR_ACCOUNT
```

Do not submit simultaneous jobs writing the same setup. Resume verifies stage
fingerprints and skips completed stages; changed physics requires a new setup.
The submission wrapper clears inherited `MAX_STEPS` so a previous smoke-test
setting cannot silently shorten the full workflow. Slurm logs are saved beside
the configuration. Post-processing is never launched by the submission script.

Post-processing requires NumPy, Matplotlib, and either `imageio-ffmpeg` or a
system FFmpeg with `libx264`. It checks that all stages are complete and both
main cases contain 100 finite frames at matching times (1.8–180 s). It averages
the final 50 frames (91.8–180 s) at the fixed hub centerline, interpolating in z,
and normalizes both deficits by the same hub velocity averaged over the full
precursor recording. There is no streamwise smoothing. Outputs in
`SETUP/run/postprocessing` are:

- `centerline_deficit.png` and `.pdf`: both deficits and their difference.
- `centerline_deficit.csv`: velocities, deficits, and difference in percentage points.
- `summary.json`: reference speed, averaging interval, and mean change over 4–12D.
- `control_wake.mp4` and `main_wake.mp4`: 100 frames each, 10 fps, with x-y/x-z
  velocity and temperature-anomaly panels. Color limits are selected separately
  per movie; use the deficit plot for quantitative comparison.

No GIF is generated. The deficit measures a fixed centerline, not total wake
momentum or recovered turbine power. Script validation includes configuration
checks, mocked submission, and synthetic-data plot/video checks; production
ROCm simulation remains to be run by the user.
