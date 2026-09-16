# HITSZ R9 wind-tunnel-scale AD-BEM case

> Run all commands below on a compute node, including configuration checks.
> This case uses schema version 1. Historical outputs cannot be resumed;
> regenerate inputs in the new format. See [verification](../../doc/verification.md).

This active case transfers the strict offline-precursor workflow used by the
DTU 10-MW benchmark to the 1:100 HITSZ wind-tunnel scale. It uses the R9 rotor
speed and reference loads from Yang, Lin & Zhou, *Renewable Energy* 220
(2024), 119625, DOI `10.1016/j.renene.2023.119625`.

The four stages are:

| Stage | Physical duration | Steps | Configuration |
| --- | ---: | ---: | --- |
| Coarse warmup | 900 s | 180,000 | `128 × 32 × 64`, `dt=0.005 s`, FV neutral ABL with AMD |
| Fine extension | 90 s | 36,000 | `256 × 64 × 128`, `dt=0.0025 s`, projected coarse state |
| Precursor | 90 s | 36,000 | FV precursor with one-plane sampling every step |
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
Integration history is reset because those are
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

The FV workflow also accepts `model = "hitsz-r9-alm"`. This uses the same
three-blade geometry, digitized polars, prescribed RPM, and Prandtl losses, but
samples and deposits the 24 elements on each instantaneous rotating blade.
`initial_azimuth_degrees` is optional (default `0`); the ALM does not require
`smearing_azimuthal_elements`. Set `smoothing_width_chord_factor` to use one
Gaussian width per radial element (for example, `0.5` gives `epsilon=0.5c`).
Gaussian weights are normalized with physical
cell and face volumes, so line loads are conserved on analytically mapped
meshes as well as uniform meshes. `openfast-alm` provides the corresponding
path for an OpenFAST rotor.

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

Run the complete FV workflow:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  jaxwind workflow \
  cases/HITSZWindTunnel/fv_workflow.toml
```

Profile the periodic warmup step with the same HITSZ configuration and write a
machine-readable report:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python tools/benchmark_hitsz_step.py \
  --json hitsz-step-benchmark.json
```

The FFT backend defaults to the custom chunked Thomas solve selected by
`numerics.fft_method`. Compare the block-parallel SPIKE path on the same
case without editing the TOML by adding
`--fft-method spike --spike-block-size 32`; `--thomas-chunk` controls the
unrolled local Thomas sweep used by both methods.

The benchmark reports production adaptive and fixed-dt throughput, decomposes
the step into CFL selection, lagged pressure gradient, three explicit
tendency/RK updates, and the FFT projection, then drills into momentum, AMD,
scalar, and projection operators. Components are separately synchronized, so
their sum is diagnostic rather than an additive prediction of fused execution.
The detailed momentum table compares the production RHS with variants that
remove the body force or log-law gradient correction, its advection-plus-AMD
subset, and addition of precomputed fields. Separate AMD and scalar tables
materialize internal phase boundaries to expose backend-specific fusion and
memory-traffic costs. The production momentum RHS applies
`jax.lax.optimization_barrier` immediately after advection to avoid the
backend-sensitive fusion regression observed on ROCm.

The runner independently resumes interrupted coarse and fine warmups, records
the precursor, executes the main turbine run, creates 100 frames, and overlays
the resulting wake with the TI-consistent Gaussian model. The two-section
velocity-plus-scalar recording is expected to use about 10.4 GB (9.7 GiB)
without compression. This case is configured but its wake acceptance envelope
should only be established after the first completed production run.

## Jet alone with two streamwise outlets

`fv_1024x256x512_l24_jet_only_two_outlets_1s.toml` runs the existing embedded
nitrogen source in initially still air, without a turbine, precursor, imposed
inflow, or background pressure forcing. The 24 x 6 x 3.6 m domain has
1024 x 256 x 512 cells (dx=dy=0.0234375 m, dz=0.00703125 m).
Both x faces have zero pressure correction at the exterior face and extrapolated
velocity; pressure-driven backflow uses ambient temperature and composition.
The y sidewalls and floor retain wall-model stress; the ceiling remains
impermeable and free-slip. Open boundaries can admit entrained ambient air;
"no inflow" means no prescribed wind, not a prohibition on pressure-driven flow.

The case uses low-Mach flow, matrix-free GMG, AMD, and RK3. The initial run is
1 s, with dt=0.00025 s and 4000 steps. It inherits the existing 0.0125 kg/s,
2 mm bore, 4.935894 m/s fully vaporized source at (6.3, 3.0, 0.876) m, including
its local latent-heat sink. The proposed shared spray framework is not yet
implemented. This is an initial transient run, not a converged far-field study.

On a compute node, run `jaxwind run` with this case. On Raven, submit from the
repository root with:

```bash
sbatch tools/submit_hitsz_jet_two_outlets.sh
```

The submission uses a distinct output directory per job. Override scheduler
resources and `JAXWIND_PYTHON` for other environments. The 134,217,728-cell case
requires a full-size memory check; small-grid validation does not establish
that it fits a particular GPU.

For interactive execution on a memory-constrained GPU, the dedicated runner
reuses consumed state buffers. Its results are checked against ordinary
advancement over consecutive blocks. Run inside a compute allocation with the
same CUDA environment as the submission script:

```bash
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0" \
  python -u tools/run_jet_interactive.py \
  cases/HITSZWindTunnel/fv_1024x256x512_l24_jet_only_two_outlets_1s.toml \
  --output outputs/hitsz_ln2_jet/interactive_unique_run
```

Autotuning is disabled here because the original full-size launch exhausted
40 GB of device memory during tuning. Buffer reuse also reduces stepping
memory without changing the physical case. This runner consumes its input
state; callers must not reuse old state arrays after advancement.

The smaller interactive variant is
`fv_512x128x256_l24_jet_only_two_outlets_1s.toml`. It inherits the same source,
solver, boundary conditions, 1 s duration, and 0.00025 s timestep, changing only
the mesh to 512 x 128 x 256 (16,777,216 cells) and the output directory. Use
that case path with the interactive runner above.

The adaptive case
`fv_512x128x256_l24_jet_only_two_outlets_adaptive_cfl0p6_1s.toml` uses RK3 with
a convective CFL ceiling of 0.6, including projected stage velocities. Trial
steps above the ceiling are retried from the original state at a smaller dt.
Molecular/SGS diffusion, startup-ramp resolution, a 25% step-growth limit, and
output-time alignment may require a lower CFL. `dt_seconds=0.01` is the maximum
step; `steps=100` defines a 1 s target duration, not 100 accepted steps.
Snapshots remain at 0.05 s intervals. History records the actual dt, peak-stage
CFL, and cumulative rejected trials. The adaptive path currently supports the
fully vaporized volume source; parcel injection remains fixed-step.

Adaptive validation (2026-09-09, Raven `ravg1197`, allocation `30104293`,
JAX 0.10.0): all three `tests/fv/test_adaptive_jet.py` checks passed on both
CPU and A100 CUDA. These cover physical-time source forcing, snapshot timing
and checkpoint state, and rejection of accelerating trials above CFL 0.6.
The fixed-step two-outlet and state-donation checks also passed (five tests).
The broader runtime suite had two failures that reproduced with the original
observer: the atmospheric adaptive diagnostic scheduler stalls at 0.02 s,
and the low-Mach continuation example inherits a record plane outside its
8-cell test mesh. These are separate from the adaptive jet path.

The ambient-flow variant
`fv_512x128x256_l24_jet_inflow5_adaptive_cfl0p6_1s.toml` sets
`physics.ambient.streamwise_velocity_m_s = 5.0` and restores
`physics.source.streamwise_boundaries = "inflow-outflow"`. The initial air
velocity is uniformly 5 m/s in x, the x-minus inlet holds that speed with
ambient thermodynamic conditions, and the x-plus pressure outlet allows
outflow. The internal nitrogen source and side/floor/ceiling boundaries
remain those of the jet-alone case. It retains adaptive RK3, CFL 0.6,
GMG, the 1 s target and 20 physical-time snapshots; full checkpoints are
written every 200 accepted steps to reduce compression overhead.

## Direct Mann inflow with AD-BEM, 512 × 128 × 256

`fv_mann_512x128x256_adbem_90s.toml` starts the turbine main run directly,
without warmup, precursor recording, or a precursor checkpoint. It uses the
24 × 6 × 3.6 m domain (dx=dy=0.046875 m, dz=0.0140625 m), the R9 AD-BEM rotor
at (12, 3, 0.876) m and 480 RPM, AMD, the floor wall model, periodic y,
impermeable free-slip ceiling, and GMG open-x pressure projection. Mean pressure
forcing is zero. The target is 90 s with dt=0.00125 s (72,000 steps), 100 wake
frames, and a checkpoint every 2,000 steps.

Both mean speed and turbulence intensity come from the measured columns of
`reference/inflow_profile.csv`, not its log-fit column. Heights are converted
from millimetres to metres; mean speed and TI are interpolated independently,
and their product sets the streamwise RMS at each receiving cell height.
Outside the measured 0.1–2.0 m range, the nearest measurement is held constant.
At the hub, U=3.3108 m/s and TI=9.3492%. The initial domain contains the same
measured mean profile without random perturbations; synthetic turbulence enters
from t=0. The first flow-through time is a startup transient.

A 4096 × 64 × 128 Mann box spans 384 × 6 × 3.6 m. Its 115.984 s frozen-advection
period exceeds the 90 s run. L=0.5 m and Gamma=3.9 are explicit modelling
assumptions: the supplied mean/TI data cannot identify the spectral shape or
integral scale. Height-dependent rescaling matches the measured streamwise RMS
after transverse interpolation, including the variance reduction from linear
streamwise interpolation over a complete box period. All components receive
the same local scaling (interpolated to w faces), preserving local Mann component
ratios. This makes the inlet inhomogeneous; it is not a homogeneous Mann tensor
at every height and it is not discretely divergence-free. Short-window realized
statistics fluctuate around the full-period calibration.

Run on a compute node:

```bash
jaxwind run cases/HITSZWindTunnel/fv_mann_512x128x256_adbem_90s.toml
# Raven batch execution, with a unique output directory per job:
sbatch tools/submit_hitsz_mann.sh
# Continue the exact state; the seeded box is regenerated deterministically:
jaxwind resume outputs/hitsz_mann/adbem_512x128x256_JOBID
```

The ordinary `jaxwind run` and `resume` commands own checkpoints and wake
frames; no inflow file needs to be recorded or stored.

Validation on 2026-09-09: 17 focused Mann/direct-run/architecture tests passed,
including measured-profile mean/RMS calibration and checkpoint continuation.
The full 512 × 128 × 256 case advanced two steps on an A100 40 GB with the
production box: CFL=0.219193, maximum FV divergence=8.56e-4 s^-1; the final
checkpoint was written successfully. Device memory observed after advancement
was about 5.2 GB (not a measured peak). This short smoke establishes execution,
not a converged wake or long-time stability. The broader schema suite has two
existing failures (cryogenic adaptive-CFL rejection and an out-of-range inherited
record plane); both reproduce with the original configuration module.
The 90 s production run was submitted separately as Raven job 30109427.
