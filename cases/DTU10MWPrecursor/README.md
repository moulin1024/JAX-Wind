# DTU 10-MW finite-volume AD-BEM and ALM cases

> Run all commands below on a compute node, including configuration checks.
> This case uses schema version 1. Historical outputs cannot be resumed;
> regenerate inputs in the new format. See [verification](../../doc/verification.md).

## ALM runner: prepare and main

[`tools/run_dtu10mw.py`](../../tools/run_dtu10mw.py) uses the requested
4096 x 2048 x 1024 m domain and 512 x 256 x 512 mesh. Its two modes are:

- `prepare`: turbine-free periodic warmup for 10 simulated hours, followed by
  1 hour of periodic inflow recording, using FFT pressure projection.
- `main`: a separate 1-hour ALM run using the completed recording and the
  warmup checkpoint at the start of that recording. The single-turbine adapter
  uses an open streamwise inlet/outlet, periodic lateral boundaries, and GMG.
  The background pressure force is disabled in main.

All three stages use the same **fixed timestep** derived from tip-sweep CFL 0.5:

```text
radius = 89.15 m; rotor speed = 9.6 rpm; minimum cell width = 2 m
blade-tip speed = 89.623355 m/s
dt_limit = 0.5 * 2 / 89.623355 = 0.011157806 s
steps_per_hour = ceil(3600 / dt_limit) = 322645
dt = 3600 / 322645 = 0.011157774024082195 s
```

The slight downward rounding makes each stage end at its requested duration.
Warmup uses 3,226,450 steps; precursor and main each use 322,645 steps, with
one recorded sample per main step. There is no adaptive stepping or inflow
subcycling. Main preflight checks the supplied OpenFAST rotor against the
configured tip-sweep limit. Flow-advection stability still depends on the flow.
Each stage schedules 100 frames and checkpoints every simulated hour.
The precursor contains approximately **630.5 GiB of uncompressed float32
inflow fields**; actual compressed storage depends on the flow. Allow additional
space for full-state checkpoints and outputs.

From the repository root, generate and check declarations without running:

```bash
python tools/run_dtu10mw.py prepare \
  --output outputs/dtu10mw_prepare --configure-only
python tools/run_dtu10mw.py main \
  --prepare-run outputs/dtu10mw_prepare \
  --output outputs/dtu10mw_main --configure-only
```

On a GPU compute node, execute preparation, then main:

```bash
python tools/run_dtu10mw.py prepare \
  --output outputs/dtu10mw_prepare --backend cuda

python tools/run_dtu10mw.py main \
  --prepare-run outputs/dtu10mw_prepare \
  --output outputs/dtu10mw_main --backend cuda \
  --openfast-model /path/to/DTU_10MW_AeroDyn15.fst
```

Use `--backend rocm` on AMD hardware. Preparation needs no turbine deck.
Main accepts `JAXWIND_DTU10MW_FAST` instead of `--openfast-model`; without a
deck, `--configure-only` validates declarations but cannot validate the rotor.
The OpenFAST deck is external and is not supplied by this repository.

For Slurm on the ROCm cluster:

```bash
bash tools/submit_dtu10mw_rocm.sh prepare --output outputs/dtu10mw_prepare

# Submit after preparation completes, or add DEPENDENCY=afterok:PREPARE_JOB_ID.
bash tools/submit_dtu10mw_rocm.sh main \
  --prepare-run outputs/dtu10mw_prepare --output outputs/dtu10mw_main \
  --openfast-model /path/to/DTU_10MW_AeroDyn15.fst
```

The launcher defaults to an **8-hour walltime**, one DCU, partition
`hx1hdnormal01`, and Conda environment `jax060`. Override with `WALLTIME`,
`PARTITION`, or `CONDA_ENV`. Logs are in `logs/dtu10mw/`. Eight hours is an
allocation limit, not a measured completion-time estimate for this resolution.

To continue an interrupted or paused run, repeat its command with `--resume`
and the same output directory. `--max-steps N` pauses each active stage after
at most N additional steps and returns exit code 3, preventing an `afterok`
main from starting before preparation is complete. Main outputs must be
separate from preparation outputs. The runner checks upstream completion and
pins preparation metadata and imported turbine input files for resume.

`--smoke` explicitly selects a small 32 x 24 x 64 mesh and four fixed steps
per stage; both prepare and main must use it. It tests execution and resume,
not production resolution or developed turbulence. The automated end-to-end
test uses the repository's NREL OpenFAST fixture to exercise the generic ALM
adapter; it does not validate DTU10MW aerodynamic data. No full-resolution
DTU10MW run has been launched.

## Historical AD-BEM open-domain workflow

[`fv_workflow.toml`](fv_workflow.toml) runs the same `128 x 64 x 256` domain
and fixed DTU operating point through the FV warmup/precursor/main workflow.
Its warmup and one-hour precursor use the periodic FFT projection. The
precursor records one `yz` layer every `0.1 s`; the main domain enforces those
layers at its inlet, disables the background pressure force, uses the
second-order open outlet, and projects with GMG. AD-BEM, nacelle, and tower
loads are active only in the main stage.

The turbine declaration remains data. Set its configured environment variable
to an AeroDyn15-compatible DTU 10 MW OpenFAST deck, then inspect or run it:

```bash
export JAXWIND_DTU10MW_FAST=/path/to/DTU_10MW_AeroDyn15.fst
jaxwind check \
  cases/DTU10MWPrecursor/fv_workflow.toml
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  jaxwind workflow \
  cases/DTU10MWPrecursor/fv_workflow.toml
```

The complete configuration advances 360,000 warmup steps, records 36,000
precursor layers, and advances the turbine domain for 36,000 steps. The four
recorded inflow fields contain about 9.5 GB of uncompressed float32 data, substantially less
than the former 11-plane HDF5 recording. Use `--max-steps 2` for a
full-resolution paused stage; continue with `--resume`. Artifacts use NPZ chunks.

The workflow preserves the physical domain, pressure driving, roughness, turbine
geometry, fixed rotor speed, and stage durations while using the FV AMD closure
and open-boundary discretization.

## DTU10MW actuator-line smoke case

[`fv_alm_smoke_512x256x512.toml`](fv_alm_smoke_512x256x512.toml) prepares
one rigid DTU10MW ALM turbine in a 4096 x 2048 x 1024 m domain with
512 x 256 x 512 cells (8 x 8 x 2 m; 67,108,864 cells).
The turbine is at (1000, 1024, 117.76719422649163) m, with 9.6 rpm,
zero pitch, and a 16 m Gaussian blade-force smoothing width. Nacelle/tower
settings are retained from the AD-BEM example.

This is a direct, horizontally periodic smoke run: the turbine is active
from startup in the prescribed log-law field, with no precursor required.
It uses FFT pressure projection, MUSCL-MC and fast-RK3, and advances
100 fixed steps of 0.011157774024082195 s (about 1.116 simulated seconds), saving 10 frames and a
final checkpoint. It exercises ALM startup, not a developed wake.
The matching 512-level initial profile is included. The `[workflow]` table
supplies the existing turbine loader's required workflow metadata; use
`run` below to execute the direct turbine-active smoke case.

Supply an AeroDyn15-compatible DTU10MW OpenFAST deck, including its
referenced blade, airfoil, and structural input files. The deck is external
and is not bundled or validated by the configuration-only check:

```bash
export JAXWIND_DTU10MW_FAST=/path/to/DTU_10MW_AeroDyn15.fst
JAX_PLATFORMS=cpu python -m jaxwind check \
  cases/DTU10MWPrecursor/fv_alm_smoke_512x256x512.toml

# Run on a GPU compute allocation with sufficient free memory.
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m jaxwind run cases/DTU10MWPrecursor/fv_alm_smoke_512x256x512.toml
# Use JAX_PLATFORMS=rocm on AMD hardware.
```

Outputs go to `outputs/dtu10mw_alm_smoke_512x256x512`.
The full-resolution case has been prepared only; its GPU memory requirement
and numerical execution have not yet been tested.

## Time- and volume-averaged historical precursor profile

The periodic warmup and recorded precursor use the same checkpointed
profile diagnostics as direct atmospheric runs. Each velocity sample averages
all horizontal cells at each height. On a uniform mesh this is the
volume-weighted mean within each height layer. Regularly spaced profiles
provide the time average; the recorded inlet-plane mean is separate.

The September 2026 fast check starts from the historical 128 x 64 x 256
checkpoint at 21,700 s (6.0278 h). Area-weighted restriction of staggered face
fluxes transfers it to 64 x 32 x 128 cells in the same 4096 x 1024 x 1024 m
domain. The transfer preserves the discrete divergence identity and mean flow;
a target-grid FFT projection removes float32 rounding divergence.
The Porte-Agel correction (`wall_gradient_correction = true`) is enabled
from that checkpoint, with local wall averaging and cell-average wall sampling.

The three comparison runs use the same coarsened initial velocity:

- `outputs/dtu10mw_coarse_porte_agel_20260916`: MUSCL-MC momentum advection,
  AB2, fixed dt = 0.2 s.
- `outputs/dtu10mw_central_cfl09_20260916`: central momentum advection,
  RK3, adaptive CFL ceiling 0.9, and a 5 s maximum timestep. The actual
  timestep and block-maximum CFL are recorded. AB2 is unsuitable for the
  variable-step driver because its history and clock assume a fixed step.
- `outputs/dtu10mw_muscl_rk3_cfl09_20260916`: MUSCL-MC momentum advection,
  RK3, adaptive CFL ceiling 0.9, and the same 5 s timestep cap as the central
  run. This pair changes only momentum advection; actual adaptive timesteps
  can differ as the velocity evolves.

All continue to 10 total simulated hours, then record one precursor hour.
The final warmup-hour mean and 360 precursor profiles at 10 s physical-time
intervals support the comparison. The adaptive inflow recording itself uses
variable time intervals. Each run saves its cases, checkpoint provenance,
initialization checks, and outputs in its own directory.

After both stages complete, generate the comparison on a compute node:

```bash
PYTHONPATH=src python tools/plot_dtu10mw_precursor.py \
  outputs/dtu10mw_central_cfl09_20260916/run
```

The tool writes PNG/PDF profiles, CSV data, quantitative comparison JSON,
and development diagnostics to the sibling `analysis/` directory. For the
central run it also writes an `advection_comparison` plot and CSV against the
MUSCL case. Advection, integration, and timestep control change together, so
this comparison does not isolate the effect of advection alone. The MUSCL/RK3
run instead compares against central/RK3 with the same CFL control. Plot labels
are read from each resolved case, and comparison CSV columns identify the
current and previous runs. Both comparisons retain the same physical averaging
window; stationarity must be assessed separately.

The reference is `U = (u_star / kappa) ln(z / z0)`, with `z0 = 0.001 m`,
`kappa = 0.4`, and pressure-balance `u_star = 0.4 m/s`. Errors use the vertical
cell average of the log law over the declared 20 m to 0.1 H band, consistent
with the finite-volume wall convention. The report also checks hub-height
velocity, wall stress, and remaining domain-mean acceleration. The stress
figure uses resolved covariance only: the existing SGS profile diagnostic
does not apply the enabled wall-gradient correction. Velocity and wall-stress
averages are unaffected by that diagnostic limitation.

Completed coarse comparison results (one-hour averages):

| Momentum / integration | Log-law RMSE, 20–100 m | Wall u* | Bulk drift per hour |
|---|---:|---:|---:|
| MUSCL-MC / AB2 / dt 0.2 s | 1.973 m/s | 0.351 m/s | +0.130 m/s |
| CENTRAL / RK3 / CFL 0.9 | 0.708 m/s | 0.434 m/s | -0.100 m/s |
| MUSCL-MC / RK3 / CFL 0.9 | 2.060 m/s | 0.356 m/s | +0.117 m/s |

All three runs retain bulk-flow drift; these finite-duration results do not establish statistical equilibrium. The matched RK3 comparison and checkpoints are saved in `outputs/dtu10mw_muscl_rk3_cfl09_20260916/`.

For central momentum transport with an explicit backflow-dependent outlet
pressure, use [`fv_central_backflow.toml`](fv_central_backflow.toml).
The [backflow notes](backflow.md) describe the pressure projection, supported
boundaries, recorded diagnostics, verification results, and current limitations.
