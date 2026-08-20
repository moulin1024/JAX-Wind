# JAX-Wind

JAX-Wind is a functional large-eddy simulation solver for atmospheric boundary
layers and wind-energy flows. The package owns numerical meaning and state
transitions; directories under `cases/` contain data only, while
`applications/` owns configuration interpretation, diagnostics, and effects.

The active cases are a pressure-driven atmospheric boundary layer and the
Andrén et al. (1994), Nieuwstadt et al. (1993), and GABLS1
intercomparisons. They use
conservative momentum transport, Lagrangian scale-dependent dynamic (LASD)
closure, AB2 integration, and compatible spectral/finite-difference pressure
projection.

## Install

JAX-Wind requires Python 3.11 or newer. Initialize and install the pressure
solver together with the package:

```bash
git submodule update --init --recursive
python -m pip install -e external/bw1000_benchmark
python -m pip install -e .
```

Install the JAX build appropriate for the CPU or accelerator on the target
machine.

## Run the pressure-driven LASD case

Validate and display the resolved case without importing JAX:

```bash
python -m applications.pressure_driven_lasd \
  cases/PressureDrivenLASD/config.toml --dry-run
```

Run the configured case:

```bash
python -m applications.pressure_driven_lasd \
  cases/PressureDrivenLASD/config.toml
```

For a short integration, use `--max-steps`. Restart and overwrite behavior are
explicit application arguments:

```bash
python -m applications.pressure_driven_lasd \
  cases/PressureDrivenLASD/config.toml --max-steps 10 --overwrite
python -m applications.pressure_driven_lasd \
  cases/PressureDrivenLASD/config.toml \
  --restart outputs/pressure_driven_lasd_64x64x64_gpu/checkpoint_latest.npz
```

There is no package runner registry or case-name dispatch. An explicitly
selected application reads case data and constructs the JAX-Wind solver.

JAX-Wind uses the WIRE-LES legacy nonlinear discretization exclusively.
Momentum convection is evaluated in rotational form on the base grid. At the
start of every RHS evaluation, the exact `nint(N/3)` box mask is applied to
the complete velocity and to any transported scalar. Rotational products are
not filtered, and terminal projection leaves the accepted/output state
unfiltered until the next RHS evaluation. There is no advection or dealiasing
selector in the case schema or command line.

For neutral runs that deliberately freeze the passive scalar at zero, LASD
offers two independent performance controls in the `[sgs]` table:

```toml
scalar_lasd_enabled = false
reuse_rhs_momentum_context = true
```

The first skips scalar filtering and scalar Lagrangian-memory updates while
retaining the momentum LASD model. It is accepted only with the solver's
frozen-zero-scalar optimization. The second passes the momentum halo and
derivative context built during each accepted-step LASD preparation directly
into the fused RHS, avoiding a duplicate context build. Both options preserve
their legacy behavior when omitted (`true` and `false`, respectively).

CUDA float32 runs can also replace the pure-JAX LASD test filters with a
native backend that reuses persistent three- and six-component cuFFT plans and
work buffers. Build it for the target GPU architecture, then select it in the
same table:

```bash
JAXWIND_CUDA_ARCHITECTURES=86 tools/build_lasd_cufft.sh
```

```toml
[sgs]
lasd_filter_backend = "cufft" # default: "jax"
```

The backend performs one forward transform per component group and reuses its
spectrum for both LASD filter widths. Its cache is keyed by CUDA device,
stream, field shape, and component count, so separate MPI processes and local
devices own independent plans and buffers. Set
`JAXWIND_LASD_CUFFT_LIBRARY` only when the shared library is installed outside
the default `build/native/lasd_cufft` directory. The JAX backend remains the
portable fallback and is required for float64 runs.

The Andrén case uses a strict, fixed-schema TOML configuration. The generic
ABL application translates its SI values into existing physical components
without an Ekman mode, stability selector, or case branch in the solver.
Neutral, stable, and convective are consequences of scalar buoyancy coupling,
initial stratification, and surface heat transfer—not applications:

```bash
python -m applications.abl \
  cases/Andren1994/config.toml --dry-run
python -m applications.abl \
  cases/Andren1994/config.toml --max-steps 10 --overwrite
```

Run the same reference-backed case in a separate output directory:

```bash
python -m applications.abl cases/Andren1994/config.toml \
  --output outputs/andren1994_lasd_40x40x40_trial
```

The same physical case can be run with the staggered finite-volume solver, AB2,
and its direct FFT pressure backend. The FV path currently uses AMD momentum
closure and an eddy-diffusivity passive scalar, so its resolved output records
that closure distinction from the canonical LASD run. Its extended diagnostics
supply every currently registered Andrén overlay (Figures 2--8, 11, 14, and
15):

```bash
python -m applications.fv_abl cases/Andren1994/config.toml --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.fv_abl cases/Andren1994/config.toml \
  --max-steps 10 --overwrite
```

For open-streamwise calculations, the FV application also provides a
configuration-driven warmup/precursor/main workflow. It develops and records
the periodic precursor with FFT, writes only one inflow layer per time step,
then enforces those layers in an open-x main run with GMG and a second-order
outflow condition:

```bash
python -m applications.fv_abl.workflow \
  cases/Andren1994/config.toml --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.fv_abl.workflow \
  cases/Andren1994/config.toml --max-steps 2 --overwrite
```

Nieuwstadt uses that same ABL command and schema. It also has an FV comparison
runner with Boussinesq coupling and FFT pressure projection:

```bash
python -m applications.fv_abl cases/Nieuwstadt1993/config.toml --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.fv_abl cases/Nieuwstadt1993/config.toml --overwrite
```

Its scalar profile, surface flux, and buoyancy coupling are ordinary physical
inputs to the shared solver:

```bash
python -m applications.abl \
  cases/Nieuwstadt1993/config.toml --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.abl cases/Nieuwstadt1993/config.toml
python tools/overlay_nieuwstadt1993.py
```

GABLS1 also has an FV runner with evolving surface temperature and coupled
Monin–Obukhov momentum and heat fluxes. Its complete overlay covers 27 of the
30 official diagnostics:

```bash
python -m applications.fv_abl cases/GABLS1/config.toml --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.fv_abl cases/GABLS1/config.toml \
  --output gabls1_fv_fft_overlays
python tools/overlay_gabls1.py gabls1_fv_fft_overlays \
  --output-dir gabls1_fv_fft_overlays
```

## Solver boundary

Applications use `build_jax_solver` for both one-process and parallel jobs.
Process/device discovery is captured once by the effect shell; the solver owns
lowering and pressure projection without exposing its private partition:

```python
from jaxwind import build_jax_solver
from jaxwind.effects import JaxRuntime

runtime = JaxRuntime.from_initialized_jax(jax)
solver = build_jax_solver(
    execution_grid,
    runtime=runtime,
    model=model,
    integrator=integrator_config,
    normal_boundary=boundary,
    pressure_dtype="float64",
)
result = solver.advance(state)
```

`build_solver` remains the pure transition composer beneath that facade.
Checkpointing, diagnostic gathering, statistics, acceptance, timing, and
reporting remain effects and never enter the physical transition.

## Wind-farm turbine models

`jaxwind.windfarm` owns physical turbine parameterizations and OpenFAST input
adapters. The smallest turbine model is a uniform, non-rotating actuator disk
specified in SI units and lowered explicitly to the solver scales:

```python
from jaxwind.physics import WindTunnelModel
from jaxwind.windfarm import SimpleActuatorDisk

turbine = SimpleActuatorDisk(
    x_m=400.0,
    y_m=250.0,
    hub_height_m=90.0,
    rotor_diameter_m=120.0,
    thrust_coefficient_prime=4.0 / 3.0,
    smoothing_width_m=10.0,
)
wind_tunnel = WindTunnelModel(
    actuator_disk=turbine.to_actuator_disk(scales=scales),
)
```

The disk reuses the force-conserving, filtered pure-thrust kernel. The upstream
OpenFAST source is pinned under `src/jaxwind/windfarm/reference/openfast` only
for implementation and input-format reference; it is excluded from package
discovery and is never imported or linked by JAX-Wind.

## Offline precursor sections

A developed checkpoint can be advanced as a standalone precursor while its
inflow and outflow x-normal sections are recorded in HDF5. The initial state
must retain its previous AB2 tendency, so recording starts without an Euler
cold-start step:

```python
from jaxwind.effects import PrecursorRecordingConfig, run_precursor

state = run_precursor(
    developed_state,
    steps=10_000,
    advance=solver.advance,
    path="outputs/precursor.h5",
    runtime=solver.runtime,
    recording=PrecursorRecordingConfig(
        sample_every=10,
        buffer_samples=8,
        section_width=11,
        inflow_start_index=9,
        compression=None,
    ),
)
```

Each sample is the accepted pre-step state. `/velocity` has layout
`(sample, section, component, z, y, x_section)`. The generic defaults record
one plane; the strict Fortran workflow records the distinct 11-plane slab at
one-based planes 10--20. `/scalar`, `/step`, `/time`, and staggered coordinates
are stored alongside it.

On one process, the requested path is the data file. On multiple processes,
each rank writes only its owned z slab to
`precursor.process-NNNNN.h5`. Rank 0 then publishes `precursor.h5` as a small
HDF5 virtual-dataset catalog over those shards. No global field is gathered,
no ranks contend for one writable HDF5 file, and an MPI-enabled HDF5 build is
not required. Plane slices remain device-side until a full buffer is copied in
one transfer. All ranks must call `run_precursor` with the same path and
recording policy.

The pressure-driven LASD checkpoint can be exercised end to end with the
wind-farm precursor application:

```bash
python -m applications.windfarm_precursor --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.windfarm_precursor \
  --restart outputs/pressure_driven_lasd_64x64x64_gpu/checkpoint_final.npz \
  --output outputs/windfarm_precursor \
  --precursor-steps 10000 --main-steps 10000
```

The workflow reloads the developed checkpoint for both domains, writes
`precursor.h5` and `precursor_final.npz`, then replays the clock-matched inflow
slab and writes `main_final.npz`. Playback reads each rank's local shard in
bounded time chunks. The main run reproduces the CUDA-Fortran
`force_inflow` operation: a cosine blend over planes 1--9, direct overwrite of
planes 10--20 every 10 steps, and one-cell progressive spanwise cycling every
four inlet updates. The main pressure gradient is disabled, and no downstream
fringe or alternative inlet mode is available in this application.

Here, strict compatibility means the same staggered-grid equations, operation
ordering, spectral cutoffs, AB2/projection sequence, inlet indices, and update
cadences as CUDA-Fortran. HDF5 replaces the legacy per-component binary files
as the intentional I/O difference. Floating-point fields are not expected to
be bitwise identical because JAX/XLA and the Fortran executable may choose
different FFT, fusion, and parallel-reduction orders.

## Package structure

| Path | Responsibility |
| --- | --- |
| `src/jaxwind/domain` | Grids, fields, locations, ownership, phases, and scales |
| `src/jaxwind/operators` | Compatible projection program |
| `src/jaxwind/physics` | Momentum, scalar, LASD, and optional wind forcing |
| `src/jaxwind/windfarm` | Turbine parameterizations and OpenFAST input adapters |
| `src/jaxwind/integrators` | Accepted AB2 transitions |
| `src/jaxwind/jax_solver.py` | Unified one/many-process JAX solver facade |
| `src/jaxwind/_jax` | Private JAX distribution lowering and kernels |
| `src/jaxwind/pressure` | Explicit adapter to `spectral-fd` |
| `src/jaxwind/solver.py` | Solver composition and repeated pure transitions |
| `src/jaxwind/effects` | Runtime discovery, diagnostics gathering, and checkpoint persistence |
| `applications` | Case schemas, initialization, diagnostics, and effects |
| `cases/PressureDrivenLASD` | Pressure-driven configuration data |
| `cases/Andren1994` | Andrén configuration and reference data |
| `cases/Nieuwstadt1993` | Nieuwstadt configuration and reference data |
| `cases/DTU10MWPrecursor` | 4096 × 1024 × 1024 m DTU 10-MW precursor domain |

The current equal vertical partition remains a private implementation detail.
One-device and distributed results share one solver path; applications neither
name the partition nor calculate shard ownership.

## Verify

```bash
python -m pytest -q
```

The default suite covers semantic fields, scaling, conservative physics, LASD,
projection, pressure adaptation, integration, checkpoint restart, OpenFAST
parsing, and the direct pressure-driven case. Multi-process CPU tests remain
opt-in because they bind coordinator sockets.

## Historical material

`legacy/` contains prior JAX, C++, and Fortran/CUDA implementations.
`legacy/cases/` preserves literature data and offline analysis from earlier
case implementations; none of it is a package execution entry point.

JAX-Wind is released under the [MIT License](LICENSE).
