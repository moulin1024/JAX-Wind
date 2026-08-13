# JAX-Wind

JAX-Wind is a functional large-eddy simulation solver for atmospheric boundary
layers and wind-energy flows. The package owns numerical meaning and state
transitions; directories under `cases/` contain data only, while
`applications/` owns configuration interpretation, diagnostics, and effects.

The active cases are a pressure-driven atmospheric boundary layer and the
Andrén et al. (1994) and Nieuwstadt et al. (1993) intercomparisons. They use
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

Nieuwstadt uses that same ABL command and schema. Its scalar profile, surface
flux, and buoyancy coupling are ordinary physical inputs to the shared solver:

```bash
python -m applications.abl \
  cases/Nieuwstadt1993/config.toml --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.abl cases/Nieuwstadt1993/config.toml
python tools/overlay_nieuwstadt1993.py
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

## Offline precursor sections

A developed checkpoint can be advanced as a standalone precursor while its
inflow and outflow planes are recorded in HDF5. The initial state must retain
its previous AB2 tendency, so recording starts without an Euler cold-start
step:

```python
from jaxwind.effects import PrecursorRecordingConfig, run_precursor

state = run_precursor(
    developed_state,
    steps=10_000,
    advance=solver.advance,
    path="outputs/precursor.h5",
    runtime=solver.runtime,
    recording=PrecursorRecordingConfig(
        sample_every=1,
        buffer_samples=16,
        compression=None,
    ),
)
```

Each sample is the accepted pre-step state. `/velocity` has layout
`(sample, section, component, z, y)`, with sections `inflow` and `outflow` at
x indices `0` and `nx - 1`; `/scalar`, `/step`, `/time`, and staggered
coordinates are stored alongside it.

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
plane through a downstream fringe and writes `main_final.npz`. Playback reads
each rank's local shard in time chunks and broadcasts the two-dimensional
plane across x on device; only the configured fringe region applies it.

## Package structure

| Path | Responsibility |
| --- | --- |
| `src/jaxwind/domain` | Grids, fields, locations, ownership, phases, and scales |
| `src/jaxwind/operators` | Compatible projection program |
| `src/jaxwind/physics` | Momentum, scalar, LASD, and optional wind forcing |
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
