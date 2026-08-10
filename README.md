# JAX-Wind

JAX-Wind is a functional large-eddy simulation solver for atmospheric boundary
layers and wind-energy flows. The package owns numerical meaning and state
transitions; directories under `cases/` contain data only, while
`applications/` owns configuration interpretation, diagnostics, and effects.

The active cases are a pressure-driven neutral atmospheric boundary layer and
the Andrén et al. (1994) neutral ABL intercomparison. Both use conservative
momentum transport, Lagrangian scale-dependent dynamic (LASD) closure, AB2
integration, and compatible spectral/finite-difference pressure projection.

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

## Solver boundary

`build_solver` closes one accepted transition over a model, numerical algebra,
pressure solver, boundary law, and closure event. `solve` repeats that pure
transition without reading files or performing output effects:

```python
from jaxwind import build_solver, solve

advance = build_solver(
    config=integrator_config,
    vector_field=vector_field,
    normal_boundary=boundary,
    algebra=algebra,
    pressure_solver=pressure_solver,
    closure_event=closure_event,
)
final_state = solve(initial_state, steps=20, advance=advance)
```

The pressure-driven application uses the same `advance` function and owns
checkpointing, statistics, acceptance, and reporting. The case supplies only
their configuration values.

## Package structure

| Path | Responsibility |
| --- | --- |
| `src/jaxwind/domain` | Grids, fields, locations, ownership, phases, and scales |
| `src/jaxwind/operators` | Compatible projection program |
| `src/jaxwind/physics` | Momentum, scalar, LASD, and optional wind forcing |
| `src/jaxwind/integrators` | Accepted AB2 transitions |
| `src/jaxwind/interpreters` | JAX numerical kernels grouped by responsibility |
| `src/jaxwind/pressure` | Explicit adapter to `spectral-fd` |
| `src/jaxwind/solver.py` | Solver composition and repeated pure transitions |
| `src/jaxwind/effects` | Versioned checkpoint persistence |
| `applications` | Case schemas, initialization, diagnostics, and effects |
| `cases/PressureDrivenLASD` | Pressure-driven configuration data |
| `cases/Andren1994` | Andrén configuration and reference data |

The interpreter retains the established equal-z-slab implementation so its
one-device and distributed results continue to share the same numerical path.
Kernel construction is grouped into projection, flow, LASD, scalar, and wind
products rather than one positional callable matrix.

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
