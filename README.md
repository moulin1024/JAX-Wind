# JAX-Wind

JAX-Wind is a functional staggered finite-volume LES solver for atmospheric
boundary layers, wind-energy flows, and cryogenic jets. Numerical kernels,
simulation construction, host execution, and workflow orchestration have
separate owners. See [architecture](doc/architecture.md) and
[ADR-0018](doc/design/decisions/0018-simulation-runtime.md).
The [refactor plan and acceptance checklist](doc/refactor-plan.md) track scope
and the remaining compute-node gate.

## Compute-node setup and verification

On a login node, restrict work to reading/searching files and editing.
Run installation, Python (including configuration checks), tests, builds,
simulations, and benchmarks **only on a compute node**.

Inside your site's compute allocation, use Python 3.10+ and the appropriate
CPU/CUDA/ROCm JAX environment:

```bash
python -m pip install -e '.[dev]'
```

The refactor is not yet numerically verified. Follow the
[guarded baseline/candidate verification procedure](doc/verification.md).
It supplies interactive and Slurm scripts, small cases, exact-resume checks,
and numerical comparisons against the pre-refactor revision. Optional AMG
requires the dependencies supplied with `external/jax-amg`.

## One user interface

The following commands belong on a compute node:

```bash
jaxwind check cases/Andren1994/config.toml
jaxwind run cases/Andren1994/config.toml --output runs/andren-smoke --max-steps 10
jaxwind resume runs/andren-smoke
```

`python -m jaxwind` is equivalent to the installed `jaxwind` command.
A new run requires an empty output directory. `--max-steps` pauses one
invocation without changing its configured target; `resume` continues the
complete saved state and accumulated diagnostics. There is no overwrite flag.

## Derive a case

```bash
jaxwind case derive cases/Andren1994/config.toml \
  --output cases/local/andren-fine.toml --cells 80 80 80 --cfl 0.5
jaxwind check cases/local/andren-fine.toml
jaxwind run cases/local/andren-fine.toml
```

This writes a small `extends` document, leaves the base untouched, and gives
the derived case an independent name/output. Tables merge recursively; arrays
replace. Inputs resolve relative to the file that declares them; outputs are
relative to invocation. ABL resolution changes explicitly select linear
resampling of tabulated initial profiles; unchanged cases retain strict matching.
CFL adaptation is supported for Boussinesq flow;
fixed-step formulations reject `--cfl`. Their resolution can still be changed
with `--cells`. Mesh changes require compatible new initialization artifacts,
not exact resume of a different mesh.

## Compose a workflow

```bash
jaxwind workflow cases/HITSZWindTunnel/fv_workflow.toml \
  --output runs/hitsz-workflow --max-steps 10
jaxwind workflow cases/HITSZWindTunnel/fv_workflow.toml \
  --output runs/hitsz-workflow --resume
```

The atmospheric recipe expands into periodic warmup, recorded precursor, and
open-inflow stages. Explicit workflows connect named artifacts with
`@stage/checkpoint` or `@stage/inflow`; see
[cases](cases/README.md). Selecting a stage includes its declared dependencies.

## Developer guide

- `config`: versioned schema, inheritance, validation, typed physical inputs.
- `numerics`, physical modules, `formulations`: arrays, operators, coupled state.
- `simulation`: model composition and compiled advancement.
- `runtime`: the shared host loop, scheduling, checkpointable observations.
- `io`: versioned checkpoints, inflow chunks, explicit initialization.
- `workflows` and `cli`: stage dependencies and thin user entry points.

Add physics through numerical functions and builder composition; reuse the
runtime instead of creating another application loop. Package code does not
import experiment tools. Case directories contain data, not execution logic.

This is a breaking configuration/API/artifact migration. Historical output
directories are left untouched but cannot be resumed by the new runtime.
Regenerate continuation inputs with the new format. See the
[migration map](doc/architecture.md#migration-map) and
[verification instructions](doc/verification.md) before production use.

## Synthetic turbine inflow

Generate neutral turbulence with a reproducible Mann box, sample staggered
inlet planes under frozen advection, or export them for an open-inflow workflow.
See the [Mann inflow example and model limits](doc/mann-inflow.md).
