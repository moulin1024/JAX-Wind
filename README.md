# JAX-Wind

JAX-Wind is a research large-eddy simulation (LES) code for atmospheric
boundary layers and wind-energy flows. It combines a pure, compositional
numerical model with JAX implementations for local and distributed execution.

The project is built around an explicit separation between physical meaning and
array representation. Fields carry their quantity, grid location, ownership,
and integration phase; JAX arrays are one interpretation of those semantic
objects rather than the public model itself. This makes conservation laws,
restart behavior, and agreement between reference and distributed
implementations directly testable.

> [!NOTE]
> JAX-Wind is under active development (`0.0.0`) and is intended for solver
> research and reproducible benchmark work. APIs and checkpoint formats may
> change. Validate a configuration before using its results for scientific or
> engineering decisions.

## Capabilities

- incompressible flow on uniform three-dimensional grids;
- standalone analytic rectilinear meshing with independent x/y/z clustering;
- staggered-MAC velocity and compatible gradient, divergence, and projection;
- matrix-free GMG-preconditioned PCG/FGMRES pressure solves;
- single-device and distributed non-spectral y-slab ABL execution;
- fixed-step AB2 integration with explicit startup, accepted-time diagnostics,
  and restart-complete tendency history;
- conservative dry-flow and scalar transport with horizontal two-thirds
  truncation;
- neutral log-law walls, pressure-gradient forcing, Coriolis forcing, static
  Smagorinsky, and Lagrangian scale-dependent dynamic (LASD) closures;
- Boussinesq buoyancy, prescribed scalar fluxes, and optional upper-level
  Rayleigh damping;
- actuator-disk and blade-element actuator-line physics, including an OpenFAST
  input-deck adapter; and
- reference and per-rank checkpoints for dry and velocity-scalar states.

The production ABL path uses the non-spectral staggered-MAC solver. A stable
high-level simulation API is not currently provided.

## Installation

JAX-Wind requires Python 3.11 or newer:

```bash
git clone https://github.com/moulin1024/JAX-Wind.git
cd JAX-Wind
```

Create an isolated environment and install the project in editable mode:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

The default JAX installation is suitable for CPU development. For GPU
execution, install the JAX build that matches the accelerator and CUDA runtime
on the target system.

The Python distribution and import package are named `jaxwind`:

```python
from jaxwind import EqualZSlab, UniformGrid
```

Project-specific environment variables use the `JAXWIND_` prefix.

## Analytic mesh generation

The meshing application is independent of simulation runners and may be
launched from any directory. It generates versioned physical face coordinates;
solvers do not depend on its analytic mapping formulas:

```bash
jaxwind-mesh generate /path/to/JAX-Wind/meshing/example.toml \
  --output "$PWD/mesh.json"
jaxwind-mesh inspect "$PWD/mesh.json"
```

Each axis independently selects uniform, single-boundary, or double-sided
interior-point clustering. See [`meshing/README.md`](meshing/README.md) for the
configuration semantics and portable artifact format.

## Verify the installation

Run the core test suite:

```bash
python -m pytest -q
```

The default suite covers semantic ownership, non-spectral projection, dry and
Boussinesq physics, LASD, checkpoints, and integrators.

## Architecture

JAX-Wind keeps dependencies flowing from effectful applications toward a pure
semantic core:

| Path | Responsibility |
| --- | --- |
| [`src/jaxwind/domain`](src/jaxwind/domain) | Grids, locations, quantities, ownership, phases, and scale systems |
| [`src/jaxwind/operators`](src/jaxwind/operators) | Backend-independent projection program and operator contracts |
| [`src/jaxwind/physics`](src/jaxwind/physics) | Pure dry-flow, Boussinesq, SGS, actuator-disk, and fringe models |
| [`src/jaxwind/integrators`](src/jaxwind/integrators) | AB2 and concurrent-precursor state transitions |
| [`src/jaxwind/interpreters`](src/jaxwind/interpreters) | Unified JAX z-slab interpretation; one shard is the local case |
| [`src/jaxwind/openfast`](src/jaxwind/openfast) | OpenFAST-compatible input parsing and turbine model adapters |
| [`src/jaxwind/meshing`](src/jaxwind/meshing) | Analytic physical mesh generation and versioned artifact I/O |
| [`src/jaxwind/pressure`](src/jaxwind/pressure) | Matrix-free GMG-preconditioned PCG/FGMRES and local/distributed MAC projection |
| [`src/jaxwind/effects`](src/jaxwind/effects) | Checkpoint and execution-side adapters |
| [`tests`](tests) | Algebraic, physical, interpretation, restart, and distribution tests |
| [`benchmark`](benchmark) | Reproducible physical cases and comparison tooling |

The governing dependency direction is:

```text
benchmarks and effects
        ↓
JAX interpreters
        ↓
integrators
        ↓
physics and operators
        ↓
semantic domain types
```

The domain and physics layers do not discover devices, read files, or choose a
process topology. Runtime concerns remain in interpreters and application
shells.

Within implementation-heavy packages, public modules retain the stable API and
semantic validation while private modules isolate numerical kernels, file
parsing, configuration loading, and diagnostics. The detailed boundaries and
the production-module size guard are described in
[`doc/design/04-architecture-and-gates.md`](doc/design/04-architecture-and-gates.md).

## Benchmarks

The repository includes research workflows for:

- the [Andrén et al. (1994) neutral Ekman
  intercomparison](benchmark/Andren1994/README.md);
- the [Nieuwstadt et al. (1993) dry convective boundary-layer
  comparison](benchmark/Nieuwstadt1993/reference/Nieuwstadt1993.md).

Benchmark scripts are case-specific research workflows rather than a stable
command-line interface. Read the corresponding case documentation before
running them; canonical cases can require long integrations, accelerator
hardware, plotting dependencies, and developed restart files.

## Design and development

The active solver is documentation-first. Public concepts are specified under
[`doc/design`](doc/design/README.md), and accepted choices are recorded as
[architecture decision records](doc/design/decisions/README.md). A change to a
public abstraction should identify:

1. its semantic inputs, outputs, and phase changes;
2. the conservation, composition, or restart laws it promises;
3. its reference and production interpretations;
4. its ownership and communication behavior; and
5. a test that fails for a plausible incorrect implementation.

The reference implementation is intentionally independent of optimized
distributed kernels. New production code should agree with the reference on
small problems before performance work begins.

## Legacy implementations

Earlier JAX, C++, and Fortran/CUDA implementations are preserved under
[`legacy`](legacy). They provide historical evidence and regression material,
but they do not define the architecture or semantics of the active
`src/jaxwind` package.

## License

JAX-Wind is released under the [MIT License](LICENSE).
