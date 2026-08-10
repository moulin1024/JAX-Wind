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
- cell-centered horizontal velocity and face-centered vertical velocity;
- compatible gradient, divergence, and pressure-projection operators;
- independent JAX reference, local, and equal-z-slab interpretations;
- JAX-native packed halo exchange for distributed z slabs;
- transpose, exact SPIKE, and adaptive SPIKE pressure solves through
  [`spectral-fd`](external/bw1000_benchmark/README.md);
- fixed-step AB2 integration with explicit startup, accepted-time diagnostics,
  and restart-complete tendency history;
- conservative dry-flow and scalar transport with horizontal three-halves
  padding dealiasing;
- neutral log-law walls, pressure-gradient forcing, Coriolis forcing, static
  Smagorinsky, and Lagrangian scale-dependent dynamic (LASD) closures;
- Boussinesq buoyancy, prescribed scalar fluxes, and optional upper-level
  Rayleigh damping;
- actuator-disk and rigid blade-element actuator-line forcing, including an
  OpenFAST input-deck adapter, plus concurrent-precursor fringe coupling; and
- reference and per-rank checkpoints for dry and velocity-scalar states.

The first production decomposition is an equal z slab. General meshes, uneven
slabs, and a stable high-level simulation API are not currently provided.

## Installation

JAX-Wind requires Python 3.11 or newer. Clone the pressure-solver submodule with
the repository:

```bash
git clone --recurse-submodules https://github.com/moulin1024/JAX-Wind.git
cd JAX-Wind
```

For an existing clone, initialize the submodule with:

```bash
git submodule update --init --recursive
```

Create an isolated environment and install both projects in editable mode:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e external/bw1000_benchmark
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

## Runner workflows

Case-oriented application shells live under [`runners`](runners/README.md).
The first workflow is a configurable [pressure-driven neutral
warmup](runners/pressure_driven_warmup/README.md) with LASD, AB2, checkpointed
restart, and restart-continuous profile statistics.

Validate its supplied `2048 × 1024 × 1024 m`, `128 × 64 × 256`, 10-hour case
without allocating a JAX state:

```bash
jaxwind runners/pressure_driven_warmup --dry-run
```

### OpenFAST rigid actuator line

The concurrent turbine runner can build a fixed-operating-point actuator line
directly from an ordinary OpenFAST primary input deck. The adapter follows its
ElastoDyn, AeroDyn, blade, and airfoil references, so those files remain the
source of truth:

```toml
[turbine]
model = "openfast_rigid_actuator_line"
location_m = [512.0, 512.0] # rotor-apex x and y in the LES domain
openfast_input_file = "OpenFAST/5MW_Land_DLL_WTurb.fst"
smoothing_width_m = 6.0

# Optional fixed operating-point overrides:
# rotor_speed_rpm = 9.0
# pitch_degrees = 0.0
# yaw_degrees = 0.0
# hub_height_m = 90.0
# initial_azimuth_degrees = 0.0
```

The first compatibility surface uses one identical rigid blade definition for
all blades, `AFTabMod = 1`, and linear airfoil interpolation. It imports the
OpenFAST initial rotor speed, pitch, yaw, shaft tilt, precone, azimuth, radii,
chord, twist, airfoil assignment, and first Cl/Cd table. ElastoDyn flexibility,
ServoDyn control, AeroDyn induction/dynamic wake, unsteady aerodynamics, and
curved/swept blade geometry are not advanced. These omissions are written into
the resolved case configuration rather than applied silently.

### JAX-native modal aeroelastic actuator line

The direct actuator-line runner supports JAX-native, two-way,
small-deflection blade coupling using ordinary ElastoDyn `BldFile` inputs as
data. OpenFAST is not linked or executed. Set the aeroelastic runner and enable
the structural block:

```toml
[case]
runner = "direct_aeroelastic_alm"

[aeroelastic]
enabled = true
air_density_kg_m3 = 1.225
gravity_m_s2 = 9.80665
maximum_tip_deflection_m = 20.0
```

`FlapDOF1`, `FlapDOF2`, and `EdgeDOF` select the active per-blade modes. The
adapter reads `NBlInpSt`, distributed `BMassDen`, `FlpStff`, `EdgStff`,
structural twist, modal damping, adjustment factors, and the `BldFl1Sh`,
`BldFl2Sh`, and `BldEdgSh` polynomial coefficients. Modal mass, elastic
stiffness, prescribed-speed centrifugal stiffening, and damping matrices are
assembled from those data.

At every accepted CFD step, deformed positions, slopes, and structural
velocities enter actuator-line sampling and force projection. Equal and
opposite aerodynamic loads plus gravity are projected back to the structural
modes and advanced with average-acceleration Newmark. The current coupling is
explicit and partitioned. It does not yet reproduce OpenFAST tight-coupling
iterations, tower/platform/drivetrain/generator/pitch/yaw motion, ServoDyn, or
BeamDyn.

## Verify the installation

Run the core test suite:

```bash
python -m pytest -q
```

The default suite covers semantic ownership, reference and z-slab operators,
projection, dry and Boussinesq physics, LASD, checkpoints, and integrators.
Tests that open local coordinator sockets are opt-in.

To exercise a true two-process CPU projection:

```bash
export JAXWIND_SPECTRAL_FD_SOURCE="$PWD/external/bw1000_benchmark"
python tools/run_distributed_projection_cpu.py \
  --processes 2 --dtype float32 --methods transpose,spike,spike-adaptive
```

To advance the real dry-flow vector field through AB2 and verify per-rank
checkpoint continuation:

```bash
python tools/run_distributed_ab2_cpu.py \
  --processes 2 --dtype float32 --method spike --steps 4 \
  --vector-field dry
```

Run the complete one-, two-, and four-process CPU gates with:

```bash
JAXWIND_RUN_MULTIPROCESS_CPU_TESTS=1 \
  python -m pytest -q \
  tests/interpreters/test_projection_multiprocess_cpu.py \
  tests/integrators/test_ab2_multiprocess_cpu.py
```

These commands bind loopback coordinator sockets and require the local
environment to permit subprocess networking.

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
| [`src/jaxwind/pressure`](src/jaxwind/pressure) | Semantic adapter around the external pressure solver |
| [`src/jaxwind/effects`](src/jaxwind/effects) | Checkpoint and execution-side adapters |
| [`src/jaxwind/runners`](src/jaxwind/runners) | Configuration, initialization, diagnostics, and application orchestration |
| [`tests`](tests) | Algebraic, physical, interpretation, restart, and distribution tests |
| [`benchmark`](benchmark) | Reproducible physical cases and comparison tooling |

The governing dependency direction is:

```text
benchmarks, runners, and effects
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

- the one-command pressure-driven neutral
  [LASD](benchmark/PressureDrivenLASD/README.md) benchmark, including automatic
  restart and log-law velocity-profile plots;
- the [Andrén et al. (1994) neutral Ekman
  intercomparison](benchmark/Andren1994/README.md);
- the [Nieuwstadt et al. (1993) dry convective boundary-layer
  comparison](benchmark/Nieuwstadt1993/reference/Nieuwstadt1993.md); and
- the [Lin & Porté-Agel (2019) wind-turbine wake
  case](benchmark/LinPorteAgel2019/README.md), including precursor,
  actuator-disk, and fringe-coupling studies.

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
