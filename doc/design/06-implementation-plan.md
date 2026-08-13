# First implementation plan

Status: **approved for implementation**.

This plan turns ADR-0007 into the first non-placeholder vertical slice. It does
not authorize a complete LES solver or a generic distributed framework.

## Implementation status

- Milestone A is implemented: array-independent domain and ownership values,
  construction validation, and exhaustive small-mesh partition laws.
- The initial Milestone B path is implemented: packed transient cell halos with
  explicit physical-boundary flags and JAX `ppermute` communication.
- The vertical-operator part of Milestone C is implemented: the private vertical
  partition
  `G_z`/`D_z` interpretation commutes with an independent test oracle for one,
  two, and four host CPU devices in float32/float64 tests.
- The compatible three-dimensional projection is implemented as one
  higher-order program over oracle and production algebras. Its horizontal
  spectral symbols, vertical face operators, physical normal-boundary
  reconstruction, pressure gauge, and correction path are tested together.
- The Milestone D owned-cell adapter is implemented against `spectral-fd>=0.2.0`.
  Transpose, exact SPIKE, and adaptive SPIKE commute with the independent
  oracle on forced one/two/four-device CPU meshes.
- True JAX multi-process CPU execution is implemented and validated for one,
  two, and four processes. Every process constructs only its owned z slab;
  global divergence, idempotence, pressure gauge, dtype preservation, and
  transpose/SPIKE agreement use distributed collectives.
- GPU validation is deliberately deferred. The public-release license decision
  for `spectral-fd` also remains incomplete.
- Milestone E is implemented for the first deterministic method: fixed-step
  AB2 with explicit Euler startup, one vector-field evaluation at `t_n`, one
  terminal compatible projection, accepted diagnostics/boundaries at
  `t_(n+1)`, explicit previous-tendency history, and a versioned fingerprint.
- Accepted-boundary checkpoint save/load is implemented for the owned z-slab
  representation. The bounded oracle validates transition laws without being a
  restart format. True one/two/four-process CPU tests save one file per rank
  and reproduce uninterrupted continuation exactly in float32 and float64.

## 1. Scope and explicit non-goals

The deliverable is a law-tested ownership and operator path for the accepted
hybrid `Cell + ZFace` complex:

```text
owned Cell / ZFace fields
    -> transient boundary and halo context
    -> compatible D_z and G_z
    -> cell-centred pressure projection
    -> owned divergence-free velocity
```

The first production distribution is an equal z slab. Turbines, spray,
moisture, SGS closures, general pencils, uneven slabs, runtime mesh changes,
and the full ABL integrator are out of scope.

## 2. Backend and remaining decision boundaries

ADR-0016 accepts one public JAX solver for the complete initialized runtime
mesh. The equal vertical partition is private; a one-process job is its
size-one execution, not a separate local solver. An independent tiny-grid
array implementation is admitted only as a test oracle.

Decision E does not block this slice because all admitted failures are static
construction errors. H, I, and J do not alter the ownership/operator laws and
remain open. No dynamic status, differentiability promise, physical milestone,
or benchmark migration is introduced implicitly.

The external pressure extra is constrained to `spectral-fd>=0.2.0`; its license
decision remains required before a public release. Development may use an
editable path only in the effect shell; semantic modules cannot import it.

## 3. Package and dependency skeleton

Create a conventional `src/jaxwind/` package. The first admitted modules have
these responsibilities:

- `domain`: axes, locations, grids, mesh topology, distribution specifications,
  owned regions, quantities, phases, and immutable field wrappers;
- `operators`: semantic definitions and signatures for vertical
  boundary reconstruction, `G_z`, `D_z`, and projection composition;
- `jax_solver`: unified production construction, field lowering, pressure
  projection, and accepted transition;
- `_jax`: private ownership, packed halo contexts, and compiled
  collectives;
- `tests/support/jax_oracle`: independent bounded global validation only;
- `effects`: configuration validation and JAX/distributed runtime ownership.

No registry, base-class hierarchy, universal `Params`, runtime symbolic tensor,
or placeholder abstract backend is admitted.

## 4. Milestone A: semantic ownership values

Implement immutable, array-independent values corresponding to ADR-0007:

```text
DomainAxis
MeshAxis
MeshTopology
Replicated | Partitioned(mesh_axis)
DistributionSpec
OwnedInterval
OwnedRegion[Location]
Field[Quantity, Location, Ownership, Phase, Payload]
```

Construction computes no device state. Runtime discovery is lowered into these
values by the effect shell.

Acceptance tests must falsify:

- a gap or overlap in cell ownership;
- a duplicated inter-slab `ZFace` owner;
- an incorrect lower/upper physical-boundary flag;
- a local extent inconsistent with location and owned interval;
- an unsupported x/y partition or non-divisible equal slab;
- any array-sized storage added by static semantic markers.

Exit criterion: exhaustive small integer meshes satisfy the partition and face
coordinate laws without importing JAX.

## 5. Milestone B: boundary and halo context

Define a transient neighborhood product that distinguishes:

- owned payload;
- lower/upper neighbor values;
- lower/upper physical boundary values;
- the stencil width and field location.

The first packed exchange groups fields only when their location, dtype,
ownership, and phase permit a common payload. Packing is an interpreter
optimization; component extraction recovers the same individual contexts.

Acceptance tests:

- `extract(halo(x)) == x`;
- two neighboring slabs see the same logical interface value;
- a physical boundary cannot be constructed as a rank neighbor;
- repeating exchange does not increase persistent shape;
- measured payload equals the formula derived from stencil width and packed
  component shapes.

Exit criterion: local fake-mesh and JAX z-slab contexts commute on tiny fields.

## 6. Milestone C: compatible vertical complex

Implement the semantic operators already accepted by ADR-0004:

```text
G_z : Cell -> ZFace
D_z : ZFace -> Cell
L_z = D_z . G_z
```

The independent JAX test oracle constructs boundary faces explicitly. The
z-slab interpretation reconstructs only the transient values required by the
owned stencil. Neither stores pressure ghosts in persistent state.

Acceptance tests:

- `G_z(constant) == 0`, including physical boundaries;
- discrete integration by parts holds to oracle precision;
- the domain integral of `D_z(w)` equals prescribed net boundary flux;
- one- through four-shard z-slab outputs agree with the oracle shard by shard;
- an intentionally reversed face-owner rule fails the commuting test.

Exit criterion: all laws hold for one through four slabs and both float32 and
float64 tolerances.

## 7. Milestone D: pressure facade adapter

Add an interpreter-only adapter around `spectral-fd`:

- the application initializes JAX and passes its runtime context;
- input and output are owned z-slab `Cell` fields;
- configuration is `cell-centered-compatible`;
- exact transpose is the correctness oracle;
- exact SPIKE with selected-row interface is the first communication-reduced
  production candidate;
- adaptive SPIKE is enabled only after exact SPIKE passes the same laws;
- all `nz` cells participate, the RHS constant mode is projected out, and the
  returned pressure has zero volume-weighted mean;
- internal y pencils, FFT arrays, factors, and endpoint systems remain
  transient backend workspaces.

The adapter MUST NOT gather, create a host global field, initialize a second
distributed runtime, or expose pressure-solver layout as semantic ownership.

Acceptance tests:

- manufactured horizontal and vertical modes;
- first-cell-only and last-cell-only compatible RHS;
- pressure-gauge invariance;
- projection idempotence and divergence elimination;
- transpose versus exact SPIKE pointwise comparison;
- exact versus adaptive SPIKE comparison;
- one/two/four-device single- and multi-shard commuting tests;
- a process-local allocation audit showing no global physical payload.

Exit criterion: the same projection program passes the independent oracle and
the unified z-slab transpose, exact SPIKE, and adaptive SPIKE paths within
declared tolerances.

## 8. Milestone E: complete deterministic step gate

Status: **implemented** by ADR-0009 and the fixed-step AB2 interpreter.

The first concrete integrator was chosen only after A--D passed. ADR-0009 states
its AB coefficients, Euler startup, single terminal projection, forcing
evaluation time, accepted diagnostic point, fixed-step limitation, and
checkpoint state.

The resulting transition consumes typed owned fields and a pure vector field,
constructs an unprojected candidate, and accepts only the projected state. It
does not yet select the first physical dry-flow vector field.

## 9. Verification and performance order

Correctness work proceeds in this order:

1. construction and law tests without JIT;
2. independent tiny-grid JAX oracle results;
3. single-shard z-slab agreement;
4. multi-shard z-slab agreement;
5. restart-independent ownership serialization checks;
6. synchronized communication and execution measurements.

Performance optimization starts only after the corresponding commuting test
exists. The first measurements report separately:

- compilation;
- local operator execution;
- halo payload and time;
- pressure FFT/local solve/interface communication;
- total synchronized projection time;
- peak addressable bytes per process.

No speedup is accepted if it weakens a partition, projection, or
single-shard/multi-shard law.
