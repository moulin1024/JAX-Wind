# First implementation plan

Status: **approved for implementation**.

This plan turns ADR-0007 into the first non-placeholder vertical slice. It does
not authorize a complete LES solver or a generic distributed framework.

## Implementation status

- Milestone A is implemented: array-independent domain and ownership values,
  construction validation, and exhaustive small-mesh partition laws.
- The initial Milestone B path is implemented: packed transient cell halos with
  explicit physical-boundary flags and JAX `ppermute` communication.
- The vertical-operator part of Milestone C is implemented: the unified z-slab
  `G_z`/`D_z` interpretation commutes with an independent test oracle for one,
  two, and four host CPU devices in float32/float64 tests.
- The compatible three-dimensional projection is implemented as one
  higher-order program over oracle and production algebras. Its face
  operators, physical normal-boundary reconstruction, pressure gauge, and
  correction path are tested together.
- The former Milestone D external spectral/finite-difference adapter has been
  retired. Active pressure projection uses the in-tree matrix-free MAC
  operator with GMG-preconditioned PCG or FGMRES, including the distributed
  y-slab implementation.
- True JAX multi-process CPU execution is implemented and validated for one,
  two, and four processes. Every process constructs only its owned z slab;
  global divergence, idempotence, pressure gauge, dtype preservation, and
  transpose/SPIKE agreement use distributed collectives.
- GPU validation of the retained matrix-free path is performed by the physical
  benchmark runners; the package has no optional external pressure backend.
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

ADR-0015 accepts one JAX z-slab production interpretation, with one shard as
the local case and no generic multi-backend public promise. An independent
tiny-grid array implementation is admitted only as a test oracle.

Decision E does not block this slice because all admitted failures are static
construction errors. H, I, and J do not alter the ownership/operator laws and
remain open. No dynamic status, differentiability promise, physical milestone,
or benchmark migration is introduced implicitly.

Pressure projection is an in-tree implementation. Semantic modules do not
select a Krylov method, preconditioner, device topology, or distributed
runtime; those choices remain in the interpreter or benchmark effect shell.

## 3. Package and dependency skeleton

Create a conventional `src/jaxwind/` package. The first admitted modules have
these responsibilities:

- `domain`: axes, locations, grids, mesh topology, distribution specifications,
  owned regions, quantities, phases, and immutable field wrappers;
- `operators`: semantic definitions and signatures for vertical
  boundary reconstruction, `G_z`, `D_z`, and projection composition;
- `interpreters/jax_zslab`: z ownership, packed halo contexts, compiled
  collectives, single-shard execution, and the pressure adapter;
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

## 7. Milestone D: matrix-free pressure projection

The active pressure path is the in-tree face-staggered MAC projection:

- the application initializes JAX and passes its runtime context;
- the discrete pressure operator is formed matrix-free from compatible
  divergence and gradient operators;
- PCG is used for the symmetric positive-definite case and FGMRES is available
  when the preconditioner requires a flexible Krylov method;
- geometric multigrid is a preconditioner, not a second pressure
  discretization;
- local and distributed y-slab projectors share the same boundary and pressure
  gauge conventions;
- the RHS constant mode is projected out and the returned pressure has zero
  volume-weighted mean.

The implementation MUST NOT gather a distributed production field merely to
solve pressure or initialize a second distributed runtime.

Acceptance tests cover manufactured solutions, pressure-gauge invariance,
projection idempotence and divergence elimination, stretched grids, local and
distributed agreement, and one/two/four-device ownership laws.

Exit criterion: the retained local and distributed matrix-free projectors pass
the declared algebraic and manufactured-solution tolerances.

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
