# Effects and distribution

## 1. Effect boundary

The following operations are effects and MUST remain outside the pure numerical
core:

- reading configuration and environment variables;
- device/process discovery and distributed-runtime initialization;
- filesystem, checkpoint, image, and table I/O;
- wall-clock timing, progress reporting, and logging;
- exceptions caused by external resources.

The effect shell validates and lowers external data into semantic values, calls
a pure program, then interprets outputs. Core functions MUST NOT inspect a
pathname, process rank, global logger, or current time.

## 2. Explicit computational effects

Some effects must participate in compiled computation and are therefore data:

- RNG state for stochastic models;
- distributed ownership and collective operations;
- optional diagnostic accumulation;
- failure/status values that must survive JIT compilation.

These are represented by explicit products or transition effects. Host
callbacks are prohibited in the production step unless a design amendment
defines their ordering and portability semantics.

## 3. Sharding functor

For mesh \(M\), sharding maps a global semantic shape to ownership metadata and
local storage. Global shape is metadata; it does not authorize global
materialization. The distributed interpretation MUST satisfy the commuting
property

\[
\mathrm{gather}_{test}(D_M(f)(D_M(x))) \approx f(x),
\]

where `gather_test` exists only in bounded tests and postprocessing. Production
code MUST NOT use it.

The stronger local law is preferred: each owned output shard equals the
corresponding slice of the reference result without first gathering the
distributed input.

## 4. Ownership and halos

Persistent prognostic state contains owned cells/faces only. A halo operation
constructs a transient neighborhood context. Conceptually this context is
comonadic: `extract` returns the owned value, and extending context must not
alter ownership.

At minimum, halo implementations MUST satisfy:

- extracting owned data after exchange returns the original owned data;
- neighboring interface values agree with the declared topology;
- physical boundaries and inter-rank boundaries are different constructors;
- repeated context construction does not accumulate ghost planes;
- communication volume is determined by the stencil and ownership type.

Fields used at the same phase SHOULD share a packed halo exchange when doing so
preserves their individual semantics.

## 5. Layout transformations

Slab-to-pencil transpose, spectral layout conversion, batching, and precision
conversion are natural transformations between interpretations of the same
semantic field. They MUST preserve component identity, coordinates, ownership,
and inverse/round-trip laws. A transform MUST NOT silently apply filtering or
boundary conditions.

Filtering is a named physical/numerical morphism, not a side effect of layout
conversion.

## 6. Collectives

Collectives MUST be expressed inside the distributed program through the
backend's SPMD primitives. MPI MAY launch processes but MUST NOT become a
second, independently ordered communication layer around compiled steps.

Each collective contract names:

- participating mesh axes;
- source and destination ownership;
- tensor payload and physical meaning;
- ordering relative to numerical phases;
- whether the operation is a permutation, reduction, or replication.

Reductions MUST use an explicit monoid. Floating-point non-associativity and
the expected tolerance across mesh sizes must be documented.

## 7. Checkpoints

A checkpoint is an effectful isomorphism between a complete persistent state
and a versioned serialized representation, subject to supported dtype and
backend constraints. Restart law:

\[
\mathrm{run}_n(\mathrm{load}(\mathrm{save}(s)))
=\mathrm{run}_n(s)
\]

bitwise where the backend promises deterministic execution, otherwise within a
declared tolerance. Integrator history, closure memory, RNG state, step/time,
and decomposition-independent metadata are part of persistent state. Halos,
FFT workspaces, compiled executables, and loggers are not.

Each rank writes only owned data. A restart on a different valid mesh MAY be
supported through an explicit resharding interpretation; it must never rely on
one process loading the global array.

## 8. Compilation and donation

Static structure and dynamic arrays MUST be separated deliberately. Large
operators or states MUST NOT be captured as hidden JIT constants. Buffer
donation is permitted in the array interpreter because semantic values remain
immutable; callers must treat donated representations as consumed.

Compilation cache keys, batch sizes, and mesh topology are interpreter
concerns. Changing them MUST preserve the semantic trajectory.
