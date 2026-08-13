# ADR-0016: Unified JAX solver with private distribution lowering

Status: **Accepted**

Supersedes [ADR-0015](0015-unified-zslab-interpreter.md).

## Context

ADR-0015 removed separate local and distributed numerical implementations, but
its public API still named the current storage algorithm. Applications had to
construct `EqualZSlab`, calculate addressable shard ids, assemble pressure and
checkpoint layouts, and in some cases reject multi-process jobs. The numerical
kernels were unified, while solver construction was not.

That leaked an implementation detail into case code and made a one-process job
look like a distinct execution mode. It also left host effects such as process
discovery, diagnostic gathering, rank-aware checkpoint names, timing, and
logging too close to numerical assembly.

## Decision

`jaxwind.build_jax_solver` is the public production construction API. It builds
one `JaxSolver` from a semantic grid/model/integrator and an already initialized
`JaxRuntime`. The runtime's global process/device mesh is always used. A single
process and device is the size-one instance of that mesh; it does not select a
different solver, representation, kernel set, or pressure path.

The current equal vertical partition and `pmap` lowering remain private
implementation mechanisms under `jaxwind._jax`. They are not named by
applications or exported as a production API. A future partitioning strategy
may replace them without changing case composition or physical models.

`JaxRuntime` belongs to the effect shell and owns:

- process and device discovery;
- addressable partition placement;
- cross-process gathering for host diagnostics only;
- process-specific checkpoint paths and barriers; and
- primary-process identity for output, timing, progress, and logging.

`JaxSolver` owns numerical lowering, compatible pressure projection, initial
field distribution, closure initialization, and the accepted transition. The
physics package continues to depend only on algebra protocols and semantic
configuration. It may not inspect process count, process index, device count,
partition ids, paths, clocks, or loggers.

Distributed collectives required by the numerical transition remain compiled
backend operations. Host diagnostic gathering must never feed a solver step.

## Required laws

- one and many processes use `build_jax_solver` and `JaxSolver.advance`;
- applications do not import private JAX modules or construct
  partitions;
- importing `jaxwind.domain` does not import JAX;
- process count does not alter model composition or stability selection;
- the coupled lower-surface law reduces over the global solver device axis;
- each process checkpoints only its owned persistent values;
- only the primary process writes shared reports; and
- one/two/four-device and one/two/four-process commuting tests remain valid.

## Consequences

"Z-slab" describes a private optimization, not an architectural layer or a
solver variant. Case runners now read as model + runtime + solver + effects.
The public API no longer implies that users choose between local and parallel
execution. Supporting another private partition will require a backend change,
not a new physics solver or a new application path.
