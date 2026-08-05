# ADR-0008: JAX-only array interpretations

Status: **Superseded by [ADR-0015](0015-unified-zslab-interpreter.md)**

## Context

The solver needs a small reference interpretation that can falsify production
operators, and it needs optimized local and distributed JAX interpretations.
Supporting NumPy as a second array backend would require an array-namespace
abstraction, parallel backend tests, and compromises around JIT and SPMD
primitives before there is a user requirement for a second backend.

Using JAX for both paths creates a different risk: if the reference calls the
same optimized kernels, agreement becomes tautological. Backend independence
and implementation independence are therefore separate concerns. We require
the latter without introducing the former.

## Decision

JAX is the only numerical array backend for the active solver. The project does
not provide a NumPy interpreter, an `xp`-style generic array namespace, or a
public multi-backend abstraction.

This does not put JAX into the semantic model. Domain, algebra, quantity,
location, ownership, phase, and physical-configuration modules remain ordinary
immutable Python and MUST import without importing JAX. The effect shell owns
JAX configuration, device discovery, distributed initialization, and external
array conversion.

### Independent JAX reference interpretation

The tiny-grid reference interpreter uses direct, readable JAX operations on a
bounded global test array. It is independent of the production interpretation:

- it MUST NOT import production operator kernels, halo implementations,
  pressure implementations, or production factor builders;
- it constructs boundary values, difference matrices, and compatible
  projection operations directly from the accepted semantic definitions;
- dense assembly and `jax.numpy.linalg.solve` are permitted on bounded grids;
- float64 is the default reference precision, following ADR-0005;
- eager execution is the default debugging interpretation, although tests MAY
  JIT the same reference function to check transformation stability;
- reference global materialization is test-only and protected by an explicit
  maximum grid-size check.

Sharing static semantic values, mathematical constants, and test case data is
allowed. Sharing the implementation under comparison is not.

### Production JAX interpretations

Production has separate local and SPMD interpretations. They may use JIT,
automatic batching, sharding, collectives, buffer donation, FFTs, and external
interpreter-only numerical libraries when those transformations preserve the
semantic laws.

Production code never falls back to the global reference implementation and
never gathers a field in order to call it. A logical global `jax.Array` is
allowed only when each process holds addressable shards consistent with
ADR-0007; it does not authorize host-global materialization.

Choosing JAX does not settle open decision H. A closure or external solver may
still have an explicitly nondifferentiable contract until differentiability is
accepted as a release law.

## Required laws

- Importing semantic domain modules does not initialize or import JAX.
- Reference results depend only on semantic inputs and explicit scale values,
  not device count or production configuration.
- Local JAX and JAX SPMD outputs commute with the independent JAX reference
  within declared precision-dependent tolerances.
- JIT and eager evaluation of a pure reference function agree within the
  declared tolerance.
- A dependency-boundary test fails if the reference interpreter imports a
  production numerical module or external pressure implementation.
- The reference interpreter rejects arrays above its documented global
  tiny-grid limit.
- No production path invokes reference global materialization.

## Failure and effect behavior

Unsupported dtype/backend combinations, missing JAX capabilities, an oversized
reference grid, and invalid runtime initialization order are construction-time
effect-shell errors in the first implementation. This decision adds no dynamic
compiled failure status and does not resolve E.

## Consequences

The active source package may depend directly on JAX in interpreter modules and
tests. It should prefer explicit JAX code over generic-backend adapters.

Correctness still has a non-tautological oracle because the reference uses
independent algorithms and dependency boundaries. The cost is that a JAX bug
shared by fundamental primitives such as indexing or linear algebra is not
cross-checked by another array library; manufactured solutions, algebraic
laws, and analytic scalar expectations provide the additional independent
evidence.

Adding another numerical backend later requires a new decision justified by a
real use case. It is not anticipated by placeholder interfaces today.
