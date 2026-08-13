# ADR-0015: Unified z-slab production interpreter

Status: **Superseded** by
[ADR-0016](0016-unified-jax-solver.md).

Supersedes [ADR-0008](0008-jax-only-array-interpretations.md).

## Context

The initial architecture exposed separate global-reference, local-kernel, and
z-slab interpreter modules. Production cases never used the global or local
entry points: they already represented a single-process run as an
`EqualZSlab` with one shard. Keeping alternate production modules duplicated
concepts, obscured the real ownership model, and invited semantic drift.

An implementation-independent oracle remains valuable for tests, but it does
not need to be installed as a production interpreter.

## Decision

`jaxwind.interpreters.jax_zslab` is the sole public production interpreter.
All production execution uses `AddressableField`, `EqualZSlab`, and
`JaxZSlabInterpreter`.

A one-shard decomposition is the local and single-process case. It follows the
same construction, field lifecycle, numerical kernels, and effect boundaries
as a multi-shard decomposition. Neighbor exchange degenerates to physical
boundary handling; no alternate local interpreter is selected.

Multi-shard construction requires the caller's addressable global shard
indices. One-shard construction defaults to shard zero.

Backend kernels are private modules beneath `interpreters`. The production
package does not expose `jax_reference`, `jax_local`, or public actuator-kernel
modules.

The bounded global JAX implementation moves to `tests/support` and is called an
oracle. It MAY use direct global arrays and dense solves to falsify production
results, but:

- it is not installed or imported by active source;
- it is not a runtime fallback;
- it is not an alternative field representation accepted by cases; and
- its maximum global test size remains explicit.

JAX remains the only active numerical array backend. This decision changes
interpreter topology, not the semantic/domain dependency boundary.

## Required laws

- One-shard and multi-shard executions use the same public builder and
  interpreter type.
- One-, two-, and four-shard results commute within declared dtype tolerances.
- Production results commute with the independent test oracle on bounded
  manufactured cases.
- No module under `src/jaxwind` imports `tests.support`.
- `jax_zslab.py` is the only non-private interpreter implementation module.
- Production never gathers an owned field to invoke the test oracle.

## Consequences

Single-process behavior is no longer a separate backend or storage model.
Fixes and optimizations apply to local and distributed runs together. The
public API and test matrix are smaller, while the independent oracle retains
non-tautological correctness evidence outside the production package.

Tests that need canonical global arrays import the oracle from `tests/support`;
applications and benchmarks cannot.
