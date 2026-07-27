# ADR-0001: Static semantic field types

Status: **Accepted**

## Context

An unqualified array does not reveal whether it contains pressure or velocity,
whether values live at cells or faces, whether it is globally described or
locally owned, or which step phase has established its invariants. Encoding all
of this in a runtime symbolic tensor system would add tracing, dispatch, and
debugging complexity to every numerical operation.

## Decision

Location, physical quantity, ownership, and phase MUST be encoded with static
Python types wherever Python's type system can express them usefully. Values
also receive cheap construction-time validation for facts the static checker
cannot prove, such as shape matching a grid, a supported location, or ownership
matching a mesh.

Validation occurs when semantic values are constructed or when external data
enters an interpreter. It MUST NOT be repeated in an inner time step merely to
simulate dependent typing.

The project MUST NOT introduce a runtime symbolic tensor DSL. The numerical
payload remains an ordinary backend array carried by an immutable semantic
wrapper or product.

A representative pseudotype is:

```text
Field[Quantity, Location, Ownership, Phase, ArrayPayload]
```

The parameters are semantic markers; they do not imply a runtime class
hierarchy or dynamic multiple dispatch.

## Required laws

- Construction rejects incompatible grid, location, shape, and ownership.
- Rewrapping or lowering does not change the physical quantity or phase.
- Product projection preserves the component's semantic type.
- Backend transformations cannot forge a stronger phase postcondition.
- Static markers add no array-sized runtime storage.

## Consequences

Functions can state narrow signatures such as a divergence accepting a
face-normal velocity and returning a cell-centred scalar. Some invariants will
still be construction-time checks because Python cannot express dependent array
shapes. This is deliberate and smaller than a symbolic DSL.

The representation of physical units is not settled by this decision and
remains an open question.
