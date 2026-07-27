# ADR-0002: Field location is independent of storage

Status: **Accepted**

## Context

Legacy implementations often let an array index convention define physical
location: for example, an element may mean the upper face of a cell, and ghost
planes may be stored beside prognostic values. Such choices leak into every
operator and make a later layout or decomposition change appear to change the
mathematical field.

## Decision

A field's semantic location MUST be specified independently of its storage
layout. Locations are domain concepts such as `Cell`, `XFace`, `YFace`,
`ZFace`, or another explicitly accepted geometric entity. A backend interpreter
maps those locations to array extents, indexing, padding, and ownership.

The first interpreter supports only the smallest accepted set of locations
needed by the first discrete operator system. Unsupported locations fail during
program construction. The domain model is extensible by adding explicit
locations, but it MUST NOT pretend every location is already implemented.

Ghost cells and exchanged halo planes are transient boundary/neighborhood
context, not semantic field locations and not persistent prognostic state.
Physical boundary faces, by contrast, are genuine geometric locations even if
an interpreter can derive rather than store their values.

Interpolation or reconstruction between locations is an explicit morphism. A
reinterpretation of the same bytes as another location is prohibited.

## Required laws

- Storage round-trip preserves semantic coordinates and location.
- Owned interior extraction is invariant under halo-context construction.
- Location conversion of a constant field preserves that constant where the
  boundary contract permits it.
- Every stored element has one declared semantic coordinate and owner.
- Decomposition changes storage ownership, not physical location.

## Consequences

We can choose a hybrid spectral/staggered discretization without making its
particular array indexing universal. Face ownership at slab interfaces must be
defined by the distributed interpreter. The precise minimal location set
depends on the still-open compatible projection decision.
