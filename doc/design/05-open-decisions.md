# Open decisions

These questions are intentionally unresolved. Choosing them in code before the
corresponding design amendment is prohibited.

## Resolved decisions

- A is accepted as [ADR-0001: static semantic field types](decisions/0001-static-semantic-field-types.md).
- B is accepted as [ADR-0002: location independent of storage](decisions/0002-location-independent-of-storage.md).
- C is accepted as [ADR-0004: compatible projection complex](decisions/0004-compatible-projection-complex.md).
- D is accepted as [ADR-0003: integrator interprets a vector field](decisions/0003-integrator-interprets-vector-field.md).
- A1 is accepted as [ADR-0005: dimensional semantics with nondimensional execution](decisions/0005-dimensional-semantics-nondimensional-execution.md).
- D1 is accepted as [ADR-0006: restart and forcing-time laws](decisions/0006-restart-and-forcing-time-laws.md).
- The first concrete D interpretation is accepted as [ADR-0009: fixed-step
  AB2 with one terminal projection](decisions/0009-fixed-step-ab2-projection.md).
- G is accepted as [ADR-0007: mesh-general ownership with a z-slab first interpreter](decisions/0007-mesh-general-ownership-z-slab-first.md).
- F is accepted as [ADR-0008: JAX-only array interpretations](decisions/0008-jax-only-array-interpretations.md).
- I's next milestone is accepted as [ADR-0012: one conserved potential-temperature perturbation supplies Boussinesq buoyancy](decisions/0012-boussinesq-scalar-and-capping-inversion.md).

Their remaining subordinate questions are listed below rather than being chosen
implicitly during implementation.

## E. Error and constraint representation

Construction-time invalidity can raise ordinary errors in the effect shell.
Dynamic failures inside compiled programs need explicit status values. We must
decide which conditions are impossible by type, which are validated once, and
which are runtime diagnostics.

## H. Differentiation and the word “adjoint”

Concurrent precursor/control domains are ensemble or coupled-domain axes, not
adjoints. We must reserve `tangent`, `cotangent`, and `adjoint` for actual
automatic/manual differentiation semantics.

Open question: should differentiability be a first-release law, or should
nondifferentiable closures and solvers be allowed with explicit contracts?

## I. First physical milestone

Recommendation: the first end-to-end milestone is a dry pressure-driven
neutral ABL without turbine or spray. The next milestone adds one conserved
scalar and Boussinesq buoyancy. This exercises the architectural core without
allowing multiphase requirements to dominate it.

## J. Case and reference-data ownership

All executable configurations live as ordinary top-level cases. Literature
inputs and reference figures are optional evidence owned by the case that uses
them; their presence does not select a different execution path. Semantic case
specifications and numerical acceptance data remain separate from the solver.
