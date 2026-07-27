# Architecture decisions

An accepted record is normative. A proposed record is explanatory and MUST NOT
be treated as an implementation decision. Superseded records remain here with
links to their replacements so the reasoning is auditable.

| Record | Status | Subject |
| --- | --- | --- |
| [ADR-0001](0001-static-semantic-field-types.md) | Accepted | Static semantic field types with construction-time validation |
| [ADR-0002](0002-location-independent-of-storage.md) | Accepted | Field location is independent of storage |
| [ADR-0003](0003-integrator-interprets-vector-field.md) | Accepted | Integrator as a higher-order interpretation |
| [ADR-0004](0004-compatible-projection-complex.md) | Accepted | Compatible divergence, gradient, Poisson, and boundary system |
| [ADR-0005](0005-dimensional-semantics-nondimensional-execution.md) | Accepted | SI semantic model with explicit nondimensional execution |
| [ADR-0006](0006-restart-and-forcing-time-laws.md) | Accepted | Accepted-step restart and explicit forcing-time semantics |
| [ADR-0007](0007-mesh-general-ownership-z-slab-first.md) | Accepted | Mesh-general ownership with a z-slab first interpreter |
| [ADR-0008](0008-jax-only-array-interpretations.md) | Accepted | JAX-only production and independent JAX reference interpretations |
| [ADR-0009](0009-fixed-step-ab2-projection.md) | Accepted | Fixed-step AB2, Euler startup, one terminal projection, and persistent history |
| [ADR-0010](0010-first-dry-flow-vector-field.md) | Accepted | Conservative dry-flow vector field with driving, wall stress, and static SGS |
| [ADR-0011](0011-coriolis-geostrophic-tendency.md) | Accepted | Additive Coriolis--geostrophic forcing and explicit no-rotation identity |
| [ADR-0012](0012-boussinesq-scalar-and-capping-inversion.md) | Accepted | Conserved potential-temperature perturbation, Boussinesq buoyancy, and explicit capping inversion |
| [ADR-0013](0013-top-rayleigh-geostrophic-damping.md) | Accepted | Additive top Rayleigh relaxation toward geostrophic flow |
| [ADR-0014](0014-lagrangian-scale-dependent-dynamic-closure.md) | Accepted | LASD accepted-step event, restart memory, scalar flux, and diagnostic variance |

Decision records constrain future code but contain no implementation. A record
is accepted only through an explicit design discussion.
