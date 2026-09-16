# Design documentation

The [shared nitrogen/water spray framework proposal](spray-framework-proposal.md) covers injection, entrainment, parcels, thermodynamics, turbine coupling, and validation. It is a design proposal, not implemented or numerically validated behavior.

[ADR-0018](decisions/0018-simulation-runtime.md) defines configuration, simulation construction, shared execution, workflows, and artifacts. [ADR-0017](decisions/0017-direct-finite-volume-solver.md) remains normative for the direct staggered finite-volume numerical implementation and its pressure backends. See the [architecture guide](../architecture.md) and [compute-node verification](../verification.md).

The numbered design specifications and ADR-0001 through ADR-0016 are retained as historical records. They describe the superseded semantic-field, interpreter, ownership, LASD, and solver-facade architecture and are not implementation requirements.

New architectural changes should amend or supersede the relevant active ADR before changing the corresponding public contract. Numerical acceptance of ADR-0018 remains pending compute-node verification.
