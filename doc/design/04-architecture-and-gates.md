# Architecture and implementation gates

## 1. Dependency direction

The intended dependency graph points downward only:

```text
effects / applications / benchmarks
                |
interpreters: JAX local, JAX SPMD, reference
                |
integrators and complete step composition
                |
physics contributions and boundary policies
                |
discrete operators and algebraic laws
                |
semantic field, grid, ownership, and phase types
```

Semantic layers MUST NOT import JAX, filesystem libraries, launchers, plotting,
or benchmark configuration. Physics MUST NOT select a process topology.
Interpreters MAY depend on the semantic layers and backend libraries.

JAX is therefore a target interpreter, not the domain model. The reference
interpreter evaluates canonical SI values; the JAX interpreter lowers the same
semantic program through the accepted explicit nondimensional `ScaleSystem`.

## 2. Proposed package responsibilities

No directories are created by this document, but the eventual source tree
SHOULD separate these responsibilities:

- `domain`: quantities, locations, topology, grids, ownership, immutable state;
- `algebra`: products/sums, diagnostic monoids, transitions, composition laws;
- `operators`: gradient, divergence, filters, fluxes, projection interfaces;
- `physics`: pure tendency contributions and closure state transitions;
- `integrators`: higher-order construction of transitions from vector fields;
- `interpreters`: reference arrays, JAX local, and JAX SPMD lowering;
- `effects`: config, launch, checkpoint, logging, and postprocessing adapters;
- `benchmarks`: semantic cases and comparison criteria, not solver internals.

A single universal `Params` record is prohibited. Each morphism receives the
smallest immutable environment product that describes its dependencies.

## 3. Public API shape

The public core is built from functions with semantic signatures such as:

```text
reconstruct_boundary : OwnedField -> BoundaryContext
differentiate        : BoundaryContext -> DifferentialContext
contribution_i        : EvaluationContext -> Tendency
combine               : Product[Tendency] -> Tendency
integrate             : VectorField -> Transition[State, Diagnostic]
project               : CandidateState -> SolenoidalState
lower_backend         : SemanticProgram -> BackendProgram
```

These are pseudotypes, not Python APIs. Concrete signatures require accepted
decisions about field location, units, grids, and error values.

Closures with persistent memory are transitions over an explicit product of
flow state and closure state. They are not objects that mutate internal arrays.

## 4. Reference interpreter

Before optimization, a small, readable JAX reference interpretation MUST
establish the laws on tiny grids, following ADR-0008. It MUST NOT share
optimized kernels, halo implementations, or pressure adapters with the
production interpreter in a way that makes agreement tautological.

The reference interpreter is allowed to materialize a global tiny test array.
Production local and distributed interpreters are not.

## 5. Implementation slices

Implementation proceeds only through vertical slices that close their laws:

1. semantic field/grid/location types, products/sums, monoids, and law harness;
2. one conservative scalar flux on a tiny periodic/physical-boundary grid;
3. discrete gradient/divergence and incompressible projection;
4. a complete deterministic dry-flow step and exact restart;
5. wall/boundary policies and neutral ABL validation;
6. scalar transport and Boussinesq buoyancy;
7. SGS closures with explicit closure state;
8. JAX SPMD interpretation and local/distributed commuting tests;
9. concurrent precursor/fringe and actuator forcing;
10. optional Lagrangian or multiphase extensions.

Later slices MUST NOT weaken laws established by earlier slices. Optimization
enters only after a slice has a synchronized performance baseline.

## 6. Code admission gate

New active solver code is admitted only when all of the following are true:

- the relevant open decisions are resolved in writing;
- semantic input/output objects and phase changes are documented;
- every new quantity has a canonical SI meaning and scale mapping;
- persistent state, evaluation time, and continuous-rate/discrete-event
  semantics follow the accepted restart and forcing-time laws;
- composition, identity, and conservation laws are listed;
- reference and production interpretations have a comparison plan;
- local and distributed ownership are specified;
- failure and effect behavior are explicit;
- tests can fail for a plausible incorrect implementation.

Scaffolding that merely creates empty modules, abstract base classes, or
pass-through wrappers does not satisfy the gate.

## 7. Test hierarchy

Tests are organized by meaning rather than source file:

- **law tests**: identity, associativity, projector, round-trip, equivariance;
- **discrete physics tests**: conservation, symmetry, manufactured solutions;
- **interpretation tests**: reference/local/distributed commuting diagrams;
- **restart tests**: uninterrupted versus serialized continuation;
- **benchmark tests**: nondimensional profiles and spectra with stated error;
- **performance tests**: memory ownership, communication volume, synchronized
  kernel time, and scaling.

Snapshot tests MAY support plots but MUST NOT replace numerical acceptance
criteria.

## 8. Change protocol

Any PR that adds a public concept first updates these documents. The design
diff must state which laws are new, strengthened, weakened, or unchanged. If an
implementation reveals that a law is impossible or undesirable, work stops at
the document boundary; code is not used to make the decision implicitly.
