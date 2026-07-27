# Design principles

## 1. Semantics precede representation

A velocity field, a face flux, and a cell-centred scalar are different semantic
objects even when all three happen to be floating-point arrays. Location,
physical quantity, units, domain, boundary ownership, and decomposition MUST be
part of their declared type or validated metadata.

Array shape alone MUST NOT define meaning. JAX arrays are an interpretation of
the model, not the model itself.

## 2. Pure core, explicit boundary

The numerical core MUST be referentially transparent: equal inputs produce
equal outputs, and an input state is never mutated. Configuration parsing,
filesystem access, clocks, logging, process initialization, device discovery,
and serialization belong to an outer effectful shell.

Randomness MUST be explicit data. A stochastic transition consumes an RNG
state and returns the successor RNG state; it MUST NOT read a hidden global
generator.

## 3. Composition is the primary API

Small transformations MUST compose without an orchestration object that knows
their concrete implementations. Dependencies flow through arguments and
products, not service locators, mutable registries, inheritance trees, or
module globals.

Higher-order constructors SHOULD assemble boundary reconstruction, discrete
operators, physical tendency terms, integration, projection, and diagnostics.
Composition MUST remain visible enough to test each law independently.

## 4. Make invalid compositions unrepresentable

The solver MUST distinguish the phases of a step. In particular, an owned
prognostic field is not a halo context, a raw tendency is not a projected
state, and a face flux is not a cell-centred source. A function SHOULD accept
the narrowest semantic phase it needs and return a distinct phase when its
postconditions change.

Boundary reconstruction and halo exchange MUST create transient context; they
MUST NOT turn ghost values into persistent prognostic state.

## 5. Laws outrank mechanisms

Every reusable abstraction MUST have executable laws. Identity, associativity,
conservation, idempotence, equivariance, and local/distributed commutation are
preferred specifications because they remain meaningful across rewrites.

A category-theoretic name without a useful composition rule or falsifiable law
MUST NOT enter the public vocabulary.

## 6. Algebraic data over boolean mode matrices

Independent data is represented by product types. Mutually exclusive model
choices are represented by tagged sums. Model selection MUST NOT be encoded as
large records of loosely related booleans whose combinations include invalid
states.

Diagnostics and independent tendency contributions SHOULD use explicit monoid
operations so parallel reduction and combination order have stated semantics.

## 7. Backend transformations preserve meaning

JIT compilation, batching, differentiation, precision selection, layout
conversion, and sharding are structure-preserving transformations. They MUST
not redefine boundary conditions, governing equations, diagnostic meaning, or
time alignment.

Local and distributed executions of the same semantic program MUST commute up
to a documented floating-point tolerance. A process MUST NOT materialize a
global field merely because global shape metadata exists.

## 8. Performance is part of the contract

Purity does not justify avoidable allocation or communication. Persistent
state MUST be minimal, large arrays MUST be explicit dynamic arguments rather
than captured constants, and communication MUST be derivable from ownership
types. Buffer donation and fusion MAY be interpreter optimizations only when
the semantic laws remain unchanged.

Performance claims MUST separate compilation, execution, communication, and
I/O, and MUST be backed by synchronized measurements.

## 9. Vocabulary must be literal

Names describe mathematical roles. A pair of concurrent forward simulations
is not an adjoint. A pressure projection is not a compressible pressure model.
Resolved, SGS, and total quantities remain distinct until an explicitly named
combination morphism is applied.

## 10. Minimality before extensibility

The first implementation slice MUST be the smallest system that demonstrates
the algebra and its laws. Spray, cloud physics, actuator models, and elaborate
SGS closures MUST NOT shape the core interfaces before scalar conservation,
projection, boundary reconstruction, and local/distributed equivalence are
established.
