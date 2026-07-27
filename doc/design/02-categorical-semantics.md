# Categorical semantics

This document uses a deliberately small part of category theory. Its purpose
is to constrain composition and interpretation, not to build an abstract DSL.

## 1. Categories and interpretations

We distinguish three levels:

- **Semantic category** \(\mathcal{S}\): objects are typed physical/discrete
  state spaces; morphisms are total, pure transformations with stated laws.
- **Array category** \(\mathcal{A}\): objects are immutable array pytrees with
  validated metadata; morphisms are pure array programs.
- **Distributed category** \(\mathcal{D}_M\): objects are arrays owned according
  to mesh \(M\); morphisms are SPMD programs whose communication is explicit in
  their interpretation.

Lowering the semantic model is a functor \(J: \mathcal{S}\to\mathcal{A}\).
Sharding for mesh \(M\) is a functor
\(D_M: \mathcal{A}\to\mathcal{D}_M\). They MUST preserve identity and
composition:

\[
J(\mathrm{id})=\mathrm{id},\qquad
J(g\circ f)=J(g)\circ J(f)
\]

and likewise for \(D_M\). These equations become law tests, not comments.

## 2. Products and sums

Coupled state is a product: velocity, thermodynamic scalars, SGS memory, and
time metadata remain independently addressable. Product projections MUST be
cheap and pure. Transformations that act on one component SHOULD be expressed
as lawful product lenses or explicit reconstruction, never mutation.

Alternative closures, boundary conditions, and integrators are tagged sums.
Elimination of a sum is an exhaustive interpreter. Adding a model therefore
requires a deliberate new case rather than a hidden boolean interaction.

## 3. Transitions with diagnostics

Let diagnostic values form a monoid \((\Delta,\oplus,e)\). A diagnostic
transition from \(X\) to \(Y\) is

\[
f: X\to Y\times\Delta.
\]

Composition is the Writer-style Kleisli composition

\[
(g\star f)(x) = \text{let }(y,d_1)=f(x),\ (z,d_2)=g(y)
                 \text{ in }(z,d_1\oplus d_2).
\]

The identity is \(x\mapsto(x,e)\). Associativity follows from the diagnostic
monoid and MUST be tested. A time step is an endomorphism in this transition
category. Diagnostics are therefore outputs, not callbacks hidden inside a
step.

Not every diagnostic combines by addition. `Maximum`, `Sum`, `Last`, and
structured products of monoids MUST be distinct types so their reduction law
is explicit.

## 4. Tendency algebra

For a fixed prognostic space, tendencies form an additive commutative monoid.
Independent physical terms are morphisms from a read-only evaluation context
to a tendency:

\[
F_i: C(S)\to T(S),\qquad F=\bigoplus_i F_i.
\]

This makes pressure-gradient forcing, buoyancy, Coriolis force, SGS stress
divergence, turbine forcing, and fringe forcing independently testable. The
commutative combination law applies only to already evaluated tendencies; it
does not license reordering context construction, boundary reconstruction, or
projection.

## 5. Typed phase graph

The minimal legal step factors through distinct objects:

\[
S_{owned}
\to C_{boundary}(S)
\to C_{differential}(S)
\to T(S)
\to S_{candidate}
\to S_{solenoidal}.
\]

The boundary context contains reconstructed physical boundary data. The
differential context contains shared gradients and filtered quantities. The
tendency contains no updated state. The candidate may violate the divergence
constraint. The solenoidal state carries the projection postcondition.

Skipping a phase or feeding a candidate state where a solenoidal state is
required MUST be rejected structurally or by validation at the interpreter
boundary.

## 6. Projectors and conservative operators

The incompressible projection \(P\) is a projector:

\[
P\circ P=P,
\]

within solver tolerance, and its output satisfies the declared discrete
divergence law. Gradient and divergence operators MUST document their discrete
duality or summation-by-parts relation. A flux divergence MUST conserve the
domain integral up to stated physical boundary fluxes.

Boundary reconstruction is a morphism into context, not an in-place edit. Its
laws include consistency with prescribed values/fluxes and invariance of owned
interior data.

## 7. Lawful lifts

Batching, ensemble axes, automatic differentiation, and distribution are
functorial lifts of a semantic morphism when applicable. For a lift \(L\):

\[
L(g\circ f)=L(g)\circ L(f).
\]

If a morphism cannot be lifted—because it performs I/O, depends on host order,
or has a nondifferentiable choice—that limitation MUST appear in its contract.
An actual tangent, cotangent, or adjoint program must be named separately from
an ensemble or concurrent-domain lift.

## 8. Pragmatic encoding rule

The implementation MAY use Python functions, immutable dataclasses/pytrees,
tagged enums, and protocols rather than encoding categories as runtime class
hierarchies. The category appears in composition functions, phase types, and
law suites. Runtime abstraction that adds dispatch or tracing complexity
without enforcing a law SHOULD be removed.
