# ADR-0005: Dimensional semantics, nondimensional execution

Status: **Accepted**

## Context

The solver must compose momentum, buoyancy, scalar transport, radiation, phase
change, particles, and turbine forcing. A completely nondimensional public
model makes coupling coefficients and conversions difficult to audit. Carrying
runtime unit objects through JAX, on the other hand, adds tracing and dispatch
complexity to the numerical kernel.

The same physical model also serves cases with very different natural scales:
an atmospheric boundary layer may use inversion height and friction velocity,
while a wind-tunnel turbine case may use rotor diameter and hub velocity.

## Decision

The semantic model, configuration, benchmark specification, and public
diagnostics use canonical SI quantities. Quantity and dimension are represented
by the static semantic types accepted in ADR-0001, with cheap validation when
external values enter the model.

JAX execution uses ordinary nondimensional arrays. A single explicit
`ScaleSystem` supplies the invertible transformation between dimensional
semantic values and an interpreted nondimensional program:

\[
q^*=\mathcal N_\Sigma(q),\qquad
q=\mathcal N_\Sigma^{-1}(q^*).
\]

The lowering boundary, not an individual physics module, applies this
transformation. Runtime unit packages and symbolic unit tensors MUST NOT enter
compiled numerical kernels.

The independent test oracle evaluates canonical SI values in float64 by
default. The production JAX interpreter evaluates the corresponding
nondimensional program. This independence makes scaling errors observable
rather than shared between the oracle and production paths.

## Quantity distinctions

Equal physical dimensions do not imply equal semantic quantities. In
particular, the type system MUST distinguish at least:

- absolute thermodynamic temperature;
- temperature difference or perturbation;
- potential temperature and virtual-potential-temperature perturbation;
- generic dimensionless scalars and moisture mixing ratios;
- Eulerian density/content, Lagrangian parcel mass, and their transfer source;
- thermodynamic pressure, kinematic pressure correction, and pressure gauge.

Absolute temperature uses a positive absolute scale. Temperature perturbations
may use a separate difference scale. An affine offset such as degrees Celsius
MUST be converted to canonical SI before semantic construction and MUST NOT be
used as an execution scale.

## Scale-system contract

A scale system is immutable, versioned program metadata. It defines coherent
scales for coordinates, time, velocity, density/mass, temperature quantities,
and any independent coupled quantity not safely derived from those bases.
Derived coefficients are produced by the scaling interpretation rather than
hand-normalized in physics code.

Case specifications choose physically meaningful references. For example, an
ABL may select \(z_i\) and \(u_*\), whereas a wind-tunnel turbine case may
select rotor diameter and hub velocity. Changing references MUST NOT change the
recovered dimensional solution.

The scale system is part of checkpoint and result metadata. Loading rejects an
incompatible or unknown scaling version. Postprocessing returns canonical SI
coordinates and quantities by default, while optional nondimensional outputs
must name their scale system.

## Required laws

### Round trip

For every supported semantic quantity,

\[
\mathcal N_\Sigma^{-1}(\mathcal N_\Sigma(q))=q
\]

within the declared dtype tolerance.

### Naturality of physical transformations

For every interpreted physical morphism \(f\),

\[
\mathcal N_\Sigma(f(q))
=f_\Sigma(\mathcal N_\Sigma(q)).
\]

The law covers states, tendencies, boundary values, closure memory, forcing,
and diagnostics—not only prognostic fields.

### Reference-scale invariance

For any two valid scale systems \(\Sigma_1\) and \(\Sigma_2\), converting both
interpreted results back to SI yields the same physical trajectory within the
documented numerical tolerance.

### Coupled conservation

Mass, momentum, scalar, and energy exchanges conserve their stated physical
totals in both the SI reference interpretation and the nondimensional JAX
interpretation. Scaling MUST NOT create an independent source or sink.

### Composition

Scaling preserves identity, products, and composition. One physics contribution
MUST NOT bypass or replace the program's scale system.

## Consequences

Public configuration remains physically readable, and dimensional mistakes are
caught before JIT compilation. Production arrays carry no unit objects. Every
new quantity requires a documented SI meaning and scale mapping before its
physics implementation is admitted.

The first vertical implementation slice must include a nontrivial scale-system
round-trip and reference/JAX commuting test; an identity-only scaling path is
not sufficient evidence.
