# KEP4/KO6 momentum and complete MP5 scalar transport

## Status

Experimental implementation on `feature/kep4-ko6-mp5`.

## Decision

The non-spectral ABL runners default to the following transport split:

- momentum: fourth-order skew/adjoint-paired KEP transport;
- momentum grid-cutoff regularization: conservative KO6;
- momentum SGS: AMD or physical-space LASD;
- scalar: complete local-Lax-Friedrichs MP5 flux;
- scalar SGS: scalar AMD;
- time integration: coupled SSPRK3 with FPJ2 after two exact startup steps;
- distributed explicit stages: one packed three-row y-halo message per neighbor.

The legacy second-order conservative MAC transport and additive MP5
regularization remain selectable for controlled comparisons.

## Energy properties

The KEP operator pairs every derivative with its negative Euclidean
transpose.  Its resolved velocity work therefore vanishes to roundoff:

```text
<u, C_kep4(u)> = 0.
```

KO6 is assembled from a third-difference operator `B` and a nonnegative local
coefficient:

```text
R_ko6(u) = -B.T nu6 B u / (64 dx),  nu6 >= 0,
<u, R_ko6(u)> <= 0.
```

This separates modeled AMD dissipation from numerical cutoff dissipation in
the diagnostics.

## Conservation qualification

The present pressure projection enforces the second-order MAC continuity
equation.  A fully conservative Morinishi fourth-order staggered scheme also
requires a matching fourth-order continuity operator, pressure gradient, and
Poisson operator.  Consequently, the implemented KEP4 operator is exactly
energy neutral but is not advertised as the fully conservative Morinishi S4
scheme.  The second-order `centered2` path remains the strict
mass/momentum/energy-compatible fallback until the pressure complex is
upgraded as one unit.

## Scalar qualification

The new scalar path reconstructs both states and uses them in the complete
MP5 numerical flux.  It replaces the former combination of a second-order
centered physical flux and an MP5-only dissipative correction.  MP5 controls
new extrema much more locally than a linear high-pass operator, although the
full multidimensional update, wall source, and SGS terms still require
boundedness checks at the selected CFL.

## Parallel layout

KEP4 needs two neighboring rows; KO6 and MP5 need three.  The y-slab path
therefore retains `halo_width = 3`.  Velocity and scalar boundary rows are
flattened into one payload, exchanged once in each direction, and unpacked
before the fused stage tendency.  The complete vertical column stays local.

The y-slab FPJ2 path predicts the pressure for the first two SSPRK3 stages
and performs one distributed PPE at the accepted stage. It falls back to
three PPEs until two pressure histories exist, or whenever the timestep ratio
exceeds the configured safety limit. Both histories are included in MPI
checkpoints.

At 64 cubed, y slabs are intended for two to four GPUs.  Eight GPUs leave only
eight owned y rows per rank and should use a future horizontal x-y pencil
decomposition instead of thinner y slabs.

## Runner controls

```text
--momentum-advection {centered2,kep4}
--momentum-regularization {none,mp5,ko6}
--ko6-strength FLOAT
--scalar-advection {centered_mp5,mp5}
--mp5-strength FLOAT
--projection-method {full,fpj2}
--coupling-integrator {strang,coupled-ssprk3}
--fpj2-timestep-ratio-limit FLOAT
```

The new defaults are `kep4`, `ko6`, and `mp5`, respectively.
