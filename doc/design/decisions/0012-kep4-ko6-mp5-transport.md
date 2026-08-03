# Morinishi S4/KO6 momentum and complete MP5 scalar transport

## Status

Experimental implementation on `feature/kep4-ko6-mp5`.

## Decision

The non-spectral ABL runners default to the following transport split:

- momentum: fully conservative Morinishi staggered `Div-S4` transport;
- pressure: compatible fourth-order staggered `D4`, `G4`, and `-D4 G4`;
- momentum grid-cutoff regularization: conservative KO6;
- momentum SGS: AMD or physical-space LASD;
- scalar: complete local-Lax-Friedrichs MP5 flux;
- scalar SGS: scalar AMD;
- time integration: coupled SSPRK3 with FPJ2 after two exact startup steps;
- distributed explicit stages: one packed three-row y-halo message per neighbor.

The legacy second-order conservative MAC transport and additive MP5
regularization remain selectable for controlled comparisons.

## Momentum and energy properties

The KEP operator implements Morinishi et al. (1998), Eq. (101), directly on
the three native MAC velocity grids. For each momentum component and transport
direction it combines one- and three-mesh fluxes,

```text
C_S4 = (9/8) delta_1(a_S4 * mean_1(u))
     - (1/8) delta_3(a_S4 * mean_3(u)),
a_S4 = (9/8) mean_1(a) - (1/8) mean_3(a),
```

where the convecting velocity is first interpolated to the transported
component's staggered control-volume face. This is the conservative
`Div-S4`, not a collocated skew approximation. It conserves each periodic
momentum component a priori and, when the matching S4 continuity constraint
is satisfied, its resolved velocity work vanishes to roundoff:

```text
<U, C_S4(U)> = 0  if  Cont-S4(U) = 0.
```

At rigid lower and upper boundaries, tangential velocity and three-mesh
momentum-flux ghosts use the uniform-grid forms of Morinishi Eqs. (146)--(150).
For the first prognostic wall-normal face, the nonlinear normal self-flux uses
the conservative S2 member of the same staggered family. Feeding the Eq. (151)
outer ghost, whose coefficient is 26, directly into the nonlinear `w*w` flux
creates a grid-cutoff wall mode for rough LES initial fields; S4 resumes one
face farther into the domain. The implementation is regression-tested with a
fully periodic horizontal streamfunction, an x-z streamfunction satisfying
zero wall-normal velocity, and grid-cutoff wall noise.

KO6 is assembled from a third-difference operator `B` and a nonnegative local
coefficient:

```text
R_ko6(u) = -B.T nu6 B u / (64 dx),  nu6 >= 0,
<u, R_ko6(u)> <= 0.
```

This separates modeled AMD dissipation from numerical cutoff dissipation in
the diagnostics.

## Compatible pressure complex

The pressure path now uses a fourth-order face-to-cell divergence and defines
the cell-to-face gradient by the exact negative transpose:

```text
D4 = -G4.T,
A4 = -D4 G4 = G4.T G4,
u(n+1) = u* - dt G4 p,
D4 u(n+1) = 0.
```

Periodic face endpoints represent one degree of freedom and carry half weight
in the face inner product. Homogeneous-Neumann walls use even pressure
reflection and odd normal-velocity reflection, which sets the physical wall
gradient to zero while retaining fourth-order accuracy. The resulting Poisson
operator is symmetric positive semidefinite and
annihilates constants. Krylov iterations apply `A4` exactly; the compact
second-order symmetric GMG V-cycle is used only as a preconditioner.

This removes the former pressure/continuity mismatch: momentum fluxes,
continuity, pressure gradient, and the Poisson complex now use the same S4
staggered geometry. The legacy second-order compatible MAC path remains
selectable for controlled comparisons.

## Scalar qualification

The new scalar path reconstructs both states and uses them in the complete
MP5 numerical flux.  It replaces the former combination of a second-order
centered physical flux and an MP5-only dissipative correction.  MP5 controls
new extrema much more locally than a linear high-pass operator, although the
full multidimensional update, wall source, and SGS terms still require
boundedness checks at the selected CFL.

## Parallel layout

Morinishi S4, KO6, and MP5 each reach at most three neighboring rows. The
y-slab path therefore retains `halo_width = 3`. Velocity and scalar boundary rows are
flattened into one payload, exchanged once in each direction, and unpacked
before the fused stage tendency.  The complete vertical column stays local.

The y-slab FPJ2 path predicts the pressure for the first two SSPRK3 stages
and performs one distributed PPE at the accepted stage. It falls back to
three PPEs until two pressure histories exist, or whenever the timestep ratio
exceeds the configured safety limit. Both histories are included in MPI
checkpoints.

The distributed pressure apply exchanges three pressure rows because the
composed `D4G4` stencil reaches three cells. Pressure gradients exchange two
rows and face divergences exchange one extra interior face. These halo paths
reproduce the serial `D4G4` operator to floating-point roundoff.

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
--pressure-discretization {centered2,kep4}
--projection-method {full,fpj2}
--coupling-integrator {strang,coupled-ssprk3}
--fpj2-timestep-ratio-limit FLOAT
```

The new defaults are Morinishi S4 momentum, KEP4 pressure, KO6 momentum
regularization, and complete MP5 scalar transport.

## Primary reference

Y. Morinishi, T. S. Lund, O. V. Vasilyev, and P. Moin, “Fully Conservative
Higher Order Finite Difference Schemes for Incompressible Flow,” *Journal of
Computational Physics* 143 (1998), 90–124,
<https://doi.org/10.1006/jcph.1998.5962>.
