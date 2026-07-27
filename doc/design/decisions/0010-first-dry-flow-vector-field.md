# ADR-0010: First dry-flow vector field is a sum of four conservative terms

Status: **Accepted**

## Context

ADR-0009 supplies a deterministic higher-order time interpretation, but its
tests use manufactured tendencies.  The first physical vector field must add
resolved transport, a driving pressure gradient, a neutral lower-wall model,
and an SGS closure without putting any of those choices into AB2 or projection.
It must also preserve the independent global reference and z-slab production
interpretations.

## Decision

The first dry incompressible momentum vector field is

\[
F(u)=F_{adv}(u)+F_{pg}+F_{wall}(u)+F_{sgs}(u).
\]

Each term returns an `Evaluated` velocity tendency at the prognostic component
locations.  Only evaluated tendencies are added.  Context construction,
boundary reconstruction, and projection are not members of this commutative
sum.

The prognostic locations remain cell-centred horizontal velocity and
z-face-normal vertical velocity.  Horizontal boundaries are periodic.  The
lower and upper normal velocities are impermeable, the upper tangential SGS
traction is zero, and the lower tangential traction is supplied exclusively by
the wall term.

### Resolved advection

Advection uses conservative flux form,

\[
F_{adv,i}=-\partial_j(u_i u_j).
\]

Cell velocities are linearly interpolated to z faces.  The vertical component
uses the matching face control volume and a cell-centred \(w^2\) flux.  Physical
normal-face tendencies are zero.  Horizontal nonlinear products use the fixed
two-thirds spectral truncation before differentiation.  This is product
truncation rather than a claim of exact 3/2-padding dealiasing, and is part of
the discretization rather than a case tuning parameter.

### Pressure-gradient driving

The configured pressure gradient is represented by its kinematic acceleration
\((a_x,a_y,0)\), in SI \(m\,s^{-2}\) at the semantic boundary.  It is a constant
body tendency, distinct from the gauge-fixed pressure correction computed by
projection.

### Neutral lower wall

The first cell centre at \(z=\Delta z/2\) supplies the local resolved horizontal
velocity.  With roughness \(z_0\),

\[
C_D=\left[\kappa/\log((\Delta z/2)/z_0)\right]^2,
\quad
(\tau_{xz},\tau_{yz})=-C_D |U_h|(u,v).
\]

The wall tendency is this lower traction divided by the first-cell height.  It
does not use a horizontal plane average and therefore remains valid when a
turbine destroys horizontal homogeneity.  The first implementation applies no
additional wall test filter.

### Static Smagorinsky SGS

The initial closure is memoryless static Smagorinsky,

\[
\nu_t=(C_s\Delta)^2|S|,\qquad
\tau_{ij}=-2\nu_tS_{ij},\qquad
F_{sgs,i}=-\partial_j\tau_{ij},
\]

with \(\Delta=(\Delta x\Delta y\Delta z)^{1/3}\).  Normal and cross derivatives
are evaluated once in a shared gradient bundle.  Cross stresses live on z
faces; normal and horizontal shear stresses live at cells.  SGS lower and upper
tangential tractions are zero so the separate wall term owns the lower physical
traction without replacement or double counting.

## Distributed interpretation

Production stores one upper z face per owned cell plus the separately
constructed lower physical face.  One packed velocity exchange builds all
cell/face interpolation and gradient context.  SGS performs one subsequent
packed stress exchange for lower cross-stress fluxes and the upper neighbour's
normal stress.  No process holds or gathers the global velocity array.

## Required laws

- Combining evaluated tendencies has identity, associativity, and commutativity.
- Uniform velocity has zero advection and zero SGS tendency.
- Conservative advection has zero domain-integrated horizontal momentum change
  for periodic horizontal boundaries and impermeable normal boundaries.
- Pressure-gradient forcing is exactly the configured acceleration at every
  horizontal-velocity degree of freedom.
- Wall work is non-positive pointwise and the wall term is supported only on
  the first physical cell layer.
- Static Smagorinsky work is non-positive up to boundary and round-off terms.
- The reference and one-, two-, and four-slab interpretations commute within
  their declared dtype tolerances, including every individual contribution.
- A dry-flow AB2 step remains restart-identical because the closure has no
  hidden state and the previous total tendency is already checkpointed.

## Consequences

This decision deliberately does not introduce LASD, molecular viscosity,
buoyancy, scalar transport, Coriolis forcing, a wall test filter, or closure
memory.  Each is a later contribution or tagged closure choice.  AB2 and the
projection program remain unchanged.
