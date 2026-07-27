# ADR-0011: Rotation is an additive Coriolis--geostrophic tendency

Status: **Accepted**

## Context

ADR-0010 defines the first dry vector field as a sum of independently evaluated
tendencies.  A neutral Ekman layer additionally needs planetary rotation and
the pressure-gradient acceleration that is in geostrophic balance aloft.  The
projection pressure is a kinematic constraint and cannot supply this physical
driving law.

## Decision

Rotation is a tagged choice, `NoRotation | CoriolisGeostrophic`, and contributes
one more evaluated tendency without changing advection, wall, SGS, AB2, or
projection.  For vertical Coriolis parameter (f), optional horizontal component
(f_h), and geostrophic wind ((U_g,V_g)),

\[
F_u=f(v-V_g)-f_h w,\qquad
F_v=-f(u-U_g),\qquad
F_w=f_h(u-U_g).
\]

The default (f_h=0) is the traditional approximation used by the existing
benchmarks.  At latitude 45 degrees the Andrén et al. case uses (f_h=f).  The
cell/face interpolations in the two cross-location terms are an adjoint pair,
so the discrete operator remains globally skew-symmetric.

`NoRotation` is the explicit additive identity; a zero value of (f) is not a
second spelling of that choice.  The existing kinematic pressure-gradient term
remains independently composable.  An Ekman case uses zero additional
pressure-gradient acceleration because the geostrophically balanced pressure
gradient is already represented by the offsets in the equations above.

The semantic configuration uses SI units: (f) and (f_h) in (s^{-1}) and
geostrophic velocity in (m\,s^{-1}).  The first choice is height-independent.  A later
baroclinic or profile-valued geostrophic law requires a distinct tagged model.

## Required laws

- A velocity equal to the geostrophic wind is a pointwise fixed point.
- With (f_h=0), Coriolis forcing conserves pointwise kinetic energy relative to
  the geostrophic wind.  With (f_h\ne0), it conserves the corresponding global
  staggered-grid inner product.
- `NoRotation` is the additive identity.
- Reference and z-slab interpretations commute for either hemisphere and both
  float32 and float64.
- Adding rotation does not add a halo exchange or global reduction.

## Consequences

The dry vector field now evaluates five inspectable terms over the same shared
context.  Pressure projection remains terminal and unchanged.  Ekman benchmark
configuration is responsible for initial perturbations, runtime convergence,
and profile statistics; these are effects rather than hidden physics state.
