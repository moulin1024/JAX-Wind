# ADR-0013: Additive top Rayleigh--geostrophic damping

Status: Accepted

## Context

The explicit potential-temperature inversion in ADR-0012 represents the
thermodynamic cap, but it does not absorb horizontal inertial motions reflected
by the rigid numerical top.  In a finite Ekman-layer domain those motions can
accumulate resolved kinetic energy in the uppermost cells.

Rayleigh damping is a numerical open-boundary treatment, not a replacement for
the inversion or free-atmosphere stratification.  It must also remain compatible
with the pure vector-field, restart, and distributed-ownership laws.

## Decision

The Boussinesq vector field has an explicit Rayleigh choice.  The identity
choice contributes zero.  The active choice relaxes horizontal velocity toward
the configured geostrophic velocity and vertical velocity toward zero:

```text
eta(z)   = clip((z - z_start) / (Lz - z_start), 0, 1)
sigma(z) = sigma_max eta(z)^2
R        = -sigma(z) (u - ug, v - vg, w)
```

The quadratic ramp is fixed model semantics: its value and first derivative are
zero at the layer base.  There is no tunable ramp exponent.  The tendency is
integrated by the selected time integrator; it is not a hidden post-step filter.
Potential temperature is not damped.

The NeutralEkman capped benchmark enables the top 100 m by default with
`sigma_max = 1.6e-3 s^-1`.  This is independent of its 700--800 m inversion and
stable free atmosphere.

## Laws

- The tendency is exactly zero at and below `z_start`.
- Its kinetic-energy work relative to `(ug, vg, 0)` is non-positive.
- Reference and z-slab interpretations commute after storage conversion.
- Each z slab constructs the rate from its global static index; no halo exchange
  or global field is needed.
- Changing the Rayleigh configuration invalidates stored multistep tendency
  history.  A restart with changed physics uses an explicit cold AB2 restart.

## Consequences

The cap and the numerical top absorber have separate, inspectable roles.  The
extra production cost is local elementwise arithmetic only.  An overly deep or
strong layer can influence the physical Ekman solution, so benchmark output
records its support, maximum rate, and target velocity.
