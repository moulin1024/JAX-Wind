# ADR-0012: One conserved potential-temperature perturbation supplies Boussinesq buoyancy

Status: **Accepted**

## Context

The finite-height dry Ekman benchmark fills the complete neutral domain and
therefore places its wind-speed maximum at the rigid top.  A physical
atmospheric boundary layer instead needs a prognostic stratifying scalar, an
inversion that caps turbulent growth, and a stable free atmosphere above it.
A Rayleigh relaxation can truncate a domain, but it is not a thermodynamic
capping inversion and must not be reported as one.

## Decision

The first coupled state is the product

\[
  X=(\boldsymbol u,\theta'),
\]

where velocity retains the hybrid cell/z-face locations and
potential-temperature perturbation is cell centred.  `Accepted` scalar state,
`Candidate` scalar state, and `Evaluated` scalar tendency are distinct static
phases.  Pressure projection changes only velocity; accepting a scalar
candidate is an explicit identity morphism with a phase change.

The scalar equation is conservative,

\[
 \partial_t\theta'=-\nabla\cdot(\boldsymbol u\theta')
                   -\nabla\cdot\boldsymbol q_{sgs},\qquad
 \boldsymbol q_{sgs}=-\frac{\nu_t}{Pr_t}\nabla\theta',
\]

with zero scalar flux at the lower and upper physical boundaries.  It uses the
same static-Smagorinsky strain magnitude and filter width as momentum.  The
buoyancy contribution is

\[
 F_w=b=g\theta'/\theta_0
\]

interpolated to vertical-velocity faces.  Horizontally uniform hydrostatic
buoyancy is a pure vertical gradient and is removed analytically by subtracting
the horizontal mean at every face before the terminal projection.  This is the
discrete pressure-gauge representative that avoids float32 cancellation;
perturbation buoyancy remains dynamically active.  Absolute potential temperature is reconstructed
as `theta_reference + theta_prime` for diagnostics.

The benchmark initializes a mixed layer, a finite-thickness positive inversion
jump, and a stable free-atmosphere lapse rate.  These are case data, not hidden
solver boundary behavior.  Rayleigh damping is not part of the thermodynamic
cap; the independent numerical top treatment is specified by ADR-0013.

## Distributed interpretation

The z-slab interpreter stores only owned cell scalar values.  Scalar neighbor
planes and SGS face fluxes are exchanged between adjacent slabs; no process
materializes a global scalar array.  The first implementation keeps the
existing packed velocity gradient bundle and a separate one-component scalar
exchange so that the dry interpretation remains unchanged.  Co-packing is a
performance refinement and cannot change the semantic program.

## Required laws

- scalar advection conserves the domain integral for impermeable boundaries;
- zero velocity and uniform scalar give zero scalar tendency;
- SGS scalar variance work is non-positive;
- buoyancy is supported only in vertical momentum and is linear in `theta'`;
- pressure projection leaves the accepted scalar payload bitwise unchanged;
- reference and one-, two-, and four-slab interpretations commute;
- uninterrupted and checkpoint-restarted coupled AB2 trajectories are equal;
- changing mechanical or temperature execution scales preserves the recovered
  SI tendency within the declared dtype tolerance.

## Consequences

An explicit capping inversion can keep turbulence away from the numerical top
and allow an interior supergeostrophic maximum.  Surface heating/cooling,
moisture, radiation, and nonlinear equations of state remain separate later
choices.
