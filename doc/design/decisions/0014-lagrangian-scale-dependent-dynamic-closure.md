# ADR-0014: LASD is an accepted-step closure event with complete restart memory

Status: **Accepted**

## Context

The static Smagorinsky choices admitted by ADR-0010 and ADR-0012 are
memoryless.  The Lagrangian-averaged scale-dependent dynamic (LASD) model is
not: it test-filters resolved products, averages Germano contractions along
pathlines, and updates its coefficients at a configured cadence.  Treating
that update as an ordinary vector-field evaluation would execute it once per
Runge--Kutta stage and make changing the integrator change the closure.

The Andrén et al. passive-scalar comparison additionally needs a prescribed
surface scalar flux, dynamic momentum and scalar diffusivities, and an
explicitly identified diagnostic SGS scalar variance.  LASD supplies SGS
stress and scalar flux, but not the stress trace or SGS scalar variance.

## Decision

### Closure event and read-only vector field

LASD coefficient advancement is a pure discrete transition at the beginning
of an accepted step:

```text
PrepareLASD(fields_n, memory_n, accepted_clock_n)
    -> fields_n × memory_prepared × event_diagnostic
```

AB2 executes this transition exactly once before evaluating the vector field
at `t_n`.  The vector field reads `memory_prepared` and remains a pure map from
one explicitly timed evaluation to tendencies and diagnostics.  The prepared
memory is carried unchanged into the accepted `n+1` state.  A future staged
integrator must retain the same once-per-accepted-step event law unless a new
decision explicitly changes it.

Every step accumulates the cell-centred trajectory velocity.  When
`(accepted_step + 1) % update_interval == 0`, the event computes Germano
contractions at the two test-filter scales, applies Lagrangian departure-point
averaging over the complete interval, updates grid-scale coefficients, and
resets the trajectory accumulators.  Otherwise it only advances those
accumulators.  The first update initializes contraction histories from the
current denominators using the documented initial coefficient.

The event configuration fingerprint includes test-filter grid ratio, test
ratio, update interval, Lagrangian time-scale coefficient, scale-dependence
and coefficient bounds.  A changed configuration invalidates LASD memory.

### Persistent memory

Momentum LASD owns cell-centred fields

```text
Cs2, LM, MM, QN, NN, u_lag, v_lag, w_lag.
```

Scalar LASD owns

```text
Ds2, scalar_LM, scalar_MM, scalar_QN, scalar_NN.
```

All are `Accepted` semantic fields with the same ownership as the transported
cell scalar.  They are closure memory rather than prognostic fluid variables
or AB2 tendencies.  A checkpoint writes every owned payload, the closure
configuration fingerprint, update schedule implied by the accepted clock,
and the ordinary AB2 tendency history.  Halos, filtered products, departure
coordinates, stresses, and scalar fluxes remain transient.

### Dynamic stress and scalar flux

The momentum coefficient determines

\[
  \nu_t=C_{s,\Delta}^2\Delta^2|S|,
  \qquad
  \tau_{ij}-\tfrac13\tau_{kk}\delta_{ij}=-2\nu_t S_{ij}.
\]

The independently dynamic scalar coefficient determines

\[
  K_c=D_{s,\Delta}^2\Delta^2|S|,
  \qquad q_i=-K_c\partial_i c.
\]

The lower scalar face takes the configured prescribed flux and the upper face
takes the configured zero or prescribed flux.  The same face flux enters the
conservative scalar tendency and diagnostics.  Dynamic contractions use
cell-centred physical gradients and therefore do not mistake a boundary ghost
reconstruction for resolved interior information.

### Diagnostic SGS energy and scalar variance

LASD does not manufacture an isotropic stress trace or scalar variance.  For
comparison plots only, WIRE-LES diagnoses SGS energy from the local
production--dissipation balance

\[
 e_{sgs}=\left[\max(P\Delta/C_\epsilon,0)\right]^{2/3},
 \qquad P=\nu_t|S|^2
\]

for the neutral passive-scalar case.  The diagnostic SGS scalar variance uses
the complete scalar-flux contraction

\[
 \sigma^2_{c,sgs}=
 \max\left(
   \frac{-2\ell_c q_i\partial_i c}
        {C_c\sqrt{e_{sgs}}},0
 \right),
 \qquad \ell_c=\Delta\sqrt{D_{s,\Delta}^2}.
\]

The constants and formula are part of the diagnostic fingerprint.  Output
must label this contribution `diagnostic SGS`; it is a fifth-model result and
must not be presented as one of the four original Andrén closures.

At the first physical cell, diagnostic production uses the vertical shear
implied by the configured neutral log wall, rather than the centered interior
gradient.  The scalar wall gradient is reconstructed from the exact prescribed
face flux and the interpreted diffusivity.  These are diagnostic boundary
semantics: they neither change the deviatoric LASD stress nor add a prognostic
energy equation, but they prevent a spurious division by vanishing centered
shear in the near-wall scalar-variance budget.

The generic diagnostic interpretation exposes both the local ratio and its
pre-division numerator.  A horizontally homogeneous benchmark may average the
numerator and energy over its statistical plane before closing the variance.
That averaging is a benchmark observation algebra, not a solver operation or
a universal field semantic; non-homogeneous wind-turbine cases must not inherit
it.

For a prescribed statistically homogeneous wall flux, that observation
algebra may explicitly reconstruct the diagnostic wall gradient with the
horizontal mean diffusivity.  This satisfies the flux relation in the
plane-mean sense without dividing by pointwise dynamic coefficients that may
legitimately be zero.  The choice is fingerprinted and defaults off; it is
invalid for a spatially localized turbine or spray forcing.

## Distributed interpretation

Horizontal test filters remain slab-local FFTs.  Lagrangian interpolation
uses periodic horizontal indices and packed nearest-neighbour z halos.  The
configured law requires

\[
  update\_interval\,\max(CFL_x,CFL_y,CFL_z)<1
\]

for a one-cell departure halo; exceeding it is reported as a warning and
invalidates the claimed interpolation accuracy but is not a hard solver clip.
No process stores or gathers a global closure field.

## Required laws

- a LASD event executes exactly once at its accepted-step boundary;
- skipped events change only trajectory accumulators;
- uninterrupted and checkpoint-restarted coefficients, histories,
  trajectories, tendencies, and diagnostics agree;
- rescaling a passive scalar changes neither its dynamic scalar coefficient
  nor its normalized variance and flux statistics;
- uniform velocity and scalar fields produce finite bounded coefficients and
  zero SGS divergence;
- reference and one-, two-, and four-slab event, stress, flux, and diagnostic
  interpretations commute within the declared dtype tolerance;
- prescribed scalar boundary flux changes the domain scalar integral by the
  exact net boundary flux;
- diagnostics do not execute an additional LASD event.

## Consequences

LASD adds substantial persistent memory and horizontal filtering cost, but it
does not alter projection or AB2 coefficients.  The Andrén run becomes a fifth
SGS-model comparison with resolved, modeled flux, diagnostic SGS variance,
and total curves kept explicitly separate.
