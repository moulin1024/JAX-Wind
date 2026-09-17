# Shared-flux moist transport and EOS/pressure coupling

`src/jaxwind/spray_low_mach.py` connects conservative warm-gas species and
enthalpy transport to a projection of its shared advective mass flux.
It is an opt-in, first-order transport stage. It does not replace the production
waterjet timestep or complete the momentum, SGS, and spatial-core coupling.

## Why projection alone is insufficient

`low_mach.project_low_mach` can enforce

```
(rho_new-rho_old)/dt + div(Fmass) = Smass.
```

Given densities and sources, this is a mass-conservative pressure solve. It does
not by itself establish that the prescribed new density agrees with transported
species and enthalpy. Independently updating species on another velocity, or
replacing total density by an EOS value afterwards, can change dry-air mass or
break the scalar budgets. The new stage solves this coupling explicitly.

## State and thermodynamic convention

The cell fields are dry-air density, vapor density and enthalpy per volume:

```
rho = rho_d + rho_v
H = cp_d*rho_d*(T-Tf) + Lv*rho_v
Rmix = (Rd*rho_d + Rv*rho_v)/rho
rho_EOS = p0/(Rmix*T).
```

This is the same warm, dilute enthalpy reference as `spray_core`, with vapor
sensible heat omitted. Temperature and the EOS are evaluated without clipping
or density renormalization. The model requires positive dry-air density,
nonnegative vapor, temperatures at least freezing, and positive mixture heat
capacity at constant volume under this approximation. A complete thermodynamic
mixture model and fog/condensation coupling remain outstanding. The prescribed
thermodynamic pressure `p0` is constant; a sealed heated box requires a separate
pressure evolution equation and is not supported by this stage.

## Coupled discrete step

The input source is a **conserved increment per cell volume over the step**, not
a rate. A core handoff or withdrawal must supply it from its ownership ledger;
the stage does not create an additional copy of the core inventory. The momentum
predictor is a separate input, to be supplied by a future coupled driver.

1. Form the source-updated conserved fields `q*=q_old+Delta_q` and their specific
   properties `psi*=q*/rho*`.
2. Compute intrinsic donor density from the EOS of `q*`, including the ambient
   reservoir on inflow. Select donor faces using the current transport direction.
   Form the predictor mass flux from this density and the prescribed predictor
   velocity. Guess new cell density using the EOS of `q*`, then use
   `low_mach.project_mass_flux` to project the predictor flux against that cell
   density, old density, and `Smass=(Delta_rho_d+Delta_rho_v)/dt`.
3. Form one corrected face mass flux `F`. Advance **all three** conserved fields
   as `q_new=q* - dt*div(F*upwind(psi*))`. Open x boundaries use the supplied
   ambient reservoir on inflow and the interior state on outflow. Other normal
   boundaries are impermeable, except periodic directions.
4. Compute `rho_EOS` from these conserved fields. Relax the density guess and
   repeat until the species-summed density agrees with the EOS and the mass
   residual is below the prescribed relative tolerance. Every iteration starts
   from the same `q*`, so sources are applied only once.
5. Recover velocity using the **same donor face density that formed the
   predictor flux**. A pressure correction may reverse a face; check that its
   density and scalar donors agree before accepting. Return this explicit
   `transport_density` and the committed species/enthalpy face fluxes.

The returned advective flux is `transport_density * velocity`, or equivalently
sum of the two species fluxes. It generally differs from
`low_mach.mass_flux(rho_new, velocity)`, which uses interpolated cell density.
The latter remains the convention of the existing low-Mach solver; callers must
not substitute it into this stage's budget. Thermodynamic donor density also
must not be silently substituted for the inertia of a future MAC momentum
control volume. A conservative momentum driver still needs its own compatible
mass and momentum transport derivation.

The pressure correction acts on face momentum, retaining the existing compact
Poisson backends. Scalar fluxes are donor-cell for this first verified stage;
first-order time/space accuracy is a limitation, not the desired final
production discretization. The surface fluxes returned by the stage permit
independent global dry-air, vapor and enthalpy budget checks.

## Positivity, convergence and rejection

For each cell, the sum of outgoing mass during the step must not exceed its
source-updated mass. Under this donor-cell bound, nonnegative incoming states
preserve nonnegative species without clipping. The entire stage is rejected if
this bound fails, a thermodynamic state is inadmissible, or the EOS iteration
fails to converge, including inconsistent donor densities after flow reversal.
`eos_error` and `donor_error` report those criteria separately.
The default relative tolerance is `max(1e-9, 50*epsilon)`
for the field precision (about `5.96e-6` in float32); explicit tolerances override
it. Pressure-solver accuracy must be compatible with that choice. Rejection
returns both input fields and predictor velocity,
zero committed pressure/fluxes/transport density, and attempted-step diagnostics. A driver must
reduce the timestep or correct incompatible boundary/pressure assumptions.
When a source is computed by a core/parcel update, the coupled driver must also
retain and restore the original core and liquid states until this carrier stage
accepts. Carrier-only rollback after committing evaporation would lose the
phase budget. This standalone stage owns only the carrier fields.

There is no invented outlet strip or global mass sink to force a closed-domain
source to pass. A conservation result alone is insufficient for acceptance:
EOS consistency is checked independently.

Heating at constant pressure drives gas out of an open domain. Adding vapor
isothermally also produces outward volume flow. By contrast, evaporation into
warm air can cool it enough to contract the gas despite adding vapor mass;
the correct pressure response can then draw ambient gas inward. The source
response must therefore follow the thermodynamic state, not just vapor mass.

## Verification and remaining integration

The gpudev script runs new coupled-stage checks alongside the existing low-Mach
tests:

```bash
sbatch --output=outputs/spray_closure_validation/low-mach-%j.out \
  tools/verify_spray_low_mach.sbatch
```

The source test constructs its actual mass/enthalpy increments using
`advance_water_core`, then checks the combined liquid/gas enthalpy ledger and
boundary flux. An independent, uniform isobaric reactor supplies a convergence
reference: with volumetric heating `Q`, constant composition and no mass source,

```
T(t) = T0 * exp[Q*Rmix*t/(p0*cp_mix)],  cp_mix=Yd*cp_d.
```

This follows from `M*cp_mix*dT/dt=Q*V` and `M*Rmix*T=p0*V` for the vented box.
It does not use the discrete pressure/transport implementation as its reference.

Remaining work includes a conservative momentum predictor and pressure-work
accounting, compatible scalar diffusion and higher-order time integration,
spatial core withdrawal/handoff and parcel ownership, a justified unresolved
energy/stress model, and measured transport/waterjet validation. The current
stage does not accept lateral pressure outlets or evolve condensate, and it
must not be advertised as a completed low-Mach spray LES solver.

## Verified results

The following initial results predate the contact-preservation correction below.
They establish budgets and source behavior, but did not establish a consistent
velocity across a translating contact.

Gpudev **30272312** completed on the A100 with **14 tests passed** in 94.09 s:
ten new checks and four existing low-Mach checks. Exact hashes for nine files,
JUnit, recorded numerical diagnostics and a summary JSON are retained in
`outputs/spray_closure_validation/low-mach-30272312/`. The hashes were verified
against the final worktree.

- Uniform isothermal-vapor and heat sources satisfied the EOS in one iteration,
  with relative errors `1.59e-15` and `2.81e-14`. The committed species and
  enthalpy changes match independent boundary-face quadrature.
- Actual core evaporation added `7.40149e-8 kg` vapor. The carrier stage
  converged in five iterations with relative EOS error `7.051e-10`; minimum
  gas temperature was `309.89079 K` from `310 K`. Net outward gas mass flux
  was **negative**, `-1.97133e-4 kg/s`, correctly representing inward ambient
  flow during cooling. The combined liquid/gas **enthalpy** budget closes
  after boundary transport; mechanical pressure work remains excluded.
- Periodic composition transport conserves dry air, vapor and enthalpy while
  satisfying the EOS. Returned velocity reconstructs the actual transported
  mass flux, and its independently recomputed continuity residual passes.
- The sealed heated box, a negative source inventory, and an excessive
  outgoing-mass timestep all reject without committing carrier changes.
- For uniform vented heating over one second, the analytic final temperature
  is `319.1299316 K`. Errors with 4, 8 and 16 steps are `0.0334453`, `0.0167634`
  and `0.00839194 K`: the expected first-order timestep refinement.
- The float32 case preserves dtype and closes its boundary budgets, with
  relative EOS error `1.07412e-7`.

Earlier attempts remain recorded. **30272288** had 11 passes and two FFT
setup failures (a GMG-specific tolerance argument was incorrectly forwarded).
**30272293** had 13 passes and exposed a float32 grid-width promotion that
violated the JAX loop's carry dtype. The final implementation explicitly casts
widths to face-flux precision and passes the added single-precision check.

These results verify the thermodynamic transport/projection stage, not measured
spray behavior, a coupled LES energy budget, or waterjet sensor accuracy.


## Translating-contact audit and correction

The additional audit exposed an error missed by the original budget tests.
With speed 1 m/s, CFL 0.25, periodic boundaries and zero sources, an isobaric
contact must retain uniform speed and zero mechanical pressure gradient. We
checked a 290/350 K temperature contact at vapor fraction 0.01 and a
0.005/0.04 vapor-fraction contact at 310 K, on 8, 16, 32 and 64 streamwise cells.
For either of these individual contacts, the ideal-mixture EOS at fixed pressure
is affine in the conserved variables. Thus the constant-speed donor update,
`q_new=0.75*q_old+0.25*q_upstream`, is an independent discrete reference. This
claim is not extended to simultaneous arbitrary temperature/composition jumps.

Original gpudev **30272343** accepted all eight cases with small EOS and mass
residuals, but generated **0.072 m/s** maximum temperature-contact velocity
error and **0.0078896 m/s** composition-contact error. Neither declined with
refinement at fixed CFL. Pressure spans were about **0.4094 Pa** and **0.04653 Pa**.
The conservative fields already matched the donor reference: the inconsistent
interpolated density used to form/recover face velocity caused the defect.

The correction projects an explicit advective mass flux and uses matching
thermodynamic donor density to recover velocity, as specified above. The
existing `project_low_mach` API retains its interpolated-density convention,
using the shared new `project_mass_flux` primitive internally. Both density
conventions are explicit; this does not derive conservative MAC inertia.

Intermediate audit **30272346** reduced velocity errors below 2e-9 m/s but
still missed the unchanged 1e-7 Pa pressure screen in three finer cases at the
default 1e-9 EOS iteration tolerance. Tightening that solver tolerance to 1e-11,
without changing any contact acceptance threshold, gave **30272365: 8/8 pass**:
maximum velocity error **1.951e-11 m/s**, pressure span **1.953e-9 Pa**, and
normalized conserved-field error **8.083e-12**. Both failed audits are retained.
Pressure accuracy requires sufficiently tight thermodynamic iteration as dt
shrinks; the default relative EOS tolerance alone is not a pressure-error bound.

Gpudev **30272366** then passed **18 tests** in 112 s, including four new
bidirectional temperature/composition contacts, the existing evaporative source,
float32, convergence and rejection checks, and four existing low-Mach tests.
Final source hashes are verified in each ignored output directory. Reproduce:

```bash
sbatch --output=outputs/spray_closure_validation/contact-%j.out tools/audit_moist_contact.sbatch
sbatch --output=outputs/spray_closure_validation/low-mach-%j.out tools/verify_spray_low_mach.sbatch
```

This fixes a demonstrated numerical defect in the opt-in stage. It neither
establishes the cause of the Montazeri temperature discrepancy nor validates the
remaining momentum, SGS, measured transport or waterjet physics.


A [conservative dual-volume momentum predictor](momentum-transport.md) now
consumes this stage's actual mass flux and checks its dual mass budgets. It
preserves uniform moving contacts, accounts for full interphase momentum
increments, and exposes open-boundary momentum/kinetic-energy fluxes and wall
impulses. This is a verified predictor, not a completed coupled timestep. Its
inertia differs from the thermodynamic donor density, so the documented
variable-coefficient pressure correction and final-flux iteration are required
before production integration.


The [new momentum-pressure primitive](momentum-pressure.md) uses the distinct
transport and inertia densities explicitly. Its full open-boundary gradient and
variable-coefficient operator are opt-in; the existing scalar-only stage still
uses its earlier mass-flux projection. The two must be joined by a common
nonlinear iteration before pressure-corrected momentum and scalar fluxes can be
committed together.


For combined momentum and thermodynamic transport, the new
[coupled carrier step](coupled-carrier.md) now iterates with the conservative
momentum predictor and inertia-consistent pressure solver. This original
scalar-only component remains useful for its verification cases; it is not the
momentum-coupled production driver.
