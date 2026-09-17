# Common-flux moist carrier step

`src/jaxwind/spray_carrier.py` connects conservative species/enthalpy transport,
staggered momentum advection, and inertia-consistent pressure in one opt-in
transaction. It advances carrier fields using a single converged mass flux,
including prescribed mass, enthalpy and full momentum increments. The production
waterjet still uses its existing timestep. Physical SGS modeling, spatial core
ownership, complete energy coupling and experimental validation remain open.

## Shared iteration

The interface is

```
step(gas, velocity, gas_increments, momentum_increment, dt)
```

`gas` contains primary-cell dry-air mass density, vapor mass density and dilute
enthalpy density. All increments are changes per volume **over dt**, not rates.
Momentum increments have three primary-cell components and contain the full
transferred-mass momentum plus force impulse. The caller must supply each owned
source once and retain old core/liquid state until the carrier transaction
accepts. The reservoirs specify dry/vapor fractions, specific enthalpy and three
velocity components at both x boundaries, used only on inflow.

Every nonlinear iteration restarts from the same source-updated inventories:

1. Compute intrinsic thermodynamic donor density and source-updated specific
   species/enthalpy. Initialize trial flow from the old velocity and donor
   density; initialize the EOS guess from the resulting trial transport.
2. Use the trial primary mass flux to determine its actual transported mass and
   the corresponding conservative staggered momentum predictor. This predictor
   uses the half-cell inventory partition and includes boundary momentum fluxes
   and explicit wall impulses.
3. Divide predicted momentum by the guessed final staggered inertia. Project
   this velocity using coefficient `rho_transport/rho_inertia` against the
   guessed final primary density and the original mass/source budget.
4. Use the resulting mass flux for **all three** conserved scalar fields. Compute
   the EOS from those transported fields, without replacing their masses.
5. Recompute momentum advection with this **same final scalar mass flux**. Check
   the actual pressure-corrected momentum equation, primary continuity, EOS,
   donor consistency after flow reversal, both primary and dual outgoing-mass
   bounds, and change in trial flow.
6. If necessary, relax the density guess toward the EOS and the trial flow
   toward the pressure-corrected flow. Repeat without reapplying any source.

This avoids accepting a momentum predictor based on an old mass flux and then
changing that flux during a scalar-only pressure solve. Pressure, donors,
transported density and momentum must all agree at acceptance. The Poisson
object provides mesh/boundary metadata; its unit-coefficient solve is not used
for the coupled momentum correction.

## Acceptance and transaction

The default relative tolerance is 1e-9, with up to 100 outer iterations and 0.7
relaxation. Momentum and flow errors use source-updated density and a velocity
scale equal to the maximum source-updated component speed, with a **1 m/s floor**.
Thus their stopping criteria have explicit reference scales even at rest.
These defaults are currently intended for float64 verification; lower precision
and production-scale performance require separate assessment.

Both the final momentum predictor and the pressure solve must accept. The
carrier also requires warm admissible thermodynamic fields, consistent donor
direction/density, nonnegative species under the outgoing-mass bound, and
converged EOS/flow/momentum/continuity errors. The final flux is authoritative:
`F=rho_transport*u` and equals the sum of returned dry-air/vapor fluxes.
Using interpolated final cell density to reconstruct another flux is incorrect.

On any failure the routine restores **all original carrier fields**, including
velocity and inertia, and returns zero committed scalar/momentum/kinetic fluxes,
pressure, impulses, energy ledgers and momentum residual. Iteration counts and
scalar diagnostic errors describe the rejected attempt. The standalone carrier
routine does not own an external parcel or core and therefore cannot roll back
one that a caller already committed. A production ownership driver must treat
carrier and liquid/core acceptance as one transaction.

## Closed-domain trial density constraint

For a closed domain, every linear pressure solve requires the trial new density
to have the mean mass implied by the original inventory and source. An EOS
estimate from an intermediate, unconverged transport field need not satisfy
that requirement. The iteration therefore projects **only its trial density**:

```
rho_guess <- rho_guess + mean(rho_old + Delta_rho - rho_guess).
```

The means are volume means (ordinary arithmetic means on the supported uniform
grid). Open domains need no such constraint because their boundary flux supplies
the mass difference. The source, transported dry/vapor masses, enthalpy and EOS
target are never modified by this projection. Final acceptance still requires
the unmodified EOS error and exact flux budgets to pass. A closed heated box at
prescribed pressure remains incompatible and must be rejected; this operation
cannot be used as a physical density repair or a mass sink.

## Momentum and mechanical-energy ledgers

All momentum/kinetic fluxes and force/work fields are on component-specific dual
control volumes, including half-width nonperiodic endpoints. The output exposes
the finite nonlinear solve error instead of relabeling it as a physical source:

```
E_i = rho_i_final*u_i_final - P_i_adv(final_flux) - pressure_impulse_i
momentum_residual = E_i
iteration_work = E_i * (u_i_adv + u_i_final)/2.
```

Here `P_i_adv` includes the source once and the final-flux advection/wall update.
`momentum_error` must be within the prescribed tolerance before acceptance.
The returned pressure work uses this same final-flux advective velocity, so

```
K_i_final - K_i_adv = pressure_work_i + iteration_work_i.
```

Together with momentum source, boundary momentum/kinetic fluxes, wall impulses,
and the predictor's numerical kinetic loss, these fields permit independent
whole-domain momentum and mechanical-energy quadrature. The small iteration
work is reported separately; it cannot be hidden in SGS energy or heating.

The thermal equation remains the warm dilute enthalpy convention shared by the
core component. Numerical kinetic loss and mechanical pressure work are
**diagnostics**, not extra enthalpy sources. Consequently these verified separate
budgets are not a claim that a full liquid/gas thermodynamic total-energy
closure has been established. Pressure-energy approximation, source deposition
work, physical interphase dissipation, numerical transfer and unresolved
kinetic energy must be combined without double counting in that next layer.

## Opt-in compatible energy extension

The preceding diagnostic-only thermal behavior remains the default. The new
[`energy_coupling=True` extension](carrier-energy.md) returns numerical mixing
loss and compatible pressure conversion to enthalpy inside every nonlinear
iterate, then evaluates the EOS from the heated state. It supplies primary
kinetic/pressure energy fluxes and explicit wall energy export. Nonlinear work
remains a reported solve error. Its constant-p0, warm dilute energy approximation
and numerical heating must not be confused with qualified physical SGS closure.
Cell/group/moving/injection transactions forward the same control.

## Verification scope

```bash
sbatch --output=outputs/spray_closure_validation/carrier-%j.out tools/verify_spray_carrier.sbatch
```

The coupled tests cover both directions of a moving temperature or humidity
contact, an independently predictable upwind shear wave, open-domain uniform
heating, and localized source increments produced by the actual finite-gas
water-droplet evaporation/drag component. They independently integrate scalar,
vector momentum and mechanical-energy budgets and verify opposite phase impulse
for the prescribed droplet source. Failure tests cover closed-domain heating,
outer-iteration exhaustion, pressure-iteration exhaustion, negative vapor and
excessive outgoing transport, checking that every carrier field and ledger rolls
back together.

Passing these numerical checks does not qualify a measured droplet inlet,
validate entrainment/profile shape, justify physical SGS energy, or establish
Montazeri waterjet accuracy. Higher-order transport, diffusion/stress, spatial
ownership/handoff and experimental convergence studies remain required.

## Multistep failure and correction

Initial gpudev **30272659** passed the first 12 coupled checks. Extending the
suite to evolving fields in **30272665** passed the full-domain contact crossing
but rejected the variable-density vortex, giving **13 passes and one failure**.
Targeted diagnostics **30272686/30272688** showed rejection on nonlinear
iteration 2: the pressure linear residual was **8.431e-12**, but physical mass
compatibility error was **4.314e-8**, above tolerance. The unconstrained EOS
trial density had the wrong closed-domain mean; the linear solver itself had
converged. Rejection preserved the original fields, so subsequent attempted
steps reproduced the same failure without partial updates.

The trial-mean constraint described above corrected this issue without changing
any final conservation, EOS or momentum gate. Targeted **30272689** passed both
the 12-step variable-density vortex and the unchanged incompatible closed-heating
rejection test. The vortex needed at most **13 outer iterations** per step, with
maximum EOS error **4.985e-12** and momentum residual **1.520e-11**. Its kinetic
energy decreased from **0.002313433 J** to **0.001866846 J**, and that change matched
the independently accumulated numerical-loss, pressure-work and iteration-work
ledgers. These separate mechanical and thermal budgets still do not establish a
full thermodynamic total-energy closure.


## Final verified run

Gpudev **30272691** completed on the A100 with **14 tests passed**, no failures or
skips, in **122.253 s**. All seven recorded source hashes match the final files.
The full run includes the corrected vortex, unchanged incompatible-source
rejection, and all original coupled checks:

- Four signed temperature/humidity contacts accepted in one iteration each.
- The independent shear-wave donor reference passed; diagnosed numerical kinetic
  loss was **0.001700516 J**, with 17 nonlinear iterations.
- Open heating used 17 iterations and had normalized momentum residual
  **1.243e-17**, with the expected outward flow and conservative boundary budgets.
- The actual finite-gas droplet source evaporated **1.618619e-7 kg**; its coupled
  carrier update used 16 iterations, EOS error **5.365e-15** and normalized
  momentum residual **1.801e-13**. Its source impulse balanced the liquid change.
- A temperature interface traversed the complete periodic domain in **32 steps**,
  retaining uniform velocity and the independent donor-cell reference. Maximum
  normalized momentum residual was **1.007e-15**.
- The 12-step variable-density vortex passed the accumulated scalar and mechanical
  budgets and the residual limits reported above.
- All five rejection cases preserved the original carrier fields and returned
  zero committed ledgers.

JUnit, diagnostics and verified hashes are ignored under
`outputs/spray_closure_validation/carrier-30272691/`. The failed intermediate
runs remain available. This establishes a tested common-flux carrier step,
not independently validated spray physics or a completed production LES model.

## Cell-owned phase transaction

The opt-in [cell exchange wrapper](cell-exchange.md) now couples a fixed group
of droplet bins directly to the live MAC inventory. It uses the actual
gather/deposit inertia, closes the source energy budget, transports an explicitly
owned unresolved mechanical reservoir with the final carrier mass flux, and
rolls back liquid as well as gas if either source or carrier fails. This covers
one cell-owned group; it does not supply spatial core ownership, moving parcels,
ambient entrainment, physical SGS dissipation or the complete carrier energy
equation.
