# Finite-gas interphase exchange

`src/jaxwind/spray_core.py` connects the existing spherical-droplet drag and
warm-water evaporation laws to the conservative `GasInventory`. It is a local
unresolved-core exchange operator, **not yet a spatial spray model or LES
coupler**. The production waterjet still uses its existing parcel path.

This closes an accounting gap between two previous components: the droplet laws
assumed a frozen carrier during a substep, whereas an entraining unresolved core
must receive the equal/opposite exchange in its own finite gas inventory.
No momentum, vapor, or heat is also deposited into the resolved LES here.

## Mechanical update

For one bin of aggregate liquid mass `M` and gas mass `mg`, freeze the shared
Morsi–Alexander relaxation rate `k` at the start of the full exchange step.
Without external forces, pairwise momentum exchange obeys

```
dv/dt = k (u-v)
du/dt = (M/mg) k (v-u)
w(t+h) = w(t) exp[-k (1+M/mg) h],  w = v-u
```

The pair barycentric velocity is constant. For reduced mass
`mu = mg*M/(mg+M)`, the exact loss of pair mean kinetic energy is

```
D = (mu/2) |w(t)|^2 {1-exp[-2 k (1+M/mg) h]} >= 0.
```

The code applies exact forward and reverse half sweeps over all bins. This
symmetric splitting is second order for fixed rates and needs only linear work
and storage. Every substep conserves vector momentum and accounts for its
nonnegative energy loss, including at high liquid loading and large `k*dt`.
For multiple bins, ordering still produces a finite-step splitting error;
stability and conservation do not establish time accuracy. Rates frozen from
nonlinear drag and the subsequent thermal split make the **combined exchange
first order**, requiring timestep refinement. Gravity is deliberately absent
from this local operator; an eventual spatial core driver must account for its
impulse and work separately.

## Evaporation, heat and reference convention

Gas species are `[dry air, water vapor]`. A bin stores mass per drop,
multiplicity, velocity, and temperature. Evaporation uses the shared
`physics.moisture.advance_water_droplet` update at the post-drag carrier state.
The carrier has finite dry-air mass `md`, so its temperature is recovered as

```
Tg = Tf + (Hg - Lv*mv)/(cp_d*md).
```

This uses the existing dilute convention `h_d=cp_d*(T-Tf)`, `h_v=Lv`,
`h_l=cp_l*(T-Tf)`. Vapor sensible heat is absent; this is not a complete mixture
EOS. Do not load a gas enthalpy using the absolute-temperature reference without
converting it to this convention.

Evaporated mass leaves each bin at its post-drag velocity. Its **full** momentum
`dm*v` enters the gas inventory, whose mass also increases. Using the existing
fixed-density velocity source `dm*(v-u)` as an extensive momentum source here
would lose momentum. Vapor carries `Lv*dm` formation enthalpy; the returned
sensible heat is debited from gas, preserving gas-plus-liquid enthalpy before
mechanical heating. The liquid velocity is unchanged by mass loss in this
zero-relative-ejection-speed approximation.

Vapor velocities can differ across bins. Their mass-weighted variance and the
energy of mixing their mean velocity into gas are accounted for once. The
caller must supply `drag_heat_fraction` in `[0,1]`, with **no default**: that
fraction of both drag and vapor-mixing energy goes to gas enthalpy; the remainder
goes to the owned unresolved-energy reservoir. Using one fraction for both
processes is a restricted parameterization, not an established physical model.
No value has been calibrated or justified for the waterjet. This unresolved
reservoir cannot be treated as isotropic SGS turbulence without an additional
stress/energy model and validation.

The discrete invariant is

```
Hg + sum(Hliquid) + Kgas_mean + sum(Kliquid) + E_unresolved.
```

This is an **enthalpy-plus-mechanical exchange ledger** under the stated
reference approximation, not verification of a closed-volume total internal
energy equation. EOS, displacement, pressure work, and the low-Mach projection
must still be made consistent in the spatial LES coupling.

## Rejection and ownership

If the candidate carrier temperature falls below freezing, or any candidate
field becomes nonfinite, the operator returns `accepted=False` and both original
states unchanged. It never clips the gas temperature while accepting liquid
changes. Diagnostic evaporation/energy/temperature fields describe the attempted
candidate, including on rejection; a driver must commit them **only** after an
accepted step. The driver must reduce the timestep and recompute on rejection.
A warm accepted state may still be supersaturated; fog/condensation competition
and validity of the shared warm spherical-drop laws remain separate requirements.

An integration check exercises the whole local ownership cycle: withdraw finite
gas from ambient, exchange with liquid bins in that owned core, then hand the
whole core back once. Repeated handoff cannot add gas or exchange twice. This
is an inventory-level integration test, not a demonstration of a physical
withdrawal shell, reconstructed profile, or solver pressure constraint.

## Verification

Run on the user-requested GPU development queue:

```bash
sbatch --output=outputs/spray_closure_validation/core-%j.out \
  tools/verify_spray_core.sbatch
```

The script records hashes, checks the CUDA backend, and runs the core tests
with the existing gas-inventory tests. The checks cover an analytic stiff
one-bin solution; multibin convergence against an independently assembled
matrix exponential; Galilean covariance and an empty bin; evaporating
species/momentum/enthalpy-plus-mechanical balance for three energy partitions;
transactional heat-exhaustion rejection; withdrawal/exchange/repeated handoff;
and combined drag/evaporation timestep refinement. Numerical verification does
not establish measured droplet transport or waterjet accuracy.

Gpudev job **30272202** completed on the A100 with **14 tests passed** in
23.45 s (seven core tests and seven existing inventory tests). JUnit and hashes
are retained at `outputs/spray_closure_validation/core-30272202/`; all seven
recorded file hashes were checked against the final worktree. The earlier
job **30272194** also passed 14 tests; the final run verifies a direct,
nonnegative unresolved-energy expression, including an initially empty
reservoir, and lint corrections. No measured validation claim follows from
these checks.
