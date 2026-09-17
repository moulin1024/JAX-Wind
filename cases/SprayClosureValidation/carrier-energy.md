# Compatible carrier energy extension

The opt-in `energy_coupling=True` control on `build_carrier_step` returns
numerical mixing loss and compatible pressure conversion to dilute enthalpy
**inside the nonlinear EOS/flow iteration**. Cell/group/moving/injection builders
forward the same control. Production waterjet settings are unchanged. Physical
SGS closure, complete moist thermodynamics and measured validation remain open.

## Explicit energy approximation

The existing dilute enthalpy H uses dry-air sensible heat and vapor latent
enthalpy, with zero vapor heat capacity. At prescribed, spatially uniform and
time-independent thermodynamic pressure p0, define internal energy density as
`E_internal = H - p0`. Write the mechanical pressure as `p0 + pi`. Keeping
mechanical work in this approximate model gives

```
dt(E_internal) + div(E_internal*u) = -(p0+pi)*div(u) + dissipation
=> dt(H) + div(H*u) = -pi*div(u) + dissipation.
```

This is the energy convention being discretized, not a claim that a leading-order
low-Mach enthalpy equation ordinarily retains every such term. The
[Nalu low-Mach derivation](https://nalu.readthedocs.io/en/latest/source/theory/lowMachNumberDerivation.html)
explicitly neglects kinetic/viscous work at its leading order and uses dynamic
pressure to enforce continuity. The present extension keeps a compatible
mechanical energy budget alongside its existing constant-p0 EOS. It does not
restore acoustic/compressible thermodynamics, vapor sensible heat or finite
liquid-volume pressure work. Reference enthalpies and the warm dilute regime
remain those of the existing phase-exchange model.

A closed heated domain at fixed p0 is still physically incompatible with fixed
volume/mass. Neither this extension nor trial-density mean projection changes
that fact. Evolving thermodynamic pressure would require another model.

## Compatible discrete transfers

Component kinetic energies occupy MAC dual volumes, with half-width boundary
volumes. `face_density_to_cell` distributes their extensive energy over adjacent
primary half cells. `kinetic_flux_to_primary` applies the matching flux map,
preserving the divergence locally, including physical endpoint fluxes. Using
full primary volumes for boundary dual cells would miscount energy.

Let Padv be the final-flux momentum predictor after the wall constraint,
Jwall its wall impulse, rho the final dual inertia and Ufinal the projected
velocity. Then

```
Uadv = Padv/rho
Umid = (Uadv+Ufinal)/2
wall_export_dual = ((Padv-Jwall)^2 - Padv^2)/(2*rho)
numerical_heat = map(kinetic_loss_dual - wall_export_dual).
```

The wall export leaves the modeled gas; it is not silently heated back into
it. The remaining upwind mixing loss goes to H. No energy term is clipped or
assigned to physical SGS k. This numerical transfer does not supply a physical
viscous or turbulent dissipation model.

Pressure is interpolated to faces with the same boundary convention as its
momentum gradient: zero pi at pressure-open x faces, copied cell pressure at
an imposed-flux inlet and impermeable walls. This yields the exact product
identity

```
map(-dt*grad(pi)*Umid) = -dt*div(pi_face*Umid) + dt*pi*div(Umid).
pressure_energy_flux = pi_face*Umid
pressure_conversion_to_H = -dt*pi*div(Umid).
```

Thus the pressure energy flux and H conversion balance mechanical pressure
work cell by cell. The midpoint includes the projection's temporal kinetic
change; the returned conversion is not solely a measurement or model of
continuum pressure dilatation. It includes the compatible discrete transfer.

Every carrier iteration reconstructs these terms from the same **final** mass
flux as scalar/momentum transport. Its candidate enthalpy is

```
H_candidate = H_staged - dt*div(H_advective_flux)
              + numerical_heat + pressure_conversion_to_H.
```

The EOS is then evaluated from that candidate. A post-acceptance temperature
patch would leave density, mass flux and pressure inconsistent and is not used.

## Budgets, source ownership and rejection

`carrier_energy_terms(result, dt, poisson)` recovers primary kinetic and
pressure fluxes, numerical heat, pressure conversion, wall export and a local
pressure-product residual from a returned carrier state. The transfers were
applied to H only if `energy_coupling` was enabled. A rejected carrier returns
zero transfers and fluxes. This helper performs no new pressure solve.

The nonlinear momentum residual remains explicit:
`iteration_work = residual_momentum * Umid`. It is a solver error, not an energy
source to be compensated by thermal or SGS adjustment. With it reported, the
cellwise carrier budget reads

```
(H+K)_final - (H+K)_staged
  + dt*div(H_advective_flux + K_flux + pressure_energy_flux)
  + wall_export = map(iteration_work).
```

The staged kinetic energy includes the phase source's full momentum and added
mass once. Phase exchange already balances that change against liquid, enthalpy
and its separately owned mechanical reservoir; numerical carrier heating does
not repeat phase drag dissipation. Transport of that reservoir uses the common
final mass flux. Injection, liquid export and external particle-force work
retain their separate per-step ledgers. All-phase rejection restores original
liquid, position, gas and reservoir through the existing transaction wrappers.

## Verification

`tools/verify_spray_energy.sbatch` runs on gpudev and retains source hashes,
logs and XML under ignored `outputs/spray_closure_validation/energy-*` paths.
The tests cover primary/dual divergence compatibility, pressure product rules
for periodic/open/fixed-inlet pressure conditions, an independently integrated
carrier energy budget, the EOS after heating, repeated injection/export with
gravity and complete source/carrier rejection. Existing carrier regressions
also run with the default energy control disabled.

Final gpudev **30273109** passed **10 energy tests** in 104.10 s, including
FutureWarnings as errors. Local pressure-product residuals were at most
**6.94e-17 J/m³**, or **3.70e-16** relative to the evaluated pressure-work scale.
The primary/dual kinetic-flux divergence tests passed all five boundary layouts.

The open carrier test deliberately combines shear with a compressive velocity
perturbation, exercising both upwind mixing and pressure projection. With
energy coupling disabled its total-energy defect was **-0.347770 J**. With the
extension it was **-1.506e-12 J** after separately reporting nonlinear work;
returned numerical heat was **0.086537 J** and compatible pressure conversion
**0.261233 J**. The latter includes the temporal projection transfer discussed
above, not just physical pressure dilatation. The heated EOS/flow iteration
converged in 20 iterations with momentum residual **3.030e-12**.

The eight-step injection/outflow/gravity case had maximum cumulative total-energy
residual **1.038e-12 J**, with **0.089959 J** injected and separately tracked
external work, gas/reservoir boundary fluxes and liquid export. Wall-loss
ownership and all-phase rollback passed. The fixed-p0 closed shear case
correctly rejected, retaining all original inventories: its unmodified EOS
error remained **5.602e-7** after 100 iterations, while the corresponding
energy-disabled carrier accepted. No density, mass or energy repair was used.

Earlier gpudev **30273084** passed all **14 existing carrier regressions** and
the injection/rollback energy cases. Six new tests failed: one used incorrect
mesh attribute capitalization, and five demanded an absolute pressure identity
error below ordinary floating-point roundoff (3e-17). The final tests use a
scale-aware bound of 32 machine epsilons and report actual residuals above.
Only test harness/roundoff checks and a docstring changed; the solver equations
were unchanged between these runs. Physical validation criteria were unchanged.
These results verify the stated discrete energy model, not measured spray
physics, full thermodynamic accuracy or a production SGS closure.
