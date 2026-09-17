# Conservative staggered momentum predictor

`src/jaxwind/spray_momentum.py` implements the source and advection part of a
variable-density momentum update on uniform MAC grids, including open x
boundaries, periodic or impermeable y, and impermeable z. It consumes the
**accepted primary gas mass flux**, including evaporation and ambient intake,
without rebuilding it from interpolated density. It remains an opt-in predictor;
it is not yet the momentum/pressure/thermodynamic timestep used by the waterjet.

The need for consistent mass and momentum transport on staggered grids is also
established in [Arrufat et al., mass-momentum consistent VOF transport](https://arxiv.org/abs/1811.12327).
That paper concerns interfacial flows. The algebra and verification here concern
our single gas inventory with discrete mass and momentum sources; its VOF
physical validation is not evidence of spray-model validity.

## Inventory partition and flux identity

For component `i`, split each primary cell in half and allocate its halves to
the adjacent velocity control volumes. Let `A_i` denote this inventory average.
Interior dual cells have the primary width; a nonperiodic endpoint dual cell has
**half width** and copies its one neighboring density. Each component has a
complete, non-overlapping partition of the physical volume.

Use `rho_i=A_i(rho)` as **momentum inertia**. The dual transport fluxes are:

- In transverse directions `j != i`, `G_ij=A_i(F_j)`.
- In direction `i`, dual faces at primary cell centers use the average of the
  two adjacent primary mass fluxes. At physical ends retain the primary boundary
  mass flux itself.

These choices give the exact uniform-grid identity

```
div_dual_i(G_i) = A_i(div_primary(F)).
```

Thus the primary mass/source budget implies every dual mass/source budget.
This identity includes the half-width endpoint cells. Using full endpoint
volumes, zeroing arbitrary boundary fluxes, or recomputing a second mass flux
would break it.

## Momentum, boundary, and energy ledgers

Inputs `Delta_rho` and `Delta_P` are conserved increments per primary volume,
not source rates or accelerations. `Delta_P` includes the full force impulse and
momentum of transferred vapor; the core exchange must supply it once.

```
rho_i* = A_i(rho_old + Delta_rho)
P_i* = A_i(rho_old) * u_i_old + A_i(Delta_P_i)
u_i* = P_i* / rho_i*
P_i_adv = P_i* - dt * div_dual_i(G_i * upwind(u_i*))
u_i_adv = P_i_adv / A_i(rho_new).
```

Both x ends use the supplied reservoir velocity on inflow. Outflow uses interior
velocity. Reservoir velocities are averaged onto each component's dual boundary
plane using the same half-cell partition. All components therefore satisfy

```
P_final_integral = P_old_integral + Delta_P_integral
                   - dt * outward_momentum_flux + wall_impulse_integral.
```

Impermeable normal velocity is set to zero with the change in gas momentum
returned as `wall_impulse`. The opposite impulse belongs to the wall; it is not
silently discarded. This is a kinematic predictor constraint, not yet the wall
pressure reaction of the full coupled flow solver.

Under the outgoing-mass CFL bound, upwind transport mixes nonnegative masses.
The diagnostic per component is

```
K_defect = (P_i*)^2/(2*rho_i*)
           - dt*div_dual_i(G_i * upwind((u_i*)^2/2))
           - (P_i_final)^2/(2*A_i(rho_new)).
```

It accounts for mean kinetic energy removed by numerical mixing and wall
clamping, after applying the source. It excludes the source kinetic-energy
change, thermal/latent energy, pressure work, viscosity and physical SGS energy.
In exact arithmetic it is nonnegative for admissible donor transport and a
consistent mass update. **It must not automatically become physical SGS k**:
its value depends on grid and advection scheme. A future total-energy driver
must distinguish numerical transfer, interphase dissipation, and physical
unresolved turbulence and close the corresponding ledgers explicitly.

Returned `kinetic_fluxes` make the advective boundary-energy term independently
integrable. All momentum fluxes, wall impulses, and energy defects are on component-specific
dual meshes. Integrating them with primary cell volumes is wrong at endpoints.

## Acceptance and integration requirements

The predictor rejects nonpositive/nonfinite densities, invalid wall mass flux,
nonfinite sources or state, a mass-budget mismatch, or an outgoing mass fraction
above one. Rejection restores input momentum/velocity/inertia and returns zero
committed flux, wall impulse, and energy defect. A future driver must roll back
carrier and liquid state together if any component rejects.

The donor scalar density `rho_transport` used by `spray_low_mach` differs from
momentum inertia `rho_i=A_i(rho_new)`. A physical pressure correction is

```
u_new = u_adv - dt*grad(p)/rho_i
F_new = rho_transport*u_new
      = F_predictor - dt*(rho_transport/rho_i)*grad(p).
```

Consequently, coupling these two stages requires a pressure operator
`div((rho_transport/rho_i)*grad(p))` (or an independently derived alternative),
not the existing unit-coefficient mass-flux correction. This is a discrete
inference from the two inventory definitions. Simply running the current scalar
stage followed by this predictor and another pressure correction would change
the mass flux after transporting species and enthalpy. The coupled iteration
must use the **final same flux** in all three equations and recover pressure
work consistently. The contact integration tests check flux compatibility only;
they do not justify that sequential production timestep.

The current predictor is first-order donor-cell transport on uniform grids.
Higher-order bounded momentum transport, coupled pressure work, turbulent stress,
spatial core ownership, measured transport and waterjet validation remain open.

## Verification

```bash
sbatch --output=outputs/spray_closure_validation/momentum-%j.out tools/verify_spray_momentum.sbatch
```

The suite checks all four x/y boundary combinations with independent surface and
volume quadrature, uniform motion with mass/momentum sources, a temperature and
humidity contact driven by the thermodynamic stage's actual returned flux, an
actual evaporating/dragging core's opposite liquid/gas impulse, and transactional
rejection for mass inconsistency and excessive outgoing transport.


Final gpudev **30272483** completed on the A100 with **10 tests passed**, no skips
or failures, in **37.673 s**. Across all four x/y boundary combinations the
largest relative dual mass residual was **2.187e-16**. Independently integrated
momentum and source-updated kinetic-energy budgets passed, including boundary
fluxes and wall impulses; numerical energy defects were nonnegative to the
1e-13 absolute roundoff allowance. The actual core test included evaporation
and nonzero drag in all three directions and matched the opposite liquid
momentum change. Source hashes for all seven recorded files match the final
verified state. Artifacts are ignored under
`outputs/spray_closure_validation/momentum-30272483/`.

Initial **30272467** also passed 10 tests; the final run additionally exposes and
checks the global kinetic-energy flux ledger and rejects nonpositive input
mass. These are numerical results only. No spray-profile, measured droplet, SGS
closure, or waterjet validation gate is advanced by calling this predictor pass
a physical validation.


The required [inertia-consistent pressure primitive](momentum-pressure.md) is now
implemented with a variable coefficient, full finite-volume gradient, independent
matrix/continuum references and a mechanical pressure-work ledger. It remains a
separate primitive: the common final-flux momentum/species/EOS iteration described
above is still required before using these components as a coupled timestep.


The [common-flux carrier step](coupled-carrier.md) now performs the shared
momentum/species/enthalpy/EOS iteration and recomputes momentum advection with
its accepted scalar mass flux. It tracks the remaining iteration residual
explicitly. This advances numerical coupling; total-energy treatment, source
ownership, physical SGS closure and production waterjet integration remain open.
