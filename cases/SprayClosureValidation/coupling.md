# Conservative gas ownership and transfer contract

`src/jaxwind/spray_coupling.py` implements inventory operations needed by the
unresolved spray/LES interface. This is **numerical conservation machinery**,
not a completed LES coupling or a validated entrainment/dispersion model.
It does not enable the candidate in any waterjet case.

## State ownership

A `GasInventory` contains extensive cell or control-volume quantities:

| Field | Unit | Meaning |
| --- | --- | --- |
| mass | kg | Gas mass, with species masses summing to it |
| momentum | kg m/s | Three-component gas momentum |
| enthalpy | J | Thermal/species enthalpy, with a common reference |
| species | kg | Individually conserved species masses |
| unresolved_energy | J | Kinetic energy excluded from the inventory mean velocity |

Ambient and core inventories are **disjoint**. Their sum is the gas represented
by the combined model. An inventory partition replaces the original ownership;
the caller cannot also advance an independent copy of the original gas.
The optional `partition_gas_inventory` operation divides an existing inventory
at unchanged specific properties. Its prescribed fraction is a mass fraction,
not a derived geometric volume fraction.

`transfer_gas_mass` removes homogeneous donor gas and merges that same gas into
the receiver. Velocity, specific enthalpy, species fractions and pre-existing
unresolved energy follow the withdrawn mass. Requests exceeding donor inventory
are bounded cellwise and the unfulfilled amount is returned explicitly. The
closure driver must reject, subcycle or recompute a supply-limited request;
accepting reduced mass with unchanged predicted momentum/heat is inconsistent.

`withdraw_gas_to_core` performs the corresponding nonlocal operation from a
spatial ambient shell into one core inventory. It accounts for both variation
of donor-cell velocities and the difference between intake and core velocity.
Shell geometry and withdrawal weights are external physical model inputs.

For a local handoff, transferring all remaining core gas into ambient empties
the core. Repeating that handoff transfers zero gas. A spatial handoff from a
single core into multiple LES cells still needs a conservative reconstruction;
these operations do not supply or calibrate that profile.

## Mechanical-energy balance

For disjoint gas masses `m1`, `m2` with mean velocities `u1`, `u2`, the loss of
mean kinetic energy on merging is the exact identity

```
K_mix = 0.5 * m1*m2/(m1+m2) * |u1-u2|^2
K_mean = |momentum|^2/(2*mass)
```

The merger adds `K_mix` once to the receiving unresolved-energy inventory.
Existing unresolved energies add independently. Thus it preserves mass, each
species, vector momentum, enthalpy, and `K_mean + K_unresolved`. Empty inventories
have zero kinetic energy and cause no division by zero. Computation uses the
velocity-difference form rather than subtracting two large kinetic energies.
Gathering a shell uses the corresponding mass-weighted velocity-variance sum.

This identity specifies the energy that must remain accounted for. It does
**not** establish that all relative-motion energy immediately becomes isotropic
SGS turbulence. A stress/energy partition, production, transport and dissipation
closure remains necessary. Do not feed this reservoir directly into the
homogeneous Langevin model without that justification. Thermal enthalpy is not
silently increased by mechanical mixing; conversion to heat requires an
explicit, conservative dissipation step.

## LES integration still required

The existing waterjet carrier advances fixed-reference-density velocity,
temperature anomaly and moisture mixing ratios. It does not carry the gas mass,
mixture enthalpy and unresolved kinetic-energy inventories above. Simply
subtracting an entrainment source from its temperature/velocity arrays and then
adding a full jet source would not implement this contract.

The [existing spray design](../../doc/design/spray-framework-proposal.md#52-handoff-and-conservative-accounting)
proposes a quasi-steady nonlocal sink/source approximation, with negligible core
storage/displacement, and requires a compatible low-Mach mass/pressure treatment.
The new inventory routines support its accounting but do not replace that
approximation with a transient two-volume model. If core storage or displacement
is retained, volume ownership and the pressure constraint must be solved too.

The [finite-gas core exchange](core-exchange.md) now connects the shared water
droplet laws to an owned gas inventory, including coupled drag reaction, vapor
mass/momentum, thermal exchange, and an explicit mechanical-energy split. It
also tests withdrawal, exchange, and handoff together. This resolves the local
interphase ledger; spatial ownership/routing, a justified energy split, and the
LES pressure/EOS interface remain outstanding.

An opt-in [moist transport/pressure stage](low-mach-transport.md) now advances
conserved dry-air mass, vapor mass and enthalpy using the same corrected mass
flux, iterating to EOS consistency without replacing inventories. Its local
source test consumes actual core evaporation increments. This addresses the
thermodynamic transport/projection interface, but does not yet connect a spatial
core or its momentum/SGS sources to the production waterjet driver.

Remaining integration work includes:

- Define the mass/species/enthalpy state and pressure constraint used by the
  coarse-grid carrier; audit EOS and pressure-work consistency.
- Derive withdrawal shells and handoff profiles from the independently assessed
  jet model, including finite-plane loss and inventory exhaustion.
- Route parcel force, evaporated mass/momentum and heat to their owning gas
  inventory with equal/opposite exchange. Suppress duplicate resolved parcel
  sources before handoff, and avoid duplicate mixing after it.
- Advance and validate unresolved stress/energy and inhomogeneous dispersion;
  demonstrate well-mixed behavior and numerical convergence.

## Verification

The dedicated gpudev script runs the new conservation tests together with the
existing closure tests, recording source hashes and JUnit output:

```bash
sbatch --output=outputs/spray_closure_validation/coupling-%j.out \
  tools/verify_spray_coupling.sbatch
```

Coverage includes analytic unequal-mass mixing; cellwise donor exhaustion;
empty inventories; partition and repeated complete handoff; repeated partial
entrainment/handoff cycles; Galilean invariance and rotation covariance;
nonlocal withdrawal from a sheared three-dimensional shell; and equivalence
of grouped and successive mergers. These checks do not test a coupled LES run
or physical profile accuracy.

Gpudev job **30272031** completed on the A100 with **20 tests passed** in
32.40 s: seven new coupling tests and thirteen existing closure tests. These
JAX tests use float64. Results are in
`outputs/spray_closure_validation/coupling-30272031/tests.xml`, with the exact
kernel/test/script hashes in `source_hashes.txt`. The earlier job 30272024
failed before testing because its script omitted the CUDA module; the corrected
script loads `cuda/13.0` and explicitly verifies the GPU backend.


The [dual-volume momentum predictor](momentum-transport.md) provides a spatially
conservative route for full core momentum increments on uniform staggered grids.
It includes explicit wall impulses and boundary fluxes, and separates diagnosed
numerical kinetic-energy loss from physical SGS energy. It must be incorporated
in a common pressure/thermodynamic iteration and ownership transaction; applying
it after committing the carrier flux is not a completed conservative LES step.


The [coupled carrier](coupled-carrier.md) joins scalar transport, momentum and
pressure with a shared final mass flux and a single carrier commit/rollback.
Gas increments enter once despite nonlinear iteration. Its actual evaporation
source test checks full opposite phase impulse, but a spatial ownership driver
must still hold and commit liquid/core state together with carrier acceptance.
The standalone carrier does not own or roll back an external liquid state.
