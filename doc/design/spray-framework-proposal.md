# Shared subgrid spray framework for nitrogen and water

Status: design proposal, 2026-09-09. No solver changes or numerical validation
are implied. This proposal complements the direct finite-volume solver and
packaged simulation contracts in ADR-0017 and ADR-0018; it does not supersede
them. Implementation that changes those contracts must amend the active ADRs.

## 1. Decision and intended predictions

Use one variable-density, low-Mach Eulerian gas solver coupled to finite-rate
Lagrangian liquid parcels. Add a reduced entraining spray model between an
unresolved atomization plane and the resolved plume. Nitrogen and water share
transport, injection, exchange accounting, runtime, and diagnostics. Material
properties and phase-change laws are separate.

Primary predictions are downstream mean temperature, temperature deficit
flux, plume trajectory and width, liquid survival, humidity, and deposition.
For turbine applications, include interaction with the resolved wake and ABL.
Nozzle internal flow, primary atomization, and blade erosion are not resolved.

The combined near-field/LES coupling below is an engineering design proposal,
not a single published and validated model. Literature supports its components;
validation must establish the coupled model's accuracy. In particular, an
unresolved nozzle does not imply that evaporation finishes within one cell.

## 2. What the repository already provides

- `src/jaxwind/cryogenic.py`: fixed-capacity JAX parcel state, deterministic
  injection, drag, gravity, cloud-in-cell sampling/deposition, and evaporation
  coupling. Diameter sampling is number-based because each injection batch
  receives a common multiplicity. `initial_diameter` is a distribution scale,
  not a Sauter mean diameter.
- `src/jaxwind/physics/cryogenic.py`: nitrogen droplet heating/boiling and moist
  saturation adjustment. The droplet heat-transfer routine uses constant air
  properties, including reference density in its Reynolds number; the parcel
  drag routine separately samples local density.
- `src/jaxwind/simulation/jet.py`: low-Mach mixture density, gas injection,
  buoyancy, and a volume-source option that injects all nitrogen as gas and
  subtracts the remaining liquid's latent heat locally. It advances temperature
  with prescribed heat capacity rather than a general mixture enthalpy state.
- `src/jaxwind/simulation/turbines.py`: turbine definitions and FV forcing.

Preserve useful numerical kernels, but audit conservation rather than assuming
the current coupling meets the new exchange contract. Existing source-radius,
temperature, and droplet defaults must not silently become physical calibration.
Existing nitrogen cases disagree on bore diameter: 10 mm versus 2 mm.

## 3. Injection contract

An injector supplies position/orientation histories, liquid material, total
mass flow, discharge state, velocity or measured thrust, and an atomized spray
distribution. Every uncertain input is recorded as an assumption, with units
and provenance. Do not require both independently specified enthalpy and quality
unless they are checked for consistency.

For the requested nitrogen baseline, assume a pressure-matched saturated
homogeneous discharge with common liquid/vapor velocity U. With area A,

    rho_mix = mdot / (A U)
    x = (1/rho_mix - 1/rho_l) / (1/rho_v - 1/rho_l)
    h_in = (1-x) h_l(p) + x h_v(p)
    thrust = mdot U

Reject x outside [0,1]; do not clip it. Permit both pure-phase endpoints.
This is a conditional closure, not a measurement of quality. An underexpanded
or slipping discharge instead needs a separate expansion/slip closure or
measured post-expansion fluxes. The low-Mach model starts after depressurization.

For ordinary water injection, use liquid mass flow and temperature, with no
injected vapor unless specified. Check mdot/(rho_l A) against the assumed speed;
an effective flow area/discharge coefficient may be needed. Atomization-plane
velocity, cone angle, and gas entrainment need not equal bore-exit quantities.

Use a consistent enthalpy reference for gas, liquid, and ice. Tabulate material
properties over the actual temperature/pressure range from an identified source,
such as [NIST fluid properties](https://webbook.nist.gov/chemistry/fluid/).
Offline property-table generation avoids host property-library calls inside JIT.

## 4. Atomized droplet distribution

Adopt a truncated, mass-based Rosin–Rammler distribution. Its untruncated CDF is
F_m(d) = 1 - exp[-(d/lambda)^k]. Normalize the CDF on [d_min,d_max] and solve
lambda so that

    1 / D32 = integral (1/d) dF_m(d)

for the truncated distribution. This identity follows from spherical droplet
area and volume. It prevents confusing a scale parameter, number mean, volume
median, and D32. Mass/volume distributions are a common spray convention; see
[NIST spray characterization](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=101399).

Provisional nitrogen prior: D32 = 150 micrometres, k = 3, bounds 0.2 D32 and
3 D32. Run D32 = 50, 150, 300 micrometres as exploratory sensitivity cases;
these are not established physical bounds. Test width separately, initially
k = 2, 3, 4. Water keeps the same representation but uses nozzle-specific data.
No universal law relating D32 to bore diameter is assumed.

Discretize initially into 12 equal-liquid-mass quantile bins. Use the harmonic
diameter in each bin to preserve its surface-area-to-mass ratio; confirm
evaporation/trajectory convergence with 24 bins. Spatial parcel sampling is
stratified across the spray cross-section. For liquid mass M_i assigned to a
parcel of diameter d_i, multiplicity is M_i/(rho_l pi d_i^3/6). Multiplicity
need not be an integer. Track injected mass and moments exactly.

Use an axisymmetric top-hat injection profile and prescribed cone angle until
measurements justify radial size/velocity correlations. Zero cone angle is an
explicit baseline, not evidence of a collimated atomized spray. The mean axial
momentum must account for off-axis droplet velocities and remain consistent
with the specified thrust. Keep distributions and liquid enthalpy correlated
with the discharge state in sensitivity studies.

Published [DLR phase-Doppler measurements](https://elib.dlr.de/138002/) provide
nitrogen size/velocity data for flash-boiling conditions. They are candidate
validation data only after matching operating conditions; they do not establish
the diameter of this nozzle's spray.

## 5. Unresolved spray and entrainment

### 5.1 Selected baseline

Use a quasi-steady, curved, top-hat integral spray model per injector. Advance
gas species mass fluxes, gas momentum flux, gas enthalpy flux, cross-section,
and the mass, velocity, temperature, and number flux of each liquid size bin
along arc length. Include gravity, ambient crossflow, and interphase exchange.
The liquid bins use the same droplet laws as resolved parcels.

This choice is motivated by
[nitrogen release/dispersion experiments and integral modeling](https://www.sciencedirect.com/science/article/abs/pii/S0957582023004354)
and [multiphase integral plume models in crossflow](https://link.springer.com/article/10.1007/s10652-018-9591-y).
The latter involves different fluids and environments; its framework supports
the architecture, not transfer of calibrated coefficients to air sprays.

For the first implementation, make the empirical entrainment law explicit:

    E = d(mdot_entrained)/ds
      = 2 pi b rho_a [alpha |u_g - u_a dot t| + beta |u_a_perp|]

Here t is the local tangent, u_g the axial gas speed, b the top-hat radius, and
ambient properties are sampled outside the modeled core. Start with alpha=0.07
and beta=0.5 as engineering priors, not literature-derived universal constants.
Sensitivity ranges are alpha=0.04–0.12 and beta=0.25–1.0. Validate still-air and
crossflow separately. A calibrated published entrainment law can replace this
function without changing the exchange contract.

Use mass, vector momentum, and enthalpy conservation to close the core. For a
slice, entrainment adds E times ambient species fractions, velocity, and
enthalpy. Gravity contributes the density-anomaly force (including liquid
loading) per unit length. Droplet drag and phase-change exchange appear with
opposite signs in gas and liquid equations. Determine area from gas mass flux,
EOS density, and axial speed; include finite liquid volume in the core geometry
where necessary. The trajectory follows core transport, while bins retain slip.

Baseline crossflow momentum transfer is by entrainment. Do not also add an
unfitted form-drag term that might duplicate it. If crossflow trajectory tests
fail, replace this baseline with a complete, consistently calibrated integral
closure rather than tuning unrelated momentum and spreading terms independently.

The initial plane is the end of primary atomization, not necessarily the bore.
Its offset, radius, and entrained-air fraction are explicit source parameters.
For pure water, initialize a finite air sheath from that prescribed entrained
air flux; withdraw its inventory from the resolved gas. A zero-gas-flux start
is singular in a gas streamtube model. Neither an arbitrary numerical epsilon
nor ambient air created without a ledger is acceptable. These atomization-plane
parameters require calibration or sensitivity alongside droplet size.

### 5.2 Handoff and conservative accounting

Transfer gas and surviving liquid when the modeled plume diameter spans at
least four local transverse cells and dilute spherical-droplet assumptions are
valid. Test three and six cells. Do not require complete evaporation. Use the
two actual transverse mesh spacings, not the streamwise spacing or global mean.

Implement the quasi-steady model as a nonlocal source/sink operator:

1. Remove entrained gas mass, species, momentum, and enthalpy from an ambient
   sampling shell along the unresolved trajectory.
2. Emit the predicted gas flux at the handoff with a normalized spatial profile.
3. Create surviving parcels with the predicted bin mass, velocity, temperature,
   and transverse distribution.
4. Account separately for core gravitational impulse and any external heating.

Then gross emitted mass minus withdrawn ambient mass equals nozzle mass. The
same accounting applies to species, momentum, and energy after external forces
and work. Shell withdrawals are bounded by available inventory; if this binds,
substep/recompute the near-field calculation rather than silently changing only
one side of the exchange. No independent nozzle vapor source, latent-heat sink,
or parcel drag acts in the LES before the handoff.

This is an effective-source approximation: LES fields inside the unresolved
core are not predictions of the true jet. The model assumes negligible core
storage and displaced volume at resolved scales. If this assumption fails,
the model is outside its domain; do not add a second overlapping gas inventory
without a gas-volume partition and a compatible pressure equation.

### 5.3 Applicability and transient releases

Compute core travel time and compare it with injector variation, ambient-flow
variation, and time since startup. The quasi-steady model is accepted only
when core transit is small relative to the times of interest; use 0.1 as an
initial timescale-ratio gate. A one-second nitrogen startup run may fail this
gate and must not be reported as transient validation of the effective source.

For rapid startup, moving jets, or long unresolved paths, a transient integral
control-volume extension is required: carry gas/liquid storage, causal axial
fluxes, conservative ambient exchange, and displaced volume. A mere time delay
on an otherwise steady source is insufficient for changing ambient conditions.
This extension is a separate implementation milestone, not a hidden capability
of the baseline. If the plume encounters a wall, rotor, another core, or loses
its coherent jet character before handoff, require a specialized interaction
model or a finer mesh; do not continue the isolated-jet law through it.

## 6. Resolved carrier and thermodynamics

Advance gas mass, species masses, momentum, and mixture sensible/formation
enthalpy consistently in the existing low-Mach FV solver. Temperature is
recovered from composition and enthalpy. EOS density and the discrete gas-mass
equation determine the projection constraint; spray evaporation is not followed
by an unrelated incompressible projection.

Use a dry-air background plus injected-N2 tracer and water vapor, with explicit
background nitrogen content when evaluating interfacial nitrogen partial
pressure. The injected tracer alone is not the total nitrogen concentration.
Density uses the full mixture gas constant. Thermal and compositional buoyancy
follow density; no additional prescribed negative-buoyancy source is applied.

Retain AMD momentum transport and the existing scalar eddy-diffusivity framework
as the baseline. Record turbulent Prandtl/Schmidt numbers and test scalar mixing
sensitivity. Do not retune spray diameters to compensate for excessive numerical
scalar diffusion. Use bounded conservative species/enthalpy transport.

Treat fine atmospheric condensate/ice as an Eulerian moist reservoir, distinct
from injected inertial droplets. At fixed total water and enthalpy, saturation
adjustment can create fog/ice; include condensate loading and settling as
appropriate. Large injected water droplets retain finite-rate thermodynamics.
Coordinate condensation on them with the fog reservoir to avoid removing vapor
twice. Surface heat flux and humidity are physical boundary inputs.

Cryogenic gas properties and moisture fits have validity ranges. Track core
states where oxygen/air condensation or solidification could matter; either
add that thermodynamics or establish its negligible effect before accepting
those cases. Ideal-gas air plus water microphysics is not a complete model of
air at all cryogenic temperatures.

## 7. Shared droplet laws

Parcel state contains position, velocity, liquid mass per droplet, liquid
enthalpy/temperature, multiplicity, material ID, persistent ID, and stochastic
dispersion state. Derive diameter from mass and liquid density. A parcel is a
statistical population, not a physical droplet enlarged to parcel size.

Use spherical finite-Re drag, local gas/film properties, and gravity with
displaced-gas buoyancy. Retain the current broad-Re drag correlation initially,
document its provenance and range, and verify terminal velocities. Use the
same drag implementation in near-field bins and resolved parcels.

For water, solve finite-rate sensible heating and evaporation/condensation
using surface saturation pressure, ambient vapor fraction, Sherwood/Nusselt
correlations, and Stefan-flow corrections. The structure is supported by the
[FDS particle equations](https://github.com/firemodels/fds/blob/master/Manuals/FDS_Technical_Reference_Guide/Particle_Chapter.tex).
Do not force water to evaporate only at its boiling point.

For nitrogen, use the same sub-boiling formulation with nitrogen properties,
and an energy-limited boiling branch near saturation at local total pressure.
The surface mass-fraction formula becomes singular as saturation approaches
pure vapor; switch through a bounded coupled boiling solve, not by evaluating
logarithms at Y_surface=1. Nitrogen vapor in ambient air participates in the
mass-transfer driving force. Validate cryogenic film-property and heat-transfer
approximations separately from water.

Solve each interacting gas volume and its droplets together, semi-implicitly
or implicitly. Mass cannot become negative, evaporation cannot consume more
energy than the coupled system supplies, and water evaporation must not
overshoot the allowed equilibrium state. Do not freeze ambient gas temperature
while many parcels independently draw heat from the same cell. This choice is
supported by [McDermott and Floyd's evaporation integration study](https://www.nist.gov/publications/development-and-evaluation-two-new-droplet-evaporation-schemes-fire-dynamics).

Baseline assumes a dilute post-atomization spray with negligible collisions and
secondary breakup. Monitor liquid volume fraction, collision timescale,
Weber number rho_g |u_g-u_p|^2 d/sigma, and droplet thermal Biot number. If these
invalidate spherical, lumped-temperature, nonbreaking droplets, activate a
validated extension. TAB is a candidate for secondary breakup, not a universal
primary-atomization law; see [O'Rourke and Amsden](https://saemobilus.sae.org/papers/tab-method-numerical-calculation-spray-droplet-breakup-872089).

## 8. Conservative parcel–gas exchange

The coupling API returns extensive increments: species mass, gas momentum,
and gas total energy, plus external-force/work and boundary ledgers. It does
not return pre-divided temperature tendencies or accelerations.

For a closed local exchange without external forces, evaluate the parcel's
before/after inventories and assign their negatives to the gas:

    delta M_g = -delta M_liquid
    delta P_g = -delta P_liquid
    delta E_g = -delta E_liquid

Separate gravity/pressure work before forming this exchange. Evaporated mass
carries momentum and enthalpy; latent heat is included once through phase
enthalpies. Drag work/dissipation must reconcile gas and parcel kinetic energy.
For the low-Mach enthalpy solver, map the total-energy ledger to thermal
enthalpy with its pressure-work convention. Do not claim exact energy
conservation from equal-and-opposite temperature changes.

Use normalized, mesh-aware, compact deposition kernels and compatible sampling;
sum cell-volume-weighted deposited sources to the exchanged increments. Deposit
along substep paths when travel exceeds the kernel width. Kernel width is a
numerical coupling scale, not an adjustable physical evaporation length.
Test both grid and kernel sensitivity. Published work shows that
[interpolation kernels and particle self-disturbance affect two-way coupling](https://arxiv.org/abs/2303.17756).

Outside the near-field core, sample resolved gas plus a consistent SGS velocity
model. Use a well-mixed Langevin fluid-seen formulation with local unresolved
variance and correlation time; use local test-filter estimates with a stated
unresolved-energy model rather than inventing independent random kicks.
Its coefficients require homogeneous and inhomogeneous dispersion benchmarks.
[LES-driven Lagrangian dispersion](https://journals.ametsoc.org/view/journals/atsc/61/23/jas-3302.1.xml)
provides the tracer-limit reference, not a complete inertial spray closure.
Record resolved/SGS energy exchanges separately and avoid adding all turbulent
variance on top of already resolved fluctuations. Begin validation with this
model disabled, then enable it for coarse turbulent application runs.

The near-field model handles the strongest unresolved temperature/humidity
correlations. The dilute resolved-parcel baseline uses filtered thermal fields;
remaining subcell clustering and thermal self-disturbance are model errors to
assess by refinement. Do not describe this as a general dense-spray closure.

## 9. Time integration, JAX, and turbine coupling

Use a symmetric exchange/transport split initially: half coupled microphysics,
full FV transport/forcing, half coupled microphysics, with density/projection
updates consistent with stage mass exchange. Verify the actual coupled temporal
order rather than inheriting an RK3 label. A converged predictor-corrector may
be preferable where coupling is stiff. Subcycle for response, evaporation, and
cell-crossing times; implicit heat exchange alone does not resolve trajectories.

Retain fixed-capacity structure-of-arrays JAX state, masks, segment reductions,
and counter-based reproducible randomness from injector ID, emission ID,
parcel ID, and time/substep identity. Near-field bins have bounded iteration
counts and convergence flags. Failed thermodynamic solves fail the step or
trigger controlled substepping; they do not clip energy silently.

Parcel capacity exhaustion must never discard injected liquid. Budget capacity
before a run, then fail clearly or conservatively merge/split with mass,
momentum, enthalpy, and size-moment error controls. Checkpoint parcel state,
injector phase, random counters, near-field state if present, wall inventories,
and integrated exchange ledgers. Distributed ownership follows cell location;
transfer complete parcels and account for cross-rank deposition without loss.

Attach injectors to a world, nacelle, or rotating-blade frame. For blade-mounted
injection, absolute velocity includes nozzle velocity plus omega cross radius.
Distinguish support reaction, pump work, and fluid injection momentum in turbine
load accounting. A moving injector may violate the quasi-steady core assumption.

Compose spray forcing with existing actuator forcing in the shared simulation
builder. Actuator disks/lines are gas-force representations, not solid collision
surfaces. If blade interception or erosion is an objective, add explicit blade
geometry/collision and film/splash models; do not infer impact from actuator
kernel overlap. Ground/wall hits move liquid to a deposition ledger. If that
liquid later cools/evaporates, advance a wall/pool reservoir instead of deleting
it or returning it instantly as vapor.

## 10. Package responsibilities and configuration

Proposed numerical modules are `spray/state.py`, `spray/injection.py`,
`spray/materials.py`, `spray/droplets.py`, `spray/near_field.py`,
`spray/coupling.py`, and `spray/diagnostics.py`. These are proposed files, not
existing APIs. Configuration and simulation construction stay in their current
package layers; runtime and artifacts follow ADR-0018. Avoid a second flow
solver or a separate water-spray application implementation.

Configuration groups:

- material and property-table version;
- injector mass-flow/state/velocity histories, frame, diameter, orientation;
- discharge closure and explicitly assumed uncertainty;
- atomization-plane position, radius, gas entrainment, cone/profile;
- size distribution expressed as D32, k, and physical bounds;
- near-field entrainment law, coefficients, timescale gate, handoff criterion;
- parcel resolution/capacity and coupling/SGS controls;
- moisture, surface reservoirs, and requested output stations.

Preserve old nitrogen cases through an explicit legacy distribution/source mode
or a documented migration. Never reinterpret `initial_diameter` as D32 without
changing the case and checking its moments.

## 11. Validation and acceptance

All execution follows ADR-0018's compute-node constraint. This document is not
a successful test report. Proposed numerical tolerances below are engineering
acceptance targets, not promises of experimental accuracy.

1. **Inventory tests:** closed gas–droplet boxes, evaporation and condensation,
   drag relaxation, phase endpoints, multiple droplets sharing finite heat,
   and saturated-water equilibrium. In a float64 reference, target relative
   mass/energy ledger residual below 1e-10; set production float32 tolerances
   from reduction-error scaling. Include buoyancy/gravity and boundary work in
   nonclosed budgets.
2. **Single droplets:** terminal velocity, transient heating, water d-squared
   behavior where applicable, humidity dependence, and nitrogen boiling/heat
   transfer against matched measurements. A d-squared law is a limiting check,
   not a universal prediction during transient heating.
3. **Injection:** exact mass-flow integral, momentum/enthalpy, D32/distribution
   moments, deterministic restart, capacity exhaustion, and decomposition
   invariance. Spatial/multiplicity refinement must not change source physics.
4. **Near-field:** non-evaporating round jets, crossflow trajectories, then
   matched nitrogen and water sprays. Check entrained air is withdrawn and that
   source-minus-sink budgets close. Reject invalid quasi-steady cases explicitly.
5. **Handoff:** vary mesh, handoff at 3/4/6 cells, kernel width, 12/24 size bins,
   and parcel population independently. Initially target changes below 5% in
   specified downstream mean temperature-deficit and centroid metrics over
   statistically converged windows. Do not demand invariance inside the core.
6. **Application:** nitrogen still-air cooling/descent, water spray in crossflow,
   then turbine wake. Run long enough for flow-through/transit and adequate
   statistics. Do not use one second by default for a 24 m downstream domain.

Output source inputs and their provenance; gas/liquid mass and energy balances;
temperature and excess-species profiles; plume centroid/width; cold-air volume
flux; liquid survival and size distributions; wall deposition; and every
validity gate. For recirculating sections distinguish signed flux, positive
throughflow, and volume averages. Nitrogen cooling capacity is approximately
mdot [h_N,g(T_ambient)-h_in], but humid-air temperature-deficit flux is not equal
to that capacity when latent heat, walls, storage, or liquid transport matter.

Calibrate droplet sizes against liquid measurements, entrainment against
velocity/concentration/spreading, and thermal state against enthalpy/temperature
data. Do not fit all unknowns to one downstream thermometer. Separate physical
uncertainty from mesh/parcel/time-integration sensitivity.

## 12. Implementation sequence

1. Generalize material/parcel state and introduce extensive exchange ledgers;
   validate isolated water and nitrogen droplets and closed boxes.
2. Introduce mixture enthalpy and conservative low-Mach phase coupling; retain
   old cases as explicitly labeled comparison models.
3. Add D32-based injection and physical wall/deposition ledgers; validate dilute
   resolved water and nitrogen sprays without an unresolved-core claim.
4. Implement the quasi-steady near-field source/sink adapter, parameterize its
   atomization-plane inputs, and validate entrainment and handoff convergence.
5. Add validated SGS dispersion, multiple injectors, and turbine-frame coupling;
   accept only applications meeting near-field validity gates.
6. Implement a transient volume-consistent core if startup, moving sources, or
   unresolved travel time require it. Add breakup/impact/coalescence models only
   for identified operating regimes, with independent validation.

The first production target is statistically steady downstream cooling and
dispersion from stationary injectors in still air or a turbine wake. It is
conditionally closed once the listed physical and empirical inputs are set.
Unknown atomization/entrainment data remain uncertainty, not quantities that
mass flow, bore diameter, and an assumed speed can determine uniquely.
