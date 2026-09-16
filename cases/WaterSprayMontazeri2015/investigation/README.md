# Discrepancy investigation

**Stopped at the user's request, 2026-09-16.** The goal is paused and the two
investigation jobs are stopped. The intended solution must model unresolved
mixing on very coarse grids. See the [handover](../../../doc/water-spray-handover.md)
for current results and preserved checkpoints. Earlier status notes below are
historical; do not automatically restart their jobs.


> **Cooling investigation reopened by user instruction (2026-09-16).** The active [persistent goal](../../../doc/water-spray-validation-goal.md) is to understand and fix the discrepancy against the uploaded Sureshkumar PDF. Historical results below remain archived; the new controlled investigation starts with conservative carrier temperature boundaries. Atomization and compressed-air injection are outside this cooling goal.

The [first coupled results and numerical refinements](results.md) reduce the fine-grid sorted error by about 68%, but the discrepancy remains open.

The original entrained-mist result is retained in `../comparison/`. This investigation is ongoing; the reference target and physical nozzle parameters are unchanged.

## Mechanisms established before fitting any outlet data

The initial liquid speed inferred using the reference nozzle coefficient is 22.08 m/s, compared with 3 m/s air. A spatially concentrated source that immediately discards this momentum and sets water temperature equal to air temperature changes both residence time and the energy budget.

`python tools/diagnose_water_spray.py` integrates a 20-point mass-distribution quadrature over 32 azimuths in uniform, frozen inlet air. It uses the prescribed cone, warm water, finite slip/temperature, gravity, and an idealized wet-wall condition. The saved `droplet_flight.json` contains three time steps. This is a no-feedback mechanism diagnostic, not a tunnel cooling prediction or a new experimental validation case.

At dt = 0.00025 s, the diagnostic gives approximately 0.313 s mass-weighted flight time, compared with 0.633 s for entrainment at air speed. Initial Sherwood numbers span about 6.9–14.8 rather than 2. About 70% of injected mass reaches a side wall under the idealized wall rule. About 5.15% evaporates. The gas supplies about 62.1 kJ/kg injected water in sensible heat, whereas vapor receives about 128.7 kJ/kg in latent enthalpy: the balance comes from the initial liquid sensible heat. Immediately assigning all latent cooling to the gas neglects this contribution. These conclusions require two-way confirmation because the diagnostic deliberately holds the air fixed.

## Implemented closure

- `physics.moisture.advance_water_droplet` uses shared saturation thermodynamics and a finite-temperature droplet heat/mass update. Ranz–Marshall heat and mass transfer depend on slip. Stefan's logarithmic driving force depends on surface temperature and local gas vapor.
- Constant liquid heat capacity and warm-air properties are explicit. The enthalpy convention remains compatible with shared moisture: vapor enthalpy is constant L_v, liquid enthalpy is cp_l(T_l−T_freeze). Vapor sensible heat is omitted, so the effective latent heat changes with liquid temperature. This approximation must be revisited in a more complete thermodynamic model.
- The implicit thermal solve prevents negative liquid mass and cooling below freezing in its warm-input domain. Multi-parcel carrier feedback is subcycled. This is first-order time integration; gas and parcel time-step convergence must be checked. Condensation onto spray, freezing, boiling, breakup and collision are not modeled.
- `water_spray.advance_water_droplet_motion` uses the reference Morsi–Alexander drag table with frozen-coefficient analytic velocity and position integration. The gas reaction excludes the full external gravitational impulse; a terminally settling drop transfers its weight to the carrier. Coefficient validity is limited to the tabulated Reynolds-number range.
- `water_parcels` injects equal physical-mass parcels from a truncated mass-based size distribution, samples/deposits with existing cloud-in-cell utilities, and couples momentum, vapor and sensible energy into the shared moist atmospheric integrator. Inertial liquid contributes drag, not an additional entrained-liquid buoyancy force.
- Droplets lose normal velocity at side walls and retain tangential velocity. There is no wall-film thickness, breakup, deposition/resuspension or detailed heat-transfer model. At the outlet, remaining liquid escapes and is counted in a mass ledger.
- The nozzle remains unresolved and is treated as a prescribed post-atomization boundary. It supplies liquid momentum; it does not supply a compressed-air mass or momentum source. The gas retains the initial benchmark's free-slip walls and uniform inlet without fluctuations, keeping those differences visible for subsequent controlled investigation.

Source provenance correction: the 2015 text labels the size histogram as 3 mm / 4 bar, but the original Sureshkumar paper Figure 6 identifies 4 mm / 3 barg. The earlier claim that it came from a different operating point was therefore not supported by the original source. The reported mean-size uncertainty remains approximately 22%.

Transfer-correlation reference: [Fluent droplet vaporization theory](https://www.afs.enea.it/project/neptunius/docs/fluent/html/th/node253.htm). This implementation uses a Stefan correction and the repository's existing simplified enthalpy convention; it is not an exact reproduction of Fluent's thermodynamics. Drag, boundary conditions and nozzle inputs: [Montazeri et al. manuscript, Sections 3.2–3.4 and Table 1](https://pure.tue.nl/ws/portalfiles/portal/32337686/15_bae_si_cpc_montazeri.pdf).

## Coupled runs

```bash
python -m jaxwind run cases/WaterSprayMontazeri2015/inertial_coarse.toml
python -m jaxwind run cases/WaterSprayMontazeri2015/inertial_fine.toml
python tools/compare_water_spray_benchmark.py \
  outputs/water_spray_montazeri2015/coarse \
  outputs/water_spray_montazeri2015/fine \
  outputs/water_spray_montazeri2015/inertial_coarse \
  outputs/water_spray_montazeri2015/inertial_fine \
  --output cases/WaterSprayMontazeri2015/investigation/comparison
```

The inertial configuration ignores the inherited Gaussian width/offset and single-diameter fields: it instead uses the reference nozzle position, radius, temperature, speed, cone and distribution. Those inherited values remain solely for compatibility with the baseline configuration. Checkpoints store the full parcel buffer and mass ledger. Capacity exhaustion aborts the run rather than silently dropping injected water.

Acceptance still requires adequate mean **and** distribution agreement, demonstrated numerical convergence, and valid physical assumptions. The comparison reports 1 K screening thresholds for mean bias and sorted-distribution RMSE; these are engineering screens, not experimental uncertainty or sufficient validation by themselves. A local parcel volume fraction above 0.001 is flagged for dilute-spray review, separately from the original entrained model's liquid mass-loading warning. The nozzle handoff region needs particular scrutiny.

## Verification and remaining work

Shared moisture and isolated droplet tests: 21 passes. Parcel injection, overflow, carrier water/enthalpy exchange and escape accounting: 3 passes. The isolated update converges at first order to an independently integrated DOP853 ODE solution. Stokes relaxation and terminal-settling reaction match analytic limits.

Four coupled runs and controlled time-step/parcel-count/mesh comparisons are complete. Remaining work includes resolving inlet turbulence, gas walls and wall-liquid behavior, establishing distribution-level mesh convergence, checking source-resolution dependence and near-nozzle validity, and adding/validating a compressed-air source for the intended snowmaking-style application. Do not mark the persistent goal complete on the basis of isolated droplet tests or agreement in a single mean temperature.

## Updated reference, inlet and measurement apparatus (2026-09-16)

The original Table 2(b), p354, provides the nine spatially identified dry-bulb
measurements directly. `../reference_table.json` preserves these values and the
reported ±0.3 C temperature uncertainty (confidence level unspecified). Their
mean is 31.60 C. Current comparisons use this table; previous figure-derived
comparisons remain archived.

| Fine-grid source/inlet | Sensor mean (C) | Spatially paired RMSE (K) | Maximum sensor error (K) |
| --- | ---: | ---: | ---: |
| Thin cone, uniform inlet | 33.689 | 2.500 | 5.117 |
| Thin cone, prescribed total inlet turbulence energy | 33.559 | 2.314 | 4.301 |
| Finite annulus, same turbulence | 33.449 | 2.346 | 4.047 |

See `turbulence/` and `source_width/` for reproducible results and spatial plots.
The total-energy inlet normalization enforces the paper's k=(U I)^2=0.09 m2/s2;
it is independent of measured outlet temperatures. The annulus uses the reported
inner/outer radius ratio range midpoint 0.475, uniform mass per annular area,
and the unchanged mean 18-degree half angle. The angular mass distribution is
an assumption. Its lower mean error does **not** improve paired RMSE.

The original apparatus locates thermocouples after wet drift plates; the companion
model includes plate heat/mass transfer. These measurements therefore require an
apparatus model, not just a last-cell spray-tunnel temperature. See the
[original experimental and model papers](https://pldhar.wordpress.com/wp-content/uploads/2010/09/spray-model-air-water-syst-suresh-kumar.pdf),
printed pp350–351 and 367.

### Conservative wet-plate sensitivity

`tools/diagnose_wet_drift.py` uses the legacy rig's equivalent straight channel:
0.195 m length, 0.64 m height, 0.036 m spacing; Gnielinski heat transfer and
Gilliland mass transfer. These are documented in
[Kachhwaha et al. (1998), pp453–454](https://pldhar.wordpress.com/wp-content/uploads/2010/09/evap-cool-water-spary-kacchwaha-1.pdf).
Their applicability to the later rig dimensions is unconfirmed. Complete
wetting, fully mixed inlet gas and co-current redistributed liquid are explicit
sensitivity assumptions. Real collection routing may differ substantially.

The diagnostic draws gas temperature/humidity/flux from completed checkpoints
and liquid flux/temperature from an outlet slab estimator. It uses shared
saturation thermodynamics and conserves water plus the model's simplified total
enthalpy. It never uses experimental outlet temperatures as inputs. Correlations
use channel bulk speed; applying them to isolated slow turbulent cells would
violate their Reynolds-number validity. The bulk Reynolds number is about 12108.

| Completed checkpoint | Gas flux mean before/after plates (C) | Liquid temperature before/after (C) | Extra sensible cooling (W) |
| --- | ---: | ---: | ---: |
| Uniform inlet | 33.091 / 32.365 | 22.313 / 22.155 | 842.5 |
| Prescribed inlet energy | 33.060 / 32.339 | 22.357 / 22.187 | 837.1 |
| Finite annulus | 33.002 / 32.280 | 22.286 / 22.116 | 838.2 |

These are instantaneous **flux means**, not time-averaged nine-sensor means;
they must not be substituted into the spatial validation scores. The outlet
liquid slab estimator also needs a time-integrated escape/collection ledger.
`wet_drift.json` records all three 256/1024/4096-step integrations. Last refinement
changes gas temperatures by less than 0.000015 K. Maximum water and enthalpy
budget residuals are below 1e-15 kg/kg and 2e-9 J/kg. Two dedicated tests cover
conservation, saturated equilibrium, and convergence to an analytic two-stream
sensible-exchange solution.

This diagnostic supports an approximately 0.72 K apparatus contribution under
its assumptions, **not closure of the validation gap**. It predicts further
liquid cooling, not warming toward the measured collected-water temperature
26.1 C. The prior hypothesis that plate exchange would also resolve the water
temperature discrepancy is not supported. Investigation must now track the
liquid energy and collection budgets, wall residence/film behavior, and
measurement routing; a calibrated temperature correction would conceal these
remaining discrepancies. Mesh convergence and air-assisted atomization also
remain unproved.

Reproduce inside the repository (no `/tmp` files):

```bash
export TMPDIR="$PWD/outputs/water_spray_montazeri2015/work"
export PYTHONDONTWRITEBYTECODE=1
python tools/diagnose_wet_drift.py \
  outputs/water_spray_montazeri2015/inertial_fine \
  outputs/water_spray_montazeri2015/inertial_turbulent_energy \
  outputs/water_spray_montazeri2015/inertial_annular \
  --output cases/WaterSprayMontazeri2015/investigation/wet_drift.json
python -m pytest tests/test_wet_drift_diagnostic.py \
  --basetemp=outputs/water_spray_montazeri2015/work/pytest_wet_drift
```

### Event-integrated liquid energy accounting

The parcel state now records injected liquid enthalpy, escaped liquid enthalpy,
and cumulative gas sensible energy transferred to droplets. The independent
identity is

```
H_liquid_inventory + H_liquid_escape + Lv * M_evaporated
    = H_liquid_injected + Q_gas_to_droplets
```

The last-second difference in escaped enthalpy divided by escaped mass and liquid
heat capacity gives a mass-flow-weighted exit temperature without an outlet-slab
estimator. The audit uses actual events, not the remaining parcel inventory.
`inertial_energy_audit.toml` repeats the prescribed-energy fine-grid case with
these diagnostics. No source or heat-transfer parameters change.
`tools/audit_water_spray_energy.py` reports the completed-run budgets. The parcel
unit tests now verify enthalpy through escape and exact checkpoint round trips.
Older checkpoints lack these cumulative ledgers and cannot be exactly resumed
with the expanded state; they remain readable for archived diagnostics. The new
run starts from its prescribed initial state, rather than inventing prior energy
history.

The Montazeri simulation itself omits the drift plates and uses outlet escape,
while retaining tangential droplet velocity at walls (Section 3.2). Thus plate
exchange alone cannot explain differences between our solver and that CFD
result. The paper also states that droplet temperatures require further
validation (Section 5); agreement with its air temperatures would not establish
correct collected-water thermodynamics. See
[Montazeri et al.](https://pure.tue.nl/ws/portalfiles/portal/32337686/15_bae_si_cpc_montazeri.pdf).

The completed 3–4 s event audit (`energy_audit.json`) gives 0.199979 kg/s
escaped liquid at **22.368 C** and 0.00763994 kg/s evaporated water. Injection
supplies 30.576 kW liquid enthalpy; exit liquid carries 18.707 kW, evaporation
carries 19.100 kW, gas sensible heat supplies 7.240 kW, and liquid storage changes
by 0.009915 kW. Their residual is 1.64e-11 W; maximum cumulative enthalpy residual
is 1.73e-9 J and mass residual is 6.46e-14 kg. The observed approximately 3.7 K
water-temperature discrepancy therefore is not an outlet-slab-estimation or
parcel enthalpy-bookkeeping artifact. It remains a physical-model or measurement
routing discrepancy. This audit does not independently prove the carrier's
transport energy budget, nor validate the simplified thermodynamics.

All 160 diagnostic times match the prior prescribed-energy case. Maximum change
in sensor mean is 8.5e-12 K; escaped and evaporated masses differ by less than
4e-15 kg. Seven focused parcel/checkpoint/wet-plate tests passed. Next physical
checks should distinguish continued droplet evaporation at wet walls from actual
film collection and establish the gas-wall/mixing and spatial-resolution effects.
A collected-water temperature after the apparatus remains distinct from our
pre-apparatus exit observable.

```bash
python -m jaxwind run cases/WaterSprayMontazeri2015/inertial_energy_audit.toml
python tools/audit_water_spray_energy.py \
  outputs/water_spray_montazeri2015/inertial_energy_audit \
  --output cases/WaterSprayMontazeri2015/investigation/energy_audit.json
```

### Next spatial-refinement experiment

`inertial_refined.toml` doubles every spatial count from 64x24x24 to
128x48x48. It retains the thin-cone physical source, normalized turbulent inlet,
32000 parcels/s, and four parcel substeps. The timestep is halved to 0.00025 s
for CFL control, with eight parcels per step and extra buffer capacity. The
4 s duration and 0.025 s diagnostic interval remain unchanged. Its results must
be compared with `inertial_energy_half_dt.toml`, which changes the timestep and
injection batch size on the existing fine mesh, to separate temporal and spatial
effects. A finite source regularization/convergence assessment remains necessary
because CIC support shrinks with the mesh while the nozzle remains unresolved.
Neither configuration changes source parameters to improve agreement.

The attempted gas-wall audit found that `Boundaries.spanwise` supports only
periodic, free-slip and open boundaries; no-slip currently applies only to the
z-walls. Therefore switching two walls would not reproduce the reference's four
no-slip walls with wall functions. The gas-wall model difference remains open;
no two-wall experiment is being described as a corrected tunnel model.

### Smooth-wall stress option

Further inspection found an existing side-wall *stress* pathway even though
no-slip y ghost boundaries are unavailable. Its rough-wall Monin–Obukhov law is
not appropriate to the reported hydraulically smooth tunnel. The new
`jaxwind.smooth_wall` implements Spalding's continuous smooth-wall law with
standard kappa=0.41 and E=9.8, without selecting a roughness from cooling data.
The relation and constants follow the
[OpenFOAM Spalding wall-function documentation](https://doc.openfoam.com/2312/tools/processing/boundary-conditions/rtm/derived/wall/nutUSpaldingWallFunction/)
and [SpaldingsLaw defaults](https://api.openfoam.com/2212/classFoam_1_1tabulatedWallFunctions_1_1SpaldingsLaw.html).

`gas_wall_model="smooth-spalding"` applies opposing tangential stress on all four
y/z walls through the existing momentum forcing interface, while resolved
free-slip fluxes prevent double-counting wall stress. Heat and vapor walls
remain adiabatic/impermeable. The law samples the first cell centre and assumes
local equilibrium. It does not reproduce Fluent's k-based RANS wall functions,
resolve wall films, or guarantee accuracy in corners or non-equilibrium spray
flow. `inertial_smooth_walls.toml` is a controlled physical-model comparison,
not a declaration that the reference boundary conditions have been reproduced.

Two tests verify inversion against prescribed Spalding profiles, the viscous
limit and rest state, integrated drag from all four wall areas, reversal of the
stress with flow, and no force away from boundary cells. These tests establish
implementation consistency, not tunnel validation. The existing baseline
remains free-slip unless the smooth-wall option is explicitly selected.

A CPU smoke check through the coupled benchmark builder advanced two steps on
an 8x4x4 grid with smooth-wall stress: finite velocity, CFL 0.0063163, and parcel
enthalpy residual -3.1e-19 J. This exercises the forcing integration; the full
smooth-wall comparison still needs its completed GPU run. The independent
128x48x48 refinement run is retained and monitored rather than restarted.

The smooth-wall test suite now also checks staggered-grid kinetic-energy
removal for a random velocity field containing local reversals, using half
control volumes at physical boundary faces. All three smooth-wall tests pass.
The comparison tool reports signed changes at every sensor, RMS change and
maximum change between adjacent runs. These are labelled as run differences,
not automatically as mesh refinements, so changes in wall/source physics cannot
be misreported as numerical convergence. A no-physics-change run pair reproduces
sensor values within 7e-12 K with this comparison.

### Completed 128x48x48 result (timestep control pending)

The refined run completed 4 s. Relative to the 64x24x24 energy-audit case,
the last-second sensor mean changes from 33.559 to 33.666 C, paired RMSE from
2.314 to 2.441 K, and maximum sensor error from 4.301 to 4.927 K. Sensor-change
RMS is 0.253 K; the middle-centre sensor changes by +0.626 K. Cooling power
changes from 7.115 to 6.822 kW. These figures combine spatial and timestep
changes until the matching fine-grid timestep control is complete. They do not
establish convergence. See `spatial_refinement/`.

`refined_energy_audit.json` gives event-integrated outlet water temperature
22.479 C, evaporation 0.0074511 kg/s and a 3–4 s energy-rate residual of
-2.82e-8 W. Energy bookkeeping remains consistent, while neither air nor water
temperature agreement improves enough to close the gap.

Liquid loading also fails to approach a benign dilute limit: the run's maximum
CIC liquid volume fraction is 0.0484. At the final state it is 0.0274 in the
outlet-bottom-side corner (x=1.8926, y=0.00609, z=0.00609 m), not at the nozzle.
About 58.3% of liquid mass lies in cells above volume fraction 0.001, occupying
2.80% of the domain volume. About 21.5% of liquid inventory is wall-contacting,
and wall-contact particles carry 48.8% of the instantaneous outlet-slab liquid
flux. Those wall particles supply 0.498 kW of the instantaneous 6.892 kW gas
sensible heat loss. See `refined_state_audit.json`.

This supports prioritizing a distinct wall-liquid/collection treatment. Keeping
wall-striking droplets as separate evaporating spheres and depositing their
mass into shrinking gas cells does not represent a resolved wall film. The
reported concentration remains a diagnostic of the current model, not a claim
that real suspended spray fills 4.8% of the tunnel volume. It also does not
justify tuning source width or diameter to the air-temperature measurements.

### User tolerance update

The user accepts 10% difference. Current comparison outputs therefore report
relative errors based on measured outlet temperature in degrees Celsius,
separately for the nine-sensor mean and every spatially paired sensor. This
normalization is explicit; it is not a percentage of absolute Kelvin temperature
or of cooling below inlet temperature. The user confirmed that **every sensor** must pass; mean-only agreement
is insufficient. The older 1 K engineering screen is
retained for historical comparison and is not the newly requested tolerance.
The refined mean passes 10% (6.54%), while the worst individual sensor does not
(15.69%). No source/physics parameters were adjusted for this criterion change.

### Completed timestep and smooth-wall controls

The 64x24x24 timestep control completed: halving dt changes the sensor mean by
+0.00261 K and any sensor by at most 0.02486 K. With dt held at 0.00025 s,
refining to 128x48x48 changes the mean by +0.10384 K, sensor RMS by 0.25768 K,
and the middle-centre sensor by +0.64185 K. Thus the observed spatial sensitivity
is not explained by the timestep/injection-batch change. `spatial_refinement/`
now includes all three completed controls.

The smooth-wall run also completed. Mean outlet temperature is 33.6933 C; eight
of nine sensors meet the user-approved 10% criterion, but the middle-centre
sensor remains 13.6729% high. Its escaped-water temperature is 22.4134 C, gas-to-
droplet sensible transfer 7.1623 kW, and maximum cumulative enthalpy residual
1.66e-9 J. See `smooth_walls/` and `smooth_walls_energy.json`. Smooth-wall stress
alone does not close the gap. Neither completed control changes the source
geometry, droplet sizes or flow rate.

A renewed check of Montazeri Sections 3.3–3.5 does not establish a specific
stochastic droplet-dispersion closure or its constants. The source confirms
hollow-cone injection and coupled trajectories, but this is insufficient to
justify adding or tuning a random-walk coefficient to cool the centre sensor.
No such coefficient has been introduced. The strongest diagnosed physical
concerns remain wall-liquid representation, unresolved source/flow mixing,
and the downstream measurement apparatus. Current results are not accepted
under the user-confirmed every-sensor criterion.

### First-impact and wall-history audit

`inertial_wall_audit.toml` repeats the smooth-wall case with passive per-parcel
wall-history tags and cumulative first-impact, tagged escape and post-impact
transfer ledgers. Tags reset on slot reuse; repeated wall contact is counted
only once. Transfer during the first collision substep belongs to free flight,
and the updated liquid mass/enthalpy enters the wall-history population at the
end of that substep. Subsequent transfer belongs to that population even if a
parcel moves away from the wall. This is an accounting convention for the
existing trajectory scheme, not a new collision or film model.

Six parcel tests pass, including independent tagged-population mass/enthalpy
closure and tag reset. The 4 s rerun changes the sensor mean by less than
8.1e-12 K from the previous smooth-wall run. `first_wall_audit.json` reports:

| Quantity, 3–4 s event budget | Result |
| --- | ---: |
| First-impact liquid flow | 0.134461 kg/s |
| First-impact mass-flow-weighted temperature | 24.363 C |
| Escaping liquid with wall history | 0.133724 kg/s at 22.911 C |
| Escaping liquid without wall history | 0.066505 kg/s at 21.414 C |
| Gas sensible heat transferred after first impact | 850.5 W |
| Evaporation after first impact | 0.00069335 kg/s |
| Maximum wall-population water residual | 8.0e-15 kg |
| Maximum wall-population enthalpy residual | 3.4e-10 J |

About 66.8% of exiting liquid has touched a wall, but post-impact transfer
accounts for only 11.9% of total gas-to-droplet sensible heat. Wall liquid is
already below the reported collected-water temperature before its subsequent
wall residence. These results do not support post-impact evaporation as the
sole explanation of the collected-water temperature gap. First-impact and
escape averages are population fluxes in a time window, not matched-cohort
cooling histories or direct measurements of a real film.

The checkpoint has zero cloud liquid and ice. `wall_profile_audit.json` adds
instantaneous size-resolved liquid temperatures/radii at x/L=0.25, 0.5 and 0.75.
At mid-tunnel, the largest current droplets (500–520 micrometres) are roughly
29–29.5 C, versus the paper's approximate 4 K reduction from 35.2 C for its
largest droplets (Section 4.2/Figure 11). Sample counts and finite axial windows
are explicit. This is an internal CFD-to-CFD comparison, **not experimental
validation of droplet temperature**; Montazeri explicitly leaves that validation
open. The discrepancy motivates examining pre-impact transport/thermal exposure
and source-to-carrier coupling, without fitting transfer coefficients.

The new state adds wall-history ledgers; prior checkpoints remain available for
read-only analyses but lack the history needed for exact resume with this state.
All new run files, renders, and test temporary directories remain in the workspace.
