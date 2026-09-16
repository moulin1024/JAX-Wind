# Water-spray investigation handover

Date: 2026-09-16. **Stopped at the user's request; discrepancy unresolved.**
The persistent goal controller reports `paused`. The two investigation jobs
were interrupted and verified stopped. No goal was marked achieved.

## Start here: the user's latest direction

The implementation is intended to work on **very coarse grids, where spray
mixing cannot be resolved**. It must model that mixing. Do not restart this as
an effort to resolve the spray jet through increasingly fine LES grids.

The next chat should focus on a physically justified coarse-grid closure for
unresolved spray/carrier mixing and entrainment, with consistent parcel
dispersion and conservative exchange. Numerical refinement can diagnose
discretization effects and closure robustness; it is not the intended solution.
Keep physical source/filter scales distinct from mesh spacing. Do not simply
increase numerical diffusion, turbulent viscosity, or source width until outlet
temperatures pass. No closure of this kind was implemented during this session.

The user requested this handover and will restart in a new chat. Do not
automatically resume the old runs or launch further experiments from this note.

## Objective and reference

Validate the water-spray implementation against
[waterjet_referece.pdf](../waterjet_referece.pdf), identify the discrepancy and
implement justified corrections without fitting source/closure parameters to
the held-out outlet temperatures.

The uploaded paper is Sureshkumar, Kale and Dhar (2008), *Heat and mass transfer
processes between a water spray and ambient air — I. Experimental data*,
DOI `10.1016/j.applthermaleng.2007.09.010`. It is the original experiment behind
the existing Montazeri benchmark, not a new independent experiment.

Verified operating point: Table 2(b), final row, PDF p.6 / printed p.354:

- Parallel flow, hot-dry air; tunnel 1.9 × 0.585 × 0.585 m.
- Water nozzle 4 mm, 3 bar gauge, 12.5 L/min, inlet water 35.2°C.
- Air 3 m/s, inlet DBT 39.2°C, WBT 18.7°C.
- Nine outlet DBTs: `[30.7, 32.2, 32.2, 31.9, 31.4, 31.7, 31.0, 31.6, 31.7]` °C.
- Paired WBTs: `[21.4, 20.8, 19.9, 20.5, 20.2, 20.2, 21.0, 20.8, 20.4]` °C.
- Sensor order: bottom/middle/top rows; left/centre/right viewed from the exit.
  Both transverse coordinates are 0.0975, 0.2925 and 0.4875 m.
- Collected water is reported at 26.1°C.

The recorded acceptance criterion is **every spatially paired DBT sensor**:
`abs(predicted_C - measured_C) / abs(measured_C) <= 0.10`.
This is the previously agreed Celsius engineering criterion, not a Kelvin
percentage, cooling-efficiency error or instrumental uncertainty. Mean-only
agreement cannot pass. Conservation and appropriate numerical/statistical
evidence are also required; current results do not meet the full objective.

The reported DBT uncertainty is ±0.3°C, confidence unspecified. Section 2.7 adds
an unquantified mist-carryover uncertainty at higher air velocities, particularly
counterflow. Do not treat ±0.3°C as the complete bound or relax the 10% criterion.

**Observation mismatch:** air sensors are after wet drift-eliminator plates;
our diagnostic plane is the last interior plane before an omitted eliminator.
Escaping modeled droplets are not the measured collected-water population.
Humidity and liquid temperatures constrain interpretation but are not valid
matched validation observables until this difference is addressed.

## Completed evidence

All values below are completed 4 s runs, averaged over 3–4 s. Every listed
inertial run passes 8/9 DBT sensors; the centre is the failing sensor.

| Case | Centre DBT °C | Worst paired error | Meaning |
| --- | ---: | ---: | --- |
| Archived smooth-wall / wall audit | 35.69329 | 13.6729% | Original reopened baseline |
| Conservative scalar boundary | 35.49809 | 13.0513% | First verified correction |
| Projected feedback | 35.47883 | 12.9899% | Corrected working baseline |
| Half timestep | 35.47533 | 12.9788% | Maximum sensor change 0.006624 K |
| Double parcel rate | 35.39236 | 12.7145% | Maximum sensor change 0.086472 K |
| Centred momentum advection | 35.33465 | 12.5307% | Centre changes only −0.144177 K |

The corrected baseline's nine predictions are
`[32.49769, 33.31908, 32.27758, 33.42790, 35.47883, 33.41686, 33.87639, 34.30411, 33.99525]` °C.
Its sensor mean is 33.62152°C versus 31.6°C measured. Reconstructed centre WBT is
25.1303°C versus 20.2°C measured, subject to the observation mismatch above.

These controls make timestep, parcel sampling and momentum-limiter damping
unlikely explanations for most of the approximately 4 K centre discrepancy.
They do not prove a unique physical cause. No completed result establishes
statistical or spatial convergence. In particular, the 2–3 s and 3–4 s block
averages differ by up to 0.33 K across sensors.

### Verified implementation corrections

1. **Retain scalar sources in physical boundary cells.** The old carrier
   boundary overwrote first/last physical cells after parcel heat deposition.
   `scalar_boundary="flux"` now prescribes incoming advective flux and retains
   those cells. The legacy `"cell"` default remains reproducible. The unchanged
   control reproduces archived temperatures within 5e-11 K. Sampled sensible
   balance residual falls from 106.7 W to 4.05 W.
2. **Project parcel momentum feedback before moisture transport.** Advecting
   with the divergent impulse produced spurious humidity variation in a
   uniform-passive-vapor diagnostic. `project_parcel_feedback=true` enforces the
   prescribed velocity boundary, then projects the impulse before carrier
   transport. Inlet and interior tests remove that error to roundoff. Lagged
   pressure is retained to avoid applying the impulse pressure twice.

These are opt-in corrections in the investigation cases. Neither was selected
by fitting experimental temperatures, and neither resolves the main discrepancy.

Relevant files:

- [open_abl.py](../src/jaxwind/open_abl.py),
  [scalar.py](../src/jaxwind/scalar.py),
  [water_parcels.py](../src/jaxwind/water_parcels.py),
  [benchmark builder](../src/jaxwind/simulation/water_spray_benchmark.py).
- [Boundary tests](../tests/fv/test_open_scalar_flux.py) and
  [feedback-projection tests](../tests/fv/test_water_projection.py).
- The builder also accepts the existing momentum-advection option from
  `[numerics]`; its default remains `"muscl-mc"`. Centred advection is a
  diagnostic control, not a chosen replacement model.

Verification logs under `outputs/waterjet_reference/` include the original
56-test physics/configuration check, 29 boundary/regression checks, 32
projection/regression checks, the final two projection tests and 16 momentum
checks. These suites overlap; do not add the counts as distinct coverage.
No new coupled coarse-grid mixing closure has been tested.

### Conservation and physical interpretation

- Corrected parcel cumulative mass and enthalpy residuals are approximately
  6.60e-14 kg and 2.06e-9 J. The carrier sampled sensible residual is 3.945 W;
  water residual is −3.62e-6 kg/s. Carrier residuals use saved-flux quadrature,
  not exact RK-stage ledgers. A complete carrier momentum budget remains absent.
- The outlet has a narrow fast, hot/humid core. At t=4 s, four central cells have
  axial velocities around 7.5–7.8 m/s. There are no measured droplet-laden gas
  velocity profiles in this reference to validate that structure directly.
- Temperature and moisture use first-order upwind transport. An exact spatial
  operator audit gives interior upwind variance loss 8.92 times AMD diffusion
  for temperature, 12.49 times for vapor in the baseline snapshot. This is not
  a full budget or a quantified temperature bias. Numerical diffusion is not
  an independently justified model of unresolved mixing.
- At t=4 s, the largest deposited liquid volume fraction is 0.00796 near a
  downstream wall corner; 15.56% of liquid mass is in cells above 0.001. About
  24% of instantaneous liquid inventory touches a wall. The first 0.1 m also
  reaches 0.00397. The existing wet-wall rule keeps droplets moving and
  evaporating along walls; it is not a film model.
- About two-thirds of escaped liquid has a wall-contact history. Continued
  exchange after first wall contact supplies about 852 W of the modeled gas
  sensible loss. Escaped liquid is about 22.40°C, which must not be equated to
  the experimental 26.1°C collected-water observation.
- A new conditional experimental balance gives evaporation 0.006637 kg/s and
  enthalpy residual −859 W assuming uniform outlet dry-air flux, equal sensor
  patches and the solver's enthalpy convention. Shifting all outlet WBTs by
  ±0.3°C moves the residual from −1994 to +290 W. This is a sensitivity check,
  not proof of inconsistency or a confidence interval; outlet velocities and
  complete apparatus heat/collection budgets are unmeasured.

## Source and closure assumptions to carry forward

The source is a prescribed post-atomization handoff, not a resolved nozzle.
The current benchmark uses speed `0.9*sqrt(2*pressure/rho_water)`, mean half-angle
18°, and a truncated mass-based Rosin–Rammler distribution: scale 369 µm,
spread 3.67, bounds 74–518 µm. The audit found no number/mass weighting bug.
Its moments are D10=206.774, D30=248.057 and D32=292.980 µm. The coarse photographic
histogram and printed approximately 330 µm D30 should not be silently identified.

Exact angle, angular mass/size/velocity correlations and inlet turbulence were
not measured sufficiently for this case. The fixed 10% inlet turbulence and
spectral length mapping are modeling assumptions. A previous finite-annulus
sensitivity used the photographed inner/outer radius ratio midpoint 0.475;
uniform mass per annular area was still an assumption and did not close the gap.

The model currently has resolved-velocity particle sampling and AMD carrier SGS,
but no explicit unresolved turbulent parcel-dispersion or spray-entrainment
closure. The user's coarse-grid requirement makes this an essential modeling
question, not something to remove by grid refinement alone. Candidate closures
need independent physical justification, conservation checks, stated regime and
filter-scale behavior. Do not select their constants using these nine targets.

An optional question about independent nozzle velocity/angle data and eliminator
dimensions was sent to the user; no answer has been received. Do not treat that
as an approval requirement. The [primary companion model](https://pldhar.wordpress.com/wp-content/uploads/2010/09/spray-model-air-water-syst-suresh-kumar.pdf) includes wet-plate
effects by reference to earlier work, but the inspected pages do not supply an
independently verified geometry for the 2008 rig. Older wet-plate calculations
in this repository remain sensitivity studies, not validated corrections.

## Stopped simulations and reproducibility

Slurm allocation `30266535` belonged to the user's interactive session. **Only
investigation steps `.11` and `.14` were stopped**, via SIGINT. Both exec sessions
(`96094`, `8434`) returned exit 130. The parent allocation and its interactive
step `.0` were left intact; it may no longer exist in a future chat.

| Output case | Saved checkpoint | Intended end | State |
| --- | --- | --- | --- |
| `inertial_projected_refined` | step 12000, 3.0 s | step 16000, 4.0 s | Interrupted; no complete summary |
| `inertial_projected_long` | step 12000, 6.0 s | step 24000, 12.0 s | Interrupted; no complete summary |

Logs had advanced beyond these saved checkpoints; resume only from the saved
state. Both include observer state. Do not feed interrupted results to the
completed-run comparison or call them convergence evidence. Exact status and
fingerprints: `outputs/waterjet_reference/handover_stop_status.json`.

The refined case uses 128×48×48, dt=0.00025 s, with the same physical parcel rate
as the corrected baseline. If later resumed for diagnostics, compare it with
the completed `inertial_projected_half_dt`, not the old free-slip refinement.
The long case uses 64×24×24, dt=0.0005 s and an extended nonrepeating Fourier
inflow: 3072 axial points over 37.8 m, preserving spectral spacing and physical
TKE/length scale. It is a new realization, not an exact extension of the earlier
4 s inlet. It was intended to compare 4–8 s and 8–12 s averages.

Completed outputs are under `outputs/water_spray_montazeri2015/<case>/`;
configurations are under `cases/WaterSprayMontazeri2015/`. Source manifests and
run logs are in `outputs/waterjet_reference/`, including
`central_source_manifest.json` and `long_source_manifest.json`.

Environment used: `/u/limo/venvs/numba_cuda_waterboa/bin/python`, `PYTHONPATH=src`.
All scratch/cache/output paths were kept inside the workspace. Normal sandbox
execution failed with a bubblewrap namespace `ENOSPC`; escalated commands were
approved and used as a workaround. Reassess that environment in the new chat.
The worktree contains many pre-existing modified and untracked files. Do not
reset or clean it. No commit was made.

If a future user explicitly chooses to resume a diagnostic, the runtime entry
point is `python -m jaxwind resume outputs/water_spray_montazeri2015/<case>` with
the original resolved case and an available compute allocation. Revalidate
source compatibility and job state first; old session/job identifiers are not
authorization to assume a live allocation or launch a duplicate.

## Evidence map

- [Persistent scope/history](water-spray-validation-goal.md): contains older
  transport-only instructions and historical blocking notes, all superseded by
  the latest user direction and stop notice at its top.
- [Uploaded-reference validation](../cases/WaterSprayMontazeri2015/uploaded_reference_validation/README.md):
  PDF checksum, direct transcription, archived comparisons and initial tests.
- [Boundary correction](../cases/WaterSprayMontazeri2015/investigation/carrier_boundary/README.md).
- [Feedback correction and field plots](../cases/WaterSprayMontazeri2015/investigation/feedback_projection/README.md).
- [Numerical controls](../cases/WaterSprayMontazeri2015/investigation/projected_controls/README.md):
  completed timestep/sampling JSON, temporal blocks, loading, scalar diffusion
  and `experimental_balance.json`.
- [Centred-advection diagnostic](../cases/WaterSprayMontazeri2015/investigation/central_advection/README.md).
- [Source weighting audit](../cases/WaterSprayMontazeri2015/investigation/source_weighting_audit.md).
- Useful tools: `compare_water_spray_benchmark.py`,
  `audit_water_spray_carrier.py`, `audit_water_spray_energy.py`,
  `audit_water_spray_statistics.py`, `audit_water_spray_state.py`,
  `audit_water_spray_scalar_diffusion.py`, `audit_waterjet_reference_balance.py`
  and `plot_water_spray_fields.py` under `tools/`.

Suggested next-chat instruction: **Read this handover, then design and validate a
conservative coarse-grid model for unresolved water-spray mixing. Preserve the
verified coupling fixes and experimental acceptance criterion; do not rely on
resolving the mixing through finer grids or calibrating against the nine outlet
temperatures.**
