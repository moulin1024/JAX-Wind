# Numerical controls for corrected coupling

**Stopped by the user, 2026-09-16.** The goal controller is paused. Refined and
long-duration simulations were interrupted; their last complete checkpoints
are 3.0 s / step 12000 and 6.0 s / step 12000, respectively. Neither has a
completed summary or a valid final comparison. Both Slurm steps are confirmed
stopped. The user clarified that unresolved mixing must be modeled on very
coarse grids. See the [handover](../../../../doc/water-spray-handover.md).


All cases use the flux thermal boundary, projected parcel feedback, smooth gas
walls and unchanged spray/turbulence inputs. Every comparison reports all nine
spatially paired sensors. A passing mean is not an acceptance test.

| Case | Mesh | dt s | Parcels/step | Parcels/s | Purpose |
| --- | --- | ---: | ---: | ---: | --- |
| projected_feedback | 64×24×24 | 0.0005 | 16 | 32000 | Corrected baseline |
| projected_half_dt | 64×24×24 | 0.00025 | 8 | 32000 | Time/exchange-step control |
| projected_more_parcels | 64×24×24 | 0.0005 | 32 | 64000 | Sampling control |
| projected_refined | 128×48×48 | 0.00025 | 8 | 32000 | Spatial control versus half_dt |

All durations are 4 s and diagnostics are spaced 0.025 s. Timestep control also
changes the grouping of deterministic injection samples, while maintaining the
physical parcel rate. Spatial refinement changes interpolation of the prescribed
inlet spectrum and CIC coupling support, so it tests the full discretization,
not only the gas operator. Physical source regularization and statistical/domain
effects remain separate completion requirements.

## Completed timestep result

Halving dt changes any sensor by at most **0.006624 K**; the centre changes by
-0.003497 K. The worst error remains **12.9788%**, and 8/9 sensors pass. Thus the
remaining approximately 4.1 K centre error is not explained by this timestep.
The sampled gas sensible residual falls from 3.945 to 2.268 W, and the sampled
water residual magnitude from 3.62e-6 to 2.06e-6 kg/s. These contain boundary-flux
sampling error and should not be described as exact integration ledgers.

Data: [timestep comparison](timestep_comparison.json),
[half-step carrier budget](half_dt_carrier_budget.json).

## Completed parcel-sampling result

Doubling the parcel rate to 64000/s changes any sensor by at most 0.086472 K,
with the centre decreasing by that amount. The worst error remains 12.7145%,
with 8/9 sensors passing. This reduces the likelihood that parcel sampling alone
explains the roughly 4 K centre discrepancy, but is only one refinement level.
The sampled sensible residual is 2.529 W and water residual -2.881e-6 kg/s.
Parcel cumulative mass/enthalpy residuals remain 6.14e-14 kg and 2.19e-9 J.

[Sampling comparison](sampling_comparison.json),
[carrier budget](sampling_carrier_budget.json),
[parcel budget](sampling_parcel_budget.json).

## Statistics and interrupted controls

The 2–3 s and 3–4 s time blocks differ by up to 0.330 K in the corrected baseline
and 0.322 K in the doubled-parcel case. These are temporal-stability diagnostics,
not confidence intervals or sufficient independent samples. See
[baseline/timestep blocks](temporal_blocks.json) and
[sampling blocks](sampling_statistics.json).

The 128×48×48 spatial control ran in exec session 96094, Slurm allocation
30266535; it is now interrupted as recorded above. Centred-advection session 63037 completed; its centre error remains
12.5307% ([report](../central_advection/README.md)).

A 12 s statistical control, `inertial_projected_long.toml`, ran in exec session
8434 and is now interrupted. Its inflow box expands from 1024 to 3072 axial points and 12.6 to 37.8 m,
preserving spectral spacing, physical turbulence intensity and length scale.
The 12.6 s box period exceeds the 12 s run, avoiding repeated forcing. This
changes the Fourier realization and low-frequency discretization, so it is not
an exact continuation of the four-second case. Compare 4–8 s and 8–12 s blocks,
with all nine sensors, before claiming stable time averages. Independent
realization uncertainty remains distinct. Source hashes are recorded in
`outputs/waterjet_reference/long_source_manifest.json`.

Do not restart automatically. A future user may choose to resume from the
preserved checkpoints; first revalidate allocation and source compatibility.

## Additional diagnostics, not new physical closures

At t=4 s, the largest CIC liquid volume fractions in both the baseline and
doubled-parcel snapshots are near downstream wall corners, not the nozzle.
Baseline maximum is 0.00796, with 15.56% of liquid mass assigned to cells above
0.001; doubled sampling gives 0.00714 and 13.11%. About 24% of the instantaneous
liquid inventory touches a side wall. These cells use the existing idealized
wet-wall droplet rule, not a resolved film. The first 0.1 m has maximum
volume fraction 0.00397 (baseline) and 0.00390 (doubled sampling), so the
near-nozzle dilute approximation also needs scrutiny. See [loading audit](loading_audit.json).

The [scalar-diffusion audit](scalar_diffusion.json) isolates an exact spatial
operator identity: first-order upwind adds face diffusion |u| dx/2 relative to
centred advection on this uniform grid. At t=4 s, its interior scalar-variance
loss is 8.92 times AMD diffusion for temperature and 12.49 times for vapor.
These are instantaneous, gradient-weighted operator sums, not a prediction of
the sign or magnitude of outlet bias. They exclude boundary flux, source,
phase-change and time-integration effects. They establish that scalar grid
resolution must be checked before interpreting the carrier as a resolved LES.

The scalar face-sum calculation matches the actual upwind-minus-centred
operators to roundoff ([identity check](scalar_diffusion_identity.json)).

## Conditional experimental heat balance

[Experimental balance](experimental_balance.json) uses uniform outlet dry-air
flux, equal sensor patches and the solver's enthalpy convention. It implies
0.006637 kg/s evaporation and an enthalpy residual of −859 W. Coherent ±0.3°C
outlet-WBT shifts move that residual from −1994 to +290 W. This is a calibration
sensitivity, not a confidence interval or evidence that the data are internally
inconsistent. Unmeasured outlet velocities, apparatus heat exchange and mist
carryover remain limitations. No physical parameters were changed from it.
