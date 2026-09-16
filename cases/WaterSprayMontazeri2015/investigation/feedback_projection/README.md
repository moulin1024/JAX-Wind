# Projected parcel feedback: completed comparison

Both 4 s cases completed with identical physical inputs and the conservative
thermal boundary. Only the projection of the parcel momentum impulse before
carrier transport changed. The six original boundary regressions and the new
inlet/interior uniform-humidity checks establish the defects independently of
outlet-temperature agreement.

| 3–4 s result | Flux boundary, no impulse projection | With impulse projection |
| --- | ---: | ---: |
| Mean DBT °C | 33.61850 | 33.62152 |
| Centre DBT °C | 35.49809 | 35.47883 |
| Worst DBT error | 13.0513% | 12.9899% |
| Passing DBT sensors | 8/9 | 8/9 |
| Reconstructed centre WBT °C | 25.1451 | 25.1303 |
| Sampled sensible-budget residual W | 4.047 | 3.945 |
| Sampled water-budget residual kg/s | 9.892e-6 | -3.623e-6 |

**The correction is numerically justified but does not explain the remaining
cooling discrepancy.** Centre measurements are 31.4°C DBT and 20.2°C WBT.
Conservation residuals above use saved boundary-flux quadrature, not exact
RK-stage ledgers. The corrected parcel ledger residuals remain 6.60e-14 kg and
2.06e-9 J. The simulation observation plane still precedes the drift eliminator.

[Comparison](comparison.json), [carrier budgets](carrier_budgets.json),
[parcel budget](parcel_budget.json).

## Flow mechanism

[Carrier-field plot](carrier_fields.svg) and [vertical profiles](vertical_profiles.svg)
show instantaneous t=4 s fields, not time averages. The failing centre sensor
coincides with a narrow fast, humid gas core. Four cells around the outlet centre
carry axial speeds about 7.5–7.8 m/s, with vapor mixing ratios around
0.016 kg/kg. The nine-temperature average alone conceals this spatial structure.

For context, Montazeri Figure 9 shows substantially broader downstream carrier
profiles. Those curves are another CFD model's results, not independent measured
velocity targets, and must not be used to tune a new closure. Their source is
[the author manuscript](https://pure.tue.nl/ws/portalfiles/portal/32337686/15_bae_si_cpc_montazeri.pdf),
PDF p.19 / printed p.18. The original cooling paper supplies no droplet-laden gas
velocity profiles. We therefore treat deficient mixing as a hypothesis to test,
not an established experimental diagnosis.

The [instantaneous momentum-tendency audit](momentum_tendencies.json) gives
limited-minus-centred advection power -0.637 W, AMD SGS power -0.137 W, molecular
power -0.00842 W and smooth-wall power -0.249 W. Powers use MAC dual volumes;
advection includes open-boundary transport, and the operator difference includes
boundary-discretization effects. This is not a complete turbulent-energy budget.
It nevertheless motivates isolating numerical damping before altering physical
source or turbulence constants. `inertial_projected_central.toml` changes only
the momentum advection operator to the existing centred implementation. Its
outlet agreement alone will not justify selecting it.

Reproduce the field and tendency diagnostics:

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/outputs/waterjet_reference/matplotlib" \
  python tools/plot_water_spray_fields.py \
  outputs/water_spray_montazeri2015/inertial_projected_feedback \
  --output cases/WaterSprayMontazeri2015/investigation/feedback_projection
PYTHONPATH=src JAX_PLATFORMS=cpu python tools/audit_water_spray_mixing.py \
  outputs/water_spray_montazeri2015/inertial_projected_feedback \
  --output cases/WaterSprayMontazeri2015/investigation/feedback_projection/momentum_tendencies.json
```

## Experimental uncertainty clarification

The original paper's Section 2.7 (PDF p.10 / printed p.358) distinguishes the
reported ±0.3°C DBT uncertainty from an additional dominant mist-carryover
uncertainty at higher air velocities, especially counterflow. No magnitude for
that additional effect is supplied. The reference metadata now retain this
qualification. It does not relax the every-sensor 10% criterion or justify
subtracting a fitted sensor-wetting correction.
