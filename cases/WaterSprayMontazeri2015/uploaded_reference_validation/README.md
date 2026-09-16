# Validation against waterjet_referece.pdf

Assessment date: 2026-09-16. **The archived simulations do not pass the recorded
10% criterion at every outlet sensor.** The current implementation passes 56
focused physics/configuration tests, which establishes numerical verification,
not experimental agreement. No new coupled simulations or parameter fitting
were performed in this assessment.

## Reference and scope

The supplied [PDF](../../../waterjet_referece.pdf) is Sureshkumar, Kale and Dhar,
*Heat and mass transfer processes between a water spray and ambient air — I.
Experimental data*, Applied Thermal Engineering 28 (2008), 349–360,
DOI 10.1016/j.applthermaleng.2007.09.010. It is the original experiment underlying
the repository's Montazeri cooling case, not an independent new spray experiment.

The final row of Table 2(b), PDF p.6 / printed p.354, was visually checked against
the upload. All nine dry-bulb values agree exactly with `reference_table.json`.
The operating point is parallel flow, hot-dry air, 4 mm nozzle, 3 bar gauge water
pressure, 12.5 L/min water (Table 1), 3 m/s air, inlet air DBT/WBT 39.2/18.7°C,
and inlet water 35.2°C. The tunnel measures 1.9 × 0.585 × 0.585 m.
Only this operating point is evaluated here; the other experimental rows have
not been simulated or validated by this assessment.

[reference.json](reference.json) records the uploaded PDF checksum, provenance,
sensor geometry, DBT and WBT measurements, and experimental uncertainty.
The paper reports ±0.3°C DBT measurement uncertainty, without a confidence level.
Section 2.7 additionally describes dominant, unquantified mist carryover effects
at higher air velocities, especially counterflow. The ±0.3°C figure is therefore
not a complete uncertainty bound for this operating point; the 10% criterion
remains unchanged.
Figure 2 identifies the 3 × 3 sensor positions viewed from the exit. Comparisons
preserve bottom/middle/top and left/centre/right identities.

This is a **cooling-apparatus** experiment. Air sensors are downstream of a drift
eliminator; collected-water temperature is measured before the sump. It does not
supply the independent downstream droplet velocity/size profiles required by
the [transport-only objective](../../../doc/water-spray-validation-goal.md).
This report assesses existing results. Following this assessment, the user
explicitly reopened cooling validation and requested an active persistent goal
to understand and fix the discrepancy; see the updated goal document above.

## Acceptance and results

The existing [acceptance record](../acceptance.json) requires every sensor to
satisfy `abs(predicted_C - measured_C) / abs(measured_C) <= 0.10`. This percentage
uses degrees Celsius, as previously recorded; it is neither a Kelvin percentage
nor a percentage of the cooling below inlet temperature. It is an engineering
tolerance, not the ±0.3°C measurement uncertainty. No near-zero temperatures
occur in this dataset. The mean cannot compensate for a failing sensor.

Predictions are arithmetic means of uniformly spaced saved diagnostics over
3–4 s (41 samples for the inertial cases, six for the entrained cases). They
sample the final interior cell-centre plane by interpolation, rather than the
experimental plane downstream of the eliminator. Archived runs are complete at
4 s; current-source coupled reruns were not performed. Current-source unit tests
are reported separately below.

| Model/run | Mean DBT °C | Paired RMSE K | Worst sensor error | Passing sensors |
| --- | ---: | ---: | ---: | ---: |
| Entrained mist, coarse | 30.025 | 7.722 | 36.79% | 1/9 |
| Entrained mist, fine | 30.422 | 7.567 | 33.29% | 0/9 |
| Inertial, energy audit | 33.559 | 2.314 | 13.70% | 8/9 |
| Inertial, half timestep | 33.562 | 2.316 | 13.65% | 8/9 |
| Inertial, refined grid | 33.666 | 2.441 | 15.69% | 7/9 |
| Inertial, smooth walls / latest wall audit | 33.693 | 2.365 | 13.67% | 8/9 |

The measured mean is 31.6°C. The latest archived case predicts 35.693°C at the
middle-centre sensor versus 31.4°C measured: +4.293 K, or 13.67%. All nine sensors
are warmer than measured. Even moving that measured centre value upward by the
reported 0.3°C leaves a 12.60% discrepancy; treating that instrumental band as favourable
does not change the failure. This is a sensitivity check, not a confidence test.

![Spatially paired temperatures](sensors.png)

Full paired values are in [sensors.csv](sensors.csv) and [tables.md](tables.md).
[comparison.json](comparison.json) retains every run, numerical diagnostics,
adjacent-run differences and hashes of the input histories/configurations.
The hashes identify the archived evidence; they do not establish its source-code
revision or make it a fresh simulation of the current working tree.

## Numerical evidence and limitations

- Timestep control at 64 × 24 × 24: halving dt from 0.0005 to 0.00025 s changes
  any sensor by at most 0.0249 K. The existing pair also changes injection batch
  size to retain parcel rate. It is not an isolated parcel-sampling study.
- At fixed dt=0.00025 s, refining to 128 × 48 × 48 changes a sensor by up to
  0.6419 K and increases the worst experimental error to 15.69%. Grid independence
  is not established. These controls use the free-slip energy-audit baseline;
  they are not a refinement sequence for the later smooth-wall case.
- The latest case's mean shifts by 0.0894 K between the early and late halves of
  the averaging window. One second of statistics and these grid controls do not
  establish full statistical, parcel-sampling, source-support or domain convergence.
- Recomputed parcel ledgers close to 6.47e-14 kg maximum cumulative mass residual
  and 1.60e-9 J enthalpy residual. The 3–4 s enthalpy-rate residual is 4.27e-10 W.
  [energy_audit.json](energy_audit.json) contains the full budget. This checks parcel
  exchange bookkeeping, not the complete carrier-flow energy/momentum budget.
- The latest case reaches a deposited liquid volume fraction of 0.0133; the
  refined case reaches 0.0484. Both exceed the existing 0.001 dilute screening
  level, particularly relevant to the idealized wall-liquid treatment.
- The latest escaped-water temperature is 22.413°C versus the paper's collected
  water at 26.1°C. These are different locations/populations because drift-plate
  collection is absent from the simulation, so this is not a valid paired sensor
  acceptance test. About 66.8% of escaped liquid has a wall-contact history.
- The measured WBT values are retained, but no averaged, spatially paired outlet
  WBT/humidity predictions are present in the saved diagnostic histories. Humidity
  validation is not claimed.

The implementation contains an entrained-mist closure and a separate inertial
parcel closure. Only the latter represents droplet slip and thermal lag. The
source Rosin–Rammler fit and idealized injection velocities used in the benchmark
are modeling inputs, not new velocity measurements furnished by this PDF.
The source paper gives coarse photographic size bins (74 µm resolution,
approximately 330 µm D30 and ±22% resolution uncertainty), not a complete measured
joint source/validation transport dataset. D30 must not be equated to the model's
area/mass-equivalent D32. Neither this comparison nor the tests validate nozzle
atomization, compressed-air assistance, freezing, or airborne transport profiles.

## Current implementation tests

**56 passed in 85.85 s**, run on the allocated compute node with the CPU JAX
backend. The selected suites check droplet ODE/analytic limits, exchange
conservation, parcel injection and transport-only operation, moisture physics,
configuration and benchmark construction. Test output and JUnit results are
retained under `outputs/waterjet_reference/`.

```bash
mkdir -p outputs/waterjet_reference/tmp
JAX_PLATFORMS=cpu TMPDIR="$PWD/outputs/waterjet_reference/tmp" \
  PYTHONPATH=src python -m pytest -q -o 'python_files=test_*.py' \
  tests/fv/test_water_droplet.py tests/fv/test_water_parcels.py \
  tests/fv/test_moisture.py tests/test_moisture_config.py \
  tests/test_water_spray_benchmark.py \
  --basetemp=outputs/waterjet_reference/tmp/pytest \
  --junitxml=outputs/waterjet_reference/tests.xml
```

Reproduce the comparisons and plot from the existing completed runs:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu \
  TMPDIR="$PWD/outputs/waterjet_reference/tmp" \
  MPLCONFIGDIR="$PWD/outputs/waterjet_reference/matplotlib" \
  python tools/validate_waterjet_reference.py
```

The script verifies the uploaded PDF checksum and the transcribed DBT values,
then calls the existing comparison and energy-audit routines. It does not tune
the model or automatically pass it when only the mean agrees.
