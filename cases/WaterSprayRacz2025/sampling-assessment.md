# Source sampling sensitivity and conditional temporal uncertainty

**The raw event statistics are not yet a qualified inlet.** Gpudev job
**30272918** audited all **34 source stations / 996,164 events**, with archived
SHA-256 verification and source-only loading. All **15 tests passed**, including
the importer guards and eight sampling-estimator checks. No downstream records
were loaded for this assessment; no model coefficient or physical gate changed.

The [frozen protocol](sampling_protocol.json) compares event weights with transit
time weights and resamples equal-duration blocks at four resolutions. This is a
sensitivity analysis, not selection of an experimentally calibrated correction.
The authors report acquisition limits of 40,000 particles or 15 s in their
[manuscript](https://arxiv.org/abs/2512.15413). Here the observed first-to-last
arrival spans are 4.043–15.001 s; those spans do not establish the full acquisition
window or processor dead time.

## Result

| Observable | Maximum relative weighting change | Stations changing by >10% |
| --- | ---: | ---: |
| Axial mean velocity | 48.286% | 27/34 |
| Number-mean diameter D10 | 17.532% | 14/34 |
| D32 | 17.026% | 18/34 |
| Axial velocity standard deviation | 20.216% | 17/34 |

These are changes between two processing scenarios, **not model errors against
truth**. The 10% column describes sensitivity on the scale of the existing
engineering criterion; it does not introduce a new validation pass/fail gate.
Signed transverse means can approach zero: their maximum absolute shift is
2.494 m/s, so a maximum relative percentage is not a useful summary.

At Y=+8 mm, the axial event mean is **11.306 m/s**, while the transit-weighted
mean is **5.847 m/s**. At the independently acquired centres, X changes from
**66.602 to 63.219 m/s** and Y from **65.862 to 62.408 m/s**. At Y=+7 mm,
D32 changes from **35.329 to 29.314 micrometres**. No station is discarded for
having a large sensitivity.

The largest individual event-weighted bootstrap half-width is 4.169% of its
axial mean and 2.314% of its D32 estimate. These conditional sampling ranges
are much smaller than the maximum weighting shifts. However, changing the
number of time blocks changes the half-width by factors as large as **2.467**
for axial mean and **3.136** for D32. We do not select one resolution as a
validated decorrelation scale or interpret these as complete experimental
confidence intervals.

The inspectable figure is
`outputs/spray_closure_validation/racz-sampling-30272918/source_sampling.png`.
It shows both source traverses, axial means and D32, with envelopes over all
four block-bootstrap ranges. The joint raw data and all station/observable
results remain available in the generated report and CSV.

## Method and limits

For each event, one weight applies jointly to size and both measured velocity
components. Event weighting uses w=1; the alternative uses w proportional to
recorded transit time. Means use sum(w*q)/sum(w), D32 uses sum(w*D^3)/sum(w*D^2),
and velocity variance uses the corresponding weighted moments. Every audited
event remains present; no outlier trimming or size-range certification occurs.

A synthetic example with equal true concentrations at velocities 1 and 3 m/s,
event rates proportional to velocity and equal probe chord lengths recovers
2 m/s with transit weighting versus the biased event mean of 2.5 m/s. That
verifies the estimator under those assumptions. It does **not** establish equal
chords, known probe area, or size-independent detection for this experiment.
The measurement has polydisperse droplets and only two velocity components.
Transit weighting alone cannot supply its spatial liquid mass-flux distribution
or qualify a gas-velocity/stress inlet.

The observed arrival span is partitioned into 8, 16, 32 and 64 equal-duration
blocks. Each bootstrap draw samples whole blocks with replacement; 2,048 draws
produce conditional 2.5/97.5 percentile ranges. Shared block membership retains
joint measured size/velocity/transit correlations. There were no empty blocks
or empty draws in this source audit. The reported weight-concentration count
is a weight diagnostic, **not the number of temporally independent droplets**.

Interpretation requires stationarity and sufficiently weak dependence between
blocks. Long correlations, nonstationarity, acquisition stopping rules,
processor dead time, optical validation, probe-volume geometry and detection
bias are not included in the conditional ranges. Systematic uncertainties
cannot be removed by increasing event count or choosing smaller error bars.

## Consequence for transport validation

A transport simulation sampled as number density cannot be compared directly
with raw PDA event distributions and called validated. The measured inlet and
downstream observation operator need consistent size-, velocity- and
probe-dependent sampling. This audit quantifies that requirement but does not
supply the missing probe calibration. The independently identified sparse gas
proxy and missing joint third velocity component remain unresolved. The next
qualification work must establish that observation/inlet mapping or use an
independent dataset with qualified carrier and flux inputs; fitting downstream
profiles or normalizing event counts to nominal liquid flow would not resolve it.

## Reproduction

```bash
sbatch --output=outputs/spray_closure_validation/racz-sampling-%j.log \
  tools/audit_racz_sampling.sbatch
/u/limo/venvs/numba_cuda_waterboa/bin/python tools/plot_racz_sampling.py \
  outputs/spray_closure_validation/racz-sampling-30272918/report.json
```

`source_hashes.txt`, `report.json`, `statistics.csv`, `tests.xml` and the figure
are excluded from version control under `outputs/`. The report records the
frozen protocol hash and every source measurement hash. Acquisition headers of
other planes may be inspected to identify source files, but held-out numerical
records are never used in the audit.
