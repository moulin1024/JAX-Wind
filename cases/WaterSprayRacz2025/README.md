# Measured water-spray transport: Rácz PDA dataset

**Status: numerical data acquired and audited; physical qualification remains
open. No droplet-transport validation pass is claimed.** This supplies a public
alternative to the source-distribution gaps in the
[earlier candidate assessment](../FreeWaterSprayTransport/README.md).
The [active goal](../../doc/water-spray-validation-goal.md) still includes
profile accuracy, conservative coarse-grid LES coupling and eventual waterjet
validation.

## Sources and frozen split

Rácz, Malý, Jedelský and Józsa, *Spatio-temporal analysis of sprays by using
Phase Doppler Anemometry data*: [public raw data, CC-BY-4.0](https://doi.org/10.5281/zenodo.17935932),
[manuscript](https://arxiv.org/abs/2512.15413),
[journal article](https://doi.org/10.1016/j.ijheatmasstransfer.2026.129210).

The selected condition is `air-blast_1.28kgh_W_0.3bar_25C_1`: water at 25 C,
0.3 bar gauge atomizing pressure. This was chosen before any model evaluation
as the lowest-pressure ambient-temperature water subset. That choice limits,
but does not establish the absence of, compressibility or thermal effects.
The [protocol](protocol.json) reserves **z=20 mm as the measured source** and
**z=40/60 mm as downstream holdouts**. Downstream data cannot prescribe or tune
the inlet or closure coefficients. The 20 mm plane still needs qualification
as a post-breakup transport boundary.

| Plane | X stations | Y stations | Role | Droplet records |
| --- | ---: | ---: | --- | ---: |
| 20 mm | 17 | 17 | Source | 996,164 |
| 40 and 60 mm | 28 | 28 | Holdout | 1,434,558 |

Each record preserves joint diameter, axial velocity, one transverse velocity,
arrival time, transit time and two phase channels. LDA4 measures radial velocity
on the X traverse and tangential velocity on the Y traverse, in the signed
instrument frame. A third jointly measured component is unavailable. The two
centre measurements at each plane remain separate.

Coordinates come from explicit file headers. File order starts downstream.
Acquisition labels such as `X_20mm_center` also cover nonzero coordinates;
these labels are preserved and never used to override positions.

## Reproduction and evidence

Run from the repository root, with the documented Python environment:

```bash
/u/limo/venvs/numba_cuda_waterboa/bin/python tools/fetch_racz_spray_data.py \
  --output outputs/spray_closure_validation/racz_water_25C_03bar
sbatch --output=outputs/spray_closure_validation/racz-%j.out \
  tools/audit_racz_spray_data.sbatch
```

The downloader retrieves only the selected 90 measurement files and readme
from the 7,047,352,118-byte ZIP through bounded HTTP ranges. The initial download
transferred 59,641,654 bytes. `provenance.json` records archive metadata, selected
member CRC-32 and SHA-256, and the raw metadata checksum. ZIP member CRCs and
individual file hashes were checked; the full-archive MD5 was **not** verified.
Downloaded measurements, research PDFs, logs and generated reports remain under
ignored `outputs/`.

Gpudev job **30272001** completed successfully on 2026-09-16:

- **7 importer tests passed**, including units, velocity-component meaning,
  source-only rejection before loading held-out records, and acquisition labels.
- **90 stations, 2,430,722 records**, all checksums matched; explicit station
  layout matched the frozen source/holdout split.
- No nonfinite or nonpositive diameter/transit records were found. Nevertheless,
  62 records are below the 514.5 nm sizing wavelength and 1,456 exceed the
  earlier apparatus paper's reported 64.1 micrometre range. These records remain
  present and flagged; they are not silently clipped or certified valid.
- Audit results and source-code hashes:
  `outputs/spray_closure_validation/racz-audit-30272001/audit.json` and
  `station_audit.csv`; tests in `tests.xml`.

The previous job 30271989 failed because its importer rejected valid acquisition
label suffixes. Its five synthetic tests passed, but it did not finish the data
audit. The corrected job above supersedes it.

## Qualification findings and remaining work

The audit statistics are **event weighted**, without detection-efficiency,
probe-volume, transit-time or outlier corrections. They do not define an
unbiased inlet number-density or mass-flux distribution.

Source-only checks already warn against silently imposing axisymmetry. The
maximum mirrored axial event-mean difference divided by the pair mean is about
29.8% along X and 15.8% along Y. The independent centre traverses give axial
means 66.602 and 65.862 m/s. These are descriptive statistics, without an
uncertainty assessment or correction for detection bias. They are neither a
model-validation result nor proof of the mechanism behind the asymmetry.

Before constructing a defensible transport case:

1. Establish size-dependent sampling/probe corrections and spatial mass-flux
   weights. The [source sampling audit](sampling-assessment.md) now quantifies
   transit-weight sensitivity and conditional time-block ranges: axial means
   shift by up to 48.3% and D32 by 17.0%, despite much narrower conditional
   sampling ranges. This does not calibrate the probe or qualify either inlet
   scenario. The [probe compatibility audit](probe-assessment.md) also finds
   that 25.85% of source records exceed a nominal hard-volume axial transit
   bound; 2.15% exceed a generous finite-sphere/uncertainty scenario. Published
   nominal dimensions therefore do not establish an effective detection-volume
   correction. The raw records are retained. Do not interpret 40,000 events as
   40,000 independent samples.
   The [conditional chord-calibration assessment](chord-assessment.md) now
   tests a Gaussian-threshold axial-cylinder alternative. Even its nonnegative
   fit leaves about 16.2% of temporal-check events beyond inferred chord support;
   no measured-inlet correction is qualified by that test.
2. Supply carrier mean velocity and stress at the source. The
   [completed source-only proxy assessment](gas-proxy-assessment.md) reproduces
   the published selection rules but finds only 3–1,238 retained events per
   station (32/34 below the preliminary count screen). The estimator is not
   sufficiently supported to qualify the full carrier inlet. The
   [Rácz et al. gas-velocity estimator](https://doi.org/10.1016/j.ijmultiphaseflow.2022.104260)
   may support a **source-only model-assisted estimate** from small droplets;
   it is not an independent carrier measurement or a substitute for validating
   dispersion. The author code inspection now confirms the scaled-MAD filter,
   diameter-prefix/Stokes rules, and 12 mm length based on the liquid bore.
3. Resolve apparatus inputs. The
   [Urbán et al. apparatus paper](https://doi.org/10.1016/j.expthermflusci.2018.11.006)
   reports a 1.6 mm outer air-annulus diameter and 0.35 g/s liquid flow, while
   the later manuscript says 1.4 mm and the archive names 1.28 kg/h. Inner
   annulus diameter is 0.8 mm. Do not silently reconcile these differences or
   derive a unique air mass flow from gauge pressure alone.
4. Assess source asymmetry, missing third velocity component, breakup,
   evaporation and collision applicability. State and test any closure used
   for unavailable joint spatial/velocity information.
5. Freeze selected physical observables, measurement corrections, uncertainty
   and absolute tolerances for near-zero values before simulation. Preserve
   the user-revised 20% criterion for selected nonzero targets. Compare simulated
   and measured sampling consistently; report all selected targets.

A successful parser/data audit is progress toward measured validation; it does
not validate finite-inertia transport, turbulent dispersion, conservative LES
coupling, or the Montazeri waterjet.
