# Source-only assessment of a conditional probe calibration

**Outcome: the tested axial-cylinder calibration is not qualified for correcting
this inlet. No measured transport pass or raw-data rejection is implied.**
The [protocol](chord_protocol.json) and numerical implementation retain all
station fits and temporal checks, including inadmissible results. Downstream
40/60 mm observations have not been used for this calibration.

## Published methods and available inputs

[Saffman (1987)](https://doi.org/10.1364/AO.26.002592) describes automatic
calibration from burst measurements, including a two-dimensional extension.
Only its abstract and indexed first page were accessible in this assessment;
the attempted public PDF download timed out. No uninspected equation from that
paper is implemented or claimed reproduced here.

[Aísa et al. (2002)](https://www.sciencedirect.com/science/article/abs/pii/S0301932201000714)
explicitly combine probe calibration with size-class flux normalization in a
nonbreaking, nonevaporating glass-particle jet. This is not independent evidence
for applying that normalization to an evaporating water spray. Their abstract
and accessible article sections were inspected; the full calibration algorithm
was not recovered.

An accessible [cloud-PDI experiment, section 2.2](https://amt.copernicus.org/articles/15/965/2022/)
uses size-dependent area proportional to the square root of log(D/Dmin), with
beam-waist and cutoff parameters obtained from the instrument's software.
Its size-dependent volume and signal-processing corrections are specific to
that instrument. They are a motivation for testing a Gaussian-threshold model,
not supplied Dantec calibration parameters.

[Christou et al. (2021), pp. 6–7](https://publikationen.bibliothek.kit.edu/1000138012/127676792)
use individual detection areas for flux estimation and check the integrated
cross-sectional result against a separate flow meter, reporting less than 9%
deviation. That independent integral check is useful methodology; it is not a
calibration or measured flux supplied for the present Rácz condition.

The Rácz README and actual eight-column exports supply arrival/transit times,
two velocity components, diameter and two phase channels. They supply no
explicit rejected-burst counts, live-time record or calibrated effective-area
field. Row number is not asserted to measure rejected events. Those omissions
remain separate from whether a geometric radius can be inferred.

## Independently derived conditional model

Assume locally homogeneous **axial** straight tracks, point droplets, complete
threshold detection, a circular optical cylinder of radius R truncated by a
constant receiver slit, and a uniform transverse impact coordinate y in [-R,R].
For axial transit length ell=abs(u_axial)*TT,

```
ell = 2 sqrt(R²-y²)
E(ell) = pi R / 2
E(ell²) = 8 R² / 3
P(ell/R <= z) = 1 - sqrt(1-z²/4),  0 <= z <= 2
```

Gaussian intensity plus a power-law scattering response motivates
`R(D)² = a log(D_um) + b`. An ordinary least-squares fit of
`(3/8)*ell_um²` against log diameter estimates a and b from training events.
The cylinder moment formula is derived above and tested by independent
geometric quadrature. It is not a full multidirectional optical model or an
implementation of the unavailable Saffman algorithm. A mean fit alone cannot
qualify its distributional, efficiency or support assumptions.

All 34 source stations are analyzed separately. Diameters in the previously
reported [0.5145,64.1] micrometre range form a stated **diagnostic subset**;
832 other events remain in the raw data and audit counts. This is not a new
measurement-validity filter. Equal-duration 8/16/32-block partitions use even
blocks for fitting and odd blocks for checking. Boundaries within eight
floating-point ulps of the absolute time scale are tied to the right block.
The temporal partitions assess stability; they are not independent experiments
or a basis for IID significance claims.

All unconstrained fits in the initial completed audit had positive slopes but
negative radius squared at the lower declared diameter. Rather than rejecting
the hypothesis solely because unconstrained least squares was inadmissible, an
explicit protocol extension added a nonnegative fit:

```
R² = a log(D/D_lower) + c,  a >= 0, c >= 0.
```

This two-variable convex least-squares problem is solved by comparing its
interior solution with both boundary solutions. No radius is clipped. A zero
radius is permitted as a limit at D_lower, but every included measured diameter
must have strictly positive radius. Both original and constrained results are
retained. This adaptive follow-up is documented, not presented as blind
validation. Neither fit uses a downstream observation or nominal nozzle flow.

## Results and verification

Final gpudev **30273040** passed **16 tests** (nine calibration tests and seven
importer regressions), checked source-file hashes and evaluated **996,164
source events**. All final audit source hashes match. The earlier job 30273029
stopped at a floating-point temporal-boundary test before loading measured
calibration data. That tie was fixed; 30273030 passed the original 15 tests and
completed the unconstrained assessment. No physical tolerance was relaxed.

| Time blocks | Unconstrained implied cutoff range (µm) | Nonnegative check events | Beyond inferred chord support | Size bins with >10% second-moment mismatch / supported bins |
| --- | --- | ---: | ---: | ---: |
| 8 | 2.642–10.014 | 492,509 | 79,614 (16.165%) | 223 / 265 |
| 16 | 2.630–9.980 | 492,443 | 79,938 (16.233%) | 226 / 266 |
| 32 | 2.662–9.390 | 492,399 | 80,063 (16.260%) | 227 / 266 |

All 102 unconstrained station/split fits are inadmissible over the stated size
range; roughly 1.49–1.51% of included source events have nonpositive inferred
radius squared. The nonnegative fits support every included diameter, but
held-source CDF distances range from **0.147 to 0.357**. Every station has chord
support exceedances. Size-bin counts in the table require at least 100 check
events; the 10% comparison is a descriptive moment mismatch, not a replacement
for the independent physical transport gate or an uncertainty interval.

Additionally, **558,964 / 996,164 events (56.112%)** have
`atan(abs(u_transverse)/abs(u_axial)) > 5 degrees`. This is a lower bound on the
angle to the axis because the third joint velocity is missing. No events were
filtered by angle. It underscores the limitation of the axial specialization;
it does not identify a unique cause of the optical mismatch. Finite droplet
size, trajectory/slit truncation, detection thresholds, transit-time processing
and validation selection remain possible contributors.

Reproduction:

```bash
sbatch --output=outputs/spray_closure_validation/racz-chord-%j.log \
  tools/audit_racz_chord.sbatch
# On a gpudev allocation, replacing JOBID with that completed job:
/u/limo/venvs/numba_cuda_waterboa/bin/python tools/plot_racz_chord.py \
  --report-directory outputs/spray_closure_validation/racz-chord-JOBID
```

Reports, hashes, XML, logs and the inspected PNG/PDF figure remain in ignored
outputs. The figure includes both traverses and all three temporal partitions.

## Implication for measured transport

This assessment does not support using the simple calibrated radius to turn
recorded events into physical mass-flux weights. It also does not prove that
all automatic calibration is impossible. A defensible next route is a
multidirectional receiver-geometry model with qualified transit processing and
validation efficiency, checked against an independent flux or concentration
measurement. Alternatively, select a transport dataset with those quantities
already qualified. The missing carrier inlet, joint third velocity, apparatus
ambiguity and transverse spatial integration remain separate requirements.
No inlet correction, source distribution, production setting or downstream
acceptance criterion has been changed.
