# Independent measured-profile transfer and momentum accounting

The best new candidate reduces maximum local Hussein profile error to
**26.018%**, from the previous best **29.524%** and original **37.805%**. It still
**fails the unchanged 10% engineering screen**. No candidate is production
enabled, and no measured droplet or waterjet validation is claimed.

## Source qualification and frozen roles

The independent source is [Panchapakesan and Lumley (1993), Part 1](https://doi.org/10.1017/S0022112093000096).
Its Figure 7 reports `U/Uc` against `r/x`, with stations `x/d=60..120` and a
least-squares spline through the full set. The paper reports half-width 0.096,
decay coefficient 6.06 and an approximate Gaussian coefficient 75.2. Calibration
RMS errors of 0.01 m/s and 0.5 degrees and total momentum agreement within 5%
are not pointwise normalized-profile confidence intervals.

[Shin et al. (2017)](https://doi.org/10.1017/jfm.2017.304) independently attribute
the solid experimental curves in Figure 2a to this study. Their
[archived figure data](https://eprints.soton.ac.uk/407939/) contain one 40-point
axial overlay in block 26, repeated exactly as block 27. The latter is excluded;
all DNS and radial-velocity blocks are excluded. The extraction pathway is not
documented. These are a secondary numeric representation of an experimental
profile, **not raw measurements or independent realizations**.

The archived curve has half-width **0.095316**, agreeing with the primary
rounded value within 0.72%. Its coordinates are retained as reported; no origin,
width or centre normalization is adjusted to match Hussein. The complete source
support is used. Source PDFs, tables, fitted artifacts and figures remain under
ignored outputs, with downloaded publisher PDF and source-archive hashes.

The [measured-profile protocol](measured_profile_protocol.json) was frozen before
fitting. The calibration process reads only that protocol and the hashed source
curve. All 40 unique points are fitted with trapezoidal radial weights, preventing
uneven extraction density from defining the objective. A single Gaussian and a
positive two-Gaussian mixture use fixed bounds and three fixed mixture starts.
The published Gaussian is an independent control. All optimizer results are
retained. A separate process opens the unchanged Hussein reference only after
the calibration JSON is written.

The assessment remains retrospective: prior holdout failures motivated this
study. The source has different Reynolds number and nozzle/development conditions.
It does not establish a universal profile or a statistical confidence statement.

## Initial measured-shape assessment

Final gpudev **30272732** passed the existing **six reconstruction conservation
checks**, then assessed all four declared candidates. The assessment exits 1
because all candidates fail the physical screen; this is not a runtime failure.
The initial **30272713** gives the same results and is retained.

| Candidate | Decay B | Half-width slope | Alpha | Maximum profile error |
| --- | ---: | ---: | ---: | ---: |
| Published Gaussian, C=75.2 | 6.13188 | 0.096007 | 0.081541 | 30.085% |
| Source-fitted Gaussian | 6.21368 | 0.094743 | 0.080468 | 35.511% |
| Source-fitted mixture | 6.25548 | 0.094509 | 0.081734 | 30.388% |
| Mixture shape with Ricou alpha | 6.39109 | 0.092503 | 0.080000 | 38.927% |

The fitted single coefficient is **77.21957**. The mixture weights are
**0.026045/0.973955**, with coefficients **1380.130/74.64874**. All starts converge
to the same interior solution without active bounds. Weighted source RMS error
falls from **0.011101 Uc** for one Gaussian to **0.008249 Uc** for the mixture.
This absolute RMS measure does not imply an accurate relative fit in the outer
low-velocity region.

At similarity radius 0.2, the secondary source gives **0.032324 Uc**, while the
unchanged Hussein fit gives **0.070644 Uc** and its 10% interval is
**0.063579–0.077708 Uc**. The source curve itself therefore differs by **54.244%**
there. A more flexible fit to this particular source cannot alone establish
transferability. The original Hussein coefficient row was visually rechecked;
no reference coefficient, range, metric or threshold was changed.

## Separate momentum-partition hypothesis

The initial reconstruction equates conserved nozzle momentum flux with the
mean advective contribution `integral rho*U^2 dA`. The full turbulent momentum
budget also contains normal-stress and pressure contributions. A separate
[follow-up protocol](momentum_partition_protocol.json) was frozen after the
initial measured-profile rejection, before evaluating this accounting change.
It uses only the independent source decay and its already-frozen profile fit;
it does not use Hussein stress or decay values as calibration inputs.

For source shape `U/Uc=f(r/x)`, define

```
I1 = integral eta*f(eta) d_eta
I2 = integral eta*f(eta)^2 d_eta
Uc = B_source*Ujet*D/x
J_total = rho*Ujet^2*pi*D^2/4
chi = J_mean/J_total = 8*B_source^2*I2
alpha_source = 2*B_source*I1.
```

This is an inferred integral partition, not a measurement of a local SGS energy.
The assessment uses `m=2*alpha*sqrt(pi*rho*J_total)*x` for entrainment and
`J_mean=chi*J_total` for mean-profile reconstruction. It explicitly retains
`J_total-J_mean` as the combined stress/pressure flux remainder; it is not removed
from the momentum budget. This remainder has units of force, not stored energy.
It cannot be inserted into an SGS-k reservoir or a cell momentum source without
a spatial stress/pressure closure and a conservative handoff.

For every source family, using `alpha_source` recovers the source decay **6.06**
and native similarity length **1** through the reconstruction equations. That
is a calibration identity, not an independent physical validation. The separate
Ricou mode keeps the independently measured entrainment coefficient **0.08**.
Every family and both modes are reported; none is retuned after evaluation.

| Profile family | Inferred chi | Entrainment mode | Decay B | Half-width slope | Maximum profile error |
| --- | ---: | --- | ---: | ---: | ---: |
| Published Gaussian | 0.976691 | Source integral | 6.06000 | 0.096007 | 30.085% |
| Published Gaussian | 0.976691 | Ricou | 6.10432 | 0.095310 | 33.105% |
| Fitted Gaussian | 0.951147 | Source integral | 6.06000 | 0.094743 | 35.511% |
| Fitted Gaussian | 0.951147 | Ricou | 5.94467 | 0.096581 | 27.547% |
| Fitted mixture | 0.938479 | Source integral | 6.06000 | 0.094509 | 30.388% |
| Fitted mixture | 0.938479 | Ricou | 5.99790 | 0.095487 | **26.018%** |

All six still fail. The best case keeps about **6.15%** of total momentum flux
in the explicit stress/pressure remainder and passes the integral screens, but
misses the outer profile. Its improved holdout result does not establish an
accurate pointwise fit to the secondary source tail either.

The partition is inferred under a top-hat, uniform-density nozzle assumption;
measurement/extraction errors and finite-domain effects are not assigned a joint
uncertainty distribution. Assuming this source-derived partition transfers to
the higher-Reynolds-number holdout remains unvalidated. A scalar chi alone does
not specify anisotropy, turbulence energy, pressure work or spatial dispersion.
Thus the result supports retaining the full momentum budget in future modeling,
not enabling an empirically corrected waterjet source.

## Reproduction

```bash
sbatch --output=outputs/spray_closure_validation/measured-profile-%j.out tools/assess_measured_jet_profile.sbatch
sbatch --output=outputs/spray_closure_validation/momentum-partition-%j.out tools/assess_jet_momentum_partition.sbatch outputs/spray_closure_validation/measured-profile-30272732/calibration.json
```

The independent measured curve has not closed the profile gap. Further work
must address stress/pressure accounting, profile applicability and uncertainties
alongside measured finite-inertia transport and conservative LES energy/ownership
coupling. The waterjet remains unvalidated and the persistent goal stays active.


## Final provenance

The final measured-profile run is **30272732** (six numerical checks passed;
four physical screens rejected). The final momentum-partition run is
**30272757** (all three source-integral identities passed; six physical screens
rejected). Earlier partition **30272752** gives the same numbers; the final run
adds the reviewable profile/error figure. Both final Slurm jobs exit 1 explicitly
for unmet physical criteria. Exact recorded source hashes, calibration hashes,
source-profile hash and the partition's link to the measured calibration were
verified. The holdout file still matches its hash in DNS assessment **30272244**.

Reports, calibrations and plots are retained under ignored
`outputs/spray_closure_validation/measured-profile-30272732/` and
`outputs/spray_closure_validation/momentum-partition-30272757/`. No research PDF,
raw table, log or generated artifact is included in version control.
