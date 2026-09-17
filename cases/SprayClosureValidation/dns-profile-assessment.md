# Independent DNS profile calibration and transfer assessment

This assessment tests whether a positive two-Gaussian reconstruction calibrated
on a separate stationary DNS supplies the missing radial shape. It is an
**empirical profile transfer test**, not a derivation or validation of a
Reynolds-stress closure. The Hussein holdout and original 10% screen are
unchanged. Nothing is enabled in the production waterjet.

## Independent source and frozen roles

The calibration source is [Shin, Sandberg and Richardson (2017)](https://doi.org/10.1017/jfm.2017.304),
with numerical figure data from the [Southampton repository](https://eprints.soton.ac.uk/407939/)
(DOI `10.5258/SOTON/D0075`). Their stationary round gas jet has `Re=7290`
and `Ma=0.304`; Figure 2a labels axial profiles over `x/D=15–40`. This differs
from the high-Reynolds-number Hussein experiment. Its transferability is an
assumption to assess, not a consequence of similarity notation.

The source archive, README, article, hashes, and selected original tables are
retained under ignored `outputs/spray_closure_validation/shin_2017/`. The archive
has 91,705,389 bytes. The source landing page and README specify CC BY-NC-ND;
raw source tables are not redistributed in the repository. The separate
Edinburgh decelerating-jet archive was also located, but is not used here.

The [protocol](dns_profile_protocol.json) was written before fitting. The numeric
source has 55 blank-separated blocks, despite its README describing column
headings. The first 26 blocks are normalized axial DNS curves (834 points each),
followed by short experimental overlays and radial-velocity curves. The original
figure was inspected to identify the fields and the collective station range;
no individual block-to-station mapping is asserted. The initial source audit
found exact duplicate blocks 1/0, 27/26 and 29/28. Only the first pair falls in
the axial-DNS calibration selection. Revision 2 removes this duplicate, leaving
25 unique curves, without selecting by fit quality or holdout performance.
All curves remain correlated observations from one DNS.

`fit_dns_jet_profile.py` opens only the protocol and independently hashed DNS
source. It interpolates all selected unique curves onto the same 601-point
`eta=0..0.3` grid and fits their equal-weight mean. No noisy/negative source
values are clipped. A single Gaussian is the control; the two-component mixture
uses three fixed optimizer starts and fixed positive bounds. All optimizer
results and each source-curve RMS residual are recorded. The held-out evaluator
is a separate process that reads the frozen calibration JSON. This separation
prevents fitting to the holdout in this implementation; the experiment is still
retrospective because earlier Hussein failures motivated the investigation.

## Reconstruction and conservation

For positive coefficients `ci` and nonnegative weights `wi` summing to one,

```
f(s) = sum_i wi exp(-ci*s^2)
I1 = integral_0^infinity s*f(s) ds = sum_i wi/(2*ci)
I2 = integral_0^infinity s*f(s)^2 ds
   = sum_ij wi*wj/[2*(ci+cj)]
U(r) = Uc*f(r/L)
```

The integrated mass flux `m` and mean-flow axial momentum `J` require

```
Uc = (J/m) I1/I2
L  = m*sqrt(I2) / [I1*sqrt(2*pi*rho*J)].
```

With `dm/dx=2*alpha*sqrt(pi*rho*J)`, the asymptotic length slope is
`sL=sqrt(2)*alpha*sqrt(I2)/I1`. A source shape given directly versus
`eta=r/(x-x0)` requires `sL=1`, hence `alpha=I1/sqrt(2*I2)`. The third candidate
retains only the DNS mixture shape and uses the independent Ricou `alpha=0.08`,
letting conservation determine the new length slope and centreline decay.
Neither candidate imposes an independently incompatible width or centre speed.
All candidates still omit Reynolds-normal-stress and pressure flux from `J`.

Analytic rectangular-face integration is also implemented for coarse-grid
reconstruction. Mass uses the individual Gaussian terms; momentum uses all
pair cross terms. Out-of-plane flux is retained as loss rather than redistributed
onto the cropped plane. Scalar profiles retain the original uniform-specific-
property assumption. This does not provide scalar mixing or an LES handoff.

## Reproduction and interpretation

With the original selected source file under the ignored source directory:

```bash
sbatch --output=outputs/spray_closure_validation/dns-profile-%j.out \
  tools/assess_dns_jet_profile.sbatch
```

The script runs independent infinite-domain and rectangular-face quadratures,
checks one-component/duplicate-component reduction to the original Gaussian,
then calibrates and evaluates all three declared candidates. All raw fits,
profiles, figures and logs stay under ignored output paths. A completed
assessment exits 1 when no candidate passes the physical screen; that is an
accuracy rejection, distinct from numerical-test or runtime failure.

The range across source curves is reported alongside the holdout. It is not a
confidence interval, and the source and holdout do not have identical nozzle,
Reynolds-number or development conditions. Published/profile-data uncertainty
is insufficient to turn this engineering screen into a statistical validation.

## Results and decision

Final gpudev job **30272244** passed **six numerical tests** in 7.70 s,
then completed the assessment with exit code 1 because **all three candidates
failed the physical screen**. The first job 30272238 also passed six tests and
failed all screens; it is retained to show the small effect of duplicate
weighting. Final reports and the frozen calibration are under
`outputs/spray_closure_validation/dns-profile-30272244/`. All eight recorded
source hashes, the calibration hash, pooled-profile hash, and JUnit result were
checked against the final files.

| Candidate | Decay B | Half-width slope | Alpha | Maximum local profile error | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| single_gaussian_dns | 6.59939 | 0.089206 | 0.075765 | 56.571% | No |
| two_gaussian_dns | 6.63067 | 0.088979 | 0.076672 | 53.355% | No |
| two_gaussian_shape_ricou | 6.35479 | 0.092842 | 0.080000 | 38.513% | No |

The DNS-only fit has RMS residual 0.007784 Uc for one Gaussian and
0.006286 Uc for the mixture. All three mixture optimizer starts converged to
the same interior solution, with a narrow component carrying about 2.18% of
the centreline amplitude; the fit did not reach a parameter bound. The fit
residuals are calibration errors, not independent validation results.

At `eta=0.2`, the original DNS curves span `U/Uc=0.01159–0.03725`,
with mean `0.02156`. The unchanged Hussein fit gives `0.07064`, with the
10% screen spanning `0.06358–0.07771`. Thus the source curves themselves
do not overlap that screening interval. A more flexible fit to this source
cannot by itself justify the larger held-out tail. This is evidence against
transferring this DNS shape unchanged across these conditions; it does not
identify whether the difference arises from development, Reynolds number,
boundaries, initial conditions or measurement/numerical uncertainty.

The mixture/Ricou result (38.513%) does not improve the original Gaussian
(37.805%) or the separately specified Huck Gaussian (29.524%). Keep this
candidate disabled. Next modeling work must address transport/stress physics
and source/holdout applicability rather than add unconstrained shape freedom.
No holdout threshold, coefficient, domain or target role has been changed.
Measured droplet transport, LES integration and waterjet validation remain open.

## Additional measured-profile lead — not yet assessed

The archived Shin et al. paper, page 7, explicitly identifies the experimental
comparison in Figure 2(a–c) as **Panchapakesan and Lumley (1993)**, with experiment
Re=11,000 and a near-laminar top-hat nozzle condition. These are distinct from the
Hussein holdout. The current DNS calibration protocol deliberately excluded the
short experimental overlay blocks; they have not been used for fitting here.
Before considering them as a separate measured calibration source, establish
block-to-curve mapping, axial station and normalization from the original paper
and figure, and check the primary measurement provenance and uncertainty. This
is a source lead, not a new accuracy result or permission to fit the Hussein
profile. Source: [Shin et al. (2017)](https://doi.org/10.1017/jfm.2017.304), with
raw figure data already held in the ignored DOI 10.5258/SOTON/D0075 archive.


That measured-source lead has now been assessed in the
[measured-profile and momentum-budget investigation](measured-profile-assessment.md).
The source-derived budget correction improves the best held-out local error to
26.018%, still failing the same 10% criterion. The existing DNS results above
are unchanged; the new source and follow-up hypothesis have separate protocols.
