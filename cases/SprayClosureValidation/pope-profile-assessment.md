# Independently sourced Pope-convention profile assessment

Current acceptance is **20%**, explicitly revised by the user on 2026-09-17.
The qualified 19.8810% candidate now passes that agreement screen. See the
[re-assessment](acceptance-20-percent.md); older 10% results below are historical.

Status: numerical qualification and independent gas-profile assessment,
**not a validated production spray model**. The best maximum local profile
error is now **19.881%**, improved from 26.018%, but still fails the unchanged
10% screen. The corresponding centerline decay coefficient is 13.930% high.
The production waterjet settings remain unchanged.

## Resolve the source convention before evaluating the holdout

A high-resolution reread of [Pope (1978), printed p. 279](https://tcg.mae.cornell.edu/pubs/Pope_AIAAJ_78.pdf)
confirmed C1=1.45 and C2=1.90. Those values were transcribed correctly. The
normal-strain term audit and a separately documented coefficient convention
were frozen before accessing any new held-out profile result.

The [official OpenFOAM v2212 standard-model documentation](https://doc.openfoam.com/2212/tools/processing/models/turbulence/ras/linear-evm/rtm/kEpsilon/)
specifies C1=1.44 and C2=1.92. An initial diagnostic used only these two
independently documented pairs; it did not optimize or interpolate coefficients.

Further primary-source research located
[Speziale, Raj and Gatski (December 1990), NASA CR-187485 / ICASE 90-88](https://ntrs.nasa.gov/api/citations/19910006993/downloads/19910006993.pdf?attachment=true).
Equation (11) and its accompanying paragraph explicitly specify the Pope
correction with **C1=1.44, C2=1.92, C3=0.79**. This report predates the 1994
holdout. Its PDF SHA256 is
`765111b1a24887f048d309f60b93ed70e55f5d4c21c0e41c4b79ce64eceaa105`.
The report also finds that this correction does not resolve deficiencies in
rotating isotropic turbulence or rotating homogeneous shear. It does not justify
a general rotation/swirl closure.

The [1990-convention protocol](pope_1990_profile_protocol.json) therefore treats
that convention separately, retaining the failed 1978-printed candidates.
It does not assert which coefficients Pope actually used in his original
calculation. Both previously specified production approximations were carried
through the holdout assessment. The source spreading criterion remains 2%, and
the holdout criterion remains 10%; neither was loosened.

## Steady solver and edge qualification

The verified downstream march provided a useful initializer for the previously
unsuccessful boundary-value formulation. The new
`tools/solve_kepsilon_profile.py` calls the shared, verified equations in
`tools/kepsilon_similarity.py` rather than maintaining another equation copy.
It solves simultaneously for the profile and an unknown zero-turbulence edge.

Let s=eta_edge-eta. The leading free-edge balances give

```
F ~ s^m, K ~ s^p, E ~ s^q,
m = 1/(2 sigma_k - sigma_e), p = sigma_k m, q = sigma_e m.
N ~ I s/(eta m), Qk ~ -I K, Qe ~ -I E.
```

The three latter relations supply outer boundary conditions at
eta=(1-delta)*eta_edge. Axis conditions are I=Qk=Qe=0 and F=1. No measured
width or entrainment constant sets the coordinate. The remaining small tail is
integrated with its leading power law, with its effect controlled by delta
refinement. Numerical exponent safeguards are permitted during Newton trials;
accepted solutions require them to be inactive.

The normal-strain diagnostic adds 2 N (Sxx²+Srr²+Stt²) to shear production.
The shared implementation is checked against the original dimensional
advection/diffusion/reaction equations and against known axis strain. This
still omits dV/dx in shear/rotation and axial diffusion/stress transport, so it
is an intermediate approximation, not full RANS.

Gpudev **30273482** first converged the corrected model and its normal-production
continuation. The standard-model initializer had an empty sparse-tail mask;
that failed attempt is retained. A guarded initializer and edge/tolerance study
**30273487** converged both models. The separate convention diagnostic
**30273490** used the fixed 1.44/1.92 pair. No held-out data were read in any of
these trials.

Final versioned run **30273498** passed **14 equation/kernel tests**, qualified
all eight combinations of coefficient pair, correction and production choice,
and verified source hashes. Its numerical study used delta=0.08, 0.04, 0.02,
0.01 and 0.005, then tightened collocation tolerance from 1e-6 to 1e-8 and changed
the initial mesh from 400 to 800 nodes. All accepted profiles were finite,
positive and monotone, with inactive trial safeguards. The maximum profile
change across the final edge refinements was **7.14e-7 Uc**; the largest
width/integral change was **3.23e-6 relative**. Tolerance and initial-mesh effects
were smaller. These are numerical changes, not experimental uncertainty.

| Coefficient pair | Correction C3 | Production | Computed spreading | Historical source rate | Source error |
|---|---:|---|---:|---:|---:|
| 1.45 / 1.90 | 0 | shear | 0.11197559 | 0.125 | 10.420% |
| 1.45 / 1.90 | 0 | normal added | 0.11388948 | 0.125 | 8.888% |
| 1.45 / 1.90 | 0.79 | shear | 0.07959133 | 0.086 | 7.452% |
| 1.45 / 1.90 | 0.79 | normal added | 0.08060281 | 0.086 | 6.276% |
| 1.44 / 1.92 | 0 | shear | 0.11987467 | 0.125 | 4.100% |
| 1.44 / 1.92 | 0 | normal added | 0.12217899 | 0.125 | 2.257% |
| 1.44 / 1.92 | 0.79 | shear | 0.08577297 | 0.086 | **0.264%** |
| 1.44 / 1.92 | 0.79 | normal added | 0.08702439 | 0.086 | **1.191%** |

Only the last two pass the 2% source-rate comparison. This does not prove a
faithful reconstruction of the undocumented 1978 numerical implementation.
Normal production alone does not explain the printed-convention discrepancy;
coefficient convention has a larger effect. No coefficient was selected using
the 1994 profile errors.

## Held-out result

The assessment checks both candidate qualifications, coefficient provenance,
source-rate gates and solver hashes **before** loading `reference.json`.
Gpudev **30273502** then evaluated the unchanged 201 radial points over
0<=r/(x-x0)<=0.2. It also retained the original mean-momentum convention for
B and alpha, without a measured width/coordinate rescaling or stress-factor fit.

| Metric | Holdout | Shear production | Normal strain added |
|---|---:|---:|---:|
| Decay coefficient B | 5.8 | 6.70912 (**15.674% high**) | 6.60793 (**13.930% high**) |
| Half-width slope | 0.094 | 0.085773 (8.752% low) | 0.087024 (7.421% low) |
| Entrainment alpha | 0.081 | 0.081295 (0.364% high) | 0.082744 (2.153% high) |
| Maximum local profile error | <=10% | **24.579%** | **19.881%** |
| Location of maximum error | — | eta=0.177 | eta=0.164 |
| Profile RMS / Uc | — | 0.040246 | 0.033461 |
| Overall screen | — | **FAIL** | **FAIL** |

Both models underpredict the normalized profile through much of the outer jet.
Good entrainment and half-width summaries do not remove the local-profile or
centerline failures. The normal-strain case is the best diagnostic so far, but
it remains an incomplete momentum/production approximation. The prior 26.018%
result remains in its original assessment; it is not overwritten.

The source-normalized model now provides a reproducible positive k/epsilon
profile, which can inform subsequent turbulence work. It does not yet supply
filter-aware LES unresolved energy, finite-inertia inhomogeneous dispersion,
measured droplet validation, or conservative embedded-core ownership/handoff.
The next physical work should address consistent stress/pressure momentum
transport and dispersion against independent sources, without tuning to these
newly observed holdout errors. Waterjet validation remains open.

The subsequent [stress/pressure and realizability audit](jet-stress-assessment.md)
finds that the omitted section flux is small and negative. It also verifies a
negative covariance eigenvalue within a thin solved edge region; raw stress
cannot be used directly there for droplet dispersion. The profile scores above
are unchanged.

## Reproduction and artifacts

- `sbatch tools/assess_kepsilon_source.sbatch` produces the numerical march.
- Set `JAXWIND_KEPSILON_MARCH` to that output and run
  `sbatch tools/assess_kepsilon_similarity.sbatch`.
- Set `JAXWIND_KEPSILON_SOLUTIONS` to its output and run
  `sbatch tools/assess_pope_1990_profile.sbatch`.

The defaults point to the verified jobs above. Numerical reports and states are
in ignored `outputs/spray_closure_validation/kepsilon-similarity-30273498/`.
The final JSON, profile CSV, PNG and PDF comparison are in ignored
`outputs/spray_closure_validation/pope-1990-30273502/`. The plot was visually
inspected. The primary NASA PDF/text and retrieval provenance are in ignored
`outputs/spray_closure_validation/speziale_1990/`. Source hashes passed and
no research PDF, log or generated artifact is added to version control.

The [consistent constrained-timescale follow-up](constrained-kepsilon.md)
re-solves transport with a physical covariance constraint. The qualified
corrected candidate still fails the independent profile gate at 19.8810%;
the uncorrected control remains numerically rejected.
