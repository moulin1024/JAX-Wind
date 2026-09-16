# Small-case correction with central precursor preservation

Use `fv_v80_small_central_open_upwind.toml`. The opt-in numerical policy is
`momentum_advection_scheme = "central-open-upwind"` with full RK3.

- Periodic streamwise domain: exactly the existing central momentum operator.
- Open streamwise domain: conservative MUSCL-MC horizontal momentum fluxes,
  centered vertical fluxes. This suppresses horizontal grid-scale modes without
  adding upwind diffusion across the mean vertical shear.
- The high-x energy backflow pressure remains available. The experimental
  lateral perturbation-energy pressure switch is removed from the integrator;
  lateral pressure outlets retain their ordinary pressure/extrapolation closure.
  Physical lateral entrainment is allowed, not clipped.
- GMG remains the pressure solver. Divergence was already small before the fix;
  changing the preconditioner was not the remedy for the observed oscillations.

The derived DTU configuration `../DTU10MWPrecursor/fv_central_backflow.toml`
now selects this policy, retaining its Porté-Agel correction. Periodic warmup
and precursor therefore remain central, while its open main stage receives
horizontal upwinding. Existing historical cases and numerical defaults are
unchanged. Uniform meshes are required for the new option.

## Completed single-turbine validation

One V80, 128 × 64 × 64 cells; 2048 × 1024 × 256 m; 16 × 16 × 4 m spacing;
uniform 10 m/s inflow; fixed dt=0.25 s; 1200 steps / 300 s. The corrected run
retains the baseline physical settings and the same reduced domain.

| Metric | Central / ordinary outlets | Corrected |
|---|---:|---:|
| Final upstream second-difference RMS [m/s] | 0.03978103 | 0.00029962 |
| Maximum adjacent upstream face jump [m/s] | 0.18297291 | 0.00978470 |
| Peak lateral inward speed [m/s] | 0.230427 | 0.099884 |
| Minimum sampled downstream u [m/s] | 3.330297 | 2.588762 |
| Final hub outlet minimum u [m/s] | 6.897274 | 5.516509 |
| Maximum divergence [1/s] | 1.453e-07 | 1.751e-07 |

The upstream oscillation measure decreases by
99.25% and the adjacent-face jump by
94.65%.
The failed lateral-pressure variant had a peak inward speed of 0.510917 m/s;
the corrected run removes its near-wall switching spikes. No sampled negative
streamwise velocity or downstream backflow occurred in the corrected run.
Sampling is every 3 s and cannot rule out shorter events.

## What is preserved, and what is not established

The saved DTU central precursor at t=39600 s (64 × 32 × 128) was evaluated
with both policies. All three complete momentum tendencies were bitwise
identical (maximum difference 0), including AMD, pressure forcing, and the
Porté-Agel wall treatment. Scalar transport and the periodic integrator are
unchanged. Thus the new policy preserves the existing central precursor
calculation and its previously measured time/volume-averaged log-law result;
this is not a claim of a newly simulated or perfectly matching log law.
The existing central reference had 20–100 m RMSE about 0.708 m/s and was not
fully stationary.

An independent open-column test with logarithmic mean shear and nonzero
vertical transport verifies centered wall-normal momentum fluxes on evolved
faces. The open integrator prescribes/extrapolates its x-end layers.
Twenty advection tests pass, including the existing full-MUSCL regressions,
periodic exact equality, central vertical flux, and dissipation of a horizontal
checkerboard mode. Eleven existing outlet/projection tests also pass. Two
initial new test fixtures incorrectly included prescribed endpoint tendencies
and nonzero impermeable-wall flux; their corrected versions pass in the
20-test advection suite. The original test log is retained.

Horizontal upwinding changes turbulent transport and the downstream wake:
the corrected far-wake deficit is deeper in this 300 s uniform-inflow check.
A long-time turbulent open-domain log-law and wake-recovery validation has
not been performed. This correction preserves the periodic log-law reference
by construction; it does not prove an unchanged profile everywhere in a wake.
No full-farm debug rerun was used.

## Artifacts

- [Upstream comparison](../../outputs/v80_small_backflow_fix_20260916/run/backflow_analysis/upstream_comparison.png)
- [Near-wall lateral comparison](../../outputs/v80_small_backflow_fix_20260916/run/backflow_analysis/lateral_near_wall_comparison.png)
- [All boundary histories](../../outputs/v80_small_backflow_fix_20260916/run/backflow_analysis/comparison_with_baseline.png)
- [Corrected wake movie](../../outputs/v80_small_backflow_fix_20260916/run/v80_single_hub_u.mp4)
- [Real-precursor equivalence check](../../outputs/v80_small_backflow_fix_20260916/precursor_equivalence.json)
- [Existing time/volume-averaged log-law reference](../../outputs/dtu10mw_central_cfl09_20260916/analysis/log_law_profile.pdf)

## Reproducible follow-up submission

Run `bash tools/submit_backflow_validation.sh` from the repository root to submit
matched small wake reruns and a paired turbulent open-domain log-law experiment.
The [submission and interpretation guide](../DTU10MWPrecursor/backflow_validation.md)
describes averaging windows, downstream subvolume checks, thresholds, and outputs.

The subsequent full turbulent open-domain log-law comparison has now completed.
It **fails the declared preservation screen**: corrected log-law RMSE is
0.7297 m/s versus 0.6408 m/s for central open flow, and downstream quarter
increases exceed the 0.1 m/s limit. Wall u* drops from 0.4317 to 0.3385 m/s.
See the [completed validation](../DTU10MWPrecursor/backflow_validation.md#completed-interactive-validation-2026-09-16).
The earlier exact periodic-operator equivalence remains valid, but does not
establish preservation of the open-domain wall layer.
