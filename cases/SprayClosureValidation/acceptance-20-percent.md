# User-authorized 20% engineering acceptance

On 2026-09-17 the user changed physical acceptance from 10% to **20%**. The
criterion applies to profile/integral observables, selected measured droplet
transport targets, and **each** of the nine waterjet temperature sensors.
Source-data qualification, numerical convergence, conservation and independence
requirements are unchanged. This is a retrospective change in the engineering
criterion, not an improvement in predictions or experimental confidence.

Gpudev **30273713** re-assessed the source-qualified constrained gas-profile
candidate: maximum local error **19.8810%**, decay error **13.9298%**, width error
**7.4209%**, and entrainment error **2.1530%** now all pass. The uncorrected
control remains numerically rejected and was not assessed against the holdout.

Gpudev **30273715** re-read the archived waterjet histories. All four recent
schemes now pass **9/9** sensors. Worst errors are **12.9899%** (MUSCL momentum /
legacy scalars), **12.5307%** (central momentum / legacy scalars), **12.5443%**
(central / upwind SSP-RK3), and **12.6945%** (central / MUSCL SSP-RK3).
Predictions and worst-error values match the historical reports to 1e-10.
The normalization remains measured temperature in Celsius, as previously agreed.

Both re-assessments completed successfully. The old 10% reports remain intact;
explicitly named 10-percent fields remain historical diagnostics in the waterjet
comparison tool, while active acceptance uses its configured threshold.
New reports are ignored outputs under `constrained-profile-30273713/` and
`outputs/acceptance_revision_20260917/waterjet-30273715/`.

Passing these agreement screens does **not** complete measured droplet transport,
production LES integration, or the full waterjet conservation and grid/timestep/
parcel/averaging verification. The persistent goal remains active.
