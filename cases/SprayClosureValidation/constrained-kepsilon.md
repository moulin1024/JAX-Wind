# Consistent constrained-timescale experiment

Current acceptance is **20%**, explicitly revised by the user on 2026-09-17.
The qualified 19.8810% candidate now passes that agreement screen. See the
[re-assessment](acceptance-20-percent.md); older 10% results below are historical.

This research candidate inserts the momentum-compatible realizability bound
into the existing normal-production jet equations. It is a **derived hybrid**,
not an independently validated published closure. It is not enabled in LES or
waterjet production. The [protocol](constrained_kepsilon_protocol.json) freezes
the previously documented constants and both C3=0 and C3=0.79 cases. No empirical
multiplier or held-out profile is used to choose coefficients.

## Verified result

Final paired solve **30273699** passed **12 equation/feasibility checks**.
The C3=0.79 candidate passed all six solves, edge/tolerance/initial-mesh
refinements, and the independent source-width gate. Its final original-coordinate
RMS residual is **2.13e-8**, boundary residual **2.84e-14**, and minimum modeled
covariance eigenvalue **-4.24e-16 K**, consistent with floating-point roundoff.
The active constraint begins at eta **0.27251218**. At the final relative edge
cutoff **3.125e-6**, viscosity is **0.234365** of its unconstrained value.
The source half-width slope **0.0870243942** differs from 0.086 by **1.1912%**.

The C3=0 control failed startup with a singular Jacobian and an active transition
parameter guard. It is **not numerically qualified**. The paired Slurm job
therefore exited **2**, deliberately; the passing candidate does not make the
whole job successful. Source hashes were manually checked after termination,
and exact matching source snapshots were preserved under the ignored job output.

Separate gpudev assessment **30273701** completed successfully, with all recorded
source hashes unchanged. The [assessment protocol](constrained_profile_assessment_protocol.json)
retains the rejected control without a held-out score. It assesses the qualified
candidate against all unchanged Hussein points and integral criteria:

| Quantity | Prediction | Relative discrepancy |
| --- | ---: | ---: |
| Maximum local profile error, at eta=0.164 | — | **19.8810%** |
| Centerline decay coefficient | 6.60792728 | 13.9298% high |
| Half-width slope | 0.0870243942 | 7.4209% low |
| Top-hat entrainment alpha | 0.0827439230 | 2.1530% high |

The candidate **fails the 10% engineering screen**. Its maximum change from the
unconstrained normal-production profile on the assessment interval is only
**5.6823e-8 Uc**. Thus the consistent edge constraint does not materially change
the measured profile discrepancy. This supports treating the edge covariance
problem separately from the interior profile error; it does not prove a cause
for that error or validate finite-inertia droplet dispersion.

The largest edge/tolerance/mesh change in the assessed profile is **8.93e-12 Uc**;
the largest corresponding relative integral/width change is **1.96e-11**. The
covariance result refers to the model's stated strain approximation and sampled
solved domain, not an independent measured Reynolds-stress validation.

## Numerical failure history

- **30273647**: the manufactured edge test did not enter the active regime at
  its largest distance. Its synthetic epsilon amplitude was corrected.
- **30273648**: the test was active, but not sufficiently far into the asymptotic
  regime for the predicted convergence order. Testing s=1e-7,1e-8,1e-9 confirms
  the derived 0.7 order; no model coefficient or validation threshold changed.
- **30273650**: 12 checks passed; 13 transport solves converged, but strict
  refinements failed. Diagnostic **30273655** localized residual peaks to the
  limiter switch with mesh spacings **6.3e-15 / 3.7e-14**.
- **30273656**: an exact logarithmic coordinate did not remove the strict
  refinement failures. Its accepted coarse states are numerical initializers.
- **30273671 / 30273672**: explicitly matching smooth inactive and active regions
  qualified the corrected candidate; the control failed startup with either
  unconstrained or already constrained initial fields.
- **30273675 / 30273692**: grading the initial meshes failed stricter checks.
  Evaluating the physical residual directly in native coordinates did not by
  itself resolve those failures. All rejected states remain rejected.
- **30273699**: uniform initial meshes and deeper edge cutoffs again qualified
  the corrected candidate; the control remains rejected. Six state components
  are continuous at the solved switch, where Ncap/Nnatural=1. Original-coordinate
  residuals are evaluated on native quadrature nodes via the exact affine
  chain rule. Acceptance limits and physical equations remain unchanged.

The uncorrected control's unresolved startup limits the robustness of this
research solver. It is not evidence that its physical equations have no solution.

## Equations

The strain approximation is the same as the earlier normal-production
experiment: radial shear and all continuity-compatible normal strains, with
dV/dx omitted. The local momentum relation is `N F'=-I F/eta`. At fixed local
I,F,K, the covariance `R=2K/3 identity-2N S` is affine in N, even though S changes
with N. The analytic implementation intersects the nonnegative diagonal and
2-by-2 determinant intervals to select the largest positive N no greater than
`Cmu K^2/E`. Infeasible physical states are rejected. The analytic result is
checked against an independent concave eigenvalue maximization.

[Durbin (1996), equations 11–15](https://meshfree.pages.fraunhofer.de/docu/Gasdynamics/Durbin_realizabilityConstraint.pdf)
provides the principal-strain bound and calls for a limited timescale wherever
that timescale enters the equations. Following that prescription, this
experiment defines `T=N/(Cmu K)`, uses N in mean momentum and turbulent diffusion,
and uses `P=2N S:S`, epsilon production `C1 P/T`, and epsilon destruction
`(C2-C3 chi)E/T`. The signed Pope invariant is `chi=T^3 trace(Omega^2 S)`.
K destruction remains E. Combining these ingredients is this repository's
experimental extension, not a claim about the paper's validated applications.

When the constraint is inactive, the equations recover the previous
normal-production system. A separate test verifies the dimensional epsilon
transport residual at unrelated velocity and length scales. Trial Newton
states may fall back to the unconstrained viscosity when the PSD interval is
infeasible; accepted states must have this fallback inactive everywhere sampled.
The trial positivity/exponent safeguards must also be inactive.

## Active edge balance

Let `s=eta_edge-eta`. For the frozen sigma_k=1, finite limiting strain implies
that the active viscosity is proportional to K. Momentum and leading
convection/diffusion balances then give

```
N ~ n s, K ~ k s, F ~ f s, E ~ e s^1.3
n = I_edge / eta_edge
Qk ~ -I K, Qe ~ -I E
```

These replace the unconstrained F,K powers 10/7 and E power 13/7.
The natural viscosity scales as s^0.7, so the constraint dominates sufficiently
close to the edge. The limited T tends to a finite value. Epsilon production
is O(s), while its leading convection/diffusion terms are O(s^0.3); hence the
relative first source correction is O(s^0.7). The manufactured edge test checks
that limiting residual order, rather than requiring the asymptotic expressions
to solve the full finite-s equations exactly. These arguments use the frozen
sigma_k=1; they are not a proof for arbitrary transport constants.

The BVP imposes the new leading flux relations and viscosity slope at a sequence
of decreasing edge cutoffs. Qualification additionally requires an active
constraint at the final edge, finite positive monotone fields, PSD covariance,
inactive trial safeguards, collocation/boundary residual limits, and edge,
tolerance, and initial-mesh refinement. Numerical qualification must precede
source-width consistency and any separate independent assessment.

## Scope and reproduction

This remains a boundary-layer reduction, with omitted axial transport and
pressure matching. Positive-semidefinite covariance does not by itself define
a well-mixed finite-inertia dispersion process, or justify identifying RANS K
with unresolved LES energy. Even a qualified gas profile would leave those
physical validation and conservative coupling requirements outstanding.

```bash
sbatch --output=outputs/spray_closure_validation/constrained-kepsilon-%j.log \
  tools/solve_constrained_kepsilon.sbatch
```

The batch default uses accepted source-only constrained states from
`outputs/spray_closure_validation/constrained-kepsilon-30273656/` as initial
guesses. Their own full refinements failed; using them as guesses does not
qualify them. The CLI also supports the earlier unconstrained edge states.
Research outputs remain ignored. The prior
[local feasibility audit](jet-realizability-assessment.md) changes no solution;
this experiment instead attempts to solve the modified transport equations.

Reproduce the independent assessment after source qualification with:

```bash
sbatch --output=outputs/spray_closure_validation/constrained-profile-%j.log \
  tools/assess_constrained_profile.sbatch
```

The default assessment inputs are the historical **30273699** states. Outputs
are `constrained-kepsilon-30273699/` and `constrained-profile-30273701/` under
ignored `outputs/spray_closure_validation/`. No production model is enabled.
