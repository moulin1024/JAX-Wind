# Source-state stress, pressure and realizability assessment

Status: a diagnostic of the independently computed gas profiles, **not a
profile correction or a validated dispersion/coupling model**. The best held-out
profile error remains 19.881%, failing 10%; its decay coefficient remains 13.930%
high. No holdout values were read in this audit.

Two results change the next modeling step:

- The thin-jet stress/pressure flux remainder for the best diagnostic is
  **-0.7414% of mean momentum**. It does not provide a direct momentum-accounting
  explanation for a 13.9% high centerline coefficient. At fixed shape its sign
  would require a slightly higher amplitude to retain the same inlet momentum.
- The raw Boussinesq covariance develops negative eigenvalues inside a very thin,
  numerically solved edge region. Its integrated turbulent-energy fraction is
  tiny, but that tensor cannot directly serve as a particle-velocity covariance.
  This is not evidence that the edge defect causes the larger mean-profile error.

## Model and independent checks

The [frozen momentum protocol](kepsilon_momentum_audit_protocol.json) uses the
normal-stress momentum convention in
[Panchapakesan and Lumley (1993), equation (7), pp. 205–206](https://doi.org/10.1017/S0022112093000096).
Write dimensionless stresses R_ij=<u_i u_j>/(A/x)² and pressure Pi=(p-p_ambient)/(rho (A/x)²).
The thin-jet radial equilibrium and section integral give

```
Pi(eta) = -Rrr(eta) + integral_eta^edge (Rrr-Rtt)/s ds
I_momentum = integral eta [F² + Rxx - (Rrr+Rtt)/2] d_eta.
```

This pressure approximation omits mean radial inertia and axial radial-stress
transport. A full finite-domain RANS/LES pressure match remains a separate task.
The stress/pressure remainder is a flux contribution, not SGS kinetic energy,
a momentum sink, or a new gas inventory.

The audit constructs the exact gradient of the existing self-similar velocity,
and also the version that omits dV/dx in shear, then computes
R=(2/3) K identity - 2 N S. Both versions retain the continuity-consistent normal
strains. This tests the stress implied by extending the viscosity ansatz to the
normal momentum terms; the original mean boundary-layer equation did not solve
those terms. No eigenvalue clipping, covariance projection, profile renormalization
or new coefficient is introduced.

Gpudev **30273537** passed four independent checks in 0.86 s:

1. Cartesian finite differences of an analytic three-dimensional axisymmetric
   velocity agree with the similarity-gradient transformation, including the axis.
2. Isotropic turbulence pressure cancels its normal-stress section flux.
3. An analytic anisotropic stress profile agrees with the independently integrated
   pressure and section identity, including its finite-boundary contribution.
4. The tensor transforms correctly under rotation and retains a known negative
   eigenvalue instead of repairing it.

## Momentum contribution

All eight previously qualified source-state combinations were retained. The
4097-point section results are:

| Coefficient convention | C3 | Production | Stress/pressure remainder / mean momentum |
|---|---:|---|---:|
| 1978 printed | 0 | shear | -0.7609% |
| 1978 printed | 0 | normal added | -0.8062% |
| 1978 printed | 0.79 | shear | -0.6219% |
| 1978 printed | 0.79 | normal added | -0.6440% |
| 1990 convention | 0 | shear | -0.8684% |
| 1990 convention | 0 | normal added | -0.9269% |
| 1990 convention | 0.79 | shear | -0.7124% |
| 1990 convention | 0.79 | normal added | **-0.7414%** |

The largest 2049-to-4097 quadrature change is **3.43e-11 of mean momentum**.
Changing the solver edge cutoff from delta=0.01 to 0.005 changes the remainder
by at most **1.11e-7 of mean momentum**. Independent pressure integration agrees
with the section normal-stress identity to **6.29e-10 of mean momentum**.
These errors are much smaller than the diagnosed flux term.

The negative remainder is physically allowed as a pressure/stress flux. It does
not mean negative turbulent energy. It also cannot be substituted for the positive
stress/pressure remainder inferred in an earlier, separate measured-profile
calibration. The present calculation does not solve the added momentum terms;
therefore it does not bound how much a consistently solved full model might
reshape the velocity profile. No adjusted B, alpha or held-out score is reported.

## Separate appended-tail behavior from the solved covariance

The initial uniform-grid audit found negative eigenvalues for the corrected
models only in the appended leading-power tail. That alone was insufficient to
claim a failure inside the computed solution. The separate
[edge protocol](kepsilon_edge_audit_protocol.json) therefore moved the numerical
boundary closer to the edge, at delta=0.0025, 0.001, 0.0005, 0.00025 and 0.0001,
with tolerance 1e-8 and 800 initial nodes. It reports computed and appended
regions separately and uses an edge-graded diagnostic grid.

Initial job **30273560** failed the first control's delta=0.0001 nonlinear solve.
It had extrapolated the previous logarithmic fields as polynomials. Final
**30273563** instead extended only the *starting guess* with the already-derived
edge powers; equations, coefficients and acceptance gates were unchanged.
It again passed the four kernel tests and retained 19 converged edge solves.
The corrected shear-only case still failed at delta=0.0001 through mesh-node
exhaustion (residual 10.829), so the job correctly exited 2 and that solution
was rejected. Its converged delta=0.00025 result is retained. Other cases reached
0.0001. Source hashes were checked separately after this partial-job exit and
all matched.

For the complete self-similar gradient, the last accepted result in each case is:

| 1990-convention case | Last delta | Minimum eigenvalue in solved domain | Modeled K fraction in violating solved region |
|---|---:|---:|---:|
| Uncorrected, shear | 0.0001 | -4.8342e-8 | 2.9172e-9 |
| Uncorrected, normal added | 0.0001 | -4.8892e-8 | 2.6957e-9 |
| Corrected, shear | 0.00025 | -4.9794e-8 | 1.4178e-8 |
| Corrected, normal added | 0.0001 | **-4.7814e-8** | **1.3759e-8** |

Eigenvalues use the common centerline-velocity-squared scale. The best diagnostic
also has a minimum eigenvalue/local-K ratio of **-0.7118** inside the solved
region. Doubling its diagnostic grid changes the minimum eigenvalue by
**9.10e-15** and the violating energy fraction by **1.17e-11**. Both gradient
conventions detect violations in the solved domain; the result is not limited
to dV/dx reconstruction or appended-tail extrapolation. No negative eigenvalue
has been removed from these reports.

There is a structural reason to investigate the edge treatment. In the model's
leading edge balance, K~s^(10/7) while N~s, with s=eta_edge-eta. Entrainment gives
Srr approaching a finite positive value at the edge. Consequently
Rrr/K=2/3-2(N/K)Srr can become negative as N/K~s^(-3/7) grows. This inference
explains why positive K and epsilon do not by themselves establish a realizable
stress tensor. It does not establish a new physical edge model.

## Consequence for the persistent goal

The current gas profiles remain failed independent physical candidates. Their
raw stress tensor needs a justified edge/realizability treatment before it can
seed anisotropic droplet dispersion. Replacing it by an arbitrary clipped tensor
would change momentum and production without a validated closure. A proposed
filter or handoff treatment must account for the displaced mass, momentum and
energy and be tested against independent observations.

The small integrated edge defect is separate from the 19.881% mean-profile
failure. The next work should address a consistently realizable stress/dispersion
model and conservative LES energy ownership, while continuing measured-inlet
qualification. This audit supplies neither a calibrated SGS-energy source nor
permission to enable the production waterjet model.

## Reproduction

- `sbatch tools/audit_jet_stress.sbatch` runs the tensor/pressure tests and the
  eight-case momentum audit.
- `sbatch tools/audit_kepsilon_edge.sbatch` runs the edge follow-up. The documented
  shear-only final-level rejection currently gives exit status 2; inspect the
  retained case reports rather than interpreting job termination as a physical pass.
- Set `JAXWIND_KEPSILON_SOLUTIONS` to use another separately qualified source run.

Reports and states remain under ignored output directories
`jet-stress-30273537`, `kepsilon-edge-30273560` and `kepsilon-edge-30273563`
beneath `outputs/spray_closure_validation/`. No research data, logs or generated
states are added to version control.

The subsequent [momentum-compatible covariance feasibility audit](jet-realizability-assessment.md)
checks whether a realizability constraint can also satisfy the existing local
momentum relation. It does not change the independently assessed profile.
