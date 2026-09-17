# Source-frozen turbulence-profile candidate

Status: equation verification, **not a qualified profile or LES closure**.
The production waterjet settings are unchanged. This page records the original
1978-printed convention. The subsequent [1990-convention assessment](pope-profile-assessment.md)
improves the best independent diagnostic to 19.881%, still above the unchanged
10% screen; the original printed-convention source failures are retained.

## Why this candidate

The Panchapakesan–Lumley primary paper explicitly reports agreement with the
Hussein group's mean profile except near the edge. Its published normalization
does not explain the outer-profile discrepancy. Transferring an independently
fitted Gaussian mixture improved the integral model but did not meet the local
profile criterion. A variable turbulent viscosity is a physical alternative to
another empirical profile fit.

The candidate uses the high-Re k-epsilon equations and vortex-stretching
correction in [Pope (1978), pp. 279–280](https://tcg.mae.cornell.edu/pubs/Pope_AIAAJ_78.pdf).
The original constants are Cmu=0.09, C1=1.45, C2=1.90, sigma_k=1,
sigma_e=1.3. The paper's corrected variant uses C3=0.79. Its printed
half-width slopes are 0.125 without the correction and 0.086 with it.
These are source-reproduction targets, not new experimental validation.
The correction was calibrated in that older study; it is not a parameter-free
prediction. The historical constants must not silently become 1.44/1.92.

The [frozen protocol](kepsilon_profile_protocol.json) specifies constants,
source SHA256, numerical qualification and the unchanged holdout. The PDF is
stored only under ignored `outputs/spray_closure_validation/pope_1978/`.
Neither the equation module nor its tests read the Hussein reference.

## Similarity reduction

Write z=x-x0, eta=r/z, U=A F/z, V=A G/z, k=A² K/z² and epsilon=A³ E/z⁴.
Set F(0)=1. Define I=int_0^eta s F(s) ds and N=Cmu K²/E, so nu_t=A N.
Continuity gives G=eta F-I/eta. Integrated mean momentum gives
eta N F'=-I F; no measured entrainment or width is imposed.

With Qk=eta N K'/sigma_k and Qe=eta N E'/sigma_e, the six equations are

```
I'  = eta F
F'  = -I F/(eta N)
K'  = sigma_k Qk/(eta N)
Qk' = -2 eta F K - I K' - eta N F'^2 + eta E
E'  = sigma_e Qe/(eta N)
Qe' = -4 eta F E - I E' - eta C1 (E/K) N F'^2
      + eta (C2-C3 chi) E²/K
chi = (K/E)^3 F'^2 (F-I/eta²)/4
```

The invariant follows from tr(rotation² strain), retaining the leading radial
axial-velocity gradient in rotation and the continuity-consistent radial and
azimuthal normal strain. It changes sign in the entraining outer flow and is
not clipped. At the axis I=Qk=Qe=0, F'=K'=E'=0 and G/eta tends to F/2.
Positive finite K and E are required; invalid states are rejected.

This is an incompressible, steady axisymmetric boundary-layer approximation:
molecular viscosity, axial turbulent diffusion and axial normal-stress gradients
are omitted. It is not a full RANS solver. Source spreading rates must be
reproduced before assuming the reduction is sufficient for profile assessment.

For a numerically qualified, decaying solution, I1=int eta F and
I2=int eta F² determine B=1/sqrt(8 I2) and alpha=2 B I1 under the existing
mean-momentum/top-hat nozzle convention. The half-width is the root F=1/2.
Any stress/pressure correction to that momentum convention must be a separate,
explicit model decision. No such correction is fitted here.

## Equation verification and solver qualification

Gpudev **30273381** passed five checks in 0.07 s. Final **30273421** repeated
all five successfully in 0.06 s after fixing default-suite import discovery
and making the numerical residual gates explicit; source hashes were verified:

- Exact constant-viscosity round jet F=(1+a eta²)^(-2), N=1/(8a).
- Manufactured K/E profiles checked against the original dimensional advection,
  radial diffusion, production and dissipation equations, both C3 variants.
- Direct 3-by-3 tensor contraction for the signed correction, including the
  negative outer region.
- Rejection of invalid turbulence states and inconsistent axis fluxes.

Run `sbatch tools/verify_kepsilon_similarity.sbatch` for these checks. They
verify the reduction and its implementation, not existence, uniqueness or
accuracy of a computed similarity solution.

Exploratory solvers and all their logs remain in the ignored
`outputs/spray_closure_validation/kepsilon_prototype/` directory. Gpudev
**30273238** (log-state collocation) and **30273285** (squared-radius flux
formulation) failed through singular/nonphysical nonlinear iterations. A
positive bounded least-squares discretization, **30273334**, reached its 2000
evaluation cap with scaled maximum residuals 0.02465 and 0.009436. Its apparent
widths are not qualified predictions. A subsequent dense-Jacobian trial
**30273388** failed before solving because its seed filename was wrong; that
harness error was corrected for **30273399**. That solver terminated at a
least-squares stationary point, but maximum scaled residuals remained **0.02150**
and **0.008514**. Optimizer success is not equation convergence; neither result
is admissible. No holdout profile was evaluated.

The model's leading zero-turbulence-edge balance suggests compact-support
exponents m=1/(2 sigma_k-sigma_e), p=sigma_k m, q=sigma_e m for F, K and E.
A free-edge collocation trial using that asymptotic relation, **30273417**, also
failed its nonlinear solve. This is a numerical failure, not evidence against
the published model. The next approach is positive downstream marching toward
a self-similar state, followed by mesh, inlet-memory and outer-boundary studies;
it must reproduce the source spreading rates before holdout assessment.
None of these exploratory outputs is a physical pass or a reason to adjust a
published coefficient.

The subsequent [downstream-marching assessment](kepsilon-marching.md) verifies
the moving mean-flow front but finds a persistent source-rate discrepancy.
It supersedes the prospective marching plan above without removing these failed
collocation attempts.

## Limits for spray coupling

Even a successful mean profile would leave unresolved energy and particle
transport work. RANS k is total modeled turbulence; assigning it directly to
unresolved LES energy could double count resolved fluctuations. Filter-aware
energy ownership, finite-inertia inhomogeneous dispersion, ambient withdrawal
and conservative handoff remain separate requirements. Neither numerical
mixing heat nor a fitted mean-profile width supplies these quantities.
