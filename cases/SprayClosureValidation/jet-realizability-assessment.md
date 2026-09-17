# Momentum-compatible covariance feasibility

Gpudev **30273619** passed **11 checks** (seven new feasibility checks and four
existing stress/gradient/pressure checks). At every sampled violating point in
the four accepted source-only edge states, a positive viscosity reduction could
make the local modeled covariance positive semidefinite while satisfying the
boundary-layer momentum relation. This is a local algebraic result. No profile,
turbulence transport solution, held-out score, or production model was changed.
The independent profile error remains **19.881%**, above the **10%** criterion.

## Source and scope

[Durbin (1996), equations 11–15](https://meshfree.pages.fraunhofer.de/docu/Gasdynamics/Durbin_realizabilityConstraint.pdf)
derives the incompressible Boussinesq bound
`N <= K / (3 lambda_max(S))` from nonnegative principal Reynolds stresses.
The paper applies a limited timescale wherever that timescale enters its model;
a viscosity-only change is not that complete model. Its illustrated application
is stagnation flow, not independent round-jet or droplet validation. This audit
uses the exact eigenvalue bound without an empirical multiplier. The downloaded
paper has SHA256
`f3862c55bd5b3fd81dd0fbb50a2202294eb07c76e0e9785f01d54b4667e8c676`;
the PDF and provenance are ignored research outputs.

The [frozen protocol](jet_realizability_protocol.json) assesses all four latest
accepted edge states from job **30273563**, with both gradient diagnostics. It
retains the rejected corrected shear-only delta=0.0001 solve and uses that
case's last accepted delta=0.00025 state. Other cases use delta=0.0001. No held-out
reference is read. Computations exclude the appended asymptotic tail and retain
only solved-domain points with K greater than 1e-12 of its peak.

## Why limiting viscosity alone is insufficient

The existing momentum equation requires `N F' = -I F / eta`. Thus its strain
changes when N changes. At fixed local I,F,K, define `H=I F/eta`, `T=F-I/eta^2`
(with axis limit T=F/2). The covariance has diagonal components

```
Rxx = 2K/3 + 2NF - 2 eta H
Rrr = 2K/3 - 2N(F-T) + 2 eta H
Rtt = 2K/3 - 2NT
```

The radial-shear approximation gives `Rxr=H`; the complete self-similar gradient
gives `Rxr=H(1-eta^2)+N eta F`. Both preserve trace `2K`. These identities are
independently checked against the existing velocity-gradient calculation at
three positive viscosity fractions, including the axis.

Consequently, with `N=t N_original`, the covariance is affine, `R=A+t B`.
The smallest eigenvalue is concave in t. A bounded maximization and bracketed
upper root find the largest feasible positive fraction in `(0,1]`. Inputs are
normalized by local K. Near-zero maxima are reported as indeterminate, rather
than incorrectly rejecting a degenerate semidefinite interval. Tests include
an interval that excludes zero, an impossible state, unchanged states, a
fixed-momentum shear counterexample, invalid inputs, and a degenerate interval.

## Results

The table uses the complete-gradient diagnostic and the finer diagnostic grid.
All eight case/gradient combinations have zero infeasible or indeterminate
sampled points. The smallest post-constraint eigenvalue across all cases is
-3.56e-15 K, consistent with numerical root tolerance.

| Source state | Smallest feasible N/N_original | Minimum eigenvalue/K after an old-gradient cap and momentum recomputation |
| --- | ---: | ---: |
| Standard, shear production | 0.683334 | -0.006922 |
| Standard, normal production included | 0.692717 | -0.003112 |
| Pope corrected, shear production | 0.695888 | -0.007478 |
| Pope corrected, normal production included | 0.485405 | -0.004480 |

For the last row, only **1.3770e-8** of the solved-domain integral of eta K lies
in the affected region. The reduction of viscosity there is locally substantial,
but that tiny region is not evidence for the cause of the much larger radial
profile error. The naive cap reduces the covariance defect but does not remove
it after momentum is enforced.

Doubling the requested edge-graded sample count from 8193 to 16385 leaves each
minimum feasible fraction unchanged; both grids include the same outer solved
point. This is sampling stability on fixed input states, **not** convergence of
a modified closure or proof of feasibility at every continuum point. The
radial-shear-only diagnostic gives the same qualitative conclusion.

## Required next step

The locally recomputed F' is not the derivative of a newly solved F profile.
A candidate must incorporate the constrained timescale consistently into the
mean and turbulence equations, derive compatible edge conditions, and repeat
numerical and source gates before another independent assessment. Reusing the
old unconstrained free-edge powers without checking their balance would be
unjustified. The complete-gradient diagnostic also does not make the existing
boundary-layer momentum equation a full RANS model. No covariance from this
audit is supplied to droplet transport or LES.

Outputs are under ignored
`outputs/spray_closure_validation/jet-realizability-30273619/`; all recorded
source checksums passed at completion. Reproduce using:

```bash
sbatch --output=outputs/spray_closure_validation/jet-realizability-%j.log \
  tools/audit_jet_realizability.sbatch
```

The default input is the historical edge-audit directory; set
`JAXWIND_EDGE_SOLUTIONS` to use another independently qualified source-only audit.
See the preceding [stress/pressure assessment](jet-stress-assessment.md) and
[current independent profile assessment](pope-profile-assessment.md).

The [consistent constrained-timescale follow-up](constrained-kepsilon.md)
re-solves transport with a physical covariance constraint. The qualified
corrected candidate still fails the independent profile gate at 19.8810%;
the uncorrected control remains numerically rejected.
