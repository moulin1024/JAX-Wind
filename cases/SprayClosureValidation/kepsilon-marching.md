# Downstream-marching verification and source-rate discrepancy

Status: a verified mean-flow marching kernel and a source-only turbulence
investigation of the 1978-printed coefficients. **Neither original variant
passes source reproduction.** This page records the pre-holdout marching
checkpoint. A later [steady-solver and 1990-convention assessment](pope-profile-assessment.md)
reaches 19.881% held-out profile error, still failing 10%. Production waterjet
defaults are unchanged.

## Conservative streamfunction coordinate

Define psi=int_0^r s U ds at constant density. The radial geometry is
r²=2 int_0^psi 1/U dpsi, and mean momentum becomes

```
dU/dx|psi = d/dpsi [D dU/dpsi],  D = r² nu_t U.
dk/dx|psi = d/dpsi [D/sigma_k dk/dpsi] + (P-epsilon)/U.
depsilon/dx|psi = d/dpsi [D/sigma_e depsilon/dpsi]
                 + [C1 epsilon P/k - (C2-C3 chi) epsilon²/k]/U.
```

The finite-volume integral int U dpsi is axial momentum divided by 2 pi rho.
`tools/jet_streamfunction.py` uses implicit diffusion and destruction with
explicit nonnegative production. Its matrix has nonpositive off-diagonal
entries and positive row sums. It does not clip or renormalize solutions.
The outward boundary transfer is returned explicitly. This supports positivity
of the discrete update, not positivity/realizability of every physical closure.

An initially attempted harmonic interpolation effectively pinned the nearly
stationary jet edge. Standard run **30273437** retained momentum but developed
an unphysical flat core and meaningless interpolated width; corrected run
**30273438** also lost finite positive states. These failures show why the
momentum integral alone is insufficient. Arithmetic face interpolation permits
entrainment and is the verified implementation. The initial code/logs remain
under ignored `outputs/spray_closure_validation/kepsilon_march/`.

## Independent moving-front check

For constant viscosity nu=1/8 and A=a=1, the exact round jet in this coordinate
is U=x^-1 [1-2 psi/x]^2 for psi<x/2 and zero otherwise. This is the same physical
solution U=x^-1 [1+(r/x)^2]^-2, whose finite mass-flux edge moves downstream.
The audit initializes exact cell averages at x=1, marches to x=2, and compares
cell velocities and radial half-width, with a stated small ambient coflow.

Gpudev **30273458** passed six kernel tests: a stiff positive reaction/diffusion
budget, constant-stream geometry and equilibrium, front penetration, and three
negative-rate rejections. Source hashes were checked. The separate analytic
audit produced:

| Cells | Max velocity error / exact Uc | Relative half-width error |
|---:|---:|---:|
| 128 | 1.69294e-4 | 2.65750e-4 |
| 256 | 5.38827e-5 | 8.54906e-6 |
| 512 | 3.04053e-5 | 6.73048e-6 |
| 1024 | 1.60621e-5 | 1.01547e-5 |

The marching increment was 0.05/cell-count. On 512 cells, halving it reduced
velocity error to **1.36165e-5**. Reducing ambient speed from 1e-8 to 1e-9 gave
**3.04080e-5**, a negligible change here. All states stayed positive and monotone;
maximum absolute momentum-budget error was **1.744e-14**. The half-width error
is not monotone under refinement because spatial and time errors compete; do
not infer an order from that column. This verifies the mean-flow numerical
method, not the turbulence model or a spray experiment.

Reproduce with `sbatch tools/verify_jet_streamfunction.sbatch`. Reports are
under ignored `outputs/spray_closure_validation/jet-streamfunction-30273458/`.

## Turbulence source-rate assessment

`tools/march_kepsilon_profile.py` preserves the independently fixed coefficients
and shear-production approximation in the [profile protocol](kepsilon_profile_protocol.json).
It reads no holdout. The inlet is an explicitly synthetic Gaussian mean flow,
with arbitrary positive k/epsilon profiles; it is not a measured spray source.
A small coflow regularizes the streamfunction coordinate. Width growth is fitted
only to the computed downstream history over x>=0.4*x_end, including a freely
computed virtual origin. Neither source nor holdout widths set that coordinate.

The exploratory 14-case matrix **30273461** separated cell count, marching
increment, ambient speed, outer mass-streamfunction boundary and inlet k level.
At a fixed increment 0.00075, changing 1000 to 2000 cells changed standard
spreading from 0.1116565 to 0.1116581 and corrected spreading from 0.0791405 to
0.0791335. On 1000 cells at increment 0.0015, a tenfold coflow change, doubled
outer boundary, and doubled initial k with epsilon multiplied by 2^1.5 each
changed the slope by less than **0.02%**. The coarse corrected 500-cell case
failed, and is retained. Its intermediate momentum drift is not accepted.

The versioned marcher also checks cumulative boundary-adjusted momentum before
accepting a step, rejects a budget error above 1e-10, and retains the last
accepted state on a rejected numerical transaction. Its source-only report
separates successful integration from the source-reproduction result.

Final gpudev **30273474**, using the verified kernel, tightened the marching
increment further on 1000 cells. Every run retained finite positive states and
verified source hashes:

| Model | Increment | Computed spreading | Published spreading | Relative source error |
|---|---:|---:|---:|---:|
| Standard | 0.000375 | 0.11183062 | 0.125 | 10.536% |
| Standard | 0.0001875 | 0.11191951 | 0.125 | 10.464% |
| Pope correction | 0.000375 | 0.07930559 | 0.086 | 7.784% |
| Pope correction | 0.0001875 | 0.07938884 | 0.086 | 7.687% |

Both fail the frozen **2% source-reproduction gate**. Their last step-refinement
changes are much smaller than the source deficit, but complete profile residual,
shape-convergence and downstream-development qualification is still outstanding.
These source-rate errors must not be confused with the unchanged 10% Hussein
profile gate or the prior best held-out profile error of 26.018%.

Reproduce with `sbatch tools/assess_kepsilon_source.sbatch`; final reports and
states are under ignored `outputs/spray_closure_validation/kepsilon-source-30273474/`.

## Omitted-production audit and next work

Pope's equation (5) defines production through the complete stress/mean-gradient
contraction. The current reduction retains shear production only. The separate
[production audit protocol](kepsilon_production_audit_protocol.json) froze a
diagnostic addition of normal-strain production, with all coefficients unchanged
and no holdout access. Direct explicit evaluation in the nearly stagnant ambient
region proved numerically unsuitable: **30273475** failed all four trials within
three steps, including momentum drift and overflow. No resulting width is used.
Adding only those terms would still be an intermediate approximation, since
streamwise radial-velocity gradients, axial diffusion and axial stress transport
remain omitted. This failure does not establish that the terms are unimportant.

Next work should audit the primary model's complete gradient/production treatment
and solve those terms consistently, or use an independently documented alternative
model. Do not retune a coefficient to remove this source deficit. RANS k still
requires a filter-aware ownership model before it can supply LES unresolved energy
or a finite-inertia dispersion model. The measured-droplet and waterjet gates
remain open.
