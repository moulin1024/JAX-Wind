# Inhomogeneous well-mixed tracer component

`src/jaxwind/inhomogeneous_dispersion.py` addresses the tracer-limit consistency
requirement before connecting a dispersion model to finite-inertia droplets.
It is not enabled in the waterjet and is not a complete LES dispersion closure.

The [Thomson well-mixed criterion](https://doi.org/10.1017/S0022112087001940)
requires a tracer population initially distributed like the carrier fluid to
retain that distribution. In three dimensions that condition does not generally
select a unique stochastic model. The original
[Pozorski–Apte component](https://doi.org/10.1016/j.ijmultiphaseflow.2008.10.005)
implemented here was restricted to homogeneous turbulence; its manuscript
explicitly identifies missing stress-gradient and mean-gradient terms for more
general flows. Simply evaluating its variance locally in a nonuniform jet does
not establish well-mixed behavior.

## Derived drift and limits

The present derivation assumes stationary, constant-density, zero-mean,
isotropic Gaussian turbulence with positive, smooth variance `q(x)` in each
velocity component and correlation time `T(x)>0`. Position and physical tracer
velocity obey an Itô system:

```
dx = u dt
du = a dt + sqrt(2 q/T) dW

a = -u/T + grad(q)/2 + u [u dot grad(q)]/(2q)
p(x,u) = constant * (2 pi q)^(-3/2) exp(-|u|^2/(2q))
```

This is our explicit derivation of one admissible isotropic drift, not a claim
that the general 3D closure is unique. The OU part of the Fokker–Planck equation
annihilates the local Gaussian velocity density. For the remaining drift `a_g`,

```
div_u(a_g p) = [3/(2q) - |u|^2/(2q^2)] (u dot grad(q)) p
            = -u dot grad_x(p)
```

so spatial transport and gradient drift cancel. Uniform concentration with the
specified local velocity PDF is stationary. An independent automatic-
differentiation test evaluates the full physical phase-space Fokker–Planck
residual with variance varying in all three spatial directions; dropping the
gradient drift fails that identity.

For numerical integration use `w=u/sigma(x)` and `sigma=sqrt(q)`. Because `x`
has finite variation, the change of variables cancels the quadratic velocity
term exactly, giving

```
dx = sigma(x) w dt
dw = [grad(sigma) - w/T] dt + sqrt(2/T) dW
```

The integrator uses an exact normalized OU half-step, explicit-midpoint transport,
and a second OU half-step. The two noise arrays are independent. There is no
particle resampling, density correction after the step, clipping, or artificial
variance floor. The caller supplies the consistent variance and its gradient,
random samples and boundary handling. Explicit transport still requires timestep
convergence checks. A deterministic refinement test verifies second-order
transport accuracy; the ensemble test separately checks stationarity at three
timesteps.

The model does **not** cover finite particle inertia, mean shear, anisotropic
Reynolds stress, non-Gaussian velocities, time-varying turbulence, variable
density, zero-variance interfaces, walls or two-way energy exchange. In
particular, replacing tracer velocity by droplet velocity in the gradient term
is not a justified finite-inertia extension. The covariance/energy input must
come from a separate physical model, not an unvalidated conversion from AMD
viscosity or from all the energy in the inventory merger.

## Prespecified numerical benchmark

The [protocol](well_mixed_protocol.json) defines a periodic 1 m box,
`q=1+0.8 cos(2 pi x)` in `(m/s)^2`, `T=0.05 s`, and a 2 s integration.
Positions initially sample a uniform distribution; `w` samples `N(0,I)`.
Two fixed seeds and timesteps 0.01, 0.005 and 0.0025 s are used. Diagnostics are
32-bin concentration, conditional normalized-velocity means and variances in
all three components, and the first position Fourier moment. These test more
than a global velocity variance, which could hide spatial accumulation.

Fixed numerical screens are: maximum density error 7%, conditional mean 0.08,
conditional variance error 0.09, and Fourier amplitude 0.012. These are Monte
Carlo screens, distinct from the 10% experimental profile criterion. Their
purpose is to detect systematic drift at the specified sample count, not assign
measurement confidence intervals.

A negative control evolves physical velocity with a local OU model and no
inhomogeneous drift, using symmetric free-flight/OU/free-flight integration.
It has the same variance field and finest timestep. Its density error must
exceed 15%; otherwise this benchmark has not demonstrated sensitivity to the
missing drift. It is a deliberately uncorrected control, not a competing
physical model eligible for production use.

## Reproduction

```bash
sbatch --output=outputs/spray_closure_validation/well-mixed-%j.out \
  tools/verify_well_mixed.sbatch
```

The script runs four numerical tests and the ensemble benchmark on `gpudev`.
Reports, plots, JUnit results and source hashes stay under ignored
`outputs/spray_closure_validation/well-mixed-JOBID/`. The report records every
seed and timestep; a failed screen causes a nonzero job exit.

### Initial ensemble and refinement

Job **30272108** passed all four unit/identity tests. With **131,072 tracers**,
all six corrected final-state screens passed. Maximum density errors were
2.44–4.03%; the uncorrected controls reached 169.70% and 171.70% excess density.
However, the overall screen failed: seed 21's initial conditional variance error
was **0.09974**, exceeding the fixed **0.09** bound before integration. This
initial-sampling failure is retained in its original report.

The protocol was refined to **262,144 tracers**, preserving both seeds, all
three timesteps and every tolerance. This reduces sampling error; it does not
repair the result by changing a seed, resampling particles or widening a gate.
Both sample sizes must be reported when assessing this verification.

### Refined result

Gpudev job **30272114** completed successfully on the GPU. All **four numerical
tests passed**, and every initial/final corrected ensemble screen passed at
262,144 particles. The six final density errors ranged from **1.98% to 3.15%**;
maximum conditional normalized mean and variance errors were **0.0412** and
**0.0577**, respectively. Maximum first Fourier amplitude was **0.003616**.
The two uncorrected controls reached **171.04% and 173.07%** density error.

The complete report and matching source hashes are in
`outputs/spray_closure_validation/well-mixed-30272114/report.json`, with the
comparison in `report.md` and `density.png`. The original lower-count report
remains at `well-mixed-30272108/report.json` and includes its frozen protocol.
Sampling scatter dominates the small differences between the three timesteps;
these results do not establish an empirical weak-convergence order for the
stochastic integrator. They do demonstrate stationarity within the fixed
screens, supported independently by the analytical PDF identity.

This completes a defined numerical tracer-limit check, not the active goal's
full inhomogeneous finite-inertia dispersion requirement. The measured-source
qualification, general fluid-seen velocity model, conservative LES coupling and
waterjet assessment remain outstanding.
