# Independent assessment of coarse-grid spray-closure components

Current acceptance is **20%**, explicitly revised by the user on 2026-09-17.
The qualified 19.8810% candidate now passes that agreement screen. See the
[re-assessment](acceptance-20-percent.md); older 10% results below are historical.

Status: component implementation and verification, **not a validated production
spray model**. The waterjet benchmark does not enable these components.

## What is implemented

`src/jaxwind/spray_closure.py` contains two separate building blocks:

1. A steady integral round-jet entrainment model with exact section-to-section
   advancement and Gaussian reconstruction. It conserves axial mean-flow
   momentum and accounts for the mass, species and enthalpy of ambient intake.
   Analytic face integrals preserve the integrated fluxes even when the jet
   radius is smaller than a receiving cell. Finite-plane truncation is reported
   as missing flux rather than silently renormalized.
2. The homogeneous residual-fluid-velocity Langevin update from
   [Pozorski and Apte (2009)](https://doi.org/10.1016/j.ijmultiphaseflow.2008.10.005),
   equations 4.1–4.6. It includes different parallel/perpendicular correlation
   times for particles crossing turbulent eddies, uses relative velocity, and
   takes explicit unresolved kinetic energy and filter width. The caller owns
   the random state. This is an auxiliary velocity update, not a complete
   particle integrator or a two-way coupled spray model.

These are separate literature components, not a published combined closure.
The Gaussian reconstruction is a different profile convention from the top-hat
spray proposed in `doc/design/spray-framework-proposal.md`; it does not silently
replace that proposal or its provisional coefficients.

## Conservative coupling operations

The separate [gas ownership and transfer contract](coupling.md) now implements
bounded ambient withdrawal, nonlocal shell intake, and inventory handoff with
explicit unresolved kinetic-energy accounting. These operations are not yet
connected to the LES mass/pressure and parcel exchange steps. They do not fix
the failed profile criterion or establish physical validation.

The opt-in carrier/parcel verification path also includes
[moving residence exchange and outflow](moving-parcels.md),
[explicit particle body forces](particle-body-force.md), and
[prescribed injection with conservative boundary accounting](parcel-injection.md).
These components preserve separate phase, external-force and injection budgets,
but do not yet qualify measured inlet distributions or complete the physical
entrainment/dispersion model.

The [compatible carrier energy extension](carrier-energy.md) adds an opt-in
energy balance to this path: numerical mixing heat, compatible pressure
conversion and explicit wall export are included without assigning numerical
losses to physical SGS k. Its energy approximation is stated separately from
experimental validation and physical turbulence closure.

The [source-frozen k-epsilon profile candidate](kepsilon-profile.md) explores
variable turbulent viscosity. The [downstream marcher](kepsilon-marching.md)
passes an independent moving-front check. A subsequently documented
[1990 coefficient convention](pope-profile-assessment.md) passes the independent
source-rate and numerical gates, but fails the held-out profile screen: the best
normal-strain diagnostic reaches **19.881%** local error, improved from 26.018%,
and its centerline decay coefficient remains **13.930% high**. The 10% threshold
and production defaults are unchanged. The subsequent
[stress/pressure audit](jet-stress-assessment.md) finds a small negative momentum
flux remainder and a covariance-realizability defect in the thin computed edge.
It supplies no profile rescaling or particle-dispersion covariance repair.

## Independent entrainment comparison

The sole entrainment coefficient is fixed at equivalent-top-hat `alpha=0.08`
from [Ricou and Spalding (1961)](https://doi.org/10.1017/S0022112061000834).
The held-out experiment is
[Hussein, Capp and George (1994)](https://turbulence-online.com/Publications/Papers/HCG94.pdf).
Its LDA/FHW results give the velocity-decay coefficient `B_u=5.8`, velocity
half-width slope `S=0.094` (printed page 44), and top-hat entrainment coefficient
`alpha=0.081` (page 47). The normalized radial-profile fit in Table 4 (page 52) is also tested over
`0 <= r/(x-x0) <= 0.2`, with a 10% local relative-error screen. Its coefficients
are held-out targets, never model inputs. Good summary slopes alone do not
satisfy this profile check.

The last page explicitly attributes `0.08` to the earlier
Ricou–Spalding direct measurement. The experiment also discusses omitted
second-order contributions to momentum; a mean-flow-only model need not recover
its centreline coefficient exactly.

The small [reference file](reference.json) records the source, definitions and
roles. These are rounded published fit summaries, not raw measurements.
Experimental uncertainty is unavailable in this extraction. The screening
tolerance is 10% relative error, an engineering criterion, not a confidence
interval. The comparison is retrospective and not blind; the observables are
correlated. No held-out coefficient, measured virtual origin or Montazeri
waterjet temperature is used to fit this implementation.

For mass flux `m`, axial momentum flux `J`, constant density `rho` and distance
`s`, the integral model is

```
dm/ds = 2 alpha sqrt(pi rho J)
dJ/ds = 0
dH/ds = h_ambient dm/ds
dm_species/ds = Y_ambient dm/ds

U(r) = Uc exp(-r²/b²)
Uc = 2 J/m
b = m/sqrt(2 pi rho J)
b_half = sqrt(log(2)) b
```

`b` is the Gaussian e-folding radius; equivalent top-hat radius is `sqrt(2)*b`
and top-hat speed is `Uc/2`. The Gaussian entrainment coefficient is therefore
`alpha/sqrt(2)`, not `alpha`. For a normalized top-hat nozzle these equations
predict `B_u=1/(2 alpha)=6.25`, `S=sqrt(2 log(2))*alpha=0.094193` and normalized
mass-flow slope `4 alpha=0.32`. Validation measures these slopes from the
computed section sequence instead of fitting a virtual origin to the data.

The exact integral advance is evaluated with axial intervals 8, 4, 1 and 0.25
nozzle diameters. Coarse-plane flux integration uses Gaussian-radius/cell-size
ratios 0.25, 0.5, 1, 2 and 4 with an offset centre. Neither check is a spatial
convergence study of the 3D LES. The plane operator does not reconstruct LES
pressure or represent the ambient withdrawal and must not be used as an
unbalanced gas source.

## Dispersion verification and remaining physical validation

The Langevin kernel is tested for stationary variance, lag correlation,
Galilean invariance, rotational covariance, and zero-energy/time limits. These
are numerical verification tests against stochastic identities, **not an
independent experimental validation of droplet dispersion**.

The selected paper supplies a useful model but explicitly requires extra drift
terms for inhomogeneous turbulence. It also reports limitations for preferential
concentration at small inertia. Our homogeneous implementation does not include
those terms, correlated spatial noise, drag integration, a transported SGS
energy budget, evaporation or feedback. No validated mapping from the current
AMD viscosity to unresolved kinetic energy exists in this implementation.
Published validation of the authors' solver does not validate this solver.

[Tsang et al. (2019)](https://journals.sagepub.com/doi/full/10.1177/1468087418772219)
also distinguish the effects of droplet dispersion from gas entrainment. A
successful spreading test cannot establish that the hot-core problem is fixed.

A full spray-closure claim still requires:

- An independent finite-inertia droplet-transport comparison with measured
  upstream size/velocity/mass-flux distributions and downstream profiles. The
  [Rüger/Khaled data audit](../FreeWaterSprayTransport/README.md) identifies the
  missing numerical 25 mm inlet distributions and 50/100/200 mm targets. Papers
  and aggregate errors alone cannot supply this validation.
- A justified SGS-energy input and inhomogeneous drift model, with a well-mixed
  tracer check. Local isotropic random kicks are insufficient in the spray core.
- Conservative exchange with the carrier, including ambient withdrawal,
  equal/opposite parcel forces and an explicit unresolved-energy budget; then
  a no-double-counting handoff between the unresolved jet and LES.
- Independent spray entrainment and evaporation checks, followed by mesh,
  parcel-count, timestep and averaging-duration studies of the coupled model.

The present free-jet evidence applies to a steady, developed, nonbuoyant,
constant-density gas jet in still unconfined surroundings. It does not establish
near-nozzle, confined-coflow, finite-inertia or evaporating-spray accuracy.

## Completed gpudev assessment (2026-09-16)

- Job **30271912**: all **13 numerical verification tests passed** on an A100.
  The numerical kernel and test-source hashes still match this tested version.
- Job **30271957**: completed the expanded experimental assessment on an A100.
  It exited **1 deliberately because the radial-profile accuracy gate failed**,
  not because of a solver crash. Reports and plots were written successfully.

| Independent observable | Model | Experimental summary/fit | Error |
|---|---:|---:|---:|
| Centreline decay coefficient | 6.25 | 5.8 | 7.759% |
| Half-width growth slope | 0.0941928 | 0.094 | 0.205% |
| Equivalent top-hat entrainment coefficient | 0.08 | 0.081 | 1.235% |
| Normalized velocity at `r/(x-x0)=0.2` | 0.0439369 | 0.0706435 | **37.805%** |

Maximum relative error in the coarse-plane integrated fluxes was `2.220e-16`.
The Gaussian profile agrees much better in the core than at the edge. Thus the
integral-summary screen passes, while the full-profile screen fails. Neither the
13 verification tests nor those passing summaries qualify the combined model
for production spray use. No coefficient was changed to repair the holdout.

Full [report](../../outputs/spray_closure_validation/job-30271957/report.md),
[machine-readable results](../../outputs/spray_closure_validation/job-30271957/report.json),
[profile plot](../../outputs/spray_closure_validation/job-30271957/radial_profile.png),
and [numerical test results](../../outputs/spray_closure_validation/job-30271912/tests.xml)
are local ignored artifacts. The comparison script records source hashes.

## Inhomogeneous tracer consistency

A separate [well-mixed tracer component](inhomogeneous-dispersion.md) now provides
one derived isotropic Gaussian drift for stationary zero-mean, constant-density
flow. Gpudev job **30272114** passed four numerical tests and the refined
262,144-particle benchmark: maximum concentration error **3.15%**, compared with
**171–173%** for the uncorrected controls. The first lower-count run's initial
sampling failure is retained. This does not extend the homogeneous
Pozorski–Apte kernel to finite-inertia droplets or validate a production LES
closure; mean shear, anisotropy, interfaces and energy feedback remain open.

## Subsequent profile alternatives

The [independent candidate comparison](profile-assessment.md) retains this
baseline and adds a separately calibrated Gaussian and a constant-eddy-viscosity
shape. Gpudev job **30272071** passed three new numerical checks, then rejected
all candidates on the unchanged physical accuracy screen. The separately
calibrated Gaussian improves maximum local error to **29.524%**; the alternative
shape gives **66.099%**. No held-out coefficient is fitted and no candidate is
promoted into production.

## Reproduce

From the repository root, on the requested queue:

```sh
sbatch --output=outputs/spray_closure_validation/slurm-%j.out tools/validate_spray_closure.sbatch
```

The output parent must exist before submission:
`mkdir -p outputs/spray_closure_validation`.
Set `JAXWIND_PYTHON` to select another CUDA-enabled environment. The job runs the
verification suite and independent comparison, writing reports under
`outputs/spray_closure_validation/job-JOBID/`. It exits unsuccessfully if a
numerical test or the experimental screening tolerance fails. Generated logs,
reports, images and downloaded research PDFs remain under ignored `outputs/`.
The source, tests and small transcribed reference file are reviewable inputs.

The subsequent [momentum-compatible covariance feasibility audit](jet-realizability-assessment.md)
checks whether a realizability constraint can also satisfy the existing local
momentum relation. It does not change the independently assessed profile.

The [consistent constrained-timescale follow-up](constrained-kepsilon.md)
re-solves transport with a physical covariance constraint. The qualified
corrected candidate still fails the independent profile gate at 19.8810%;
the uncorrected control remains numerically rejected.
