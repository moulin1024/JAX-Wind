# Independently specified radial-profile alternatives

**Result: the separately calibrated Gaussian improves the maximum local profile
error from 37.805% to 29.524%, but no assessed candidate meets the unchanged 10%
criterion.** The constant-eddy-viscosity alternative is worse. No production
profile or waterjet configuration has been switched to these candidates.

## Inputs and independence

The frozen [candidate specification](profile_candidates.json) contains:

1. The original Gaussian, with equivalent-top-hat entrainment `alpha=0.08` from
   Ricou–Spalding. This remains the baseline.
2. A Gaussian with `C=75` from Table 3 and equation 3.5 of
   [Huck et al. (2022)](https://doi.org/10.1017/jfm.2021.992), with
   `U/Uc=exp(-C eta^2)`. This is a separate air–water experiment using roughly
   1 micrometre droplets as gas tracers. The present Gaussian integral model
   requires `alpha=1/sqrt(2 C)`, so that mass, momentum, centreline speed and
   width remain mutually consistent. Only `C` is transferred; the experiment's
   other fitted constants are not imposed independently.
3. The squared-Lorentzian profile from the constant-radial-eddy-viscosity
   boundary-layer similarity solution, as discussed in
   [Basset et al. (2022), equations 3.4 and 5.10](https://doi.org/10.1017/jfm.2022.638).
   Its entrainment coefficient remains the independent Ricou–Spalding value.

The held-out Hussein LDA profile coefficients, coordinate range, three integral
summary targets and 10% threshold in `reference.json` were not changed. The
first LDA row of the original scanned Table 4 was visually rechecked: `C0=1`,
`C2=12.12`, `C4=2815`, `A=111`. A transcription error was not found.

No optimizer or parameter sweep uses the held-out target. This is a
retrospective comparison after seeing the baseline's failure, not a blind
preregistered experiment. The candidates were documented before evaluation;
all results are retained, including the worse alternative. Calibration on
Huck's gas proxy does not prove transferability to a single-phase gas jet, and
neither paper establishes a universal profile for the confined waterjet.

## Conservation fixes the reconstruction

For `U(r)=Uc/(1+(r/b)^2)^2`, integration over an infinite round section gives

```
m = pi*rho*Uc*b^2
J = pi*rho*Uc^2*b^2/3
Uc = 3*J/m
b = m/sqrt(3*pi*rho*J)
b_half = b*sqrt(sqrt(2)-1)
```

Replacing the Gaussian shape while retaining its original centreline speed and
width would violate these identities. The new reconstruction instead holds the
integrated mass and axial mean-flow momentum fixed. Like the baseline, it omits
Reynolds-normal-stress and pressure contributions to the momentum integral.
It has no new fitted shape coefficient.

## Gpudev assessment

Job **30272071** ran the final lint-clean implementation on the GPU. All **three
new numerical tests passed**: two independent infinite-domain flux quadratures
and a differential-identity check of constant eddy viscosity. The assessment
then completed with exit code **1 because all physical profile screens failed**.
This is an accuracy rejection, not a solver failure. Job 30272069 had already
completed the same numerical assessment before an attempted pending-job
cancellation; the final rerun also fixes explicit closure-variable binding and
script executable permissions. Results agree.

| Candidate | Centreline decay B | Half-width slope | Alpha | Maximum profile error | All screens pass |
| --- | ---: | ---: | ---: | ---: | --- |
| Original Gaussian / Ricou | 6.25000 | 0.094193 | 0.080000 | 37.805% | No |
| Gaussian / Huck C=75 | 6.12372 | 0.096135 | 0.081650 | 29.524% | No |
| Squared Lorentzian / Ricou | 9.37500 | 0.059453 | 0.080000 | 66.099% | No |

Each candidate was evaluated with section steps of 8, 4, 1 and 0.25 nozzle
diameters. The exact integral advance gives the same asymptotic metrics at each
step. This is not convergence evidence for an LES calculation. The observations
are published fit summaries with unavailable uncertainty intervals; the 10%
threshold remains an engineering screen, not a statistical confidence bound.

Local ignored results:

- `outputs/spray_closure_validation/profile-30272071/report.json`: every candidate,
  step, metric and error, plus source hashes.
- `report.md`, `profiles.csv`, `profiles.png`: comparison and plotted profiles.
- `tests.xml`: numerical verification results.

Reproduce from the repository root:

```bash
sbatch --output=outputs/spray_closure_validation/profile-%j.out \
  tools/assess_jet_profiles.sbatch
```

## Consequence for the active goal

A separately measured Gaussian width provides a modest improvement, but the
profile requirement is still unmet. Replacing the Gaussian with the classical
constant-eddy-viscosity shape is not a defensible correction for this holdout.
Further work needs independent radial-profile/stress information or a justified
nonuniform transport closure, with its integral budgets checked. The held-out
Hussein polynomial must remain a target, rather than become the reconstruction.
Measured droplet transport, well-mixed dispersion, LES coupling and all-sensor
waterjet validation remain separate, outstanding requirements.

A subsequent [independent DNS calibration assessment](dns-profile-assessment.md)
fits a positive Gaussian mixture to a separate stationary jet and tests its
conservative reconstruction. Gpudev **30272244** passed six numerical checks
but rejected all three declared candidates (best maximum error **38.513%**).
The DNS source itself has substantially lower outer-tail velocity than the
Hussein holdout. This does not improve the best result above or justify a
production model change.
