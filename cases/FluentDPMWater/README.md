# Fluent-reference water-spray DPM baseline

The user selected public documentation as the reference on 2026-09-17. The
baseline is pinned to **ANSYS Fluent 2026 R1**, for individual LES cells of
**16 × 16 × 4 m**. [baseline.json](baseline.json) records the selected options.
This is an independent implementation of documented equations, not a copy of
proprietary source or a claim of complete Fluent numerical equivalence.

## Selected model and reference

- Transient spherical water droplets; Morsi–Alexander drag, gravity and finite
  slip. The existing analytic frozen-drag motion is reusable. See the public
  [motion](https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/flu_th/flu_th_disp_equations_particle_motion.html)
  and [integration](https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/flu_th/flu_th_sect_pt_numerics.html)
  sections. Thermal integration here uses standard Cash–Karp 5(4), with explicit
  error controls; Fluent's internal control logic has not been reproduced.
- Warm, pure-water Law 2, diffusion-controlled baseline. The optional
  convection/diffusion model includes both logarithmic mass transfer and its
  associated heat-transfer correction. Droplet and gas temperatures enter the
  diffusion model's concentrations separately. No droplet condensation is
  enabled. Reference: [Law 2](https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/flu_th/flu_th_sec_disp_law2.html).
- Ranz–Marshall transfer and finite droplet temperature. Constant material
  properties and the saturation function are explicit repository inputs,
  not assumed Fluent database values. Vapor sensible heat is retained and the
  latent heat follows the same species enthalpy reference. Inert heating is
  checked against an analytic solution; see [Law 1](https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/flu_th/flu_th_sec_disp_law1.html).
- Isotropic discrete random walk (DRW) with persistent Gaussian eddies, lifetime
  and crossing events. Deterministic transport is the verification control.
  LES input is subgrid turbulence only. The documented characteristic-scale
  mapping now uses the carrier SGS viscosity with the scale providers described
  below. AMD inference is an explicitly identified extension. The documentation also identifies
  inhomogeneous-flow limitations of DRW. Reference:
  [turbulent dispersion](https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/flu_th/flu_th_sect_pt_turbdispersion.html).
- Two-way mass, momentum and energy exchange with the **local CFD gas state**.
  A separate humid plume/entrainment model is not part of this baseline.
  Reference: [phase coupling](https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/flu_th/flu_th_sec_discrete_couple.html).

The source inputs (mass flow, sizes, velocity, temperature and nozzle placement)
must be prescribed independently. Primary atomization, breakup, collision,
radiation, boiling, freezing, wall films and blade impacts are disabled, not
implicitly modeled. Warm kernels reject out-of-range states. Any later enabling
of these processes changes the baseline and requires its own validation.

## Implemented pieces and remaining integration

| Component | Current status |
| --- | --- |
| Law 1/2 rates and coupled mass/enthalpy integration | Implemented in `physics/fluent_dpm.py`; independently checked against analytic heating and a separate mass/temperature ODE integration |
| Disappearance | Explicit residual transfer below 1e-8 of starting drop mass; residual reported, not silently deleted |
| DRW draws and LES characteristic-scale mapping | Implemented; Gaussian variance, lifetime and crossing checks pass |
| Persistent DRW motion | Implemented in `fluent_dpm_dispersion.py`; event and restart/step-partition verification recorded below |
| Finite local gas exchange | Implemented in `fluent_dpm_cell.py`; binary species sensible/formation enthalpy and gas/particle momentum accounted together |
| Spatial tracking, injection, deposition and exit ledgers | Implemented in `fluent_dpm_spatial.py`: cell residence, persistent DRW, gravity, injection, periodic crossing, ground trap and open escape |
| Atmospheric/turbine carrier | Connected to the single-turbine recorded-inflow workflow through a separate full-species-enthalpy DPM state; Boussinesq approximation retained |
| Fluent-to-JAX trajectory/source comparison | Not performed; no Fluent executable found on PATH and no matched reference run supplied |
| Coarse-grid wake cooling validation | Not performed |

The finite-cell operator conserves mass, vector momentum and the sum of species
enthalpy, mean kinetic energy and an owned unresolved-energy reservoir. Its
split gas/bin update and configurable dissipation allocation are explicit
numerical choices, not claimed reproductions of Fluent's coupling iterations.
It holds source pressure fixed; the spatial driver supplies transport while
retaining a Boussinesq carrier. Its gas inventory is the local cell gas,
not an additional spray-core inventory.

## Verification and illustrative response

Gpudev **30273744** passed **22 checks**, including two independent evaporation
trajectory comparisons, analytic inert heating, saturation/no-condensation,
complete disappearance, rollback, DRW statistics, and finite-cell budgets and
substep convergence. Earlier **30273740** found a cancellation-scale assertion
(1.55e-18 W); its tolerance now includes roundoff in the contributing terms.
**30273743** found a missing species argument in the new test fixture. Both
failures remain archived; neither was a physical acceptance decision.

Gpudev **30273746** passed **4 additional checks** for persistent DRW motion:
step-partition invariance with identical random state, retaining eddies until
expiration, analytic laminar settling/turbulence restart, and transactional
rollback on event-capacity exhaustion. The combined total is **26 passing
checks**. This does not remove the documented inhomogeneous limitations of DRW
or establish physical accuracy of an LES characteristic-scale inference.

The same accepted job advanced an illustrative **0.1 kg pulse** split equally
between 50 and 200 micrometre drops in the gas inventory initially occupying
1,024 m³ at 310 K and 101325 Pa. Initial vapor mass fraction was 0.006; gas speed
8 m/s and droplet speeds/temperatures are explicit in the audit script. After
2 s, diffusion-controlled evaporation left **0.0424448 kg** liquid and reduced
the gas temperature by **0.118172 K**. The corrected convection/diffusion option
left **0.0426786 kg** and cooled by **0.117762 K**. These are source-operator
responses, not spatial wake predictions or measurements.

Final total-budget residuals across the two options were at most
9.10e-13 kg, 1.46e-11 kg m/s and 5.97e-8 J. Thermal response should not be called
validated from these residuals. Physical acceptance remains the user-selected
20%; numerical verification uses much stricter tolerances.

Generated evidence is in
`outputs/fluent_dpm_verification/30273744/{tests.xml,cell_assessment.json,source.sha256}`.
Public reference pages/equation graphics and their hashes are under ignored
`outputs/fluent_dpm_reference/`. Research downloads and run artifacts are not
version-control inputs.

Reproduce on the requested queue:

```bash
sbatch --output=outputs/fluent_dpm_verification/%j.log tools/verify_fluent_dpm.sbatch
sbatch --output=outputs/fluent_dpm_verification/%j.log tools/verify_fluent_dpm_dispersion.sbatch
```

The next validation gate is a matched downstream transport/cooling experiment.
Match material
functions, source distributions, interpolation/deposition, timestep controls,
LES turbulence scales and boundary conditions before interpreting differences
from a Fluent run. Preserve numerical equivalence and physical validation as
separate requirements.


## Spatial turbine integration and AMD inference

Select `physics.water_spray.model = "fluent-dpm"` and supply the nested `dpm`
settings shown in [v80_coarse_smoke.toml](v80_coarse_smoke.toml). Source position
is turbine position plus the streamwise offset, at hub height. Size bins carry
prescribed mass fractions, temperature and three-component velocity. This
point source represents a declared post-atomization condition, not a resolved
nozzle. One parcel per bin is injected each carrier step. Capacity exhaustion
rejects the step; no merging, mass deletion or automatic source weakening occurs.

`fluent_dpm_source.py` gathers adjacent MAC faces and deposits their momentum
adjointly. Residence intervals assign evaporation/heat sources to crossed
cells. `fluent_dpm_spatial.py` records injection, escape, trapping, gravity,
stochastic work and wall reactions. Its frozen-velocity path with split forces
is first order in tracking timestep; it is not Fluent's trajectory interpolator.
Use tracking-substep refinement. Injection timing is also first order.

The carrier transports vapor/dry-air ratio and species enthalpy with common
conservative MUSCL-MC/SSP-RK3 fluxes and bounded subcycling. Temperature follows
`H = (cp_d + q*cp_v)*(T-Tref) + q*Lv_ref`; vapor sensible heat is retained.
The common diffusivity assumes unity carrier Lewis number. The carrier remains
Boussinesq with fixed reference dry-air density, prescribed thermodynamic
pressure and divergence-free volume flow. Gas properties used in droplet laws
are evaluated from local T, vapor fraction and pressure. This does not reproduce
Fluent's variable-density pressure/EOS coupling. Phase equilibrium, imposed wall
heat fluxes and coupled thermal surface models are not enabled in this path.

Two explicit LES inputs are available:

- `les_model = "fluent-smagorinsky"` (documented comparison default): both carrier
  stress and DPM use the same static model. `l = min(kappa*z, Cs*V^(1/3))`, with
  Cs=0.1 and kappa=0.41 by default. The upper atmospheric lid is a symmetry plane.
- `les_model = "amd-inferred"`: retain carrier AMD and require
  `amd_length_scale_m > 0`. Use the actual AMD viscosity and the declared length
  in `v = nu/l`, `k = v^2`, `epsilon = v^3/l`. **This length and inferred energy
  are additional modeling assumptions**, not AMD predictions or a documented
  Fluent AMD model. The smoke case's 1 m is illustrative and uncalibrated.

AMD is appropriate for anisotropic LES grids; its viscosity describes modeled
energy transfer/dissipation, which does not uniquely determine unresolved
velocity variance or correlation time. A zero AMD viscosity therefore yields
zero inferred dispersion in this particular extension and does not prove that
all unresolved fluctuations vanish. An isotropic DRW also does not reproduce
anisotropic SGS velocity statistics. Do not add resolved wake variance to the
SGS input, which would count resolved particle forcing again. References:
[AMD paper](https://authors.library.caltech.edu/records/kefzd-2py54) and
[Fluent LES scales](https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/flu_th/flu_th_sec_turb_scales_LES.html).

Source conservation is checked against actual gas/parcel inventories. Prescribed
stochastic velocities can supply mechanical work; that work is explicitly
reported and is not withdrawn from an evolved SGS-energy reservoir. Projection,
carrier SGS dissipation, open-boundary work and turbine extraction also prevent
interpreting source conservation as a closed total-energy LES solver. The current
serial parcel implementation is a verification baseline; large-farm throughput
has not been established. Multi-turbine controlled-farm assembly is not wired.

Gpudev **30274377** passed **62 checks**, including 10 new spatial/LES/carrier
checks and regressions of previous moisture, moving-source and Fluent kernels.
The new checks cover actual MAC source budgets, wall/stochastic work, cell
crossing, evaporation, trapping/escape, capacity rollback, both LES inputs,
coarse-cell cooling, species consistency and exact restart. Gpudev **30274406**
passed **3 additional checks** for first-order tracking refinement, retaining
actual DRW eddies and RNG across checkpoint restart, and path-capacity rollback.
Tracking position errors relative to 16 substeps decreased from 0.07241 m to
0.02768 m to 0.01085 m with 1, 2 and 4 substeps over 0.02 s. This is a numerical
refinement check, not a measurement comparison.

Reproduce these checks and the V80 smoke audit on gpudev:

```bash
sbatch --output=outputs/fluent_dpm_verification/spatial-%j.log tools/verify_fluent_dpm_spatial.sbatch
sbatch --output=outputs/fluent_dpm_verification/tracking-%j.log tools/verify_fluent_dpm_tracking.sbatch
sbatch --output=outputs/fluent_dpm_verification/v80-%j.log tools/audit_fluent_dpm_spatial.sbatch
```

The V80 smoke script supplies uniform initial/recorded inflow explicitly, uses
the repository's V80 rotor deck and shared `build_open_components` turbine
workflow, and compares dry versus spraying for only 4 s. The 256 × 128 × 128 m
**domain** has 16 × 8 × 32 cells of **16 × 16 × 4 m**. This short, narrow-domain
startup is an integration check; it is not a developed atmospheric wake, a
validated injection prescription or a production wake-cooling prediction.

Gpudev **30274404** completed the matched V80 dry/spray startup audit with AMD
and the explicit 1 m dispersion length. At 4 s, 0.400000 kg had been injected,
0.114709397 kg evaporated, and 0.285290603 kg remained liquid. No parcels had
escaped or deposited. Maximum cooling relative to the dry control was
**0.0443125 K** in a cell; this is a startup minimum, not a plane-averaged or
rotor-averaged cooling benefit. Forty parcels remained active, with eight DRW
draws. The parcel-water residual was 1.67e-16 kg; the independently accumulated
carrier vapor change, after net transport, differed from evaporation by
6.22e-12 kg. Maximum phase-source energy residual was 7.28e-11 J. Reported
stochastic work was 0.00569 J and buoyancy-corrected gravity work 3.08102 J.
Evidence: ignored `outputs/fluent_dpm_verification/30274404/` contains the JSON
report, saved fields and source hashes. The case's source parameters and length
are exploratory, with no 20% physical-acceptance decision implied.

Final gpudev regression **30274407** passed **69 checks** after adding early
rejection of unsupported carrier boundaries and malformed numeric settings.
Together with the three tracking checks, **72 distinct checks pass**. The V80
audit's evolution kernels are unchanged; the later source changes only add
input guards. A subsequent test-only dictionary-literal/comment cleanup is
nonfunctional. The final regression hashes are archived with its report.
