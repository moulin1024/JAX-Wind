# Active persistent spray and waterjet validation goal

## Fluent DPM reference selected by the user — 2026-09-17

The user next requested reproduction of Fluent DPM for this application and
explicitly selected **public documentation and an explicit baseline**. The
[Fluent 2026 R1 baseline](../cases/FluentDPMWater/README.md) now takes precedence
for the immediate implementation. Use local CFD gas conditions, documented
water-droplet laws and LES-only stochastic dispersion inputs. The separate
conditional plume proposal below is not part of the Fluent baseline and must
not be silently added to a comparison. Coarse-cell mixing sensitivity remains
an application validation question.

New equation/thermal and finite-cell source kernels passed **22 checks** on
gpudev **30273744**. An illustrative source audit at 16 × 16 × 4 m cell volume
closed mass/momentum/energy budgets and produced about 0.118 K cooling from a
0.1 kg pulse over 2 s. This is not a spatial wake run or a Fluent solver
comparison. Persistent DRW event tracking passed **4 additional checks** on gpudev
**30273746** (26 checks combined);
Subsequent spatial/turbine integration now has a separate consistent species-
enthalpy carrier, parcel lifecycle accounting and persistent DRW state. It
supports documented static Smagorinsky scales and optional AMD inference with
an explicitly declared length assumption. Gpudev 30274407 and 30274406 passed
72 distinct numerical checks including regressions. This is not physical validation;
source budgets do not close an evolved LES SGS-energy reservoir. See the
baseline README for current evidence and remaining limitations.

A matched V80 dry/spray startup check on gpudev 30274404 used individual
16 × 16 × 4 m cells, the carrier AMD viscosity, and an explicit uncalibrated
1 m dispersion length. It advanced 4 s: 0.114709 kg evaporated from 0.4 kg
injected and maximum local cooling was 0.044313 K. The carrier vapor residual
was 6.22e-12 kg. This establishes runnable turbine coupling, not a developed
wake response, Fluent parity, or satisfaction of the 20% physical threshold.

Keep the broader goal incomplete. The controller remains paused; no status or
budget change was requested by this baseline selection.

## Application priority clarified by the user — 2026-09-17

**16 m × 16 m × 4 m are individual LES cell dimensions (1,024 m³).**
The intended application is downstream water-droplet transport, evaporation,
and cooling of a wind-turbine wake. The nozzle jet is entirely subgrid.
This explicit clarification takes precedence over the profile-first ordering
below and the historical goal-controller objective. The controller is currently
paused; this document records the revised scope without changing that status.

The current implementation target is an unresolved spray/plume state coupled
conservatively to the existing atmospheric LES, with size-dependent droplet
slip, settling, heat/mass transfer, and justified unresolved dispersion.
Subgrid temperature/humidity and finite ambient entrainment must control local
evaporation while the plume is narrower than a cell. Do not equate conservative
cell deposition with instantaneous physical mixing through the whole cell.
The unresolved plume may persist across many cells; no assumption of complete
evaporation or a resolved plume within the injection cell is permitted.

See the [coarse-cell application contract](design/coarse-cell-water-spray.md).
Fine radial jet profiles are supporting closure evidence, not the main production
output or a reason to keep optimizing a resolved jet. Retain the waterjet as a
component validation case. Prioritize downstream liquid survival, vapor/thermal
fluxes, plume trajectory/spread, and wake cooling/buoyancy at the application
scale. Keep the user-authorized 20% physical screen and strict numerical and
conservation checks; define observable-specific error normalization in advance.

The untested profile-eddy-viscosity experiment introduced immediately before
this clarification has been removed. It was never submitted or validated; no
waterjet improvement is claimed. Existing source/transport/conservation work
and archived results are retained. The target turbine case and nozzle placement
remain to be specified; do not silently reuse laboratory nozzle settings.

## Reactivated by the user — 2026-09-16

The user explicitly requested a persistent goal to improve profile accuracy,
validate measured droplet transport, implement conservative LES coupling, and
ultimately validate the waterjet. The goal controller successfully created this
objective with **active** status. No token budget was specified.

This instruction supersedes the historical stop, transport-only restriction,
and goal-controller limitation recorded below. Preserve historical results and
checkpoints; do not assume old process IDs or jobs are still running.

The solution must remain useful on **very coarse grids**. Mesh studies test
closure robustness and numerical error; resolving the nozzle or jet by brute
force is not a substitute for an unresolved-spray model.

### Acceptance revision — 2026-09-17

The user explicitly changed physical engineering acceptance from **10% to 20%**.
This applies to radial profiles and integral quantities, selected measured
transport observables, and every one of the nine waterjet temperature sensors.
It supersedes the original 10% wording in the goal-controller objective and older
plans. Source qualification, numerical convergence, conservation and data
independence requirements are unchanged. Historical 10% reports remain archived;
a changed acceptance result is not an improvement in the prediction itself.

### Completion criteria

1. **Profile accuracy:** improve or replace the Gaussian candidate using
   physically justified modeling and independent calibration. Assess radial
   profiles as well as integral quantities, preserving held-out data and the
   user-revised 20% engineering screen. Report uncertainty and near-zero handling;
   do not fit the validation profile or waterjet temperatures.
2. **Measured droplet transport:** qualify a measured upstream source and
   independent downstream size, velocity and spatial/mass-flux observations.
   Freeze source/target roles and observables before evaluation. Demonstrate
   agreement per selected observable with numerical convergence evidence.
3. **Conservative LES coupling:** account for ambient mass/species/enthalpy
   withdrawal, equal/opposite interphase exchange, justified unresolved
   turbulence energy and inhomogeneous dispersion, and a handoff that avoids
   double-counting gas inventory or resolved mixing. Verify conservation and
   well-mixed behavior before interpreting cooling predictions.
4. **Waterjet validation:** assess all nine sensors against the existing
   experimental criterion, with conservation, timestep, parcel-count,
   coarse-grid and averaging-duration checks. Resolve observation-plane and
   apparatus compatibility; distinguish collected-water measurements from
   escaping droplet temperatures. Do not substitute a passing average for a
   failed sensor.

Use **gpudev** for computational jobs. Keep research PDFs, logs, caches and
other generated artifacts under ignored workspace output paths. No author
contact is authorized merely by this goal. Keep numerical verification,
independent physical validation and missing data distinct in reports. The goal
remains active until all completion criteria are met or an external dependency
prevents meaningful progress under the goal controller's blocking rules.

### Starting evidence and next work

- The [user-authorized 20% re-assessment](../cases/SprayClosureValidation/acceptance-20-percent.md)
  completed on gpudev **30273713 / 30273715**. The qualified constrained profile
  now passes the physical engineering screen at **19.8810%** maximum local
  error; the numerically rejected control stays rejected. All four archived
  recent waterjet transport schemes now pass **9/9** temperature sensors, with
  unchanged worst errors **12.5307–12.9899%**. Historical 10% reports are retained.
  These classifications do not close measured droplet, production coupling,
  conservation, or convergence requirements. The user next requested an opt-in
  spray-closure integration and a controlled waterjet comparison.

- The [consistent constrained-timescale experiment](../cases/SprayClosureValidation/constrained-kepsilon.md)
  now re-solves momentum and turbulence transport with a realizability-limited
  timescale and independently checked active-edge powers. Final gpudev
  **30273699** passed **12 checks**; the corrected candidate passes all numerical
  refinements and source-width agreement. Its final modeled covariance minimum
  is **-4.24e-16 K**, physical residual **2.13e-8**, and boundary residual
  **2.84e-14**. The uncorrected control remains numerically rejected, so the
  paired job correctly exits 2. Separate independent assessment **30273701**
  completed with unchanged source hashes and a retained rejected control. The
  candidate still **fails** the unchanged profile gate: **19.8810%** local error,
  **13.9298%** high centerline coefficient. Its profile changes only
  **5.6823e-8 Uc** from the earlier unconstrained normal-production case. Thus
  resolving the edge covariance does not materially resolve the interior
  discrepancy. No production enablement, finite-inertia measured validation,
  or identification of RANS K with SGS energy follows. Future profile work
  should address the remaining transport/pressure/anisotropy approximations;
  measured droplet and conservative LES coupling gates remain open. Rejected
  numerical experiments and the control startup failure are explicitly retained.

- The [momentum-compatible realizability audit](../cases/SprayClosureValidation/jet-realizability-assessment.md)
  passed **11 checks** on gpudev **30273619**. All sampled covariance violations
  in four accepted source-only edge states admit a positive local viscosity
  reduction with the boundary-layer momentum relation enforced; both gradient
  diagnostics have zero infeasible or indeterminate samples. For the best
  diagnostic, the minimum viscosity fraction is **0.485405**, with resulting
  minimum eigenvalue within **3.56e-15 K** of zero. An old-gradient viscosity
  cap still gives **-0.004480 K** after momentum recomputation, so that simpler
  correction is insufficient. Only **1.3770e-8** of the solved-domain integral
  of eta K is affected. No profile or transport equation was re-solved, no
  held-out data was read, and the **19.881%** profile error is unchanged. The
  next closure step needs a consistent constrained timescale and newly checked
  edge conditions; local PSD feasibility is not physical validation.

- The [source-state stress/pressure audit](../cases/SprayClosureValidation/jet-stress-assessment.md)
  rules out a simple omitted-flux rescaling as an explanation for the best
  diagnostic's 13.930% high centerline coefficient: its modeled thin-jet
  stress/pressure remainder is **-0.7414% of mean momentum**, with the opposite
  sign for that proposed fix. No velocity amplitude, profile, or held-out score
  was changed. Gpudev **30273537** passed **four** Cartesian-gradient, pressure
  identity and tensor checks; eight source-state cases were retained. Pressure
  identity error was at most **6.29e-10 of mean momentum**, quadrature change
  **3.43e-11**, and edge-cutoff change **1.11e-7**. A follow-up distinguished
  appended tails from genuinely computed covariance: final **30273563** passed
  the four tests and retained **19** converged edge solves. The best diagnostic
  has a solved-domain minimum covariance eigenvalue **-4.7814e-8** in Uc² units,
  with only **1.3759e-8** of integrated K in the violating solved region. A doubled
  diagnostic grid changes that eigenvalue by **9.10e-15**. This prevents direct
  use of the raw tensor for particle dispersion there, but does not explain
  the much larger mean-profile error. The corrected shear-only delta=0.0001
  solve was rejected (mesh exhaustion, residual 10.829), so the audit correctly
  exited 2 and retains its converged delta=0.00025 result. Source hashes were
  checked after exit and matched. Earlier **30273560** failed numerical tail
  extrapolation; the initializer was fixed without changing model equations.
  Next: a justified realizable stress/dispersion treatment and conservative
  filter/energy ownership, not covariance clipping or holdout tuning. Best
  profile error remains **19.881%**, failing **10%**; measured transport,
  embedded-core LES coupling and final waterjet validation remain incomplete.
- The [independently sourced 1990 Pope-convention assessment](../cases/SprayClosureValidation/pope-profile-assessment.md)
  improves the best diagnostic maximum local radial-profile error from
  **26.018% to 19.881%**, but still fails the unchanged **10%** gate. The
  normal-strain case predicts centerline decay B=**6.60793**, **13.930% high**;
  width and entrainment errors are **7.421% / 2.153%**. The shear-only case
  has **24.579%** local profile and **15.674%** decay error. Both were frozen
  before holdout assessment; all results are retained, with no fit or rescaling.
  Primary NASA CR-187485 (December 1990), equation (11), explicitly documents
  C1=1.44, C2=1.92, C3=0.79 for Pope’s correction. This is a separate convention
  from the confirmed 1.45/1.90 pair printed in Pope 1978; the original failures
  were not erased or reclassified. The new corrected variants pass independent
  source-rate comparisons by **0.264% / 1.191%** within the frozen 2% gate.
  Gpudev **30273498** passed **14 tests** and qualified all eight steady
  diagnostic combinations using a marched initializer, free-edge refinement,
  tighter tolerance and changed initial mesh. Maximum final edge-refinement
  profile change was **7.14e-7 Uc**; width/integral change **3.23e-6 relative**.
  Source hashes passed. Gpudev **30273502** performed the unchanged holdout
  test only after numerical/source gates passed; its plot was inspected.
  The normal-production extension is still a partial approximation, not full
  RANS or a production LES closure. Next: consistent stress/pressure momentum
  treatment and independently validated dispersion/energy ownership, without
  tuning to these observed holdout errors. Measured droplet validation,
  embedded-core withdrawal/handoff and final waterjet gates remain incomplete.
- The [downstream streamfunction marcher](../cases/SprayClosureValidation/kepsilon-marching.md)
  now provides a positive, conservative numerical path beyond the failed
  similarity boundary-value solves. Gpudev **30273458** passed **six** kernel
  tests and an independent moving constant-viscosity jet audit: 1024-cell
  maximum velocity error **1.606e-5 Uc**, half-width error **1.016e-5**,
  momentum-budget error at most **1.744e-14**. A harmonic-face prototype had
  pinned the front despite conserving momentum; it was rejected and retained.
  Source-only matrix **30273461** separated mesh, downstream step, coflow,
  boundary and inlet-memory effects. Final versioned runs **30273474** give
  spreading **0.1119195** standard and **0.0793888** corrected at the smaller
  step, still **10.464% / 7.687%** below the primary paper's **0.125 / 0.086**.
  Both fail the frozen **2% source-reproduction gate**; no new Hussein profile
  assessment is permitted. Direct addition of omitted normal-strain production
  in source-only audit **30273475** failed all four trials within three steps.
  Next: reconcile the complete primary production/gradient treatment and solve
  it consistently, or select a separately documented model; do not retune
  coefficients. Profile error remains **26.018%**, failing **10%**, and measured
  droplet transport, LES energy ownership/embedded-core handoff and waterjet
  validation remain incomplete. All computational jobs used gpudev and all
  research data, logs and generated states remain ignored.
- The [source-frozen k-epsilon profile candidate](../cases/SprayClosureValidation/kepsilon-profile.md)
  now has verified similarity equations for Pope’s original standard and
  vortex-stretching variants. The primary paper fixes Cmu=0.09, C1=1.45,
  C2=1.90, sigma_k=1, sigma_e=1.3 and corrected C3=0.79; no Hussein fit or
  measured-width rescaling is used. Gpudev **30273381**, repeated on final files
  as **30273421**, passed **five** analytic,
  dimensional-PDE and tensor checks with source hashes verified. This is
  equation verification only. Collocation trials **30273238/30273285** failed;
  bounded **30273334** hit its evaluation cap; better-conditioned **30273399**
  stopped with unacceptable scaled residuals **0.02150/0.008514**. The latter’s
  optimizer-success flags are explicitly rejected as physical predictions.
  **30273388** was a retained seed-path harness failure, and free-edge trial
  **30273417** also failed nonlinear convergence. No new holdout assessment was
  run, so the best independent error remains **26.018%**, failing **10%**.
  Next: positive downstream marching and independent source-spreading
  reproduction (0.125 standard, 0.086 corrected), then numerical qualification
  before the unchanged holdout. RANS k cannot yet supply LES unresolved energy;
  measured inlet qualification, conservative embedded-core ownership/handoff
  and waterjet validation remain open. Research and solver artifacts are ignored.
- The [compatible carrier energy extension](../cases/SprayClosureValidation/carrier-energy.md)
  now closes the opt-in warm-dilute H+K budget through the EOS iteration:
  numerical mixing loss heats enthalpy, pressure conversion has a matching
  boundary energy flux, wall energy is exported, and nonlinear iteration work
  remains an explicit error. It is available to cell/group/moving/injection
  transactions through `energy_coupling=True`; defaults and production waterjet
  settings are unchanged. Gpudev **30273109** passed **10 energy tests** in
  104.10 s. A shear/compressive carrier case improved from **-0.347770 J**
  uncompensated energy defect to **-1.506e-12 J**, with consistent heated EOS
  and momentum residual **3.030e-12**. Eight injected/outflow/gravity steps
  closed cumulative energy to **1.038e-12 J**. Local pressure-product error
  was at most **3.70e-16** relative to pressure-work scale across five boundary
  layouts. Wall-energy ownership and rollback passed; incompatible fixed-p0
  closed heating rejected without repairing conserved fields (EOS error
  **5.602e-7**). Earlier **30273084** passed all **14 default carrier
  regressions** plus two energy cases; new-test attribute/roundoff failures
  were corrected without changing solver equations or physical gates.
  This is an explicit constant-p0, dilute-energy approximation: compatible
  pressure conversion includes temporal projection transfer, numerical heat
  is not physical SGS k, and full moist thermodynamics, physical SGS/dispersion,
  embedded-core ambient withdrawal/handoff and measured validation remain open.
  Profile error remains **26.018%**, and waterjet validation is not achieved.
- The [source-only chord calibration assessment](../cases/WaterSprayRacz2025/chord-assessment.md)
  tests whether transit lengths can qualify a Gaussian-threshold detection
  radius instead of relying on nominal probe dimensions. Gpudev **30273040**
  passed **16 tests**, checked all source hashes and assessed **996,164 events**
  at all 34 inlet stations with 8/16/32 temporal blocks. All 102 unconstrained
  radius fits are inadmissible over the declared size range (implied cutoffs
  **2.630–10.014 micrometres**). A documented nonnegative least-squares
  follow-up fixes admissibility but still leaves **16.165–16.260%** of internal
  temporal-check events beyond predicted chord support, CDF distances
  **0.147–0.357**, and **223/265, 226/266, 227/266** sufficiently sampled
  size bins with second-moment mismatch above 10%. That last diagnostic does
  not replace the physical transport gate. **56.112%** of raw source events
  have a measured lower-bound trajectory angle above 5 degrees. The simple
  axial-cylinder model is therefore not qualified as an inlet correction;
  multidirectional receiver/processor calibration or an independently qualified
  dataset remains the route forward. Both unconstrained and constrained results
  are retained, no raw measurements were removed, and downstream holdouts were
  untouched. An initial time-boundary floating-point test failure (30273029)
  was fixed before the first measured calibration run (30273030); no physical
  tolerance was relaxed. The inspected figure and all reports remain ignored.
  Carrier qualification, complete conservative LES/SGS coupling, the 26.018%
  profile error and final nine-sensor waterjet validation remain outstanding.
- The [prescribed parcel-injection component](../cases/SprayClosureValidation/parcel-injection.md)
  now stages explicit mass-flow quadrature with per-parcel birth ages, exact
  external mass/momentum/enthalpy/kinetic-energy ledgers, start-of-step capacity
  reservation and pre-injection rollback on source/carrier failure. Gpudev
  **30272996** passed **9 injection tests** in 128.53 s, including a 12-step
  recycle/export test: injected **1.2e-7 kg**, exported **1.199897442574e-7 kg**,
  maximum cumulative vapor/liquid/boundary water residual **1.8185e-19 kg**.
  The coupled birth case took 9 carrier iterations, momentum residual
  **4.3285e-13**. Earlier **30272993** passed 26 tests (including all 19
  body-force/moving regressions) and failed one test-only JAX list-indexing
  operation, corrected before the final injection run. Solver sources were
  unchanged between runs, final source hashes match, and lint/diff checks pass.
  Exact birth ages do not remove first-order source/path splitting. No measured
  source distribution, observation correction, gas-nozzle entrainment, physical
  SGS closure or complete carrier energy closure is supplied. Independent
  profile error remains **26.018%** against the unchanged 10% screen; measured
  inlet qualification and all-nine-sensor waterjet validation remain open.
- The [particle body-force extension](../cases/SprayClosureValidation/particle-body-force.md)
  adds explicit liquid acceleration to moving residence exchange, with separate
  external impulse/work, pre/post-evaporation mass, early-exit force limits and
  all-phase rollback. Gpudev **30272964** passed **19 regression tests**. An
  independently derived finite-gas Stokes-settling comparison initially failed
  the coarsest position-rate check (**30272970**); it retained conservation and
  its error ratios approached the expected order under refinement. Extended
  job **30272972** passed with all coarse results preserved: final position
  reduction factors **1.947/1.974**, velocity factors approaching **4**, finest
  errors **3.705e-7 m / 2.674e-7 m/s**, and exact-total-impulse/global source
  budgets at six resolutions. No numerical solver or physical gate was changed
  to resolve that test failure. The constant-mass verification is not measured
  transport validation. Paths remain first order; gravity kick work balances
  kinetic energy but not exact gravitational potential along frozen paths.
  Gas-pressure/buoyancy reactions, full energy closure, injection, physical
  dispersion/SGS, embedded-core entrainment and waterjet validation remain open.
- The [source probe/observation assessment](../cases/WaterSprayRacz2025/probe-assessment.md)
  recovered and visually inspected the primary apparatus geometry, then tested
  its nominal axial extent against recorded velocity/transit time. Gpudev
  **30272931** passed **15 tests** and audited all **996,164 source events** with
  verified hashes. **25.848%** exceed the nominal point-particle hard-support
  bound; **2.155%** still exceed a generous finite-sphere bound with velocity
  and diameter allowances. Every station has exceedances, including **5.284%**
  of 20–40 micrometre events in the generous scenario. This rejects treating
  nominal dimensions as an already established hard detection volume, not the
  raw measurements. No events were filtered and no geometry was fit to force
  compatibility. A documented occupation-time flux operator now makes the
  missing effective volume, efficiency and acquisition-time inputs explicit.
  Published auto-calibration methods offer a follow-up route but have not yet
  been qualified for this apparatus. PDFs and all reports remain ignored;
  source/holdout roles and physical acceptance gates are unchanged.
- The [measured-source sampling assessment](../cases/WaterSprayRacz2025/sampling-assessment.md)
  quantifies a missing validation requirement without using downstream records.
  Gpudev **30272918** passed **15 tests**, verified all source hashes and audited
  **34 stations / 996,164 events**. Transit-time weighting changes axial means
  by up to **48.286%** (27/34 stations exceed 10%) and D32 by **17.026%**
  (18/34 exceed 10%). Conditional temporal-block ranges are much narrower but
  remain block-size sensitive: half-width ratios reach 2.467 for axial mean
  and 3.136 for D32. These are processing sensitivities, not physical model
  errors or qualified corrections. Both scenarios preserve all raw events;
  8/16/32/64-block, 2048-replicate results and an inspected source-profile figure
  are retained in ignored output. This establishes why a size/velocity/probe
  observation operator and qualified flux weighting are required before a
  measured transport claim; sparse carrier proxies, missing third velocity,
  apparatus ambiguity and physical SGS/entrainment closure remain unresolved.
  The original source/holdout split and all physical acceptance gates are unchanged.
- The [moving-parcel residence driver](../cases/SprayClosureValidation/moving-parcels.md)
  now traces cell crossings, exchanges for each visited-cell residence interval,
  exports remaining liquid at open-x boundaries exactly once, and commits
  positions/outflow with all phases through one carrier solve. Final gpudev
  **30272906** passed **10 tests** in 61.59 s with FutureWarnings treated as
  errors. Analytic path tests cover reverse motion, exact-face/corner crossings,
  repeated periodic wraps, stationary/immediate exits and capacity/wall rejection.
  Moving-source conservation includes exported liquid mass, momentum and energy;
  tested local source relative energy residuals were at most **2.372e-19**.
  The coupled crossing/outflow case took 17 iterations with momentum residual
  **1.576e-13**. Over a fixed 8e-4 s trajectory, dt=1e-4, 5e-5, 2.5e-5,
  1.25e-5 s gave position errors **5.669e-4, 2.739e-4, 1.326e-4, 6.328e-5 m**
  against a 1024-step numerical reference (reduction factors **2.070, 2.065,
  2.095**). The 512/1024-step reference difference was **4.188e-6 m**; global
  source budgets passed at all six resolutions. Initial jobs **30272896** and
  **30272902** passed but exposed an integer scatter-width warning, corrected
  before the final run. Paths freeze old velocity: this is first-order
  deterministic splitting, not measured-transport validation. Gravity/buoyancy,
  injection, wall interaction, finite-inertia stochastic dispersion, physical
  SGS/energy closure and embedded-core ambient withdrawal remain required.
  Production waterjet and all held-out physical gates remain unchanged.
- The [dynamic group exchange stage](../cases/SprayClosureValidation/group-exchange.md)
  now lets several droplet groups share cells/faces while each samples the
  updated live gas inventory. All groups commit through one carrier solve;
  an invalid later recipient or failed carrier rolls back every group. Gpudev
  **30272846** passed **22 tests** in 176.43 s, including the preceding 13 cell
  regressions. Adjacent/coincident source relative energy residuals were at most
  **3.288e-18**. Reversing group order gave one-step velocity differences
  **1.2711e-5, 3.2339e-6, 8.1562e-7 m/s** at dt=2e-5, 1e-5, 5e-6 s (ratios
  **3.931, 3.965**): consistent with a second-order local splitting difference,
  not a second-order global method. The combined carrier accepted in 17
  iterations with momentum residual **1.615e-13**. Three successive recipient
  reassignments conserved the global source inventories without creating
  another gas or liquid owner. This enables dynamic ownership but does not
  yet integrate trajectories, crossing residence times, injection/outflow or
  embedded-core entrainment. Serial group scaling, nearest-cell sampling,
  physical SGS/energy closure and all measured validation gates remain open.
- The [cell-owned MAC phase exchange](../cases/SprayClosureValidation/cell-exchange.md)
  now uses the actual adjacent dual-face masses for drag response, deposits full
  vapor momentum, and closes the source energy budget including an explicitly
  owned unresolved mechanical reservoir. Gas, liquid and that reservoir commit
  together with the carrier; any rejected source/carrier restores all old live
  inventories. Gpudev **30272831** passed **13 tests** in 107.92 s: analytic stiff
  drag and Galilean invariance, independent global half-volume budgets at an
  open boundary and interior cell for three heat fractions, common-flux reservoir
  and vapor budgets, rejected-source rollback, two carrier rejection modes and
  clean retries. Maximum source relative energy residual was **4.451e-19**;
  the accepted carrier took 11 iterations with momentum residual **2.467e-13**.
  Job **30272817** was deliberately stopped to correct test-harness velocity
  field names before the verified rerun. This is a fixed cell-owned bin group,
  not a moving parcel or embedded-core entrainment/handoff driver. Impermeable
  wall-adjacent recipients are unsupported, and physical heat/SGS partition,
  complete carrier total energy, measured transport and waterjet gates remain
  open. Production waterjet is unchanged; generated output remains ignored.
- [Independent measured-profile and momentum-budget assessment](../cases/SprayClosureValidation/measured-profile-assessment.md)
  improves the best maximum local profile error from **29.524% to 26.018%**, still
  failing the unchanged 10% screen. The PL1993 curve is a secondary author-archived
  representation, not raw measured data. Gpudev **30272732** passed six numerical
  checks but rejected all four initial profile candidates. Separate source-only
  calibration of mean versus total momentum flux gave **30272757**: all source
  normalization identities passed, all six physical screens failed. The best
  mixture/Ricou case retains a 6.15% stress/pressure flux remainder explicitly;
  it is not an SGS-energy closure and is not production enabled. The original
  holdout hash is unchanged. Physical source transferability, spatial stress and
  energy/ownership coupling, measured droplet transport and waterjet validation
  remain open; the two-Gaussian fit also misses the source's outer relative profile.
- The [common-flux carrier step](../cases/SprayClosureValidation/coupled-carrier.md)
  now joins momentum, pressure, dry-air/vapor mass and enthalpy in one carrier
  transaction. Gpudev **30272691** passed **14 tests**, including actual evaporating
  droplet increments, all budget/rollback checks, a 32-step contact crossing and
  a 12-step variable-density vortex. Multistep testing exposed and corrected a
  closed-domain trial-density mean incompatibility; no conserved field or final
  EOS gate was modified. Momentum residuals in the vortex were at most
  **1.520e-11**. Production waterjet remains unchanged. Core/liquid ownership,
  complete total-energy treatment, higher-order coupled transport and physical
  SGS/entrainment/dispersion closure still require integration and validation.
- A separate measured-profile source lead is recorded in the
  [DNS assessment](../cases/SprayClosureValidation/dns-profile-assessment.md):
  Shin et al. Figure 2 compares Panchapakesan & Lumley (1993), distinct from the
  Hussein holdout. Qualify the original measurement/overlay provenance before
  considering calibration; none of those experimental overlays has been fit.
- The [inertia-consistent pressure primitive](../cases/SprayClosureValidation/momentum-pressure.md)
  now applies pressure using staggered momentum inertia and projects transport
  with coefficient `rho_transport/rho_dual`. Gpudev **30272522** passed **12 tests**:
  independent dense matrices, second-order continuum convergence, pressure impulse,
  local/global mechanical-work ledgers, and rejected incompatible sources.
  Maximum tested relative continuity residual: **3.776e-12**. The full FV gradient
  retains transverse open-end pressure gradients. The next required integration
  is a common final-flux momentum/species/enthalpy/EOS iteration and transactional
  source ownership. Neither this primitive nor the preceding predictor is yet
  a production coupled LES timestep; all experimental validation gates stay open.
- A [conservative staggered momentum predictor](../cases/SprayClosureValidation/momentum-transport.md)
  now uses the accepted carrier mass flux on half-cell dual control volumes.
  Gpudev **30272483** passed **10 tests**, covering all x/y boundary combinations,
  uniform source-fed motion, actual scalar-stage contacts, full evaporating-core
  impulse, boundary momentum/kinetic-energy ledgers, and rollback. Largest dual
  mass residual: **2.187e-16**. Wall impulses and numerical energy loss are
  explicit; numerical dissipation is not assigned to physical SGS energy.
  The next integration requirement is a pressure correction using
  `rho_transport/rho_dual` and a common final-flux momentum/species/EOS iteration.
  A sequential scalar step followed by this predictor is not a validated LES
  timestep. Production cases remain unchanged; all physical gates remain open.
- A moving-contact audit exposed a numerical defect in the new carrier stage:
  conserved fields and EOS passed while uniform translation acquired a **7.2%**
  temperature-contact velocity error that persisted under refinement. Matching
  thermodynamic donor density in the projected mass flux and recovered velocity
  fixed it. Gpudev **30272365** passed all eight contact cases (maximum velocity
  error **1.951e-11 m/s**), and **30272366** passed **18 regression tests**.
  The explicit advective-flux contract and tolerance sensitivity are documented
  in [low-Mach transport](../cases/SprayClosureValidation/low-mach-transport.md).
  Conservative MAC momentum integration is still required. This new-stage bug
  does not establish the cause of the original waterjet discrepancy.
- The latest central-momentum/upwind-scalar waterjet remains at **8/9** passing
  sensors, with centre temperature **35.33465 C** versus **31.4 C** measured.
  Conservative MUSCL scalar transport did not close that discrepancy.
- The initial integral-jet/homogeneous-Langevin components passed **13 numerical
  tests** on gpudev job **30271912**. Independent profile assessment job
  **30271957** completed and failed the accuracy gate: maximum local normalized
  velocity-profile error **37.805%**, despite small spreading-rate error.
- The components remain disabled in production waterjet cases. See
  [component assessment](../cases/SprayClosureValidation/README.md),
  [scalar-transport investigation](../cases/WaterSprayMontazeri2015/investigation/scalar_transport/README.md),
  and [measured-transport data audit](../cases/FreeWaterSprayTransport/README.md).
- Public measured water-spray data are now acquired: the
  [Rácz PDA candidate](../cases/WaterSprayRacz2025/README.md) has 90 stations
  and 2,430,722 joint droplet records, with 34 source and 56 held-out stations.
  Gpudev job **30272001** passed seven importer tests and the full checksum/layout
  audit. Raw-data availability is resolved for this candidate; carrier inputs,
  flux-weight corrections, source asymmetry and physical applicability still
  need qualification. No measured-transport pass is claimed.
- Conservative gas-inventory operations now cover bounded ambient withdrawal,
  nonlocal shell intake and handoff without duplicate gas ownership, retaining
  mean-motion mixing energy in an explicit unresolved reservoir. Gpudev job
  **30272031** passed **20 tests** (seven new coupling and thirteen existing
  closure tests) on the A100. See the [coupling contract](../cases/SprayClosureValidation/coupling.md).
  LES mass/pressure integration, parcel ownership and physical SGS-energy/
  dispersion closure remain outstanding; this is numerical verification only.
- Profile job **30272071** assessed two independently specified alternatives
  against the unchanged Hussein target. A Gaussian using Huck et al.'s separate
  measured coefficient reduced maximum error from **37.805% to 29.524%**, still
  failing the 10% screen. The constant-eddy-viscosity profile failed at **66.099%**.
  Three new numerical checks passed; no alternative is enabled in the waterjet.
  See the [profile assessment](../cases/SprayClosureValidation/profile-assessment.md).
- Inhomogeneous tracer consistency now has a derived stationary isotropic
  Gaussian drift and a [well-mixed verification](../cases/SprayClosureValidation/inhomogeneous-dispersion.md).
  Gpudev **30272114** passed four numerical tests and all refined ensemble
  screens (262,144 tracers, two seeds, three timesteps): density error at most
  **3.15%**, versus **171–173%** for uncorrected controls. The first smaller
  ensemble's initial sampling failure is preserved. Finite-inertia, sheared,
  anisotropic dispersion and the physical SGS-energy closure are still missing.
- Measured inlet gas-proxy assessment **30272161** passed 12 software tests and
  processed only the 34 source stations. The published nominal filter leaves
  **3–1,238 events/station**, with **32/34** below the preliminary count screen;
  edge estimates vary strongly across time blocks. Thus this proxy does not yet
  qualify the carrier inlet. The 12 mm estimator length convention is resolved
  from the author code. See [source assessment](../cases/WaterSprayRacz2025/gas-proxy-assessment.md).
- Finite-gas [core interphase exchange](../cases/SprayClosureValidation/core-exchange.md)
  now connects shared droplet drag/evaporation laws to the owned gas inventory.
  Gpudev **30272202** passed **14 tests**, including a multibin matrix-exponential
  reference, evaporation species/momentum/enthalpy-plus-mechanical conservation,
  coupled timestep refinement, heat-exhaustion rejection, and the complete local
  withdrawal–exchange–handoff cycle. The caller must supply a physically justified
  mechanical heat/unresolved-energy partition; no production value is assumed.
  This is local numerical verification. Spatial core transport, measured
  validation, pressure/EOS coupling and a justified SGS closure remain incomplete.
- Independent [DNS profile calibration](../cases/SprayClosureValidation/dns-profile-assessment.md)
  now assesses a positive two-Gaussian shape fitted only to Shin et al.'s separate
  stationary jet. Gpudev **30272244** passed six numerical reconstruction checks,
  then rejected all three candidates: best maximum local error **38.513%**.
  One exact source-export duplicate was removed and both assessment revisions
  retained. The source DNS outer-tail range lies below the holdout's 10% interval;
  added empirical shape flexibility does not establish transferable accuracy.
  The best earlier profile remains **29.524%**, still above the required 10%.
- An opt-in [moist transport/EOS-pressure stage](../cases/SprayClosureValidation/low-mach-transport.md)
  now uses one corrected mass flux for dry air, vapor and enthalpy, iterating to
  EOS consistency without overwriting inventories. Gpudev **30272312** passed
  **14 tests**, including actual core-evaporation sources, boundary budgets,
  returned-velocity flux consistency, float32, transactional rejection, and
  analytic vented-heating timestep refinement. The evaporation case converged
  in five iterations (relative EOS error **7.051e-10**) and correctly drew
  ambient gas inward during cooling. This first-order stage still needs the
  conservative momentum/SGS predictor, pressure-work accounting, spatial core
  routing, higher-order integration, and production waterjet integration.
- Next: investigate an independently supported profile/entrainment closure,
  separating reconstruction error from missing momentum/stress physics; qualify
  the numerical source/target data for measured droplet transport. Continue
  conservation/interface work that does not depend on unavailable measurements.

---

# Historical validation objectives and handover

All status and stop instructions below predate the active objective above.
They are retained for provenance, not as current instructions.


## Stopped by the user — handover, 2026-09-16

The user requested a handover and stopped this investigation. The goal
controller reports **paused**; completion has not been claimed. Both running
investigation steps were interrupted and verified stopped, with checkpoints
preserved. Do not automatically resume them.

**Latest modeling direction:** this implementation must work on very coarse
grids, so unresolved spray mixing must be modeled. Resolving the jet through
finer LES grids is not the intended solution. This instruction supersedes any
older wording that implies otherwise.

Read [the standalone handover](water-spray-handover.md) in the next chat. It
records corrected code, completed evidence, exact stopped checkpoints and the
remaining coarse-grid closure questions. All investigation/status text below
is historical as of this stop, including former live process references.


## Cooling discrepancy investigation before the stop, 2026-09-16

The user explicitly requested a persistent goal to understand and fix the
discrepancy against `waterjet_referece.pdf`. That goal has now been created and
was active until the user-requested stop above. This instruction reopened cooling validation and superseded the
transport-only scope and prohibition on further cooling/wall investigations
recorded below. The earlier text is retained as historical context.

Baseline evidence is in
[the uploaded-reference validation report](../cases/WaterSprayMontazeri2015/uploaded_reference_validation/README.md).
The PDF is Sureshkumar et al. (2008); Table 2(b)'s final row exactly matches the
existing nine-sensor reference. The latest archived inertial case passes 8/9
sensors, with the middle-centre prediction 35.693°C versus 31.4°C measured
(13.67% error). Current focused implementation checks pass 56 tests. These
results do not establish experimental validation.

Required outcome: explain the discrepancy, implement physically justified
corrections, and demonstrate every-sensor agreement under the existing 10%
Celsius criterion, together with conservation and timestep/grid/parcel-sampling
and statistical convergence evidence. Preserve baseline results. Do not tune
source or closure parameters to held-out temperatures, silently substitute a
mean-only criterion, or equate pre-collector water temperature with the measured
collected-water observable. Keep new files and temporary data in the workspace.

Next investigation: audit source distributions, trajectory/thermal coupling,
carrier transport/mixing and observation locations, then use controlled runs to
distinguish implementation errors from physical-model and apparatus effects.
The existing fine/refined free-slip controls and smooth-wall results must remain
separate so changes in boundary physics are not reported as grid convergence.

### First reopened investigation: thermal boundary conservation

The carrier's legacy scalar boundary overwrites physical end cells after parcel
exchange. An opt-in `scalar_boundary="flux"` now retains that heat and prescribes
incoming advective boundary flux instead. The legacy default and archived cases
remain reproducible. Six new boundary tests plus existing regression tests pass
(29 total). The unchanged-boundary control reproduces the archived sensor
histories within 5e-11 K and shows a sampled sensible-energy residual of 106.7 W.
The completed matched flux-boundary run lowers that residual to 4.05 W and the
centre DBT error from 13.67% to 13.05%, with 8/9 sensors passing. No source or
turbulence parameters changed. The correction is justified but does not close
the discrepancy. See the [completed boundary investigation](../cases/WaterSprayMontazeri2015/investigation/carrier_boundary/README.md).

Both new runs record gas sensible-energy storage and paired vapor measurements.
The control's reconstructed centre WBT is 25.21°C versus the measured 20.2°C,
which adds a constraint on the concentrated hot/humid core. The simulation
observation plane still precedes the unmodeled drift eliminator. No validation
pass is implied by the correction or test results.

### Completed feedback correction and active convergence checks

Parcel momentum feedback was advected before pressure projection, producing a
0.146% spurious humidity variation in a source-free diagnostic. Projecting the
impulse after enforcing the prescribed inlet removes that variation to roundoff.
The opt-in correction passes inlet/interior tests and the completed 4 s run has
8/9 passing DBT sensors, with the centre error 12.9899%. Its sampled sensible
residual is 3.945 W. This correction is justified, but does not resolve the main
discrepancy. See [feedback results](../cases/WaterSprayMontazeri2015/investigation/feedback_projection/README.md).

Halving dt with the same physical parcel rate changes any sensor by at most
0.006624 K; the centre error remains 12.9788%. Doubling parcel count changes
sensors by at most 0.086472 K, still 8/9 passing. The centred-advection control
also completed, with centre error 12.5307% after only a 0.144 K decrease.
MUSCL-MC remains the default; no physical spray/inlet/AMD parameter changed.
See [numerical controls](../cases/WaterSprayMontazeri2015/investigation/projected_controls/README.md)
and [advection diagnostic](../cases/WaterSprayMontazeri2015/investigation/central_advection/README.md).

The matched 128×48×48 spatial control runs in exec session 96094. A 12 s
statistical control with a longer nonrepeating Fourier inflow runs in session
8434; both use Slurm allocation 30266535. Its 3072-point, 37.8 m box retains
spectral spacing and physical turbulence statistics but is a new realization,
not an exact restart. Compare 4–8 s with 8–12 s. Check logs and completed
summaries before restarting either process. Short-run 2–3 s and 3–4 s blocks
already differ by up to 0.33 K; statistical convergence is not established.

The latest field diagnostics show a narrow, fast, hot/humid centre core.
Momentum-limiter damping is a hypothesis being tested, not an established cause.
Statistical, grid and parcel convergence, apparatus compatibility and global
momentum evidence remain outstanding. The paper's reported ±0.3°C DBT uncertainty
also excludes an additional unquantified mist-carryover effect discussed in
Section 2.7; that caveat does not alter the declared 10% acceptance criterion.

### Current mechanism evidence and next decisions

At the completed corrected t=4 s checkpoint, maximum liquid concentration lies
at downstream wall corners, with 24% of liquid inventory touching walls and
15.56% deposited into cells above volume fraction 0.001. The idealized wet-wall
rule remains a model limitation; this is not evidence of a dense nozzle alone.
The first-order scalar operator's interior variance loss is 8.92 times AMD
for temperature and 12.49 times for vapor. The exact face-sum identity was
cross-checked against the solver to roundoff. These are instantaneous spatial
operator diagnostics, not a quantified temperature bias or a full budget.

Next: process the completed refined and long-duration runs before selecting
another change. Compare the refined mesh against **half_dt**, not the old
free-slip refinement; compare long-run 4–8 s and 8–12 s means at every sensor.
If spatial changes remain material, separate scalar-advection accuracy from
CIC source support and unresolved wall concentration. Do not infer convergence
from a passing single grid. Temperature and humidity station alignment with the
unmodeled eliminator must also be resolved before declaring validation.

An optional user question is pending about independent nozzle velocity/angle
measurements and drift-eliminator dimensions. Numerical work continues while
waiting; no approval or answer is needed to finish the active simulations.

## Historical transport-only objective (superseded)

User-directed scope replacement, 2026-09-16. This supersedes the Montazeri
wall-bounded discrepancy investigation as the working objective.

Validate **unresolved water-droplet transport**, preferably in an unconfined
air-assisted spray against experimental downstream droplet measurements. Nozzle geometry
and primary breakup remain subgrid. Stop investigating wall impacts, films,
drift eliminators, sump collection, and the old tunnel temperature discrepancy.
Preserve that work as historical evidence, not a completion gate.

## Required outcome

1. Select an airborne-spray transport experiment with measured droplet velocities, sizes and
   spatial distributions at multiple downstream locations. Prefer accessible
   numerical data and measured gas-flow conditions. Confirm that observation
   stations are outside primary breakup and unaffected by collecting surfaces.
2. Prescribe a documented post-breakup droplet/air handoff at the upstream
   boundary. Reserve downstream stations for validation. Do not derive the
   source distribution or tune drag/mixing from those held-out targets.
3. Implement and run a transport-only case with negligible or disabled
   evaporation. Prefer remote open boundaries; retain experimentally required
   carrier boundaries when their effects cannot be neglected. Wall-impact
   physics must not determine the selected transport observables.
   Represent the carrier air jet, or prescribe independently supported gas data;
   air-assisted transport cannot be tested by silently discarding air momentum.
4. Apply the user-approved **10% relative-error tolerance to every selected
   measured nonzero observable**, not only an average or aggregate score.
   Predeclare observables, normalization, uncertainties and treatment of
   near-zero velocity components before evaluating. A near-zero component needs
   an explicit absolute-error criterion, not an arbitrary denominator floor.
5. Check timestep, parcel sampling, spatial resolution, domain-size effects and
   mass/momentum conservation. Retain source provenance, reproducible cases,
   numerical results, limitations and appropriate tests.
6. Complete when transport agreement and these checks are established. This
   goal does not require validation of atomization, evaporation, cooling or
   negative buoyancy. Retain shared humidity/cloud compatibility in the feature;
   thermal validation is a separate future task.

All new files, downloads, caches and test temporary files stay in the workspace;
never write to `/tmp`. No more runs of the superseded wall-impact investigation.

## Initial re-evaluation

Montazeri/Sureshkumar is unsuitable as the primary **transport-only** case:
its selected validation observables concern a confined cooling apparatus.
An initially promising candidate is Xia et al. (2018), *Spray characteristics of free
air-on-water impinging jets*, DOI 10.1016/j.ijmultiphaseflow.2017.12.007.
It uses air and water and measures droplet velocities and sizes with PDA.
Jet-on-jet impingement in this title is the atomization mechanism, not impact
on a solid wall. We will start downstream of this mechanism.

Candidate acceptance is conditional on obtaining adequate upstream handoff
and independent downstream measurements; an abstract's peak velocity or SMD
alone is insufficient. See the candidate assessment in
[cases/FreeWaterSprayTransport](../cases/FreeWaterSprayTransport/README.md).

## Re-evaluation decision

The old temperature comparisons cannot establish a pass or fail for this new
transport objective. No qualified transport run or per-sensor error table exists
yet. Do not continue the old parameter studies or transfer their acceptance
scores to this task.

The first milestone is now **qualification of independent source and validation
data**. Xia's detailed velocity profiles are at z=75 mm (Figures 19–20), while
Figure 14 supplies downstream size statistics. Those statistics alone do not
specify the size-conditioned injection velocities and gas field. The KIT 2022
paper supplies water-spray and airflow measurements, but its reported stations
share z=40 mm. Neither paper is presently an accepted complete transport case;
additional source data or another experiment is needed. This is a data-selection
limitation, not evidence that the implementation fails the 10% requirement.

After qualification, freeze the input dataset and held-out sensor list before
running: each nonzero measurement must satisfy
`abs(simulated - measured) / abs(measured) <= 0.10`. Report individual errors and
the maximum; a passing mean cannot compensate for a failed sensor. Treat values
indistinguishable from zero using a separately justified absolute criterion.

A stronger source/target split has now been identified in the Rüger experiment
revisited by Khaled et al. (2026): measured injection at 25 mm and independent
profiles at 50, 100 and 200 mm. Qualification of its numerical data is the next
step. Its carrier enclosure requires assessment, but enclosure geometry alone
does not contradict the user's exclusion of wall-impact validation. The earlier
unconfined-only wording was an unnecessarily strict interpretation, not an
additional user requirement.

## Goal-controller limitation

The latest session goal-controller check reports the old goal as active; an earlier check reported it paused. Neither status changes the user-directed scope replacement. Its tools support
creation and complete/blocked status, but not editing an unfinished objective.
An explicit replacement attempt was rejected because that goal is unfinished.
The old objective has not been marked complete. This document
records the user's replacement objective and governs subsequent working scope
until the session-level objective can be edited by the application.

## Completion and blocking audit, 2026-09-16

Status: incomplete; blocked on a qualified experimental input dataset after
repeated source checks across three consecutive continuation turns. The last
two turns made implementation progress, but did not remove this same data gap.

| Requirement | Authoritative evidence | Status |
| --- | --- | --- |
| Suitable transport experiment | Rüger/Khaled has separate source and observation planes; assessment in case README | Candidate identified; data qualification incomplete |
| Independent measured source | No Rüger inlet dataset in workspace; KIT audit has no verified coordinate mapping or upstream plane | Missing |
| Runnable literature case | Case directory contains assessment, audit and unsent request, not a measured case configuration | Missing |
| Transport-only physics | Parcel thermal-exchange switch and side-escape option; 13 parcel tests passed | Prerequisites verified |
| Every-sensor 10% comparison | No qualified run or prediction/measurement table | Not evaluated |
| Numerical convergence | No selected literature case to refine | Not evaluated |
| Conservation | Unit tests cover thermal ledgers, transport-only axial momentum and side escapes | Unit-level evidence only |

The current sources cannot supply a defensible numerical inlet distribution
without inventing unmeasured inputs. Further source or turbulence tuning would
not establish experimental validation. The prepared data request has not been
sent. Resume when the measured inlet and held-out observations are supplied,
or another documented dataset with the necessary source conditions is available.

The persistent controller still stores the superseded Montazeri objective and
cannot edit it. Its blocked status records the present inability to continue
the user-directed replacement task, not completion of the old investigation.
