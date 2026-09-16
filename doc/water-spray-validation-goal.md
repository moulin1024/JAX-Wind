# Revised water-spray validation goal

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
