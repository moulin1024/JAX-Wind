# Persistent goal: preserve the log law and eliminate upstream oscillations

User request: make this a persistent goal (2026-09-16).
Status: active; the two requirements have not yet been met together.

## Objective and current direction

Preserve the central neutral-ABL mean profile and wall stress while eliminating
upstream numerical oscillations in the small single-V80 case. Start from the
user-directed central-advection plus downstream-damping-zone approach. Use small
cases; do not return to full-farm debugging. Do not hide alternating face modes
by evaluating only cell-centered velocity.

## Evidence required before completion

- Matched turbulent log-law experiment from the developed DTU checkpoint:
  1800 s spin-up, 3600 s time/volume averaging, Porté-Agel enabled, full RK3,
  CFL ceiling 0.9. Compare four streamwise quarters, identifying the sponge
  region separately.
- Preserve the existing declared screens: corrected quarter log-law RMSE at
  most 1.0 m/s; profile change from central at most 0.2 m/s; log-law RMSE
  increase at most 0.1 m/s; near-wall maximum quarter-profile change at most
  0.2 m/s; relative wall-u* change at most 5%; split-window profile drift at
  most 0.2 m/s; sampled divergence at most 1e-5 /s.
- Matched 300 s single-V80 test with raw-face diagnostics. The existing minimum
  acceptance gate is at least 90% reduction in upstream second-difference RMS,
  with inlet error at most 1e-5 m/s and divergence at most 1e-5 /s. Inspect
  residual spatial/temporal grid modes before describing them as eliminated.
- Keep acceptance thresholds fixed. Distinguish outlet attenuation, lateral
  entrainment, streamwise reversal, and upstream grid-scale oscillations.
- Record validation scope: preserving the central reference does not remove
  its existing log-law bias or prove converged turbulent wake accuracy.

## Current completed result

`outputs/backflow_sponge_validation_30270853` contains the completed experiment.
Central advection plus a quadratic outlet sponge starting at 0.75 Lx, with a
5 s outlet relaxation time, passes all log-law checks:

- Log-law RMSE: central 0.611299 m/s; sponge 0.613041 m/s.
- Maximum quarter-profile change outside sponge: 0.003035 m/s.
- Wall u*: central 0.430758 m/s; sponge 0.428006 m/s (0.639% change).
- Upstream oscillation RMS: central 0.0397812 m/s; sponge 0.0398294 m/s.
  This fails the oscillation gate.
- Minimum sampled outlet u rises from 3.32188 to 9.52201 m/s, but lateral
  inward volume flux increases. Neither wake run samples downstream reversal.

The outlet sponge is implemented and available in
`cases/HornsRev1/fv_v80_small_central_sponge.toml`.
See `cases/HornsRev1/central_outlet_sponge.md` for implementation and plot links.

## Earlier investigations and constraints

- Full MUSCL and horizontal MUSCL in open flow damaged the wall layer.
- Global weak fourth-difference damping also reduced wall u* excessively.
- Wider streamwise AD-BEM force projection alone reduced oscillations by about
  65–68%, below the 90% gate. Legacy disk widths were 0.32–3.93 m on a 16 m
  streamwise mesh; the ordinary smoothing_width_m field does not control that
  legacy disk projection.
- A rotor-local stabilization trial was stopped at the user's request when
  switching to the downstream damping-zone approach. Do not silently resume it.
- Experimental options remain in the working tree, disabled in the sponge case.
  Automatic approval rejected a proposed bulk cleanup because of possible
  loss/corruption of unrelated work. Preserve the heavily modified workspace.
- GMG was already used with small divergence; changing the preconditioner
  alone did not address the observed upstream modes.
- Oscillations were previously observed before the wake reached the outlet.
  The next investigation should distinguish internal mode generation and
  propagation from boundary reflection, while respecting the user's current
  central-scheme direction.

## Resuming

Check the current host/allocation and GPU availability; do not assume allocation
30270853 still exists. The last host was ravg1130 and the Python environment was
`/u/limo/venvs/numba_cuda_waterboa/bin/python`, with `cuda/13.0` and
`PYTHONPATH=src:tools`. Source/input snapshots are stored with experiments.

Use fresh output directories. The validation runner binds prepared inputs to
source hashes and does not automatically resume interrupted experiments.
The completed numerical run's reports were relabeled afterward to distinguish
log-law PASS from combined wake FAIL; postprocessing provenance is recorded.

Do not mark the persistent goal complete until both requirements are supported
by the matched numerical evidence above.

## User constraint and origin controls (2026-09-16)

The user objected to over-smoothing because it compromises AD-BEM. Preserve the
existing aerodynamic sampling and annular force widths; do not pursue broader
normal/radial kernels as the oscillation fix. Keep central advection and the
damping-zone direction. Any proposed remedy must assess rotor-loading fidelity
as well as log-law retention and raw-face oscillations.

Completed 300 s no-actuator controls on the same 128×64×64 uniform-flow grid,
central/RK3, dt=0.25 s, and downstream sponge:
- Rough wall retained: upstream raw-face second-difference RMS 0.000185157 m/s,
  maximum sampled divergence 6.15e-8 /s, maximum sampled CFL 0.16658.
- Wall stress removed: exactly uniform flow; zero RMS and divergence.
- Turbine plus rough wall reference: RMS 0.0398294 m/s.

Artifacts: `outputs/backflow_origin_controls_30270853`. These controls strongly
implicate actuator-generated modes, but do not establish a specific cause.
The brief radial-width experiment was stopped before a wake/log-law run, its
new edits reversed exactly, and its patch/test archived in that output directory.
The existing normal-width and rotor-local stabilization options remain disabled
in the current sponge case. No claim of a successful combined fix is justified.

## Wake-study constraint (latest user correction)

Do not extend a damping buffer toward the disk. The purpose is to study the
wake, so a treatment that damps the incoming flow/induction or the measured
wake is not an acceptable solution merely because its upstream metric passes.
Keep any outlet sponge outside the turbine/wake measurement region. Preserve
AD-BEM sampling/force kernels and solve interior modes through justified
discretization, source/projection consistency, or resolution changes.

Two upstream-buffer diagnostics used an eighth-order open-x operator ending
at 0.2 Lx with legacy actuator widths unchanged. Relaxation times 1 s and
0.25 s reduced the fixed x<=336 m RMS by 83.9% and 95.3%, respectively, but
left appreciable modes nearer the disk. These are diagnostic results, not an
accepted fix. Final-state thrust/torque changes were below 1%, which does not
establish unaltered turbulent wake physics.

No extension toward the disk was implemented. The paired buffer log-law trial
`outputs/backflow_mode_sponge_loglaw_30270853` was stopped before completing
its averaging window; it cannot establish log-law retention. Default central
plus outlet-sponge cases do not enable the upstream buffer. The optional
prototype and artifacts remain for reproducibility; do not promote or resume
this approach as the solution without addressing the wake-study constraint.

## Reconsidering MUSCL (latest user direction)

The user explicitly asks whether MUSCL/upwind can be used properly while
retaining the log law. Investigate that route; the earlier preference for
central does not preclude this user-directed reconsideration. The restrictions
against changing AD-BEM kernels or damping the turbine/wake study region remain.

A frozen-field energy audit on the developed periodic precursor is saved in
`outputs/muscl_energy_audit_30270853`. Full MUSCL adds a domain-mean kinetic-energy
sink 0.00519126 m²/s³ versus AMD 0.000753718 (6.89x). Horizontal-only MUSCL adds
0.00484141 (6.42x), streamwise-only 0.00362846 (4.81x). Central advective work
is -3.7e-10. This strongly supports excessive numerical dissipation, but is one
snapshot, not a full evolved stress/energy budget or proof of sole causation.

A conservative low-dissipation central/MUSCL face-flux blend, with measured
limiter activity and a numerical-energy budget, is a plausible next candidate.
It has not been implemented or validated. Avoid claiming that turning off AMD,
changing CFL/RK3, or enabling Porté-Agel alone solves the issue. Retain all
original physical validation requirements plus wake/rotor fidelity.

## User rejects a tunable central/MUSCL blend

The requested 5% blend test was cancelled before implementation or simulation.
No central-MUSCL blend option was added. The user requests a first-principles
approach rather than a new empirical blending coefficient. Focus next on a
consistent discrete momentum/kinetic-energy budget, staggered SGS-gradient
and stress operators, force sampling/spreading work consistency, and resolution
of actuator-generated modes. Do not promise that conservation alone suppresses
oscillations or that an energy-balanced LES closure guarantees the log law.
Do not replace the rejected constant blend with an unvalidated adaptive blend
and call it parameter-free. The existing measured MUSCL energy sink already
exceeds AMD by roughly 6.9x on the audited field, so reducing/removing a positive
SGS viscosity cannot simply offset it.

## Completed first-principles checks 1–3 (2026-09-16)

Artifacts and reproducible scripts: `outputs/operator_audit_30270853/README.md`.
Audited the saved t=300 s small V80 central checkpoint, plus manufactured
fields and one dt=0.25 s production RK3 step with frozen rotor speed. No
production numerical changes or evolved wake/log-law validation were made.

1. SGS: staggered-to-cell averaging exactly annihilates a transverse
   alternating gradient (natural RMS 0.2 /s, cell RMS zero). Actual upstream
   off-diagonal squared-gradient retention is 16.3–57.7%. This identifies a
   possible blind spot, not a proven AMD error or oscillation cause: smooth
   pure shear also correctly has zero modeled AMD transfer.
2. Actuator: sampling/spreading work pairing agrees within 2e-8 relative at
   highest matrix-product precision. However, 33/108 streamwise Gaussian
   ring weight arrays underflow entirely to zero on GPU; their normalization
   fails. A read-only reference subtracts the maximum log weight before exp,
   without changing width/support or adding a parameter. It restores force
   conservation to about 3e-8, and changes integrated frozen-state thrust by
   +2.2733% and extracted work by +2.7137% (+1.8788% each in uniform flow).
   This is a concrete numerical defect, not proof that correction eliminates
   upstream oscillations. Work adjointness alone does not guarantee normalized
   weights or adequately resolved force kernels.
3. Budget: independent central dual-volume momentum/energy flux identities
   close. One production RK3 step is reproduced bitwise. Including boundary
   resets, pressure boundary and constrained-layer work, projection loss,
   and measured float32 arithmetic gives maximum stage energy residual
   9.05e-6 and momentum residual 5.50e-10 in volume-integrated units per
   density. Final divergence 8.29e-8 /s. This is one-step bookkeeping, not a
   physical certification of the boundaries or a time-averaged LES budget.

The strongest next correction is stable Gaussian normalization, followed by
unchanged paired wake/log-law gates and rotor-load checks. It has only been
evaluated in an isolated reference kernel, not installed in production. Do
not add empirical blending, broaden kernels, or extend the damping buffer.
The combined persistent goal remains unresolved.

## Stable Gaussian normalization implemented (2026-09-16)

User authorized production implementation. `src/jaxwind/_jax/wind.py` now
subtracts a per-ring maximum before exponentiating normal/radial Gaussian
weights. Radial maxima are global over vertical partitions; zero-area faces
are excluded from the maximum. No widths, physical support, blending settings,
or production precision policy changed.

Fourteen targeted GPU tests pass at highest matrix-product precision, covering
mapped-grid quadrature, uniform sampling, force/work conservation, vertical
partition invariance, and zero-area top boundaries. Four new narrow-kernel cases
fail against the captured pre-change kernel. The existing strict normal-width
assertion fails at default GPU precision with both versions and passes at
highest precision; it was not weakened. Actual checkpoint and uniform-flow
force/MAC-work relative errors are approximately 3e-8 at highest precision.
Artifacts: `outputs/stable_gaussian_implementation_30270853/README.md`.

This supersedes the earlier statement that the correction exists only in a
reference kernel. Paired evolved wake/log-law validation has not been rerun;
the persistent combined goal remains unresolved. Do not claim normalized
weights alone remove underresolution, upstream oscillations, or log-law error.
