# ADR-0006: Restart and forcing-time laws

Status: **Accepted**

## Context

ADR-0003 makes an integrator a higher-order interpretation of a pure vector
field. That separation is incomplete unless the program defines which data is
persistent across restart and which physical time every vector-field evaluation
observes.

Multistep integrators require history beyond the current prognostic fields.
Runge–Kutta methods evaluate the vector field at stage times. Closures,
particles, concurrent domains, averaging diagnostics, and scheduled injection
may also carry memory or discrete event state. Letting each module infer time
from an integer step would make restart and integrator changes alter the
physical program.

## Decision

### Accepted-state clock

`State.time` is the physical time of the last accepted full-step state. A
vector-field evaluation receives an explicit `EvaluationTime`; physics modules
MUST NOT derive physical time from `step * dt`, a process-local counter, or a
global clock.

For a staged method, evaluation \(i\) observes the method-defined time

\[
t_{n,i}=t_n+c_i\Delta t.
\]

Prescribed boundary data, actuator and fringe forcing, continuous injection or
cooling rates, radiation, surface fluxes, and other time-dependent terms use
that same evaluation time unless an explicit time policy says otherwise.

### Time policies

Stage-consistent evaluation is the default. A forcing that is intentionally
held fixed over a complete step must use a separately named `FrozenOverStep`
policy. The policy samples once at the documented step time and makes the held
value an explicit input to all stage evaluations.

Freezing MUST NOT be an optimization silently introduced by an interpreter.
Changing time policy changes the semantic program and therefore its
configuration fingerprint.

### Continuous rates and discrete events

A continuous physical rate belongs to the vector field and is integrated by the
chosen method. A discrete event is a separately typed transition at a named
accepted-step boundary. It MUST NOT be disguised as a vector-field rate or be
executed once per RK stage accidentally.

Examples requiring explicit classification include parcel injection, periodic
closure-coefficient updates, controller samples, mesh/ownership changes, and
concurrent-domain mailbox advancement.

### Checkpoint boundary

The first implementation supports checkpoints only at accepted full-step
boundaries. Mid-stage checkpointing is out of scope. Stage arrays and stage
diagnostics are ephemeral.

A checkpoint contains every value needed to reproduce both future states and
declared diagnostic streams:

- prognostic fields and accepted physical time/step;
- method-specific integrator history;
- closure and filter memory;
- explicit RNG state;
- Lagrangian and other coupled subsystem state;
- persistent diagnostic/window accumulators;
- discrete-event schedules and counters;
- concurrent-domain time alignment and persistent mailbox state;
- the versioned `ScaleSystem`;
- semantic, numerical, time-policy, and decomposition-independent fingerprints.

Transient halos, reconstructed boundaries, gradient bundles, FFT/linear-solver
workspaces, compiled executables, loggers, and ordinary stage arrays are not
persistent state.

The distributed checkpoint contract from the effects specification and the
scaling metadata contract from ADR-0005 still apply: each rank serializes owned
data only, and no process gathers a global field.

## Diagnostic time semantics

A full-step diagnostic emitted after acceptance is labelled at \(t_{n+1}\).
A diagnostic of an intermediate vector-field evaluation carries its exact
`EvaluationTime` and stage identity. Time-window diagnostics define whether
samples are point values or quadrature contributions and persist their
accumulator state across checkpoint.

Diagnostics MUST NOT trigger an extra stateful physics update merely to report
a value.

## Required laws

### Restart equivalence

For an accepted initial state \(s_0\),

\[
\operatorname{run}_m(
  \operatorname{load}(\operatorname{save}(\operatorname{run}_n(s_0))))
=\operatorname{run}_{n+m}(s_0).
\]

The equality is bitwise where the accepted backend promises determinism and
otherwise uses a declared tolerance. It covers persistent diagnostics and event
schedules, not only prognostic arrays.

### Stage-time correctness

Each vector-field evaluation and every stage-consistent forcing term observes
the method-defined \(t_n+c_i\Delta t\). A manufactured time-dependent forcing
test MUST fail if any stage uses \(t_n\) or \(t_{n+1}\) incorrectly.

### Frozen-policy coherence

All stage evaluations under `FrozenOverStep` observe exactly the same explicitly
sampled forcing value. The sample time is part of the policy contract.

### Event uniqueness

A discrete accepted-step event executes exactly once when its schedule is
crossed, including across save/load boundaries. It never executes once per
stage or twice after restart.

### Integrator independence of physics

Changing the integrator changes evaluation times and numerical composition but
does not change the semantic definition, units, or dependencies of the vector
field.

## Consequences

AB2 must checkpoint its required previous tendency/history. An RK method need
not checkpoint ordinary stage arrays because checkpoints occur only after full
step acceptance. Closure update cadence, particle injection, and concurrent
mailbox advancement become explicit event/rate decisions rather than hidden
integer-step branches inside unrelated physics functions.

The first concrete integrator may now be proposed, but it must document its
coefficients, persistent history, forcing evaluation times, projection points,
and diagnostic acceptance point against this ADR.
