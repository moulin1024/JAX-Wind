# ADR-0009: First integrator is fixed-step AB2 with one terminal projection

Status: **Accepted**

## Context

ADR-0003 requires an integrator to interpret a pure vector field. ADR-0006
fixes accepted-state clocks, evaluation-time semantics, and restart contents.
Milestones A--D now provide typed owned fields and one compatible projection
program. The first complete deterministic transition can therefore be chosen
without embedding a time method in physics.

## Decision

The first method is fixed-step Adams--Bashforth 2. For an accepted projected
state \(u_n\), evaluate the vector field exactly once at \(t_n\):

\[
F_n = F(u_n,t_n,E_n).
\]

After a cold start, use explicit Euler:

\[
u^* = u_n + \Delta t F_n.
\]

Once history exists, use

\[
u^* = u_n + \Delta t\left(\frac{3}{2}F_n-\frac{1}{2}F_{n-1}\right).
\]

Apply the compatible pressure projection exactly once to \(u^*\). The normal
velocity boundary supplied to that projection is sampled at the accepted
target time \(t_{n+1}\), because it constrains the newly accepted state rather
than the vector-field evaluation at \(t_n\). The result is the only accepted
prognostic state.

The method has no RK stages. `EvaluationTime` is therefore \(t_n\) with the
identity `ab2-current`, while the emitted full-step diagnostic and accepted
clock are labelled \(t_{n+1}\). Continuous forcing is evaluated by the vector
field at \(t_n\). Discrete events remain separate accepted-boundary
transitions and are not introduced by this milestone.

## Persistent state and checkpoint boundary

The persistent AB2 state contains:

- the projected prognostic velocity at the last accepted time;
- accepted physical time and integer accepted-step count;
- either the explicit `ColdStart` tag or `PreviousTendency(F_n)`;
- a versioned fingerprint containing the method, fixed `dt`, projection law,
  and forcing-time policy.

After accepting step \(n+1\), the saved history is \(F_n\). Pressure,
candidate velocity, boundary/gradient contexts, solver workspaces, and step
diagnostics are ephemeral. Checkpoint save/load occurs only between accepted
steps and writes owned payloads only.

Changing `dt` invalidates existing AB2 history and therefore changes the
fingerprint. Variable-step AB2 is not silently approximated by the fixed-step
coefficients; it requires a later decision and an explicit previous-step size.

## Why this method first

AB2 needs one vector-field evaluation and one projection per accepted step,
matching the intended LES execution cost. It also forces restart history and
accepted-time semantics to become executable now. RK3 remains a later tagged
integrator for problems whose stability requirements justify additional
vector-field evaluations and projections.

## Required laws

- The zero vector field leaves an already projected state unchanged.
- The first step uses Euler coefficients and every later step uses AB2
  coefficients.
- Every vector-field evaluation observes exactly the accepted \(t_n\).
- The normal boundary and accepted diagnostic observe \(t_{n+1}\).
- Every accepted state satisfies the compatible divergence tolerance.
- Saving and loading after any accepted step gives the same continuation as an
  uninterrupted run, including the cold/history tag and evaluation times.
- Reference, local-device, and multi-process z-slab interpretations commute
  within their declared precision tolerance.
- Float32 execution preserves float32 state and tendency history.

## Consequences

The first deterministic step is intentionally not a Navier--Stokes model. It
is a lawful higher-order transition that can interpret any admitted pure dry
vector field. Adding advection, buoyancy, SGS, turbine, or fringe terms extends
the vector field product; it does not alter AB2 semantics.
