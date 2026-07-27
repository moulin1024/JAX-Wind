# ADR-0003: Integrator as a higher-order interpretation

Status: **Accepted**

## Context

When physics, forcing-time choices, stage updates, logging, and checkpoint
history are fused into one step function, changing the integrator changes the
meaning of the model. AB2 and Runge–Kutta methods also require different
persistent or ephemeral state, so pretending they share an unqualified state
record makes restart semantics ambiguous.

## Decision

The physical model is a pure vector field: it evaluates a read-only state and
an explicitly time-labelled environment and returns tendencies plus evaluation
diagnostics. An integrator is a higher-order interpretation that converts this
vector field into a transition.

```text
VectorField : Evaluation[State, Time, Forcing] -> Tendency × Diagnostic

Integrate(method, time_law, VectorField)
    : PersistentIntegratorState -> PersistentIntegratorState × Diagnostic
```

The vector field MUST NOT know whether it is evaluated as an AB history term,
an RK stage, a manufactured-solution probe, or a Jacobian-vector product.

No concrete integrator is selected until two subordinate contracts are
accepted. Both are now fixed by
[ADR-0006](0006-restart-and-forcing-time-laws.md):

1. the restart law specifies exactly which history, time, closure, and RNG
   values are persistent;
2. the forcing-time law specifies the physical time used by boundary data,
   actuator/fringe forcing, pressure forcing, and diagnostics at every
   evaluation or stage.

Stage arrays are ephemeral unless the accepted restart model explicitly allows
mid-step checkpointing. Diagnostics MUST declare whether they describe vector
field evaluations, accepted full steps, or time-window reductions.

## Required laws

- Integrating the zero vector field leaves prognostic state unchanged.
- Equivalent vector-field compositions give equivalent integrated transitions
  under the method's stated tolerance.
- Restarted and uninterrupted transitions agree according to the checkpoint
  law.
- Forcing and diagnostics observe the time defined by the forcing-time law.
- Changing integrator does not change the semantic definition of a tendency.

## Consequences

AB2, RK3, or another method becomes an interpreter choice rather than a physics
mode flag. Persistent integrator state is method-specific but explicit. The
first concrete interpretation is now selected by
[ADR-0009](0009-fixed-step-ab2-projection.md); restart and forcing-time laws
remain its admission criteria.
