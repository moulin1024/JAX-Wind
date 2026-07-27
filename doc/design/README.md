# Design specification

Status: **normative design with implementation in progress**.

This directory defines the reconstruction before source code exists. The words
MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are normative. A code change cannot
override these documents silently: it must first amend the relevant design,
state the affected laws, and update the associated law tests.

Read in this order:

1. [`01-principles.md`](01-principles.md) — engineering and functional design
   constraints.
2. [`02-categorical-semantics.md`](02-categorical-semantics.md) — the small
   categorical vocabulary used by the solver.
3. [`03-effects-and-distribution.md`](03-effects-and-distribution.md) — effect
   boundaries, sharding, halos, and checkpoint semantics.
4. [`04-architecture-and-gates.md`](04-architecture-and-gates.md) — intended
   layers, dependency direction, and implementation gates.
5. [`decisions/README.md`](decisions/README.md) — accepted and proposed
   architecture decision records.
6. [`05-open-decisions.md`](05-open-decisions.md) — choices that must remain
   visible until we resolve them.
7. [`06-implementation-plan.md`](06-implementation-plan.md) — the first
   decision-gated vertical implementation slice.

## Normative hierarchy

When documents disagree, the priority is:

1. stated physical/discrete conservation laws;
2. categorical and algebraic laws;
3. public semantic types and phase boundaries;
4. backend and performance contracts;
5. implementation convenience.

Legacy behavior is evidence, not a sixth source of truth. A legacy result may
motivate a compatibility test, but a legacy data layout or call graph does not
constrain the new architecture.

## Documentation-first rule

Before a new public abstraction is implemented, its documentation MUST name:

- the objects it accepts and returns;
- the morphism or transformation it represents;
- the laws it promises;
- its allowed effects;
- its local and distributed interpretations;
- at least one falsifiable acceptance test.

Examples and pseudotypes MAY appear in these documents. They are contracts,
not an authorization to create placeholder modules.
