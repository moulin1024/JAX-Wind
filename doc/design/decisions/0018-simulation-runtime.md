# ADR-0018: Packaged simulations and a shared runtime

Status: Accepted for implementation; compute-node verification pending.

Supersedes ADR-0017's application ownership boundary. The direct staggered
finite-volume implementation, canonical SI inputs, explicit JAX state, and
compatible pressure projection remain unchanged.

## Decision

The installed package owns configuration, simulation construction, execution,
artifacts, and workflow orchestration. Numerical modules own equations and
discrete operators, and never import runtime, configuration, or applications.
Formulation state and compiled advancement remain explicit and functional.

Configuration resolves before construction. A simulation supplies initialization,
compiled block advancement and diagnostics to the runtime. Workflows bind named
simulation stages through declared artifacts, not private application helpers.
FFT, GMG and optional AMG remain pressure backends, not flow solvers.

The public entry points are `load_case`, `build_simulation`, `run`, `resume`,
and the installed `jaxwind` command. Case inheritance merges tables, replaces
arrays and scalar values, and resolves input paths at their declaring file.
Checkpoint schemas carry full evolving state and execution metadata. Historical
artifacts are not a compatibility target. Numerical algorithms and existing
fixed/adaptive timestep policies must be preserved during migration.

## Verification constraint

Development is on a login node. File inspection and edits are permitted;
Python execution, tests, builds, installation and numerical verification run
only on a compute node. Supply verification scripts; never infer a successful
test result from static inspection. Numerical acceptance remains pending until
the compute-node report is reviewed.

## Migration acceptance

All active formulation families must use package services; workflows must not
own numerical stepping; installed code must not import `applications`.
Small deterministic trajectories, conservation/projection residuals, resume
equivalence and differentiated inlet behavior are the numerical gates.
Documentation must distinguish implemented behavior from pending migration.

## Atmospheric moisture extension

The optional recorded-inflow moisture state extends the seven atmospheric fields
with a named water record (vapor, cloud liquid/ice, spray liquid and droplet number).
Existing dry and cryogenic state schemas remain unchanged. Generic full-state
checkpointing and resume serialize this record; frame observers also export water
slices. Injection geometry is separate from shared water thermodynamics. The
carrier keeps fast-RK3; split microphysics and subcycled first-order moisture
transport do not claim third-order coupled accuracy. See
[water-spray scope and configuration](../../water-spray.md).
