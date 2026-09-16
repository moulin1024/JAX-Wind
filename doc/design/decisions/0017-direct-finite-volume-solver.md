# ADR-0017: Direct finite-volume solver with pressure backends

Status: **Accepted**

Supersedes ADR-0001 through ADR-0016 as the description of the active solver architecture. The SI-boundary rule from ADR-0005 and the compatible projection principle from ADR-0004 are retained here without their former semantic-interpreter implementation.

## Context

The previous architecture wrapped arrays in semantic field, phase, ownership, vector-field, integrator, and effect abstractions. Production development converged on a direct staggered finite-volume implementation instead. Keeping both structures made configuration appear to select behavior that the active solver ignored and left a large unreachable code path.

## Decision

JAX-Wind has one flow solver: the staggered finite-volume modules in `src/jaxwind`. Velocity components live on MAC faces and transported scalars and pressure live at cell centers. Applications construct these arrays, physical models, pressure projection, and time integration directly.

FFT, geometric multigrid, and optional AMG are pressure backends within that solver. FFT remains the direct periodic-uniform option; mapped or open domains use multigrid. A pressure backend is not a separate flow solver.

For a nonperiodic streamwise axis, GMG supports either the original prescribed
inlet/pressure-outlet pair or pressure outlets at both ends (`open_x_low=True`).
The latter uses zero exterior-face pressure at x− and x+, consistently in the
gradient, multigrid operator/diagonals, residual, and low-Mach correction. It
does not impose a streamwise inlet velocity. Cryogenic volume-source cases
select it with `physics.source.streamwise_boundaries = "outflow-outflow"`;
sidewalls and vertical wall conditions are unchanged. Outlet backflow uses
ambient scalar values. Other backends reject this optional boundary mode.

Cryogenic fully vaporized volume-source runs may select adaptive RK3 through
`time.cfl`. Their `dt_seconds` is a maximum and `steps * dt_seconds` defines
physical duration. The adaptive state additionally carries accepted dt,
peak-stage CFL, and rejected-trial count. Source ramps use physical time;
projection, microphysics, and stage updates use the accepted step. Diffusion,
startup, and CFL checks can shorten a step. Snapshot scheduling for adaptive
runs uses physical time and is preserved across checkpoints.

Case files use canonical SI values. The ABL application stores pressure acceleration, rotation, geostrophic velocity, wall and surface parameters, scalar flux and buoyancy, time step, precision, and sampling controls directly. The ABL solver advances those SI values directly; there is no semantic-field interpreter, ownership DSL, or separate integrator package.

The ABL momentum closure is AMD. Removed LASD parameters are not accepted as configuration. Adding another closure requires an implemented FV closure, a real configuration choice, and direct tests.

Application packages own configuration parsing, initialization, checkpoints, diagnostics, and artifacts. The numerical package owns grid geometry, discretization, physical tendencies, pressure projection, integration, and turbine forcing. Turbine records under `jaxwind.physics` are parameter data consumed by the FV turbine adapters.

## Consequences

- The public code path matches the implementation used by applications.
- Removed spectral-flow and semantic-interpreter APIs are not compatibility surfaces.
- Projection tests must check divergence and backend agreement through the FV operators.
- Configuration schemas reject dead choices instead of silently ignoring them.
- Distributed execution can be added around FV arrays when required; no unused ownership abstraction is retained in advance.
