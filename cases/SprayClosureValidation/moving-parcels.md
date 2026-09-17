# Moving parcel residence exchange and conservative outflow

`spray_paths.py` traces a straight, frozen-velocity trajectory through a uniform
grid. It records the time spent in every visited cell, with simultaneous corner
crossings, periodic x/y wrapping and explicit open-x exit. Negative motion from
an exact face starts in the cell on its negative side. There is no position
nudge that changes the physical residence time. Segment durations sum to the
step duration or the earlier open-boundary exit time.

`spray_moving.py` advances each parcel's liquid state through those residence
intervals, exchanging with the current live gas inventory in each visited cell.
Mass, momentum and energy sources are accumulated once, then passed through one
common-flux carrier transaction. Positions and all outflow budgets commit only
if source and carrier both accept. A rejected step restores old positions,
gas, velocity, liquid and unresolved mechanical energy.

The interface is

```
step(gas, velocity, unresolved_density, liquid, dt, position, residence_times=None)
```

Positions and liquid velocities have shape `(3, parcels)`, in xyz coordinates;
water mass per droplet, multiplicity and temperature have shape `(parcels,)`.
Optional residence times have shape `(parcels,)`, are positive and no greater
than `dt`, and default to `dt`. The [injection wrapper](parcel-injection.md)
uses them to give newborns only their post-birth portion of a timestep.
The grid origin is zero. Each array entry owns one parcel exactly once; callers
must not duplicate the same physical parcel in several entries.

At an open-x exit, the remaining liquid mass, full momentum, sensible enthalpy
and kinetic energy enter explicit **per-step** outflow arrays. Its live
multiplicity becomes zero. Repeating a step cannot export it again. Inactive
slots keep their old position. Cumulative export belongs to the driver; adding
a per-step export more than once would double count it.

## Accuracy and scope

The path uses the **old velocity** over each timestep. Drag and evaporation
change the liquid velocity/temperature along the path, which affects the next
step's trajectory. This is first-order splitting, not the exact trajectory of
an accelerating evaporating droplet. Exact geometric residence is not a claim
of exact physical motion. Parcel ordering also remains a first-order source
splitting. The nearest-cell gas gather is spatially discontinuous, so crossing
and timestep tests are essential on a coarse mesh.

If the fixed path capacity is exhausted, the entire step rejects rather than
skipping cells or accepting a truncated trajectory. A driver may then shorten
the timestep or increase capacity. Impaction and wall heat transfer are not
implemented: closed-wall hits, and source exchange in wall-adjacent cells,
reject rather than silently reflecting or deleting liquid.

The [particle body-force extension](particle-body-force.md) now supplies
explicit liquid acceleration with separate external impulse/work. Displaced-gas
buoyancy, stochastic dispersion, adaptive stepping, physical SGS dissipation
and embedded-core entrainment/ambient withdrawal remain absent. Prescribed
injection is available through the separate wrapper; measured inlet qualification
remains open. These remain required for the requested waterjet model. It also
retains the carrier's existing pressure-work and numerical-energy diagnostics;
it does not claim a complete thermodynamic total-energy equation. Production
waterjet settings are unchanged.

## Verification

`tools/verify_spray_moving.sbatch` tests analytic positive/negative residence
intervals, exact-face/corner motion, repeated periodic wraps, stationary and
immediate-exit parcels, capacity/wall rejection, full source inventories with
outflow, no repeated export, sources on both sides of a crossed face, accepted
carrier vapor/liquid/boundary budgets and rejected-carrier position rollback.

`tools/verify_spray_moving_convergence.sbatch` advances the same finite-inertia,
evaporating crossing case over a fixed duration at successively smaller
timesteps. It checks global source conservation at every resolution and compares
positions with a finer numerical reference. That reference assesses numerical
convergence of these equations, not agreement with measured droplet transport.

Both run on gpudev. Their logs, source hashes and XML results are under ignored
`outputs/spray_closure_validation/moving-*` paths. The active goal record names
the final verified runs and their results.

Final gpudev **30272906** passed **10 tests** in 61.59 s, including the timestep
study, with FutureWarnings treated as errors. An integer index-width warning in
earlier passing runs 30272896/30272902 was fixed before this final verification.
The moving-source cases closed their local energy ledgers to relative residual
at most **2.372e-19**. The accepted carrier case took 17 iterations with momentum
residual **1.576e-13**.

For a fixed 8e-4 s integration, timesteps 1e-4, 5e-5, 2.5e-5 and 1.25e-5 s
gave position-error norms **5.669e-4, 2.739e-4, 1.326e-4 and 6.328e-5 m** against
a 1024-step reference, reduction factors **2.070, 2.065 and 2.095**. The difference
between 512- and 1024-step references was **4.188e-6 m**. Global source budgets
passed at all six resolutions. These float64 results support first-order
numerical convergence of the current deterministic model; they do not qualify
physical dispersion, production precision or measured waterjet accuracy.
