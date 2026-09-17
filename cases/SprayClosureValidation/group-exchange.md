# Dynamically addressed droplet groups sharing the live gas inventory

`spray_group_exchange.py` extends the verified cell exchange to several groups
in one source stage and one carrier transaction. Groups may occupy the same
cell or adjacent cells with a shared velocity face. Every group uses the gas
state left by the preceding group, including changed face inertia, velocity,
vapor and enthalpy. Computing every source from the same old gas would omit
these interactions and would invalidate the independent source energy budgets.

The source interface is

```
step(gas, velocity, unresolved_density, groups, dt, recipients)
```

Water-bin mass, multiplicity and temperature have shape `(groups, bins)`;
velocity has shape `(groups, 3, bins)`. Integer recipients have shape
`(groups, 3)` with coordinates `(z,y,x)`. Array index owns each group's liquid
once. Changing its recipient between calls does not clone the liquid or store
another gas inventory. Callers remain responsible for unique physical parcels:
the routine cannot infer that two distinct array entries duplicate a parcel.

The common wrapper accumulates each group's conservative species, enthalpy and
full momentum increments, then runs **one** carrier timestep for all groups.
The owned unresolved mechanical reservoir uses the final carrier mass flux.
Any rejected group, carrier solve or reservoir check restores all original
liquid groups, gas, velocity and unresolved energy. All committed source and
carrier ledgers are zeroed; attempted diagnostics remain available. Invalid
runtime cell addresses reject instead of silently wrapping or clipping ownership.

## Splitting and limits

Sequential source processing is first order in group ordering. This is explicit
operator splitting, not a claim that group order is physically irrelevant. Tests
reverse the groups and halve the source timestep to check convergence of that
difference. The original symmetric bin drag sweep within each group is retained.
Rates and evaporation still use the existing frozen-reservoir source laws.

Recipient reassignment in this group API is bookkeeping support for motion,
**not position integration**. A separate [moving-parcel driver](moving-parcels.md)
now supplies first-order position integration, cell residence exchange and
conservative open-x outflow. Gravity, wall impaction, injection and stochastic
finite-inertia dispersion remain to be integrated. The current nearest-cell gather has discontinuities
at cell boundaries. No coarse-grid dispersion accuracy follows from source
conservation or ownership alone.

Uniform-grid and impermeable-wall restrictions from [cell exchange](cell-exchange.md)
remain. This first implementation processes groups serially and revisits gas
arrays per group; production parcel-count scaling needs a separate assessment.
It introduces no physical SGS dissipation or fitted heat partition and does not
supply the embedded-core ambient-withdrawal/entrainment/handoff model. Carrier
pressure work and numerical kinetic loss remain explicit diagnostics rather
than a complete thermodynamic total-energy closure.

## Verification

`tools/verify_spray_groups.sbatch` runs both the original 13 cell transaction
checks and group tests on gpudev. The latter cover independent global extensive
budgets for adjacent and coincident groups, agreement with sequential single-cell
exchange, group-order timestep convergence, three invalid later recipients,
three successive recipient reassignments, one coupled carrier with vapor and
reservoir boundary budgets, and rollback of every group after carrier rejection.
Source hashes and results live under ignored
`outputs/spray_closure_validation/groups-<jobid>`. These checks establish
numerical properties only; measured transport and waterjet validation remain open.

Gpudev **30272846** passed all **22 tests** in 176.43 s. Source relative energy
residuals were at most **3.288e-18** for the group cases. Reversing group order
at timesteps 2e-5, 1e-5 and 5e-6 s gave one-step liquid-velocity difference
norms 1.271127e-5, 3.233946e-6 and 8.156181e-7 m/s, reduction factors **3.931**
and **3.965**. This is the expected second-order *local difference* between
first-order ordered source splittings; no second-order global accuracy is
claimed. The multi-group carrier case accepted in 17 iterations with relative
momentum residual **1.615e-13**. All source and carrier rollback checks passed.
