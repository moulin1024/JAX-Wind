# Prescribed parcel injection and atomic carrier coupling

`spray_injection.py` stages prescribed droplets in free parcel slots, then
commits their moving-source and carrier update as one transaction. It is an
opt-in numerical component. It does not prescribe a measured nozzle source,
qualify the PDA sampling operator, or change production waterjet settings.

## Source contract

`ParcelBirths` contains `WaterCoreBins`, xyz positions `(3, births)`, and birth
time offsets `(births,)` measured from the current step start. All entries must
be finite, masses/counts nonnegative, and temperatures at least the freezing
reference used by the warm-water model. An active birth needs positive droplet
mass. Offsets lie in `[0, dt)`; a birth exactly at the end belongs to the next
step. Zero-count entries consume no slots.

`births_from_mass_flow` provides an optional deterministic inlet quadrature.
The caller supplies total mass flow, diameter, **mass fractions**, velocity,
temperature, position and fractional birth times. For node i,

```
single_drop_mass_i = water_density * pi * diameter_i**3 / 6
multiplicity_i = mass_flow * dt * mass_fraction_i / single_drop_mass_i
```

Fractions must sum to one (absolute tolerance 1e-12), and are never silently
renormalized. This quadrature exactly accounts for the prescribed step mass;
it does not reconstruct a joint size/velocity distribution, imply that event
counts are mass fractions, or establish measurement representativeness. The
current tests use float64. No production-precision claim is made.

`stage_parcel_births` reserves slots whose existing mass or multiplicity is
zero, in array order, at the **start** of a step. It does not anticipate later
exits. If capacity is insufficient, the entire batch rejects; no partial flow
is accepted or silently discarded. Its output is provisional until the whole
coupled transaction accepts. Existing active parcels retain their state.

## Transport and ownership

The coupled interface is

```
step(gas, velocity, unresolved_density, liquid, dt, position, births)
```

Existing parcels receive duration `dt`; each newborn receives
`dt - time_offset`. Moving sources trace and exchange only for this duration,
or until earlier open-boundary exit. Body-force impulses/work use the same
residence intervals. The source/carrier transaction still spans `dt`.

Per-step injected mass, full momentum, sensible liquid enthalpy relative to
freezing, and kinetic energy are explicit external-boundary ledgers. They are
not an interphase gas withdrawal: subsequent drag and evaporation supply their
own conservative gas exchanges. Outflow remains owned by the existing moving
step, which retires exited multiplicities. A driver accumulates each accepted
injection/export ledger exactly once.

Any injection, residence/source or carrier rejection restores **pre-injection**
gas, velocity, unresolved energy, liquid and positions. All committed injection,
outflow and external-force ledgers are zero, and birth slot assignments are -1.
`injection_accepted`, requested/available capacity and path diagnostics describe
the attempted substeps; only `moving.phase.accepted` authorizes a commit. A
stochastic caller must likewise retain its original RNG state until acceptance.

Exact birth ages do not make the source update a chronological event solver:
parcels exchange serially with the current gas, frozen paths use old velocity,
and the carrier receives the aggregate source. The method retains first-order
splitting. There is no breakup, collision, impaction or adaptive capacity.
Physical SGS/dispersion, gas-nozzle entrainment/ambient withdrawal, complete
carrier energy closure and independent measured transport remain outstanding.

The [compatible carrier energy extension](carrier-energy.md) is available through
`energy_coupling=True`. It can balance injection, gas/liquid transport and body
work in the stated warm dilute model; it does not remove the physical closure
or measured-inlet limitations above.

## Verification

`tools/verify_spray_injection.sbatch` runs on gpudev and records source hashes,
logs and XML in ignored `outputs/spray_closure_validation/injection-*` paths.
Tests check exact mass-flow quadrature and birth ages; preservation of existing
parcels; zero flow at full capacity; invalid times/fractions; full source
mass/momentum/energy budgets including external injection/body work/outflow;
pre-injection rollback for capacity/source/carrier failures; deterministic retry;
and accepted carrier water budgets.

A separate 12-step continuous-injection test injects near the open boundary,
exports liquid and reuses the same slot every step. It checks cumulative
prescribed flow and vapor/liquid/boundary water conservation after every step,
so a one-step pass cannot conceal dropped flow or repeated export. It is a
numerical bookkeeping test, not a steady spray or physical validation case.

Final gpudev **30272996** passed **9 injection tests** in 128.53 s, with
FutureWarnings treated as errors. The 12-step case injected **1.2e-7 kg**,
exported **1.199897442574e-7 kg**, and had maximum cumulative water residual
**1.8185e-19 kg**, including vapor boundary flux. The accepted coupled birth
case converged in 9 iterations with momentum residual **4.3285e-13**. Source
hashes match the tested files.

Earlier run **30272993** passed 26 tests, including all 19 body-force/moving
regressions, and failed one test-only JAX list-indexing operation. Correcting
that indexing required no solver change; the final targeted run also added the
continuous-injection test. Neither run changes physical acceptance criteria.
