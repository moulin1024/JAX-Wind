# Cell-owned droplet exchange with the live MAC carrier

`spray_cell_exchange.py` and `spray_cell_step.py` are opt-in verification
components. A fixed group of water bins belongs to one Eulerian cell. The gas
is a view of the live carrier, not a second stored core-gas inventory. This is
not yet an embedded-core entrainment model or a moving-parcel driver.

## Force response and energy

The cell gather is the arithmetic mean of two faces per velocity component;
its adjoint deposits half of the component impulse to each face. With actual
extensive dual-cell masses M_minus and M_plus, the inverse effective inertia is

    1 / M_eff = (1/4) / M_minus + (1/4) / M_plus.

For uniform density at an interior cell M_eff is twice the primary-cell gas
mass. Using the primary mass in a lumped drag solve would give the wrong slip
response after MAC deposition. Boundary faces use their actual half volumes.
Each frozen-rate bin/face pair relaxes exactly, with symmetric forward/reverse
half sweeps for multiple bins. The gas impulse opposes the full liquid impulse.
The computed kinetic loss uses the reduced effective mass for each component.

Evaporation uses the existing water law at the source-updated sampled velocity
and gas temperature. Vapor mass and its full momentum split equally to the two
faces. The lost kinetic energy is the sum of evaporated-velocity variance and
the exact mass-weighted mixing loss on the six receiving faces. This accounts
for both finite gas inertia and the staggered representation; no extra mapping
loss is inserted.

The caller must specify the fraction of mechanical loss going to sensible
heat. The remainder credits a separately owned unresolved mechanical-energy
inventory. This parameter has **no production default or physical SGS
justification**. Source conservation includes gas dilute enthalpy, actual MAC
kinetic energy, liquid sensible and kinetic energy, and that inventory. It does
not prove the carrier has a complete total-energy equation.

## Atomic transaction

The wrapper computes source increments once from the old state, runs the
common-flux carrier solve, and transports unresolved energy with its final gas
mass flux and source-updated specific energy. It commits gas, velocity, liquid,
and unresolved energy together. Any failed source, carrier, or reservoir
admissibility check restores all old live inventories and zeros all committed
flux/source/work ledgers. Attempted solver and source diagnostics remain
available. Retrying a rejected step therefore does not evaporate a bin twice.

The carrier's pressure work, numerical kinetic loss, wall impulse and finite
iteration work retain their existing explicit ledgers. They are not silently
converted to heat or assigned to physical SGS turbulence by default. The
separate [compatible energy extension](carrier-energy.md), enabled with
`energy_coupling=True`, adds numerical mixing heat and pressure conversion to
enthalpy within the EOS iteration and retains wall energy export. It does not
change the phase source's drag/evaporation energy partition or supply physical
SGS dissipation.

## Scope and verification

Uniform grids only. Periodic gather directions require at least two cells.
The recipient may touch a pressure-open x boundary. Cells touching an
impermeable y/z wall are rejected: wall reaction, deposition and heat partition
need an explicit wall model. Independent groups that overlap must not each
source the same old gas. The [group exchange stage](group-exchange.md) now
processes them against updated live inventories and commits with one carrier
solve. A separate [moving-parcel driver](moving-parcels.md) adds first-order
residence exchange and outflow; embedded-core ownership and the remaining
trajectory physics are still outstanding.

`tools/verify_spray_cell.sbatch` runs `tests/fv/test_spray_cell_step.py` on
**gpudev**. Tests use independently assembled global half-volume quadrature,
analytic stiff single-bin relaxation, Galilean shifts, nonuniform gas density,
three heat fractions, open-boundary and interior recipients, reservoir boundary
transport, vapor/liquid conservation, failed-source rollback and failed-carrier
rollback followed by a clean retry. These are numerical verification tests,
not measured droplet-transport or waterjet validation.

Gpudev **30272831** passed all **13 tests** in 107.92 s. The largest source
relative energy residual was **4.451e-19**. The accepted coupled case took
11 carrier iterations with relative momentum residual **2.467e-13**. Both
forced carrier rejection modes restored liquid and all committed ledgers, and
retries matched a fresh step exactly. These results use float64. Initial job
30272817 was stopped after identifying test-harness field-name mistakes.

Test results and source hashes are recorded under ignored
`outputs/spray_closure_validation/cell-<jobid>`; see the active goal record for
the final verified job. Physical SGS energy, inhomogeneous finite-inertia
transport, ambient withdrawal and no-double-counted core handoff remain open.
