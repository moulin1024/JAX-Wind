# Particle gravity with separate external impulse and work

The opt-in moving-source and moving-carrier builders now accept
`particle_acceleration=(ax, ay, az)` in m/s². The default is zero, preserving
unforced verification; `(0,0,-9.81)` supplies liquid gravity. This parameter is
an explicit external acceleration, not an inferred pressure/buoyancy force or
a modification of the drag coefficient. Production waterjet settings are unchanged.

For each in-domain residence interval, the source applies half a constant
acceleration velocity kick, exchanges mass/heat/momentum with the live gas, then
applies the second half kick using the remaining liquid mass. Each kick records

```
external_impulse = current_liquid_mass * acceleration * half_duration
external_work = external_impulse dot midpoint_liquid_velocity
```

The work is signed and exactly equals that kick's kinetic-energy change. A
Galilean boost changes it by boost dot impulse, as it should. Empty/inactive bins
receive no kick. Gravity's impulse is not directly deposited as an opposing gas
force; the gas receives only the subsequent physical drag and vapor-transfer
impulse. Mechanical drag loss retains the existing explicit heat/reservoir
partition, with no extra gravity heating.

Evaporating mass therefore receives body force only while it belongs to the
liquid at each split stage. An early-exiting parcel receives force only during
its path residence, not the full nominal timestep. Its final mass, momentum and
energy are exported once. Source or carrier rejection restores all old live
states and positions and zeros attempted outflow, external impulse and work.
Returned `external_impulse` has shape `(3, parcels)` and `external_work` has
shape `(parcels,)`; both are per-step extensive ledgers.

## Numerical scope

Paths still use the old velocity over a timestep. Symmetric force kicks do not
make the whole trajectory second order. In particular, kick work balances
kinetic energy exactly but is not generally identical to gravity dotted into
the frozen-path displacement. Thus this scheme does not exactly conserve
kinetic plus gravitational potential energy at finite dt. Timestep convergence
is required. A complete carrier total-energy treatment remains outstanding.

The gas equations retain their existing body-force treatment. No displaced-gas
buoyancy, pressure-gradient particle force, lift, stochastic dispersion or wall
reaction is silently inserted. Their physical assumptions and gas reaction/work
must be resolved separately before production coupling.

## Verification

`tools/verify_spray_body_force.sbatch` exercises analytic velocity kicks,
kinetic work and Galilean transforms, full source budgets including outflow and
external work, residence-limited force on an evaporating exiting parcel,
zero force after export, source/carrier rollback, accepted carrier ledgers,
and forced/unforced fixed-duration trajectory refinement. It includes the
preceding motion regressions.

Gpudev **30272964** passed **19 tests** in 118.20 s. The gravity/outflow carrier
accepted in 17 iterations with momentum residual **1.576e-13**. The gravity
crossing case retained the expected first-order position refinement against a
finer numerical reference, with reduction factors **2.070, 2.065, 2.095**.

`tools/verify_spray_settling.sbatch` additionally compares against an independent
continuous two-mass Stokes solution. A 20-micrometre parcel remains in one cell,
in the actual Stokes branch, with finite effective gas inertia from the two MAC
faces. Slight supersaturation disables evaporation in the existing
no-condensation water law; all mechanical loss stays in the unresolved reservoir
to isolate constant-mass dynamics. No drag or heat-transfer function is mocked.

With liquid mass m, effective gas mass M, Stokes rate k, acceleration g and
initial rest, the reference has lambda=k*(1+m/M), centre acceleration a=m*g/(M+m),

```
slip(t) = g/lambda * (1-exp(-lambda*t))
v_liquid(t) = a*t + M/(M+m)*slip(t)
u_gas(t) = a*t - m/(M+m)*slip(t)
z_liquid(t) = z0 + a*t²/2
                + M/(M+m)*g/lambda*(t-(1-exp(-lambda*t))/lambda)
```

Initial job **30272970** passed conservation but failed its coarse position-rate
check: 32/64/128/256-step position errors were 7.199e-6, 4.784e-6, 2.695e-6 and
1.424e-6 m (ratios 1.505, 1.775, 1.892). At the coarsest lambda*dt near 0.63,
first-order position lag and second-order velocity error partially cancel.
The follow-up **30272972** retained those results and extended to 512/1024
steps; **the analytic test passed** in 10.05 s. Fine position errors are
7.313e-7 and 3.705e-7 m, with final reduction factors **1.947 and 1.974**.
Velocity reduction factors progress from **3.980 to 4.000**; the finest velocity
error is **2.674e-7 m/s**. All six resolutions passed global source budgets and
the analytic total impulse m*g*T check. The numerical solver and physical
acceptance gates were not changed. This establishes asymptotic convergence;
the original coarsest timestep remains outside the tested position-rate band.

These are numerical verification cases, not measured settling/dispersion or
waterjet validation. All logs, source hashes and test artifacts remain under
ignored `outputs/spray_closure_validation/` paths.
