# Pressure correction using momentum inertia

`src/jaxwind/spray_pressure.py` supplies the pressure primitive needed to couple
thermodynamic mass transport and the conservative staggered momentum predictor.
It is opt-in and currently restricted to uniform grids. It does not replace the
production pressure solver or replace the production nonlinear transport timestep. The separate
[coupled carrier](coupled-carrier.md) uses this primitive in a common iteration.

## Equation and mechanical budget

The thermodynamic donor density `rho_t` defines the transported mass flux. The
staggered inventory density `rho_i` defines momentum inertia. Both are positive
face fields supplied by the caller and held fixed during a pressure solve:

```
F_star = rho_t * u_star
beta = rho_t / rho_i
R = (rho_new - rho_old - Delta_rho)/dt + div(F_star)
div(beta * grad(p)) = R/dt
u_new = u_star - dt * grad(p)/rho_i
F_new = rho_t * u_new.
```

Here `Delta_rho` is the conserved mass increment per volume, not a rate. This
makes the pressure impulse on momentum exactly `-dt*grad(p)` while satisfying
the primary mass budget. Applying the previous unit-coefficient correction to
`F` and then dividing by `rho_t` would multiply that impulse by `rho_i/rho_t`.
Mass conservation alone would not detect the wrong force.

The returned component impulse is momentum change per dual volume. Its midpoint
mechanical work is

```
W_i = (-dt*grad_i(p)) * (u_i_star + u_i_new)/2
    = rho_i * (u_i_new^2 - u_i_star^2)/2.
```

This closes the pressure substep's kinetic-energy change at fixed inertia. It
does **not** determine the full-step sensible heat, enthalpy, compressive work,
or SGS-energy treatment. The driver must combine source, advection and pressure
ledgers, including boundary pressure work, with a consistent thermodynamic
energy equation. Adding this diagnostic to heat automatically is not justified.

## Boundary operator

The new gradient retains all transverse pressure differences, including those
in the x boundary columns. The legacy open-domain gradient suppresses those
transverse differences; it is not reused here. Other existing callers retain
their current behavior.

- Periodic x/y use wrapped differences.
- Open x faces have zero mechanical pressure and use the half-cell distance.
- `open_x_low=False` instead prescribes zero pressure correction to the inlet
  normal velocity; the high x end remains pressure-open.
- Impermeable y/z faces have zero normal pressure gradient and must enter with
  zero normal predictor velocity. Lateral pressure outlets are not supported.

The full operator is `-div(beta*grad)` with the standard primary finite-volume
divergence. On a uniform grid it is symmetric positive definite when x is
pressure-open, and positive semidefinite with a constant null mode for periodic
x. Pressure and its residual are kept at zero mean in the latter case.

## Solve, acceptance, and rejection

The matrix-free preconditioned conjugate-gradient solve uses the positive
operator and its Jacobi diagonal. The recursive residual controls iteration,
but a **fresh operator evaluation** checks the true residual before acceptance.
Its guard is the larger of ten times the requested relative linear tolerance
and 100 machine epsilons, multiplied by the RHS norm. The actual unmodified
mass residual must also satisfy the separate physical relative tolerance.
An exactly zero RHS needs no iterations.

Removing the pressure gauge does not grant permission to remove mass. A closed
source/density incompatibility remains in the physical residual and rejects the
step. Nonpositive face densities, nonfinite input, nonpositive dt, invalid wall
velocity, nonconvergence, or excess continuity error also reject. Rejection
restores input velocity and returns zero committed pressure, mass flux, impulse
and work. The caller must keep old carrier and liquid state until all coupled
stages accept.

The routine is presently assessed in float64. It does not claim scalable
multigrid performance, mapped-grid support, full high-density-ratio robustness,
or mixed-precision production readiness. Those require their own evidence.

## Independent verification and remaining coupling

```bash
sbatch --output=outputs/spray_closure_validation/pressure-%j.out tools/verify_spray_pressure.sbatch
```

The tests compare with a separately assembled NumPy matrix for periodic,
pressure-open and fixed-inlet boundary combinations, with random independent
inertia and transport density fields. Pointwise pressure impulse and kinetic
work are compared against independent differences. Independent primary/dual
volume quadrature also checks global pressure work by integration by parts,
including the boundary work of a fixed-flux inlet. An analytic continuum field
`sin(kx*x)*cos(2*pi*y)*cos(pi*z)` supplies an independent grid-convergence
reference on 8, 16 and 32 streamwise cells, with a variable periodic coefficient
or pressure-open x boundaries. The source is calculated by analytic derivatives,
not by applying the implementation to the exact field. A unit-coefficient
negative control retains mass conservation but gives the wrong momentum force.
Rejection tests include an incompatible closed source and exhausted iterations.

The [coupled carrier step](coupled-carrier.md) now supplies a common iteration
of momentum prediction, this pressure solve, and species/enthalpy transport using
the **same final mass flux**. Predictor fluxes, donor directions, density and inertia must all agree
at acceptance. Passing this standalone pressure verification does not justify
sequentially committing the existing scalar step and then correcting its flow.


## Verified results

Gpudev **30272522** completed on the A100 with **12 tests passed**, no failures
or skips, in **26.127 s**. Five random-field matrix comparisons required 31–35
iterations, with maximum relative physical continuity error **3.776e-12**.
Pointwise force/kinetic-work and independent global pressure-work quadrature
passed for all five boundary configurations, including the fixed-flux inlet.

The continuum manufactured-solution RMS pressure errors were:

| Streamwise cells | Periodic variable coefficient | Pressure-open x |
| --- | ---: | ---: |
| 8 | 0.01844543 | 0.01630968 |
| 16 | 0.00450605 | 0.00400070 |
| 32 | 0.00112010 | 0.00099546 |

Each halving of grid spacing reduced error by factors **4.02–4.09**, consistent
with second-order spatial accuracy. The unit-coefficient control failed the
momentum-force criterion despite its mass-conservative construction. Incompatible
closed sources, invalid density and iteration exhaustion were rejected without
committing flux or pressure work. Zero RHS required zero iterations.

The initial **30272509** also passed all 12 tests. The final run adds the global
integration-by-parts pressure-work check. Final hashes for all five recorded
files match the verified state. Reports and JUnit are ignored under
`outputs/spray_closure_validation/pressure-30272522/`. These results do not
advance the physical profile, measured droplet, SGS or waterjet accuracy gates.
