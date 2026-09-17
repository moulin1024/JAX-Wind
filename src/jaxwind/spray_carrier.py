"""Common-flux conservative moist-gas/momentum/pressure iteration.

Opt-in first-order uniform-grid carrier step. It joins the verified transport,
MAC momentum and variable-coefficient pressure components in one transaction.
It does not yet supply the physical SGS closure, parcel/core ownership driver,
or a physically validated total-energy/SGS closure. An opt-in compatible
energy extension returns numerical mixing losses and pressure conversion to
enthalpy while keeping wall energy export and nonlinear work explicit.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .numerics.discretization import divergence
from .spray_energy import compatible_energy_terms
from .spray_low_mach import (
    MoistGasFields,
    _admissible,
    _donor_faces,
    _outgoing_mass_rate,
    _specific_fluxes,
    moist_gas_eos_density,
)
from .spray_momentum import build_momentum_transport, dual_average, dual_mass_flux
from .spray_pressure import build_momentum_projection
from .state import StaggeredVelocity


class CarrierStep(NamedTuple):
    gas: MoistGasFields
    velocity: StaggeredVelocity
    inertia_density: StaggeredVelocity
    transport_density: StaggeredVelocity
    mass_flux: StaggeredVelocity
    scalar_fluxes: StaggeredVelocity
    momentum_fluxes: tuple[StaggeredVelocity, ...]
    kinetic_fluxes: tuple[StaggeredVelocity, ...]
    wall_impulse: StaggeredVelocity
    numerical_kinetic_loss: StaggeredVelocity
    pressure: jax.Array
    pressure_impulse: StaggeredVelocity
    pressure_work: StaggeredVelocity
    iteration_work: StaggeredVelocity
    momentum_residual: StaggeredVelocity
    accepted: jax.Array
    iterations: jax.Array
    pressure_iterations: jax.Array
    eos_error: jax.Array
    donor_error: jax.Array
    momentum_error: jax.Array
    flow_error: jax.Array
    continuity_error: jax.Array
    outgoing_fraction: jax.Array
    pressure_continuity_error: jax.Array
    pressure_linear_error: jax.Array


def build_carrier_step(
    poisson,
    config,
    ambient_specific,
    ambient_velocity,
    *,
    tolerance=1e-9,
    max_iterations=100,
    relaxation=0.7,
    pressure_tolerance=1e-11,
    pressure_max_iterations=400,
    energy_coupling=False,
):
    """Build a transactional common-flux source/advection/pressure timestep.

    step(gas, velocity, gas_increments, momentum_increment, dt).

    Sources are increments per primary volume over dt, not rates. Momentum is
    the full vector impulse plus transferred-mass momentum, shape (3,nz,ny,nx).
    The ambient fields have shape (3,nz,ny); scalar order is dry/vapor fractions
    and specific dilute enthalpy. Existing Poisson metadata supplies geometry
    and BCs; its unit-coefficient solver is not used for momentum pressure.

    Each iterate restarts from the same source-updated inventories. Final
    momentum advection is recomputed with the FINAL scalar mass flux before
    acceptance. EOS, density-donor direction, flow fixed point, primal mass,
    dual CFL and the actual momentum equation must all pass. Error normalization
    for momentum/flow uses source-updated density and a speed scale at least
    1 m/s. Returned momentum_residual and iteration_work expose finite nonlinear
    solve error; neither is relabeled as a physical source.

    Rejection restores gas, velocity and inertia, zeros all committed fluxes,
    impulses and energy ledgers, and retains scalar diagnostics. The caller must
    also retain the old core/liquid until this transaction accepts. The thermal
    equation remains the existing conservative dilute enthalpy equation; pressure
    and numerical kinetic work remain diagnostics by default. energy_coupling
    enables compatible mixing heat and perturbation-pressure conversion in each
    iterate BEFORE the EOS check; wall loss remains an external export. This
    retains H_dilute - p0 as internal energy at constant thermodynamic p0. It is
    an explicit energy approximation, not a full compressible or physical SGS
    model. A closed heated domain at fixed p0 can still be incompatible.
    """
    grid = poisson.grid
    if not grid.is_uniform or poisson.open_y:
        raise ValueError("carrier requires uniform grids with periodic/impermeable y")
    if tolerance <= 0 or max_iterations < 1 or not 0 < relaxation <= 1:
        raise ValueError("invalid carrier iteration controls")
    ambient = jnp.asarray(ambient_specific)
    ambient_u = jnp.asarray(ambient_velocity)
    if ambient.shape != (3, grid.nz, grid.ny) or ambient_u.shape != ambient.shape:
        raise ValueError("ambient scalar/velocity shape must be (3,nz,ny)")
    periodic = (False, poisson.periodic_y, poisson.periodic_x)
    momentum_step = build_momentum_transport(
        grid,
        periodic_x=poisson.periodic_x,
        periodic_y=poisson.periodic_y,
        tolerance=tolerance,
    )
    pressure_step = build_momentum_projection(
        grid,
        periodic_x=poisson.periodic_x,
        periodic_y=poisson.periodic_y,
        open_x_low=poisson.open_x_low,
        tolerance=tolerance,
        linear_tolerance=pressure_tolerance,
        max_iterations=pressure_max_iterations,
    )
    inertia = lambda rho: StaggeredVelocity(
        *(dual_average(rho, c, periodic[c]) for c in (2, 1, 0))
    )
    maximum = lambda tree: jnp.max(jnp.stack([jnp.max(jnp.abs(x)) for x in tree]))

    def step(gas, velocity, increments, momentum_increment, dt):
        old = gas.dry_density + gas.vapor_density
        staged = jax.tree.map(lambda a, b: a + b, gas, increments)
        staged_rho = staged.dry_density + staged.vapor_density
        mass_increment = increments.dry_density + increments.vapor_density
        specific = jnp.stack(staged) / staged_rho
        donor_rho = moist_gas_eos_density(staged, config)
        ambient_rho = moist_gas_eos_density(MoistGasFields(*ambient), config)
        rho_star_faces = inertia(staged_rho)
        old_inertia = inertia(old)
        source_velocity = StaggeredVelocity(
            *(
                (r * u + dual_average(momentum_increment[k], c, periodic[c])) / s
                for k, (c, r, u, s) in enumerate(
                    zip((2, 1, 0), old_inertia, velocity, rho_star_faces)
                )
            )
        )
        speed_scale = jnp.maximum(jnp.asarray(1.0, old.dtype), maximum(source_velocity))

        def face_density(direction):
            return StaggeredVelocity(
                *(
                    r[0]
                    for r in _donor_faces(
                        donor_rho[None], direction, ambient_rho[None], poisson
                    )
                )
            )

        def transport(flow):
            fluxes = _specific_fluxes(specific, flow, ambient, poisson)
            q = jnp.stack(staged) - dt * jax.vmap(lambda f: divergence(f, grid))(fluxes)
            return MoistGasFields(*q), fluxes

        zero_face = jax.tree.map(jnp.zeros_like, velocity)
        zero_scalar = jax.tree.map(
            lambda u: jnp.zeros((3, *u.shape), u.dtype), velocity
        )
        zero_dual = tuple(dual_mass_flux(zero_face, c, periodic) for c in (2, 1, 0))
        inf = jnp.asarray(jnp.inf, old.dtype)
        initial_result = CarrierStep(
            gas,
            velocity,
            old_inertia,
            zero_face,
            zero_face,
            zero_scalar,
            zero_dual,
            zero_dual,
            zero_face,
            zero_face,
            jnp.zeros_like(old),
            zero_face,
            zero_face,
            zero_face,
            zero_face,
            jnp.array(False),
            jnp.array(0),
            jnp.array(0),
            inf,
            inf,
            inf,
            inf,
            inf,
            jnp.asarray(0.0, old.dtype),
            inf,
            inf,
        )
        initial_density = face_density(velocity)
        initial_flow = StaggeredVelocity(
            *(r * u for r, u in zip(initial_density, velocity))
        )
        initial_gas, _ = transport(initial_flow)
        valid = (
            _admissible(gas, config)
            & _admissible(staged, config)
            & _admissible(initial_gas, config)
            & (dt > 0)
        )
        valid &= _admissible(MoistGasFields(*ambient), config)
        valid &= jnp.all(jnp.abs(ambient[0] + ambient[1] - 1) <= tolerance) & jnp.all(
            jnp.isfinite(ambient_u)
        )

        def compatible_guess(rho):
            # A closed pressure solve cannot change the domain mass. Project
            # only the nonlinear TRIAL density onto that affine constraint;
            # no conserved inventory or EOS target is modified. A truly
            # incompatible closed source still fails the unmodified EOS gate.
            return rho + jnp.mean(staged_rho - rho) if poisson.periodic_x else rho

        initial = (
            compatible_guess(moist_gas_eos_density(initial_gas, config)),
            initial_flow,
            initial_result,
            valid,
        )

        def condition(state):
            _, _, result, valid = state
            return (result.iterations < max_iterations) & ~result.accepted & valid

        def iterate(state):
            guess, previous_flow, previous_result, valid = state
            # Momentum advection needs the mass advanced by its own trial flux,
            # not the EOS guess. Only pressure inertia uses the guessed density.
            trial_rho = staged_rho - dt * divergence(previous_flow, grid)
            previous_momentum = momentum_step(
                old,
                trial_rho,
                velocity,
                mass_increment,
                momentum_increment,
                previous_flow,
                ambient_u,
                dt,
            )
            guessed_inertia = inertia(guess)
            predictor = StaggeredVelocity(
                *(p / r for p, r in zip(previous_momentum.momentum, guessed_inertia))
            )
            rho_t = face_density(previous_flow)
            projection = pressure_step(
                predictor, old, guess, rho_t, guessed_inertia, mass_increment, dt
            )
            flow = projection.mass_flux
            candidate, fluxes = transport(flow)
            rho = candidate.dry_density + candidate.vapor_density
            final_inertia = inertia(rho)
            final_momentum = momentum_step(
                old,
                rho,
                velocity,
                mass_increment,
                momentum_increment,
                flow,
                ambient_u,
                dt,
            )
            residual = StaggeredVelocity(
                *(
                    r * u - p - j
                    for r, u, p, j in zip(
                        final_inertia,
                        projection.velocity,
                        final_momentum.momentum,
                        projection.impulse,
                    )
                )
            )
            if energy_coupling:
                energy = compatible_energy_terms(
                    projection.pressure,
                    final_momentum.momentum,
                    final_momentum.inertia_density,
                    final_momentum.wall_impulse,
                    final_momentum.kinetic_defect,
                    final_momentum.kinetic_fluxes,
                    projection.velocity,
                    dt,
                    grid,
                    periodic_x=poisson.periodic_x,
                    periodic_y=poisson.periodic_y,
                    open_x_low=poisson.open_x_low,
                )
                candidate = candidate._replace(
                    enthalpy_density=candidate.enthalpy_density
                    + energy.numerical_heat
                    + energy.pressure_conversion
                )
            eos = moist_gas_eos_density(candidate, config)
            eos_error = jnp.max(jnp.abs(rho - eos) / eos)
            momentum_error = maximum(
                StaggeredVelocity(
                    *(e / (r * speed_scale) for e, r in zip(residual, rho_star_faces))
                )
            )
            flow_error = maximum(
                StaggeredVelocity(
                    *(
                        (a - b) / (r * speed_scale)
                        for a, b, r in zip(flow, previous_flow, rho_t)
                    )
                )
            )
            chosen_density = face_density(flow)
            donor_error = maximum(
                StaggeredVelocity(*((a - b) / b for a, b in zip(rho_t, chosen_density)))
            )
            continuity_error = jnp.max(
                jnp.abs(rho - old - mass_increment + dt * divergence(flow, grid)) / old
            )
            outgoing = jnp.maximum(
                jnp.max(dt * _outgoing_mass_rate(flow, grid, poisson) / staged_rho),
                final_momentum.outgoing_fraction,
            )
            valid &= (
                previous_momentum.accepted
                & final_momentum.accepted
                & projection.accepted
            )
            valid &= _admissible(candidate, config) & (outgoing <= 1)
            errors = jnp.stack(
                (eos_error, donor_error, momentum_error, flow_error, continuity_error)
            )
            valid &= jnp.all(jnp.isfinite(errors))
            accepted = valid & jnp.all(errors <= tolerance)
            midpoint = StaggeredVelocity(
                *(
                    (a + b) / 2
                    for a, b in zip(final_momentum.velocity, projection.velocity)
                )
            )
            pressure_work = StaggeredVelocity(
                *(j * u for j, u in zip(projection.impulse, midpoint))
            )
            iteration_work = StaggeredVelocity(
                *(e * u for e, u in zip(residual, midpoint))
            )
            result = CarrierStep(
                candidate,
                projection.velocity,
                final_inertia,
                rho_t,
                flow,
                fluxes,
                final_momentum.fluxes,
                final_momentum.kinetic_fluxes,
                final_momentum.wall_impulse,
                final_momentum.kinetic_defect,
                projection.pressure,
                projection.impulse,
                pressure_work,
                iteration_work,
                residual,
                accepted,
                previous_result.iterations + 1,
                previous_result.pressure_iterations + projection.iterations,
                eos_error,
                donor_error,
                momentum_error,
                flow_error,
                continuity_error,
                outgoing,
                projection.continuity_error,
                projection.linear_error,
            )
            next_flow = jax.tree.map(
                lambda a, b: (1 - relaxation) * a + relaxation * b, previous_flow, flow
            )
            return (
                compatible_guess((1 - relaxation) * guess + relaxation * eos),
                next_flow,
                result,
                valid,
            )

        _, _, result, _ = jax.lax.while_loop(condition, iterate, initial)
        # All carrier fields and ledgers commit together; diagnostic errors and
        # iteration counts describe the attempted step even on rejection.
        committed = {}
        for name in CarrierStep._fields[:15]:
            committed[name] = jax.tree.map(
                lambda a, b: jnp.where(result.accepted, a, b),
                getattr(result, name),
                getattr(initial_result, name),
            )
        return result._replace(**committed)

    return step
