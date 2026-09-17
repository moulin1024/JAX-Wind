"""Conservative moist-gas transport coupled to the existing mass projection.

This opt-in first-order thermodynamic transport stage advances dry-air mass,
vapor mass and the shared dilute enthalpy with ONE pressure-corrected mass
flux. A fixed-point iteration enforces the EOS without overwriting inventories.
It is not yet a full momentum/SGS/spray LES timestep. No production default is
changed. Pressure is prescribed and spatially uniform; heat release in a closed
box is rejected if it requires changing the thermodynamic pressure.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .low_mach import project_mass_flux
from .numerics.discretization import divergence
from .state import StaggeredVelocity


class MoistGasFields(NamedTuple):
    """Cell-centred extensive quantities per volume: kg/m3, kg/m3, J/m3.

    H=cp_d*rho_d*(T-Tf)+Lv*rho_v, matching spray_core. This dilute convention
    omits vapor sensible heat and is restricted to warm, dry-air-rich gas.
    """

    dry_density: jax.Array
    vapor_density: jax.Array
    enthalpy_density: jax.Array


class MoistGasTransport(NamedTuple):
    gas: MoistGasFields
    velocity: StaggeredVelocity
    pressure: jax.Array
    # Thermodynamic donor density used with velocity to form advective mass flux.
    transport_density: StaggeredVelocity
    # Leading axis [dry air, vapor, enthalpy]; per area per time, committed only.
    fluxes: StaggeredVelocity
    accepted: jax.Array
    iterations: jax.Array
    eos_error: jax.Array
    donor_error: jax.Array
    continuity_error: jax.Array
    outgoing_fraction: jax.Array


def moist_gas_temperature(gas, config):
    return config.freezing_temperature + (
        gas.enthalpy_density - config.water_vapor_latent_heat * gas.vapor_density
    ) / (config.dry_air_heat_capacity * gas.dry_density)


def moist_gas_eos_density(gas, config):
    """Unclipped ideal-mixture density from current temperature/composition."""
    rho = gas.dry_density + gas.vapor_density
    gas_constant = (
        config.dry_air_gas_constant * gas.dry_density
        + config.water_vapor_gas_constant * gas.vapor_density
    ) / rho
    return config.pressure / (gas_constant * moist_gas_temperature(gas, config))


def moist_gas_from_primitive(temperature, vapor_fraction, config):
    """Construct EOS-consistent fields; vapor_fraction is kg/kg total gas."""
    temperature, vapor_fraction = jnp.broadcast_arrays(temperature, vapor_fraction)
    gas_constant = (
        1 - vapor_fraction
    ) * config.dry_air_gas_constant + vapor_fraction * config.water_vapor_gas_constant
    rho = config.pressure / (gas_constant * temperature)
    dry, vapor = rho * (1 - vapor_fraction), rho * vapor_fraction
    return MoistGasFields(
        dry,
        vapor,
        config.dry_air_heat_capacity * dry * (temperature - config.freezing_temperature)
        + config.water_vapor_latent_heat * vapor,
    )


def _admissible(gas, config):
    temperature = moist_gas_temperature(gas, config)
    # The shared dilute convention must retain positive mixture cv.
    capacity = config.dry_air_heat_capacity * gas.dry_density
    gas_constant_density = (
        config.dry_air_gas_constant * gas.dry_density
        + config.water_vapor_gas_constant * gas.vapor_density
    )
    return (
        jnp.all(jnp.isfinite(jnp.stack(gas)))
        & jnp.all(jnp.isfinite(temperature))
        & jnp.all(gas.dry_density > 0)
        & jnp.all(gas.vapor_density >= 0)
        & jnp.all(temperature >= config.freezing_temperature)
        & jnp.all(capacity > gas_constant_density)
    )


def _donor_faces(specific, carrier_flux, ambient, poisson):
    """Donor values selected by the oriented face transport direction."""
    parts = []
    # Spatial axes z,y,x become 1,2,3 after the leading species/enthalpy axis.
    for axis, face, periodic in (
        (2, carrier_flux.x, poisson.periodic_x),
        (1, carrier_flux.y, poisson.periodic_y),
        (0, carrier_flux.z, False),
    ):
        field_axis = axis + 1
        if periodic:
            left, right = jnp.roll(specific, 1, field_axis), specific
        else:
            if axis == 2:
                low = high = jnp.asarray(ambient)[..., None]
            else:
                low = jnp.take(specific, jnp.array([0]), axis=field_axis)
                high = jnp.take(specific, jnp.array([-1]), axis=field_axis)
            left = jnp.concatenate((low, specific), axis=field_axis)
            right = jnp.concatenate((specific, high), axis=field_axis)
        parts.append(jnp.where(face[None] >= 0, left, right))
    return StaggeredVelocity(*parts)


def _specific_fluxes(specific, carrier_flux, ambient, poisson):
    donors = _donor_faces(specific, carrier_flux, ambient, poisson)
    return StaggeredVelocity(*(f[None] * q for f, q in zip(carrier_flux, donors)))


def _outgoing_mass_rate(flux, grid, poisson):
    result = jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=flux.x.dtype)
    for axis, face, periodic, widths in (
        (2, flux.x, poisson.periodic_x, grid.x_widths),
        (1, flux.y, poisson.periodic_y, grid.y_widths),
        (0, flux.z, False, grid.z_widths),
    ):
        if periodic:
            low, high = face, jnp.roll(face, -1, axis)
        else:
            low = jnp.take(face, jnp.arange(face.shape[axis] - 1), axis=axis)
            high = jnp.take(face, jnp.arange(1, face.shape[axis]), axis=axis)
        shape = [1, 1, 1]
        shape[axis] = len(widths)
        result += (jnp.maximum(high, 0) - jnp.minimum(low, 0)) / jnp.asarray(
            widths, dtype=face.dtype
        ).reshape(shape)
    return result


def build_moist_gas_transport(
    poisson,
    config,
    ambient_specific,
    *,
    tolerance=None,
    max_iterations=60,
    relaxation=0.7,
):
    """Build a transactional source/transport/EOS-projection stage.

    step(gas, predictor_velocity, increments, dt) takes conserved source
    INCREMENTS per cell volume over dt, not source rates. Ambient has shape
    (3,nz,ny): dry/vapor mass fractions summing to one and specific enthalpy.
    Both open x ends use this reservoir only on inflow. y is periodic or
    impermeable; z is impermeable. Predictor wall-normal speeds must be zero.

    Every Picard iteration restarts transport from gas+increments, so sources
    are applied once. The EOS updates the density guess for pressure; it never
    replaces conserved masses. Species and enthalpy use that same mass flux.
    At acceptance, velocity uses the same thermodynamic donor face density as
    the predictor flux. This transport_density is returned explicitly; it is
    NOT the interpolated final cell density used by low_mach.mass_flux.
    Returned species fluxes are authoritative for the committed mass budget.

    Donor-cell transport is first order. A step is rejected on nonconvergence,
    outgoing mass fraction >1, negative species, subfreezing temperature, or
    nonpositive heat capacity at constant volume. Default relative tolerance is
    max(1e-9,50*machine_epsilon); explicit tolerances override it. No clipping or global
    redistribution fixes a failed step. Rejection preserves input fields and
    predictor velocity, returns zero committed fluxes/pressure, and retains
    candidate diagnostics. The driver must reduce dt/recompute. This stage
    excludes momentum prediction, gravity, diffusion, condensate, SGS energy,
    mechanical pressure work and changing thermodynamic pressure.
    """
    if poisson.open_y:
        raise ValueError("moist transport currently requires impermeable or periodic y")
    if (
        (tolerance is not None and tolerance <= 0)
        or max_iterations < 1
        or not 0 < relaxation <= 1
    ):
        raise ValueError("invalid EOS iteration controls")
    grid = poisson.grid
    ambient = jnp.asarray(ambient_specific)
    if ambient.shape != (3, grid.nz, grid.ny):
        raise ValueError("ambient must have shape (3,nz,ny)")

    def step(gas, predictor, increments, dt):
        previous = gas.dry_density + gas.vapor_density
        allowed = (
            max(1e-9, 50 * jnp.finfo(previous.dtype).eps)
            if tolerance is None
            else tolerance
        )
        staged = jax.tree.map(lambda a, b: a + b, gas, increments)
        staged_density = staged.dry_density + staged.vapor_density
        specific = jnp.stack(staged) / staged_density
        source = (increments.dry_density + increments.vapor_density) / dt
        donor_density = moist_gas_eos_density(staged, config)
        ambient_density = moist_gas_eos_density(MoistGasFields(*ambient), config)

        def transport_density(direction):
            return StaggeredVelocity(
                *(
                    r[0]
                    for r in _donor_faces(
                        donor_density[None], direction, ambient_density[None], poisson
                    )
                )
            )

        zero_flux = jax.tree.map(lambda v: jnp.zeros((3, *v.shape), v.dtype), predictor)
        zero_p = jnp.zeros_like(previous)
        initial = (
            jnp.array(0),
            moist_gas_eos_density(staged, config),
            staged,
            predictor,
            transport_density(predictor),
            zero_p,
            zero_flux,
            jnp.array(jnp.inf, dtype=previous.dtype),
            jnp.array(jnp.inf, dtype=previous.dtype),
            jnp.array(0.0, dtype=previous.dtype),
            jnp.array(jnp.inf, dtype=previous.dtype),
            _admissible(staged, config) & (dt > 0),
        )

        def condition(state):
            n, _, _, _, _, _, _, error, continuity, _, donor_error, valid = state
            return (
                (n < max_iterations)
                & ((error > allowed) | (continuity > allowed) | (donor_error > allowed))
                & valid
            )

        def iterate(state):
            n, guess, _, direction, _, _, _, _, _, _, _, _ = state
            rho_faces = transport_density(direction)
            predictor_flux = StaggeredVelocity(
                *(r * u for r, u in zip(rho_faces, predictor))
            )
            flow, pressure = project_mass_flux(
                predictor_flux, previous, guess, poisson, dt, mass_source=source
            )
            fluxes = _specific_fluxes(specific, flow, ambient, poisson)
            tendency = -jax.vmap(lambda f: divergence(f, grid))(fluxes)
            q = jnp.stack(staged) + dt * tendency
            candidate = MoistGasFields(*q)
            density = candidate.dry_density + candidate.vapor_density
            eos = moist_gas_eos_density(candidate, config)
            error = jnp.max(jnp.abs(density - eos) / eos)
            continuity = jnp.max(
                jnp.abs(density - previous + dt * (divergence(flow, grid) - source))
                / previous
            )
            outgoing = jnp.max(
                dt * _outgoing_mass_rate(flow, grid, poisson) / staged_density
            )
            valid = (
                _admissible(candidate, config) & jnp.isfinite(error) & (outgoing <= 1)
            )
            recovered = StaggeredVelocity(*(f / r for f, r in zip(flow, rho_faces)))
            # A pressure correction can reverse a face. Require the density
            # and scalar donors to agree before accepting this Picard iterate.
            corrected_faces = transport_density(flow)
            donor_error = jnp.max(
                jnp.stack(
                    [
                        jnp.max(jnp.abs(a - b) / b)
                        for a, b in zip(rho_faces, corrected_faces)
                    ]
                )
            )
            return (
                n + 1,
                (1 - relaxation) * guess + relaxation * eos,
                candidate,
                recovered,
                rho_faces,
                pressure,
                fluxes,
                error,
                continuity,
                outgoing,
                donor_error,
                valid,
            )

        (
            n,
            _,
            candidate,
            velocity,
            rho_faces,
            pressure,
            fluxes,
            error,
            continuity,
            outgoing,
            donor_error,
            valid,
        ) = jax.lax.while_loop(condition, iterate, initial)
        accepted = (
            valid
            & (error <= allowed)
            & (continuity <= allowed)
            & (donor_error <= allowed)
        )
        choose = lambda a, b: jnp.where(accepted, a, b)
        return MoistGasTransport(
            jax.tree.map(choose, candidate, gas),
            jax.tree.map(choose, velocity, predictor),
            choose(pressure, zero_p),
            jax.tree.map(choose, rho_faces, jax.tree.map(jnp.zeros_like, predictor)),
            jax.tree.map(choose, fluxes, zero_flux),
            accepted,
            n,
            error,
            donor_error,
            continuity,
            outgoing,
        )

    return step
