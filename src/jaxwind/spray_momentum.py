"""Mass-consistent momentum transport on uniform staggered control volumes.

Opt-in advective/source predictor, not a pressure-coupled LES timestep. Each
component uses the half-cell partition of the SAME primary gas mass flux.
Boundary dual cells have half width. Wall impulses and numerical kinetic-energy
loss are explicit; neither is silently assigned to the physical SGS reservoir.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .state import StaggeredVelocity


class MomentumTransport(NamedTuple):
    velocity: StaggeredVelocity
    momentum: StaggeredVelocity
    inertia_density: StaggeredVelocity
    fluxes: tuple[StaggeredVelocity, ...]
    kinetic_fluxes: tuple[StaggeredVelocity, ...]
    wall_impulse: StaggeredVelocity
    kinetic_defect: StaggeredVelocity
    accepted: jax.Array
    mass_error: jax.Array
    outgoing_fraction: jax.Array


def _slice(values, axis, start, stop):
    slices = [slice(None)] * values.ndim
    slices[axis] = slice(start, stop)
    return values[tuple(slices)]


def dual_average(values, component, periodic):
    """Half-cell inventory average, with copied nonperiodic endpoints."""
    if periodic:
        return 0.5 * (values + jnp.roll(values, 1, component))
    return jnp.concatenate(
        (
            _slice(values, component, 0, 1),
            0.5
            * (_slice(values, component, 0, -1) + _slice(values, component, 1, None)),
            _slice(values, component, -1, None),
        ),
        axis=component,
    )


def dual_widths(grid, component, periodic, dtype):
    """Widths of the dual cells in their component direction."""
    n = (grid.nz, grid.ny, grid.nx)[component]
    h = (grid.dz, grid.dy, grid.dx)[component]
    widths = jnp.full((n if periodic else n + 1,), h, dtype=dtype)
    if not periodic:
        widths = widths.at[0].multiply(0.5).at[-1].multiply(0.5)
    return widths


def dual_volumes(grid, component, periodic, dtype):
    shape = [1, 1, 1]
    shape[component] = -1
    cross_section = grid.dx * grid.dy * grid.dz / (grid.dz, grid.dy, grid.dx)[component]
    return dual_widths(grid, component, periodic, dtype).reshape(shape) * cross_section


def dual_mass_flux(flow, component, periodic):
    """Three face fluxes for one component's dual mesh, ordered x,y,z."""
    parts = []
    for axis, face in zip((2, 1, 0), flow):
        if axis != component or periodic[component]:
            parts.append(dual_average(face, component, periodic[component]))
        else:
            # Interior dual faces lie at primary cell centres. Physical end
            # faces retain the primary boundary mass flux exactly.
            middle = 0.5 * (_slice(face, axis, 0, -1) + _slice(face, axis, 1, None))
            parts.append(
                jnp.concatenate(
                    (_slice(face, axis, 0, 1), middle, _slice(face, axis, -1, None)),
                    axis=axis,
                )
            )
    return StaggeredVelocity(*parts)


def _dual_divergence(flux, grid, component, periodic):
    result = 0.0
    for axis, face in zip((2, 1, 0), flux):
        difference = (
            jnp.roll(face, -1, axis) - face
            if periodic[axis]
            else jnp.diff(face, axis=axis)
        )
        if axis == component:
            shape = [1, 1, 1]
            shape[axis] = -1
            width = dual_widths(
                grid, component, periodic[component], face.dtype
            ).reshape(shape)
        else:
            width = jnp.asarray((grid.dz, grid.dy, grid.dx)[axis], dtype=face.dtype)
        result = result + difference / width
    return result


def _donors(values, flow, ambient, periodic):
    parts = []
    for axis, face in zip((2, 1, 0), flow):
        if periodic[axis]:
            left, right = jnp.roll(values, 1, axis), values
        else:
            low = high = ambient[..., None] if axis == 2 else None
            if axis != 2:
                low, high = _slice(values, axis, 0, 1), _slice(values, axis, -1, None)
            left = jnp.concatenate((low, values), axis=axis)
            right = jnp.concatenate((values, high), axis=axis)
        parts.append(jnp.where(face >= 0, left, right))
    return StaggeredVelocity(*parts)


def _wall_zero(values, component, periodic):
    # x is periodic or open; y is periodic or impermeable; z impermeable.
    if component == 0 or (component == 1 and not periodic[1]):
        slices = [slice(None)] * 3
        slices[component] = 0
        values = values.at[tuple(slices)].set(0.0)
        slices[component] = -1
        values = values.at[tuple(slices)].set(0.0)
    return values


def build_momentum_transport(grid, *, periodic_x, periodic_y, tolerance=1e-9):
    """Build a conservative source/advective predictor on a uniform grid.

    step(rho_old, rho_new, velocity, mass_increment, momentum_increment,
         mass_flux, ambient_velocity, dt)

    Densities/sources are primary-cell quantities; momentum_increment has shape
    (3,nz,ny,nx) in kg/(m2 s), integrated over dt, including FULL evaporated
    momentum and force impulse. It is not an acceleration or a source-relative
    force. ambient_velocity has shape (3,nz,ny), in m/s, for both x inflows.
    Source-updated velocity is the transported donor quantity. rho_new must
    agree with the supplied mass flux and source. Rejecting this predictor must
    roll back the carrier/core step in a future driver; this function cannot
    roll back another component's previously committed state.

    Impenetrable normal velocities are imposed with an explicit gas impulse.
    The opposite integrated impulse belongs to the wall. Momentum fluxes and
    kinetic_defect use component dual volumes, not primary cell volumes.
    kinetic_defect includes upwind mixing and wall removal of mean kinetic
    energy, excludes the source's kinetic change, and is NOT physical SGS k.
    """
    if not grid.is_uniform:
        raise ValueError(
            "conservative spray momentum currently requires a uniform grid"
        )
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    periodic = (False, periodic_y, periodic_x)
    shape = (grid.nz, grid.ny, grid.nx)

    def step(old, new, velocity, mass_increment, momentum_increment, flow, ambient, dt):
        if momentum_increment.shape != (3, *shape) or ambient.shape != (
            3,
            grid.nz,
            grid.ny,
        ):
            raise ValueError("invalid momentum source or ambient velocity shape")
        staged = old + mass_increment
        valid = jnp.all(old > 0) & jnp.all(staged > 0) & jnp.all(new > 0) & (dt > 0)
        valid &= jnp.all(jnp.isfinite(jnp.stack((old, new, staged, mass_increment))))
        valid &= jnp.all(jnp.isfinite(momentum_increment)) & jnp.all(
            jnp.isfinite(ambient)
        )
        # Boundary fluxes must agree with the supported impermeable walls.
        valid &= jnp.all(flow.z[0] == 0) & jnp.all(flow.z[-1] == 0)
        if not periodic_y:
            valid &= jnp.all(flow.y[:, 0] == 0) & jnp.all(flow.y[:, -1] == 0)
        momenta, densities, speeds, fluxes, impulses, defects = [], [], [], [], [], []
        kinetic_fluxes = []
        mass_error, cfl = jnp.array(0.0, old.dtype), jnp.array(0.0, old.dtype)
        for index, (component, u) in enumerate(zip((2, 1, 0), velocity)):
            rho_old = dual_average(old, component, periodic[component])
            rho = dual_average(new, component, periodic[component])
            rho_star = dual_average(staged, component, periodic[component])
            pstar = rho_old * u + dual_average(
                momentum_increment[index], component, periodic[component]
            )
            ustar = pstar / rho_star
            g = dual_mass_flux(flow, component, periodic)
            ambient_i = ambient[index]
            if component != 2:
                ambient_i = dual_average(ambient_i, component, periodic[component])
            donors = _donors(ustar, g, ambient_i, periodic)
            pf = StaggeredVelocity(*(f * a for f, a in zip(g, donors)))
            kf = StaggeredVelocity(*(0.5 * f * a**2 for f, a in zip(g, donors)))
            predicted = pstar - dt * _dual_divergence(pf, grid, component, periodic)
            momentum = _wall_zero(predicted, component, periodic)
            impulse = momentum - predicted
            defect = (
                pstar**2 / (2 * rho_star)
                - dt * _dual_divergence(kf, grid, component, periodic)
                - momentum**2 / (2 * rho)
            )
            expected_mass = rho_star - dt * _dual_divergence(
                g, grid, component, periodic
            )
            mass_error = jnp.maximum(
                mass_error, jnp.max(jnp.abs(rho - expected_mass) / rho_star)
            )
            # Sum positive upper and negative lower outgoing fluxes;
            # net divergence alone would cancel a steady throughflow.
            outgoing = 0.0
            for axis, f in zip((2, 1, 0), g):
                low = f if periodic[axis] else _slice(f, axis, 0, -1)
                high = (
                    jnp.roll(f, -1, axis)
                    if periodic[axis]
                    else _slice(f, axis, 1, None)
                )
                width = jnp.asarray((grid.dz, grid.dy, grid.dx)[axis], old.dtype)
                if axis == component:
                    dims = [1, 1, 1]
                    dims[axis] = -1
                    width = dual_widths(
                        grid, component, periodic[component], old.dtype
                    ).reshape(dims)
                outgoing = (
                    outgoing + (jnp.maximum(high, 0) - jnp.minimum(low, 0)) / width
                )
            cfl = jnp.maximum(cfl, jnp.max(dt * outgoing / rho_star))
            valid &= jnp.all(jnp.isfinite(momentum)) & jnp.all(jnp.isfinite(defect))
            momenta.append(momentum)
            densities.append(rho)
            speeds.append(momentum / rho)
            fluxes.append(pf)
            kinetic_fluxes.append(kf)
            impulses.append(impulse)
            defects.append(defect)
        accepted = valid & (mass_error <= tolerance) & (cfl <= 1)
        choose = lambda a, b: jnp.where(accepted, a, b)
        old_rho = StaggeredVelocity(
            *(dual_average(old, c, periodic[c]) for c in (2, 1, 0))
        )
        old_p = StaggeredVelocity(*(r * u for r, u in zip(old_rho, velocity)))
        commit = lambda tree: jax.tree.map(lambda x: choose(x, jnp.zeros_like(x)), tree)
        return MomentumTransport(
            jax.tree.map(choose, StaggeredVelocity(*speeds), velocity),
            jax.tree.map(choose, StaggeredVelocity(*momenta), old_p),
            jax.tree.map(choose, StaggeredVelocity(*densities), old_rho),
            commit(tuple(fluxes)),
            commit(tuple(kinetic_fluxes)),
            commit(StaggeredVelocity(*impulses)),
            commit(StaggeredVelocity(*defects)),
            accepted,
            mass_error,
            cfl,
        )

    return step
