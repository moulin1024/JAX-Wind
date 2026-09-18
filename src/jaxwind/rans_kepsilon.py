"""Opt-in incompressible standard k-epsilon carrier closure for spray audits.

Launder-Spalding constants; ANSYS Fluent 12 theory sections 4.4.1 and 4.12.2.
The wall treatment includes the established low-Re stepwise epsilon-wall
branch (OpenFOAM epsilonWallFunction), so this is not an exact Fluent match.
This is standard, not realizable, k-epsilon. No buoyancy production, particle
fluctuation production, or stochastic dispersion is asserted. The prescribed
mean inlet supplies k and epsilon, without simultaneous resolved turbulence.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .cryogenic import _cell_to_velocity_faces
from .numerics.discretization import cell_velocity
from .scalar_transport import transport_scalars
from .sgs import (
    _to_cell_from_xy_edge,
    _to_cell_from_xz_edge,
    _to_cell_from_yz_edge,
    _to_xy_edge_from_cell,
    _to_xz_edge_from_cell,
    _to_yz_edge_from_cell,
    edge_gradients,
)
from .state import spanwise_is_periodic, streamwise_is_periodic

CMU = 0.09
C1 = 1.44
C2 = 1.92
KAPPA = 0.4187
E_WALL = 9.793
FLOOR = 1e-12


class KEpsilonState(NamedTuple):
    kinetic_energy: jax.Array
    dissipation: jax.Array


class RANSInertialSolution(NamedTuple):
    velocity: object
    pressure: jax.Array
    momentum_tendency: object
    scalar: jax.Array
    scalar_tendency: jax.Array
    time: jax.Array
    step: jax.Array
    moisture: object
    parcels: object
    turbulence: KEpsilonState


def turbulent_viscosity(state):
    return CMU * state.kinetic_energy**2 / jnp.maximum(state.dissipation, FLOOR)


def production(velocity, grid, boundaries, viscosity, *, gradients=None):
    """Transfer native modeled-stress work into cell-centred turbulent energy.

    Form stress/strain products at the same locations as momentum stresses,
    then average their powers to cells. Averaging gradients before squaring
    loses resolved-to-modeled energy on a staggered grid. On a uniform mesh
    this quadrature balances stress work when boundary energy flux is zero.
    With supplied native edge gradients, use the same gradients as momentum
    stresses, including physical inlet traction. Boundary work then closes
    the identity; no separate inlet-production source is added here.
    The surrounding integrator substitutes the separate wall-function source
    in wall-adjacent cells; that closure is not part of this bulk identity.
    """
    if not grid.is_uniform:
        raise ValueError("RANS stress-work production requires a uniform grid")
    g = edge_gradients(velocity, grid, boundaries) if gradients is None else gradients
    nu = jnp.broadcast_to(jnp.asarray(viscosity, g["xx"].dtype), g["xx"].shape)
    open_x = not streamwise_is_periodic(velocity, grid)
    wall_y = not spanwise_is_periodic(velocity, grid)
    value = 2 * nu * (g["xx"] ** 2 + g["yy"] ** 2 + g["zz"] ** 2)
    value += _to_cell_from_xy_edge(
        _to_xy_edge_from_cell(nu, open_x=open_x, wall_y=wall_y)
        * (g["xy"] + g["yx"]) ** 2,
        open_x=open_x,
        wall_y=wall_y,
    )
    value += _to_cell_from_xz_edge(
        _to_xz_edge_from_cell(nu, open_x=open_x) * (g["xz"] + g["zx"]) ** 2,
        open_x=open_x,
    )
    value += _to_cell_from_yz_edge(
        _to_yz_edge_from_cell(nu, wall_y=wall_y) * (g["yz"] + g["zy"]) ** 2,
        wall_y=wall_y,
    )
    return value


def local_sources(state, generation, dt):
    """Positive first-order Patankar update of the standard local source ODE.

    Destruction is implicit with frozen epsilon/k. No clipping of a negative
    candidate and no empirical production limiter are used.
    """
    k, eps = state
    rate = eps / jnp.maximum(k, FLOOR)
    return KEpsilonState(
        (k + dt * generation) / (1 + dt * rate),
        (eps + dt * C1 * rate * generation) / (1 + dt * C2 * rate),
    )


def wall_terms(velocity, state, grid, molecular_viscosity):
    """k-based wall traction and equilibrium epsilon at four adiabatic walls.

    k has zero normal flux. Wall epsilon is a boundary-cell constraint, not a
    transported equation. At corners production and epsilon are area-averaged;
    wall tractions add. Linear stress applies below y*=11.225, with no
    turbulent production and epsilon=2*nu*k/y**2. The stepwise epsilon
    condition is discontinuous at this threshold, as in the established
    low-Re wall-function branch; it is not a low-Re bulk turbulence model.
    """
    cells = cell_velocity(velocity)
    k, _ = state
    force = [jnp.zeros_like(k) for _ in range(3)]
    generation, epsilon, weight = (jnp.zeros_like(k) for _ in range(3))
    for axis, widths, components in (
        (0, grid.z_widths, (0, 1)),
        (1, grid.y_widths, (0, 2)),
    ):
        for index in (0, -1):
            width = float(widths[index])
            distance = width / 2
            location = [slice(None)] * 3
            location[axis] = index
            location = tuple(location)
            kp = jnp.maximum(k[location], FLOOR)
            scale = CMU**0.25 * jnp.sqrt(kp)
            ystar = scale * distance / molecular_viscosity
            loglaw = jnp.log(E_WALL * jnp.maximum(ystar, 11.225)) / KAPPA
            logarithmic = ystar > 11.225
            coefficient = jnp.where(
                logarithmic, scale / loglaw, molecular_viscosity / distance
            )
            speed2 = sum(cells[c][location] ** 2 for c in components)
            stress2 = coefficient**2 * speed2
            for c in components:
                force[c] = (
                    force[c].at[location].add(-coefficient * cells[c][location] / width)
                )
            # The Cmu**.25 factor is required: with tau/rho=scale**2,
            # Pk=scale**3/(kappa*y)=epsilon at equilibrium. Continuing this
            # expression into the viscous layer spuriously diverges as k->0.
            pk = jnp.where(logarithmic, stress2 / (KAPPA * scale * distance), 0.0)
            ep = jnp.where(
                logarithmic,
                CMU**0.75 * kp**1.5 / (KAPPA * distance),
                2 * molecular_viscosity * kp / distance**2,
            )
            area_weight = 1 / width
            generation = generation.at[location].add(area_weight * pk)
            epsilon = epsilon.at[location].add(area_weight * ep)
            weight = weight.at[location].add(area_weight)
    return (
        _cell_to_velocity_faces(*force, grid),
        generation / jnp.maximum(weight, FLOOR),
        epsilon / jnp.maximum(weight, FLOOR),
        weight > 0,
    )


def constrain_wall_epsilon(state, velocity, grid, molecular_viscosity):
    _, _, eps, mask = wall_terms(velocity, state, grid, molecular_viscosity)
    return state._replace(dissipation=jnp.where(mask, eps, state.dissipation))


def advance_turbulence(
    state,
    velocity,
    grid,
    boundaries,
    molecular_viscosity,
    inlet,
    dt,
    *,
    model="standard-k-epsilon",
    transport_scheme="upwind",
    gradients=None,
):
    """Conservative scalar transport plus positive source splitting.

    Both transported quantities use prescribed inlet flux and zero wall flux.
    Eddy viscosity is frozen during this step; the carrier uses the same field.
    Optional native edge gradients must correspond to this velocity and its
    physical boundary data. They supply both stress-work production and the
    realizable invariants. Inlet scalar fluxes and wall substitutions are
    unchanged; None retains the original gradient reconstruction.
    """
    if transport_scheme not in ("upwind", "muscl-mc"):
        raise ValueError("turbulence transport must be upwind or muscl-mc")
    state = constrain_wall_epsilon(state, velocity, grid, molecular_viscosity)
    if model not in ("standard-k-epsilon", "realizable-k-epsilon"):
        raise ValueError("unsupported transported turbulence model")
    realizable = model == "realizable-k-epsilon"
    if realizable:
        from . import rans_realizable

        strain, ustar, a_s = rans_realizable.velocity_invariants(
            velocity, grid, boundaries, gradients=gradients
        )
        nu_t = rans_realizable.viscosity_from_invariants(state, ustar, a_s)
    else:
        nu_t = turbulent_viscosity(state)
    fields = []
    for field, sigma, ambient in zip(state, (1.0, 1.2 if realizable else 1.3), inlet):
        reservoir = jnp.full((1, grid.nz, grid.ny), ambient, field.dtype)
        fields.append(
            transport_scalars(
                field[None],
                velocity,
                grid,
                dt,
                reservoir,
                molecular_viscosity + nu_t / sigma,
                scheme=transport_scheme,
            )[0]
        )
    current = KEpsilonState(*fields)
    source = production(velocity, grid, boundaries, nu_t, gradients=gradients)
    _, wall_source, eps_wall, mask = wall_terms(
        velocity, current, grid, molecular_viscosity
    )
    source = jnp.where(mask, wall_source, source)
    current = current._replace(
        dissipation=jnp.where(mask, eps_wall, current.dissipation)
    )
    result = (
        rans_realizable.local_sources(current, source, strain, molecular_viscosity, dt)
        if realizable
        else local_sources(current, source, dt)
    )
    return constrain_wall_epsilon(result, velocity, grid, molecular_viscosity)
