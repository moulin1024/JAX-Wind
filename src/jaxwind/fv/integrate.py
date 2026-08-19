"""Fractional-step time integration for the staggered finite-volume solver.

Both schemes advance an explicit tendency and then project, so the velocity
carried between steps is discretely divergence-free at every step and, for the
Runge-Kutta scheme, at every substage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxwind.domain.grid import UniformGrid

from .operators import advection, diffusion, pressure_gradient
from .poisson import PressurePoisson, project
from .sgs import AnisotropicMinimumDissipation, subfilter_tendency
from .state import Boundaries, StaggeredVelocity, enforce_impermeability, zeros
from .wall import MoninObukhovWall, wall_tendency


# Wray's low-storage three-stage scheme, the standard explicit choice for
# wall-bounded incompressible flow.
_RK3_CURRENT = (8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0)
_RK3_PREVIOUS = (0.0, -17.0 / 60.0, -5.0 / 12.0)


class Solution(NamedTuple):
    """Everything the integrator needs to take the next step."""

    velocity: StaggeredVelocity
    pressure: jnp.ndarray
    tendency: StaggeredVelocity
    time: jnp.ndarray
    step: jnp.ndarray


@dataclass(frozen=True, slots=True)
class FlowModel:
    """Constant-property incompressible flow with an optional body force.

    ``subfilter`` selects the large-eddy closure.  It defaults to none, so a
    direct simulation is what an unconfigured model gives; set it to an
    :class:`~jaxwind.fv.sgs.AnisotropicMinimumDissipation` instance to run a
    large-eddy simulation.
    """

    viscosity: float = 0.0
    body_force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    forcing: Callable[[StaggeredVelocity, jnp.ndarray], StaggeredVelocity] | None = None
    subfilter: AnisotropicMinimumDissipation | None = None
    surface: MoninObukhovWall | None = None


def initial_solution(
    grid: UniformGrid,
    velocity: StaggeredVelocity | None = None,
    *,
    dtype: str = "float64",
) -> Solution:
    """Wrap a velocity field as a solution at rest at ``t = 0``."""
    field = zeros(grid, dtype) if velocity is None else velocity
    resolved = field.x.dtype
    return Solution(
        enforce_impermeability(field),
        jnp.zeros((grid.nz, grid.ny, grid.nx), resolved),
        StaggeredVelocity(
            jnp.zeros_like(field.x),
            jnp.zeros_like(field.y),
            jnp.zeros_like(field.z),
        ),
        jnp.asarray(0.0, resolved),
        jnp.asarray(0, jnp.int32),
    )


def _add(left: StaggeredVelocity, right: StaggeredVelocity) -> StaggeredVelocity:
    return StaggeredVelocity(
        left.x + right.x,
        left.y + right.y,
        left.z + right.z,
    )


def _combine(
    velocity: StaggeredVelocity,
    current: StaggeredVelocity,
    previous: StaggeredVelocity,
    current_weight: jnp.ndarray,
    previous_weight: jnp.ndarray,
) -> StaggeredVelocity:
    return StaggeredVelocity(
        velocity.x + current_weight * current.x + previous_weight * previous.x,
        velocity.y + current_weight * current.y + previous_weight * previous.y,
        velocity.z + current_weight * current.z + previous_weight * previous.z,
    )


def build_tendency(
    grid: UniformGrid,
    boundaries: Boundaries,
    model: FlowModel,
) -> Callable[[StaggeredVelocity, jnp.ndarray], StaggeredVelocity]:
    """Return the explicit right-hand side of the momentum equations."""
    force_x, force_y, force_z = model.body_force

    def tendency(
        velocity: StaggeredVelocity,
        time: jnp.ndarray,
    ) -> StaggeredVelocity:
        total = advection(velocity, grid)
        if model.viscosity:
            total = _add(total, diffusion(velocity, grid, boundaries, model.viscosity))
        if force_x or force_y or force_z:
            wall = jnp.zeros_like(velocity.z[:1])
            interior = jnp.full_like(velocity.z[1:-1], force_z)
            total = _add(
                total,
                StaggeredVelocity(
                    jnp.full_like(velocity.x, force_x),
                    jnp.full_like(velocity.y, force_y),
                    jnp.concatenate((wall, interior, wall), axis=0),
                ),
            )
        if model.surface is not None:
            total = _add(total, wall_tendency(velocity, grid, model.surface))
        if model.subfilter is not None:
            subfilter, _ = subfilter_tendency(
                velocity,
                grid,
                boundaries,
                model.subfilter,
            )
            total = _add(total, subfilter)
        if model.forcing is not None:
            total = _add(total, model.forcing(velocity, time))
        return total

    return tendency


def build_step(
    grid: UniformGrid,
    boundaries: Boundaries,
    poisson: PressurePoisson,
    model: FlowModel,
    *,
    scheme: str = "rk3",
) -> Callable[[Solution, float], Solution]:
    """Build one time step of the fractional-step method.

    ``rk3`` projects every substage, so each intermediate field is discretely
    solenoidal.  ``fast-rk3`` projects only the final substage and carries the
    previous step's pressure gradient through the intermediate ones, which
    costs one pressure solve per step instead of three.  ``ab2`` is the
    two-step Adams-Bashforth alternative.
    """
    if scheme not in ("rk3", "fast-rk3", "ab2"):
        raise ValueError(f"unsupported time scheme: {scheme!r}")
    tendency = build_tendency(grid, boundaries, model)

    def rk3_step(solution: Solution, dt: float) -> Solution:
        velocity = solution.velocity
        time = solution.time
        previous = solution.tendency
        pressure = solution.pressure
        step_size = jnp.asarray(dt, velocity.x.dtype)
        for current_weight, previous_weight in zip(_RK3_CURRENT, _RK3_PREVIOUS):
            current = tendency(velocity, time)
            candidate = _combine(
                velocity,
                current,
                previous,
                step_size * current_weight,
                step_size * previous_weight,
            )
            substep = dt * (current_weight + previous_weight)
            velocity, pressure = project(
                enforce_impermeability(candidate),
                poisson,
                substep,
            )
            previous = current
            time = time + substep
        return Solution(
            velocity,
            pressure,
            previous,
            solution.time + step_size,
            solution.step + 1,
        )

    def fast_rk3_step(solution: Solution, dt: float) -> Solution:
        """Runge-Kutta 3 with a single projection at the end of the step.

        The pressure gradient of the previous step is applied at every
        substage, so the intermediate fields stay close to solenoidal and the
        final projection only has to remove what the step itself introduced.
        The projection returns that increment, which accumulates into the
        pressure carried to the next step; a converged steady state therefore
        needs no correction at all.
        """
        velocity = solution.velocity
        time = solution.time
        previous = solution.tendency
        pressure = solution.pressure
        step_size = jnp.asarray(dt, velocity.x.dtype)
        # One gradient per step: the applied pressure is frozen across the
        # substages, so recomputing it would give the same answer three times.
        lagged = pressure_gradient(pressure, grid)
        last = len(_RK3_CURRENT) - 1
        for stage, (current_weight, previous_weight) in enumerate(
            zip(_RK3_CURRENT, _RK3_PREVIOUS)
        ):
            current = tendency(velocity, time)
            candidate = _combine(
                velocity,
                current,
                previous,
                step_size * current_weight,
                step_size * previous_weight,
            )
            substep = dt * (current_weight + previous_weight)
            scale = jnp.asarray(substep, velocity.x.dtype)
            candidate = enforce_impermeability(
                StaggeredVelocity(
                    candidate.x - scale * lagged.x,
                    candidate.y - scale * lagged.y,
                    candidate.z - scale * lagged.z,
                )
            )
            if stage == last:
                velocity, correction = project(candidate, poisson, substep)
                # The substage weights sum to one, so the lagged gradient
                # already applies an impulse of dt * grad(p) over the step.
                # The final projection adds a further substep * grad(correction),
                # so only that fraction of the correction belongs in the
                # pressure carried forward; taking all of it would over-relax
                # the pressure by dt / substep and the step would diverge.
                pressure = pressure + correction * (substep / dt)
            else:
                velocity = candidate
            previous = current
            time = time + substep
        return Solution(
            velocity,
            pressure,
            previous,
            solution.time + step_size,
            solution.step + 1,
        )

    def ab2_step(solution: Solution, dt: float) -> Solution:
        step_size = jnp.asarray(dt, solution.velocity.x.dtype)
        current = tendency(solution.velocity, solution.time)
        first = solution.step == 0
        current_weight = jnp.where(first, 1.0, 1.5).astype(step_size.dtype)
        previous_weight = jnp.where(first, 0.0, -0.5).astype(step_size.dtype)
        candidate = _combine(
            solution.velocity,
            current,
            solution.tendency,
            step_size * current_weight,
            step_size * previous_weight,
        )
        velocity, pressure = project(
            enforce_impermeability(candidate),
            poisson,
            dt,
        )
        return Solution(
            velocity,
            pressure,
            current,
            solution.time + step_size,
            solution.step + 1,
        )

    return {"rk3": rk3_step, "fast-rk3": fast_rk3_step, "ab2": ab2_step}[scheme]


def build_run(
    step: Callable[[Solution, float], Solution],
    *,
    jit: bool = True,
) -> Callable[[Solution, float, int], Solution]:
    """Build a fixed-step driver that keeps the whole loop on the device."""

    def run(solution: Solution, dt: float, steps: int) -> Solution:
        def body(_: int, carry: Solution) -> Solution:
            return step(carry, dt)

        return jax.lax.fori_loop(0, steps, body, solution)

    return jax.jit(run, static_argnums=2) if jit else run


__all__ = [
    "FlowModel",
    "Solution",
    "build_run",
    "build_step",
    "build_tendency",
    "initial_solution",
]
