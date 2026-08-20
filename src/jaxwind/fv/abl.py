"""Coupled momentum and scalar integration for finite-volume ABL cases."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxwind.domain.grid import UniformGrid

from .buoyancy import LinearBoussinesqBuoyancy, boussinesq_tendency
from .integrate import FlowModel, build_adaptive_run, build_tendency
from .operators import pressure_gradient
from .poisson import PressurePoisson, project
from .scalar import PassiveScalar, scalar_tendency
from .sgs import eddy_viscosity
from .state import Boundaries, StaggeredVelocity, enforce_impermeability, zeros
from .surface import (
    MoninObukhovSurface,
    coupled_surface_exchange,
    surface_momentum_tendency,
)


class AtmosphericSolution(NamedTuple):
    """AB2 state for velocity, pressure, and a cell-centred scalar."""

    velocity: StaggeredVelocity
    pressure: jnp.ndarray
    momentum_tendency: StaggeredVelocity
    scalar: jnp.ndarray
    scalar_tendency: jnp.ndarray
    time: jnp.ndarray
    step: jnp.ndarray


def initial_atmospheric_solution(
    grid: UniformGrid,
    velocity: StaggeredVelocity | None = None,
    scalar: jnp.ndarray | None = None,
    *,
    dtype: str = "float32",
) -> AtmosphericSolution:
    """Create a coupled state with zero tendency and pressure."""

    field = zeros(grid, dtype) if velocity is None else velocity
    resolved = field.x.dtype
    scalar_field = (
        jnp.zeros((grid.nz, grid.ny, grid.nx), resolved)
        if scalar is None
        else jnp.asarray(scalar, resolved)
    )
    if scalar_field.shape != (grid.nz, grid.ny, grid.nx):
        raise ValueError("the passive scalar must be cell centred")
    zero_velocity = StaggeredVelocity(
        jnp.zeros_like(field.x),
        jnp.zeros_like(field.y),
        jnp.zeros_like(field.z),
    )
    return AtmosphericSolution(
        enforce_impermeability(field),
        jnp.zeros_like(scalar_field),
        zero_velocity,
        scalar_field,
        jnp.zeros_like(scalar_field),
        jnp.asarray(0.0, resolved),
        jnp.asarray(0, jnp.int32),
    )


def build_atmospheric_step(
    grid: UniformGrid,
    boundaries: Boundaries,
    poisson: PressurePoisson,
    momentum: FlowModel,
    scalar: PassiveScalar,
    buoyancy: LinearBoussinesqBuoyancy | None = None,
    surface_transfer: MoninObukhovSurface | None = None,
    *,
    scheme: str = "ab2",
) -> Callable[[AtmosphericSolution, float], AtmosphericSolution]:
    """Build one coupled AB2 or single-projection RK3 atmospheric step."""

    if scheme not in ("ab2", "fast-rk3", "rk3"):
        raise ValueError(f"unsupported atmospheric time scheme: {scheme!r}")
    if surface_transfer is not None and momentum.surface is not None:
        raise ValueError(
            "independent and coupled FV surface models are mutually exclusive"
        )
    momentum_rhs = build_tendency(grid, boundaries, momentum)

    def tendencies(velocity, scalar_field, execution_time):
        current_momentum = momentum_rhs(velocity, execution_time)
        exchange = None
        if surface_transfer is not None:
            exchange = coupled_surface_exchange(
                velocity,
                scalar_field,
                execution_time,
                grid,
                surface_transfer,
            )
            source = surface_momentum_tendency(velocity, exchange, grid)
            current_momentum = StaggeredVelocity(
                current_momentum.x + source.x,
                current_momentum.y + source.y,
                current_momentum.z + source.z,
            )
        if buoyancy is not None:
            source = boussinesq_tendency(scalar_field, buoyancy)
            current_momentum = StaggeredVelocity(
                current_momentum.x + source.x,
                current_momentum.y + source.y,
                current_momentum.z + source.z,
            )
        subfilter_viscosity = (
            0.0
            if momentum.subfilter is None
            else eddy_viscosity(
                velocity,
                grid,
                boundaries,
                momentum.subfilter,
            )
        )
        current_scalar = scalar_tendency(
            scalar_field,
            velocity,
            grid,
            scalar,
            eddy_viscosity=subfilter_viscosity,
            lower_flux=(None if exchange is None else exchange.scalar_flux),
        )
        return current_momentum, current_scalar

    def ab2_step(
        solution: AtmosphericSolution,
        dt: float,
    ) -> AtmosphericSolution:
        step_size = jnp.asarray(dt, solution.velocity.x.dtype)
        current_momentum, current_scalar = tendencies(
            solution.velocity,
            solution.scalar,
            solution.time,
        )
        first = solution.step == 0
        current_weight = jnp.where(first, 1.0, 1.5).astype(step_size.dtype)
        previous_weight = jnp.where(first, 0.0, -0.5).astype(step_size.dtype)
        candidate = StaggeredVelocity(
            solution.velocity.x
            + step_size
            * (
                current_weight * current_momentum.x
                + previous_weight * solution.momentum_tendency.x
            ),
            solution.velocity.y
            + step_size
            * (
                current_weight * current_momentum.y
                + previous_weight * solution.momentum_tendency.y
            ),
            solution.velocity.z
            + step_size
            * (
                current_weight * current_momentum.z
                + previous_weight * solution.momentum_tendency.z
            ),
        )
        velocity, pressure = project(
            enforce_impermeability(candidate),
            poisson,
            dt,
        )
        next_scalar = solution.scalar + step_size * (
            current_weight * current_scalar
            + previous_weight * solution.scalar_tendency
        )
        next_step = solution.step + 1
        next_time = step_size * next_step.astype(step_size.dtype)
        return AtmosphericSolution(
            velocity,
            pressure,
            current_momentum,
            next_scalar,
            current_scalar,
            next_time,
            next_step,
        )

    def rk3_step(
        solution: AtmosphericSolution,
        dt: float,
    ) -> AtmosphericSolution:
        current_weights = (8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0)
        previous_weights = (0.0, -17.0 / 60.0, -5.0 / 12.0)
        velocity = solution.velocity
        scalar_field = solution.scalar
        execution_time = solution.time
        previous_momentum = solution.momentum_tendency
        previous_scalar = solution.scalar_tendency
        pressure = solution.pressure
        step_size = jnp.asarray(dt, velocity.x.dtype)

        for current_weight, previous_weight in zip(
            current_weights,
            previous_weights,
        ):
            current_momentum, current_scalar = tendencies(
                velocity,
                scalar_field,
                execution_time,
            )
            current_scale = step_size * current_weight
            previous_scale = step_size * previous_weight
            candidate = StaggeredVelocity(
                velocity.x
                + current_scale * current_momentum.x
                + previous_scale * previous_momentum.x,
                velocity.y
                + current_scale * current_momentum.y
                + previous_scale * previous_momentum.y,
                velocity.z
                + current_scale * current_momentum.z
                + previous_scale * previous_momentum.z,
            )
            scalar_field = (
                scalar_field
                + current_scale * current_scalar
                + previous_scale * previous_scalar
            )
            substep = dt * (current_weight + previous_weight)
            velocity, pressure = project(
                enforce_impermeability(candidate),
                poisson,
                substep,
            )
            previous_momentum = current_momentum
            previous_scalar = current_scalar
            execution_time = execution_time + jnp.asarray(
                substep,
                execution_time.dtype,
            )

        return AtmosphericSolution(
            velocity,
            pressure,
            previous_momentum,
            scalar_field,
            previous_scalar,
            solution.time + step_size,
            solution.step + 1,
        )

    def fast_rk3_step(
        solution: AtmosphericSolution,
        dt: float,
    ) -> AtmosphericSolution:
        current_weights = (8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0)
        previous_weights = (0.0, -17.0 / 60.0, -5.0 / 12.0)
        velocity = solution.velocity
        scalar_field = solution.scalar
        execution_time = solution.time
        previous_momentum = solution.momentum_tendency
        previous_scalar = solution.scalar_tendency
        pressure = solution.pressure
        step_size = jnp.asarray(dt, velocity.x.dtype)
        lagged = pressure_gradient(pressure, grid)
        last = len(current_weights) - 1

        for stage, (current_weight, previous_weight) in enumerate(
            zip(current_weights, previous_weights)
        ):
            current_momentum, current_scalar = tendencies(
                velocity,
                scalar_field,
                execution_time,
            )
            momentum_current_scale = step_size * current_weight
            momentum_previous_scale = step_size * previous_weight
            candidate = StaggeredVelocity(
                velocity.x
                + momentum_current_scale * current_momentum.x
                + momentum_previous_scale * previous_momentum.x,
                velocity.y
                + momentum_current_scale * current_momentum.y
                + momentum_previous_scale * previous_momentum.y,
                velocity.z
                + momentum_current_scale * current_momentum.z
                + momentum_previous_scale * previous_momentum.z,
            )
            next_scalar = (
                scalar_field
                + momentum_current_scale * current_scalar
                + momentum_previous_scale * previous_scalar
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
                pressure = pressure + correction * (substep / dt)
            else:
                velocity = candidate
            scalar_field = next_scalar
            previous_momentum = current_momentum
            previous_scalar = current_scalar
            execution_time = execution_time + scale

        return AtmosphericSolution(
            velocity,
            pressure,
            previous_momentum,
            scalar_field,
            previous_scalar,
            solution.time + step_size,
            solution.step + 1,
        )

    return {
        "ab2": ab2_step,
        "fast-rk3": fast_rk3_step,
        "rk3": rk3_step,
    }[scheme]


def build_atmospheric_run(
    step: Callable[[AtmosphericSolution, float], AtmosphericSolution],
) -> Callable[[AtmosphericSolution, float, int], AtmosphericSolution]:
    """JIT a fixed-size block of coupled atmospheric steps."""

    def run(
        solution: AtmosphericSolution,
        dt: float,
        steps: int,
    ) -> AtmosphericSolution:
        return jax.lax.fori_loop(0, steps, lambda _, state: step(state, dt), solution)

    return jax.jit(run, static_argnums=2)


def build_adaptive_atmospheric_run(
    step: Callable[[AtmosphericSolution, float], AtmosphericSolution],
    grid: UniformGrid,
    *,
    cfl_ceiling: float,
    maximum_dt: float,
) -> Callable[[AtmosphericSolution, float, int], AtmosphericSolution]:
    """Build the adaptive-CFL driver for coupled RK3 atmospheric steps."""
    return build_adaptive_run(
        step,
        grid,
        cfl_ceiling=cfl_ceiling,
        maximum_dt=maximum_dt,
    )


__all__ = [
    "AtmosphericSolution",
    "build_adaptive_atmospheric_run",
    "build_atmospheric_run",
    "build_atmospheric_step",
    "initial_atmospheric_solution",
]
