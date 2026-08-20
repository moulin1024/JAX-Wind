"""Open-streamwise atmospheric integration driven by precursor planes."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jaxwind.domain.grid import UniformGrid

from .abl import AtmosphericSolution
from .buoyancy import LinearBoussinesqBuoyancy, boussinesq_tendency
from .integrate import FlowModel, build_tendency
from .open_boundary import (
    InflowPlane,
    enforce_open_scalar,
    enforce_open_velocity,
)
from .operators import pressure_gradient
from .poisson import PressurePoisson, project
from .scalar import PassiveScalar, scalar_tendency
from .sgs import eddy_viscosity
from .state import OPEN, Boundaries, StaggeredVelocity
from .surface import (
    MoninObukhovSurface,
    coupled_surface_exchange,
    surface_momentum_tendency,
)


def build_open_atmospheric_step(
    grid: UniformGrid,
    boundaries: Boundaries,
    poisson: PressurePoisson,
    momentum: FlowModel,
    scalar: PassiveScalar | None,
    buoyancy: LinearBoussinesqBuoyancy | None = None,
    surface_transfer: MoninObukhovSurface | None = None,
    *,
    scheme: str = "ab2",
) -> Callable[[AtmosphericSolution, float, InflowPlane], AtmosphericSolution]:
    """Build an open-boundary AB2 or single-projection fast-RK3 step."""
    if scheme not in ("ab2", "fast-rk3"):
        raise ValueError(f"unsupported open atmospheric time scheme: {scheme!r}")
    if boundaries.streamwise != OPEN:
        raise ValueError("open atmospheric integration requires open boundaries")
    if poisson.periodic_x:
        raise ValueError("open atmospheric integration requires nonperiodic pressure")
    if scalar is None and (buoyancy is not None or surface_transfer is not None):
        raise ValueError(
            "buoyancy and coupled surface transfer require an active scalar"
        )
    if surface_transfer is not None and momentum.surface is not None:
        raise ValueError(
            "independent and coupled FV surface models are mutually exclusive"
        )
    momentum_rhs = build_tendency(grid, boundaries, momentum)

    def tendencies(velocity, scalar_field, execution_time, inflow):
        current_velocity = enforce_open_velocity(velocity, inflow, grid)
        current_scalar_field = (
            scalar_field
            if scalar is None
            else enforce_open_scalar(scalar_field, inflow, grid)
        )
        current_momentum = momentum_rhs(current_velocity, execution_time)
        exchange = None
        if surface_transfer is not None:
            exchange = coupled_surface_exchange(
                current_velocity,
                current_scalar_field,
                execution_time,
                grid,
                surface_transfer,
            )
            source = surface_momentum_tendency(
                current_velocity,
                exchange,
                grid,
            )
            current_momentum = StaggeredVelocity(
                current_momentum.x + source.x,
                current_momentum.y + source.y,
                current_momentum.z + source.z,
            )
        if buoyancy is not None:
            source = boussinesq_tendency(
                current_scalar_field,
                buoyancy,
                x_face_count=current_velocity.x.shape[-1],
            )
            current_momentum = StaggeredVelocity(
                current_momentum.x + source.x,
                current_momentum.y + source.y,
                current_momentum.z + source.z,
            )
        if scalar is None:
            current_scalar = jnp.zeros_like(current_scalar_field)
        else:
            subfilter_viscosity = (
                0.0
                if momentum.subfilter is None
                else eddy_viscosity(
                    current_velocity,
                    grid,
                    boundaries,
                    momentum.subfilter,
                )
            )
            current_scalar = scalar_tendency(
                current_scalar_field,
                current_velocity,
                grid,
                scalar,
                eddy_viscosity=subfilter_viscosity,
                lower_flux=None if exchange is None else exchange.scalar_flux,
            )
        return (
            current_velocity,
            current_scalar_field,
            current_momentum,
            current_scalar,
        )

    def ab2_step(
        solution: AtmosphericSolution,
        dt: float,
        inflow: InflowPlane,
    ) -> AtmosphericSolution:
        step_size = jnp.asarray(dt, solution.velocity.x.dtype)
        (
            current_velocity,
            current_scalar_field,
            current_momentum,
            current_scalar,
        ) = tendencies(
            solution.velocity,
            solution.scalar,
            solution.time,
            inflow,
        )
        first = solution.step == 0
        current_weight = jnp.where(first, 1.0, 1.5).astype(step_size.dtype)
        previous_weight = jnp.where(first, 0.0, -0.5).astype(step_size.dtype)
        candidate = StaggeredVelocity(
            current_velocity.x
            + step_size
            * (
                current_weight * current_momentum.x
                + previous_weight * solution.momentum_tendency.x
            ),
            current_velocity.y
            + step_size
            * (
                current_weight * current_momentum.y
                + previous_weight * solution.momentum_tendency.y
            ),
            current_velocity.z
            + step_size
            * (
                current_weight * current_momentum.z
                + previous_weight * solution.momentum_tendency.z
            ),
        )
        candidate = enforce_open_velocity(candidate, inflow, grid)
        velocity, pressure = project(
            candidate, poisson, dt, solution.pressure
        )
        if scalar is None:
            next_scalar = current_scalar_field
        else:
            next_scalar = current_scalar_field + step_size * (
                current_weight * current_scalar
                + previous_weight * solution.scalar_tendency
            )
            next_scalar = enforce_open_scalar(next_scalar, inflow, grid)
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

    def fast_rk3_step(
        solution: AtmosphericSolution,
        dt: float,
        inflow: InflowPlane,
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
        lagged = pressure_gradient(pressure, grid, periodic_x=False)
        last = len(current_weights) - 1

        for stage, (current_weight, previous_weight) in enumerate(
            zip(current_weights, previous_weights)
        ):
            (
                current_velocity,
                current_scalar_field,
                current_momentum,
                current_scalar,
            ) = tendencies(
                velocity,
                scalar_field,
                execution_time,
                inflow,
            )
            current_scale = step_size * current_weight
            previous_scale = step_size * previous_weight
            candidate = StaggeredVelocity(
                current_velocity.x
                + current_scale * current_momentum.x
                + previous_scale * previous_momentum.x,
                current_velocity.y
                + current_scale * current_momentum.y
                + previous_scale * previous_momentum.y,
                current_velocity.z
                + current_scale * current_momentum.z
                + previous_scale * previous_momentum.z,
            )
            next_scalar = (
                current_scalar_field
                + current_scale * current_scalar
                + previous_scale * previous_scalar
            )
            substep = dt * (current_weight + previous_weight)
            scale = jnp.asarray(substep, velocity.x.dtype)
            candidate = enforce_open_velocity(
                StaggeredVelocity(
                    candidate.x - scale * lagged.x,
                    candidate.y - scale * lagged.y,
                    candidate.z - scale * lagged.z,
                ),
                inflow,
                grid,
            )
            if stage == last:
                velocity, correction = project(candidate, poisson, substep)
                pressure = pressure + correction * (substep / dt)
            else:
                velocity = candidate
            scalar_field = (
                current_scalar_field
                if scalar is None
                else enforce_open_scalar(next_scalar, inflow, grid)
            )
            previous_momentum = current_momentum
            previous_scalar = current_scalar
            execution_time = execution_time + scale

        next_step = solution.step + 1
        next_time = step_size * next_step.astype(step_size.dtype)
        return AtmosphericSolution(
            velocity,
            pressure,
            previous_momentum,
            scalar_field,
            previous_scalar,
            next_time,
            next_step,
        )

    return {"ab2": ab2_step, "fast-rk3": fast_rk3_step}[scheme]

def build_open_atmospheric_run(
    step: Callable[[AtmosphericSolution, float, InflowPlane], AtmosphericSolution],
) -> Callable[[AtmosphericSolution, float, InflowPlane], AtmosphericSolution]:
    """JIT a clock-matched sequence of precursor layers with one scan."""

    def run(
        solution: AtmosphericSolution,
        dt: float,
        inflows: InflowPlane,
    ) -> AtmosphericSolution:
        def advance(current, inflow):
            return step(current, dt, inflow), None

        final, _ = jax.lax.scan(advance, solution, inflows)
        return final

    return jax.jit(run)


__all__ = ["build_open_atmospheric_run", "build_open_atmospheric_step"]
