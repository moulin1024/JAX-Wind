"""Open-streamwise atmospheric integration driven by precursor planes."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jaxwind.domain.grid import Grid

from .abl import AtmosphericSolution
from .buoyancy import LinearBoussinesqBuoyancy, boussinesq_tendency
from jaxwind.numerics.integrate import FlowModel, build_tendency
from .open_boundary import (
    InflowPlane,
    backflow_outlet_pressure,
    enforce_open_scalar,
    enforce_open_velocity,
)
from jaxwind.numerics.discretization import pressure_gradient
from jaxwind.numerics.poisson import PressurePoisson, project
from .scalar import PassiveScalar, open_scalar_tendency, scalar_tendency
from .sgs import eddy_viscosity
from .state import OPEN, Boundaries, StaggeredVelocity
from .surface import (
    MoninObukhovSurface,
    coupled_surface_exchange,
    surface_momentum_tendency,
)


def build_open_atmospheric_step(
    grid: Grid,
    boundaries: Boundaries,
    poisson: PressurePoisson,
    momentum: FlowModel,
    scalar: PassiveScalar | None,
    buoyancy: LinearBoussinesqBuoyancy | None = None,
    surface_transfer: MoninObukhovSurface | None = None,
    *,
    scalar_source: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    scheme: str = "ab2",
    scalar_boundary: str = "cell",
    transport_scalar: bool = True,
) -> Callable[[AtmosphericSolution, float, InflowPlane], AtmosphericSolution]:
    """Build an open-boundary AB2, RK3, or single-projection fast-RK3 step.

    scalar_boundary="cell" retains legacy end-cell prescription. "flux"
    evolves every physical scalar cell and prescribes ambient incoming
    advective flux, with zero diffusive flux at the open x faces. The latter
    retains local parcel/source heat exchange in the inlet and outlet cells.
    Energy backflow uses this conservative scalar boundary automatically,
    with the supplied inflow scalar as the external reservoir value.
    It requires full RK3 and a prescribed low-x inlet. Open lateral outlets
    use perturbation energy relative to the ambient streamwise inflow; active
    scalar transport with open lateral boundaries is not yet supported.
    Outlet pressure is explicit in each stage. Set transport_scalar=False for
    externally split scalar transport; the scalar still supplies buoyancy.
    """
    if scalar_boundary not in ("cell", "flux"):
        raise ValueError("scalar_boundary must be cell or flux")
    if scheme not in ("ab2", "rk3", "fast-rk3"):
        raise ValueError(f"unsupported open atmospheric time scheme: {scheme!r}")
    if boundaries.streamwise != OPEN:
        raise ValueError("open atmospheric integration requires open boundaries")
    if poisson.periodic_x:
        raise ValueError("open atmospheric integration requires nonperiodic pressure")
    if scalar is None and (buoyancy is not None or surface_transfer is not None):
        raise ValueError(
            "buoyancy and coupled surface transfer require an active scalar"
        )
    if scalar is None and scalar_source is not None:
        raise ValueError("a scalar source requires an active scalar")
    if surface_transfer is not None and momentum.surface is not None:
        raise ValueError(
            "independent and coupled FV surface models are mutually exclusive"
        )
    if momentum.outlet_backflow not in {"none", "energy"}:
        raise ValueError("outlet_backflow must be none or energy")
    backflow = momentum.outlet_backflow == "energy"
    if backflow and scheme != "rk3":
        raise ValueError("energy outlet backflow requires rk3 with a projection at every stage")
    if backflow and poisson.open_x_low:
        raise ValueError("energy outlet backflow requires a prescribed low-x inlet")
    if backflow and poisson.open_y and scalar is not None:
        raise ValueError("energy lateral backflow currently requires an inactive scalar")
    if backflow and scalar is not None:
        scalar_boundary = "flux"
    if not transport_scalar and (scalar_boundary != "flux" or scalar_source is not None or surface_transfer is not None):
        raise ValueError("external scalar transport requires flux boundaries and no carrier scalar sources")
    momentum_rhs = build_tendency(grid, boundaries, momentum)
    sponge = None
    if momentum.outlet_sponge_start_fraction is not None:
        from .sponge import outlet_sponge_tendency
        sponge = outlet_sponge_tendency(grid, start_fraction=momentum.outlet_sponge_start_fraction,
                                       timescale=momentum.outlet_sponge_timescale_seconds)

    mode_sponge = None
    if momentum.upstream_mode_sponge_end_fraction is not None:
        from .numerics.mode_sponge import upstream_mode_sponge_tendency
        mode_sponge = upstream_mode_sponge_tendency(grid,
            end_fraction=momentum.upstream_mode_sponge_end_fraction,
            timescale=momentum.upstream_mode_sponge_timescale_seconds)

    def apply_scalar_boundary(field, inflow):
        if scalar_boundary == "flux":
            return field
        return enforce_open_scalar(field, inflow, grid)

    def tendencies(velocity, scalar_field, execution_time, inflow, buoyancy_offset):
        current_velocity = enforce_open_velocity(
            velocity, inflow, grid, open_y=poisson.open_y,
            extrapolate_normal_outflow=not backflow,
        )
        current_scalar_field = (
            scalar_field
            if scalar is None
            else apply_scalar_boundary(scalar_field, inflow)
        )
        current_momentum = momentum_rhs(current_velocity, execution_time)
        if sponge is not None:
            damping = sponge(current_velocity, inflow)
            current_momentum = StaggeredVelocity(*(a+b for a,b in zip(current_momentum,damping)))
        if mode_sponge is not None:
            damping = mode_sponge(current_velocity)
            current_momentum = StaggeredVelocity(*(a+b for a,b in zip(current_momentum,damping)))
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
                current_scalar_field + buoyancy_offset,
                buoyancy,
                x_face_count=current_velocity.x.shape[-1],
                y_face_count=current_velocity.y.shape[1],
            )
            current_momentum = StaggeredVelocity(
                current_momentum.x + source.x,
                current_momentum.y + source.y,
                current_momentum.z + source.z,
            )
        if scalar is None or not transport_scalar:
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
            scalar_rhs = (
                open_scalar_tendency if scalar_boundary == "flux" else scalar_tendency
            )
            boundary_values = (
                {"ambient": inflow.scalar} if scalar_boundary == "flux" else {}
            )
            current_scalar = scalar_rhs(
                current_scalar_field,
                current_velocity,
                grid,
                scalar,
                eddy_viscosity=subfilter_viscosity,
                lower_flux=None if exchange is None else exchange.scalar_flux,
                **boundary_values,
            )
            if scalar_source is not None:
                current_scalar = current_scalar + scalar_source(execution_time)
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
        buoyancy_offset=0.0,
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
            buoyancy_offset,
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
        candidate = enforce_open_velocity(candidate, inflow, grid, open_y=poisson.open_y)
        velocity, pressure = project(
            candidate, poisson, dt, solution.pressure
        )
        if scalar is None or not transport_scalar:
            next_scalar = current_scalar_field
        else:
            next_scalar = current_scalar_field + step_size * (
                current_weight * current_scalar
                + previous_weight * solution.scalar_tendency
            )
            next_scalar = apply_scalar_boundary(next_scalar, inflow)
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
        inflow: InflowPlane,
        buoyancy_offset=0.0,
    ) -> AtmosphericSolution:
        velocity, scalar_field = solution.velocity, solution.scalar
        execution_time = solution.time
        previous_momentum, previous_scalar = solution.momentum_tendency, solution.scalar_tendency
        step_size = jnp.asarray(dt, velocity.x.dtype)
        pressure = solution.pressure
        previous_boundary_pressure = jnp.zeros((grid.nz, grid.ny), velocity.x.dtype)
        for current_weight, previous_weight in zip(
            (8. / 15., 5. / 12., 3. / 4.), (0., -17. / 60., -5. / 12.)
        ):
            current_velocity, current_scalar_field, current_momentum, current_scalar = tendencies(
                velocity, scalar_field, execution_time, inflow, buoyancy_offset,
            )
            candidate = StaggeredVelocity(*(
                field + step_size * (current_weight * current + previous_weight * previous)
                for field, current, previous in zip(current_velocity, current_momentum, previous_momentum)
            ))
            candidate = enforce_open_velocity(candidate, inflow, grid, open_y=poisson.open_y)
            substep = step_size * (current_weight + previous_weight)
            # Evaluate backflow from the incoming stage's projected normal flux.
            # Predictor extrapolation must not erase a pressure-driven reversal.
            boundary_pressure = None
            if backflow:
                current_boundary_pressure = backflow_outlet_pressure(velocity, grid)
                # The known boundary-gradient forcing uses the same Wray
                # weights as every other explicit RHS term. project() scales
                # it by substep, hence division by the sum of the weights.
                boundary_pressure = (
                    current_weight * current_boundary_pressure
                    + previous_weight * previous_boundary_pressure
                ) / (current_weight + previous_weight)
                previous_boundary_pressure = current_boundary_pressure
            velocity, pressure = project(candidate, poisson, substep,
                                         outlet_pressure=boundary_pressure)
            if scalar is not None and transport_scalar:
                next_scalar = current_scalar_field + step_size * (
                    current_weight * current_scalar + previous_weight * previous_scalar
                )
                scalar_field = apply_scalar_boundary(next_scalar, inflow)
            previous_momentum, previous_scalar = current_momentum, current_scalar
            execution_time = execution_time + substep
        return AtmosphericSolution(
            velocity, pressure, previous_momentum, scalar_field, previous_scalar,
            solution.time + step_size, solution.step + 1,
        )

    def fast_rk3_step(
        solution: AtmosphericSolution,
        dt: float,
        inflow: InflowPlane,
        buoyancy_offset=0.0,
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
        lagged = pressure_gradient(pressure, grid, periodic_x=False, periodic_y=poisson.periodic_y, open_y=poisson.open_y)
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
                buoyancy_offset,
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
                open_y=poisson.open_y,
            )
            if stage == last:
                velocity, correction = project(candidate, poisson, substep)
                pressure = pressure + correction * (substep / dt)
            else:
                velocity = candidate
            scalar_field = (
                current_scalar_field
                if scalar is None
                else apply_scalar_boundary(next_scalar, inflow)
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

    return {"ab2": ab2_step, "rk3": rk3_step, "fast-rk3": fast_rk3_step}[scheme]

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
