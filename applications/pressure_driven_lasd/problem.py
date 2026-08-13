"""Shared construction of the pressure-driven LASD numerical problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import CaseConfig


@dataclass(frozen=True, slots=True)
class PressureDrivenProblem:
    physical_grid: Any
    scales: Any
    momentum_sgs: Any
    scalar_sgs: Any
    physics_fingerprint: str
    integrator: Any
    solver: Any

    @property
    def closure_fingerprint(self) -> str:
        return self.momentum_sgs.fingerprint + "|" + self.scalar_sgs.fingerprint


def build_pressure_driven_problem(
    case: CaseConfig,
    *,
    runtime: Any,
    wind_tunnel_model: Any = None,
) -> PressureDrivenProblem:
    """Build the reusable pressure-driven model, scales, and JAX solver."""

    from jaxwind import build_jax_solver
    from jaxwind.domain import ScaleSystem, UniformGrid, VerticalBoundary
    from jaxwind.integrators import AB2Config
    from jaxwind.physics import (
        BoussinesqModel,
        ConservativeAdvection,
        ConservativeScalarAdvection,
        DryFlowModel,
        FilteredNeutralLogWall,
        KinematicPressureGradient,
        LagrangianScaleDependentDynamic,
        LagrangianScaleDependentScalarFlux,
        NoBuoyancy,
        NoRayleighDamping,
        NoRotation,
        ScalarFluxBoundary,
    )

    physical_grid = UniformGrid(
        case.domain.nx,
        case.domain.ny,
        case.domain.nz,
        case.domain.lx_m,
        case.domain.ly_m,
        case.domain.lz_m,
    )
    scales = ScaleSystem(
        case.flow.forcing_height_m,
        case.flow.friction_velocity_m_s,
    )
    grid = scales.to_execution_grid(physical_grid)
    momentum_sgs = LagrangianScaleDependentDynamic(
        filter_grid_ratio=case.sgs.filter_grid_ratio,
        test_filter_ratio=case.sgs.test_filter_ratio,
        update_interval=case.sgs.update_interval_steps,
        timescale_coefficient=case.sgs.timescale_coefficient,
        initial_coefficient=case.sgs.initial_coefficient,
        minimum_coefficient=case.sgs.minimum_coefficient,
        maximum_coefficient=case.sgs.maximum_coefficient,
    )
    scalar_sgs = LagrangianScaleDependentScalarFlux()
    model = BoussinesqModel(
        DryFlowModel(
            ConservativeAdvection(),
            KinematicPressureGradient(
                scales.to_execution_acceleration(
                    case.flow.pressure_acceleration_m_s2
                )
            ),
            FilteredNeutralLogWall(
                scales.to_execution_length(case.flow.roughness_length_m),
                von_karman=case.flow.von_karman,
                filter_grid_ratio=case.wall.filter_grid_ratio,
                test_filter_ratio=case.wall.test_filter_ratio,
                porte_agel_correction=case.wall.porte_agel_correction,
            ),
            momentum_sgs,
            NoRotation(),
        ),
        ConservativeScalarAdvection(),
        scalar_sgs,
        NoBuoyancy(),
        NoRayleighDamping(),
        ScalarFluxBoundary(),
    )
    physics_fingerprint = (
        momentum_sgs.fingerprint
        + "|advection=conservative"
        + "|dealiasing=three-halves-padding"
        + "|coefficient-padding=bounded"
    )
    integrator = AB2Config(scales.to_execution_time(case.time.dt_seconds))

    def boundary(_clock, _environment):
        return VerticalBoundary(0.0, 0.0)

    solver = build_jax_solver(
        grid,
        runtime=runtime,
        model=model,
        integrator=integrator,
        normal_boundary=boundary,
        pressure_dtype=case.numerics.dtype,
        pressure_method=case.numerics.pressure_method,
        nonlinear_padding_ratio=1.5,
        optimize_frozen_zero_scalar=True,
        wind_tunnel_model=wind_tunnel_model,
    )
    return PressureDrivenProblem(
        physical_grid,
        scales,
        momentum_sgs,
        scalar_sgs,
        physics_fingerprint,
        integrator,
        solver,
    )


__all__ = ["PressureDrivenProblem", "build_pressure_driven_problem"]
