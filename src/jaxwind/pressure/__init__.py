"""Matrix-free pressure solve and full MAC projection."""

from .mac_projection import (
    _velocity_sum as _velocity_sum,
    MACProjectionResult,
    MACStageProjector,
    MACVelocity,
    SSPRK3ProjectionResult,
    VelocityPressureProjection,
    mac_divergence,
    mac_pressure_gradient,
    projected_ssprk3_step,
    projected_ssprk3_velocity_pressure_step,
    projected_ssprk3_velocity_step,
)
from .matrix_free_gmg import (
    BoundaryCondition,
    GMGConfig,
    MatrixFreeGMG,
    MatrixFreePoissonOperator,
    MatrixFreePoissonSolver,
    PoissonBoundaryConditions,
    RectilinearGrid,
)
from .pcg import PCGConfig, PCGResult, pcg
from jaxwind.domain.multilevel import MultigridHierarchy

__all__ = [
    "BoundaryCondition",
    "GMGConfig",
    "MACProjectionResult",
    "MACStageProjector",
    "MACVelocity",
    "MatrixFreeGMG",
    "MatrixFreePoissonOperator",
    "MatrixFreePoissonSolver",
    "MultigridHierarchy",
    "PCGConfig",
    "PCGResult",
    "PoissonBoundaryConditions",
    "RectilinearGrid",
    "SSPRK3ProjectionResult",
    "VelocityPressureProjection",
    "mac_divergence",
    "mac_pressure_gradient",
    "pcg",
    "projected_ssprk3_step",
    "projected_ssprk3_velocity_pressure_step",
    "projected_ssprk3_velocity_step",
]
