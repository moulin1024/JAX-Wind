"""The first pure dry-flow vector field and its static physical choices.

This semantic module contains no JAX code.  Array layout, halo exchange, and
operator lowering belong to the algebra supplied to ``DryFlowVectorField``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

from .lasd import LagrangianScaleDependentDynamic


@dataclass(frozen=True, slots=True)
class ConservativeAdvection:
    """Conservative flux form with horizontal three-halves dealiasing."""


@dataclass(frozen=True, slots=True)
class RotationalAdvection:
    """Rotational ``omega x u`` advection.

    The fused Boussinesq lowering evaluates its products with three-halves
    horizontal padding.
    """


@dataclass(frozen=True, slots=True)
class KinematicPressureGradient:
    """Constant pressure-gradient acceleration in canonical SI units."""

    x_acceleration: float
    y_acceleration: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.x_acceleration) or not math.isfinite(
            self.y_acceleration
        ):
            raise ValueError("pressure-gradient acceleration must be finite")


@dataclass(frozen=True, slots=True)
class NeutralLogWall:
    """Point-local neutral logarithmic lower-wall traction."""

    roughness_length: float
    von_karman: float = 0.4

    def __post_init__(self) -> None:
        if not math.isfinite(self.roughness_length) or self.roughness_length <= 0.0:
            raise ValueError("roughness length must be finite and positive")
        if not math.isfinite(self.von_karman) or self.von_karman <= 0.0:
            raise ValueError("von Karman constant must be finite and positive")


@dataclass(frozen=True, slots=True)
class FilteredNeutralLogWall:
    """Local neutral log wall evaluated from a 2-D filtered velocity.

    The filter matches the legacy JAX-Wind wall path: the first-level
    horizontal velocity is sharply filtered at the combined grid/test-filter
    scale before the local speed and stress direction are evaluated.
    """

    roughness_length: float
    von_karman: float = 0.4
    filter_grid_ratio: float = 1.5
    test_filter_ratio: float = 2.0
    porte_agel_correction: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.roughness_length) or self.roughness_length <= 0.0:
            raise ValueError("roughness length must be finite and positive")
        if not math.isfinite(self.von_karman) or self.von_karman <= 0.0:
            raise ValueError("von Karman constant must be finite and positive")
        ratios = (self.filter_grid_ratio, self.test_filter_ratio)
        if not all(math.isfinite(value) and value > 0.0 for value in ratios):
            raise ValueError("wall filter ratios must be finite and positive")
        if not isinstance(self.porte_agel_correction, bool):
            raise TypeError("Porté-Agel wall correction flag must be boolean")


@dataclass(frozen=True, slots=True)
class StaticSmagorinsky:
    """Memoryless static Smagorinsky momentum closure."""

    coefficient: float = 0.16

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient) or self.coefficient < 0.0:
            raise ValueError("Smagorinsky coefficient must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class NoRotation:
    """Explicit additive identity for non-rotating dry flow."""


@dataclass(frozen=True, slots=True)
class CoriolisGeostrophic:
    """Constant Earth-rotation components about a geostrophic wind."""

    coriolis_parameter: float
    geostrophic_x_velocity: float
    geostrophic_y_velocity: float = 0.0
    horizontal_coriolis_parameter: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.coriolis_parameter,
            self.geostrophic_x_velocity,
            self.geostrophic_y_velocity,
            self.horizontal_coriolis_parameter,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Coriolis--geostrophic parameters must be finite")
        if self.coriolis_parameter == 0.0:
            raise ValueError("use NoRotation instead of a zero Coriolis parameter")


@dataclass(frozen=True, slots=True)
class DryFlowModel:
    """Small product of independent choices required by the first model."""

    advection: ConservativeAdvection | RotationalAdvection
    pressure_gradient: KinematicPressureGradient
    wall: NeutralLogWall | FilteredNeutralLogWall
    sgs: StaticSmagorinsky | LagrangianScaleDependentDynamic
    rotation: NoRotation | CoriolisGeostrophic = NoRotation()

    def __post_init__(self) -> None:
        expected = (
            (
                self.advection,
                (ConservativeAdvection, RotationalAdvection),
                "advection",
            ),
            (self.pressure_gradient, KinematicPressureGradient, "pressure gradient"),
            (
                self.wall,
                (NeutralLogWall, FilteredNeutralLogWall),
                "wall",
            ),
            (
                self.sgs,
                (
                    StaticSmagorinsky,
                    LagrangianScaleDependentDynamic,
                ),
                "SGS",
            ),
            (self.rotation, (NoRotation, CoriolisGeostrophic), "rotation"),
        )
        for value, choice_type, name in expected:
            if not isinstance(value, choice_type):
                raise TypeError(f"dry-flow {name} has an unsupported choice")


@dataclass(frozen=True, slots=True)
class DryFlowContributions:
    """Five independently inspectable evaluated tendencies."""

    advection: Any
    pressure_gradient: Any
    wall: Any
    sgs: Any
    coriolis_geostrophic: Any

    def values(self) -> tuple[Any, Any, Any, Any, Any]:
        return (
            self.advection,
            self.pressure_gradient,
            self.wall,
            self.sgs,
            self.coriolis_geostrophic,
        )


@dataclass(frozen=True, slots=True)
class DryFlowDiagnostic:
    """Cheap proof that one shared context fed the named physical terms."""

    evaluation_time: Any
    terms: tuple[str, ...] = (
        "advection",
        "pressure_gradient",
        "wall",
        "sgs",
        "coriolis_geostrophic",
    )
    shared_context_builds: int = 1


@dataclass(frozen=True, slots=True)
class DryFlowVectorFieldResult:
    tendency: Any
    diagnostic: DryFlowDiagnostic


class DryFlowAlgebra(Protocol):
    def dry_flow_context(self, velocity: Any) -> Any: ...

    def advection_tendency(
        self, context: Any, config: Any, wall: Any | None = None
    ) -> Any: ...

    def pressure_gradient_tendency(self, context: Any, config: Any) -> Any: ...

    def wall_stress_tendency(self, context: Any, config: Any) -> Any: ...

    def sgs_tendency(
        self, context: Any, config: Any, wall: Any | None = None
    ) -> Any: ...

    def coriolis_geostrophic_tendency(self, context: Any, config: Any) -> Any: ...

    def combine_tendencies(self, tendencies: tuple[Any, ...]) -> Any: ...


@dataclass(frozen=True, slots=True)
class DryFlowVectorField:
    """Interpret five pure contributions over one shared differential context."""

    algebra: DryFlowAlgebra
    model: DryFlowModel

    def evaluate_contributions(self, evaluation: Any) -> DryFlowContributions:
        context = self.algebra.dry_flow_context(evaluation.velocity)
        return DryFlowContributions(
            self.algebra.advection_tendency(
                context,
                self.model.advection,
                self.model.wall,
            ),
            self.algebra.pressure_gradient_tendency(
                context,
                self.model.pressure_gradient,
            ),
            self.algebra.wall_stress_tendency(context, self.model.wall),
            self.algebra.sgs_tendency(
                context,
                self.model.sgs,
                self.model.wall,
            ),
            self.algebra.coriolis_geostrophic_tendency(
                context,
                self.model.rotation,
            ),
        )

    def __call__(self, evaluation: Any) -> DryFlowVectorFieldResult:
        contributions = self.evaluate_contributions(evaluation)
        tendency = self.algebra.combine_tendencies(contributions.values())
        return DryFlowVectorFieldResult(
            tendency,
            DryFlowDiagnostic(evaluation.time),
        )
