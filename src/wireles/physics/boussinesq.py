"""Pure coupled velocity--potential-temperature Boussinesq program."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Generic, Protocol, TypeVar

from .dry_flow import DryFlowModel
from .lasd import (
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    NoClosureMemory,
)


V = TypeVar("V")
S = TypeVar("S")


@dataclass(frozen=True, slots=True)
class BoussinesqFields(Generic[V, S]):
    velocity: V
    potential_temperature: S
    closure: Any = NoClosureMemory()


@dataclass(frozen=True, slots=True)
class BoussinesqTendency(Generic[V, S]):
    velocity: V
    potential_temperature: S


@dataclass(frozen=True, slots=True)
class ConservativeScalarAdvection:
    """Conservative flux-form transport with horizontal product truncation."""


@dataclass(frozen=True, slots=True)
class StaticSmagorinskyScalarFlux:
    turbulent_prandtl: float = 0.4

    def __post_init__(self) -> None:
        if not math.isfinite(self.turbulent_prandtl) or self.turbulent_prandtl <= 0.0:
            raise ValueError("turbulent Prandtl number must be finite and positive")


@dataclass(frozen=True, slots=True)
class ScalarFluxBoundary:
    """Prescribed positive-z scalar fluxes at lower and upper physical faces."""

    lower_flux: float = 0.0
    upper_flux: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower_flux) or not math.isfinite(self.upper_flux):
            raise ValueError("scalar boundary fluxes must be finite")


@dataclass(frozen=True, slots=True)
class LinearBoussinesqBuoyancy:
    """Acceleration per stored potential-temperature perturbation unit."""

    acceleration_per_temperature: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.acceleration_per_temperature):
            raise ValueError("buoyancy coefficient must be finite")


@dataclass(frozen=True, slots=True)
class NoBuoyancy:
    """Explicit identity for a transported passive scalar."""


@dataclass(frozen=True, slots=True)
class NoRayleighDamping:
    """Identity choice for domains that do not require a momentum sponge."""


@dataclass(frozen=True, slots=True)
class RayleighGeostrophicDamping:
    """Quadratic top-layer relaxation toward geostrophic horizontal flow."""

    start_height: float
    maximum_rate: float
    geostrophic_x_velocity: float
    geostrophic_y_velocity: float

    def __post_init__(self) -> None:
        values = (
            self.start_height,
            self.maximum_rate,
            self.geostrophic_x_velocity,
            self.geostrophic_y_velocity,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Rayleigh damping parameters must be finite")
        if self.start_height < 0.0:
            raise ValueError("Rayleigh damping start height must be nonnegative")
        if self.maximum_rate <= 0.0:
            raise ValueError("Rayleigh damping maximum rate must be positive")


@dataclass(frozen=True, slots=True)
class BoussinesqModel:
    momentum: DryFlowModel
    scalar_advection: ConservativeScalarAdvection
    scalar_sgs: StaticSmagorinskyScalarFlux | LagrangianScaleDependentScalarFlux
    buoyancy: LinearBoussinesqBuoyancy | NoBuoyancy
    rayleigh_damping: NoRayleighDamping | RayleighGeostrophicDamping = (
        NoRayleighDamping()
    )
    scalar_boundary: ScalarFluxBoundary = ScalarFluxBoundary()

    def __post_init__(self) -> None:
        expected = (
            (self.momentum, DryFlowModel, "momentum"),
            (self.scalar_advection, ConservativeScalarAdvection, "scalar advection"),
            (
                self.scalar_sgs,
                (StaticSmagorinskyScalarFlux, LagrangianScaleDependentScalarFlux),
                "scalar SGS",
            ),
            (self.buoyancy, (LinearBoussinesqBuoyancy, NoBuoyancy), "buoyancy"),
        )
        for value, choice, name in expected:
            if not isinstance(value, choice):
                raise TypeError(f"Boussinesq {name} has an unsupported choice")
        if not isinstance(
            self.rayleigh_damping,
            (NoRayleighDamping, RayleighGeostrophicDamping),
        ):
            raise TypeError("Boussinesq Rayleigh damping has an unsupported choice")
        if not isinstance(self.scalar_boundary, ScalarFluxBoundary):
            raise TypeError("Boussinesq scalar boundary has an unsupported choice")
        momentum_lasd = isinstance(self.momentum.sgs, LagrangianScaleDependentDynamic)
        scalar_lasd = isinstance(self.scalar_sgs, LagrangianScaleDependentScalarFlux)
        if momentum_lasd != scalar_lasd:
            raise TypeError("momentum and scalar LASD must be selected together")


@dataclass(frozen=True, slots=True)
class BoussinesqContributions:
    advection: Any
    pressure_gradient: Any
    wall: Any
    momentum_sgs: Any
    coriolis_geostrophic: Any
    buoyancy: Any
    rayleigh_damping: Any
    scalar_advection: Any
    scalar_sgs: Any

    def momentum_values(self) -> tuple[Any, ...]:
        return (
            self.advection,
            self.pressure_gradient,
            self.wall,
            self.momentum_sgs,
            self.coriolis_geostrophic,
            self.buoyancy,
            self.rayleigh_damping,
        )

    def scalar_values(self) -> tuple[Any, Any]:
        return self.scalar_advection, self.scalar_sgs


@dataclass(frozen=True, slots=True)
class BoussinesqDiagnostic:
    evaluation_time: Any
    terms: tuple[str, ...] = (
        "advection",
        "pressure_gradient",
        "wall",
        "momentum_sgs",
        "coriolis_geostrophic",
        "buoyancy",
        "rayleigh_damping",
        "scalar_advection",
        "scalar_sgs",
    )
    shared_velocity_context_builds: int = 1


@dataclass(frozen=True, slots=True)
class BoussinesqVectorFieldResult:
    tendency: BoussinesqTendency
    diagnostic: BoussinesqDiagnostic


class BoussinesqAlgebra(Protocol):
    def boussinesq_context(self, fields: BoussinesqFields) -> Any: ...

    def momentum_context(self, context: Any) -> Any: ...

    def advection_tendency(self, context: Any, config: Any) -> Any: ...

    def pressure_gradient_tendency(self, context: Any, config: Any) -> Any: ...

    def wall_stress_tendency(self, context: Any, config: Any) -> Any: ...

    def sgs_tendency(self, context: Any, config: Any) -> Any: ...

    def coriolis_geostrophic_tendency(self, context: Any, config: Any) -> Any: ...

    def buoyancy_tendency(self, context: Any, config: Any) -> Any: ...

    def rayleigh_damping_tendency(self, context: Any, config: Any) -> Any: ...

    def scalar_advection_tendency(self, context: Any, config: Any) -> Any: ...

    def scalar_sgs_tendency(
        self, context: Any, momentum_sgs: Any, config: Any, boundary: Any
    ) -> Any: ...

    def combine_tendencies(self, tendencies: tuple[Any, ...]) -> Any: ...

    def combine_scalar_tendencies(self, tendencies: tuple[Any, ...]) -> Any: ...


@dataclass(frozen=True, slots=True)
class BoussinesqVectorField:
    algebra: BoussinesqAlgebra
    model: BoussinesqModel

    def evaluate_contributions(self, evaluation: Any) -> BoussinesqContributions:
        context = self.algebra.boussinesq_context(evaluation.velocity)
        momentum = self.algebra.momentum_context(context)
        model = self.model
        return BoussinesqContributions(
            self.algebra.advection_tendency(momentum, model.momentum.advection),
            self.algebra.pressure_gradient_tendency(
                momentum, model.momentum.pressure_gradient
            ),
            self.algebra.wall_stress_tendency(momentum, model.momentum.wall),
            self.algebra.sgs_tendency(momentum, model.momentum.sgs),
            self.algebra.coriolis_geostrophic_tendency(
                momentum, model.momentum.rotation
            ),
            self.algebra.buoyancy_tendency(context, model.buoyancy),
            self.algebra.rayleigh_damping_tendency(context, model.rayleigh_damping),
            self.algebra.scalar_advection_tendency(context, model.scalar_advection),
            self.algebra.scalar_sgs_tendency(
                context,
                model.momentum.sgs,
                model.scalar_sgs,
                model.scalar_boundary,
            ),
        )

    def __call__(self, evaluation: Any) -> BoussinesqVectorFieldResult:
        contributions = self.evaluate_contributions(evaluation)
        return BoussinesqVectorFieldResult(
            BoussinesqTendency(
                self.algebra.combine_tendencies(contributions.momentum_values()),
                self.algebra.combine_scalar_tendencies(contributions.scalar_values()),
            ),
            BoussinesqDiagnostic(evaluation.time),
        )
