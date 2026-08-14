"""Pure configuration and persistent-memory products for dynamic LASD."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class LagrangianScaleDependentDynamic:
    """Momentum LASD configuration; coefficients store squared model lengths."""

    filter_grid_ratio: float = 1.5
    test_filter_ratio: float = 2.0
    update_interval: int = 10
    timescale_coefficient: float = 1.5
    initial_coefficient: float = 0.03
    minimum_coefficient: float = 1.0e-6
    maximum_coefficient: float = 0.81
    scale_dependent: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.filter_grid_ratio,
            self.test_filter_ratio,
            self.timescale_coefficient,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("LASD filter ratios and time scale must be positive")
        if self.test_filter_ratio <= 1.0:
            raise ValueError("LASD test-filter ratio must exceed one")
        if self.update_interval <= 0:
            raise ValueError("LASD update interval must be positive")
        bounds = (
            self.initial_coefficient,
            self.minimum_coefficient,
            self.maximum_coefficient,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in bounds):
            raise ValueError("LASD coefficient values must be finite and nonnegative")
        if self.maximum_coefficient < self.minimum_coefficient:
            raise ValueError("LASD coefficient bounds are reversed")
        if (
            not self.minimum_coefficient
            <= self.initial_coefficient
            <= self.maximum_coefficient
        ):
            raise ValueError("LASD initial coefficient must lie inside its bounds")

    @property
    def fingerprint(self) -> str:
        return (
            "jaxwind.lasd.momentum.v1"
            f"|fgr={self.filter_grid_ratio.hex()}"
            f"|tfr={self.test_filter_ratio.hex()}"
            f"|interval={self.update_interval}"
            f"|timescale={self.timescale_coefficient.hex()}"
            f"|initial={self.initial_coefficient.hex()}"
            f"|min={self.minimum_coefficient.hex()}"
            f"|max={self.maximum_coefficient.hex()}"
            f"|scale-dependent={int(self.scale_dependent)}"
        )


@dataclass(frozen=True, slots=True)
class LagrangianScaleDependentScalarFlux:
    """Independent dynamic coefficient for one conservative scalar flux."""

    initial_coefficient: float = 0.03
    minimum_coefficient: float = 0.0
    maximum_coefficient: float = 1.0
    scale_dependent: bool = True
    stability_buoyancy_coefficient: float = 0.0
    stability_beta: float = 30.0
    stability_power: float = 2.0
    dynamic_updates_enabled: bool = True

    def __post_init__(self) -> None:
        values = (
            self.initial_coefficient,
            self.minimum_coefficient,
            self.maximum_coefficient,
            self.stability_buoyancy_coefficient,
            self.stability_beta,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("scalar LASD coefficients must be finite and nonnegative")
        if self.maximum_coefficient < self.minimum_coefficient:
            raise ValueError("scalar LASD coefficient bounds are reversed")
        if (
            not self.minimum_coefficient
            <= self.initial_coefficient
            <= self.maximum_coefficient
        ):
            raise ValueError(
                "scalar LASD initial coefficient must lie inside its bounds"
            )
        if not math.isfinite(self.stability_power) or self.stability_power <= 0.0:
            raise ValueError("scalar stability power must be finite and positive")
        if not isinstance(self.dynamic_updates_enabled, bool):
            raise TypeError("scalar LASD update flag must be boolean")

    @property
    def fingerprint(self) -> str:
        fingerprint = (
            "jaxwind.lasd.scalar.v1"
            f"|initial={self.initial_coefficient.hex()}"
            f"|min={self.minimum_coefficient.hex()}"
            f"|max={self.maximum_coefficient.hex()}"
            f"|scale-dependent={int(self.scale_dependent)}"
            f"|stability-buoyancy={self.stability_buoyancy_coefficient.hex()}"
            f"|stability-beta={self.stability_beta.hex()}"
            f"|stability-power={self.stability_power.hex()}"
        )
        if not self.dynamic_updates_enabled:
            fingerprint += "|dynamic-updates=0"
        return fingerprint


@dataclass(frozen=True, slots=True)
class DiagnosticLasdConstants:
    """Constants used only to diagnose unresolved energy and scalar variance."""

    sgs_dissipation_coefficient: float = 0.93
    scalar_variance_coefficient: float = 2.02
    horizontal_homogeneous_wall: bool = False

    def __post_init__(self) -> None:
        values = (
            self.sgs_dissipation_coefficient,
            self.scalar_variance_coefficient,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("diagnostic LASD constants must be finite and positive")
        if not isinstance(self.horizontal_homogeneous_wall, bool):
            raise TypeError("horizontal_homogeneous_wall must be boolean")

    @property
    def fingerprint(self) -> str:
        return (
            "jaxwind.lasd.diagnostics.v3"
            f"|ce={self.sgs_dissipation_coefficient.hex()}"
            f"|cc={self.scalar_variance_coefficient.hex()}"
            "|production-wall=neutral-log-if-configured"
            "|scalar-wall-gradient=prescribed-flux"
            "|exports=local-and-predivision"
            f"|homogeneous-wall={int(self.horizontal_homogeneous_wall)}"
        )


@dataclass(frozen=True, slots=True)
class LasdDiagnosticFields:
    """Array-interpreted diagnostic fields, not prognostic closure state."""

    momentum_diffusivity: Any
    scalar_diffusivity: Any
    scalar_flux_x: Any
    scalar_flux_y: Any
    scalar_flux_z: Any
    sgs_tke: Any
    scalar_variance_numerator: Any
    scalar_variance: Any


@dataclass(frozen=True, slots=True)
class NoClosureMemory:
    """Explicit identity memory for a memoryless closure."""


@dataclass(frozen=True, slots=True)
class MomentumLasdMemory:
    coefficient: Any
    lm: Any
    mm: Any
    qn: Any
    nn: Any
    trajectory_x: Any
    trajectory_y: Any
    trajectory_z: Any

    def fields(self) -> tuple[Any, ...]:
        return (
            self.coefficient,
            self.lm,
            self.mm,
            self.qn,
            self.nn,
            self.trajectory_x,
            self.trajectory_y,
            self.trajectory_z,
        )


@dataclass(frozen=True, slots=True)
class ScalarLasdMemory:
    coefficient: Any
    lm: Any
    mm: Any
    qn: Any
    nn: Any

    def fields(self) -> tuple[Any, ...]:
        return (self.coefficient, self.lm, self.mm, self.qn, self.nn)


@dataclass(frozen=True, slots=True)
class LasdClosureMemory:
    momentum: MomentumLasdMemory
    scalar: ScalarLasdMemory
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        if not self.configuration_fingerprint:
            raise ValueError("LASD memory fingerprint must be non-empty")

    def fields(self) -> tuple[Any, ...]:
        return self.momentum.fields() + self.scalar.fields()


@dataclass(frozen=True, slots=True)
class LasdClosureEventDiagnostic:
    updated: Any
    accepted_step: int
    update_interval: int


@dataclass(frozen=True, slots=True)
class IdentityClosureEventDiagnostic:
    updated: bool = False


@dataclass(frozen=True, slots=True)
class IdentityClosureEvent:
    def __call__(self, fields: Any, clock: Any, environment: Any) -> tuple[Any, Any]:
        del clock, environment
        return fields, IdentityClosureEventDiagnostic()


@dataclass(frozen=True, slots=True)
class LasdAcceptedStepEvent:
    """Delegate one accepted-boundary LASD transition to the solver algebra."""

    algebra: Any
    model: Any
    dt: float
    reuse_rhs_momentum_context: bool = False
    fused_update_evaluation: Any = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("LASD event dt must be finite and positive")
        if not isinstance(self.reuse_rhs_momentum_context, bool):
            raise TypeError("LASD momentum-context reuse flag must be boolean")
        if self.fused_update_evaluation is not None and not callable(
            self.fused_update_evaluation
        ):
            raise TypeError("fused LASD update evaluation must be callable")

    def __call__(self, fields: Any, clock: Any, environment: Any) -> Any:
        del environment
        if self.reuse_rhs_momentum_context:
            should_update = (
                clock.step + 1
            ) % self.model.momentum.sgs.update_interval == 0
            if (
                self.fused_update_evaluation is not None
                and isinstance(should_update, bool)
                and should_update
            ):
                from jaxwind.domain import EvaluationTime
                from jaxwind.integrators.ab2 import (
                    PreparedVectorEvaluation,
                    VectorFieldResult,
                )
                from jaxwind.physics.boussinesq import BoussinesqDiagnostic

                first_update = (
                    clock.step == self.model.momentum.sgs.update_interval - 1
                )
                prepared, tendency = self.fused_update_evaluation(
                    fields,
                    clock.time,
                    first_update,
                )
                diagnostic = LasdClosureEventDiagnostic(
                    True,
                    clock.step,
                    self.model.momentum.sgs.update_interval,
                )
                evaluation_time = EvaluationTime(
                    clock.time,
                    clock.step,
                    "ab2-current",
                )
                return PreparedVectorEvaluation(
                    prepared,
                    VectorFieldResult(
                        tendency,
                        BoussinesqDiagnostic(evaluation_time),
                    ),
                    diagnostic,
                )

            from jaxwind.integrators.ab2 import PreparedEvaluation

            prepared, diagnostic, context = (
                self.algebra.prepare_lasd_closure_reusing_update_context(
                    fields,
                    self.model,
                    clock,
                    self.dt,
                )
            )
            return PreparedEvaluation(prepared, context, diagnostic)
        return self.algebra.prepare_lasd_closure(
            fields,
            self.model,
            clock,
            self.dt,
        )
