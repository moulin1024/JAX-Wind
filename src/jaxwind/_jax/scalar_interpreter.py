"""Scalar and scalar-diagnostic methods for the z-slab discretization."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from jaxwind.domain import (
    AddressableField,
    Cell,
    Evaluated,
    PassiveScalarTendency,
    PotentialTemperaturePerturbation,
    PotentialTemperatureTendency,
)
from jaxwind.operators import VelocityVector
from jaxwind.physics.boussinesq import (
    ConservativeScalarAdvection,
    LinearBoussinesqBuoyancy,
    NoBuoyancy,
    NoRayleighDamping,
    RayleighGeostrophicDamping,
    ScalarFluxBoundary,
    StaticSmagorinskyScalarFlux,
)
from jaxwind.physics.dry_flow import (
    FilteredNeutralLogWall,
    NeutralLogWall,
    StaticSmagorinsky,
)
from jaxwind.physics.lasd import (
    DiagnosticLasdConstants,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LasdClosureMemory,
    LasdDiagnosticFields,
)

if TYPE_CHECKING:
    from .discretization import ZSlabBoussinesqContext


class ZSlabScalarMixin:
    """Distributed scalar transport and diagnostics."""

    __slots__ = ()

    def _scalar_tendency(
        self,
        context: ZSlabBoussinesqContext,
        payload: Any,
    ) -> AddressableField:
        quantity = (
            PotentialTemperatureTendency
            if context.potential_temperature.quantity
            is PotentialTemperaturePerturbation
            else PassiveScalarTendency
        )
        return AddressableField(
            quantity,
            Cell,
            self._expected_regions(Cell),
            Evaluated,
            payload.astype(context.potential_temperature.payload.dtype),
        )

    def buoyancy_tendency(
        self,
        context: ZSlabBoussinesqContext,
        config: LinearBoussinesqBuoyancy | NoBuoyancy,
    ) -> VelocityVector:
        velocity = context.momentum.velocity
        if isinstance(config, NoBuoyancy):
            return self._dry_tendency(
                jnp.zeros_like(velocity.x.payload),
                jnp.zeros_like(velocity.y.payload),
                jnp.zeros_like(velocity.z.owned.payload),
            )
        if not isinstance(config, LinearBoussinesqBuoyancy):
            raise TypeError("unsupported Boussinesq buoyancy choice")
        z = self.scalar.buoyancy(
            context.arrays,
            config.acceleration_per_temperature,
        )
        return self._dry_tendency(
            jnp.zeros_like(velocity.x.payload),
            jnp.zeros_like(velocity.y.payload),
            z,
        )

    def rayleigh_damping_tendency(
        self,
        context: ZSlabBoussinesqContext,
        config: NoRayleighDamping | RayleighGeostrophicDamping,
    ) -> VelocityVector:
        velocity = context.momentum.velocity
        if isinstance(config, NoRayleighDamping):
            return self._dry_tendency(
                jnp.zeros_like(velocity.x.payload),
                jnp.zeros_like(velocity.y.payload),
                jnp.zeros_like(velocity.z.owned.payload),
            )
        if not isinstance(config, RayleighGeostrophicDamping):
            raise TypeError("unsupported Rayleigh damping choice")
        if config.start_height >= self.decomposition.grid.lz:
            raise ValueError("Rayleigh damping must start below the domain top")
        x, y, z = self.scalar.rayleigh_damping(
            velocity.x.payload,
            velocity.y.payload,
            velocity.z.owned.payload,
            config.start_height,
            config.maximum_rate,
            config.geostrophic_x_velocity,
            config.geostrophic_y_velocity,
        )
        return self._dry_tendency(x, y, z)

    def scalar_advection_tendency(
        self,
        context: ZSlabBoussinesqContext,
        config: ConservativeScalarAdvection,
    ) -> AddressableField:
        if not isinstance(config, ConservativeScalarAdvection):
            raise TypeError("unsupported conservative scalar advection choice")
        return self._scalar_tendency(
            context,
            self.scalar.advection(context.arrays, context.momentum.arrays),
        )

    def scalar_sgs_tendency(
        self,
        context: ZSlabBoussinesqContext,
        momentum_config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
        config: StaticSmagorinskyScalarFlux | LagrangianScaleDependentScalarFlux,
        boundary: ScalarFluxBoundary = ScalarFluxBoundary(),
    ) -> AddressableField:
        static = isinstance(momentum_config, StaticSmagorinsky) and isinstance(
            config, StaticSmagorinskyScalarFlux
        )
        dynamic = isinstance(
            momentum_config,
            LagrangianScaleDependentDynamic,
        ) and isinstance(config, LagrangianScaleDependentScalarFlux)
        if not (static or dynamic):
            raise TypeError("unsupported or inconsistent scalar SGS choice")
        if static:
            coefficient = jnp.full_like(
                context.arrays.theta,
                momentum_config.coefficient**2 / config.turbulent_prandtl,
            )
            coefficient_bounds = (0.0, math.inf)
        else:
            closure = context.momentum.closure
            if not isinstance(closure, LasdClosureMemory):
                raise TypeError("scalar LASD requires initialized closure memory")
            coefficient = closure.scalar.coefficient.payload
            coefficient_bounds = (
                config.minimum_coefficient,
                config.maximum_coefficient,
            )
        return self._scalar_tendency(
            context,
            self.scalar.sgs(
                context.arrays,
                context.momentum.arrays,
                coefficient,
                *coefficient_bounds,
                boundary.lower_flux,
                boundary.upper_flux,
                config.stability_buoyancy_coefficient if dynamic else 0.0,
                config.stability_beta if dynamic else 0.0,
                config.stability_power if dynamic else 1.0,
            ),
        )

    def lasd_diagnostic_fields(
        self,
        context: ZSlabBoussinesqContext,
        momentum_config: LagrangianScaleDependentDynamic,
        scalar_config: LagrangianScaleDependentScalarFlux,
        boundary: ScalarFluxBoundary = ScalarFluxBoundary(),
        constants: DiagnosticLasdConstants = DiagnosticLasdConstants(),
        wall: NeutralLogWall | FilteredNeutralLogWall | None = None,
    ) -> LasdDiagnosticFields:
        """Return owned cell diagnostics and owned upper-face scalar flux."""
        if not isinstance(
            momentum_config, LagrangianScaleDependentDynamic
        ) or not isinstance(scalar_config, LagrangianScaleDependentScalarFlux):
            raise TypeError("LASD diagnostics require momentum and scalar LASD")
        if not isinstance(constants, DiagnosticLasdConstants):
            raise TypeError("unsupported LASD diagnostic constants")
        wall_gradient_factor = 0.0
        if wall is not None:
            if not isinstance(wall, (NeutralLogWall, FilteredNeutralLogWall)):
                raise TypeError("LASD diagnostic wall must be NeutralLogWall")
            reference_height = 0.5 * self.decomposition.grid.dz
            if wall.roughness_length >= reference_height:
                raise ValueError("wall roughness must be below the first cell centre")
            wall_gradient_factor = 1.0 / (
                math.log(reference_height / wall.roughness_length) * reference_height
            )
        closure = context.momentum.closure
        if not isinstance(closure, LasdClosureMemory):
            raise TypeError("LASD diagnostics require initialized closure memory")
        values = self.lasd.diagnostics(
            context.arrays,
            context.momentum.arrays,
            closure.momentum.coefficient.payload,
            closure.scalar.coefficient.payload,
            boundary.lower_flux,
            boundary.upper_flux,
            constants.sgs_dissipation_coefficient,
            constants.scalar_variance_coefficient,
            wall_gradient_factor,
            constants.horizontal_homogeneous_wall,
            scalar_config.stability_buoyancy_coefficient,
            scalar_config.stability_beta,
            scalar_config.stability_power,
        )
        return LasdDiagnosticFields(*values)

    def combine_scalar_tendencies(
        self,
        tendencies: tuple[AddressableField, ...],
    ) -> AddressableField:
        if not tendencies:
            raise ValueError("at least one scalar tendency is required")
        first = tendencies[0]
        for tendency in tendencies:
            self._validate_field(tendency, Cell)
            if (
                tendency.quantity
                not in (PotentialTemperatureTendency, PassiveScalarTendency)
                or tendency.phase is not Evaluated
            ):
                raise TypeError("only evaluated scalar tendencies may be combined")
            if tendency.payload.dtype != first.payload.dtype:
                raise TypeError("combined scalar tendencies must preserve dtype")
        return AddressableField(
            first.quantity,
            Cell,
            self._expected_regions(Cell),
            Evaluated,
            self.flow.combine_payloads(tuple(term.payload for term in tendencies)),
        )
