"""Independent, bounded, global tiny-grid JAX test oracle.

This module deliberately does not import production operator or pressure
modules. Its explicit full-face representation is a correctness oracle for the
different z-slab storage interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import jax
import jax.numpy as jnp

from jaxwind_archiv.domain import (
    Accepted,
    Cell,
    Candidate,
    Divergence,
    Evaluated,
    Field,
    PressureCorrection,
    PressureRhs,
    PassiveScalarConcentration,
    PassiveScalarTendency,
    PotentialTemperaturePerturbation,
    PotentialTemperatureTendency,
    Projected,
    VerticalBoundary,
    VerticalVelocity,
    VerticalVelocityTendency,
    XPressureGradient,
    XVelocity,
    XVelocityTendency,
    YPressureGradient,
    YVelocity,
    YVelocityTendency,
    ZFace,
)
from jaxwind_archiv.operators import PressureGradient, VelocityVector
from jaxwind_archiv.physics.boussinesq import BoussinesqFields

from .jax_oracle_core import (
    MAX_ORACLE_CELLS,
    OracleBoussinesqContext,
    OracleDryFlowContext,
    _boundary_plane,
    _cell_gradient_on_full_faces,
    _cell_to_full_faces,
    _horizontal_derivative,
    _horizontal_filter,
    _horizontal_symbols,
    _require_tiny_global,
    _require_velocity_component,
    divergence_z,
    pressure_gradient_z,
)

from .jax_oracle_flow import OracleFlowMixin
from .jax_oracle_lasd import OracleLasdMixin


@dataclass(frozen=True, slots=True)
class JaxOracleProjection(OracleLasdMixin, OracleFlowMixin):
    """Independent tiny-grid interpretation of the hybrid projection algebra."""

    def enforce_normal_boundary(
        self,
        velocity: VelocityVector,
        boundary: VerticalBoundary[Any],
    ) -> VelocityVector:
        x_ownership = _require_velocity_component(velocity.x, XVelocity)
        y_ownership = _require_velocity_component(velocity.y, YVelocity)
        z_ownership = _require_tiny_global(velocity.z, ZFace)
        if velocity.z.quantity is not VerticalVelocity:
            raise TypeError("vertical velocity requires VerticalVelocity")
        if velocity.z.phase not in (Candidate, Projected):
            raise TypeError("vertical velocity must be Candidate or Projected")
        if not (x_ownership.grid == y_ownership.grid == z_ownership.grid):
            raise ValueError("velocity components must share one grid")
        grid = x_ownership.grid
        lower = _horizontal_filter(
            _boundary_plane(boundary.lower, velocity.z),
            grid=grid,
        )
        upper = _horizontal_filter(
            _boundary_plane(boundary.upper, velocity.z),
            grid=grid,
        )
        z_payload = _horizontal_filter(velocity.z.payload, grid=grid)
        z_payload = z_payload.at[0].set(lower).at[-1].set(upper)
        return VelocityVector(
            Field(
                XVelocity,
                Cell,
                x_ownership,
                velocity.x.phase,
                _horizontal_filter(velocity.x.payload, grid=grid),
            ),
            Field(
                YVelocity,
                Cell,
                y_ownership,
                velocity.y.phase,
                _horizontal_filter(velocity.y.payload, grid=grid),
            ),
            Field(
                VerticalVelocity,
                ZFace,
                z_ownership,
                velocity.z.phase,
                z_payload,
            ),
        )

    def dry_flow_context(self, velocity: VelocityVector) -> OracleDryFlowContext:
        """Build the reference interpolation and gradient bundle exactly once."""
        x_ownership = _require_velocity_component(velocity.x, XVelocity)
        y_ownership = _require_velocity_component(velocity.y, YVelocity)
        z_ownership = _require_tiny_global(velocity.z, ZFace)
        if velocity.z.quantity is not VerticalVelocity:
            raise TypeError("dry-flow vertical velocity requires VerticalVelocity")
        if not (
            velocity.x.phase is Projected
            and velocity.y.phase is Projected
            and velocity.z.phase is Projected
        ):
            raise TypeError("dry-flow context requires projected velocity")
        if not (x_ownership.grid == y_ownership.grid == z_ownership.grid):
            raise ValueError("dry-flow velocity components must share one grid")
        grid = x_ownership.grid
        u = velocity.x.payload
        v = velocity.y.payload
        w = velocity.z.payload
        u_on_faces = _cell_to_full_faces(u)
        v_on_faces = _cell_to_full_faces(v)
        w_at_cells = 0.5 * (w[:-1] + w[1:])
        dudz_on_faces = _cell_gradient_on_full_faces(u, grid.dz)
        dvdz_on_faces = _cell_gradient_on_full_faces(v, grid.dz)
        wall_correction = 1.0 / math.log(3.0)
        dudz_on_faces = dudz_on_faces.at[1].multiply(wall_correction)
        dvdz_on_faces = dvdz_on_faces.at[1].multiply(wall_correction)
        dudz_at_cells = 0.5 * (dudz_on_faces[:-1] + dudz_on_faces[1:])
        dvdz_at_cells = 0.5 * (dvdz_on_faces[:-1] + dvdz_on_faces[1:])
        return OracleDryFlowContext(
            velocity,
            u_on_faces,
            v_on_faces,
            w_at_cells,
            _horizontal_derivative(u, grid=grid, axis="x"),
            _horizontal_derivative(u, grid=grid, axis="y"),
            dudz_at_cells,
            _horizontal_derivative(v, grid=grid, axis="x"),
            _horizontal_derivative(v, grid=grid, axis="y"),
            dvdz_at_cells,
            _horizontal_derivative(w_at_cells, grid=grid, axis="x"),
            _horizontal_derivative(w_at_cells, grid=grid, axis="y"),
            (w[1:] - w[:-1]) / grid.dz,
            dudz_on_faces,
            dvdz_on_faces,
            _horizontal_derivative(w, grid=grid, axis="x"),
            _horizontal_derivative(w, grid=grid, axis="y"),
        )

    def boussinesq_context(
        self,
        fields: BoussinesqFields,
    ) -> OracleBoussinesqContext:
        momentum = self.dry_flow_context(fields.velocity)
        scalar = fields.potential_temperature
        ownership = _require_tiny_global(scalar, Cell)
        if scalar.quantity not in (
            PotentialTemperaturePerturbation,
            PassiveScalarConcentration,
        ):
            raise TypeError("Boussinesq context requires a supported scalar quantity")
        if scalar.phase is not Accepted:
            raise TypeError("Boussinesq context requires accepted scalar state")
        if ownership.grid != momentum.velocity.x.ownership.grid:
            raise ValueError("velocity and scalar must share one grid")
        grid = ownership.grid
        momentum = replace(momentum, closure=fields.closure)
        return OracleBoussinesqContext(
            momentum,
            scalar,
            _cell_to_full_faces(scalar.payload),
            _horizontal_derivative(scalar.payload, grid=grid, axis="x"),
            _horizontal_derivative(scalar.payload, grid=grid, axis="y"),
            _cell_gradient_on_full_faces(scalar.payload, grid.dz),
        )

    def velocity_divergence(self, velocity: VelocityVector) -> Field:
        x_ownership = _require_velocity_component(velocity.x, XVelocity)
        y_ownership = _require_velocity_component(velocity.y, YVelocity)
        vertical = divergence_z(velocity.z)
        if not (x_ownership == y_ownership == vertical.ownership):
            raise ValueError("velocity components must share one ownership")
        grid = x_ownership.grid
        payload = (
            _horizontal_derivative(velocity.x.payload, grid=grid, axis="x")
            + _horizontal_derivative(velocity.y.payload, grid=grid, axis="y")
            + vertical.payload
        )
        return Field(Divergence, Cell, x_ownership, Evaluated, payload)

    def pressure_rhs(self, divergence: Field, inverse_dt: float) -> Field:
        ownership = _require_tiny_global(divergence, Cell)
        if divergence.quantity is not Divergence:
            raise TypeError("pressure RHS requires Divergence")
        return Field(
            PressureRhs,
            Cell,
            ownership,
            Evaluated,
            divergence.payload * inverse_dt,
        )

    def pressure_gradient(self, pressure: Field) -> PressureGradient:
        ownership = _require_tiny_global(pressure, Cell)
        if pressure.quantity is not PressureCorrection:
            raise TypeError("pressure gradient requires PressureCorrection")
        grid = ownership.grid
        return PressureGradient(
            Field(
                XPressureGradient,
                Cell,
                ownership,
                Evaluated,
                _horizontal_derivative(pressure.payload, grid=grid, axis="x"),
            ),
            Field(
                YPressureGradient,
                Cell,
                ownership,
                Evaluated,
                _horizontal_derivative(pressure.payload, grid=grid, axis="y"),
            ),
            pressure_gradient_z(pressure, VerticalBoundary(0.0, 0.0)),
        )

    def correct_velocity(
        self,
        velocity: VelocityVector,
        gradient: PressureGradient,
        dt: float,
    ) -> VelocityVector:
        x_dt = jnp.asarray(dt, dtype=velocity.x.payload.dtype)
        y_dt = jnp.asarray(dt, dtype=velocity.y.payload.dtype)
        z_dt = jnp.asarray(dt, dtype=velocity.z.payload.dtype)
        return VelocityVector(
            Field(
                XVelocity,
                Cell,
                velocity.x.ownership,
                Projected,
                velocity.x.payload - x_dt * gradient.x.payload,
            ),
            Field(
                YVelocity,
                Cell,
                velocity.y.ownership,
                Projected,
                velocity.y.payload - y_dt * gradient.y.payload,
            ),
            Field(
                VerticalVelocity,
                ZFace,
                velocity.z.ownership,
                Projected,
                velocity.z.payload - z_dt * gradient.z.payload,
            ),
        )

    def ab2_candidate_velocity(
        self,
        velocity: VelocityVector,
        current_tendency: VelocityVector,
        previous_tendency: VelocityVector,
        *,
        dt: float,
        current_weight: float,
        previous_weight: float,
    ) -> VelocityVector:
        """Form an Euler/AB2 candidate without changing the vector field."""
        components = (
            (
                velocity.x,
                current_tendency.x,
                previous_tendency.x,
                XVelocity,
                XVelocityTendency,
                Cell,
            ),
            (
                velocity.y,
                current_tendency.y,
                previous_tendency.y,
                YVelocity,
                YVelocityTendency,
                Cell,
            ),
            (
                velocity.z,
                current_tendency.z,
                previous_tendency.z,
                VerticalVelocity,
                VerticalVelocityTendency,
                ZFace,
            ),
        )
        candidates = []
        for (
            state,
            current,
            previous,
            state_quantity,
            tendency_quantity,
            location,
        ) in components:
            ownership = _require_tiny_global(state, location)
            if state.quantity is not state_quantity or state.phase is not Projected:
                raise TypeError("AB2 requires projected velocity state components")
            for tendency in (current, previous):
                tendency_ownership = _require_tiny_global(tendency, location)
                if tendency.quantity is not tendency_quantity:
                    raise TypeError("AB2 received an incorrect tendency quantity")
                if tendency.phase is not Evaluated:
                    raise TypeError("AB2 tendencies must be Evaluated")
                if tendency_ownership != ownership:
                    raise ValueError("AB2 state and tendencies must share ownership")
            local_dt = jnp.asarray(dt, dtype=state.payload.dtype)
            current_coefficient = jnp.asarray(
                current_weight,
                dtype=state.payload.dtype,
            )
            previous_coefficient = jnp.asarray(
                previous_weight,
                dtype=state.payload.dtype,
            )
            payload = state.payload + local_dt * (
                current_coefficient * current.payload
                + previous_coefficient * previous.payload
            )
            candidates.append(
                Field(
                    state_quantity,
                    location,
                    ownership,
                    Candidate,
                    payload,
                )
            )
        return VelocityVector(*candidates)

    def ab2_candidate_scalar(
        self,
        scalar: Field,
        current_tendency: Field,
        previous_tendency: Field,
        *,
        dt: float,
        current_weight: float,
        previous_weight: float,
    ) -> Field:
        ownership = _require_tiny_global(scalar, Cell)
        tendency_quantity = (
            PotentialTemperatureTendency
            if scalar.quantity is PotentialTemperaturePerturbation
            else PassiveScalarTendency
        )
        if (
            scalar.quantity
            not in (
                PotentialTemperaturePerturbation,
                PassiveScalarConcentration,
            )
            or scalar.phase is not Accepted
        ):
            raise TypeError("AB2 requires an accepted supported scalar")
        for tendency in (current_tendency, previous_tendency):
            if _require_tiny_global(tendency, Cell) != ownership:
                raise ValueError("scalar state and tendency must share ownership")
            if (
                tendency.quantity is not tendency_quantity
                or tendency.phase is not Evaluated
            ):
                raise TypeError("AB2 requires evaluated scalar tendency")
        local_dt = jnp.asarray(dt, dtype=scalar.payload.dtype)
        payload = scalar.payload + local_dt * (
            jnp.asarray(current_weight, dtype=scalar.payload.dtype)
            * current_tendency.payload
            + jnp.asarray(previous_weight, dtype=scalar.payload.dtype)
            * previous_tendency.payload
        )
        return Field(
            scalar.quantity,
            Cell,
            ownership,
            Candidate,
            payload,
        )

    def accept_scalar(self, scalar: Field) -> Field:
        ownership = _require_tiny_global(scalar, Cell)
        if (
            scalar.quantity
            not in (
                PotentialTemperaturePerturbation,
                PassiveScalarConcentration,
            )
            or scalar.phase is not Candidate
        ):
            raise TypeError("only a candidate supported scalar may be accepted")
        return Field(
            scalar.quantity,
            Cell,
            ownership,
            Accepted,
            scalar.payload,
        )


@dataclass(frozen=True, slots=True)
class JaxOraclePressureSolver:
    """Independent spectral/dense-z solve for bounded global test fields."""

    def solve(self, rhs: Field) -> Field:
        ownership = _require_tiny_global(rhs, Cell)
        if rhs.quantity is not PressureRhs:
            raise TypeError("reference pressure solver requires PressureRhs")
        grid = ownership.grid
        if grid.nx % 2 or grid.ny % 2:
            raise ValueError("reference projection requires even nx and ny")
        compatible_rhs = rhs.payload - jnp.mean(rhs.payload)
        spectrum = jnp.fft.rfftn(compatible_rhs, axes=(-2, -1))
        kx, ky, keep = _horizontal_symbols(grid, compatible_rhs.dtype)
        k2 = ky[:, None] ** 2 + kx[None, :] ** 2
        dz2 = grid.dz * grid.dz
        base = jnp.diag(jnp.full((grid.nz,), -2.0 / dz2))
        base = base + jnp.diag(jnp.full((grid.nz - 1,), 1.0 / dz2), 1)
        base = base + jnp.diag(jnp.full((grid.nz - 1,), 1.0 / dz2), -1)
        base = base.at[0, 0].add(1.0 / dz2)
        base = base.at[-1, -1].add(1.0 / dz2)
        mode_rhs = jnp.moveaxis(spectrum, 0, -1)
        flat_rhs = mode_rhs.reshape((-1, grid.nz))
        flat_k2 = k2.reshape(-1)
        flat_keep = keep.reshape(-1)
        tolerance = jnp.finfo(compatible_rhs.dtype).eps * 128.0

        def solve_mode(values, horizontal_k2, retained):
            operator = base - horizontal_k2 * jnp.eye(
                grid.nz,
                dtype=compatible_rhs.dtype,
            )
            is_zero = jnp.abs(horizontal_k2) < tolerance
            gauged = operator.at[0].set(
                jnp.full((grid.nz,), 1.0 / grid.nz, compatible_rhs.dtype)
            )
            gauged_rhs = values.at[0].set(0.0)
            selected_operator = jnp.where(is_zero, gauged, operator)
            selected_rhs = jnp.where(is_zero, gauged_rhs, values)
            solution = jnp.linalg.solve(selected_operator, selected_rhs)
            return jnp.where(retained > 0.0, solution, jnp.zeros_like(solution))

        flat_pressure = jax.vmap(solve_mode)(flat_rhs, flat_k2, flat_keep)
        pressure_spectrum = jnp.moveaxis(
            flat_pressure.reshape(mode_rhs.shape),
            -1,
            0,
        )
        pressure = jnp.fft.irfftn(
            pressure_spectrum,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(rhs.payload.dtype)
        pressure = pressure - jnp.mean(pressure)
        return Field(
            PressureCorrection,
            Cell,
            ownership,
            Evaluated,
            pressure,
        )
