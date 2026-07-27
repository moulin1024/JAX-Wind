"""Independent, bounded, global tiny-grid JAX reference operators.

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

from wireles.interpreters.jax_actuator_disk import (
    filtered_disk_velocity_correction,
    gaussian_convolved_annulus,
)

from wireles.domain import (
    Accepted,
    Cell,
    Candidate,
    Divergence,
    Evaluated,
    Field,
    GlobalTestRegion,
    LasdTrajectoryXVelocity,
    LasdTrajectoryYVelocity,
    LasdTrajectoryZVelocity,
    MomentumLasdCoefficient,
    MomentumLasdLm,
    MomentumLasdMm,
    MomentumLasdNn,
    MomentumLasdQn,
    PressureCorrection,
    PressureRhs,
    PassiveScalarConcentration,
    PassiveScalarTendency,
    PotentialTemperaturePerturbation,
    PotentialTemperatureTendency,
    Projected,
    ScalarLasdCoefficient,
    ScalarLasdLm,
    ScalarLasdMm,
    ScalarLasdNn,
    ScalarLasdQn,
    VerticalBoundary,
    VerticalPressureGradient,
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
from wireles.operators import PressureGradient, VelocityVector
from wireles.physics.dry_flow import (
    ConservativeAdvection,
    CoriolisGeostrophic,
    FilteredNeutralLogWall,
    KinematicPressureGradient,
    NeutralLogWall,
    NoRotation,
    StaticSmagorinsky,
)
from wireles.physics.boussinesq import (
    BoussinesqFields,
    ConservativeScalarAdvection,
    LinearBoussinesqBuoyancy,
    NoBuoyancy,
    NoRayleighDamping,
    RayleighGeostrophicDamping,
    ScalarFluxBoundary,
    StaticSmagorinskyScalarFlux,
)
from wireles.physics.lasd import (
    DiagnosticLasdConstants,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LasdClosureEventDiagnostic,
    LasdClosureMemory,
    LasdDiagnosticFields,
    MomentumLasdMemory,
    ScalarLasdMemory,
)
from wireles.physics.wind_tunnel import (
    ConcurrentPrecursorEnvironment,
    ConcurrentPrecursorFringe,
    NoActuatorDisk,
    NoFringe,
    PureThrustActuatorDisk,
    WindTunnelModel,
)


MAX_REFERENCE_CELLS = 32_768


def _require_tiny_global(field: Field, location: type) -> GlobalTestRegion:
    ownership = field.ownership
    if not isinstance(ownership, GlobalTestRegion):
        raise TypeError("the JAX reference interpreter requires global test ownership")
    if field.location is not location:
        raise TypeError(f"reference operator requires {location.__name__} input")
    if ownership.grid.cell_count > MAX_REFERENCE_CELLS:
        raise ValueError(
            "reference grid exceeds the bounded global limit of "
            f"{MAX_REFERENCE_CELLS} cells"
        )
    return ownership


def _boundary_plane(value: Any, field: Field):
    plane = jnp.asarray(value, dtype=field.payload.dtype)
    return jnp.broadcast_to(
        plane,
        (field.ownership.grid.ny, field.ownership.grid.nx),
    )


def pressure_gradient_z(
    pressure: Field,
    boundary_gradient: VerticalBoundary[Any],
) -> Field:
    """Interpret ``G_z: Cell -> ZFace`` with all ``nz+1`` faces explicit."""
    ownership = _require_tiny_global(pressure, Cell)
    if pressure.quantity is not PressureCorrection:
        raise TypeError("pressure_gradient_z requires PressureCorrection")
    lower = _boundary_plane(boundary_gradient.lower, pressure)
    upper = _boundary_plane(boundary_gradient.upper, pressure)
    interior = (pressure.payload[1:] - pressure.payload[:-1]) / ownership.grid.dz
    payload = jnp.concatenate((lower[None, ...], interior, upper[None, ...]), axis=0)
    return Field(
        VerticalPressureGradient,
        ZFace,
        ownership.at_location(ZFace),
        Evaluated,
        payload,
    )


def divergence_z(vertical_faces: Field) -> Field:
    """Interpret ``D_z: ZFace -> Cell`` by an explicit face flux difference."""
    ownership = _require_tiny_global(vertical_faces, ZFace)
    if vertical_faces.quantity not in (
        VerticalVelocity,
        VerticalPressureGradient,
    ):
        raise TypeError("divergence_z requires a vertical face-normal quantity")
    payload = (
        vertical_faces.payload[1:] - vertical_faces.payload[:-1]
    ) / ownership.grid.dz
    return Field(
        Divergence,
        Cell,
        ownership.at_location(Cell),
        Evaluated,
        payload,
    )


def _horizontal_symbols(grid, dtype):
    kx = 2.0 * jnp.pi * jnp.fft.rfftfreq(grid.nx, d=grid.lx / grid.nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(grid.ny, d=grid.ly / grid.ny)
    kx = kx.astype(dtype)
    ky = ky.astype(dtype)
    keep = jnp.ones((grid.ny, grid.nx // 2 + 1), dtype=dtype)
    if grid.nx % 2 == 0:
        kx = kx.at[-1].set(0.0)
        keep = keep.at[:, -1].set(0.0)
    if grid.ny % 2 == 0:
        ky = ky.at[grid.ny // 2].set(0.0)
        keep = keep.at[grid.ny // 2, :].set(0.0)
    return kx, ky, keep


def _horizontal_derivative(values, *, grid, axis: str):
    kx, ky, keep = _horizontal_symbols(grid, values.real.dtype)
    spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
    if axis == "x":
        multiplier = 1j * kx[None, None, :]
    elif axis == "y":
        multiplier = 1j * ky[None, :, None]
    else:
        raise ValueError("horizontal derivative axis must be 'x' or 'y'")
    return jnp.fft.irfftn(
        spectrum * multiplier * keep[None, ...],
        s=(grid.ny, grid.nx),
        axes=(-2, -1),
    ).astype(values.dtype)


def _horizontal_filter(values, *, grid):
    _, _, keep = _horizontal_symbols(grid, values.real.dtype)
    spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
    return jnp.fft.irfftn(
        spectrum * keep,
        s=(grid.ny, grid.nx),
        axes=(-2, -1),
    ).astype(values.dtype)


def _wall_filter(values, *, grid, filter_width: float):
    """Apply the legacy sharp two-dimensional filter at the wall."""
    spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
    x_mode = jnp.arange(grid.nx // 2 + 1)
    y_mode = jnp.abs(jnp.fft.fftfreq(grid.ny) * grid.ny)
    cutoff_x = jnp.floor(grid.nx / (2.0 * filter_width))
    cutoff_y = jnp.floor(grid.ny / (2.0 * filter_width))
    keep = (y_mode[:, None] < cutoff_y) & (x_mode[None, :] < cutoff_x)
    return jnp.fft.irfftn(
        spectrum * keep,
        s=(grid.ny, grid.nx),
        axes=(-2, -1),
    ).astype(values.dtype)


def _two_thirds_mask(grid, dtype):
    """Fixed sharp mask used only for nonlinear horizontal products."""
    x_mode = jnp.arange(grid.nx // 2 + 1)
    y_mode = jnp.fft.fftfreq(grid.ny) * grid.ny
    keep_x = x_mode <= grid.nx // 3
    keep_y = jnp.abs(y_mode) <= grid.ny // 3
    return (keep_y[:, None] & keep_x[None, :]).astype(dtype)


def _two_thirds_filter(values, *, grid):
    spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
    mask = _two_thirds_mask(grid, values.real.dtype)
    return jnp.fft.irfftn(
        spectrum * mask,
        s=(grid.ny, grid.nx),
        axes=(-2, -1),
    ).astype(values.dtype)


def _truncated_horizontal_derivative(values, *, grid, axis: str):
    kx, ky, _ = _horizontal_symbols(grid, values.real.dtype)
    spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
    if axis == "x":
        multiplier = 1j * kx[None, None, :]
    elif axis == "y":
        multiplier = 1j * ky[None, :, None]
    else:
        raise ValueError("horizontal derivative axis must be 'x' or 'y'")
    mask = _two_thirds_mask(grid, values.real.dtype)
    return jnp.fft.irfftn(
        spectrum * multiplier * mask,
        s=(grid.ny, grid.nx),
        axes=(-2, -1),
    ).astype(values.dtype)


def _require_velocity_component(field: Field, quantity: type) -> GlobalTestRegion:
    ownership = _require_tiny_global(field, Cell)
    if field.quantity is not quantity:
        raise TypeError(f"velocity component requires {quantity.__name__}")
    if field.phase not in (Candidate, Projected):
        raise TypeError("velocity component must be Candidate or Projected")
    return ownership


@dataclass(frozen=True, slots=True)
class ReferenceDryFlowContext:
    """Global tiny-grid velocity, interpolation, and shared gradient bundle."""

    velocity: VelocityVector
    u_on_faces: Any
    v_on_faces: Any
    w_at_cells: Any
    dudx: Any
    dudy: Any
    dudz_at_cells: Any
    dvdx: Any
    dvdy: Any
    dvdz_at_cells: Any
    dwdx_at_cells: Any
    dwdy_at_cells: Any
    dwdz: Any
    dudz_on_faces: Any
    dvdz_on_faces: Any
    dwdx_on_faces: Any
    dwdy_on_faces: Any
    closure: Any = None


@dataclass(frozen=True, slots=True)
class ReferenceBoussinesqContext:
    momentum: ReferenceDryFlowContext
    potential_temperature: Field
    theta_on_faces: Any
    dtheta_dx: Any
    dtheta_dy: Any
    dtheta_dz_on_faces: Any


def _cell_to_full_faces(values):
    interior = 0.5 * (values[:-1] + values[1:])
    return jnp.concatenate(
        (values[:1], interior, values[-1:]),
        axis=0,
    )


def _cell_gradient_on_full_faces(values, dz: float):
    zero = jnp.zeros_like(values[:1])
    interior = (values[1:] - values[:-1]) / dz
    return jnp.concatenate((zero, interior, zero), axis=0)


def _strain_magnitude(
    dudx,
    dudy,
    dudz,
    dvdx,
    dvdy,
    dvdz,
    dwdx,
    dwdy,
    dwdz,
):
    sxy = 0.5 * (dudy + dvdx)
    sxz = 0.5 * (dudz + dwdx)
    syz = 0.5 * (dvdz + dwdy)
    symmetric_dot = (
        dudx * dudx
        + dvdy * dvdy
        + dwdz * dwdz
        + 2.0 * (sxy * sxy + sxz * sxz + syz * syz)
    )
    return jnp.sqrt(jnp.maximum(2.0 * symmetric_dot, 0.0))


def _strain_tensor(context: ReferenceDryFlowContext):
    return jnp.stack(
        (
            context.dudx,
            0.5 * (context.dudy + context.dvdx),
            0.5 * (context.dudz_at_cells + context.dwdx_at_cells),
            context.dvdy,
            0.5 * (context.dvdz_at_cells + context.dwdy_at_cells),
            context.dwdz,
        ),
        axis=-1,
    )


def _symmetric_dot(left, right):
    return (
        left[..., 0] * right[..., 0]
        + 2.0 * left[..., 1] * right[..., 1]
        + 2.0 * left[..., 2] * right[..., 2]
        + left[..., 3] * right[..., 3]
        + 2.0 * left[..., 4] * right[..., 4]
        + left[..., 5] * right[..., 5]
    )


def _tensor_magnitude(tensor):
    return jnp.sqrt(jnp.maximum(2.0 * _symmetric_dot(tensor, tensor), 0.0))


def _lasd_filter(values, *, grid, filter_width: float):
    spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
    x_mode = jnp.arange(grid.nx // 2 + 1)
    y_mode = jnp.abs(jnp.fft.fftfreq(grid.ny) * grid.ny)
    cutoff_x = jnp.floor(grid.nx / (2.0 * filter_width) + 0.5)
    cutoff_y = jnp.floor(grid.ny / (2.0 * filter_width) + 0.5)
    mask = (y_mode[:, None] < cutoff_y) & (x_mode[None, :] < cutoff_x)
    return jnp.fft.irfftn(
        spectrum * mask,
        s=(grid.ny, grid.nx),
        axes=(-2, -1),
    ).astype(values.dtype)


def _lasd_filter_components(values, *, grid, filter_width: float):
    """Filter z,y,x,component arrays without transforming the component axis."""
    moved = jnp.moveaxis(values, -1, 0)
    filtered = jax.vmap(
        lambda value: _lasd_filter(value, grid=grid, filter_width=filter_width)
    )(moved)
    return jnp.moveaxis(filtered, 0, -1)


def _safe_divide(numerator, denominator):
    valid = jnp.abs(denominator) > 1.0e-30
    return jnp.where(valid, numerator / jnp.where(valid, denominator, 1.0), 0.0)


def _lasd_beta(
    coefficient_2d,
    coefficient_4d,
    config,
    *,
    scale_dependent: bool | None = None,
):
    exponent = math.log(config.test_filter_ratio) / (
        math.log(config.test_filter_ratio**2) - math.log(config.test_filter_ratio)
    )
    raw = jnp.maximum(_safe_divide(coefficient_4d, coefficient_2d), 0.0) ** exponent
    beta = jnp.maximum(raw, 1.0 / config.test_filter_ratio**3)
    enabled = config.scale_dependent if scale_dependent is None else scale_dependent
    return beta if enabled else jnp.ones_like(beta)


def _history_boundary(values):
    if values.shape[0] < 2:
        return values
    return values.at[0].set(values[1]).at[-1].set(values[-2])


def _departure_interpolate(
    values, trajectory_x, trajectory_y, trajectory_z, *, grid, dt
):
    z_index = jnp.arange(grid.nz, dtype=trajectory_x.dtype)[:, None, None]
    y_index = jnp.arange(grid.ny, dtype=trajectory_x.dtype)[None, :, None]
    x_index = jnp.arange(grid.nx, dtype=trajectory_x.dtype)[None, None, :]
    xi = jnp.mod(x_index - trajectory_x * dt / grid.dx, grid.nx)
    eta = jnp.mod(y_index - trajectory_y * dt / grid.dy, grid.ny)
    zeta = jnp.clip(z_index - trajectory_z * dt / grid.dz, 0.0, grid.nz - 1.0)
    i0 = jnp.floor(xi).astype(jnp.int32)
    j0 = jnp.floor(eta).astype(jnp.int32)
    k0 = jnp.floor(zeta).astype(jnp.int32)
    i1 = (i0 + 1) % grid.nx
    j1 = (j0 + 1) % grid.ny
    k1 = jnp.minimum(k0 + 1, grid.nz - 1)
    fx = xi - i0
    fy = eta - j0
    fz = zeta - k0
    while fx.ndim < values.ndim:
        fx = fx[..., None]
        fy = fy[..., None]
        fz = fz[..., None]
    q000 = values[k0, j0, i0]
    q100 = values[k0, j0, i1]
    q010 = values[k0, j1, i0]
    q110 = values[k0, j1, i1]
    q001 = values[k1, j0, i0]
    q101 = values[k1, j0, i1]
    q011 = values[k1, j1, i0]
    q111 = values[k1, j1, i1]
    q00 = (1.0 - fx) * q000 + fx * q100
    q10 = (1.0 - fx) * q010 + fx * q110
    q01 = (1.0 - fx) * q001 + fx * q101
    q11 = (1.0 - fx) * q011 + fx * q111
    q0 = (1.0 - fy) * q00 + fy * q10
    q1 = (1.0 - fy) * q01 + fy * q11
    return (1.0 - fz) * q0 + fz * q1


def _lagrangian_average(
    current_a,
    current_b,
    old_a,
    old_b,
    trajectory_x,
    trajectory_y,
    trajectory_z,
    *,
    grid,
    interval_dt,
    timescale_coefficient,
    timescale_a=None,
    timescale_b=None,
):
    scale_a = old_a if timescale_a is None else timescale_a
    scale_b = old_b if timescale_b is None else timescale_b
    product = scale_a * scale_b
    valid = (scale_a > 0.0) & (scale_b >= 0.0) & (product > 0.0)
    delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
    timescale = (
        timescale_coefficient
        * delta
        * jnp.where(
            valid,
            product ** (-0.125),
            1.0,
        )
    )
    weight = jnp.where(
        valid,
        (interval_dt / timescale) / (1.0 + interval_dt / timescale),
        0.0,
    )
    a_departure = _departure_interpolate(
        old_a,
        trajectory_x,
        trajectory_y,
        trajectory_z,
        grid=grid,
        dt=interval_dt,
    )
    b_departure = _departure_interpolate(
        old_b,
        trajectory_x,
        trajectory_y,
        trajectory_z,
        grid=grid,
        dt=interval_dt,
    )
    return (
        weight * current_a + (1.0 - weight) * a_departure,
        jnp.maximum(weight * current_b + (1.0 - weight) * b_departure, 0.0),
    )


def _momentum_lasd_contractions(context, config, ratio: float):
    grid = context.velocity.x.ownership.grid
    tensor = _strain_tensor(context)
    magnitude = _tensor_magnitude(tensor)
    velocity = jnp.stack(
        (
            context.velocity.x.payload,
            context.velocity.y.payload,
            context.w_at_cells,
        ),
        axis=-1,
    )
    products = jnp.stack(
        (
            velocity[..., 0] * velocity[..., 0],
            velocity[..., 0] * velocity[..., 1],
            velocity[..., 0] * velocity[..., 2],
            velocity[..., 1] * velocity[..., 1],
            velocity[..., 1] * velocity[..., 2],
            velocity[..., 2] * velocity[..., 2],
        ),
        axis=-1,
    )
    width = config.filter_grid_ratio * ratio
    velocity_hat = _lasd_filter_components(velocity, grid=grid, filter_width=width)
    products_hat = _lasd_filter_components(products, grid=grid, filter_width=width)
    tensor_hat = _lasd_filter_components(tensor, grid=grid, filter_width=width)
    magnitude_tensor_hat = _lasd_filter_components(
        magnitude[..., None] * tensor,
        grid=grid,
        filter_width=width,
    )
    resolved = jnp.stack(
        (
            products_hat[..., 0] - velocity_hat[..., 0] ** 2,
            products_hat[..., 1] - velocity_hat[..., 0] * velocity_hat[..., 1],
            products_hat[..., 2] - velocity_hat[..., 0] * velocity_hat[..., 2],
            products_hat[..., 3] - velocity_hat[..., 1] ** 2,
            products_hat[..., 4] - velocity_hat[..., 1] * velocity_hat[..., 2],
            products_hat[..., 5] - velocity_hat[..., 2] ** 2,
        ),
        axis=-1,
    )
    delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
    model = (
        2.0
        * delta**2
        * (
            magnitude_tensor_hat
            - ratio**2 * _tensor_magnitude(tensor_hat)[..., None] * tensor_hat
        )
    )
    return _symmetric_dot(resolved, model), _symmetric_dot(model, model)


def _scalar_cell_gradient(context):
    scalar = context.potential_temperature.payload
    if scalar.shape[0] == 1:
        vertical = jnp.zeros_like(scalar)
    else:
        vertical = jnp.empty_like(scalar)
        vertical = vertical.at[0].set(
            (scalar[1] - scalar[0]) / context.momentum.velocity.x.ownership.grid.dz
        )
        vertical = vertical.at[-1].set(
            (scalar[-1] - scalar[-2]) / context.momentum.velocity.x.ownership.grid.dz
        )
        if scalar.shape[0] > 2:
            vertical = vertical.at[1:-1].set(
                (scalar[2:] - scalar[:-2])
                / (2.0 * context.momentum.velocity.x.ownership.grid.dz)
            )
    return jnp.stack((context.dtheta_dx, context.dtheta_dy, vertical), axis=-1)


def _scalar_lasd_contractions(context, config, ratio: float):
    momentum = context.momentum
    grid = momentum.velocity.x.ownership.grid
    tensor = _strain_tensor(momentum)
    magnitude = _tensor_magnitude(tensor)
    velocity = jnp.stack(
        (
            momentum.velocity.x.payload,
            momentum.velocity.y.payload,
            momentum.w_at_cells,
        ),
        axis=-1,
    )
    scalar = context.potential_temperature.payload
    # The test filter is horizontal at fixed z, so subtracting the plane mean
    # leaves the exact Germano identity unchanged while avoiding cancellation
    # against a large scalar reference value in float32.
    scalar = scalar - jnp.mean(scalar, axis=(-2, -1), keepdims=True)
    gradient = _scalar_cell_gradient(context)
    velocity_scalar = velocity * scalar[..., None]
    width = config.filter_grid_ratio * ratio
    velocity_hat = _lasd_filter_components(velocity, grid=grid, filter_width=width)
    scalar_hat = _lasd_filter(scalar, grid=grid, filter_width=width)
    velocity_scalar_hat = _lasd_filter_components(
        velocity_scalar,
        grid=grid,
        filter_width=width,
    )
    gradient_hat = _lasd_filter_components(gradient, grid=grid, filter_width=width)
    strain_gradient_hat = _lasd_filter_components(
        magnitude[..., None] * gradient,
        grid=grid,
        filter_width=width,
    )
    tensor_hat = _lasd_filter_components(tensor, grid=grid, filter_width=width)
    resolved = velocity_scalar_hat - velocity_hat * scalar_hat[..., None]
    delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
    model = delta**2 * (
        strain_gradient_hat
        - ratio**2 * _tensor_magnitude(tensor_hat)[..., None] * gradient_hat
    )
    return jnp.sum(resolved * model, axis=-1), jnp.sum(model * model, axis=-1)


def _reference_tendency_from_velocity(
    velocity: VelocityVector, x, y, z
) -> VelocityVector:
    return VelocityVector(
        Field(
            XVelocityTendency,
            Cell,
            velocity.x.ownership,
            Evaluated,
            x.astype(velocity.x.payload.dtype),
        ),
        Field(
            YVelocityTendency,
            Cell,
            velocity.y.ownership,
            Evaluated,
            y.astype(velocity.y.payload.dtype),
        ),
        Field(
            VerticalVelocityTendency,
            ZFace,
            velocity.z.ownership,
            Evaluated,
            z.astype(velocity.z.payload.dtype),
        ),
    )


def _reference_tendency(context: ReferenceDryFlowContext, x, y, z) -> VelocityVector:
    return _reference_tendency_from_velocity(context.velocity, x, y, z)


@dataclass(frozen=True, slots=True)
class JaxReferenceProjection:
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

    def dry_flow_context(self, velocity: VelocityVector) -> ReferenceDryFlowContext:
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
        return ReferenceDryFlowContext(
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
    ) -> ReferenceBoussinesqContext:
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
        return ReferenceBoussinesqContext(
            momentum,
            scalar,
            _cell_to_full_faces(scalar.payload),
            _horizontal_derivative(scalar.payload, grid=grid, axis="x"),
            _horizontal_derivative(scalar.payload, grid=grid, axis="y"),
            _cell_gradient_on_full_faces(scalar.payload, grid.dz),
        )

    @staticmethod
    def _reference_closure_field(
        template: Field, quantity: type, payload: Any
    ) -> Field:
        return Field(
            quantity,
            Cell,
            template.ownership,
            Accepted,
            payload.astype(template.payload.dtype),
        )

    def initialize_lasd_closure(
        self, fields: BoussinesqFields, model: Any
    ) -> BoussinesqFields:
        momentum_config = model.momentum.sgs
        scalar_config = model.scalar_sgs
        if not isinstance(
            momentum_config, LagrangianScaleDependentDynamic
        ) or not isinstance(
            scalar_config,
            LagrangianScaleDependentScalarFlux,
        ):
            raise TypeError("LASD initialization requires momentum and scalar LASD")
        scalar = fields.potential_temperature
        _require_tiny_global(scalar, Cell)
        zero = jnp.zeros_like(scalar.payload)
        momentum_coefficient = jnp.full_like(
            scalar.payload,
            momentum_config.initial_coefficient,
        )
        scalar_coefficient = jnp.full_like(
            scalar.payload,
            scalar_config.initial_coefficient,
        )
        field = lambda quantity, payload: self._reference_closure_field(  # noqa: E731
            scalar,
            quantity,
            payload,
        )
        closure = LasdClosureMemory(
            MomentumLasdMemory(
                field(MomentumLasdCoefficient, momentum_coefficient),
                field(MomentumLasdLm, zero),
                field(MomentumLasdMm, zero),
                field(MomentumLasdQn, zero),
                field(MomentumLasdNn, zero),
                field(LasdTrajectoryXVelocity, zero),
                field(LasdTrajectoryYVelocity, zero),
                field(LasdTrajectoryZVelocity, zero),
            ),
            ScalarLasdMemory(
                field(ScalarLasdCoefficient, scalar_coefficient),
                field(ScalarLasdLm, zero),
                field(ScalarLasdMm, zero),
                field(ScalarLasdQn, zero),
                field(ScalarLasdNn, zero),
            ),
            momentum_config.fingerprint + "|" + scalar_config.fingerprint,
        )
        return replace(fields, closure=closure)

    def prepare_lasd_closure(
        self,
        fields: BoussinesqFields,
        model: Any,
        clock: Any,
        dt: float,
    ) -> tuple[BoussinesqFields, LasdClosureEventDiagnostic]:
        momentum_config = model.momentum.sgs
        scalar_config = model.scalar_sgs
        if not isinstance(
            momentum_config, LagrangianScaleDependentDynamic
        ) or not isinstance(
            scalar_config,
            LagrangianScaleDependentScalarFlux,
        ):
            raise TypeError("LASD event requires momentum and scalar LASD")
        closure = fields.closure
        if not isinstance(closure, LasdClosureMemory):
            raise TypeError("LASD event requires initialized closure memory")
        fingerprint = momentum_config.fingerprint + "|" + scalar_config.fingerprint
        if closure.configuration_fingerprint != fingerprint:
            raise ValueError("LASD memory fingerprint does not match the model")
        context = self.boussinesq_context(fields)
        momentum = context.momentum
        old_m = closure.momentum
        old_s = closure.scalar
        interval = momentum_config.update_interval
        trajectory_x = (
            old_m.trajectory_x.payload + momentum.velocity.x.payload / interval
        )
        trajectory_y = (
            old_m.trajectory_y.payload + momentum.velocity.y.payload / interval
        )
        trajectory_z = old_m.trajectory_z.payload + momentum.w_at_cells / interval
        should_update = (clock.step + 1) % interval == 0
        field = lambda template, payload: self._reference_closure_field(  # noqa: E731
            template,
            template.quantity,
            payload,
        )
        if should_update:
            ratio = momentum_config.test_filter_ratio
            lm, mm = _momentum_lasd_contractions(
                context.momentum, momentum_config, ratio
            )
            qn, nn = _momentum_lasd_contractions(
                context.momentum,
                momentum_config,
                ratio**2,
            )
            first_update = clock.step == interval - 1
            old_lm = jnp.where(
                first_update, momentum_config.initial_coefficient * mm, old_m.lm.payload
            )
            old_mm = jnp.where(first_update, mm, old_m.mm.payload)
            old_qn = jnp.where(
                first_update, momentum_config.initial_coefficient * nn, old_m.qn.payload
            )
            old_nn = jnp.where(first_update, nn, old_m.nn.payload)
            old_lm, old_mm, old_qn, old_nn = (
                _history_boundary(value) for value in (old_lm, old_mm, old_qn, old_nn)
            )
            interval_dt = dt * interval
            lm_avg, mm_avg = _lagrangian_average(
                lm,
                mm,
                old_lm,
                old_mm,
                trajectory_x,
                trajectory_y,
                trajectory_z,
                grid=momentum.velocity.x.ownership.grid,
                interval_dt=interval_dt,
                timescale_coefficient=momentum_config.timescale_coefficient,
            )
            qn_avg, nn_avg = _lagrangian_average(
                qn,
                nn,
                old_qn,
                old_nn,
                trajectory_x,
                trajectory_y,
                trajectory_z,
                grid=momentum.velocity.x.ownership.grid,
                interval_dt=interval_dt,
                timescale_coefficient=momentum_config.timescale_coefficient,
            )
            coefficient_2d = jnp.maximum(_safe_divide(lm_avg, mm_avg), 0.0)
            coefficient_4d = jnp.maximum(_safe_divide(qn_avg, nn_avg), 0.0)
            momentum_coefficient = jnp.clip(
                _safe_divide(
                    coefficient_2d,
                    _lasd_beta(coefficient_2d, coefficient_4d, momentum_config),
                ),
                momentum_config.minimum_coefficient,
                momentum_config.maximum_coefficient,
            )

            scalar_lm, scalar_mm = _scalar_lasd_contractions(
                context,
                momentum_config,
                ratio,
            )
            scalar_qn, scalar_nn = _scalar_lasd_contractions(
                context,
                momentum_config,
                ratio**2,
            )
            old_scalar_lm = jnp.where(
                first_update,
                scalar_config.initial_coefficient * scalar_mm,
                old_s.lm.payload,
            )
            old_scalar_mm = jnp.where(first_update, scalar_mm, old_s.mm.payload)
            old_scalar_qn = jnp.where(
                first_update,
                scalar_config.initial_coefficient * scalar_nn,
                old_s.qn.payload,
            )
            old_scalar_nn = jnp.where(first_update, scalar_nn, old_s.nn.payload)
            old_scalar_lm, old_scalar_mm, old_scalar_qn, old_scalar_nn = (
                _history_boundary(value)
                for value in (
                    old_scalar_lm,
                    old_scalar_mm,
                    old_scalar_qn,
                    old_scalar_nn,
                )
            )
            scalar_lm_avg, scalar_mm_avg = _lagrangian_average(
                scalar_lm,
                scalar_mm,
                old_scalar_lm,
                old_scalar_mm,
                trajectory_x,
                trajectory_y,
                trajectory_z,
                grid=momentum.velocity.x.ownership.grid,
                interval_dt=interval_dt,
                timescale_coefficient=momentum_config.timescale_coefficient,
                timescale_a=lm_avg,
                timescale_b=mm_avg,
            )
            scalar_qn_avg, scalar_nn_avg = _lagrangian_average(
                scalar_qn,
                scalar_nn,
                old_scalar_qn,
                old_scalar_nn,
                trajectory_x,
                trajectory_y,
                trajectory_z,
                grid=momentum.velocity.x.ownership.grid,
                interval_dt=interval_dt,
                timescale_coefficient=momentum_config.timescale_coefficient,
                timescale_a=qn_avg,
                timescale_b=nn_avg,
            )
            scalar_lm_avg = jnp.where(scalar_lm_avg > 0.0, scalar_lm_avg, 1.0e-32)
            scalar_qn_avg = jnp.where(scalar_qn_avg > 0.0, scalar_qn_avg, 1.0e-32)
            scalar_2d = jnp.maximum(_safe_divide(scalar_lm_avg, scalar_mm_avg), 0.0)
            scalar_4d = jnp.maximum(_safe_divide(scalar_qn_avg, scalar_nn_avg), 0.0)
            scalar_coefficient = jnp.clip(
                _safe_divide(
                    scalar_2d,
                    _lasd_beta(
                        scalar_2d,
                        scalar_4d,
                        momentum_config,
                        scale_dependent=scalar_config.scale_dependent,
                    ),
                ),
                scalar_config.minimum_coefficient,
                scalar_config.maximum_coefficient,
            )
            zero = jnp.zeros_like(trajectory_x)
            new_momentum = MomentumLasdMemory(
                field(old_m.coefficient, momentum_coefficient),
                field(old_m.lm, lm_avg),
                field(old_m.mm, mm_avg),
                field(old_m.qn, qn_avg),
                field(old_m.nn, nn_avg),
                field(old_m.trajectory_x, zero),
                field(old_m.trajectory_y, zero),
                field(old_m.trajectory_z, zero),
            )
            new_scalar = ScalarLasdMemory(
                field(old_s.coefficient, scalar_coefficient),
                field(old_s.lm, scalar_lm_avg),
                field(old_s.mm, scalar_mm_avg),
                field(old_s.qn, scalar_qn_avg),
                field(old_s.nn, scalar_nn_avg),
            )
        else:
            new_momentum = MomentumLasdMemory(
                old_m.coefficient,
                old_m.lm,
                old_m.mm,
                old_m.qn,
                old_m.nn,
                field(old_m.trajectory_x, trajectory_x),
                field(old_m.trajectory_y, trajectory_y),
                field(old_m.trajectory_z, trajectory_z),
            )
            new_scalar = old_s
        prepared = replace(
            fields,
            closure=LasdClosureMemory(new_momentum, new_scalar, fingerprint),
        )
        return prepared, LasdClosureEventDiagnostic(should_update, clock.step, interval)

    def momentum_context(
        self,
        context: ReferenceBoussinesqContext,
    ) -> ReferenceDryFlowContext:
        return context.momentum

    def buoyancy_tendency(
        self,
        context: ReferenceBoussinesqContext,
        config: LinearBoussinesqBuoyancy | NoBuoyancy,
    ) -> VelocityVector:
        momentum = context.momentum
        if isinstance(config, NoBuoyancy):
            return _reference_tendency(
                momentum,
                jnp.zeros_like(momentum.velocity.x.payload),
                jnp.zeros_like(momentum.velocity.y.payload),
                jnp.zeros_like(momentum.velocity.z.payload),
            )
        if not isinstance(config, LinearBoussinesqBuoyancy):
            raise TypeError("unsupported Boussinesq buoyancy choice")
        hydrostatic_free_theta = context.theta_on_faces - jnp.mean(
            context.theta_on_faces,
            axis=(-2, -1),
            keepdims=True,
        )
        z = config.acceleration_per_temperature * hydrostatic_free_theta
        z = z.at[0].set(0.0).at[-1].set(0.0)
        return _reference_tendency(
            momentum,
            jnp.zeros_like(momentum.velocity.x.payload),
            jnp.zeros_like(momentum.velocity.y.payload),
            z,
        )

    def rayleigh_damping_tendency(
        self,
        context: ReferenceBoussinesqContext,
        config: NoRayleighDamping | RayleighGeostrophicDamping,
    ) -> VelocityVector:
        momentum = context.momentum
        velocity = momentum.velocity
        if isinstance(config, NoRayleighDamping):
            return _reference_tendency(
                momentum,
                jnp.zeros_like(velocity.x.payload),
                jnp.zeros_like(velocity.y.payload),
                jnp.zeros_like(velocity.z.payload),
            )
        if not isinstance(config, RayleighGeostrophicDamping):
            raise TypeError("unsupported Rayleigh damping choice")
        grid = velocity.x.ownership.grid
        if config.start_height >= grid.lz:
            raise ValueError("Rayleigh damping must start below the domain top")
        dtype = velocity.x.payload.dtype
        depth = grid.lz - config.start_height
        cell_height = (jnp.arange(grid.nz, dtype=dtype) + 0.5) * grid.dz
        face_height = jnp.arange(grid.nz + 1, dtype=dtype) * grid.dz
        cell_eta = jnp.clip((cell_height - config.start_height) / depth, 0.0, 1.0)
        face_eta = jnp.clip((face_height - config.start_height) / depth, 0.0, 1.0)
        cell_rate = jnp.asarray(config.maximum_rate, dtype=dtype) * cell_eta**2
        face_rate = (
            jnp.asarray(
                config.maximum_rate,
                dtype=velocity.z.payload.dtype,
            )
            * face_eta.astype(velocity.z.payload.dtype) ** 2
        )
        return _reference_tendency(
            momentum,
            -cell_rate[:, None, None]
            * (velocity.x.payload - config.geostrophic_x_velocity),
            -cell_rate[:, None, None]
            * (velocity.y.payload - config.geostrophic_y_velocity),
            -face_rate[:, None, None] * velocity.z.payload,
        )

    def _reference_scalar_tendency(
        self,
        context: ReferenceBoussinesqContext,
        payload: Any,
    ) -> Field:
        scalar = context.potential_temperature
        quantity = (
            PotentialTemperatureTendency
            if scalar.quantity is PotentialTemperaturePerturbation
            else PassiveScalarTendency
        )
        return Field(
            quantity,
            Cell,
            scalar.ownership,
            Evaluated,
            payload.astype(scalar.payload.dtype),
        )

    def scalar_advection_tendency(
        self,
        context: ReferenceBoussinesqContext,
        config: ConservativeScalarAdvection,
    ) -> Field:
        if not isinstance(config, ConservativeScalarAdvection):
            raise TypeError("unsupported conservative scalar advection choice")
        momentum = context.momentum
        grid = context.potential_temperature.ownership.grid
        theta = context.potential_temperature.payload
        vertical_flux = _two_thirds_filter(
            momentum.velocity.z.payload * context.theta_on_faces,
            grid=grid,
        )
        tendency = -(
            _truncated_horizontal_derivative(
                momentum.velocity.x.payload * theta,
                grid=grid,
                axis="x",
            )
            + _truncated_horizontal_derivative(
                momentum.velocity.y.payload * theta,
                grid=grid,
                axis="y",
            )
            + (vertical_flux[1:] - vertical_flux[:-1]) / grid.dz
        )
        return self._reference_scalar_tendency(context, tendency)

    def scalar_sgs_tendency(
        self,
        context: ReferenceBoussinesqContext,
        momentum_config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
        config: StaticSmagorinskyScalarFlux | LagrangianScaleDependentScalarFlux,
        boundary: ScalarFluxBoundary = ScalarFluxBoundary(),
    ) -> Field:
        static = isinstance(momentum_config, StaticSmagorinsky) and isinstance(
            config,
            StaticSmagorinskyScalarFlux,
        )
        dynamic = isinstance(
            momentum_config,
            LagrangianScaleDependentDynamic,
        ) and isinstance(config, LagrangianScaleDependentScalarFlux)
        if not (static or dynamic):
            raise TypeError("unsupported or inconsistent scalar SGS choice")
        momentum = context.momentum
        grid = context.potential_temperature.ownership.grid
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        cell_magnitude = _strain_magnitude(
            momentum.dudx,
            momentum.dudy,
            momentum.dudz_at_cells,
            momentum.dvdx,
            momentum.dvdy,
            momentum.dvdz_at_cells,
            momentum.dwdx_at_cells,
            momentum.dwdy_at_cells,
            momentum.dwdz,
        )
        face_magnitude = _strain_magnitude(
            _cell_to_full_faces(momentum.dudx),
            _cell_to_full_faces(momentum.dudy),
            momentum.dudz_on_faces,
            _cell_to_full_faces(momentum.dvdx),
            _cell_to_full_faces(momentum.dvdy),
            momentum.dvdz_on_faces,
            momentum.dwdx_on_faces,
            momentum.dwdy_on_faces,
            _cell_to_full_faces(momentum.dwdz),
        )
        if static:
            scalar_coefficient = jnp.full_like(
                cell_magnitude,
                momentum_config.coefficient**2 / config.turbulent_prandtl,
            )
        else:
            closure = momentum.closure
            if not isinstance(closure, LasdClosureMemory):
                raise TypeError("scalar LASD requires initialized closure memory")
            scalar_coefficient = closure.scalar.coefficient.payload
        stability = jnp.ones_like(cell_magnitude)
        if dynamic and config.stability_buoyancy_coefficient > 0.0:
            n2 = jnp.maximum(
                config.stability_buoyancy_coefficient
                * _scalar_cell_gradient(context)[..., 2],
                0.0,
            )
            richardson = n2 / jnp.maximum(cell_magnitude**2, 1.0e-24)
            stability = (1.0 + config.stability_beta * richardson) ** (
                -config.stability_power
            )
        effective_scalar_coefficient = scalar_coefficient * stability
        cell_diffusivity = effective_scalar_coefficient * delta**2 * cell_magnitude
        face_diffusivity = (
            _cell_to_full_faces(effective_scalar_coefficient)
            * delta**2
            * face_magnitude
        )
        qx = -cell_diffusivity * context.dtheta_dx
        qy = -cell_diffusivity * context.dtheta_dy
        qz = -face_diffusivity * context.dtheta_dz_on_faces
        qz = qz.at[0].set(boundary.lower_flux).at[-1].set(boundary.upper_flux)
        qz = _two_thirds_filter(qz, grid=grid)
        tendency = -(
            _truncated_horizontal_derivative(qx, grid=grid, axis="x")
            + _truncated_horizontal_derivative(qy, grid=grid, axis="y")
            + (qz[1:] - qz[:-1]) / grid.dz
        )
        return self._reference_scalar_tendency(context, tendency)

    def lasd_diagnostic_fields(
        self,
        context: ReferenceBoussinesqContext,
        momentum_config: LagrangianScaleDependentDynamic,
        scalar_config: LagrangianScaleDependentScalarFlux,
        boundary: ScalarFluxBoundary = ScalarFluxBoundary(),
        constants: DiagnosticLasdConstants = DiagnosticLasdConstants(),
        wall: NeutralLogWall | FilteredNeutralLogWall | None = None,
    ) -> LasdDiagnosticFields:
        """Diagnose LASD energy/variance without adding prognostic state."""
        if not isinstance(
            momentum_config, LagrangianScaleDependentDynamic
        ) or not isinstance(
            scalar_config,
            LagrangianScaleDependentScalarFlux,
        ):
            raise TypeError("LASD diagnostics require momentum and scalar LASD")
        if not isinstance(constants, DiagnosticLasdConstants):
            raise TypeError("unsupported LASD diagnostic constants")
        closure = context.momentum.closure
        if not isinstance(closure, LasdClosureMemory):
            raise TypeError("LASD diagnostics require initialized closure memory")
        momentum = context.momentum
        grid = context.potential_temperature.ownership.grid
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        magnitude = _strain_magnitude(
            momentum.dudx,
            momentum.dudy,
            momentum.dudz_at_cells,
            momentum.dvdx,
            momentum.dvdy,
            momentum.dvdz_at_cells,
            momentum.dwdx_at_cells,
            momentum.dwdy_at_cells,
            momentum.dwdz,
        )
        diagnostic_magnitude = magnitude
        wall_gradient_factor = None
        if wall is not None:
            if not isinstance(
                wall,
                (NeutralLogWall, FilteredNeutralLogWall),
            ):
                raise TypeError("LASD diagnostic wall must be NeutralLogWall")
            reference_height = 0.5 * grid.dz
            if wall.roughness_length >= reference_height:
                raise ValueError("wall roughness must be below the first cell centre")
            wall_gradient_factor = 1.0 / (
                math.log(reference_height / wall.roughness_length) * reference_height
            )
            diagnostic_dudz = momentum.dudz_at_cells.at[0].set(
                momentum.velocity.x.payload[0] * wall_gradient_factor
            )
            diagnostic_dvdz = momentum.dvdz_at_cells.at[0].set(
                momentum.velocity.y.payload[0] * wall_gradient_factor
            )
            diagnostic_magnitude = _strain_magnitude(
                momentum.dudx,
                momentum.dudy,
                diagnostic_dudz,
                momentum.dvdx,
                momentum.dvdy,
                diagnostic_dvdz,
                momentum.dwdx_at_cells,
                momentum.dwdy_at_cells,
                momentum.dwdz,
            )
        face_magnitude = _strain_magnitude(
            _cell_to_full_faces(momentum.dudx),
            _cell_to_full_faces(momentum.dudy),
            momentum.dudz_on_faces,
            _cell_to_full_faces(momentum.dvdx),
            _cell_to_full_faces(momentum.dvdy),
            momentum.dvdz_on_faces,
            momentum.dwdx_on_faces,
            momentum.dwdy_on_faces,
            _cell_to_full_faces(momentum.dwdz),
        )
        momentum_diffusivity = (
            closure.momentum.coefficient.payload * delta**2 * magnitude
        )
        scalar_coefficient = closure.scalar.coefficient.payload
        stability = jnp.ones_like(magnitude)
        if scalar_config.stability_buoyancy_coefficient > 0.0:
            n2 = jnp.maximum(
                scalar_config.stability_buoyancy_coefficient
                * _scalar_cell_gradient(context)[..., 2],
                0.0,
            )
            richardson = n2 / jnp.maximum(magnitude**2, 1.0e-24)
            stability = (1.0 + scalar_config.stability_beta * richardson) ** (
                -scalar_config.stability_power
            )
        effective_scalar_coefficient = scalar_coefficient * stability
        scalar_diffusivity = effective_scalar_coefficient * delta**2 * magnitude
        face_diffusivity = (
            _cell_to_full_faces(effective_scalar_coefficient)
            * delta**2
            * face_magnitude
        )
        diagnostic_face_diffusivity = face_diffusivity
        if wall_gradient_factor is not None:
            zero_wall_cross_gradient = jnp.zeros_like(momentum.dwdx_on_faces[:1])
            wall_face_magnitude = _strain_magnitude(
                momentum.dudx[:1],
                momentum.dudy[:1],
                momentum.velocity.x.payload[:1] * wall_gradient_factor,
                momentum.dvdx[:1],
                momentum.dvdy[:1],
                momentum.velocity.y.payload[:1] * wall_gradient_factor,
                zero_wall_cross_gradient,
                zero_wall_cross_gradient,
                momentum.dwdz[:1],
            )
            wall_scalar_diffusivity = (
                effective_scalar_coefficient[:1]
                * delta**2
                * wall_face_magnitude
            )
            if constants.horizontal_homogeneous_wall:
                wall_scalar_diffusivity = jnp.full_like(
                    wall_scalar_diffusivity,
                    jnp.mean(wall_scalar_diffusivity),
                )
            diagnostic_face_diffusivity = face_diffusivity.at[0].set(
                wall_scalar_diffusivity[0]
            )
        flux_x = -scalar_diffusivity * context.dtheta_dx
        flux_y = -scalar_diffusivity * context.dtheta_dy
        flux_z = -face_diffusivity * context.dtheta_dz_on_faces
        flux_z = flux_z.at[0].set(boundary.lower_flux).at[-1].set(boundary.upper_flux)
        flux_z = _two_thirds_filter(flux_z, grid=grid)

        shear_production = momentum_diffusivity * diagnostic_magnitude**2
        buoyancy_destruction = (
            scalar_diffusivity
            * scalar_config.stability_buoyancy_coefficient
            * _scalar_cell_gradient(context)[..., 2]
        )
        sgs_tke = jnp.maximum(
            (shear_production - buoyancy_destruction)
            * delta
            / constants.sgs_dissipation_coefficient,
            0.0,
        ) ** (2.0 / 3.0)
        diagnostic_gradient_faces = (
            context.dtheta_dz_on_faces.at[0]
            .set(
                jnp.where(
                    diagnostic_face_diffusivity[0] > 0.0,
                    -flux_z[0] / diagnostic_face_diffusivity[0],
                    0.0,
                )
            )
            .at[-1]
            .set(
                jnp.where(
                    face_diffusivity[-1] > 0.0,
                    -flux_z[-1] / face_diffusivity[-1],
                    0.0,
                )
            )
        )
        gradient_z = 0.5 * (
            diagnostic_gradient_faces[:-1] + diagnostic_gradient_faces[1:]
        )
        flux_z_at_cells = 0.5 * (flux_z[:-1] + flux_z[1:])
        scalar_dissipation = -(
            flux_x * context.dtheta_dx
            + flux_y * context.dtheta_dy
            + flux_z_at_cells * gradient_z
        )
        scalar_length = delta * jnp.sqrt(
            jnp.maximum(effective_scalar_coefficient, 0.0)
        )
        sqrt_tke = jnp.sqrt(jnp.maximum(sgs_tke, 0.0))
        valid = sqrt_tke > jnp.finfo(sqrt_tke.dtype).tiny
        scalar_variance_numerator = (
            2.0
            * scalar_length
            * scalar_dissipation
            / constants.scalar_variance_coefficient
        )
        scalar_variance = jnp.where(
            valid,
            scalar_variance_numerator / jnp.where(valid, sqrt_tke, 1.0),
            0.0,
        )
        scalar_variance = jnp.maximum(scalar_variance, 0.0)
        return LasdDiagnosticFields(
            momentum_diffusivity,
            scalar_diffusivity,
            flux_x,
            flux_y,
            flux_z,
            sgs_tke,
            scalar_variance_numerator,
            scalar_variance,
        )

    def combine_scalar_tendencies(self, tendencies: tuple[Field, ...]) -> Field:
        if not tendencies:
            raise ValueError("at least one scalar tendency is required")
        first = tendencies[0]
        for tendency in tendencies:
            _require_tiny_global(tendency, Cell)
            if (
                tendency.quantity
                not in (PotentialTemperatureTendency, PassiveScalarTendency)
                or tendency.phase is not Evaluated
            ):
                raise TypeError("only evaluated scalar tendencies may be combined")
            if tendency.ownership != first.ownership:
                raise ValueError("combined scalar tendencies must share ownership")
            if tendency.payload.dtype != first.payload.dtype:
                raise TypeError("combined scalar tendencies must share one dtype")
        return Field(
            first.quantity,
            Cell,
            first.ownership,
            Evaluated,
            sum(
                (term.payload for term in tendencies),
                jnp.zeros_like(first.payload),
            ),
        )

    def advection_tendency(
        self,
        context: ReferenceDryFlowContext,
        config: ConservativeAdvection,
    ) -> VelocityVector:
        if not isinstance(config, ConservativeAdvection):
            raise TypeError("unsupported reference advection choice")
        velocity = context.velocity
        grid = velocity.x.ownership.grid
        u = velocity.x.payload
        v = velocity.y.payload
        w = velocity.z.payload
        vertical_u_flux = _two_thirds_filter(
            w * context.u_on_faces,
            grid=grid,
        )
        vertical_v_flux = _two_thirds_filter(
            w * context.v_on_faces,
            grid=grid,
        )
        x_tendency = -(
            _truncated_horizontal_derivative(u * u, grid=grid, axis="x")
            + _truncated_horizontal_derivative(v * u, grid=grid, axis="y")
            + (vertical_u_flux[1:] - vertical_u_flux[:-1]) / grid.dz
        )
        y_tendency = -(
            _truncated_horizontal_derivative(u * v, grid=grid, axis="x")
            + _truncated_horizontal_derivative(v * v, grid=grid, axis="y")
            + (vertical_v_flux[1:] - vertical_v_flux[:-1]) / grid.dz
        )
        vertical_w_flux = _two_thirds_filter(
            context.w_at_cells * context.w_at_cells,
            grid=grid,
        )
        vertical_w_derivative = _cell_gradient_on_full_faces(
            vertical_w_flux,
            grid.dz,
        )
        z_tendency = -(
            _truncated_horizontal_derivative(
                context.u_on_faces * w,
                grid=grid,
                axis="x",
            )
            + _truncated_horizontal_derivative(
                context.v_on_faces * w,
                grid=grid,
                axis="y",
            )
            + vertical_w_derivative
        )
        z_tendency = z_tendency.at[0].set(0.0).at[-1].set(0.0)
        return _reference_tendency(context, x_tendency, y_tendency, z_tendency)

    def pressure_gradient_tendency(
        self,
        context: ReferenceDryFlowContext,
        config: KinematicPressureGradient,
    ) -> VelocityVector:
        if not isinstance(config, KinematicPressureGradient):
            raise TypeError("unsupported pressure-gradient forcing choice")
        velocity = context.velocity
        x = jnp.full_like(velocity.x.payload, config.x_acceleration)
        y = jnp.full_like(velocity.y.payload, config.y_acceleration)
        z = jnp.zeros_like(velocity.z.payload)
        return _reference_tendency(context, x, y, z)

    def wall_stress_tendency(
        self,
        context: ReferenceDryFlowContext,
        config: NeutralLogWall | FilteredNeutralLogWall,
    ) -> VelocityVector:
        if not isinstance(
            config,
            (NeutralLogWall, FilteredNeutralLogWall),
        ):
            raise TypeError("unsupported wall-stress choice")
        velocity = context.velocity
        grid = velocity.x.ownership.grid
        reference_height = 0.5 * grid.dz
        if config.roughness_length >= reference_height:
            raise ValueError("wall roughness must be below the first cell centre")
        drag = (
            config.von_karman / math.log(reference_height / config.roughness_length)
        ) ** 2
        u0 = velocity.x.payload[0]
        v0 = velocity.y.payload[0]
        if isinstance(config, FilteredNeutralLogWall):
            width = config.filter_grid_ratio * config.test_filter_ratio
            filtered = _wall_filter(
                jnp.stack((u0, v0)),
                grid=grid,
                filter_width=width,
            )
            u0, v0 = filtered[0], filtered[1]
        speed = jnp.sqrt(u0 * u0 + v0 * v0)
        x = jnp.zeros_like(velocity.x.payload).at[0].set(-drag * speed * u0 / grid.dz)
        y = jnp.zeros_like(velocity.y.payload).at[0].set(-drag * speed * v0 / grid.dz)
        z = jnp.zeros_like(velocity.z.payload)
        return _reference_tendency(context, x, y, z)

    def sgs_tendency(
        self,
        context: ReferenceDryFlowContext,
        config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
    ) -> VelocityVector:
        if not isinstance(config, (StaticSmagorinsky, LagrangianScaleDependentDynamic)):
            raise TypeError("unsupported SGS choice")
        velocity = context.velocity
        grid = velocity.x.ownership.grid
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        magnitude = _strain_magnitude(
            context.dudx,
            context.dudy,
            context.dudz_at_cells,
            context.dvdx,
            context.dvdy,
            context.dvdz_at_cells,
            context.dwdx_at_cells,
            context.dwdy_at_cells,
            context.dwdz,
        )
        coefficient = self._momentum_sgs_coefficient(context, config)
        eddy_viscosity = coefficient * delta**2 * magnitude
        txx = -2.0 * eddy_viscosity * context.dudx
        txy = -eddy_viscosity * (context.dudy + context.dvdx)
        tyy = -2.0 * eddy_viscosity * context.dvdy
        tzz = -2.0 * eddy_viscosity * context.dwdz
        txz, tyz = self.sgs_vertical_flux(context, config)
        tzz = _two_thirds_filter(tzz, grid=grid)
        x = -(
            _truncated_horizontal_derivative(txx, grid=grid, axis="x")
            + _truncated_horizontal_derivative(txy, grid=grid, axis="y")
            + (txz[1:] - txz[:-1]) / grid.dz
        )
        y = -(
            _truncated_horizontal_derivative(txy, grid=grid, axis="x")
            + _truncated_horizontal_derivative(tyy, grid=grid, axis="y")
            + (tyz[1:] - tyz[:-1]) / grid.dz
        )
        z = -(
            _truncated_horizontal_derivative(txz, grid=grid, axis="x")
            + _truncated_horizontal_derivative(tyz, grid=grid, axis="y")
            + _cell_gradient_on_full_faces(tzz, grid.dz)
        )
        z = z.at[0].set(0.0).at[-1].set(0.0)
        return _reference_tendency(context, x, y, z)

    def sgs_vertical_flux(
        self,
        context: ReferenceDryFlowContext,
        config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
    ) -> tuple[Any, Any]:
        """Return filtered SGS xz and yz stresses on full vertical faces."""
        if not isinstance(config, (StaticSmagorinsky, LagrangianScaleDependentDynamic)):
            raise TypeError("unsupported SGS choice")
        grid = context.velocity.x.ownership.grid
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        face_magnitude = _strain_magnitude(
            _cell_to_full_faces(context.dudx),
            _cell_to_full_faces(context.dudy),
            context.dudz_on_faces,
            _cell_to_full_faces(context.dvdx),
            _cell_to_full_faces(context.dvdy),
            context.dvdz_on_faces,
            context.dwdx_on_faces,
            context.dwdy_on_faces,
            _cell_to_full_faces(context.dwdz),
        )
        coefficient = self._momentum_sgs_coefficient(context, config)
        viscosity_on_faces = (
            _cell_to_full_faces(coefficient) * delta**2 * face_magnitude
        )
        txz = -viscosity_on_faces * (context.dudz_on_faces + context.dwdx_on_faces)
        tyz = -viscosity_on_faces * (context.dvdz_on_faces + context.dwdy_on_faces)
        txz = txz.at[0].set(0.0).at[-1].set(0.0)
        tyz = tyz.at[0].set(0.0).at[-1].set(0.0)
        txz = _two_thirds_filter(txz, grid=grid)
        tyz = _two_thirds_filter(tyz, grid=grid)
        return txz, tyz

    @staticmethod
    def _momentum_sgs_coefficient(
        context: ReferenceDryFlowContext,
        config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
    ):
        if isinstance(config, StaticSmagorinsky):
            return jnp.full_like(context.dudx, config.coefficient**2)
        closure = context.closure
        if not isinstance(closure, LasdClosureMemory):
            raise TypeError("momentum LASD requires initialized closure memory")
        return closure.momentum.coefficient.payload

    def coriolis_geostrophic_tendency(
        self,
        context: ReferenceDryFlowContext,
        config: NoRotation | CoriolisGeostrophic,
    ) -> VelocityVector:
        velocity = context.velocity
        if isinstance(config, NoRotation):
            x = jnp.zeros_like(velocity.x.payload)
            y = jnp.zeros_like(velocity.y.payload)
        elif isinstance(config, CoriolisGeostrophic):
            local_f = jnp.asarray(
                config.coriolis_parameter,
                dtype=velocity.x.payload.dtype,
            )
            horizontal_f = jnp.asarray(
                config.horizontal_coriolis_parameter,
                dtype=velocity.x.payload.dtype,
            )
            x = (
                local_f * (velocity.y.payload - config.geostrophic_y_velocity)
                - horizontal_f * context.w_at_cells
            )
            y = -local_f * (velocity.x.payload - config.geostrophic_x_velocity)
            z = horizontal_f.astype(velocity.z.payload.dtype) * (
                context.u_on_faces - config.geostrophic_x_velocity
            )
            z = z.at[0].set(0.0).at[-1].set(0.0)
        else:
            raise TypeError("unsupported rotation choice")
        if isinstance(config, NoRotation):
            z = jnp.zeros_like(velocity.z.payload)
        return _reference_tendency(
            context,
            x,
            y,
            z,
        )

    def wind_tunnel_tendency(
        self,
        velocity: VelocityVector,
        model: WindTunnelModel,
        environment: Any,
    ) -> VelocityVector:
        """Evaluate pure-thrust ADM and concurrent fringe on a tiny grid."""
        if not isinstance(model, WindTunnelModel):
            raise TypeError("unsupported wind-tunnel model")
        x_ownership = _require_velocity_component(velocity.x, XVelocity)
        y_ownership = _require_velocity_component(velocity.y, YVelocity)
        z_ownership = _require_tiny_global(velocity.z, ZFace)
        if not (x_ownership.grid == y_ownership.grid == z_ownership.grid):
            raise ValueError("wind-tunnel velocity components must share ownership")
        grid = x_ownership.grid
        source_x = jnp.zeros_like(velocity.x.payload)
        source_y = jnp.zeros_like(velocity.y.payload)
        source_z = jnp.zeros_like(velocity.z.payload)

        disk = model.actuator_disk
        if isinstance(disk, PureThrustActuatorDisk):
            dtype = velocity.x.payload.dtype
            x = (jnp.arange(grid.nx, dtype=dtype) + 0.5) * grid.dx
            y = (jnp.arange(grid.ny, dtype=dtype) + 0.5) * grid.dy
            z = (jnp.arange(grid.nz, dtype=dtype) + 0.5) * grid.dz
            dx = jnp.mod(x - disk.x + 0.5 * grid.lx, grid.lx) - 0.5 * grid.lx
            dy = jnp.mod(y - disk.y + 0.5 * grid.ly, grid.ly) - 0.5 * grid.ly
            yaw = jnp.deg2rad(jnp.asarray(disk.yaw_degrees, dtype=dtype))
            normal_x = jnp.cos(yaw)
            normal_y = jnp.sin(yaw)
            normal_distance = (
                dx[None, None, :] * normal_x
                + dy[None, :, None] * normal_y
            )
            in_plane = (
                -dx[None, None, :] * normal_y
                + dy[None, :, None] * normal_x
            )
            radius = jnp.sqrt(
                in_plane**2 + (z[:, None, None] - disk.z) ** 2
            )
            streamwise = jnp.exp(
                -(normal_distance / disk.normal_smoothing_width) ** 2
            )
            radial = gaussian_convolved_annulus(
                radius,
                outer_radius=0.5 * disk.diameter,
                inner_radius=0.5 * disk.hub_diameter,
                smoothing_width=disk.transverse_smoothing_width,
            )
            kernel = radial * streamwise
            disk_area = 0.25 * jnp.pi * (
                disk.diameter**2 - disk.hub_diameter**2
            )
            kernel_integral = jnp.sum(kernel) * grid.dx * grid.dy * grid.dz
            kernel = kernel * disk_area / jnp.maximum(
                kernel_integral,
                jnp.finfo(dtype).tiny,
            )
            normal_velocity = (
                velocity.x.payload * normal_x + velocity.y.payload * normal_y
            )
            disk_velocity = jnp.sum(normal_velocity * kernel) / jnp.maximum(
                jnp.sum(kernel), jnp.finfo(dtype).tiny
            )
            correction = jnp.where(
                disk.filtered_velocity_correction,
                filtered_disk_velocity_correction(
                    disk.thrust_coefficient_prime,
                    outer_radius=0.5 * disk.diameter,
                    inner_radius=0.5 * disk.hub_diameter,
                    smoothing_width=disk.transverse_smoothing_width,
                    dtype=dtype,
                ),
                1.0,
            )
            disk_velocity = correction * disk_velocity
            acceleration = (
                -0.5
                * disk.thrust_coefficient_prime
                * disk_velocity
                * jnp.abs(disk_velocity)
                * kernel
            )
            source_x = source_x + acceleration * normal_x
            source_y = source_y + acceleration * normal_y
        elif not isinstance(disk, NoActuatorDisk):
            raise TypeError("unsupported actuator-disk choice")

        fringe = model.fringe
        if isinstance(fringe, ConcurrentPrecursorFringe):
            if not isinstance(environment, ConcurrentPrecursorEnvironment):
                raise TypeError(
                    "concurrent fringe requires ConcurrentPrecursorEnvironment"
                )
            target = environment.velocity
            target_x = _require_velocity_component(target.x, XVelocity)
            target_y = _require_velocity_component(target.y, YVelocity)
            target_z = _require_tiny_global(target.z, ZFace)
            if not (
                target_x.grid
                == target_y.grid
                == target_z.grid
                == x_ownership.grid
            ):
                raise ValueError("precursor target must share main-domain ownership")
            dtype = velocity.x.payload.dtype
            x = (jnp.arange(grid.nx, dtype=dtype) + 0.5) * grid.dx
            half_width = 0.5 * (grid.lx - fringe.start_x)
            if half_width <= 0.0:
                raise ValueError("fringe start must lie before the periodic seam")

            def cinf_step(coordinate):
                epsilon = jnp.finfo(dtype).eps
                safe = jnp.clip(coordinate, epsilon, 1.0 - epsilon)
                interior = jax.nn.sigmoid(1.0 / (1.0 - safe) - 1.0 / safe)
                return jnp.where(
                    coordinate <= 0.0,
                    0.0,
                    jnp.where(coordinate >= 1.0, 1.0, interior),
                )

            mask = cinf_step((x - fringe.start_x) / half_width) * cinf_step(
                (grid.lx - x) / half_width
            )
            rate = mask / fringe.relaxation_time
            source_x = source_x + rate[None, None, :] * (
                target.x.payload - velocity.x.payload
            )
            source_y = source_y + rate[None, None, :] * (
                target.y.payload - velocity.y.payload
            )
            source_z = source_z + rate[None, None, :] * (
                target.z.payload - velocity.z.payload
            )
        elif not isinstance(fringe, NoFringe):
            raise TypeError("unsupported wind-tunnel fringe choice")

        return _reference_tendency_from_velocity(
            velocity,
            source_x,
            source_y,
            source_z,
        )

    def combine_tendencies(
        self,
        tendencies: tuple[VelocityVector, ...],
    ) -> VelocityVector:
        if not tendencies:
            raise ValueError("at least one evaluated tendency is required")
        first = tendencies[0]
        expected_components = (
            (first.x, XVelocityTendency, Cell),
            (first.y, YVelocityTendency, Cell),
            (first.z, VerticalVelocityTendency, ZFace),
        )
        for tendency in tendencies:
            for component, expected in zip(
                (tendency.x, tendency.y, tendency.z),
                expected_components,
                strict=True,
            ):
                first_component, quantity, location = expected
                _require_tiny_global(component, location)
                if (
                    component.quantity is not quantity
                    or component.phase is not Evaluated
                ):
                    raise TypeError(
                        "only evaluated velocity tendencies may be combined"
                    )
                if component.ownership != first_component.ownership:
                    raise ValueError("combined tendencies must share one ownership")
                if component.payload.dtype != first_component.payload.dtype:
                    raise TypeError("combined tendencies must share one dtype")
        velocity = VelocityVector(
            Field(
                XVelocity,
                Cell,
                first.x.ownership,
                Projected,
                jnp.zeros_like(first.x.payload),
            ),
            Field(
                YVelocity,
                Cell,
                first.y.ownership,
                Projected,
                jnp.zeros_like(first.y.payload),
            ),
            Field(
                VerticalVelocity,
                ZFace,
                first.z.ownership,
                Projected,
                jnp.zeros_like(first.z.payload),
            ),
        )
        return _reference_tendency_from_velocity(
            velocity,
            sum(
                (term.x.payload for term in tendencies), jnp.zeros_like(first.x.payload)
            ),
            sum(
                (term.y.payload for term in tendencies), jnp.zeros_like(first.y.payload)
            ),
            sum(
                (term.z.payload for term in tendencies), jnp.zeros_like(first.z.payload)
            ),
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
class JaxReferencePressureSolver:
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
