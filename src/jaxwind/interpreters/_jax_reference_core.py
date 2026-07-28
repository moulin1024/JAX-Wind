"""Numerical kernels for the bounded global JAX reference oracle.

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

from jaxwind.interpreters.jax_actuator_disk import (
    filtered_disk_velocity_correction,
    gaussian_convolved_annulus,
)
from jaxwind.interpreters.jax_actuator_line import (
    actuator_line_deformed_kinematics,
    blade_element_kinematic_forces,
    gaussian_weights,
)
from jaxwind.interpreters.jax_fringe import plateau_fringe_mask

from jaxwind.domain import (
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
from jaxwind.operators import PressureGradient, VelocityVector
from jaxwind.physics.dry_flow import (
    ConservativeAdvection,
    CoriolisGeostrophic,
    FilteredNeutralLogWall,
    KinematicPressureGradient,
    NeutralLogWall,
    NoRotation,
    StaticSmagorinsky,
)
from jaxwind.physics.boussinesq import (
    BoussinesqFields,
    ConservativeScalarAdvection,
    LinearBoussinesqBuoyancy,
    NoBuoyancy,
    NoRayleighDamping,
    RayleighGeostrophicDamping,
    ScalarFluxBoundary,
    StaticSmagorinskyScalarFlux,
)
from jaxwind.physics.lasd import (
    DiagnosticLasdConstants,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LasdClosureEventDiagnostic,
    LasdClosureMemory,
    LasdDiagnosticFields,
    MomentumLasdMemory,
    ScalarLasdMemory,
)
from jaxwind.physics.wind_tunnel import (
    BladeElementActuatorLine,
    ConcurrentPrecursorEnvironment,
    ConcurrentPrecursorFringe,
    NoActuatorDisk,
    NoActuatorLine,
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

