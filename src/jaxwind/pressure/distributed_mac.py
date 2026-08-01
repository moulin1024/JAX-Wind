"""Y-slab distributed MAC projection using the matrix-free GMG solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp

from .distributed_gmg import YSlabMatrixFreePoissonSolver, shard_y_field
from .fgmres import FGMRESResult
from .mac_projection import MACVelocity, _gradient_axis
from .pcg import PCGResult


Array = jax.Array


class YSlabMACVelocity(NamedTuple):
    """Device-leading MAC velocity with duplicated inter-slab y faces."""

    x: Array  # (device, nz, local_y, nx + 1)
    y: Array  # (device, nz, local_y + 1, nx)
    z: Array  # (device, nz + 1, local_y, nx)


@dataclass(frozen=True, slots=True)
class YSlabMACProjectionResult:
    velocity: YSlabMACVelocity
    pressure_correction: Array
    divergence_before: Array
    divergence_after: Array
    target_divergence: Array
    linear_result: FGMRESResult | PCGResult


def shard_y_mac_velocity(
    velocity: MACVelocity,
    device_count: int,
) -> YSlabMACVelocity:
    nz, ny, nx_plus_one = velocity.x.shape
    nx = nx_plus_one - 1
    if velocity.y.shape != (nz, ny + 1, nx):
        raise ValueError("global y-face velocity shape is inconsistent")
    if velocity.z.shape != (nz + 1, ny, nx):
        raise ValueError("global z-face velocity shape is inconsistent")
    if ny % device_count:
        raise ValueError("global y cells must divide across devices")
    local_y = ny // device_count
    y_slabs = tuple(
        velocity.y[
            :,
            device * local_y : (device + 1) * local_y + 1,
            :,
        ]
        for device in range(device_count)
    )
    return YSlabMACVelocity(
        shard_y_field(velocity.x, device_count),
        jnp.stack(y_slabs),
        shard_y_field(velocity.z, device_count),
    )


def gather_y_mac_velocity(velocity: YSlabMACVelocity) -> MACVelocity:
    devices = velocity.x.shape[0]
    x = jnp.reshape(
        jnp.transpose(velocity.x, (1, 0, 2, 3)),
        (
            velocity.x.shape[1],
            devices * velocity.x.shape[2],
            velocity.x.shape[3],
        ),
    )
    z = jnp.reshape(
        jnp.transpose(velocity.z, (1, 0, 2, 3)),
        (
            velocity.z.shape[1],
            devices * velocity.z.shape[2],
            velocity.z.shape[3],
        ),
    )
    y = jnp.concatenate(
        tuple(velocity.y[device, :, :-1, :] for device in range(devices))
        + (velocity.y[-1, :, -1:, :],),
        axis=1,
    )
    return MACVelocity(x, y, z)


class YSlabMACProjector:
    """Project device-leading y-slab MAC stages."""

    def __init__(self, solver: YSlabMatrixFreePoissonSolver) -> None:
        self.solver = solver
        mapped = solver.pmap_options
        self._mapped_divergence = jax.pmap(
            self._divergence_local,
            **mapped,
        )
        self._mapped_gradient = jax.pmap(
            self._gradient_local,
            **mapped,
        )

    def _divergence_local(self, velocity: YSlabMACVelocity) -> Array:
        operator = self.solver.operator
        wx, wy, wz = operator._level.widths
        local_y = operator.shape[1] // self.solver.device_count
        start = self.solver._rank() * local_y
        local_wy = lax.dynamic_slice_in_dim(wy, start, local_y)
        return (
            (velocity.x[..., 1:] - velocity.x[..., :-1])
            / wx[None, None, :]
            + (velocity.y[:, 1:, :] - velocity.y[:, :-1, :])
            / local_wy[None, :, None]
            + (velocity.z[1:, ...] - velocity.z[:-1, ...])
            / wz[:, None, None]
        )

    def _gradient_local(self, pressure: Array) -> YSlabMACVelocity:
        operator = self.solver.operator
        boundaries = operator.boundaries
        x_gradient = _gradient_axis(
            pressure,
            axis=-1,
            faces=operator.grid.x_faces,
            lower_kind=boundaries.x_lower.kind,
            lower_value=boundaries.x_lower.value,
            upper_kind=boundaries.x_upper.kind,
            upper_value=boundaries.x_upper.value,
        )
        z_gradient = _gradient_axis(
            pressure,
            axis=-3,
            faces=operator.grid.z_faces,
            lower_kind=boundaries.z_lower.kind,
            lower_value=boundaries.z_lower.value,
            upper_kind=boundaries.z_upper.kind,
            upper_value=boundaries.z_upper.value,
        )
        _, centers = operator._level.widths[1], operator._level.centers[1]
        global_y = operator.shape[1]
        local_y = global_y // self.solver.device_count
        rank = self.solver._rank()
        start = rank * local_y
        local_centers = lax.dynamic_slice_in_dim(centers, start, local_y)
        gradient = jnp.zeros(
            (pressure.shape[0], local_y + 1, pressure.shape[2]),
            dtype=pressure.dtype,
        )
        if local_y > 1:
            gradient = gradient.at[:, 1:-1, :].set(
                (pressure[:, 1:, :] - pressure[:, :-1, :])
                / (local_centers[1:] - local_centers[:-1])[None, :, None]
            )
        periodic = boundaries.y_lower.kind == "periodic"
        left, right = self.solver._exchange_y(pressure, periodic)
        previous = lax.dynamic_index_in_dim(
            centers,
            jnp.maximum(start - 1, 0),
            keepdims=False,
        )
        following = lax.dynamic_index_in_dim(
            centers,
            jnp.minimum(start + local_y, global_y - 1),
            keepdims=False,
        )
        lower_internal = (
            pressure[:, 0, :] - left
        ) / (local_centers[0] - previous)
        upper_internal = (
            right - pressure[:, -1, :]
        ) / (following - local_centers[-1])
        if periodic:
            wy = operator._level.widths[1]
            wrap = 0.5 * (wy[0] + wy[-1])
            lower_physical = (pressure[:, 0, :] - left) / wrap
            upper_physical = (right - pressure[:, -1, :]) / wrap
        else:
            if boundaries.y_lower.kind == "dirichlet":
                lower_physical = (
                    pressure[:, 0, :] - boundaries.y_lower.value
                ) / (local_centers[0] - operator.grid.y_faces[0])
            else:
                lower_physical = jnp.full_like(
                    pressure[:, 0, :],
                    -boundaries.y_lower.value,
                )
            if boundaries.y_upper.kind == "dirichlet":
                upper_physical = (
                    boundaries.y_upper.value - pressure[:, -1, :]
                ) / (operator.grid.y_faces[-1] - local_centers[-1])
            else:
                upper_physical = jnp.full_like(
                    pressure[:, -1, :],
                    boundaries.y_upper.value,
                )
        gradient = gradient.at[:, 0, :].set(
            jnp.where(rank > 0, lower_internal, lower_physical)
        )
        gradient = gradient.at[:, -1, :].set(
            jnp.where(
                rank < self.solver.device_count - 1,
                upper_internal,
                upper_physical,
            )
        )
        return YSlabMACVelocity(x_gradient, gradient, z_gradient)

    def _check_velocity(self, velocity: YSlabMACVelocity) -> None:
        local_devices = self.solver.local_device_count
        devices = self.solver.device_count
        nz, ny, nx = self.solver.operator.shape
        local_y = ny // devices
        expected = (
            (local_devices, nz, local_y, nx + 1),
            (local_devices, nz, local_y + 1, nx),
            (local_devices, nz + 1, local_y, nx),
        )
        actual = tuple(tuple(component.shape) for component in velocity)
        if actual != expected:
            raise ValueError(f"expected y-slab MAC shapes {expected}, got {actual}")

    def divergence(self, velocity: YSlabMACVelocity) -> Array:
        self._check_velocity(velocity)
        return self._mapped_divergence(velocity)

    def gradient(self, pressure: Array) -> YSlabMACVelocity:
        self.solver._check_sharded_shape(pressure)
        return self._mapped_gradient(pressure)

    def project(
        self,
        velocity: YSlabMACVelocity,
        *,
        timestep: float,
        target_divergence: Array | float = 0.0,
        initial_pressure: Array | None = None,
    ) -> YSlabMACProjectionResult:
        if not (timestep > 0.0 and timestep < float("inf")):
            raise ValueError("projection timestep must be positive and finite")
        divergence_before = self.divergence(velocity)
        target = jnp.broadcast_to(
            jnp.asarray(target_divergence, dtype=divergence_before.dtype),
            divergence_before.shape,
        )
        linear_result = self.solver.solve(
            (target - divergence_before) / timestep,
            initial=initial_pressure,
        )
        gradient = self.gradient(linear_result.solution)
        corrected = YSlabMACVelocity(
            velocity.x - timestep * gradient.x,
            velocity.y - timestep * gradient.y,
            velocity.z - timestep * gradient.z,
        )
        divergence_after = self.divergence(corrected)
        return YSlabMACProjectionResult(
            corrected,
            linear_result.solution,
            divergence_before,
            divergence_after,
            target,
            linear_result,
        )


__all__ = [
    "YSlabMACProjectionResult",
    "YSlabMACProjector",
    "YSlabMACVelocity",
    "gather_y_mac_velocity",
    "shard_y_mac_velocity",
]
