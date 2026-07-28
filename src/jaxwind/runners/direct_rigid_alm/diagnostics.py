"""Initialization and diagnostic output for direct ALM runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .models import CaseConfig


def _initial_velocity(
    *,
    jnp,
    case: CaseConfig,
    decomposition,
    addressable_shards: tuple[int, ...],
    scales,
):
    from jaxwind.domain import (
        AddressableField,
        Cell,
        Projected,
        VerticalVelocity,
        XVelocity,
        YVelocity,
        ZFace,
    )
    from jaxwind.interpreters.jax_zslab import ZFaceFieldContext
    from jaxwind.operators import VelocityVector

    dtype = getattr(jnp, case.numerics.dtype)
    domain = case.domain
    local_nz = domain.nz // decomposition.shard_count
    payload_shape = (
        len(addressable_shards),
        local_nz,
        domain.ny,
        domain.nx,
    )
    z_m = (jnp.arange(domain.nz, dtype=dtype) + 0.5) * domain.dz_m
    u_m_s = (
        case.flow.friction_velocity_m_s
        / case.flow.von_karman
        * jnp.log(z_m / case.flow.roughness_length_m)
    )
    u = jnp.broadcast_to(
        u_m_s[:, None, None],
        (domain.nz, domain.ny, domain.nx),
    )
    u = scales.to_execution_velocity(u).reshape(payload_shape)
    zero = jnp.zeros(payload_shape, dtype=dtype)
    cell_regions = tuple(
        decomposition.regions(Cell)[index]
        for index in addressable_shards
    )
    face_regions = tuple(
        decomposition.regions(ZFace)[index]
        for index in addressable_shards
    )
    return VelocityVector(
        AddressableField(
            XVelocity,
            Cell,
            cell_regions,
            Projected,
            u,
        ),
        AddressableField(
            YVelocity,
            Cell,
            cell_regions,
            Projected,
            zero,
        ),
        ZFaceFieldContext(
            AddressableField(
                VerticalVelocity,
                ZFace,
                face_regions,
                Projected,
                zero,
            ),
            jnp.zeros((domain.ny, domain.nx), dtype=dtype),
        ),
    )


def _diagnostics(
    state,
    *,
    divergence,
    case: CaseConfig,
    scales,
    jnp,
) -> dict[str, float | bool]:
    velocity = state.velocity
    u = scales.from_execution_velocity(velocity.x.payload)
    v = scales.from_execution_velocity(velocity.y.payload)
    w = scales.from_execution_velocity(velocity.z.owned.payload)
    cfl_x = (
        float(jnp.max(jnp.abs(u)))
        * case.time.dt_seconds
        / case.domain.dx_m
    )
    cfl_y = (
        float(jnp.max(jnp.abs(v)))
        * case.time.dt_seconds
        / case.domain.dy_m
    )
    cfl_z = (
        float(jnp.max(jnp.abs(w)))
        * case.time.dt_seconds
        / case.domain.dz_m
    )
    finite = bool(
        jnp.all(jnp.isfinite(velocity.x.payload))
        & jnp.all(jnp.isfinite(velocity.y.payload))
        & jnp.all(jnp.isfinite(velocity.z.owned.payload))
    )
    maximum_execution_divergence = float(
        jnp.max(jnp.abs(divergence))
    )
    return {
        "cfl_x": cfl_x,
        "cfl_y": cfl_y,
        "cfl_z": cfl_z,
        "maximum_cfl": max(cfl_x, cfl_y, cfl_z),
        "maximum_execution_divergence": maximum_execution_divergence,
        "maximum_divergence_s_inv": (
            maximum_execution_divergence * scales.inverse_time
        ),
        "fields_finite": finite,
        "maximum_u_m_s": float(jnp.max(jnp.abs(u))),
        "maximum_v_m_s": float(jnp.max(jnp.abs(v))),
        "maximum_w_m_s": float(jnp.max(jnp.abs(w))),
    }


def _interpolation_indices(
    position: float,
    spacing: float,
    count: int,
) -> tuple[int, int, float]:
    coordinate = position / spacing - 0.5
    lower = max(0, min(int(np.floor(coordinate)), count - 1))
    upper = min(lower + 1, count - 1)
    fraction = max(0.0, min(coordinate - lower, 1.0))
    return lower, upper, fraction


def _capture_flow_frame(
    state,
    *,
    case: CaseConfig,
    scales,
    jax,
    jnp,
) -> tuple[np.ndarray, np.ndarray]:
    domain = case.domain
    u = scales.from_execution_velocity(
        state.velocity.x.payload
    ).reshape((domain.nz, domain.ny, domain.nx))
    z_m = (
        jnp.arange(domain.nz, dtype=u.dtype) + 0.5
    ) * domain.dz_m
    baseline = (
        case.flow.friction_velocity_m_s
        / case.flow.von_karman
        * jnp.log(z_m / case.flow.roughness_length_m)
    )
    delta_u = u - baseline[:, None, None]

    x_lower, x_upper, x_fraction = _interpolation_indices(
        case.turbine.location_m[0],
        domain.dx_m,
        domain.nx,
    )
    z_lower, z_upper, z_fraction = _interpolation_indices(
        case.turbine.hub_height_m,
        domain.dz_m,
        domain.nz,
    )
    rotor_plane = (
        (1.0 - x_fraction) * delta_u[:, :, x_lower]
        + x_fraction * delta_u[:, :, x_upper]
    )
    hub_plane = (
        (1.0 - z_fraction) * delta_u[z_lower]
        + z_fraction * delta_u[z_upper]
    )
    rotor_plane, hub_plane = jax.device_get(
        (rotor_plane, hub_plane)
    )
    return (
        np.asarray(rotor_plane, dtype=np.float32),
        np.asarray(hub_plane, dtype=np.float32),
    )


def _save_flow_frames(
    path: Path,
    *,
    case: CaseConfig,
    times_seconds: list[float],
    rotor_planes: list[np.ndarray],
    hub_planes: list[np.ndarray],
    blade_positions_m: list[np.ndarray],
) -> None:
    metadata = {
        "schema": "jaxwind.direct-alm.flow-slices.v2",
        "quantities": {
            "rotor_plane_delta_u_m_s": "u(y,z)-u_initial(z)",
            "hub_plane_delta_u_m_s": "u(x,y)-u_initial(z_hub)",
        },
        "domain": {
            "nx": case.domain.nx,
            "ny": case.domain.ny,
            "nz": case.domain.nz,
            "lx_m": case.domain.lx_m,
            "ly_m": case.domain.ly_m,
            "lz_m": case.domain.lz_m,
        },
        "turbine": {
            "model": (
                "openfast_modal_aeroelastic_actuator_line"
                if case.aeroelastic.enabled
                else "openfast_rigid_actuator_line"
            ),
            "x_m": case.turbine.location_m[0],
            "y_m": case.turbine.location_m[1],
            "hub_height_m": case.turbine.hub_height_m,
            "hub_radius_m": case.turbine.openfast.hub_radius_m,
            "tip_radius_m": case.turbine.openfast.tip_radius_m,
            "rotor_speed_rpm": case.turbine.rotor_speed_rpm,
            "angular_velocity_rad_s": (
                case.turbine.openfast.angular_velocity_rad_s
            ),
            "initial_azimuth_degrees": (
                case.turbine.initial_azimuth_degrees
            ),
            "precone_degrees": (
                case.turbine.openfast.precone_degrees
            ),
            "blade_count": case.turbine.openfast.blade_count,
            "smoothing_width_m": case.turbine.smoothing_width_m,
        },
    }
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=np.asarray(json.dumps(metadata)),
            times_seconds=np.asarray(times_seconds, dtype=np.float64),
            rotor_plane_delta_u_m_s=np.stack(rotor_planes),
            hub_plane_delta_u_m_s=np.stack(hub_planes),
            blade_positions_m=np.stack(blade_positions_m),
        )
    temporary.replace(path)


def _blade_positions_m(
    case: CaseConfig,
    *,
    modal_model,
    modal_state,
    time_seconds: float,
) -> np.ndarray:
    """Return physical blade-element centers for animation diagnostics."""

    turbine = case.turbine.openfast
    yaw = math.radians(case.turbine.yaw_degrees)
    tilt = math.radians(-turbine.shaft_tilt_degrees)
    precone = math.radians(turbine.precone_degrees)
    normal = np.asarray(
        (
            math.cos(tilt) * math.cos(yaw),
            math.cos(tilt) * math.sin(yaw),
            math.sin(tilt),
        )
    )
    horizontal = np.asarray((-math.sin(yaw), math.cos(yaw), 0.0))
    vertical = np.cross(normal, horizontal)
    blade_phase = (
        2.0 * math.pi * np.arange(turbine.blade_count)
        / turbine.blade_count
    )
    speed = case.turbine.rotor_speed_rpm * 2.0 * math.pi / 60.0
    if turbine.mirror_rotor:
        speed = -speed
    theta = (
        math.radians(case.turbine.initial_azimuth_degrees)
        + speed * time_seconds
        + blade_phase[:, None]
    )
    radial = (
        np.cos(theta)[..., None] * vertical
        + np.sin(theta)[..., None] * horizontal
    )
    tangent = (
        -np.sin(theta)[..., None] * vertical
        + np.cos(theta)[..., None] * horizontal
    )
    coned_radial = math.cos(precone) * radial + math.sin(precone) * normal
    if modal_model is None or modal_state is None:
        shape = (turbine.blade_count, len(turbine.element_radii_m))
        flap = np.zeros(shape)
        edge = np.zeros(shape)
    else:
        fields = modal_model.deformation_fields(modal_state)
        flap = fields["flap_displacements_m"]
        edge = fields["edge_displacements_m"]
    hub = np.asarray(
        (
            case.turbine.location_m[0],
            case.turbine.location_m[1],
            case.turbine.hub_height_m,
        )
    )
    radii = np.asarray(turbine.element_radii_m)
    return (
        hub
        + radii[None, :, None] * coned_radial
        + flap[..., None] * normal
        + edge[..., None] * tangent
    ).astype(np.float32)
