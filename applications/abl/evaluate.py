"""Materialize and evaluate a fully composed ABL case."""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np

from applications.boussinesq import BoussinesqCase
from applications.initial_conditions import build_initial_fields


SOURCE_ROOT = Path(__file__).resolve().parents[2]


class ProfileStatistics:
    """Restartable averages of instantaneous horizontal-plane statistics."""

    NAMES = (
        "u",
        "v",
        "w",
        "scalar",
        "u_variance",
        "v_variance",
        "w_variance",
        "scalar_variance",
    )

    DIAGNOSTIC_NAMES = (
        "resolved_uw",
        "resolved_vw",
        "resolved_wc",
        "sgs_tke",
        "sgs_uw",
        "sgs_vw",
        "sgs_wc",
        "resolved_tke_sgs_transfer",
        "momentum_diffusivity",
        "scalar_diffusivity",
        "sgs_scalar_variance",
        "pressure_variance",
        "w_third_moment",
        "updraft_fraction",
        "updraft_w",
        "updraft_scalar_excess",
        "resolved_energy_vertical_transport",
        "pressure_vertical_transport",
    )

    SPECTRUM_NAMES = (
        "mode",
        "u",
        "v",
        "w",
        "scalar",
        "height_m",
        "radial_wavenumber_reference",
        "radial_horizontal",
        "radial_w",
        "radial_scalar",
    )

    def __init__(self, nz: int) -> None:
        self.count = 0
        self.diagnostic_count = 0
        self.spectrum_count = 0
        self.ustar_sum = 0.0
        self.surface_scalar_flux_count = 0
        self.surface_scalar_flux_sum = 0.0
        self.sums = {name: np.zeros(nz, dtype=np.float64) for name in self.NAMES}
        self.diagnostic_sums = {
            name: np.zeros(nz, dtype=np.float64)
            for name in self.DIAGNOSTIC_NAMES
        }
        self.spectrum_sums: dict[str, np.ndarray] = {}

    def sample(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        scalar: np.ndarray,
        *,
        ustar: float,
        surface_scalar_flux: float | None = None,
        diagnostics: dict[str, np.ndarray] | None = None,
        spectra: dict[str, np.ndarray] | None = None,
    ) -> None:
        fields = {"u": u, "v": v, "w": w, "scalar": scalar}
        for name, values in fields.items():
            mean = np.mean(values, axis=(-2, -1))
            self.sums[name] += mean
            self.sums[f"{name}_variance"] += np.mean(
                (values - mean[:, None, None]) ** 2,
                axis=(-2, -1),
            )
        self.ustar_sum += ustar
        if surface_scalar_flux is not None:
            self.surface_scalar_flux_sum += surface_scalar_flux
            self.surface_scalar_flux_count += 1
        self.count += 1
        if diagnostics is not None:
            missing = set(self.DIAGNOSTIC_NAMES) - set(diagnostics)
            if missing:
                raise ValueError(
                    "profile diagnostics are missing: " + ", ".join(sorted(missing))
                )
            for name in self.DIAGNOSTIC_NAMES:
                values = np.asarray(diagnostics[name], dtype=np.float64)
                if values.shape != self.diagnostic_sums[name].shape:
                    raise ValueError(f"diagnostic profile {name} has the wrong shape")
                self.diagnostic_sums[name] += values
            self.diagnostic_count += 1
        if spectra is not None:
            missing = set(self.SPECTRUM_NAMES) - set(spectra)
            if missing:
                raise ValueError(
                    "spectral diagnostics are missing: " + ", ".join(sorted(missing))
                )
            for name in self.SPECTRUM_NAMES:
                values = np.asarray(spectra[name], dtype=np.float64)
                if not self.spectrum_sums:
                    self.spectrum_sums = {
                        key: np.zeros_like(np.asarray(spectra[key]), dtype=np.float64)
                        for key in self.SPECTRUM_NAMES
                    }
                if values.shape != self.spectrum_sums[name].shape:
                    raise ValueError(f"spectrum {name} has the wrong shape")
                self.spectrum_sums[name] += values
            self.spectrum_count += 1

    def save(self, path: Path) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                count=np.asarray(self.count),
                diagnostic_count=np.asarray(self.diagnostic_count),
                spectrum_count=np.asarray(self.spectrum_count),
                ustar_sum=np.asarray(self.ustar_sum),
                surface_scalar_flux_count=np.asarray(
                    self.surface_scalar_flux_count
                ),
                surface_scalar_flux_sum=np.asarray(self.surface_scalar_flux_sum),
                **self.sums,
                **{
                    f"diagnostic_{name}": values
                    for name, values in self.diagnostic_sums.items()
                },
                **{
                    f"spectrum_{name}": values
                    for name, values in self.spectrum_sums.items()
                },
            )
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path, nz: int) -> "ProfileStatistics":
        result = cls(nz)
        with np.load(path, allow_pickle=False) as archive:
            result.count = int(archive["count"])
            result.ustar_sum = float(archive["ustar_sum"])
            result.surface_scalar_flux_count = (
                int(archive["surface_scalar_flux_count"])
                if "surface_scalar_flux_count" in archive
                else 0
            )
            result.surface_scalar_flux_sum = (
                float(archive["surface_scalar_flux_sum"])
                if "surface_scalar_flux_sum" in archive
                else 0.0
            )
            for name in cls.NAMES:
                values = np.asarray(archive[name], dtype=np.float64)
                if values.shape != (nz,):
                    raise ValueError("statistics shape does not match the case grid")
                result.sums[name] = values.copy()
            result.diagnostic_count = (
                int(archive["diagnostic_count"])
                if "diagnostic_count" in archive
                else 0
            )
            if result.diagnostic_count:
                for name in cls.DIAGNOSTIC_NAMES:
                    key = f"diagnostic_{name}"
                    if key not in archive:
                        raise ValueError(f"statistics are missing {key}")
                    values = np.asarray(archive[key], dtype=np.float64)
                    if values.shape != (nz,):
                        raise ValueError(
                            "diagnostic shape does not match the case grid"
                        )
                    result.diagnostic_sums[name] = values.copy()
            result.spectrum_count = (
                int(archive["spectrum_count"])
                if "spectrum_count" in archive
                else 0
            )
            if result.spectrum_count:
                for name in cls.SPECTRUM_NAMES:
                    key = f"spectrum_{name}"
                    if key not in archive:
                        raise ValueError(f"statistics are missing {key}")
                    result.spectrum_sums[name] = np.asarray(
                        archive[key], dtype=np.float64
                    ).copy()
        return result

    def profiles(self) -> dict[str, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("no profile samples have been collected")
        profiles = {
            name: values / self.count for name, values in self.sums.items()
        }
        if self.diagnostic_count:
            profiles.update(
                {
                    name: values / self.diagnostic_count
                    for name, values in self.diagnostic_sums.items()
                }
            )
        return profiles

    def spectra(self) -> dict[str, np.ndarray]:
        if self.spectrum_count == 0:
            raise RuntimeError("no spectrum samples have been collected")
        return {
            name: values / self.spectrum_count
            for name, values in self.spectrum_sums.items()
        }

    @property
    def mean_ustar(self) -> float:
        if self.count == 0:
            raise RuntimeError("no friction-velocity samples have been collected")
        return self.ustar_sum / self.count

    @property
    def mean_surface_scalar_flux(self) -> float:
        if self.surface_scalar_flux_count == 0:
            raise RuntimeError("no surface scalar-flux samples have been collected")
        return self.surface_scalar_flux_sum / self.surface_scalar_flux_count


def _configure_pressure_source() -> None:
    source = Path(
        os.environ.get(
            "JAXWIND_SPECTRAL_FD_SOURCE",
            SOURCE_ROOT / "external" / "bw1000_benchmark",
        )
    )
    if source.exists() and str(source) not in sys.path:
        sys.path.insert(0, str(source))


def resolved(case: BoussinesqCase) -> dict[str, Any]:
    """Return a JSON-compatible view without reverse-dispatching components."""

    grid = case.physical_grid
    momentum = case.model.momentum
    rotation = momentum.rotation
    wall = momentum.wall
    scalar_flux = case.scalar_scales.from_execution_flux(
        case.model.scalar_boundary.lower_flux
    )
    surface_transfer = case.model.surface_transfer
    rotation_values = _rotation_values(case)
    buoyancy_coefficient = (
        case.scalar_scales.from_execution_buoyancy_coefficient(
            case.model.buoyancy.acceleration_per_temperature
        )
    )
    return {
        "case": case.name,
        "citation": case.citation,
        "domain": {
            "cells": [grid.nx, grid.ny, grid.nz],
            "lengths_m": [grid.lx, grid.ly, grid.lz],
            "spacing_m": [grid.dx, grid.dy, grid.dz],
        },
        "physics": {
            "advection": type(momentum.advection).__name__,
            "pressure_gradient": type(momentum.pressure_gradient).__name__,
            "wall": type(wall).__name__,
            "roughness_length_m": case.mechanical_scales.from_execution_length(
                wall.roughness_length
            ),
            "momentum_sgs": type(momentum.sgs).__name__,
            "scalar_sgs": type(case.model.scalar_sgs).__name__,
            "rotation": type(rotation).__name__,
            "coriolis_vertical_s": rotation_values[0],
            "coriolis_horizontal_s": rotation_values[1],
            "geostrophic_velocity_m_s": [
                rotation_values[2],
                rotation_values[3],
            ],
            "scalar_quantity": case.scalar_scales.quantity,
            "scalar_reference_value": case.scalar_scales.reference_value,
            "scalar_surface_flux": scalar_flux,
            "surface_transfer": type(surface_transfer).__name__,
            "surface_scalar_initial": (
                case.scalar_scales.from_execution_scalar(
                    surface_transfer.surface_scalar_initial
                )
                if hasattr(surface_transfer, "surface_scalar_initial")
                else None
            ),
            "surface_scalar_rate_per_second": (
                surface_transfer.surface_scalar_rate
                * case.scalar_scales.magnitude
                / case.mechanical_scales.time
                if hasattr(surface_transfer, "surface_scalar_rate")
                else None
            ),
            "surface_scalar_roughness_length_m": (
                case.mechanical_scales.from_execution_length(
                    surface_transfer.scalar_roughness_length
                )
                if hasattr(surface_transfer, "scalar_roughness_length")
                else None
            ),
            "buoyancy_acceleration_per_scalar": buoyancy_coefficient,
        },
        "time": {
            "method": "AB2",
            "dt_seconds": case.dt_seconds,
            "steps": case.steps,
            "duration_hours": case.duration_seconds / 3600.0,
            "statistics_start_step": case.output.sample_start_step,
            "statistics_start_hours": (
                case.output.sample_start_step * case.dt_seconds / 3600.0
            ),
        },
        "numerics": {
            "dtype": case.pressure.dtype,
            "pressure_method": case.pressure.method,
            "thomas_chunk": case.pressure.thomas_chunk,
            "nonlinear_padding_ratio": case.nonlinear_padding_ratio,
            "advection_frame_velocity_m_s": list(
                case.advection_frame_velocity_m_s
            ),
        },
        "diagnostic_reference": {
            "length_m": case.diagnostic_reference.length_m,
            "velocity_m_s": case.diagnostic_reference.velocity_m_s,
            "scalar": case.diagnostic_reference.scalar,
            "inversion_search_max_height_m": (
                case.diagnostic_reference.inversion_search_max_height_m
            ),
            "spectrum_heights_m": list(
                case.diagnostic_reference.spectrum_heights_m
            ),
        },
        "output": {
            "directory": str(case.output.directory),
            "sample_every_steps": case.output.sample_every_steps,
            "log_every_steps": case.output.log_every_steps,
            "checkpoint_every_steps": case.output.checkpoint_every_steps,
        },
    }


def _rotation_values(case: BoussinesqCase) -> tuple[float, float, float, float]:
    """Return dimensional rotation/forcing values, including their zero limit."""

    rotation = case.model.momentum.rotation
    vertical = getattr(rotation, "coriolis_parameter", 0.0)
    horizontal = getattr(rotation, "horizontal_coriolis_parameter", 0.0)
    geostrophic_x = getattr(rotation, "geostrophic_x_velocity", 0.0)
    geostrophic_y = getattr(rotation, "geostrophic_y_velocity", 0.0)
    return (
        case.mechanical_scales.from_execution_inverse_time(vertical),
        case.mechanical_scales.from_execution_inverse_time(horizontal),
        case.mechanical_scales.from_execution_velocity(geostrophic_x)
        + case.advection_frame_velocity_m_s[0],
        case.mechanical_scales.from_execution_velocity(geostrophic_y)
        + case.advection_frame_velocity_m_s[1],
    )


def _physics_fingerprints(case: BoussinesqCase) -> tuple[str, str]:
    momentum = case.model.momentum.sgs
    scalar = case.model.scalar_sgs
    closure = momentum.fingerprint + "|" + scalar.fingerprint
    flow = case.model.momentum
    rotation = flow.rotation
    wall = flow.wall
    buoyancy = case.model.buoyancy
    scalar_boundary = case.model.scalar_boundary
    surface_transfer = case.model.surface_transfer
    physics = (
        "jaxwind.application-boussinesq-physics.v2"
        + f"|closure={closure}"
        + f"|advection={type(flow.advection).__name__}"
        + f"|pressure-x={flow.pressure_gradient.x_acceleration.hex()}"
        + f"|pressure-y={flow.pressure_gradient.y_acceleration.hex()}"
        + f"|wall={type(wall).__name__}"
        + f":{wall.roughness_length.hex()}:{wall.von_karman.hex()}"
        + f"|rotation={type(rotation).__name__}:{rotation!r}"
        + f"|buoyancy={type(buoyancy).__name__}"
        + f":{buoyancy.acceleration_per_temperature.hex()}"
        + f"|scalar-boundary={scalar_boundary.lower_flux.hex()}"
        + f":{scalar_boundary.upper_flux.hex()}"
        + f"|surface-transfer={surface_transfer!r}"
        + f"|rayleigh={case.model.rayleigh_damping!r}"
        + f"|padding={case.nonlinear_padding_ratio.hex()}"
    )
    return closure, physics


def _physical_fields(state, case: BoussinesqCase, jnp, solver):
    velocity = state.fields.velocity
    u = case.mechanical_scales.from_execution_velocity(
        solver.global_array(velocity.x.payload)
    ) + case.advection_frame_velocity_m_s[0]
    v = case.mechanical_scales.from_execution_velocity(
        solver.global_array(velocity.y.payload)
    ) + case.advection_frame_velocity_m_s[1]
    w_upper = case.mechanical_scales.from_execution_velocity(
        solver.global_array(velocity.z.owned.payload)
    )
    lower = jnp.concatenate((jnp.zeros_like(w_upper[:1]), w_upper[:-1]), axis=0)
    w = 0.5 * (lower + w_upper)
    scalar = case.scalar_scales.from_execution_scalar(
        solver.global_array(state.fields.potential_temperature.payload)
    )
    return u, v, w, w_upper, scalar


def _friction_velocity(u, v, case: BoussinesqCase, jnp) -> float:
    _tau_x, _tau_y, ustar = _wall_stress(u, v, case, jnp)
    return ustar


def _wall_stress(u, v, case: BoussinesqCase, jnp):
    wall = case.model.momentum.wall
    roughness = case.mechanical_scales.from_execution_length(
        wall.roughness_length
    )
    drag = (
        wall.von_karman
        / math.log(0.5 * case.physical_grid.dz / roughness)
    ) ** 2
    speed = jnp.hypot(u[0], v[0])
    tau_x = -drag * speed * u[0]
    tau_y = -drag * speed * v[0]
    ustar = math.sqrt(
        math.hypot(float(jnp.mean(tau_x)), float(jnp.mean(tau_y)))
    )
    return tau_x, tau_y, ustar


def _x_spectrum(values: np.ndarray, level: int) -> np.ndarray:
    """Return the one-sided streamwise variance spectrum at one z level."""

    signal = values[level] - np.mean(values[level], axis=-1, keepdims=True)
    coefficients = np.fft.rfft(signal, axis=-1) / signal.shape[-1]
    energy = np.mean(np.abs(coefficients) ** 2, axis=0)
    factors = np.full(energy.shape, 2.0, dtype=np.float64)
    factors[0] = 1.0
    if signal.shape[-1] % 2 == 0:
        factors[-1] = 1.0
    return energy * factors


def _radial_spectrum(
    values: np.ndarray,
    *,
    dx: float,
    dy: float,
    edges: np.ndarray,
) -> np.ndarray:
    """Return a variance-conserving horizontal radial spectrum."""

    ny, nx = values.shape
    signal = values - np.mean(values)
    transformed = np.fft.fft2(signal) / (nx * ny)
    energy = np.abs(transformed) ** 2
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    radius = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    bins = np.digitize(radius.ravel(), edges) - 1
    valid = (bins >= 0) & (bins < edges.size - 1)
    result = np.bincount(
        bins[valid],
        weights=energy.ravel()[valid],
        minlength=edges.size - 1,
    )
    return result[: edges.size - 1]


def _diagnostic_observables(
    state,
    pressure,
    case: BoussinesqCase,
    solver,
    jax,
    jnp,
    *,
    include_spectra: bool,
):
    """Evaluate paper-aligned instantaneous diagnostics in physical units."""

    u, v, w, _w_upper, scalar = _physical_fields(state, case, jnp, solver)
    mean_u = jnp.mean(u, axis=(-2, -1))
    mean_v = jnp.mean(v, axis=(-2, -1))
    mean_w = jnp.mean(w, axis=(-2, -1))
    mean_scalar = jnp.mean(scalar, axis=(-2, -1))
    u_fluctuation = u - mean_u[:, None, None]
    v_fluctuation = v - mean_v[:, None, None]
    w_fluctuation = w - mean_w[:, None, None]
    scalar_fluctuation = scalar - mean_scalar[:, None, None]
    numerical = solver.diagnostic_fields(state.fields, state.clock)
    surface_transfer = numerical.surface_transfer
    if surface_transfer is None:
        tau_x, tau_y, ustar = _wall_stress(u, v, case, jnp)
        surface_scalar_flux = case.scalar_scales.from_execution_flux(
            case.model.scalar_boundary.lower_flux
        )
        surface_scalar = math.nan
        obukhov_length = math.nan
    else:
        def replicated(value):
            return value.reshape(-1)[0]

        tau_x_value = (
            -replicated(surface_transfer.stress_x)
            * case.mechanical_scales.kinematic_pressure
        )
        tau_y_value = (
            -replicated(surface_transfer.stress_y)
            * case.mechanical_scales.kinematic_pressure
        )
        tau_x = jnp.full_like(u[0], tau_x_value)
        tau_y = jnp.full_like(v[0], tau_y_value)
        ustar = float(
            case.mechanical_scales.from_execution_velocity(
                replicated(surface_transfer.friction_velocity)
            )
        )
        surface_scalar_flux = float(
            case.scalar_scales.from_execution_flux(
                replicated(surface_transfer.scalar_flux)
            )
        )
        surface_scalar = float(
            case.scalar_scales.from_execution_scalar(
                surface_transfer.surface_scalar
                if getattr(surface_transfer.surface_scalar, "ndim", 0) == 0
                else replicated(surface_transfer.surface_scalar)
            )
        )
        obukhov_length = float(
            case.mechanical_scales.from_execution_length(
                replicated(surface_transfer.obukhov_length)
            )
        )

    velocity_squared = case.mechanical_scales.kinematic_pressure
    sgs_tke = solver.global_array(numerical.sgs_tke) * velocity_squared
    momentum_diffusivity = solver.global_array(numerical.momentum_diffusivity) * (
        case.mechanical_scales.kinematic_viscosity
    )
    scalar_diffusivity = solver.global_array(numerical.scalar_diffusivity) * (
        case.mechanical_scales.kinematic_viscosity
    )
    sgs_scalar_variance = solver.global_array(numerical.scalar_variance) * (
        case.scalar_scales.magnitude**2
    )
    scalar_flux_upper = case.scalar_scales.from_execution_flux(
        solver.global_array(numerical.scalar_flux_z)
    )
    lower_scalar_flux = jnp.concatenate(
        (
            jnp.full_like(scalar_flux_upper[:1], surface_scalar_flux),
            scalar_flux_upper[:-1],
        ),
        axis=0,
    )
    sgs_scalar_flux = 0.5 * (lower_scalar_flux + scalar_flux_upper)

    pressure_physical = (
        solver.global_array(pressure.payload)
        * case.mechanical_scales.kinematic_pressure
    )
    pressure_fluctuation = pressure_physical - jnp.mean(
        pressure_physical, axis=(-2, -1), keepdims=True
    )
    w_face = case.mechanical_scales.from_execution_velocity(
        solver.global_array(state.fields.velocity.z.owned.payload)
    )
    scalar_face = (
        solver.global_array(numerical.scalar_upper)
        * case.scalar_scales.magnitude
    )
    w_face_fluctuation = w_face - jnp.mean(
        w_face, axis=(-2, -1), keepdims=True
    )
    scalar_face_fluctuation = scalar_face - jnp.mean(
        scalar_face, axis=(-2, -1), keepdims=True
    )
    resolved_scalar_flux_upper = jnp.mean(
        w_face_fluctuation * scalar_face_fluctuation,
        axis=(-2, -1),
    )
    resolved_scalar_flux = 0.5 * jnp.concatenate(
        (
            resolved_scalar_flux_upper[:1],
            resolved_scalar_flux_upper[:-1] + resolved_scalar_flux_upper[1:],
        )
    )

    resolved_tke_sgs_transfer = solver.global_array(
        numerical.momentum_sgs_tke_transfer
    ) * (
        case.mechanical_scales.kinematic_pressure
        * case.mechanical_scales.inverse_time
    )
    txz_upper = solver.global_array(numerical.sgs_flux_xz) * velocity_squared
    tyz_upper = solver.global_array(numerical.sgs_flux_yz) * velocity_squared
    lower_txz = jnp.concatenate((tau_x[None], txz_upper[:-1]), axis=0)
    lower_tyz = jnp.concatenate((tau_y[None], tyz_upper[:-1]), axis=0)

    def profile(values) -> np.ndarray:
        return np.asarray(
            jax.device_get(jnp.mean(values, axis=(-2, -1))),
            dtype=np.float64,
        )

    resolved_tke = profile(
        0.5
        * (
            u_fluctuation**2
            + v_fluctuation**2
            + w_fluctuation**2
        )
    )
    sgs_tke_profile = profile(sgs_tke)
    updraft = w_fluctuation > 0.0
    updraft_count = jnp.sum(updraft, axis=(-2, -1))
    safe_updraft_count = jnp.maximum(updraft_count, 1)
    updraft_fraction = updraft_count / float(
        case.physical_grid.nx * case.physical_grid.ny
    )
    updraft_w = jnp.sum(jnp.where(updraft, w_fluctuation, 0.0), axis=(-2, -1)) / (
        safe_updraft_count
    )
    updraft_scalar_excess = jnp.sum(
        jnp.where(updraft, scalar_fluctuation, 0.0), axis=(-2, -1)
    ) / safe_updraft_count
    resolved_energy = 0.5 * (
        u_fluctuation**2 + v_fluctuation**2 + w_fluctuation**2
    )
    diagnostics = {
        "resolved_uw": profile(u_fluctuation * w_fluctuation),
        "resolved_vw": profile(v_fluctuation * w_fluctuation),
        "resolved_wc": np.asarray(
            jax.device_get(resolved_scalar_flux), dtype=np.float64
        ),
        "sgs_tke": sgs_tke_profile,
        "sgs_uw": profile(0.5 * (lower_txz + txz_upper)),
        "sgs_vw": profile(0.5 * (lower_tyz + tyz_upper)),
        "sgs_wc": profile(sgs_scalar_flux),
        "resolved_tke_sgs_transfer": profile(resolved_tke_sgs_transfer),
        "momentum_diffusivity": profile(momentum_diffusivity),
        "scalar_diffusivity": profile(scalar_diffusivity),
        "sgs_scalar_variance": profile(sgs_scalar_variance),
        "pressure_variance": profile(pressure_fluctuation**2),
        "w_third_moment": profile(w_fluctuation**3),
        "updraft_fraction": np.asarray(
            jax.device_get(updraft_fraction), dtype=np.float64
        ),
        "updraft_w": np.asarray(jax.device_get(updraft_w), dtype=np.float64),
        "updraft_scalar_excess": np.asarray(
            jax.device_get(updraft_scalar_excess), dtype=np.float64
        ),
        "resolved_energy_vertical_transport": profile(
            w_fluctuation * resolved_energy
        ),
        "pressure_vertical_transport": profile(
            pressure_fluctuation * w_fluctuation
        ),
    }

    mean_u_host, mean_v_host = jax.device_get((mean_u, mean_v))
    mean_u_host = np.asarray(mean_u_host, dtype=np.float64)
    mean_v_host = np.asarray(mean_v_host, dtype=np.float64)
    surface_uw = float(jnp.mean(tau_x))
    surface_vw = float(jnp.mean(tau_y))
    coriolis, _horizontal_coriolis, geostrophic_u, geostrophic_v = (
        _rotation_values(case)
    )
    integrated_u_deficit = np.sum(mean_u_host - geostrophic_u) * (
        case.physical_grid.dz
    )
    integrated_v_deficit = np.sum(mean_v_host - geostrophic_v) * (
        case.physical_grid.dz
    )
    history = {
        "ustar_m_s": ustar,
        "surface_scalar": surface_scalar,
        "surface_scalar_flux": surface_scalar_flux,
        "obukhov_length_m": obukhov_length,
        "surface_uw_m2_s2": surface_uw,
        "surface_vw_m2_s2": surface_vw,
        "momentum_stationarity_cu": (
            -coriolis * integrated_v_deficit / surface_uw
            if surface_uw != 0.0
            else math.nan
        ),
        "momentum_stationarity_cv": (
            coriolis * integrated_u_deficit / surface_vw
            if surface_vw != 0.0
            else math.nan
        ),
        "integrated_resolved_tke_m3_s2": float(
            np.sum(resolved_tke) * case.physical_grid.dz
        ),
        "integrated_sgs_tke_m3_s2": float(
            np.sum(sgs_tke_profile) * case.physical_grid.dz
        ),
    }
    history["integrated_total_tke_m3_s2"] = (
        history["integrated_resolved_tke_m3_s2"]
        + history["integrated_sgs_tke_m3_s2"]
    )

    fields = {
        "u": np.asarray(jax.device_get(u), dtype=np.float64),
        "v": np.asarray(jax.device_get(v), dtype=np.float64),
        "w": np.asarray(jax.device_get(w), dtype=np.float64),
        "scalar": np.asarray(jax.device_get(scalar), dtype=np.float64),
    }
    spectra = None
    if include_spectra:
        z = (
            np.arange(case.physical_grid.nz, dtype=np.float64) + 0.5
        ) * case.physical_grid.dz
        targets = np.asarray(
            [
                int(np.argmin(np.abs(z - height)))
                for height in case.diagnostic_reference.spectrum_heights_m
            ],
            dtype=np.int64,
        )
        modes = np.arange(case.physical_grid.nx // 2 + 1, dtype=np.float64)
        radial_edges = np.linspace(
            0.0,
            np.hypot(
                np.pi / case.physical_grid.dx,
                np.pi / case.physical_grid.dy,
            ),
            modes.size + 1,
        )
        radial_wavenumber = 0.5 * (radial_edges[:-1] + radial_edges[1:])
        radial = {
            name: np.stack(
                [
                    _radial_spectrum(
                        fields[name][level],
                        dx=case.physical_grid.dx,
                        dy=case.physical_grid.dy,
                        edges=radial_edges,
                    )
                    for level in targets
                ]
            )
            for name in ("u", "v", "w", "scalar")
        }
        spectra = {
            "mode": np.broadcast_to(modes, (targets.size, modes.size)),
            "u": np.stack([_x_spectrum(fields["u"], level) for level in targets]),
            "v": np.stack([_x_spectrum(fields["v"], level) for level in targets]),
            "w": np.stack([_x_spectrum(fields["w"], level) for level in targets]),
            "scalar": np.stack(
                [_x_spectrum(fields["scalar"], level) for level in targets]
            ),
            "height_m": np.broadcast_to(z[targets, None], (targets.size, modes.size)),
            "radial_wavenumber_reference": np.broadcast_to(
                radial_wavenumber[None] * case.diagnostic_reference.length_m,
                (targets.size, modes.size),
            ),
            "radial_horizontal": radial["u"] + radial["v"],
            "radial_w": radial["w"],
            "radial_scalar": radial["scalar"],
        }
    return fields, history, diagnostics, spectra


def _diagnostics(state, divergence, case: BoussinesqCase, jnp, solver):
    u, v, _w, w_upper, _scalar = _physical_fields(state, case, jnp, solver)
    grid = case.physical_grid
    cfl_x = float(jnp.max(jnp.abs(u))) * case.dt_seconds / grid.dx
    cfl_y = float(jnp.max(jnp.abs(v))) * case.dt_seconds / grid.dy
    cfl_z = float(jnp.max(jnp.abs(w_upper))) * case.dt_seconds / grid.dz
    maximum_cfl = max(cfl_x, cfl_y, cfl_z)
    update_interval = case.model.momentum.sgs.update_interval
    return {
        "cfl_x": cfl_x,
        "cfl_y": cfl_y,
        "cfl_z": cfl_z,
        "maximum_cfl": maximum_cfl,
        # Horizontal departure interpolation is periodic and supports travel over
        # multiple cells.  The vertical interpolation has one neighboring plane,
        # so its trajectory limit applies only to vertical displacement.
        "lasd_trajectory_cfl": cfl_z * update_interval,
        "ustar_m_s": _friction_velocity(u, v, case, jnp),
        "maximum_execution_divergence": float(
            jnp.max(jnp.abs(solver.global_array(divergence)))
        ),
    }


def _write_profiles(
    path: Path,
    case: BoussinesqCase,
    statistics: ProfileStatistics,
) -> None:
    fields = statistics.profiles()
    z = (
        np.arange(case.physical_grid.nz, dtype=np.float64) + 0.5
    ) * case.physical_grid.dz
    resolved_tke = 0.5 * (
        fields["u_variance"]
        + fields["v_variance"]
        + fields["w_variance"]
    )
    ustar = statistics.mean_ustar
    coriolis = _rotation_values(case)[0]
    columns = {
        "z_m": z,
        "z_f_over_ustar": z * coriolis / ustar,
        "z_over_reference_length": z / case.diagnostic_reference.length_m,
        "mean_u_m_s": fields["u"],
        "mean_v_m_s": fields["v"],
        "mean_w_m_s": fields["w"],
        "mean_scalar": fields["scalar"],
        "mean_scalar_kg_m3": fields["scalar"],
        "resolved_u_variance_m2_s2": fields["u_variance"],
        "resolved_v_variance_m2_s2": fields["v_variance"],
        "resolved_w_variance_m2_s2": fields["w_variance"],
        "resolved_tke_m2_s2": resolved_tke,
        "resolved_scalar_variance": fields["scalar_variance"],
        "resolved_scalar_variance_kg2_m6": fields["scalar_variance"],
    }
    if statistics.diagnostic_count:
        columns.update(
            {
                "sgs_tke_m2_s2": fields["sgs_tke"],
                "resolved_uw_m2_s2": fields["resolved_uw"],
                "resolved_vw_m2_s2": fields["resolved_vw"],
                "sgs_uw_m2_s2": fields["sgs_uw"],
                "sgs_vw_m2_s2": fields["sgs_vw"],
                "total_uw_m2_s2": fields["resolved_uw"] + fields["sgs_uw"],
                "total_vw_m2_s2": fields["resolved_vw"] + fields["sgs_vw"],
                "resolved_wc_kg_m2_s": fields["resolved_wc"],
                "sgs_wc_kg_m2_s": fields["sgs_wc"],
                "total_wc_kg_m2_s": fields["resolved_wc"] + fields["sgs_wc"],
                "resolved_scalar_flux": fields["resolved_wc"],
                "sgs_scalar_flux": fields["sgs_wc"],
                "total_scalar_flux": fields["resolved_wc"] + fields["sgs_wc"],
                "resolved_tke_sgs_transfer_m2_s3": fields[
                    "resolved_tke_sgs_transfer"
                ],
                "momentum_diffusivity_m2_s": fields["momentum_diffusivity"],
                "scalar_diffusivity_m2_s": fields["scalar_diffusivity"],
                "sgs_scalar_variance": fields["sgs_scalar_variance"],
                "pressure_variance_m4_s4": fields["pressure_variance"],
                "w_third_moment_m3_s3": fields["w_third_moment"],
                "updraft_fraction": fields["updraft_fraction"],
                "updraft_w_m_s": fields["updraft_w"],
                "updraft_scalar_excess": fields["updraft_scalar_excess"],
                "resolved_energy_vertical_transport_m3_s3": fields[
                    "resolved_energy_vertical_transport"
                ],
                "pressure_vertical_transport_m3_s3": fields[
                    "pressure_vertical_transport"
                ],
            }
        )
    np.savetxt(
        path,
        np.column_stack(tuple(columns.values())),
        delimiter=",",
        header=",".join(columns),
        comments="",
    )


def _write_spectra(
    path: Path,
    case: BoussinesqCase,
    statistics: ProfileStatistics,
) -> None:
    spectra = statistics.spectra()
    selected = spectra["mode"] > 0.0
    modes = spectra["mode"][selected]
    ustar = statistics.mean_ustar
    coriolis = _rotation_values(case)[0]
    scalar_flux = case.scalar_scales.from_execution_flux(
        case.model.scalar_boundary.lower_flux
    )
    if statistics.surface_scalar_flux_count:
        scalar_flux = statistics.mean_surface_scalar_flux
    concentration_scale = scalar_flux / ustar
    wavenumber = 2.0 * np.pi * modes / case.physical_grid.lx
    columns = {
        "k_reference_length": wavenumber * case.diagnostic_reference.length_m,
        "k_ustar_over_f": (
            wavenumber * ustar / coriolis
            if coriolis != 0.0
            else np.full(modes.shape, np.nan)
        ),
        "kEu_over_ustar2": modes * spectra["u"][selected] / ustar**2,
        "kEv_over_ustar2": modes * spectra["v"][selected] / ustar**2,
        "kEw_over_ustar2": modes * spectra["w"][selected] / ustar**2,
        "kEc_over_cstar2": (
            modes * spectra["scalar"][selected] / concentration_scale**2
            if concentration_scale != 0.0
            else np.full(modes.shape, np.nan)
        ),
        "sample_height_m": spectra["height_m"][selected],
    }
    np.savetxt(
        path,
        np.column_stack(tuple(columns.values())),
        delimiter=",",
        header=",".join(columns),
        comments="",
    )


def _write_radial_spectra(
    path: Path,
    statistics: ProfileStatistics,
) -> None:
    spectra = statistics.spectra()
    selected = spectra["radial_wavenumber_reference"] > 0.0
    columns = {
        "wavenumber_reference_length": spectra[
            "radial_wavenumber_reference"
        ][selected],
        "horizontal_energy": spectra["radial_horizontal"][selected],
        "vertical_energy": spectra["radial_w"][selected],
        "scalar_energy": spectra["radial_scalar"][selected],
        "sample_height_m": spectra["height_m"][selected],
    }
    np.savetxt(
        path,
        np.column_stack(tuple(columns.values())),
        delimiter=",",
        header=",".join(columns),
        comments="",
    )


def _bulk_metrics(
    case: BoussinesqCase,
    statistics: ProfileStatistics,
) -> dict[str, float]:
    """Compute normalization-driven metrics without identifying a case."""

    if statistics.count == 0:
        return {}
    metrics = {
        "surface_friction_velocity_m_s": statistics.mean_ustar,
        "surface_friction_velocity_ratio": (
            statistics.mean_ustar / case.diagnostic_reference.velocity_m_s
        )
    }
    if statistics.surface_scalar_flux_count:
        metrics["surface_scalar_flux"] = statistics.mean_surface_scalar_flux
    if statistics.diagnostic_count == 0:
        return metrics
    profiles = statistics.profiles()
    z = (
        np.arange(case.physical_grid.nz, dtype=np.float64) + 0.5
    ) * case.physical_grid.dz
    search = z <= case.diagnostic_reference.inversion_search_max_height_m
    if not np.any(search):
        return metrics
    total_flux = profiles["resolved_wc"] + profiles["sgs_wc"]
    selected_indices = np.flatnonzero(search)
    inversion_index = selected_indices[np.argmin(total_flux[search])]
    boundary_height = float(z[inversion_index])
    surface_flux = (
        statistics.mean_surface_scalar_flux
        if statistics.surface_scalar_flux_count
        else case.scalar_scales.from_execution_flux(
            case.model.scalar_boundary.lower_flux
        )
    )
    buoyancy_coefficient = (
        case.scalar_scales.from_execution_buoyancy_coefficient(
            case.model.buoyancy.acceleration_per_temperature
        )
    )
    buoyancy_velocity = float(
        np.cbrt(buoyancy_coefficient * surface_flux * boundary_height)
    )
    metrics.update(
        {
            "boundary_layer_height_m": boundary_height,
            "boundary_layer_height_ratio": (
                boundary_height / case.diagnostic_reference.length_m
            ),
            "buoyancy_velocity_ratio": (
                buoyancy_velocity / case.diagnostic_reference.velocity_m_s
            ),
            "entrainment_flux_ratio": (
                -float(total_flux[inversion_index]) / surface_flux
                if surface_flux != 0.0
                else math.nan
            ),
        }
    )
    return metrics


def _compare_metrics(
    measured: dict[str, float],
    reference: dict[str, Any],
    *,
    evaluate_acceptance: bool,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for name, bounds in reference.get("metrics", {}).items():
        value = measured.get(name)
        if "target" in bounds:
            lower = float(bounds["target"]) - float(bounds["tolerance"])
            upper = float(bounds["target"]) + float(bounds["tolerance"])
        else:
            lower = float(bounds["minimum"])
            upper = float(bounds["maximum"])
        comparisons[name] = {
            "value": value,
            "minimum": lower,
            "maximum": upper,
            "accepted": (
                lower <= value <= upper
                if evaluate_acceptance and value is not None
                else None
            ),
        }
    accepted = [item["accepted"] for item in comparisons.values()]
    return {
        "metrics": comparisons,
        "reference_acceptance_evaluated": evaluate_acceptance,
        "all_reference_metrics_accepted": (
            all(accepted) if evaluate_acceptance and accepted else None
        ),
    }


def evaluate(
    case: BoussinesqCase,
    *,
    output_dir: Path,
    restart: Path | None,
    max_steps: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    """Evaluate one already-composed case without selecting implementations."""

    if max_steps is not None and max_steps < 0:
        raise ValueError("maximum steps must be nonnegative")
    _configure_pressure_source()
    import jax

    jax.config.update("jax_enable_x64", case.pressure.dtype == "float64")
    import jax.numpy as jnp
    from jaxwind import build_jax_solver
    from jaxwind.domain import (
        AcceptedClock,
        VerticalBoundary,
    )
    from jaxwind.effects import (
        JaxRuntime,
        load_boussinesq_checkpoint,
        save_boussinesq_checkpoint,
    )
    runtime = JaxRuntime.from_initialized_jax(jax)

    if (
        restart is None
        and output_dir.exists()
        and any(output_dir.iterdir())
        and not overwrite
    ):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use --overwrite"
        )
    if restart is not None and not runtime.checkpoint_path(restart).exists():
        raise FileNotFoundError(runtime.checkpoint_path(restart))
    output_dir.mkdir(parents=True, exist_ok=True)
    if runtime.is_primary:
        (output_dir / "resolved_case.json").write_text(
            json.dumps(resolved(case), indent=2) + "\n"
        )
    runtime.synchronize("jaxwind-abl-output-ready")

    grid = case.mechanical_scales.to_execution_grid(case.physical_grid)

    def boundary(_clock, _environment):
        return VerticalBoundary(0.0, 0.0)

    solver = build_jax_solver(
        grid,
        runtime=runtime,
        model=case.model,
        integrator=case.integrator,
        normal_boundary=boundary,
        pressure_dtype=case.pressure.dtype,
        pressure_method=case.pressure.method,
        pressure_thomas_chunk=case.pressure.thomas_chunk,
        nonlinear_padding_ratio=case.nonlinear_padding_ratio,
    )
    closure_fingerprint, physics_fingerprint = _physics_fingerprints(case)
    scale_fingerprint = case.scalar_scales.fingerprint
    layout = solver.checkpoint_layout(jnp.asarray)
    statistics_path = output_dir / "statistics_latest.npz"
    latest_checkpoint = output_dir / "checkpoint_latest.npz"
    local_latest_checkpoint = runtime.checkpoint_path(latest_checkpoint)

    if restart is None:
        fields = build_initial_fields(
            case,
            jax=jax,
            jnp=jnp,
            solver=solver,
        )
        state = solver.cold_start(fields, clock=AcceptedClock(0.0, 0))
        statistics = ProfileStatistics(case.physical_grid.nz)
    else:
        state = load_boussinesq_checkpoint(
            runtime.checkpoint_path(restart),
            layout=layout,
            config=case.integrator,
            scale_fingerprint=scale_fingerprint,
            closure_fingerprint=closure_fingerprint,
            physics_fingerprint=physics_fingerprint,
        )
        restart_statistics = restart.parent / "statistics_latest.npz"
        statistics = (
            ProfileStatistics.load(restart_statistics, case.physical_grid.nz)
            if restart_statistics.exists()
            else ProfileStatistics(case.physical_grid.nz)
        )

    if state.clock.step > case.steps:
        raise ValueError("restart lies beyond the configured final step")
    remaining = case.steps - state.clock.step
    steps_to_run = remaining if max_steps is None else min(remaining, max_steps)
    initial_step = state.clock.step
    advance = solver.advance

    history_path = output_dir / "history.csv"
    fieldnames = (
        "step",
        "time_hours",
        "cfl_x",
        "cfl_y",
        "cfl_z",
        "maximum_cfl",
        "lasd_trajectory_cfl",
        "ustar_m_s",
        "maximum_execution_divergence",
        "surface_scalar",
        "surface_scalar_flux",
        "obukhov_length_m",
        "surface_uw_m2_s2",
        "surface_vw_m2_s2",
        "momentum_stationarity_cu",
        "momentum_stationarity_cv",
        "integrated_resolved_tke_m3_s2",
        "integrated_sgs_tke_m3_s2",
        "integrated_total_tke_m3_s2",
        "elapsed_seconds",
    )
    prior_history: list[dict[str, str]] = []
    if runtime.is_primary and restart is not None and history_path.exists():
        with history_path.open(newline="") as stream:
            prior_history = [
                row
                for row in csv.DictReader(stream)
                if int(row["step"]) <= state.clock.step
            ]
    history_stream = (
        history_path.open("w", newline="")
        if runtime.is_primary
        else io.StringIO()
    )
    writer = csv.DictWriter(history_stream, fieldnames=fieldnames, restval=math.nan)
    writer.writeheader()
    writer.writerows(prior_history)
    history_stream.flush()

    latest_diagnostic: dict[str, float] = {}
    warned_cfl = False
    started = time.perf_counter()
    timing_warmup_steps = min(
        steps_to_run,
        2 * case.model.momentum.sgs.update_interval,
    )
    timing_end_step = max(timing_warmup_steps, steps_to_run - 1)
    timing_warmup_elapsed: float | None = None
    timing_end_elapsed: float | None = None
    solver_elapsed: float | None = None
    try:
        for local_step in range(1, steps_to_run + 1):
            next_step = state.clock.step + 1
            should_log = (
                next_step % case.output.log_every_steps == 0
                or local_step == steps_to_run
            )
            result = advance(
                state,
                compute_projection_residual=should_log,
            )
            state = result.state
            if local_step == timing_warmup_steps:
                state.fields.velocity.x.payload.block_until_ready()
                timing_warmup_elapsed = time.perf_counter() - started
            if local_step == timing_end_step:
                state.fields.velocity.x.payload.block_until_ready()
                timing_end_elapsed = time.perf_counter() - started
            if local_step == steps_to_run:
                state.fields.velocity.x.payload.block_until_ready()
                solver_elapsed = time.perf_counter() - started
            accepted_step = state.clock.step
            should_sample = (
                accepted_step >= case.output.sample_start_step
                and (accepted_step - case.output.sample_start_step)
                % case.output.sample_every_steps
                == 0
            )
            paper_history: dict[str, float] = {}
            if should_sample or should_log:
                (
                    observed_fields,
                    paper_history,
                    paper_profiles,
                    paper_spectra,
                ) = _diagnostic_observables(
                    state,
                    result.diagnostic.projection.pressure,
                    case,
                    solver,
                    jax,
                    jnp,
                    include_spectra=should_sample,
                )
            if should_sample:
                statistics.sample(
                    observed_fields["u"],
                    observed_fields["v"],
                    observed_fields["w"],
                    observed_fields["scalar"],
                    ustar=paper_history["ustar_m_s"],
                    surface_scalar_flux=paper_history["surface_scalar_flux"],
                    diagnostics=paper_profiles,
                    spectra=paper_spectra,
                )

            if should_log:
                latest_diagnostic = _diagnostics(
                    state,
                    result.diagnostic.projection.divergence.payload,
                    case,
                    jnp,
                    solver,
                )
                latest_diagnostic.update(paper_history)
                if latest_diagnostic["maximum_cfl"] >= case.cfl_abort:
                    raise RuntimeError("CFL abort limit reached")
                if (
                    latest_diagnostic["lasd_trajectory_cfl"]
                    >= case.trajectory_cfl_abort
                ):
                    raise RuntimeError("LASD trajectory CFL abort limit reached")
                if (
                    latest_diagnostic["maximum_cfl"] > case.cfl_warning
                    and not warned_cfl
                ):
                    warnings.warn(
                        "CFL warning limit exceeded: "
                        f"{latest_diagnostic['maximum_cfl']:.3f}",
                        stacklevel=1,
                    )
                    warned_cfl = True
                row = {
                    "step": accepted_step,
                    "time_hours": accepted_step * case.dt_seconds / 3600.0,
                    **latest_diagnostic,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                if runtime.is_primary:
                    dimensionless_time = (
                        accepted_step
                        * case.dt_seconds
                        * case.diagnostic_reference.velocity_m_s
                        / case.diagnostic_reference.length_m
                    )
                    writer.writerow(row)
                    history_stream.flush()
                    print(
                        f"step={accepted_step}/{case.steps} "
                        "t*="
                        f"{dimensionless_time:.3f} "
                        f"CFL={latest_diagnostic['maximum_cfl']:.3f} "
                        f"u*={latest_diagnostic['ustar_m_s']:.4f} m/s",
                        flush=True,
                    )

            should_checkpoint = (
                accepted_step % case.output.checkpoint_every_steps == 0
                or local_step == steps_to_run
            )
            if should_checkpoint:
                save_boussinesq_checkpoint(
                    local_latest_checkpoint,
                    state,
                    scale_fingerprint=scale_fingerprint,
                    physics_fingerprint=physics_fingerprint,
                )
                if runtime.is_primary:
                    statistics.save(statistics_path)
    finally:
        history_stream.close()

    if steps_to_run == 0:
        save_boussinesq_checkpoint(
            local_latest_checkpoint,
            state,
            scale_fingerprint=scale_fingerprint,
            physics_fingerprint=physics_fingerprint,
        )
        if runtime.is_primary:
            statistics.save(statistics_path)
    if runtime.is_primary and statistics.count:
        _write_profiles(output_dir / "profiles.csv", case, statistics)
    if runtime.is_primary and statistics.spectrum_count:
        _write_spectra(output_dir / "spectra.csv", case, statistics)
        _write_radial_spectra(output_dir / "radial_spectra.csv", statistics)
    if state.clock.step == case.steps:
        save_boussinesq_checkpoint(
            runtime.checkpoint_path(output_dir / "checkpoint_final.npz"),
            state,
            scale_fingerprint=scale_fingerprint,
            physics_fingerprint=physics_fingerprint,
        )

    reference = json.loads(case.reference_results.read_text())
    complete = state.clock.step == case.steps
    measured_metrics = _bulk_metrics(case, statistics)
    elapsed_seconds = time.perf_counter() - started
    post_warmup_steps = timing_end_step - timing_warmup_steps
    post_warmup_elapsed = (
        timing_end_elapsed - timing_warmup_elapsed
        if (
            timing_end_elapsed is not None
            and timing_warmup_elapsed is not None
            and post_warmup_steps
        )
        else None
    )
    summary = {
        **resolved(case),
        "runtime": {
            "jax_backend": jax.default_backend(),
            "process_count": runtime.process_count,
            "global_device_count": runtime.global_devices,
            "local_device_count": runtime.local_devices,
            "restart": None if restart is None else str(restart),
            "initial_step": initial_step,
            "steps_run": steps_to_run,
            "elapsed_seconds": elapsed_seconds,
            "steps_per_second": (
                steps_to_run / elapsed_seconds if steps_to_run else None
            ),
            "solver_elapsed_seconds": solver_elapsed,
            "timing_warmup_steps": timing_warmup_steps,
            "timing_warmup_elapsed_seconds": timing_warmup_elapsed,
            "timing_end_step": timing_end_step,
            "timing_end_elapsed_seconds": timing_end_elapsed,
            "post_warmup_steps": post_warmup_steps,
            "post_warmup_elapsed_seconds": post_warmup_elapsed,
            "post_warmup_steps_per_second": (
                post_warmup_steps / post_warmup_elapsed
                if post_warmup_elapsed
                else None
            ),
            "final_step": state.clock.step,
            "final_time_hours": state.clock.step * case.dt_seconds / 3600.0,
            "profile_samples": statistics.count,
            "diagnostic_profile_samples": statistics.diagnostic_count,
            "spectrum_samples": statistics.spectrum_count,
            "reached_final_time": complete,
            **latest_diagnostic,
        },
        "diagnostic_metrics": measured_metrics,
        "comparison": _compare_metrics(
            measured_metrics,
            reference,
            evaluate_acceptance=complete,
        ),
    }
    if runtime.is_primary:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    return summary


__all__ = ["ProfileStatistics", "evaluate", "resolved"]
