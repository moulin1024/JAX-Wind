"""Materialize and evaluate a fully composed ABL case."""

from __future__ import annotations

import csv
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
from applications.abl.config import derive_abl_stability


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
    )

    SPECTRUM_NAMES = (
        "mode",
        "u",
        "v",
        "w",
        "scalar",
        "height_m",
    )

    def __init__(self, nz: int) -> None:
        self.count = 0
        self.diagnostic_count = 0
        self.spectrum_count = 0
        self.ustar_sum = 0.0
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
    scalar_flux = case.scalar_scales.from_execution_concentration_flux(
        case.model.scalar_boundary.lower_flux
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
            "stability": derive_abl_stability(case),
            "advection": type(momentum.advection).__name__,
            "pressure_gradient": type(momentum.pressure_gradient).__name__,
            "wall": type(wall).__name__,
            "roughness_length_m": case.mechanical_scales.from_execution_length(
                wall.roughness_length
            ),
            "momentum_sgs": type(momentum.sgs).__name__,
            "scalar_sgs": type(case.model.scalar_sgs).__name__,
            "rotation": type(rotation).__name__,
            "coriolis_vertical_s": (
                case.mechanical_scales.from_execution_inverse_time(
                    rotation.coriolis_parameter
                )
            ),
            "coriolis_horizontal_s": (
                case.mechanical_scales.from_execution_inverse_time(
                    rotation.horizontal_coriolis_parameter
                )
            ),
            "geostrophic_velocity_m_s": [
                case.mechanical_scales.from_execution_velocity(
                    rotation.geostrophic_x_velocity
                ),
                case.mechanical_scales.from_execution_velocity(
                    rotation.geostrophic_y_velocity
                ),
            ],
            "passive_scalar_surface_flux_kg_m2_s": scalar_flux,
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
        },
        "output": {
            "directory": str(case.output.directory),
            "sample_every_steps": case.output.sample_every_steps,
            "log_every_steps": case.output.log_every_steps,
            "checkpoint_every_steps": case.output.checkpoint_every_steps,
        },
    }


def _physics_fingerprints(case: BoussinesqCase) -> tuple[str, str]:
    momentum = case.model.momentum.sgs
    scalar = case.model.scalar_sgs
    closure = momentum.fingerprint + "|" + scalar.fingerprint
    physics = (
        closure
        + f"|advection={type(case.model.momentum.advection).__name__}"
        + f"|wall={type(case.model.momentum.wall).__name__}"
        + f"|rotation={type(case.model.momentum.rotation).__name__}"
        + f"|padding={case.nonlinear_padding_ratio.hex()}"
    )
    return closure, physics


def _physical_fields(state, case: BoussinesqCase, jnp):
    shape = (
        case.physical_grid.nz,
        case.physical_grid.ny,
        case.physical_grid.nx,
    )
    velocity = state.fields.velocity
    u = case.mechanical_scales.from_execution_velocity(
        velocity.x.payload
    ).reshape(shape)
    v = case.mechanical_scales.from_execution_velocity(
        velocity.y.payload
    ).reshape(shape)
    w_upper = case.mechanical_scales.from_execution_velocity(
        velocity.z.owned.payload
    ).reshape(shape)
    lower = jnp.concatenate((jnp.zeros_like(w_upper[:1]), w_upper[:-1]), axis=0)
    w = 0.5 * (lower + w_upper)
    scalar = case.scalar_scales.from_execution_concentration(
        state.fields.potential_temperature.payload
    ).reshape(shape)
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


def _diagnostic_observables(
    state,
    case: BoussinesqCase,
    algebra,
    jax,
    jnp,
    *,
    include_spectra: bool,
):
    """Evaluate paper-aligned instantaneous diagnostics in physical units."""

    u, v, w, _w_upper, scalar = _physical_fields(state, case, jnp)
    mean_u = jnp.mean(u, axis=(-2, -1))
    mean_v = jnp.mean(v, axis=(-2, -1))
    mean_w = jnp.mean(w, axis=(-2, -1))
    mean_scalar = jnp.mean(scalar, axis=(-2, -1))
    u_fluctuation = u - mean_u[:, None, None]
    v_fluctuation = v - mean_v[:, None, None]
    w_fluctuation = w - mean_w[:, None, None]
    scalar_fluctuation = scalar - mean_scalar[:, None, None]
    tau_x, tau_y, ustar = _wall_stress(u, v, case, jnp)

    context = algebra.boussinesq_context(state.fields)
    from jaxwind.physics import DiagnosticLasdConstants

    diagnostic = algebra.lasd_diagnostic_fields(
        context,
        case.model.momentum.sgs,
        case.model.scalar_sgs,
        case.model.scalar_boundary,
        constants=DiagnosticLasdConstants(horizontal_homogeneous_wall=True),
        wall=case.model.momentum.wall,
    )
    shape = u.shape
    velocity_squared = case.mechanical_scales.kinematic_pressure
    sgs_tke = diagnostic.sgs_tke.reshape(shape) * velocity_squared
    momentum_diffusivity = diagnostic.momentum_diffusivity.reshape(shape) * (
        case.mechanical_scales.kinematic_viscosity
    )
    scalar_diffusivity = diagnostic.scalar_diffusivity.reshape(shape) * (
        case.mechanical_scales.kinematic_viscosity
    )
    scalar_flux_upper = case.scalar_scales.from_execution_concentration_flux(
        diagnostic.scalar_flux_z.reshape(shape)
    )
    surface_scalar_flux = case.scalar_scales.from_execution_concentration_flux(
        case.model.scalar_boundary.lower_flux
    )
    lower_scalar_flux = jnp.concatenate(
        (
            jnp.full_like(scalar_flux_upper[:1], surface_scalar_flux),
            scalar_flux_upper[:-1],
        ),
        axis=0,
    )
    sgs_scalar_flux = 0.5 * (lower_scalar_flux + scalar_flux_upper)

    resolved_tke_sgs_transfer = algebra.momentum_sgs_tke_transfer(
        context.momentum,
        case.model.momentum.sgs,
        wall=case.model.momentum.wall,
    ).reshape(shape) * (
        case.mechanical_scales.kinematic_pressure
        * case.mechanical_scales.inverse_time
    )
    txz_upper, tyz_upper = algebra.sgs_vertical_flux(
        context.momentum,
        case.model.momentum.sgs,
    )
    txz_upper = txz_upper.reshape(shape) * velocity_squared
    tyz_upper = tyz_upper.reshape(shape) * velocity_squared
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
    diagnostics = {
        "resolved_uw": profile(u_fluctuation * w_fluctuation),
        "resolved_vw": profile(v_fluctuation * w_fluctuation),
        "resolved_wc": profile(w_fluctuation * scalar_fluctuation),
        "sgs_tke": sgs_tke_profile,
        "sgs_uw": profile(0.5 * (lower_txz + txz_upper)),
        "sgs_vw": profile(0.5 * (lower_tyz + tyz_upper)),
        "sgs_wc": profile(sgs_scalar_flux),
        "resolved_tke_sgs_transfer": profile(resolved_tke_sgs_transfer),
        "momentum_diffusivity": profile(momentum_diffusivity),
        "scalar_diffusivity": profile(scalar_diffusivity),
    }

    mean_u_host, mean_v_host = jax.device_get((mean_u, mean_v))
    mean_u_host = np.asarray(mean_u_host, dtype=np.float64)
    mean_v_host = np.asarray(mean_v_host, dtype=np.float64)
    surface_uw = float(jnp.mean(tau_x))
    surface_vw = float(jnp.mean(tau_y))
    rotation = case.model.momentum.rotation
    coriolis = case.mechanical_scales.from_execution_inverse_time(
        rotation.coriolis_parameter
    )
    geostrophic_u = case.mechanical_scales.from_execution_velocity(
        rotation.geostrophic_x_velocity
    )
    geostrophic_v = case.mechanical_scales.from_execution_velocity(
        rotation.geostrophic_y_velocity
    )
    integrated_u_deficit = np.sum(mean_u_host - geostrophic_u) * (
        case.physical_grid.dz
    )
    integrated_v_deficit = np.sum(mean_v_host - geostrophic_v) * (
        case.physical_grid.dz
    )
    history = {
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
        target = int(np.argmin(np.abs(z * coriolis / ustar - 0.1)))
        modes = np.arange(case.physical_grid.nx // 2 + 1, dtype=np.float64)
        spectra = {
            "mode": modes,
            "u": _x_spectrum(fields["u"], target),
            "v": _x_spectrum(fields["v"], target),
            "w": _x_spectrum(fields["w"], target),
            "scalar": _x_spectrum(fields["scalar"], target),
            "height_m": np.full_like(modes, z[target]),
        }
    return fields, history, diagnostics, spectra


def _diagnostics(state, divergence, case: BoussinesqCase, jnp):
    u, v, _w, w_upper, _scalar = _physical_fields(state, case, jnp)
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
        "lasd_trajectory_cfl": maximum_cfl * update_interval,
        "ustar_m_s": _friction_velocity(u, v, case, jnp),
        "maximum_execution_divergence": float(jnp.max(jnp.abs(divergence))),
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
    coriolis = case.mechanical_scales.from_execution_inverse_time(
        case.model.momentum.rotation.coriolis_parameter
    )
    columns = {
        "z_m": z,
        "z_f_over_ustar": z * coriolis / ustar,
        "mean_u_m_s": fields["u"],
        "mean_v_m_s": fields["v"],
        "mean_w_m_s": fields["w"],
        "mean_scalar_kg_m3": fields["scalar"],
        "resolved_u_variance_m2_s2": fields["u_variance"],
        "resolved_v_variance_m2_s2": fields["v_variance"],
        "resolved_w_variance_m2_s2": fields["w_variance"],
        "resolved_tke_m2_s2": resolved_tke,
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
                "resolved_tke_sgs_transfer_m2_s3": fields[
                    "resolved_tke_sgs_transfer"
                ],
                "momentum_diffusivity_m2_s": fields["momentum_diffusivity"],
                "scalar_diffusivity_m2_s": fields["scalar_diffusivity"],
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
    coriolis = case.mechanical_scales.from_execution_inverse_time(
        case.model.momentum.rotation.coriolis_parameter
    )
    scalar_flux = case.scalar_scales.from_execution_concentration_flux(
        case.model.scalar_boundary.lower_flux
    )
    concentration_scale = scalar_flux / ustar
    wavenumber = 2.0 * np.pi * modes / case.physical_grid.lx
    columns = {
        "k_ustar_over_f": wavenumber * ustar / coriolis,
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
    from spectral_fd import runtime_from_initialized_jax

    from jaxwind import build_solver
    from jaxwind.domain import (
        AcceptedClock,
        DistributionSpec,
        EqualZSlab,
        MeshAxis,
        MeshTopology,
        VerticalBoundary,
    )
    from jaxwind.effects import (
        ZSlabCheckpointLayout,
        load_boussinesq_checkpoint,
        save_boussinesq_checkpoint,
    )
    from jaxwind.integrators import cold_start_boussinesq
    from jaxwind.interpreters.jax_zslab import build_zslab_interpreter
    from jaxwind.physics import (
        BoussinesqVectorField,
        LasdAcceptedStepEvent,
    )
    from jaxwind.pressure import build_spectral_fd_pressure_adapter

    if jax.process_count() != 1 or jax.device_count() != 1:
        raise RuntimeError("this case requires one JAX process and one device")

    if (
        restart is None
        and output_dir.exists()
        and any(output_dir.iterdir())
        and not overwrite
    ):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use --overwrite"
        )
    if restart is not None and not restart.exists():
        raise FileNotFoundError(restart)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_case.json").write_text(
        json.dumps(resolved(case), indent=2) + "\n"
    )

    grid = case.mechanical_scales.to_execution_grid(case.physical_grid)
    decomposition = EqualZSlab(
        grid,
        MeshTopology((MeshAxis("z", 1),)),
        DistributionSpec.z_slab(),
    )
    algebra = build_zslab_interpreter(
        decomposition,
        addressable_shards=(0,),
        nonlinear_padding_ratio=case.nonlinear_padding_ratio,
    )
    pressure_solver = build_spectral_fd_pressure_adapter(
        decomposition,
        addressable_shards=(0,),
        runtime=runtime_from_initialized_jax(jax),
        dtype=case.pressure.dtype,
        method=case.pressure.method,
        thomas_chunk=case.pressure.thomas_chunk,
    )
    closure_fingerprint, physics_fingerprint = _physics_fingerprints(case)
    scale_fingerprint = case.scalar_scales.fingerprint
    layout = ZSlabCheckpointLayout(decomposition, (0,), jnp.asarray)
    statistics_path = output_dir / "statistics_latest.npz"
    latest_checkpoint = output_dir / "checkpoint_latest.npz"

    if restart is None:
        fields = build_initial_fields(
            case,
            jax=jax,
            jnp=jnp,
            decomposition=decomposition,
            algebra=algebra,
            pressure_solver=pressure_solver,
        )
        state = cold_start_boussinesq(
            fields,
            clock=AcceptedClock(0.0, 0),
            config=case.integrator,
        )
        statistics = ProfileStatistics(case.physical_grid.nz)
    else:
        state = load_boussinesq_checkpoint(
            restart,
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
    vector_field = BoussinesqVectorField(algebra, case.model)
    closure_event = LasdAcceptedStepEvent(
        algebra,
        case.model,
        case.integrator.dt,
    )

    def boundary(_clock, _environment):
        return VerticalBoundary(0.0, 0.0)

    advance = build_solver(
        config=case.integrator,
        vector_field=vector_field,
        normal_boundary=boundary,
        algebra=algebra,
        pressure_solver=pressure_solver,
        closure_event=closure_event,
    )

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
    if restart is not None and history_path.exists():
        with history_path.open(newline="") as stream:
            prior_history = [
                row
                for row in csv.DictReader(stream)
                if int(row["step"]) <= state.clock.step
            ]
    history_stream = history_path.open("w", newline="")
    writer = csv.DictWriter(history_stream, fieldnames=fieldnames, restval=math.nan)
    writer.writeheader()
    writer.writerows(prior_history)
    history_stream.flush()

    latest_diagnostic: dict[str, float] = {}
    coriolis = case.mechanical_scales.from_execution_inverse_time(
        case.model.momentum.rotation.coriolis_parameter
    )
    warned_cfl = False
    started = time.perf_counter()
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
                    case,
                    algebra,
                    jax,
                    jnp,
                    include_spectra=should_sample,
                )
            if should_sample:
                ustar = _friction_velocity(
                    observed_fields["u"],
                    observed_fields["v"],
                    case,
                    np,
                )
                statistics.sample(
                    observed_fields["u"],
                    observed_fields["v"],
                    observed_fields["w"],
                    observed_fields["scalar"],
                    ustar=ustar,
                    diagnostics=paper_profiles,
                    spectra=paper_spectra,
                )

            if should_log:
                latest_diagnostic = _diagnostics(
                    state,
                    result.diagnostic.projection.divergence.payload,
                    case,
                    jnp,
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
                writer.writerow(row)
                history_stream.flush()
                print(
                    f"step={accepted_step}/{case.steps} "
                    f"tf={accepted_step * case.dt_seconds * coriolis:.3f} "
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
                    latest_checkpoint,
                    state,
                    scale_fingerprint=scale_fingerprint,
                    physics_fingerprint=physics_fingerprint,
                )
                statistics.save(statistics_path)
    finally:
        history_stream.close()

    if steps_to_run == 0:
        save_boussinesq_checkpoint(
            latest_checkpoint,
            state,
            scale_fingerprint=scale_fingerprint,
            physics_fingerprint=physics_fingerprint,
        )
        statistics.save(statistics_path)
    if statistics.count:
        _write_profiles(output_dir / "profiles.csv", case, statistics)
    if statistics.spectrum_count:
        _write_spectra(output_dir / "spectra.csv", case, statistics)
    if state.clock.step == case.steps:
        save_boussinesq_checkpoint(
            output_dir / "checkpoint_final.npz",
            state,
            scale_fingerprint=scale_fingerprint,
            physics_fingerprint=physics_fingerprint,
        )

    geostrophic_speed = math.hypot(
        case.mechanical_scales.from_execution_velocity(
            case.model.momentum.rotation.geostrophic_x_velocity
        ),
        case.mechanical_scales.from_execution_velocity(
            case.model.momentum.rotation.geostrophic_y_velocity
        ),
    )
    ustar = (
        statistics.mean_ustar
        if statistics.count
        else latest_diagnostic.get("ustar_m_s")
    )
    reference = json.loads(case.reference_results.read_text())
    published = tuple(reference["ustar_over_ug"].values())
    complete = state.clock.step == case.steps
    summary = {
        **resolved(case),
        "runtime": {
            "jax_backend": jax.default_backend(),
            "restart": None if restart is None else str(restart),
            "initial_step": initial_step,
            "steps_run": steps_to_run,
            "final_step": state.clock.step,
            "final_time_hours": state.clock.step * case.dt_seconds / 3600.0,
            "profile_samples": statistics.count,
            "diagnostic_profile_samples": statistics.diagnostic_count,
            "spectrum_samples": statistics.spectrum_count,
            "reached_final_time": complete,
            **latest_diagnostic,
        },
        "comparison": {
            "ustar_over_ug": None if ustar is None else ustar / geostrophic_speed,
            "published_ustar_over_ug_min": min(published),
            "published_ustar_over_ug_max": max(published),
            "reference_acceptance_evaluated": complete,
            "inside_published_envelope": (
                min(published) <= ustar / geostrophic_speed <= max(published)
                if complete and ustar is not None
                else None
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


__all__ = ["ProfileStatistics", "evaluate", "resolved"]
