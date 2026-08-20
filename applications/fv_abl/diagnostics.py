"""Case-independent FV ABL statistics and artifact writers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from applications.initial_conditions import load_initial_profile


PROFILE_NAMES = (
    "u",
    "v",
    "w",
    "scalar",
    "u_variance",
    "v_variance",
    "w_variance",
    "scalar_variance",
    "resolved_uw",
    "resolved_vw",
    "resolved_wc",
    "resolved_uc",
    "resolved_vc",
    "sgs_tke",
    "sgs_uw",
    "sgs_vw",
    "sgs_wc",
    "sgs_uc",
    "sgs_vc",
    "resolved_tke_sgs_transfer",
    "momentum_diffusivity",
    "scalar_diffusivity",
    "pressure_variance",
    "w_third_moment",
    "updraft_fraction",
    "updraft_w",
    "updraft_scalar_excess",
    "resolved_energy_vertical_transport",
    "pressure_vertical_transport",
)


def steps_to_next_sample(current: int, start: int, interval: int) -> int:
    """Return the steps to the next configured sample after ``current``."""

    if current < start:
        return start - current
    return interval - (current - start) % interval


def _unit_plane_noise(jax, jnp, key, shape, dtype):
    noise = jax.random.uniform(key, shape, dtype, minval=-0.5, maxval=0.5)
    noise -= jnp.mean(noise, axis=(-2, -1), keepdims=True)
    rms = jnp.sqrt(jnp.mean(noise * noise, axis=(-2, -1), keepdims=True))
    return noise / jnp.maximum(rms, jnp.finfo(dtype).tiny)


def initial_fields(case, jax, jnp):
    """Materialize the shared tabulated mean-plus-RMS initial state."""

    table = load_initial_profile(case)
    grid = case.physical_grid
    dtype = getattr(jnp, case.pressure.dtype)
    shape = (grid.nz, grid.ny, grid.nx)
    keys = jax.random.split(jax.random.PRNGKey(case.initial_condition.seed), 3)
    u_noise = _unit_plane_noise(jax, jnp, keys[0], shape, dtype)
    v_noise = _unit_plane_noise(jax, jnp, keys[1], shape, dtype)
    coupled_noise = _unit_plane_noise(jax, jnp, keys[2], shape, dtype)

    def profile(name: str):
        return jnp.asarray(table[name], dtype)[:, None, None]

    u = profile("u_m_s") + profile("u_rms_m_s") * u_noise
    v = profile("v_m_s") + profile("v_rms_m_s") * v_noise
    w_upper = profile("w_upper_m_s") + profile("w_upper_rms_m_s") * coupled_noise
    w = jnp.concatenate((jnp.zeros_like(w_upper[:1]), w_upper), axis=0)
    w = w.at[-1].set(0.0)
    scalar = profile("scalar") + profile("scalar_rms") * coupled_noise
    return u, v, w, scalar


def _x_spectrum(values: np.ndarray, level: int) -> np.ndarray:
    signal = values[level] - np.mean(values[level], axis=-1, keepdims=True)
    coefficients = np.fft.rfft(signal, axis=-1) / signal.shape[-1]
    energy = np.mean(np.abs(coefficients) ** 2, axis=0)
    factors = np.full(energy.shape, 2.0, dtype=np.float64)
    factors[0] = 1.0
    if signal.shape[-1] % 2 == 0:
        factors[-1] = 1.0
    return energy * factors


class ProfileAccumulator:
    """Profile statistics with optional streamwise spectra."""

    def __init__(self, nz: int, nx: int) -> None:
        self.count = 0
        self.ustar_sum = 0.0
        self.sums = {
            name: np.zeros(nz, dtype=np.float64) for name in PROFILE_NAMES
        }
        spectrum_size = nx // 2 + 1
        self.spectrum_sums = {
            name: np.zeros(spectrum_size, dtype=np.float64)
            for name in ("u", "v", "w", "scalar")
        }

    def sample(self, fields, profiles, *, ustar: float, spectrum_level: int):
        for name in PROFILE_NAMES:
            self.sums[name] += np.asarray(profiles[name], dtype=np.float64)
        for name, values in zip(
            ("u", "v", "w", "scalar"), fields, strict=True
        ):
            self.spectrum_sums[name] += _x_spectrum(
                np.asarray(values), spectrum_level
            )
        self.ustar_sum += ustar
        self.count += 1

    def profiles(self) -> dict[str, np.ndarray]:
        return {name: values / self.count for name, values in self.sums.items()}

    def means(self) -> dict[str, np.ndarray]:
        return self.profiles()

    def spectra(self) -> dict[str, np.ndarray]:
        return {
            name: values / self.count
            for name, values in self.spectrum_sums.items()
        }

    @property
    def ustar(self) -> float:
        return self.ustar_sum / self.count


def _radial_spectrum(
    values: np.ndarray,
    *,
    dx: float,
    dy: float,
    edges: np.ndarray,
) -> np.ndarray:
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


class RadialAccumulator:
    """Profile statistics with horizontal radial spectra at configured levels."""

    def __init__(self, case) -> None:
        grid = case.physical_grid
        self.count = 0
        self.ustar_sum = 0.0
        self.sums = {
            name: np.zeros(grid.nz, dtype=np.float64)
            for name in PROFILE_NAMES
        }
        z = (np.arange(grid.nz, dtype=np.float64) + 0.5) * grid.dz
        self.levels = np.asarray(
            [
                int(np.argmin(np.abs(z - height)))
                for height in case.diagnostic_reference.spectrum_heights_m
            ],
            dtype=np.int64,
        )
        self.heights = z[self.levels]
        modes = grid.nx // 2 + 1
        self.radial_edges = np.linspace(
            0.0,
            np.hypot(np.pi / grid.dx, np.pi / grid.dy),
            modes + 1,
        )
        self.radial_wavenumber = 0.5 * (
            self.radial_edges[:-1] + self.radial_edges[1:]
        )
        shape = (self.levels.size, modes)
        self.radial_sums = {
            name: np.zeros(shape, dtype=np.float64)
            for name in ("horizontal", "w", "scalar")
        }

    def sample(self, fields, profiles, *, ustar: float, grid) -> None:
        for name in PROFILE_NAMES:
            self.sums[name] += np.asarray(profiles[name], dtype=np.float64)
        u, v, w, scalar = (np.asarray(values) for values in fields)
        for row, level in enumerate(self.levels):
            self.radial_sums["horizontal"][row] += _radial_spectrum(
                u[level], dx=grid.dx, dy=grid.dy, edges=self.radial_edges
            ) + _radial_spectrum(
                v[level], dx=grid.dx, dy=grid.dy, edges=self.radial_edges
            )
            self.radial_sums["w"][row] += _radial_spectrum(
                w[level], dx=grid.dx, dy=grid.dy, edges=self.radial_edges
            )
            self.radial_sums["scalar"][row] += _radial_spectrum(
                scalar[level],
                dx=grid.dx,
                dy=grid.dy,
                edges=self.radial_edges,
            )
        self.ustar_sum += ustar
        self.count += 1

    def profiles(self) -> dict[str, np.ndarray]:
        return {name: values / self.count for name, values in self.sums.items()}

    def radial_spectra(self) -> dict[str, np.ndarray]:
        return {
            name: values / self.count for name, values in self.radial_sums.items()
        }

    @property
    def ustar(self) -> float:
        return self.ustar_sum / self.count


ConvectiveAccumulator = RadialAccumulator


def profile_columns(case, accumulator) -> dict[str, np.ndarray]:
    fields = accumulator.profiles()
    grid = case.physical_grid
    z = (np.arange(grid.nz, dtype=np.float64) + 0.5) * grid.dz
    resolved_tke = 0.5 * (
        fields["u_variance"] + fields["v_variance"] + fields["w_variance"]
    )
    columns = {
        "z_m": z,
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
        "sgs_tke_m2_s2": fields["sgs_tke"],
        "resolved_scalar_variance": fields["scalar_variance"],
        "resolved_scalar_variance_kg2_m6": fields["scalar_variance"],
        "resolved_uw_m2_s2": fields["resolved_uw"],
        "resolved_vw_m2_s2": fields["resolved_vw"],
        "sgs_uw_m2_s2": fields["sgs_uw"],
        "sgs_vw_m2_s2": fields["sgs_vw"],
        "total_uw_m2_s2": fields["resolved_uw"] + fields["sgs_uw"],
        "total_vw_m2_s2": fields["resolved_vw"] + fields["sgs_vw"],
        "resolved_wc_kg_m2_s": fields["resolved_wc"],
        "resolved_uc": fields["resolved_uc"],
        "resolved_vc": fields["resolved_vc"],
        "sgs_wc_kg_m2_s": fields["sgs_wc"],
        "sgs_uc": fields["sgs_uc"],
        "sgs_vc": fields["sgs_vc"],
        "total_wc_kg_m2_s": fields["resolved_wc"] + fields["sgs_wc"],
        "resolved_scalar_flux": fields["resolved_wc"],
        "sgs_scalar_flux": fields["sgs_wc"],
        "total_scalar_flux": fields["resolved_wc"] + fields["sgs_wc"],
        "resolved_tke_sgs_transfer_m2_s3": fields[
            "resolved_tke_sgs_transfer"
        ],
        "momentum_diffusivity_m2_s": fields["momentum_diffusivity"],
        "scalar_diffusivity_m2_s": fields["scalar_diffusivity"],
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
    rotation = case.model.momentum.rotation
    coriolis = (
        case.mechanical_scales.from_execution_inverse_time(
            rotation.coriolis_parameter
        )
        if hasattr(rotation, "coriolis_parameter")
        else 0.0
    )
    if coriolis != 0.0:
        columns = {
            "z_m": z,
            "z_f_over_ustar": z * coriolis / accumulator.ustar,
            **{name: values for name, values in columns.items() if name != "z_m"},
        }
    return columns


def write_columns(path: Path, columns: dict[str, np.ndarray]) -> None:
    np.savetxt(
        path,
        np.column_stack(tuple(columns.values())),
        delimiter=",",
        header=",".join(columns),
        comments="",
    )


def write_profiles(path: Path, case, accumulator) -> None:
    write_columns(path, profile_columns(case, accumulator))


def write_streamwise_spectra(path: Path, case, accumulator) -> None:
    spectra = accumulator.spectra()
    modes = np.arange(case.physical_grid.nx // 2 + 1, dtype=np.float64)
    selected = modes > 0.0
    modes = modes[selected]
    ustar = accumulator.ustar
    rotation = case.model.momentum.rotation
    coriolis = case.mechanical_scales.from_execution_inverse_time(
        rotation.coriolis_parameter
    )
    scalar_flux = case.scalar_scales.from_execution_flux(
        case.model.scalar_boundary.lower_flux
    )
    concentration_scale = scalar_flux / ustar
    wavenumber = 2.0 * np.pi * modes / case.physical_grid.lx
    write_columns(
        path,
        {
            "k_reference_length": wavenumber
            * case.diagnostic_reference.length_m,
            "k_ustar_over_f": wavenumber * ustar / coriolis,
            "kEu_over_ustar2": modes * spectra["u"][selected] / ustar**2,
            "kEv_over_ustar2": modes * spectra["v"][selected] / ustar**2,
            "kEw_over_ustar2": modes * spectra["w"][selected] / ustar**2,
            "kEc_over_cstar2": modes
            * spectra["scalar"][selected]
            / concentration_scale**2,
            "sample_height_m": np.full(
                modes.shape, case.diagnostic_reference.spectrum_heights_m[0]
            ),
        },
    )


def write_radial_spectra(path: Path, case, accumulator) -> None:
    spectra = accumulator.radial_spectra()
    wavenumber = np.broadcast_to(
        accumulator.radial_wavenumber[None]
        * case.diagnostic_reference.length_m,
        spectra["horizontal"].shape,
    )
    heights = np.broadcast_to(
        accumulator.heights[:, None], spectra["horizontal"].shape
    )
    selected = wavenumber > 0.0
    write_columns(
        path,
        {
            "wavenumber_reference_length": wavenumber[selected],
            "horizontal_energy": spectra["horizontal"][selected],
            "vertical_energy": spectra["w"][selected],
            "scalar_energy": spectra["scalar"][selected],
            "sample_height_m": heights[selected],
        },
    )


def bulk_metrics(case, accumulator) -> dict[str, float]:
    profiles = accumulator.profiles()
    grid = case.physical_grid
    z = (np.arange(grid.nz, dtype=np.float64) + 0.5) * grid.dz
    search = z <= case.diagnostic_reference.inversion_search_max_height_m
    total_flux = profiles["resolved_wc"] + profiles["sgs_wc"]
    candidates = np.flatnonzero(search)
    inversion_index = candidates[np.argmin(total_flux[search])]
    boundary_height = float(z[inversion_index])
    surface_flux = case.scalar_scales.from_execution_flux(
        case.model.scalar_boundary.lower_flux
    )
    coefficient = case.scalar_scales.from_execution_buoyancy_coefficient(
        case.model.buoyancy.acceleration_per_temperature
    )
    buoyancy_velocity = float(
        np.cbrt(coefficient * surface_flux * boundary_height)
    )
    return {
        "surface_friction_velocity_m_s": accumulator.ustar,
        "boundary_layer_height_m": boundary_height,
        "boundary_layer_height_ratio": (
            boundary_height / case.diagnostic_reference.length_m
        ),
        "buoyancy_velocity_ratio": (
            buoyancy_velocity / case.diagnostic_reference.velocity_m_s
        ),
        "entrainment_flux_ratio": -float(total_flux[inversion_index])
        / surface_flux,
    }


def reference_comparison(case, metrics: dict[str, float]) -> dict:
    reference = json.loads(case.reference_results.read_text(encoding="utf-8"))
    compared = {}
    for name, specification in reference["metrics"].items():
        lower = specification["target"] - specification["tolerance"]
        upper = specification["target"] + specification["tolerance"]
        value = metrics[name]
        compared[name] = {
            "value": value,
            "target": specification["target"],
            "minimum": lower,
            "maximum": upper,
            "within_reference_envelope": lower <= value <= upper,
        }
    return compared


_profile_columns = profile_columns
_write_columns = write_columns
_write_profiles = write_profiles
_write_radial_spectra = write_radial_spectra
_write_spectra = write_streamwise_spectra


__all__ = [
    "ConvectiveAccumulator",
    "PROFILE_NAMES",
    "ProfileAccumulator",
    "RadialAccumulator",
    "bulk_metrics",
    "initial_fields",
    "profile_columns",
    "reference_comparison",
    "steps_to_next_sample",
    "write_columns",
    "write_profiles",
    "write_radial_spectra",
    "write_streamwise_spectra",
]
