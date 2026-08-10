"""Restartable profile statistics and plots for the GABLS1 runner."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from .config import CaseConfig


class ProfileStatistics:
    """Restartable host accumulation of cell and native-face profiles."""

    CELL_NAMES = ("u", "v", "w", "theta", "u2", "v2", "w2", "theta2")
    FACE_NAMES = (
        "uw_resolved",
        "vw_resolved",
        "uw_sgs",
        "vw_sgs",
        "wtheta_resolved",
        "wtheta_sgs",
    )

    def __init__(self, nz: int) -> None:
        self.count = 0
        self.sums = {
            **{name: np.zeros(nz, dtype=np.float64) for name in self.CELL_NAMES},
            **{
                name: np.zeros(nz + 1, dtype=np.float64)
                for name in self.FACE_NAMES
            },
        }

    def sample(self, values: dict[str, np.ndarray]) -> None:
        for name in self.sums:
            self.sums[name] += np.asarray(values[name], dtype=np.float64)
        self.count += 1

    def save(self, path: Path) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez(stream, count=np.asarray(self.count), **self.sums)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path, nz: int) -> "ProfileStatistics":
        result = cls(nz)
        with np.load(path, allow_pickle=False) as archive:
            result.count = int(archive["count"])
            for name, template in result.sums.items():
                value = np.asarray(archive[name], dtype=np.float64)
                if value.shape != template.shape:
                    raise ValueError("GABLS1 statistics shape does not match grid")
                result.sums[name] = value.copy()
        return result

    def means(self) -> dict[str, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("no GABLS1 statistics samples were collected")
        return {name: value / self.count for name, value in self.sums.items()}


def write_profiles(
    output_dir: Path,
    case: CaseConfig,
    statistics: ProfileStatistics,
) -> None:
    fields = statistics.means()
    z = (np.arange(case.domain.nz) + 0.5) * case.domain.dz_m
    z_faces = np.arange(case.domain.nz + 1) * case.domain.dz_m
    variances = {
        name: np.maximum(fields[f"{name}2"] - fields[name] ** 2, 0.0)
        for name in ("u", "v", "w", "theta")
    }
    with (output_dir / "profiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "z_m",
                "mean_u_m_s",
                "mean_v_m_s",
                "mean_w_m_s",
                "mean_potential_temperature_k",
                "u_variance_m2_s2",
                "v_variance_m2_s2",
                "w_variance_m2_s2",
                "temperature_variance_k2",
                "resolved_tke_m2_s2",
            )
        )
        writer.writerows(
            zip(
                z,
                fields["u"],
                fields["v"],
                fields["w"],
                fields["theta"],
                variances["u"],
                variances["v"],
                variances["w"],
                variances["theta"],
                0.5 * (variances["u"] + variances["v"] + variances["w"]),
                strict=True,
            )
        )
    with (output_dir / "flux_profiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "z_face_m",
                "uw_resolved_m2_s2",
                "uw_sgs_m2_s2",
                "uw_total_m2_s2",
                "vw_resolved_m2_s2",
                "vw_sgs_m2_s2",
                "vw_total_m2_s2",
                "wtheta_resolved_k_m_s",
                "wtheta_sgs_k_m_s",
                "wtheta_total_k_m_s",
            )
        )
        writer.writerows(
            zip(
                z_faces,
                fields["uw_resolved"],
                fields["uw_sgs"],
                fields["uw_resolved"] + fields["uw_sgs"],
                fields["vw_resolved"],
                fields["vw_sgs"],
                fields["vw_resolved"] + fields["vw_sgs"],
                fields["wtheta_resolved"],
                fields["wtheta_sgs"],
                fields["wtheta_resolved"] + fields["wtheta_sgs"],
                strict=True,
            )
        )


def plot_profiles(output_dir: Path, case: CaseConfig) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    mpl_cache = output_dir / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib.pyplot as plt

    profiles = np.genfromtxt(output_dir / "profiles.csv", delimiter=",", names=True)
    fluxes = np.genfromtxt(
        output_dir / "flux_profiles.csv", delimiter=",", names=True
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    z = profiles["z_m"]
    axes[0, 0].plot(profiles["mean_u_m_s"], z, label="u")
    axes[0, 0].plot(profiles["mean_v_m_s"], z, label="v")
    axes[0, 0].set_xlabel("mean velocity [m/s]")
    axes[0, 0].legend()
    axes[0, 1].plot(profiles["mean_potential_temperature_k"], z)
    axes[0, 1].set_xlabel("potential temperature [K]")
    axes[0, 2].plot(profiles["resolved_tke_m2_s2"], z)
    axes[0, 2].set_xlabel("resolved TKE [m2/s2]")
    zf = fluxes["z_face_m"]
    axes[1, 0].plot(fluxes["uw_total_m2_s2"], zf, label="uw")
    axes[1, 0].plot(fluxes["vw_total_m2_s2"], zf, label="vw")
    axes[1, 0].set_xlabel("total momentum flux [m2/s2]")
    axes[1, 0].legend()
    axes[1, 1].plot(fluxes["wtheta_total_k_m_s"], zf)
    axes[1, 1].set_xlabel("total heat flux [K m/s]")
    axes[1, 2].plot(profiles["w_variance_m2_s2"], z)
    axes[1, 2].set_xlabel("vertical velocity variance [m2/s2]")
    for axis in axes.ravel():
        axis.set_ylabel("z [m]")
        axis.grid(True, alpha=0.3)
    fig.suptitle(f"GABLS1 {case.sgs.model.upper()}, hours 8--9 mean")
    fig.tight_layout()
    fig.savefig(output_dir / "gabls1_profiles.png", dpi=180)
    plt.close(fig)
