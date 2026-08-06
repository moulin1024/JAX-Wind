#!/usr/bin/env python3
"""Plot a declarative pressure-driven neutral log-law result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from jaxwind.domain import AnalyticAxisMapping, RectilinearGrid


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _grid(config: dict) -> RectilinearGrid:
    specification = config["grid"]
    nx, ny, nz = (int(value) for value in specification["shape"])
    lx, ly, lz = (float(value) for value in specification["extent"])
    mappings = specification.get("mapping", {})

    def mapping(axis: str) -> AnalyticAxisMapping:
        value = mappings.get(axis)
        if value is None:
            return AnalyticAxisMapping()
        return AnalyticAxisMapping(
            function=str(value["function"]),
            focus=(None if value.get("focus") is None else float(value["focus"])),
            strength=float(value.get("strength", 0.0)),
        )

    return RectilinearGrid.analytic(
        nx,
        ny,
        nz,
        lx=lx,
        ly=ly,
        lz=lz,
        x=mapping("x"),
        y=mapping("y"),
        z=mapping("z"),
    )


def _filtered_log_denominator(
    lower: np.ndarray,
    upper: np.ndarray,
    roughness_length: float,
) -> np.ndarray:
    width = upper - lower
    integration_lower = np.maximum(lower, roughness_length)
    integration_upper = np.maximum(upper, roughness_length)

    def primitive(height: np.ndarray) -> np.ndarray:
        return height * np.log(height / roughness_length) - height

    return np.where(
        upper > roughness_length,
        primitive(integration_upper) - primitive(integration_lower),
        0.0,
    ) / width


def main() -> None:
    args = parse_args()
    config = json.loads(
        (args.result_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (args.result_dir / "summary.json").read_text(encoding="utf-8")
    )
    profile = np.genfromtxt(
        args.result_dir / "profiles.csv",
        delimiter=",",
        names=True,
    )
    history = np.atleast_1d(
        np.genfromtxt(
            args.result_dir / "history.csv",
            delimiter=",",
            names=True,
        )
    )

    grid = _grid(config)
    z = np.asarray(profile["z_m"], dtype=float)
    height = float(config["grid"]["extent"][2])
    ustar = float(config["momentum"]["friction_velocity"])
    roughness = float(config["momentum"]["roughness_length"])
    von_karman = float(config["momentum"].get("von_karman", 0.4))
    lower = np.asarray(grid.z_faces[:-1], dtype=float) - grid.z_faces[0]
    upper = np.asarray(grid.z_faces[1:], dtype=float) - grid.z_faces[0]
    if not np.allclose(z, np.asarray(grid.z_centers), rtol=2.0e-6, atol=2.0e-6):
        raise ValueError("profile heights do not match the resolved grid")

    denominator = _filtered_log_denominator(lower, upper, roughness)
    target_velocity = ustar / von_karman * denominator
    velocity_error_plus = (profile["mean_u_m_s"] - target_velocity) / ustar
    fit = (z / height >= 0.05) & (z / height <= 0.3)
    fitted_ustar = von_karman * np.dot(
        denominator[fit], profile["mean_u_m_s"][fit]
    ) / np.dot(denominator[fit], denominator[fit])
    rmse_plus = float(np.sqrt(np.mean(velocity_error_plus[fit] ** 2)))
    wall_ustar = (
        von_karman * float(profile["mean_u_m_s"][0]) / denominator[0]
    )

    output = args.output or args.result_dir / "neutral_loglaw_imex_profiles.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), constrained_layout=True)
    normalized_height = z / height

    axis = axes[0, 0]
    axis.plot(profile["mean_u_m_s"] / ustar, normalized_height, lw=2, label="LES")
    axis.plot(
        target_velocity / ustar,
        normalized_height,
        "k--",
        lw=1.5,
        label="FV-filtered log law",
    )
    axis.set_yscale("log")
    axis.set(
        xlabel=r"$\langle u\rangle/u_*$",
        ylabel=r"$z/H$",
        ylim=(0.8 * normalized_height[0], 1.0),
        title=f"fixed-$z_0$ fit: $u_*/u_{{*,target}}$={fitted_ustar / ustar:.3f}",
    )
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    axis.plot(velocity_error_plus, normalized_height, lw=2)
    axis.axvline(0.0, color="k", ls="--", lw=1)
    axis.axhspan(0.05, 0.3, color="0.88", label="fit interval")
    axis.set(
        xlabel=r"$(\langle u\rangle-U_{log,FV})/u_*$",
        ylabel=r"$z/H$",
        ylim=(0.0, 1.0),
        title=(
            f"fit RMSE={rmse_plus:.3f}; "
            "wall-cell mean-profile inferred "
            f"$u_*/u_{{*,target}}$={wall_ustar / ustar:.3f}"
        ),
    )
    axis.legend(fontsize=8)

    axis = axes[0, 2]
    for field, label in (
        ("var_u_m2_s2", r"$\langle u'^2\rangle$"),
        ("var_v_m2_s2", r"$\langle v'^2\rangle$"),
        ("var_w_m2_s2", r"$\langle w'^2\rangle$"),
    ):
        axis.plot(profile[field] / ustar**2, normalized_height, lw=1.8, label=label)
    axis.set(
        xlabel=r"resolved variance $/u_*^2$",
        ylabel=r"$z/H$",
        ylim=(0.0, 1.0),
    )
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    axis.plot(
        -profile["resolved_uw_m2_s2"] / ustar**2,
        normalized_height,
        lw=2,
        label=r"resolved $-\langle u'w'\rangle$",
    )
    axis.plot(
        -profile["resolved_vw_m2_s2"] / ustar**2,
        normalized_height,
        lw=1.5,
        label=r"resolved $-\langle v'w'\rangle$",
    )
    axis.plot(
        1.0 - normalized_height,
        normalized_height,
        "k--",
        lw=1.5,
        label="target total $1-z/H$",
    )
    axis.set(
        xlabel=r"stress $/u_*^2$",
        ylabel=r"$z/H$",
        ylim=(0.0, 1.0),
        title="resolved contribution (SGS stress not saved)",
    )
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    axis.plot(profile["sgs_viscosity_m2_s"], normalized_height, lw=2)
    axis.set(
        xlabel=r"$\langle\nu_{sgs}\rangle$ (m$^2$ s$^{-1}$)",
        ylabel=r"$z/H$",
        ylim=(0.0, 1.0),
    )

    axis = axes[1, 2]
    time_scale = float(config.get("display", {}).get("time_scale", height / ustar))
    nondimensional_time = history["time_s"] / time_scale
    axis.plot(nondimensional_time, history["advective_cfl"], "o-", label="CFL")
    axis.plot(nondimensional_time, history["diffusive_cfl"], "s-", label="CFLnu")
    axis.axhline(
        float(config["numerics"]["target_cfl"]),
        color="k",
        ls="--",
        lw=1,
        label="targets",
    )
    axis.axhline(
        float(config["numerics"]["target_diffusive_cfl"]),
        color="k",
        ls="--",
        lw=1,
    )
    stability_maximum = max(
        float(np.nanmax(history["advective_cfl"])),
        float(np.nanmax(history["diffusive_cfl"])),
        float(config["numerics"]["target_cfl"]),
        float(config["numerics"]["target_diffusive_cfl"]),
    )
    axis.set(
        xlabel=r"$t u_*/H$",
        ylabel="stability number",
        xlim=(0.0, float(config["time"]["end"]) / time_scale),
        ylim=(0.0, max(0.55, 1.08 * stability_maximum)),
    )
    axis.legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    integration = config["numerics"]["sgs_time_integration"]
    sgs_model = str(summary["sgs_model"]).replace("_", " ").upper()
    figure.suptitle(
        "Pressure-driven neutral log layer: "
        f"{grid.shape[2]}³ z-stretched {sgs_model} {integration}, "
        f"t*={float(summary['final_time_s']) * ustar / height:.2f}"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
