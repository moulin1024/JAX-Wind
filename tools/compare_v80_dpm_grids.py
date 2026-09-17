"""Compare completed V80 startups on coarse/refined grids at common physical planes."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def plane(field, centers, coordinate):
    upper = int(np.searchsorted(centers, coordinate))
    if not 0 < upper < len(centers):
        raise ValueError("comparison coordinate is outside interpolation support")
    fraction = (coordinate - centers[upper - 1]) / (centers[upper] - centers[upper - 1])
    return (1 - fraction) * field[..., upper - 1] + fraction * field[..., upper]


def disk_cell_areas(y, z, radius=40.0):
    """Circle/rectangle intersections; integrate the chord over each z interval."""
    dy, dz = y[1] - y[0], z[1] - z[0]
    areas = np.zeros((len(z), len(y)))
    for j, zc in enumerate(z - 70.0):
        low, high = max(zc - dz / 2, -radius), min(zc + dz / 2, radius)
        if low >= high:
            continue
        for i, yc in enumerate(y - 512.0):
            yl, yh = yc - dy / 2, yc + dy / 2
            if yl >= radius or yh <= -radius:
                continue
            points = []
            for edge in (yl, yh):
                if abs(edge) < radius:
                    root = np.sqrt(radius**2 - edge**2)
                    points.extend(p for p in (-root, root) if low < p < high)

            def width(zz, yl=yl, yh=yh):
                half = np.sqrt(max(radius**2 - zz**2, 0.0))
                return max(0.0, min(yh, half) - max(yl, -half))

            areas[j, i] = quad(
                width, low, high, points=sorted(set(points)), epsabs=1e-10, epsrel=1e-11
            )[0]
    np.testing.assert_allclose(np.sum(areas), np.pi * radius**2, rtol=2e-10)
    return areas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coarse", type=Path, default=Path("outputs/v80_uniform10_20kgs_200um_290K")
    )
    parser.add_argument(
        "--refined",
        type=Path,
        default=Path("outputs/v80_uniform10_20kgs_200um_290K_refined"),
    )
    args = parser.parse_args()
    result = {
        "definition": "60 s startup; linear x interpolation to common planes; circle/cell intersection areas for the same physical 80 m disk",
        "runs": {},
    }
    curves = {}
    for label, folder in [("coarse", args.coarse), ("refined", args.refined)]:
        status = json.loads((folder / "status.json").read_text())
        if status["status"] != "complete" or abs(status["time_s"] - 60.0) > 1e-9:
            raise ValueError("comparison requires both completed 60-second runs")
        fields = np.load(folder / "fields.npz")
        x, y, z = (fields[key] for key in ("x_m", "y_m", "z_m"))
        temp, u = fields["temperature_K"], fields["u_m_s"]
        yy, zz = np.meshgrid(y, z)
        mask = ((yy - 512.0) ** 2 + (zz - 70.0) ** 2) <= 40.0**2
        area = disk_cell_areas(y, z)
        mean_t = np.einsum("zy,zyx->x", area, temp) / np.sum(area)
        mean_u = np.einsum("zy,zyx->x", area, u) / np.sum(area)
        position = np.unravel_index(np.argmin(temp), temp.shape)
        metrics = {
            key: status[key]
            for key in [
                "dpm_injected_mass_kg",
                "dpm_evaporated_mass_kg",
                "dpm_liquid_mass_kg",
                "dpm_trapped_mass_kg",
                "dpm_escaped_mass_kg",
                "minimum_temperature_K",
                "maximum_relative_humidity",
                "maximum_cfl",
                "dpm_water_budget_error_kg",
                "carrier_vapor_budget_error_kg",
            ]
        }
        metrics["minimum_temperature_location_m"] = [
            float(x[position[2]]),
            float(y[position[1]]),
            float(z[position[0]]),
        ]
        metrics["maximum_local_cooling_K"] = 300.0 - status["minimum_temperature_K"]
        metrics["evaporated_fraction"] = (
            status["dpm_evaporated_mass_kg"] / status["dpm_injected_mass_kg"]
        )
        metrics["native_center_mask_area_m2"] = float(
            np.sum(mask) * (y[1] - y[0]) * (z[1] - z[0])
        )
        metrics["sampling_disk_area_m2"] = float(np.sum(area))
        metrics["downstream"] = []
        for d in [1, 2, 4, 6, 8]:
            coordinate = 256.0 + 80.0 * d
            metrics["downstream"].append(
                {
                    "x_over_D": d,
                    "x_m": coordinate,
                    "mean_temperature_K": float(plane(mean_t, x, coordinate)),
                    "mean_cooling_K": float(300.0 - plane(mean_t, x, coordinate)),
                    "mean_u_m_s": float(plane(mean_u, x, coordinate)),
                }
            )
        result["runs"][label] = metrics
        curves[label] = (x, mean_t, mean_u)
    output = args.refined / "grid_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for label, (x, temp, u) in curves.items():
        xd = (x - 256.0) / 80.0
        axes[0].plot(xd, 300.0 - temp, label=label)
        axes[1].plot(xd, u, label=label)
    axes[0].set(
        ylabel="Disk-area mean cooling (K below 300 K)",
        xlabel="Distance downstream (rotor diameters)",
        xlim=(0, 9),
    )
    axes[1].set(
        ylabel="Disk-area mean streamwise velocity (m/s)",
        xlabel="Distance downstream (rotor diameters)",
        xlim=(0, 9),
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("Single V80 at 60 s · coarse 16×16×4 m vs refined 8×8×2 m cells")
    fig.savefig(args.refined / "grid_comparison.png", dpi=150)
    print(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
