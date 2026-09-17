"""Plot saved fields from the uniform-inflow V80 water-spray run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    folder = args.directory
    status = json.loads((folder / "status.json").read_text())
    if status["status"] != "complete":
        raise RuntimeError("plotting final fields requires a completed run")
    data = np.load(folder / "fields.npz")
    x, y, z = (data[key] for key in ("x_m", "y_m", "z_m"))
    temperature, u = data["temperature_K"], data["u_m_s"]
    iz = int(np.searchsorted(z, 70.0))
    z_weight = (70.0 - z[iz - 1]) / (z[iz] - z[iz - 1])
    iy = int(np.searchsorted(y, 512.0))
    weight = (512.0 - y[iy - 1]) / (y[iy] - y[iy - 1])
    cooling = 300.0 - temperature
    # Interpolate the two neighboring cell centers to the source plane.
    vmax = max(float(np.max(cooling)), 0.01)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    panels = [
        (
            x,
            y,
            (1 - z_weight) * cooling[iz - 1] + z_weight * cooling[iz],
            axes[0, 0],
            "Cooling at z = 70 m (interpolated)",
            "K below 300 K",
            0.0,
            vmax,
            "Blues",
        ),
        (
            x,
            z,
            (1 - weight) * cooling[:, iy - 1, :] + weight * cooling[:, iy, :],
            axes[1, 0],
            "Cooling at y = 512 m (interpolated)",
            "K below 300 K",
            0.0,
            vmax,
            "Blues",
        ),
        (
            x,
            y,
            (1 - z_weight) * u[iz - 1] + z_weight * u[iz],
            axes[0, 1],
            "Streamwise velocity at hub height",
            "m/s",
            0.0,
            12.0,
            "viridis",
        ),
        (
            x,
            z,
            (1 - weight) * u[:, iy - 1, :] + weight * u[:, iy, :],
            axes[1, 1],
            "Streamwise velocity at y = 512 m (interpolated)",
            "m/s",
            0.0,
            12.0,
            "viridis",
        ),
    ]
    for xx, vertical, field, ax, title, label, lo, hi, cmap in panels:
        artist = ax.pcolormesh(
            xx, vertical, field, shading="nearest", vmin=lo, vmax=hi, cmap=cmap
        )
        ax.set(
            xlabel="x (m)", ylabel="y (m)" if ax in axes[0] else "z (m)", title=title
        )
        ax.set_xlim(192.0, 960.0)
        ax.set_ylim((384.0, 640.0) if ax in axes[0] else (0.0, 160.0))
        ax.axvline(256.0, color="white", lw=0.8, ls="--")
        fig.colorbar(artist, ax=ax, label=label)
    fig.suptitle(
        "Single V80 · t = {:.1f} s · 20 kg/s, 200 µm, 290 K water\n10 m/s inflow · 300 K air · RH 80% · transient startup".format(
            status["time_s"]
        )
    )
    fig.savefig(folder / "fields.png", dpi=150)
    plt.close(fig)
    rows = [
        json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()
    ]
    rows = list({r["step"]: r for r in rows}.values())
    rows.sort(key=lambda r: r["step"])
    t = [r["time_s"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for key, label in [
        ("dpm_injected_mass_kg", "Injected"),
        ("dpm_liquid_mass_kg", "Airborne liquid"),
        ("dpm_evaporated_mass_kg", "Evaporated"),
        ("dpm_trapped_mass_kg", "Ground trapped"),
        ("dpm_escaped_mass_kg", "Escaped"),
    ]:
        axes[0].plot(t, [r[key] for r in rows], label=label)
    axes[0].set(xlabel="Time (s)", ylabel="Water mass (kg)", title="Water inventory")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, [300.0 - r["minimum_temperature_K"] for r in rows])
    axes[1].set(
        xlabel="Time (s)",
        ylabel="Maximum local cooling (K)",
        title="Cell-average cooling",
    )
    for d in (1, 2, 4, 6, 8):
        axes[2].plot(
            t,
            [300.0 - r[f"x{d}D_rotor_area_temperature_K"] for r in rows],
            label=f"Near {d}D",
        )
    axes[2].set(
        xlabel="Time (s)",
        ylabel="Disk-area mean cooling (K)",
        title="Downstream 80 m sampling disks",
    )
    axes[2].legend(fontsize=8)
    fig.savefig(folder / "history.png", dpi=150)
    plt.close(fig)
    print(folder / "fields.png")
    print(folder / "history.png")


if __name__ == "__main__":
    main()
