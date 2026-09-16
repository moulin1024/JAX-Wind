"""Plot completed spray checkpoint fields and vertical carrier profiles.

These are instantaneous mechanism diagnostics, not additional experimental
validation targets or time-averaged flow statistics.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.run / "checkpoint.npz", allow_pickle=False) as data:
        header = json.loads(str(data["metadata"]))
        doc = header["resolved_case"]
        faces = [np.asarray(header["mesh"][a + "_faces"]) for a in "xyz"]
        x, y, z = [(f[:-1] + f[1:]) / 2 for f in faces]
        velocity = []
        for component, axis in zip("xyz", (2, 1, 0)):
            values = data[f"state/velocity/{component}"]
            velocity.append(
                0.5
                * (
                    np.take(values, range(values.shape[axis] - 1), axis=axis)
                    + np.take(values, range(1, values.shape[axis]), axis=axis)
                )
            )
        speed = np.sqrt(sum(v * v for v in velocity))
        temperature = (
            data["state/scalar"]
            + doc["physics"]["moisture"]["temperature_offset_k"]
            - 273.15
        )
        vapor = data["state/moisture/vapor"]
        vapor_fraction = 1000 * vapor / (1 + vapor)
        time = float(data["state/time"])
    _, ly, lz = doc["mesh"]["lengths_m"]
    fields = [speed, temperature, vapor_fraction]
    labels = [
        "Gas speed (m/s)",
        "Dry-bulb temperature (°C)",
        "Vapor mass fraction (g/kg moist air)",
    ]
    interpolators = [RegularGridInterpolator((z, y, x), f) for f in fields]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    zz, xx = np.meshgrid(z, x, indexing="ij")
    points = np.stack((zz, np.full_like(zz, ly / 2), xx), axis=-1)
    for i, (field, interpolate, label) in enumerate(zip(fields, interpolators, labels)):
        centre = interpolate(points)
        first = axes[0, i].pcolormesh(faces[0], faces[2], centre, shading="flat")
        axes[0, i].set(xlabel="x (m)", ylabel="z (m)", title=label)
        fig.colorbar(first, ax=axes[0, i])
        last = axes[1, i].pcolormesh(faces[1], faces[2], field[..., -1], shading="flat")
        axes[1, i].set(xlabel="y (m)", ylabel="z (m)", aspect="equal")
        sy, sz = np.meshgrid([0.0975, 0.2925, 0.4875], [0.0975, 0.2925, 0.4875])
        axes[1, i].scatter(sy, sz, facecolors="none", edgecolors="white", s=35)
        fig.colorbar(last, ax=axes[1, i])
    fig.suptitle(
        f"{args.run.name}: instantaneous t={time:g} s\nCentre plane (top); last interior plane and sensor positions (bottom)"
    )
    fig.savefig(args.output / "carrier_fields.svg")
    fig.savefig(args.output / "carrier_fields.png", dpi=150)
    plt.close(fig)

    stations = [0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9]
    fig, axes = plt.subplots(
        3, len(stations), figsize=(15, 8), sharey=True, layout="constrained"
    )
    profiles = []
    for j, station in enumerate(stations):
        sample_x = float(np.clip(station, x[0], x[-1]))
        points = np.stack(
            (z, np.full_like(z, ly / 2), np.full_like(z, sample_x)), axis=-1
        )
        values = [f(points) for f in interpolators]
        for i, value in enumerate(values):
            axes[i, j].plot(value, z - lz / 2)
            axes[i, j].grid(alpha=0.2)
            if j == 0:
                axes[i, j].set_ylabel("z − Lz/2 (m)")
            if i == 0:
                axes[i, j].set_title(f"x={station:g} m")
            axes[i, j].set_xlabel(labels[i], fontsize=8)
        profiles.append(
            {
                "station_x_m": station,
                "sample_x_m": sample_x,
                "z_m": z.tolist(),
                "gas_speed_m_s": values[0].tolist(),
                "temperature_c": values[1].tolist(),
                "vapor_mass_fraction_g_kg": values[2].tolist(),
            }
        )
    fig.suptitle("Instantaneous vertical profiles; CFD mechanism comparison only")
    fig.savefig(args.output / "vertical_profiles.svg")
    fig.savefig(args.output / "vertical_profiles.png", dpi=150)
    plt.close(fig)
    (args.output / "profiles.json").write_text(
        json.dumps(
            {
                "run": str(args.run),
                "time_seconds": time,
                "profiles": profiles,
                "interpretation": "Instantaneous profiles at y=Ly/2; no experimental velocity targets supplied by the cooling paper.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
