"""Plot and quantify a completed single-rotor open-boundary diagnostic run."""
import argparse
import json
from pathlib import Path
import tomllib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from render_fv_wake_gif import _ffmpeg_executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    doc = tomllib.loads((directory / "resolved_case.toml").read_text())
    layout = doc["physics"]["wind_farm"]["layout"]
    if len(layout) != 1:
        raise ValueError("this diagnostic requires exactly one rotor")
    rotor = layout[0]
    speed = doc["physics"]["inflow"]["speed_m_s"]
    with np.load(directory / "flow_frames.npz") as archive:
        u = archive["u_hub_yx"]
        times = archive["time_seconds"]
        xf, yf, zf = (archive[name] for name in ("x_faces_m", "y_faces_m", "z_faces_m"))
    assert len(times) == doc["time"]["frame_count"]
    assert np.isfinite(u).all()
    xc, yc, zc = ((f[:-1] + f[1:]) / 2 for f in (xf, yf, zf))
    upstream = xc < rotor["x_m"] - 160.
    row = int(np.argmin(abs(yc - rotor["y_m"])))
    hub = int(np.argmin(abs(zc - rotor["hub_height_m"])))
    # Keep raw faces: cell averaging can hide alternating-grid oscillations.
    with np.load(directory / "checkpoint.npz") as archive:
        faces = archive["state/velocity/x"]
        assert np.isfinite(faces).all()
        inlet_error = float(np.max(abs(faces[..., 0] - speed)))
        hub_faces = faces[hub].copy()
    upstream_faces = xf < rotor["x_m"] - 160.
    observed = u[:, :, upstream]
    minimum = observed.min(axis=(1, 2))
    maximum = observed.max(axis=(1, 2))
    metrics = {
        "time_seconds": float(times[-1]),
        "turbine_count": len(layout),
        "frame_count": len(times),
        "upstream_region_x_max_m": rotor["x_m"] - 160.,
        "final_inlet_face_max_abs_error_m_s": inlet_error,
        "final_upstream_hub_cell_min_m_s": float(minimum[-1]),
        "final_upstream_hub_cell_max_m_s": float(maximum[-1]),
        "final_upstream_hub_face_max_adjacent_jump_m_s": float(np.max(abs(np.diff(hub_faces[:, upstream_faces], axis=1)))),
        "final_outlet_hub_cell_min_m_s": float(u[-1, :, -1].min()),
        "final_frame_checkpoint_max_abs_difference_m_s": float(np.max(abs(u[-1] - .5 * (hub_faces[:, :-1] + hub_faces[:, 1:])))),
        "profile_y_m": float(yc[row]),
        "note": "Upstream extrema include induction and boundary effects; adjacent jumps alone do not identify the cause. Raw-face profile uses nearest hub-height cell.",
    }
    (directory / "debug_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, field, limits, title, cmap in (
        (axes[0, 0], u[-1], (0, 12), "Final hub-height u [m/s]", "viridis"),
        (axes[0, 1], u[-1] - speed, (-1, 1), "Final u − inlet [m/s] (clipped colour scale)", "RdBu_r"),
    ):
        mesh = ax.pcolormesh(xf, yf, field, vmin=limits[0], vmax=limits[1], cmap=cmap)
        ax.plot([rotor["x_m"]] * 2, [rotor["y_m"] - 40, rotor["y_m"] + 40], "k-")
        ax.set(title=title, xlabel="x [m]", ylabel="y [m]", aspect="equal")
        fig.colorbar(mesh, ax=ax)
    axes[1, 0].plot(times, minimum, label="upstream minimum")
    axes[1, 0].plot(times, maximum, label="upstream maximum")
    axes[1, 0].axhline(speed, color="k", linestyle=":")
    axes[1, 0].set(xlabel="Time [s]", ylabel="Hub-height u [m/s]", title=f"Upstream cells: x < {rotor['x_m'] - 160:g} m")
    axes[1, 0].legend()
    axes[1, 1].plot(xf[upstream_faces], hub_faces[row, upstream_faces], ".-", label="raw staggered faces", markersize=3)
    axes[1, 1].plot(xc[upstream], u[-1, row, upstream], label="cell averages")
    axes[1, 1].set(xlim=(0, rotor["x_m"] - 160), xlabel="x [m]", ylabel="u [m/s]", title=f"Final upstream profile at y={yc[row]:g} m")
    axes[1, 1].legend()
    fig.savefig(directory / "debug_summary.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    mesh = ax.pcolormesh(xf, yf, u[0], vmin=0, vmax=12, cmap="viridis")
    ax.plot([rotor["x_m"]] * 2, [rotor["y_m"] - 40, rotor["y_m"] + 40], "k-")
    ax.set(xlabel="x [m]", ylabel="y [m]", aspect="equal")
    fig.colorbar(mesh, ax=ax, label="u [m/s]")
    matplotlib.rcParams["animation.ffmpeg_path"] = _ffmpeg_executable()
    writer = FFMpegWriter(fps=10, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    with writer.saving(fig, str(directory / "v80_single_hub_u.mp4"), dpi=140):
        for field, time in zip(u, times):
            mesh.set_array(field.ravel())
            ax.set_title(f"Single V80 — open boundaries / GMG — t = {time:g} s")
            writer.grab_frame()
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
