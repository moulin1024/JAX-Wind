"""Render the configured hub-height u frames of a Horns Rev main run."""
import argparse
from pathlib import Path
import tomllib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from render_fv_wake_gif import _ffmpeg_executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    doc = tomllib.loads((args.directory / "resolved_case.toml").read_text())
    with np.load(args.directory / "flow_frames.npz") as archive:
        u = archive["u_hub_yx"]
        times = archive["time_seconds"]
        x, y = archive["x_faces_m"] / 1000., archive["y_faces_m"] / 1000.
    count = doc["time"]["frame_count"]
    turbines = len(doc["physics"]["wind_farm"]["layout"])
    hub = doc["physics"]["turbine"]["hub_height_m"]
    inflow = doc["physics"].get("inflow", {})
    target = doc.get("initial_conditions", {}).get("stage_options", {}).get("target_hub_wind_speed_m_s")
    inflow_label = (f"{inflow['speed_m_s']:g} m/s uniform inlet" if inflow.get("model") == "uniform"
                    else f"Scaled precursor; target Uhub={target:.3f} m/s" if target is not None
                    else "Recorded precursor")
    if count < 1 or len(times) != count or len(u) != count or not np.isfinite(u).all():
        raise ValueError(f"expected {count} finite hub-height frames")
    expected_steps = np.arange(1, count + 1) * doc["time"]["steps"] // count
    expected = expected_steps * doc["time"]["dt_seconds"]
    np.testing.assert_allclose(times, expected, atol=.01)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    mesh = ax.pcolormesh(x, y, u[0], shading="flat", cmap="viridis", vmin=0., vmax=12.)
    segments = [[(row["x_m"]/1000., (row["y_m"]-40.)/1000.),
                 (row["x_m"]/1000., (row["y_m"]+40.)/1000.)]
                for row in doc["physics"]["wind_farm"]["layout"]]
    ax.add_collection(LineCollection(segments, colors="black", linewidths=.8))
    ax.set(xlabel="Downstream x [km]", ylabel="Cross-stream y [km]", aspect="equal")
    fig.colorbar(mesh, ax=ax, label="Streamwise velocity u [m/s]")
    title = ax.set_title("")
    matplotlib.rcParams["animation.ffmpeg_path"] = _ffmpeg_executable()
    output = args.directory / "hornsrev1_hub_u.mp4"
    writer = FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    with writer.saving(fig, str(output), dpi=140):
        for field, time in zip(u, times):
            mesh.set_array(field.ravel())
            title.set_text(f"Horns Rev 1 — {turbines} V80 turbines — u at z = {hub:g} m\n{inflow_label}; t = {time/60:.1f} min")
            writer.grab_frame()
    fig.savefig(args.directory / "hornsrev1_hub_u_final.png", dpi=140)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
