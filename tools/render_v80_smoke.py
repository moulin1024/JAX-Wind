"""Render runtime V80 smoke frames as a two-plane near-wake MP4."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
from render_fv_wake_gif import _render_two_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    case = tomllib.loads((args.directory / "resolved_case.toml").read_text())
    turbine = case["physics"]["turbine"]
    x0, y0, z0 = turbine["x_m"], turbine["y_m"], turbine["hub_height_m"]
    with np.load(args.directory / "flow_frames.npz") as a:
        times = a["time_seconds"]
        assert len(times) == 100 and np.allclose(times, np.arange(1, 101) * 36., atol=.1)
        x, y, z = a["x_m"], a["y_m"], a["z_m"]
        def interval(values, lo, hi):
            indices = np.flatnonzero((values >= lo) & (values <= hi))
            return slice(indices[0], indices[-1] + 1)
        sx = interval(x, x0 - 400, x0 + 1600)
        sy = interval(y, y0 - 320, y0 + 320)
        sz = interval(z, 0, 200)
        horizontal = a["u_hub_yx"][:, sy, sx]
        vertical = a["u_center_zx"][:, sz, sx]
        assert np.isfinite(horizontal).all() and np.isfinite(vertical).all()
        grid = SimpleNamespace(
            x_faces=a["x_faces_m"][sx.start:sx.stop+1],
            y_faces=a["y_faces_m"][sy.start:sy.stop+1],
            z_faces=a["z_faces_m"][sz.start:sz.stop+1],
        )
    disk = SimpleNamespace(x=x0, y=y0, z=z0, tip_radius=40.)
    output = args.directory / "v80_smoke.mp4"
    _render_two_panel(horizontal, vertical, times, grid, disk, output,
                      args.fps, "V80 periodic smoke — near-wake view")
    print(output)


if __name__ == "__main__":
    main()
