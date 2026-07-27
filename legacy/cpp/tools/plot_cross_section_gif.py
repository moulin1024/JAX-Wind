#!/usr/bin/env python3
"""Render x-z cross-section frames and a GIF from WiRE-LES HDF5 field dumps."""

import argparse
import re
from pathlib import Path

try:
    import h5py
except ModuleNotFoundError as exc:
    raise SystemExit("HDF5 plotting requires h5py.") from exc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing fields_step_*.h5 files.")
    parser.add_argument("--pattern", default="fields_step_*.h5", help="Glob pattern relative to input dir.")
    parser.add_argument("--component", default="w", choices=("u", "v", "w", "p", "theta"), help="Field component to render.")
    parser.add_argument("--y-index", type=int, help="Y index for the x-z slice. Defaults to ny/2.")
    parser.add_argument("--frames-dir", type=Path, default=Path("outputs/cross_section_png_frames"), help="PNG frame directory.")
    parser.add_argument("--gif", type=Path, default=Path("outputs/cross_section.gif"), help="Output GIF path.")
    parser.add_argument("--fps", type=float, default=10.0, help="GIF playback frames per second.")
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap.")
    parser.add_argument("--symmetric", action="store_true", help="Use symmetric color limits around zero.")
    parser.add_argument("--vmin", type=float, help="Manual lower color limit.")
    parser.add_argument("--vmax", type=float, help="Manual upper color limit.")
    parser.add_argument("--max-frames", type=int, help="Uniformly subsample to at most this many frames.")
    parser.add_argument("--dpi", type=int, default=140, help="PNG resolution.")
    parser.add_argument("--no-gif", action="store_true", help="Only render PNG frames.")
    return parser.parse_args()


def step_from_path(path):
    match = re.search(r"step_(\d+)", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def selected_files(args):
    files = sorted(args.input_dir.glob(args.pattern), key=step_from_path)
    if not files:
        raise SystemExit("No HDF5 field files matched: {}".format(args.input_dir / args.pattern))
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise SystemExit("--max-frames must be positive.")
        if len(files) > args.max_frames:
            indices = np.linspace(0, len(files) - 1, args.max_frames, dtype=int)
            files = [files[int(i)] for i in indices]
    return files


def read_step_and_time(handle, path):
    step = int(handle.attrs.get("step", step_from_path(path)))
    time = float(handle.attrs.get("time", 0.0))
    return step, time


def read_coords(handle, field_shape):
    nx, _, nz = field_shape
    if "coords/x" in handle and handle["coords/x"].shape == (nx,):
        x = np.asarray(handle["coords/x"], dtype=np.float64)
    else:
        lx = float(handle.attrs.get("lx", nx))
        x = (np.arange(nx, dtype=np.float64) + 0.5) * (lx / float(nx))

    if "coords/z" in handle and handle["coords/z"].shape == (nz,):
        z = np.asarray(handle["coords/z"], dtype=np.float64)
    else:
        lz = float(handle.attrs.get("lz", nz))
        z = (np.arange(nz, dtype=np.float64) + 0.5) * (lz / float(nz))
    return x, z


def load_frame(path, component, requested_y_index):
    with h5py.File(path, "r") as handle:
        dataset_name = "fields/{}".format(component)
        if dataset_name not in handle:
            raise SystemExit("Missing dataset {} in {}".format(dataset_name, path))
        dataset = handle[dataset_name]
        if len(dataset.shape) != 3:
            raise SystemExit("Dataset {} must have shape (nx, ny, nz).".format(dataset_name))
        nx, ny, nz = dataset.shape
        y_index = ny // 2 if requested_y_index is None else requested_y_index
        if y_index < 0 or y_index >= ny:
            raise SystemExit("--y-index {} is outside [0, {}].".format(y_index, ny - 1))
        field = np.asarray(dataset[:, y_index, :], dtype=np.float64).T
        x, z = read_coords(handle, (nx, ny, nz))
        step, time = read_step_and_time(handle, path)
    return x, z, field, step, time, y_index


def color_limits(files, args):
    if args.vmin is not None and args.vmax is not None:
        return args.vmin, args.vmax

    global_min = np.inf
    global_max = -np.inf
    min_info = None
    max_info = None
    for path in files:
        _, _, field, _, _, _ = load_frame(path, args.component, args.y_index)
        local_min = float(np.nanmin(field))
        local_max = float(np.nanmax(field))
        if local_min < global_min:
            global_min = local_min
            min_info = (local_min, path)
        if local_max > global_max:
            global_max = local_max
            max_info = (local_max, path)

    if args.vmin is not None:
        global_min = args.vmin
    if args.vmax is not None:
        global_max = args.vmax
    if args.symmetric:
        scale = max(abs(global_min), abs(global_max))
        global_min, global_max = -scale, scale

    if min_info is not None:
        print("[range] global min={:.6g} file={}".format(min_info[0], min_info[1].name))
    if max_info is not None:
        print("[range] global max={:.6g} file={}".format(max_info[0], max_info[1].name))
    return global_min, global_max


def output_path(frames_dir, source):
    return frames_dir / source.with_suffix(".png").name


def render_frame(source, output, clim, args):
    x, z, field, step, time, y_index = load_frame(source, args.component, args.y_index)
    output.parent.mkdir(parents=True, exist_ok=True)
    xx, zz = np.meshgrid(x, z)

    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    image = ax.pcolormesh(
        xx,
        zz,
        field,
        shading="nearest",
        cmap=args.cmap,
        vmin=clim[0],
        vmax=clim[1],
    )
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(z.min()), float(z.max()))
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("x-z slice {}, y index {}, step {}, t={:.4g}".format(args.component, y_index, step, time))
    fig.colorbar(image, ax=ax)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)


def make_gif(frames, output, fps):
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / fps
    try:
        import imageio.v2 as imageio

        images = [imageio.imread(path) for path in frames]
        imageio.mimsave(output, images, duration=duration)
        return
    except ModuleNotFoundError:
        pass

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit("GIF output requires imageio or Pillow; rerun with --no-gif.") from exc

    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frames]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=int(duration * 1000.0),
        loop=0,
    )


def main():
    args = parse_args()
    files = selected_files(args)
    clim = color_limits(files, args)
    frames = []
    for source in files:
        target = output_path(args.frames_dir, source)
        render_frame(source, target, clim, args)
        frames.append(target)

    print("Rendered {} PNG frame(s) to: {}".format(len(frames), args.frames_dir))
    print("Color limits: vmin={:.6g}, vmax={:.6g}".format(clim[0], clim[1]))
    if not args.no_gif:
        make_gif(frames, args.gif, args.fps)
        print("Wrote GIF: {}".format(args.gif))


if __name__ == "__main__":
    main()
