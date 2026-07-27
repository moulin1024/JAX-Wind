#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
import tomllib
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render x-z cross-section frames and GIF from WiRE-LES JAX HDF5 dumps.")
    parser.add_argument("--input-dir", type=Path, default=Path("jax_fields"), help="Directory containing HDF5 field dumps.")
    parser.add_argument("--pattern", default="fields_step_*.h5", help="Glob pattern used inside --input-dir.")
    parser.add_argument("--file", action="append", type=Path, help="Explicit HDF5 file. Can be passed more than once.")
    parser.add_argument("--config", type=Path, help="Run TOML config used to fill missing domain metadata.")
    parser.add_argument(
        "--component",
        choices=(
            "u",
            "v",
            "w",
            "speed",
            "p",
            "theta",
            "qv",
            "theta_v",
            "u_prime",
            "v_prime",
            "w_prime",
            "p_prime",
            "theta_prime",
            "qv_prime",
            "theta_v_prime",
            "wtheta",
        ),
        default="u",
        help="Field to render. wtheta renders the local x-z slice of w' theta'.",
    )
    parser.add_argument("--y-index", type=int, help="Y index for the x-z slice. Defaults to ny//2.")
    parser.add_argument("--start-step", type=int, help="Only include files with step >= this value.")
    parser.add_argument("--end-step", type=int, help="Only include files with step <= this value.")
    parser.add_argument("--max-frames", type=int, help="Uniformly subsample to at most this many frames.")
    parser.add_argument("--frames-dir", type=Path, default=Path("cross_section_frames"), help="Directory for PNG frames.")
    parser.add_argument("--frame-prefix", default="xz", help="Prefix for PNG frame names.")
    parser.add_argument("--gif", type=Path, default=Path("cross_section_xz.gif"), help="Output GIF path.")
    parser.add_argument("--fps", type=float, default=12.0, help="GIF playback frames per second.")
    parser.add_argument("--dpi", type=int, default=150, help="PNG frame DPI.")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap.")
    parser.add_argument("--vmin", type=float, help="Fixed colorbar minimum.")
    parser.add_argument("--vmax", type=float, help="Fixed colorbar maximum.")
    parser.add_argument("--symmetric", action="store_true", help="Use a symmetric colorbar around zero.")
    parser.add_argument("--report-ranges", action="store_true", help="Print per-frame min/max values before rendering.")
    parser.add_argument("--no-gif", action="store_true", help="Write PNG frames only.")
    return parser.parse_args()


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def config_value(config: dict, section: str, key: str) -> float | None:
    value = config.get(section, {}).get(key)
    return None if value is None else float(value)


def read_step(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        return int(handle.attrs.get("step", -1))


def selected_files(args: argparse.Namespace) -> list[Path]:
    files = [Path(path) for path in args.file] if args.file else sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No HDF5 files matched input: {args.input_dir / args.pattern}")

    stepped = [(read_step(path), path) for path in files]
    if args.start_step is not None:
        stepped = [(step, path) for step, path in stepped if step >= args.start_step]
    if args.end_step is not None:
        stepped = [(step, path) for step, path in stepped if step <= args.end_step]
    if not stepped:
        raise SystemExit("No HDF5 files remain after step filtering.")

    stepped.sort(key=lambda item: item[0])
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise SystemExit("--max-frames must be positive.")
        if len(stepped) > args.max_frames:
            indices = np.linspace(0, len(stepped) - 1, args.max_frames, dtype=int)
            stepped = [stepped[int(index)] for index in indices]
    return [path for _, path in stepped]


def first_available(*values: float | None, name: str) -> float:
    for value in values:
        if value is not None:
            return float(value)
    raise SystemExit(f"Missing {name}; provide it in HDF5 attrs or --config.")


def read_attrs(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        return dict(handle.attrs)


def config_string(config: dict, section: str, key: str) -> str | None:
    value = config.get(section, {}).get(key)
    return None if value is None else str(value)


def warn_if_metadata_mismatch(files: list[Path], config: dict) -> None:
    expected_wall_treatment = config_string(config, "wall", "stress_treatment")
    expected_sgs_model = config_string(config, "sgs", "model")
    missing_metadata: list[str] = []
    mismatched_wall_treatment: list[str] = []
    mismatched_sgs_model: list[str] = []
    for path in files:
        attrs = read_attrs(path)
        if expected_wall_treatment is not None and "wall_stress_treatment" not in attrs:
            missing_metadata.append(path.name)
            continue
        if expected_wall_treatment is not None:
            actual = str(attrs["wall_stress_treatment"])
            if actual != expected_wall_treatment:
                mismatched_wall_treatment.append(f"{path.name}: {actual}")
        if expected_sgs_model is not None and "sgs_model" in attrs:
            actual_sgs = str(attrs["sgs_model"])
            if actual_sgs != expected_sgs_model:
                mismatched_sgs_model.append(f"{path.name}: {actual_sgs}")

    if missing_metadata:
        sample = ", ".join(missing_metadata[:5])
        suffix = "" if len(missing_metadata) <= 5 else f", ... ({len(missing_metadata)} files)"
        print(
            "[warning] selected HDF5 files are missing wall_stress_treatment metadata "
            f"needed to compare against the config: {sample}{suffix}.",
            flush=True,
        )
    if mismatched_wall_treatment:
        sample = ", ".join(mismatched_wall_treatment[:5])
        suffix = "" if len(mismatched_wall_treatment) <= 5 else f", ... ({len(mismatched_wall_treatment)} files)"
        print(
            "[warning] selected HDF5 wall_stress_treatment differs from the config "
            f"({expected_wall_treatment}): {sample}{suffix}.",
            flush=True,
        )
    if mismatched_sgs_model:
        sample = ", ".join(mismatched_sgs_model[:5])
        suffix = "" if len(mismatched_sgs_model) <= 5 else f", ... ({len(mismatched_sgs_model)} files)"
        print(
            "[warning] selected HDF5 sgs_model differs from the config "
            f"({expected_sgs_model}): {sample}{suffix}.",
            flush=True,
        )


def domain_extents(path: Path, config: dict) -> tuple[float, float]:
    attrs = read_attrs(path)
    lx = first_available(attrs.get("lx"), config_value(config, "grid", "lx"), name="lx")
    lz = first_available(attrs.get("lz"), config_value(config, "grid", "lz"), name="lz")
    return lx, lz


def plot_coordinates(path: Path, config: dict, section_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    nx, nz = section_shape
    lx, lz = domain_extents(path, config)
    x = np.linspace(0.0, lx, nx, endpoint=False, dtype=np.float64)
    z = (np.arange(nz, dtype=np.float64) + 0.5) * (lz / float(nz))
    with h5py.File(path, "r") as handle:
        if "coords/x" in handle:
            x_data = np.asarray(handle["coords/x"], dtype=np.float64)
            if x_data.shape == (nx,):
                x = x_data
        if "coords/z_center" in handle:
            z_data = np.asarray(handle["coords/z_center"], dtype=np.float64)
            if z_data.shape == (nz,):
                z = z_data
        elif "coords/z" in handle:
            z_data = np.asarray(handle["coords/z"], dtype=np.float64)
            if z_data.shape == (nz,):
                z = z_data
    return x, z


def component_slice(handle: h5py.File, component: str, y_index: int) -> np.ndarray:
    fields = handle["fields"]
    if component == "speed":
        u = np.asarray(fields["u"][:, y_index, :])
        v = np.asarray(fields["v"][:, y_index, :])
        w = np.asarray(fields["w"][:, y_index, :])
        return np.sqrt(u * u + v * v + w * w)
    if component == "wtheta":
        w = np.asarray(fields["w"][:])
        theta = np.asarray(fields["theta"][:])
        w_mean_z = np.mean(w, axis=(0, 1), keepdims=True)
        theta_mean_z = np.mean(theta, axis=(0, 1), keepdims=True)
        return ((w - w_mean_z) * (theta - theta_mean_z))[:, y_index, :]
    if component.endswith("_prime"):
        base_component = component.removesuffix("_prime")
        field = np.asarray(fields[base_component][:])
        mean_z = np.mean(field, axis=(0, 1), keepdims=True)
        return (field - mean_z)[:, y_index, :]
    return np.asarray(fields[component][:, y_index, :])


def validate_y_index(path: Path, y_index: int | None) -> int:
    with h5py.File(path, "r") as handle:
        ny = int(handle["fields/u"].shape[1])
    if y_index is None:
        return ny // 2
    if y_index < 0 or y_index >= ny:
        raise SystemExit(f"--y-index {y_index} is outside stored y range [0, {ny - 1}].")
    return y_index


def color_limits(files: list[Path], component: str, y_index: int, args: argparse.Namespace) -> tuple[float, float]:
    if args.vmin is not None and args.vmax is not None:
        return args.vmin, args.vmax

    mins: list[float] = []
    maxs: list[float] = []
    min_info: tuple[float, int, Path] | None = None
    max_info: tuple[float, int, Path] | None = None
    for path in files:
        with h5py.File(path, "r") as handle:
            section = component_slice(handle, component, y_index)
            step = int(handle.attrs.get("step", read_step(path)))
            local_min = float(np.nanmin(section))
            local_max = float(np.nanmax(section))
            if args.report_ranges:
                print(
                    f"[range] step={step:06d} min={local_min:.6g} max={local_max:.6g}",
                    flush=True,
                )
        mins.append(local_min)
        maxs.append(local_max)
        if min_info is None or local_min < min_info[0]:
            min_info = (local_min, step, path)
        if max_info is None or local_max > max_info[0]:
            max_info = (local_max, step, path)

    vmin = min(mins) if args.vmin is None else args.vmin
    vmax = max(maxs) if args.vmax is None else args.vmax
    if args.symmetric:
        bound = max(abs(vmin), abs(vmax))
        vmin, vmax = -bound, bound
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        raise SystemExit(f"Invalid color limits: vmin={vmin}, vmax={vmax}")
    if min_info is not None and max_info is not None:
        print(
            f"[range] global min={min_info[0]:.6g} at step={min_info[1]:06d} file={min_info[2].name}",
            flush=True,
        )
        print(
            f"[range] global max={max_info[0]:.6g} at step={max_info[1]:06d} file={max_info[2].name}",
            flush=True,
        )
    return vmin, vmax


def frame_path(args: argparse.Namespace, path: Path) -> Path:
    step = read_step(path)
    return args.frames_dir / f"{args.frame_prefix}_{args.component}_step_{step:06d}.png"


def render_frame(
    source: Path,
    output: Path,
    component: str,
    y_index: int,
    config: dict,
    clim: tuple[float, float],
    args: argparse.Namespace,
) -> None:
    with h5py.File(source, "r") as handle:
        section = component_slice(handle, component, y_index)
        step = int(handle.attrs.get("step", -1))
        time_value = float(handle.attrs.get("time", np.nan))

    output.parent.mkdir(parents=True, exist_ok=True)
    x, z = plot_coordinates(source, config, section.shape)
    vmin, vmax = clim
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    image = ax.pcolormesh(
        x,
        z,
        section.T,
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
        shading="nearest",
    )
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(z)), float(np.max(z)))
    title = f"{component} x-z slice, y index {y_index}, step {step}"
    if np.isfinite(time_value):
        title += f", t={time_value:.4g}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(component)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)


def make_gif(frames: list[Path], output: Path, fps: float) -> None:
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
        raise SystemExit("GIF output requires imageio or Pillow; rerun with --no-gif to write PNG frames only.") from exc

    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frames]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=int(duration * 1000.0),
        loop=0,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    files = selected_files(args)
    warn_if_metadata_mismatch(files, config)
    y_index = validate_y_index(files[0], args.y_index)
    clim = color_limits(files, args.component, y_index, args)

    frames: list[Path] = []
    for source in files:
        output = frame_path(args, source)
        render_frame(source, output, args.component, y_index, config, clim, args)
        frames.append(output)

    print(f"Rendered {len(frames)} PNG frame(s) to: {args.frames_dir}")
    print(f"Y index: {y_index}")
    print(f"Color limits: vmin={clim[0]:.6g}, vmax={clim[1]:.6g}")
    if not args.no_gif:
        make_gif(frames, args.gif, args.fps)
        print(f"Wrote GIF: {args.gif}")


if __name__ == "__main__":
    main()
