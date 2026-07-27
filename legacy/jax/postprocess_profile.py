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
    parser = argparse.ArgumentParser(description="Postprocess WiRE-LES JAX HDF5 velocity fields.")
    parser.add_argument("--input-dir", type=Path, default=Path("jax_fields"), help="Directory containing HDF5 field dumps.")
    parser.add_argument("--pattern", default="fields_step_*.h5", help="Glob pattern used inside --input-dir.")
    parser.add_argument("--file", action="append", type=Path, help="Explicit HDF5 file. Can be passed more than once.")
    parser.add_argument("--config", type=Path, help="Run TOML config used to fill missing physics metadata.")
    parser.add_argument("--average", choices=("last", "all"), default="last", help="Use the latest file or average all selected files.")
    parser.add_argument("--start-step", type=int, help="Only include files with step >= this value.")
    parser.add_argument("--end-step", type=int, help="Only include files with step <= this value.")
    parser.add_argument("--u-fric", type=float, help="Friction velocity for the log-law reference.")
    parser.add_argument("--zo", type=float, help="Roughness length for the log-law reference.")
    parser.add_argument("--vonk", type=float, help="von Karman constant for the log-law reference.")
    parser.add_argument("--output", type=Path, default=Path("u_profile.png"), help="Output plot path.")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path for z, mean u, and log-law u.")
    parser.add_argument("--linear-z", action="store_true", help="Use a linear z axis instead of log scale.")
    parser.add_argument(
        "--include-top-boundary",
        action="store_true",
        help="Include the top physical boundary plane. By default it is dropped to match Fortran log/profile statistics.",
    )
    parser.add_argument(
        "--include-bottom-boundary",
        action="store_true",
        help="Include the bottom boundary plane if it exists. By default z <= 0 is dropped for log-axis plots.",
    )
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
    if args.average == "last":
        return [stepped[-1][1]]
    return [path for _, path in stepped]


def attr_float(attrs: h5py.AttributeManager, key: str) -> float | None:
    value = attrs.get(key)
    return None if value is None else float(value)


def first_available(*values: float | None, name: str) -> float:
    for value in values:
        if value is not None:
            return float(value)
    raise SystemExit(f"Missing {name}; provide it in HDF5 attrs, --config, or command-line arguments.")


def load_profiles(files: list[Path]) -> tuple[np.ndarray, dict]:
    profiles = []
    first_attrs: dict = {}
    for path in files:
        with h5py.File(path, "r") as handle:
            u = np.asarray(handle["fields/u"])
            profiles.append(u.mean(axis=(0, 1)))
            if not first_attrs:
                first_attrs = dict(handle.attrs)
    return np.mean(np.stack(profiles, axis=0), axis=0), first_attrs


def vertical_coordinates(path: Path, attrs: dict, nz: int, config: dict) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "coords/z_center" in handle:
            z = np.asarray(handle["coords/z_center"], dtype=np.float64)
            if z.shape == (nz,):
                return z
        if "coords/z" in handle:
            z = np.asarray(handle["coords/z"], dtype=np.float64)
            if z.shape == (nz,):
                return z

    lz = attrs.get("lz")
    if lz is None:
        lz = config_value(config, "grid", "lz")
    if lz is None:
        raise SystemExit("Missing lz; provide a newer HDF5 dump or --config.")
    dz = float(lz) / float(nz)
    return (np.arange(nz, dtype=np.float64) + 0.5) * dz


def log_law_profile(z: np.ndarray, u_fric: float, zo: float, vonk: float) -> np.ndarray:
    profile = np.full_like(z, np.nan, dtype=np.float64)
    valid = z > zo
    profile[valid] = (u_fric / vonk) * np.log(z[valid] / zo)
    return profile


def save_csv(path: Path, z: np.ndarray, u_mean: np.ndarray, u_log: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((z, u_mean, u_log))
    np.savetxt(path, data, delimiter=",", header="z,u_mean,u_loglaw", comments="")


def plot_profile(path: Path, z: np.ndarray, u_mean: np.ndarray, u_log: np.ndarray, files: list[Path], linear_z: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 6.0), constrained_layout=True)
    ax.plot(u_mean, z, marker="o", markersize=3.0, linewidth=1.6, label="mean u")
    ax.plot(u_log, z, linestyle="--", linewidth=1.6, label="log law")
    if not linear_z:
        ax.set_yscale("log")
    ax.set_xlabel("u")
    ax.set_ylabel("z")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax.legend()
    ax.set_title(f"u profile ({len(files)} snapshot{'s' if len(files) != 1 else ''})")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    files = selected_files(args)
    u_mean, attrs = load_profiles(files)
    stored_nz = u_mean.shape[0]
    z = vertical_coordinates(files[0], attrs, stored_nz, config)
    if not args.include_bottom_boundary:
        keep = z > 0.0
        z = z[keep]
        u_mean = u_mean[keep]
    if not args.include_top_boundary:
        z = z[:-1]
        u_mean = u_mean[:-1]

    u_fric = first_available(args.u_fric, attr_float(attrs, "u_fric"), config_value(config, "physics", "u_fric"), name="u_fric")
    zo = first_available(args.zo, attr_float(attrs, "zo"), config_value(config, "physics", "zo"), name="zo")
    vonk = first_available(args.vonk, attr_float(attrs, "vonk"), config_value(config, "physics", "vonk"), 0.4, name="vonk")
    u_log = log_law_profile(z, u_fric, zo, vonk)

    plot_profile(args.output, z, u_mean, u_log, files, args.linear_z)
    if args.csv is not None:
        save_csv(args.csv, z, u_mean, u_log)

    print(f"Processed {len(files)} file(s).")
    print(f"Wrote plot: {args.output}")
    if args.csv is not None:
        print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
