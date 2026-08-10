#!/usr/bin/env python3
"""Plot final precursor/main/difference velocity slices from local adjoint shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "jax"))

from run_single import RUN_DEFAULTS, load_config_file, params_from_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final three-plane velocity slices from a saved adjoint run."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hub-height", type=float, default=1.5)
    parser.add_argument("--centre-y", type=float)
    parser.add_argument("--wake-x", type=float)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _slice_bounds(value: list[int | None], extent: int) -> tuple[int, int]:
    start = 0 if value[0] is None else int(value[0])
    stop = extent if value[1] is None else int(value[1])
    return start, stop


def load_velocity_roles(directory: Path) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    slabs: dict[int, list[tuple[int, np.ndarray]]] = {0: [], 1: []}
    steps: dict[int, int] = {}
    metadata_paths = sorted(directory.glob("rank_*.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No rank_*.json metadata found in {directory}")
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        global_shape = metadata["global_field_shape"]
        role_start, role_stop = _slice_bounds(
            metadata["indices"]["u"][0], global_shape[0]
        )
        z_start, _ = _slice_bounds(metadata["indices"]["u"][3], global_shape[3])
        if role_stop - role_start != 1:
            raise ValueError(f"Expected one adjoint role in {metadata_path}")
        archive_path = metadata_path.with_suffix(".npz")
        with np.load(archive_path) as archive:
            slabs[role_start].append((z_start, np.asarray(archive["u"])[0]))
            step = int(np.asarray(archive["step"])[0])
        if role_start in steps and steps[role_start] != step:
            raise ValueError(f"Inconsistent step across role {role_start} shards")
        steps[role_start] = step
    fields = {
        role: np.concatenate(
            [slab for _, slab in sorted(role_slabs)], axis=2
        )
        for role, role_slabs in slabs.items()
    }
    if any(field.size == 0 for field in fields.values()):
        raise ValueError("Both precursor and turbine adjoint roles are required")
    return fields, steps


def interpolate_plane(
    field: np.ndarray,
    coordinates: np.ndarray,
    value: float,
    axis: int,
) -> np.ndarray:
    if not coordinates[0] <= value <= coordinates[-1]:
        raise ValueError(
            f"Slice coordinate {value:g} lies outside "
            f"[{coordinates[0]:g}, {coordinates[-1]:g}]"
        )
    upper = int(np.searchsorted(coordinates, value))
    if upper == 0:
        return np.take(field, 0, axis=axis)
    if upper == len(coordinates):
        return np.take(field, -1, axis=axis)
    lower = upper - 1
    alpha = (value - coordinates[lower]) / (
        coordinates[upper] - coordinates[lower]
    )
    return (1.0 - alpha) * np.take(field, lower, axis=axis) + alpha * np.take(
        field, upper, axis=axis
    )


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")

    import jax.numpy as jnp

    settings = dict(RUN_DEFAULTS)
    settings.update(load_config_file(args.config))
    params = params_from_settings(settings, jnp)
    lx = params.lx * params.z_i
    ly = params.ly * params.z_i
    lz = params.lz * params.z_i
    centre_y = 0.5 * ly if args.centre_y is None else args.centre_y
    wake_x = (
        params.actuator_disk_x + 3.0 * params.actuator_disk_diameter
        if args.wake_x is None
        else args.wake_x
    ) % lx

    fields, steps = load_velocity_roles(args.input_dir)
    x = (np.arange(params.nx) + 0.5) * params.dx * params.z_i
    y = (np.arange(params.ny) + 0.5) * params.dy * params.z_i
    z = (np.arange(params.nz) + 0.5) * params.dz * params.z_i

    planes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for role, field in fields.items():
        planes[role] = (
            interpolate_plane(field, z, args.hub_height, axis=2),
            interpolate_plane(field, y, centre_y, axis=1),
            interpolate_plane(field, x, wake_x, axis=0),
        )
    difference = tuple(main - precursor for main, precursor in zip(
        planes[1], planes[0], strict=True
    ))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    velocity_values = np.concatenate(
        [plane.ravel() for role in (0, 1) for plane in planes[role]]
    )
    velocity_min, velocity_max = np.percentile(velocity_values, (0.5, 99.5))
    difference_limit = np.percentile(
        np.abs(np.concatenate([plane.ravel() for plane in difference])), 99.0
    )
    difference_limit = max(float(difference_limit), 1.0e-6)

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15.2, 10.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (2.0, 2.0, 1.0)},
    )
    extents = ((0.0, lx, 0.0, ly), (0.0, lx, 0.0, lz), (0.0, ly, 0.0, lz))
    column_titles = (
        f"Horizontal: z = {args.hub_height:g} m",
        f"Streamwise: y = {centre_y:g} m",
        f"Cross-wake: x = {wake_x:g} m",
    )
    row_titles = (
        f"Precursor (step {steps[0]})",
        f"Main + actuator (step {steps[1]})",
        "Main − precursor",
    )
    image_rows = (planes[0], planes[1], difference)
    images = []
    for row, row_planes in enumerate(image_rows):
        for column, (axis, plane, extent) in enumerate(
            zip(axes[row], row_planes, extents, strict=True)
        ):
            image = axis.imshow(
                plane.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                interpolation="bilinear",
                cmap="viridis" if row < 2 else "RdBu_r",
                vmin=velocity_min if row < 2 else -difference_limit,
                vmax=velocity_max if row < 2 else difference_limit,
                rasterized=True,
            )
            images.append(image)
            if row == 0:
                axis.set_title(column_titles[column])
            if column == 0:
                axis.set_ylabel(f"{row_titles[row]}\n\ny [m]")
            elif column == 1:
                axis.set_ylabel("z [m]")
            else:
                axis.set_ylabel("z [m]")
            axis.set_xlabel("x [m]" if column < 2 else "y [m]")

            if column < 2:
                axis.axvspan(
                    params.fringe_start_x,
                    lx,
                    facecolor="white",
                    edgecolor="white",
                    hatch="///",
                    linewidth=0.8,
                    alpha=0.13,
                )
            if column in (1, 2) and params.sponge_enabled:
                axis.axhspan(
                    params.sponge_start_height,
                    lz,
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.10,
                )
            if row in (1, 2) and params.actuator_disk_enabled:
                if column == 0:
                    axis.plot(
                        [params.actuator_disk_x, params.actuator_disk_x],
                        [
                            params.actuator_disk_y
                            - 0.5 * params.actuator_disk_diameter,
                            params.actuator_disk_y
                            + 0.5 * params.actuator_disk_diameter,
                        ],
                        color="white",
                        linewidth=3.0,
                        solid_capstyle="round",
                    )
                elif column == 1:
                    axis.add_patch(
                        Rectangle(
                            (
                                params.actuator_disk_x
                                - 0.5 * params.actuator_disk_thickness,
                                params.actuator_disk_z
                                - 0.5 * params.actuator_disk_diameter,
                            ),
                            params.actuator_disk_thickness,
                            params.actuator_disk_diameter,
                            facecolor="none",
                            edgecolor="white",
                            linewidth=1.8,
                        )
                    )

    velocity_colorbar = fig.colorbar(
        images[0], ax=axes[:2, :], location="right", shrink=0.86, pad=0.018
    )
    velocity_colorbar.set_label(r"$u$ [m s$^{-1}$]")
    difference_colorbar = fig.colorbar(
        images[-1], ax=axes[2, :], location="right", shrink=0.86, pad=0.018
    )
    difference_colorbar.set_label(r"$\Delta u$ [m s$^{-1}$]")
    lead_seconds = (steps[0] - steps[1]) * params.dt_physical
    fig.suptitle(
        "Final concurrent-adjoint streamwise velocity slices\n"
        f"precursor pipeline lead = {lead_seconds:.3f} s; "
        "hatched x-region = fringe; pale top region = Rayleigh layer",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
