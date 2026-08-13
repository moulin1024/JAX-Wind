#!/usr/bin/env python3
"""Overlay uniform ABL diagnostics on selected Nieuwstadt et al. figures."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "outputs" / "nieuwstadt1993_lasd_40x40x48"
DEFAULT_REFERENCE = (
    ROOT / "cases" / "Nieuwstadt1993" / "reference" / "figures"
)
FIGURES = (2, 3, 6, 8, 11, 14, 15)
SCALE = 4
RED = (205, 35, 45, 255)
BLUE = (0, 114, 178, 255)
GREEN = (0, 135, 95, 255)


@dataclass(frozen=True, slots=True)
class Axes:
    left: float
    right: float
    top: float
    bottom: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    xlog: bool = False
    ylog: bool = False


@dataclass(frozen=True, slots=True)
class Series:
    label: str
    x: np.ndarray
    y: np.ndarray
    color: tuple[int, int, int, int] = RED
    dashed: bool = False


AXES = {
    2: Axes(159, 781, 43, 660, -0.4, 1.1, 0.0, 1.5),
    3: Axes(212, 921, 65, 756, 0.0, 0.8, 0.0, 1.5),
    6: Axes(188, 806, 31, 639, 0.0, 0.5, 0.0, 1.5),
    8: Axes(177, 796, 40, 642, -4.0, 4.0, 0.0, 1.5),
    11: Axes(169, 786, 31, 635, -0.6, 0.4, 0.0, 1.5),
    14: Axes(255, 872, 39, 642, -1.0, 2.0, 0.0, 1.5),
    15: Axes(193, 810, 24, 627, 1.0, 100.0, 1.0e-4, 1.0e1, True, True),
}


def _csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"diagnostic CSV is empty: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in rows[0]
    }


def _normalization(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    physics = summary["physics"]
    metrics = summary["diagnostic_metrics"]
    reference_length = float(summary["diagnostic_reference"]["length_m"])
    boundary_height = float(metrics["boundary_layer_height_m"])
    buoyancy_velocity = (
        float(physics["buoyancy_acceleration_per_scalar"])
        * float(physics["scalar_surface_flux"])
        * boundary_height
    ) ** (1.0 / 3.0)
    scalar_scale = float(physics["scalar_surface_flux"]) / buoyancy_velocity
    return reference_length, boundary_height, buoyancy_velocity, scalar_scale


def _series(
    profiles: dict[str, np.ndarray],
    radial: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> dict[int, list[Series]]:
    reference_length, boundary_height, velocity_scale, scalar_scale = (
        _normalization(summary)
    )
    z = profiles["z_m"]
    z_over_reference = z / reference_length
    z_over_boundary = z / boundary_height
    sgs_component_variance = (2.0 / 3.0) * profiles["sgs_tke_m2_s2"]
    resolved_w_variance = profiles["resolved_w_variance_m2_s2"]
    pressure_rms = np.sqrt(
        np.maximum(profiles["pressure_variance_m4_s4"], 0.0)
    ) / velocity_scale**2
    skewness = np.divide(
        profiles["w_third_moment_m3_s3"],
        np.maximum(resolved_w_variance, 0.0) ** 1.5,
        out=np.zeros_like(resolved_w_variance),
        where=resolved_w_variance > 0.0,
    )
    pressure_transport_gradient = np.gradient(
        profiles["pressure_vertical_transport_m3_s3"], z
    ) / (velocity_scale**3 / boundary_height)
    result = {
        2: [
            Series(
                "total scalar flux",
                profiles["total_scalar_flux"]
                / float(summary["physics"]["scalar_surface_flux"]),
                z_over_reference,
            ),
            Series(
                "SGS contribution",
                profiles["sgs_scalar_flux"]
                / float(summary["physics"]["scalar_surface_flux"]),
                z_over_reference,
                BLUE,
                True,
            ),
        ],
        3: [
            Series(
                "total vertical variance",
                (resolved_w_variance + sgs_component_variance)
                / velocity_scale**2,
                z_over_boundary,
            ),
            Series(
                "SGS contribution",
                sgs_component_variance / velocity_scale**2,
                z_over_boundary,
                BLUE,
                True,
            ),
        ],
        6: [Series("pressure RMS", pressure_rms, z_over_boundary)],
        8: [Series("vertical-velocity skewness", skewness, z_over_boundary)],
        11: [
            Series(
                "pressure transport",
                pressure_transport_gradient,
                z_over_boundary,
            )
        ],
        14: [
            Series(
                "updraft scalar excess",
                profiles["updraft_scalar_excess"] / scalar_scale,
                z_over_boundary,
            )
        ],
    }

    heights = np.unique(radial["sample_height_m"])
    colors = (RED, BLUE, GREEN)
    offsets = (1.0, 10.0, 100.0)
    spectra: list[Series] = []
    for index, height in enumerate(heights[:3]):
        selected = np.isclose(radial["sample_height_m"], height)
        wavenumber = radial["wavenumber_reference_length"][selected]
        energy = radial["horizontal_energy"][selected]
        order = np.argsort(wavenumber)
        offset = offsets[index]
        spectra.append(
            Series(
                f"z/zi={height / boundary_height:.2f} ×{offset:g}",
                wavenumber[order],
                wavenumber[order]
                * energy[order]
                / velocity_scale**2
                * offset,
                colors[index],
                dashed=index > 0,
            )
        )
    result[15] = spectra
    return result


def _coordinate(value: float, lower: float, upper: float, log: bool) -> float:
    if log:
        return (math.log10(value) - math.log10(lower)) / (
            math.log10(upper) - math.log10(lower)
        )
    return (value - lower) / (upper - lower)


def _point(x: float, y: float, axes: Axes) -> tuple[float, float]:
    x_fraction = _coordinate(x, axes.xmin, axes.xmax, axes.xlog)
    y_fraction = _coordinate(y, axes.ymin, axes.ymax, axes.ylog)
    return (
        SCALE * (axes.left + x_fraction * (axes.right - axes.left)),
        SCALE * (axes.bottom - y_fraction * (axes.bottom - axes.top)),
    )


def _segments(series: Series, axes: Axes) -> list[list[tuple[float, float]]]:
    valid = np.isfinite(series.x) & np.isfinite(series.y)
    if axes.xlog:
        valid &= series.x > 0.0
    if axes.ylog:
        valid &= series.y > 0.0
    valid &= (series.x >= axes.xmin) & (series.x <= axes.xmax)
    valid &= (series.y >= axes.ymin) & (series.y <= axes.ymax)
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for x, y, keep in zip(series.x, series.y, valid):
        if keep:
            current.append(_point(float(x), float(y), axes))
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def _dashed(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    for start, end in zip(points[:-1], points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        distance = 0.0
        while distance < length:
            dash_end = min(distance + 12 * SCALE, length)
            draw.line(
                (
                    (start[0] + dx * distance / length, start[1] + dy * distance / length),
                    (start[0] + dx * dash_end / length, start[1] + dy * dash_end / length),
                ),
                fill=fill,
                width=width,
            )
            distance += 20 * SCALE


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _overlay(
    source: Path,
    axes: Axes,
    series: list[Series],
    label: str,
) -> Image.Image:
    original = Image.open(source).convert("RGBA")
    size = (original.width * SCALE, original.height * SCALE)
    curves = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(curves)
    for item in series:
        for points in _segments(item, axes):
            if len(points) < 2:
                continue
            if item.dashed:
                _dashed(draw, points, fill=item.color, width=4 * SCALE)
            else:
                draw.line(points, fill=item.color, width=5 * SCALE, joint="curve")
    clip = Image.new("L", size, 0)
    ImageDraw.Draw(clip).rectangle(
        tuple(int(value * SCALE) for value in (
            axes.left,
            axes.top,
            axes.right,
            axes.bottom,
        )),
        fill=255,
    )
    curves.putalpha(ImageChops.multiply(curves.getchannel("A"), clip))
    legend = Image.new("RGBA", size, (0, 0, 0, 0))
    legend_draw = ImageDraw.Draw(legend)
    x = min((axes.right + 45) * SCALE, (original.width - 330) * SCALE)
    y = (axes.top + 45) * SCALE
    legend_draw.text((x, y), label, font=_font(20 * SCALE), fill=(0, 0, 0, 255))
    for index, item in enumerate(series):
        line_y = y + (45 + 36 * index) * SCALE
        points = [(x, line_y), (x + 64 * SCALE, line_y)]
        if item.dashed:
            _dashed(legend_draw, points, fill=item.color, width=4 * SCALE)
        else:
            legend_draw.line(points, fill=item.color, width=5 * SCALE)
        legend_draw.text(
            (x + 76 * SCALE, line_y - 12 * SCALE),
            item.label,
            font=_font(17 * SCALE),
            fill=(0, 0, 0, 255),
        )
    curves = curves.resize(original.size, Image.Resampling.LANCZOS)
    legend = legend.resize(original.size, Image.Resampling.LANCZOS)
    return Image.alpha_composite(
        Image.alpha_composite(original, curves), legend
    ).convert("RGB")


def _montage(images: list[tuple[int, Image.Image]]) -> Image.Image:
    tile_width, tile_height = 660, 470
    columns = 3
    rows = (len(images) + columns - 1) // columns
    result = Image.new(
        "RGB",
        (columns * tile_width + 32, rows * tile_height + 100),
        (242, 244, 247),
    )
    draw = ImageDraw.Draw(result)
    draw.text(
        (16, 14),
        "Nieuwstadt et al. (1993): uniform JAX-Wind diagnostic checkout",
        font=_font(28),
        fill=(20, 25, 32),
    )
    draw.text(
        (16, 55),
        "red = total/resolved; blue dashed = SGS; spectra colors = sampling heights",
        font=_font(17),
        fill=(70, 76, 84),
    )
    for index, (number, image) in enumerate(images):
        column, row = index % columns, index // columns
        x0 = 8 + column * tile_width
        y0 = 92 + row * tile_height
        draw.text((x0 + 8, y0), f"Figure {number}", font=_font(18), fill=(20, 25, 32))
        fitted = ImageOps.contain(
            image,
            (tile_width - 16, tile_height - 34),
            Image.Resampling.LANCZOS,
        )
        result.paste(fitted, (x0 + (tile_width - fitted.width) // 2, y0 + 28))
    return result


def overlay_results(
    results: Path,
    reference: Path,
    output: Path,
    *,
    legend_label: str = "JAX-Wind uniform LASD 40×40×48 GPU",
) -> list[Path]:
    profiles = _csv(results / "profiles.csv")
    radial = _csv(results / "radial_spectra.csv")
    summary = json.loads((results / "summary.json").read_text())
    figure_series = _series(profiles, radial, summary)
    output.mkdir(parents=True, exist_ok=True)
    images: list[tuple[int, Image.Image]] = []
    written: list[Path] = []
    for number in FIGURES:
        image = _overlay(
            reference / f"fig{number}.png",
            AXES[number],
            figure_series[number],
            legend_label,
        )
        path = output / f"figure_{number:02d}_overlay.png"
        image.save(path)
        images.append((number, image))
        written.append(path)
    montage = output / "nieuwstadt1993_selected_overlays.png"
    _montage(images).save(montage)
    written.append(montage)
    manifest = output / "overlay_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "case": summary["case"],
                "figures": list(FIGURES),
                "results": str(results),
                "reference": str(reference),
                "diagnostic_metrics": summary["diagnostic_metrics"],
                "outputs": [path.name for path in written],
            },
            indent=2,
        )
        + "\n"
    )
    written.append(manifest)
    return written


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--legend-label",
        default="JAX-Wind uniform LASD 40×40×48 GPU",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output or args.results / "overlays"
    for path in overlay_results(
        args.results,
        args.reference,
        output,
        legend_label=args.legend_label,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
