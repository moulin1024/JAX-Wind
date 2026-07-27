#!/usr/bin/env python3
"""Overlay new semantic WIRE-LES diagnostics on Nieuwstadt Figs. 1--17."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


SCALE = 4
RED = (205, 35, 45, 255)
BLUE = (0, 114, 178, 255)
GREEN = (0, 135, 95, 255)
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    BENCHMARK_DIR / "Nieuwstadt1993_LASD_complete_overlay.png"
)
COMPOSITE_COLUMNS = 4
COMPOSITE_TILE_SIZE = (724, 505)
COMPOSITE_GAP = 8
COMPOSITE_HEADER_HEIGHT = 132
COMPOSITE_TITLE_HEIGHT = 50

FIGURE_STATUS = {
    1: "WIRE-LES total energy",
    2: "WIRE-LES total + SGS heat flux",
    3: "WIRE-LES total + SGS variance",
    4: "WIRE-LES total + SGS variance",
    5: "WIRE-LES total + SGS variance",
    6: "WIRE-LES pressure RMS",
    7: "WIRE-LES third moment",
    8: "WIRE-LES skewness",
    9: "WIRE-LES dissipation",
    10: "WIRE-LES velocity transport",
    11: "WIRE-LES pressure transport",
    12: "WIRE-LES updraft fraction",
    13: "WIRE-LES conditional velocity",
    14: "WIRE-LES temperature excess",
    15: "WIRE-LES horizontal spectra",
    16: "WIRE-LES vertical spectra",
    17: "WIRE-LES temperature spectra",
}


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Series:
    label: str
    x: np.ndarray
    y: np.ndarray
    color: tuple[int, int, int, int] = RED
    dashed: bool = False


AXES = {
    1: Axes(205, 819, 27, 633, 0.0, 12.0, 0.0, 1.0),
    2: Axes(159, 781, 43, 660, -0.4, 1.1, 0.0, 1.5),
    3: Axes(212, 921, 65, 756, 0.0, 0.8, 0.0, 1.5),
    4: Axes(248, 957, 45, 739, 0.0, 0.6, 0.0, 1.5),
    5: Axes(168, 785, 38, 641, 0.0, 35.0, 0.0, 1.5),
    6: Axes(188, 806, 31, 639, 0.0, 0.5, 0.0, 1.5),
    7: Axes(247, 956, 54, 745, -0.1, 0.3, 0.0, 1.5),
    8: Axes(177, 796, 40, 642, -4.0, 4.0, 0.0, 1.5),
    9: Axes(164, 783, 34, 644, 0.0, 1.5, 0.0, 1.5),
    10: Axes(184, 803, 27, 647, -0.5, 0.5, 0.0, 1.5),
    11: Axes(169, 786, 31, 635, -0.6, 0.4, 0.0, 1.5),
    12: Axes(212, 830, 19, 614, 0.2, 0.6, 0.0, 1.5),
    13: Axes(183, 801, 25, 632, 0.0, 0.7, 0.0, 1.5),
    14: Axes(255, 872, 39, 642, -1.0, 2.0, 0.0, 1.5),
    15: Axes(193, 810, 24, 627, 1.0, 100.0, 1.0e-4, 1.0e1, True, True),
    16: Axes(192, 810, 23, 631, 1.0, 100.0, 1.0e-3, 1.0e2, True, True),
    17: Axes(196, 813, 49, 652, 1.0, 100.0, 1.0e-2, 2.0e3, True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=BENCHMARK_DIR / "reference" / "figures",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "Nieuwstadt1993",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--legend-label",
        default="WIRE-LES new LASD 40×40×48 GPU",
        help="Header shown in the added legend.",
    )
    return parser.parse_args()


def coordinate(value: float, lower: float, upper: float, log_scale: bool) -> float:
    if log_scale:
        return (np.log10(value) - np.log10(lower)) / (np.log10(upper) - np.log10(lower))
    return (value - lower) / (upper - lower)


def data_to_pixel(x: float, y: float, axes: Axes) -> tuple[float, float]:
    x_fraction = coordinate(x, axes.xmin, axes.xmax, axes.xlog)
    y_fraction = coordinate(y, axes.ymin, axes.ymax, axes.ylog)
    return (
        SCALE * (axes.left + x_fraction * (axes.right - axes.left)),
        SCALE * (axes.bottom - y_fraction * (axes.bottom - axes.top)),
    )


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash: float = 12 * SCALE,
    gap: float = 8 * SCALE,
) -> None:
    for start, end in zip(points[:-1], points[1:], strict=True):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0.0:
            continue
        distance = 0.0
        while distance < length:
            dash_end = min(distance + dash, length)
            p0 = (start[0] + dx * distance / length, start[1] + dy * distance / length)
            p1 = (start[0] + dx * dash_end / length, start[1] + dy * dash_end / length)
            draw.line((p0, p1), fill=fill, width=width)
            distance += dash + gap


def finite_segments(series: Series, axes: Axes) -> list[list[tuple[float, float]]]:
    valid = np.isfinite(series.x) & np.isfinite(series.y)
    if axes.xlog:
        valid &= series.x > 0.0
    if axes.ylog:
        valid &= series.y > 0.0
    valid &= (series.x >= axes.xmin) & (series.x <= axes.xmax)
    valid &= (series.y >= axes.ymin) & (series.y <= axes.ymax)

    # Split at out-of-range samples. Drawing one continuous polyline and then
    # clipping the raster can create artificial horizontal strokes along the
    # axes boundary when a noisy profile leaves and re-enters the plot box.
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for x, y, keep in zip(series.x, series.y, valid, strict=True):
        if keep:
            current.append(data_to_pixel(float(x), float(y), axes))
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


def build_series(
    profiles: dict[str, np.ndarray],
    time_series: dict[str, np.ndarray],
    stats: np.lib.npyio.NpzFile,
) -> dict[int, list[Series]]:
    z = profiles["z_over_zi"]
    z_over_zi0 = profiles["z"] / float(stats["zi0"])
    result: dict[int, list[Series]] = {
        1: [Series("total energy", time_series["time_over_tstar0"], time_series["energy_bl_over_wstar0_sq"])],
        2: [
            Series("total", profiles["heat_flux_total_over_qs"], z_over_zi0),
            Series("SGS contribution", profiles["heat_flux_sgs_over_qs"], z_over_zi0, BLUE, True),
        ],
        3: [
            Series("total", profiles["w_var_over_wstar_sq"], z),
            Series("SGS contribution", profiles["w_var_sgs_over_wstar_sq"], z, BLUE, True),
        ],
        4: [
            Series("total", profiles["horizontal_var_over_wstar_sq"], z),
            Series("SGS contribution", profiles["horizontal_var_sgs_over_wstar_sq"], z, BLUE, True),
        ],
        5: [
            Series("total", profiles["theta_var_over_thetastar_sq"], z),
            Series(
                "SGS contribution",
                profiles["theta_var_sgs_over_thetastar_sq"],
                z,
                BLUE,
                True,
            ),
        ],
        6: [Series("pressure RMS", np.sqrt(np.maximum(profiles["p_var_over_wstar4"], 0.0)), z)],
        7: [Series("third moment", profiles["w3_over_wstar3"], z)],
        8: [Series("skewness", profiles["skewness"], z)],
        9: [Series("dissipation", profiles["epsilon_zi_over_wstar3"], z)],
        10: [Series("velocity transport", -profiles["d_w_transport"], z)],
        11: [Series("pressure transport", profiles["d_p_transport"], z)],
        12: [Series("updraft fraction", profiles["alpha_u"], z)],
        13: [Series("conditional velocity", profiles["w_u_over_wstar"], z)],
        14: [Series("temperature excess", profiles["theta_u_excess_over_thetastar"], z)],
    }

    kzi = np.asarray(stats["spectrum_kzi"])
    wstar2 = float(stats["wstar_mean"]) ** 2
    theta_star2 = float(stats["theta_star_mean"]) ** 2
    level_fraction = np.asarray(stats["spectrum_level_fraction"])
    spectrum_specs = {
        15: (np.asarray(stats["spectrum_u"]) / wstar2, (1.0, 10.0, 100.0)),
        16: (np.asarray(stats["spectrum_w"]) / wstar2, (1.0, 10.0, 1000.0)),
        17: (np.asarray(stats["spectrum_theta"]) / theta_star2, (1.0, 100.0, 1000.0)),
    }
    colors = (RED, BLUE, GREEN)
    for figure, (spectrum, offsets) in spectrum_specs.items():
        result[figure] = [
            Series(
                f"z/zi={level_fraction[level]:.1f} ×{offset:g}",
                kzi,
                kzi * spectrum[level] * offset,
                colors[level],
                dashed=level > 0,
            )
            for level, offset in enumerate(offsets)
        ]
    return result


def overlay_figure(
    path: Path,
    axes: Axes,
    series_list: list[Series],
    legend_label: str,
) -> Image.Image:
    original = Image.open(path).convert("RGBA")
    size = (original.width * SCALE, original.height * SCALE)
    curves = Image.new("RGBA", size, (0, 0, 0, 0))
    curve_draw = ImageDraw.Draw(curves)
    for series in series_list:
        for points in finite_segments(series, axes):
            if len(points) < 2:
                continue
            if series.dashed:
                draw_dashed_line(curve_draw, points, fill=series.color, width=4 * SCALE)
            else:
                curve_draw.line(points, fill=series.color, width=5 * SCALE, joint="curve")

    # Confine data to the scanned plot rectangle while leaving a separate
    # unclipped layer for the color legend in the unused page margin.
    clip = Image.new("L", size, 0)
    ImageDraw.Draw(clip).rectangle(
        (
            int(axes.left * SCALE),
            int(axes.top * SCALE),
            int(axes.right * SCALE),
            int(axes.bottom * SCALE),
        ),
        fill=255,
    )
    curves.putalpha(ImageChops.multiply(curves.getchannel("A"), clip))

    legend = Image.new("RGBA", size, (0, 0, 0, 0))
    legend_draw = ImageDraw.Draw(legend)
    font_path = FONT_PATH if Path(FONT_PATH).exists() else "DejaVuSans.ttf"
    header_font = ImageFont.truetype(font_path, 21 * SCALE)
    label_font = ImageFont.truetype(font_path, 21 * SCALE)
    legend_x = min((axes.right + 45) * SCALE, (original.width - 300) * SCALE)
    legend_y = (axes.top + 48) * SCALE
    legend_draw.text(
        (legend_x, legend_y),
        legend_label,
        font=header_font,
        fill=(0, 0, 0, 255),
    )
    for index, series in enumerate(series_list):
        y = legend_y + (48 + 39 * index) * SCALE
        start = (legend_x, y)
        end = (legend_x + 70 * SCALE, y)
        if series.dashed:
            draw_dashed_line(legend_draw, [start, end], fill=series.color, width=4 * SCALE)
        else:
            legend_draw.line((start, end), fill=series.color, width=5 * SCALE)
        legend_draw.text(
            (legend_x + 88 * SCALE, y - 13 * SCALE),
            series.label,
            font=label_font,
            fill=(0, 0, 0, 255),
        )

    curves = curves.resize(original.size, Image.Resampling.LANCZOS)
    legend = legend.resize(original.size, Image.Resampling.LANCZOS)
    result = Image.alpha_composite(Image.alpha_composite(original, curves), legend)
    return result.convert("RGB")


def composite_font(size: int) -> ImageFont.FreeTypeFont:
    font_path = FONT_PATH if Path(FONT_PATH).exists() else "DejaVuSans.ttf"
    return ImageFont.truetype(font_path, size)


def compose_figures(figures: list[Image.Image]) -> Image.Image:
    rows = (len(figures) + COMPOSITE_COLUMNS - 1) // COMPOSITE_COLUMNS
    tile_width, tile_height = COMPOSITE_TILE_SIZE
    width = (
        COMPOSITE_COLUMNS * tile_width
        + (COMPOSITE_COLUMNS + 1) * COMPOSITE_GAP
    )
    height = (
        COMPOSITE_HEADER_HEIGHT
        + rows * tile_height
        + (rows + 1) * COMPOSITE_GAP
    )
    composite = Image.new(
        "RGB",
        (width, height),
        (242, 244, 247),
    )
    draw = ImageDraw.Draw(composite)
    draw.text(
        (COMPOSITE_GAP, 10),
        "Nieuwstadt et al. (1993): paper figures + new semantic WIRE-LES LASD",
        font=composite_font(34),
        fill=(20, 25, 32),
    )
    legend_y = 63
    draw.line((COMPOSITE_GAP, legend_y + 10, 78, legend_y + 10), fill=RED, width=5)
    draw.text(
        (92, legend_y),
        "red = total/resolved diagnostic; blue dashed = SGS contribution; spectra use red/blue/green by height",
        font=composite_font(18),
        fill=(40, 46, 54),
    )
    draw.text(
        (COMPOSITE_GAP, 96),
        "Official paper scan (DOI 10.1007/978-3-642-77674-8_24); averaging window 10 < t/t* < 11",
        font=composite_font(18),
        fill=(80, 86, 94),
    )
    for index, figure in enumerate(figures):
        figure_number = index + 1
        column = index % COMPOSITE_COLUMNS
        row = index // COMPOSITE_COLUMNS
        x0 = COMPOSITE_GAP + column * (tile_width + COMPOSITE_GAP)
        y0 = (
            COMPOSITE_HEADER_HEIGHT
            + COMPOSITE_GAP
            + row * (tile_height + COMPOSITE_GAP)
        )
        draw.rounded_rectangle(
            (x0, y0, x0 + tile_width, y0 + tile_height),
            radius=7,
            fill="white",
            outline=(207, 212, 220),
            width=1,
        )
        draw.text(
            (x0 + 9, y0 + 8),
            f"Figure {figure_number}",
            font=composite_font(19),
            fill=(20, 25, 32),
        )
        status = FIGURE_STATUS[figure_number]
        status_font = composite_font(14)
        status_width = draw.textbbox((0, 0), status, font=status_font)[2]
        draw.text(
            (x0 + tile_width - status_width - 9, y0 + 11),
            status,
            font=status_font,
            fill=RED,
        )
        fitted = ImageOps.contain(
            figure,
            (tile_width - 12, tile_height - COMPOSITE_TITLE_HEIGHT - 6),
            Image.Resampling.LANCZOS,
        )
        x = x0 + (tile_width - fitted.width) // 2
        y = y0 + COMPOSITE_TITLE_HEIGHT + (
            tile_height - COMPOSITE_TITLE_HEIGHT - fitted.height
        ) // 2
        composite.paste(fitted, (x, y))
    return composite


def main() -> None:
    args = parse_args()
    profiles = read_csv_columns(args.result_dir / "profiles.csv")
    time_series = read_csv_columns(args.result_dir / "time_series.csv")
    stats = np.load(args.result_dir / "benchmark_stats.npz")
    figure_series = build_series(profiles, time_series, stats)
    overlaid = []
    for figure in range(1, 18):
        source = args.figure_dir / f"fig{figure}.png"
        overlaid.append(
            overlay_figure(
                source,
                AXES[figure],
                figure_series[figure],
                args.legend_label,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compose_figures(overlaid).save(args.output)
    print(f"[overlay] wrote complete figure collection {args.output}")


if __name__ == "__main__":
    main()
