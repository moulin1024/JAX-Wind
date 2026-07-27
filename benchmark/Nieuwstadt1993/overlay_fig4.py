#!/usr/bin/env python3
"""Overlay a JAX-Wind horizontal-variance profile on Nieuwstadt Fig. 4."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent

# Pixel calibration of reference/figures/fig4.png. The axes span
# x=<u_h'^2>/w_*^2 in [0, 0.6] and y=z/z_i in [0, 1.5].
AXIS_LEFT = 248.0
AXIS_RIGHT = 957.0
AXIS_TOP = 45.0
AXIS_BOTTOM = 739.0
X_MIN = 0.0
X_MAX = 0.6
Y_MIN = 0.0
Y_MAX = 1.5

SCALE = 4
TOTAL_COLOR = (205, 35, 45, 255)
SGS_COLOR = (0, 114, 178, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure",
        type=Path,
        default=BENCHMARK_DIR / "reference" / "figures" / "fig4.png",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=ROOT / "benchmark_results" / "Nieuwstadt1993" / "profiles.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark_results" / "Nieuwstadt1993" / "overlays" / "fig4_overlay.png",
    )
    parser.add_argument("--label", default="JAX LASD 40×40×48, FP32")
    return parser.parse_args()


def data_to_pixel(x_value: float, y_value: float) -> tuple[float, float]:
    x_pixel = AXIS_LEFT + (x_value - X_MIN) * (AXIS_RIGHT - AXIS_LEFT) / (X_MAX - X_MIN)
    y_pixel = AXIS_BOTTOM - (y_value - Y_MIN) * (AXIS_BOTTOM - AXIS_TOP) / (Y_MAX - Y_MIN)
    return SCALE * x_pixel, SCALE * y_pixel


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash: float,
    gap: float,
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


def main() -> None:
    args = parse_args()
    with args.profiles.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    samples = [
        (
            float(row["z_over_zi"]),
            float(row["horizontal_var_over_wstar_sq"]),
            float(row["horizontal_var_sgs_over_wstar_sq"]),
        )
        for row in rows
        if Y_MIN <= float(row["z_over_zi"]) <= Y_MAX
    ]
    samples.sort(key=lambda value: value[0])

    original = Image.open(args.figure).convert("RGBA")
    overlay = Image.new("RGBA", (original.width * SCALE, original.height * SCALE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    total_points = [data_to_pixel(total, z) for z, total, _ in samples]
    sgs_points = [data_to_pixel(sgs, z) for z, _, sgs in samples]
    draw.line(total_points, fill=TOTAL_COLOR, width=5 * SCALE, joint="curve")
    draw_dashed_line(
        draw,
        sgs_points,
        fill=SGS_COLOR,
        width=4 * SCALE,
        dash=12 * SCALE,
        gap=8 * SCALE,
    )

    # Put the new legend in the original scan's unused right margin so none of
    # the paper curves or measurements are obscured.
    legend_x = 1025 * SCALE
    legend_y = 180 * SCALE
    line_length = 92 * SCALE
    system_font = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    font_path = str(system_font) if system_font.exists() else "DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 30 * SCALE)
    label_font = ImageFont.truetype(font_path, 27 * SCALE)
    draw.text((legend_x, legend_y), args.label, font=font, fill=(0, 0, 0, 255))
    first_y = legend_y + 58 * SCALE
    draw.line(
        ((legend_x, first_y), (legend_x + line_length, first_y)),
        fill=TOTAL_COLOR,
        width=5 * SCALE,
    )
    draw.text(
        (legend_x + 112 * SCALE, first_y - 17 * SCALE),
        "total",
        font=label_font,
        fill=(0, 0, 0, 255),
    )
    second_y = first_y + 52 * SCALE
    draw_dashed_line(
        draw,
        [(legend_x, second_y), (legend_x + line_length, second_y)],
        fill=SGS_COLOR,
        width=4 * SCALE,
        dash=12 * SCALE,
        gap=8 * SCALE,
    )
    draw.text(
        (legend_x + 112 * SCALE, second_y - 17 * SCALE),
        "SGS contribution",
        font=label_font,
        fill=(0, 0, 0, 255),
    )

    overlay = overlay.resize(original.size, Image.Resampling.LANCZOS)
    result = Image.alpha_composite(original, overlay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.convert("RGB").save(args.output)
    print(f"[overlay] wrote {args.output}")


if __name__ == "__main__":
    main()
