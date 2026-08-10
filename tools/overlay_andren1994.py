#!/usr/bin/env python3
"""Overlay active JAX-Wind diagnostics on Andrén et al. figure panels.

The reference panel directory is data-only. This offline tool reads the panel
registration manifest and a completed result directory; it never participates
in case composition or solver execution.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    REPOSITORY_ROOT / "cases" / "Andren1994" / "reference" / "figure_panels"
)
OVERLAY_COLOR = (210, 24, 72)
SGS_COLOR = (22, 105, 178)
REFERENCE_ONLY_COLOR = (94, 104, 116)


@dataclass(frozen=True, slots=True)
class Axis:
    left: int
    top: int
    right: int
    bottom: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    xlog: bool = False
    ylog: bool = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        nargs="?",
        help="directory containing history, profile, spectrum, and summary outputs",
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--extract-from-pdf",
        type=Path,
        help="regenerate the reference panels from the cited article PDF",
    )
    return parser


def _manifest(reference: Path) -> dict[str, Any]:
    return json.loads((reference / "manifest.json").read_text(encoding="utf-8"))


def _render_page(pdf: Path, page: int, *, dpi: int) -> Image.Image:
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise RuntimeError("panel extraction requires the pdftoppm executable")
    with tempfile.TemporaryDirectory(prefix="andren1994-page-") as temporary:
        prefix = Path(temporary) / "page"
        subprocess.run(
            (
                executable,
                "-f",
                str(page),
                "-l",
                str(page),
                "-singlefile",
                "-r",
                str(dpi),
                "-png",
                str(pdf),
                str(prefix),
            ),
            check=True,
        )
        with Image.open(prefix.with_suffix(".png")) as image:
            return image.convert("L")


def extract_reference_panels(pdf: Path, reference: Path) -> tuple[Path, ...]:
    """Reproduce the checked-in panel crops from the cited article scan."""

    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    reference.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(reference)
    dpi = int(manifest["render"]["dpi"])
    expected_size = tuple(manifest["render"]["page_size_pixels"])
    pages = {
        int(specification["page"])
        for specification in manifest["panels"].values()
    }
    rendered = {
        page: _render_page(pdf, page, dpi=dpi)
        for page in sorted(pages)
    }
    for page, image in rendered.items():
        if image.size != expected_size:
            raise ValueError(
                f"article page {page} rendered as {image.size}, expected "
                f"{expected_size}; use the DLR scan recorded in manifest.json"
            )

    outputs = []
    for specification in manifest["panels"].values():
        source = rendered[int(specification["page"])]
        panel = source.crop(tuple(specification["crop"]))
        output = reference / specification["file"]
        panel.save(output, optimize=True)
        outputs.append(output)
    return tuple(outputs)


def _axis(manifest: dict[str, Any], name: str) -> Axis:
    registration = manifest["overlay_axes"][name]
    left, top, right, bottom = registration["frame"]
    xmin, xmax, ymin, ymax = registration["limits"]
    scale = registration.get("scale", ["linear", "linear"])
    return Axis(
        left,
        top,
        right,
        bottom,
        xmin,
        xmax,
        ymin,
        ymax,
        xlog=scale[0] == "log",
        ylog=scale[1] == "log",
    )


def _points(axis: Axis, x: np.ndarray, y: np.ndarray) -> list[tuple[int, int]]:
    finite = np.isfinite(x) & np.isfinite(y)
    inside = (
        finite
        & (x >= axis.xmin)
        & (x <= axis.xmax)
        & (y >= axis.ymin)
        & (y <= axis.ymax)
    )
    if axis.xlog:
        inside &= x > 0.0
    if axis.ylog:
        inside &= y > 0.0
    x = x[inside]
    y = y[inside]
    transform_x = np.log10 if axis.xlog else np.asarray
    transform_y = np.log10 if axis.ylog else np.asarray
    transformed_x = transform_x(x)
    transformed_y = transform_y(y)
    xmin, xmax = transform_x(np.asarray((axis.xmin, axis.xmax)))
    ymin, ymax = transform_y(np.asarray((axis.ymin, axis.ymax)))
    px = axis.left + (transformed_x - xmin) * (axis.right - axis.left) / (
        xmax - xmin
    )
    py = axis.bottom - (transformed_y - ymin) * (axis.bottom - axis.top) / (
        ymax - ymin
    )
    return [
        (int(round(x_pixel)), int(round(y_pixel)))
        for x_pixel, y_pixel in zip(px, py, strict=True)
    ]


def _draw_curve(
    image: Image.Image,
    axis: Axis,
    x,
    y,
    *,
    color: tuple[int, int, int] = OVERLAY_COLOR,
) -> None:
    points = _points(axis, np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if len(points) < 2:
        raise ValueError("fewer than two result points lie inside a paper axis")
    draw = ImageDraw.Draw(image)
    draw.line(points, fill="white", width=9, joint="curve")
    draw.line(points, fill=color, width=5, joint="curve")


def _profile(results: Path) -> np.ndarray:
    path = results / "profiles.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run through the configured statistics window first"
        )
    profile = np.genfromtxt(path, delimiter=",", names=True)
    required = {
        "z_m",
        "z_f_over_ustar",
        "mean_u_m_s",
        "mean_v_m_s",
        "mean_scalar_kg_m3",
        "resolved_u_variance_m2_s2",
        "resolved_v_variance_m2_s2",
        "resolved_w_variance_m2_s2",
        "resolved_scalar_variance_kg2_m6",
    }
    missing = required - set(profile.dtype.names or ())
    if missing:
        raise ValueError("profiles.csv is missing: " + ", ".join(sorted(missing)))
    return np.atleast_1d(profile)


def _normalization(results: Path) -> tuple[float, float, float]:
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    physics = summary["physics"]
    geostrophic = physics["geostrophic_velocity_m_s"]
    geostrophic_speed = math.hypot(float(geostrophic[0]), float(geostrophic[1]))
    ratio = summary.get("diagnostic_metrics", {}).get(
        "surface_friction_velocity_ratio"
    )
    if ratio is None:
        ratio = summary.get("comparison", {}).get("ustar_over_ug")
    if ratio is None:
        ustar = summary["runtime"].get("ustar_m_s")
        if ustar is None:
            raise ValueError("summary.json does not contain a friction velocity")
        friction_velocity = float(ustar)
    else:
        friction_velocity = float(ratio) * geostrophic_speed
    scalar_flux = float(
        physics.get(
            "scalar_surface_flux",
            physics.get("passive_scalar_surface_flux_kg_m2_s"),
        )
    )
    coriolis = float(physics.get("coriolis_vertical_s", 1.0e-4))
    return friction_velocity, scalar_flux, coriolis


def _csv(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if candidate.is_file():
        return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _complete_montage(
    manifest: dict[str, Any],
    reference: Path,
    overlaid: dict[str, Image.Image],
) -> Image.Image:
    """Lay out every published figure, substituting available overlays."""

    columns = 3
    tile_width = 700
    maximum_image_height = 900
    outer_gap = 28
    column_gap = 22
    row_gap = 28
    title_height = 58
    header_height = 142
    title_font = _font(25)
    status_font = _font(17)
    header_font = _font(34)
    legend_font = _font(20)

    prepared = []
    for number, specification in manifest["panels"].items():
        if number in overlaid:
            image = overlaid[number]
        else:
            with Image.open(reference / specification["file"]) as source:
                image = source.convert("RGB")
        scale = min(
            tile_width / image.width,
            maximum_image_height / image.height,
        )
        resized = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
        prepared.append((number, resized))

    rows = [
        prepared[index : index + columns]
        for index in range(0, len(prepared), columns)
    ]
    row_heights = [
        title_height + max(image.height for _, image in row)
        for row in rows
    ]
    width = 2 * outer_gap + columns * tile_width + (columns - 1) * column_gap
    height = (
        2 * outer_gap
        + header_height
        + sum(row_heights)
        + (len(rows) - 1) * row_gap
    )
    montage = Image.new("RGB", (width, height), (242, 244, 247))
    draw = ImageDraw.Draw(montage)
    draw.text(
        (outer_gap, outer_gap),
        "Andrén et al. (1994): complete reference + JAX-Wind diagnostic overlays",
        font=header_font,
        fill=(20, 25, 32),
    )
    legend_y = outer_gap + 58
    draw.line(
        (outer_gap, legend_y + 12, outer_gap + 70, legend_y + 12),
        fill=OVERLAY_COLOR,
        width=5,
    )
    draw.text(
        (outer_gap + 88, legend_y),
        "red: active JAX-Wind result",
        font=legend_font,
        fill=(40, 46, 54),
    )
    draw.text(
        (outer_gap, legend_y + 36),
        "reference only: this run did not record the required observable",
        font=legend_font,
        fill=REFERENCE_ONLY_COLOR,
    )

    y = outer_gap + header_height
    for row, row_height in zip(rows, row_heights, strict=True):
        for column, (number, image) in enumerate(row):
            x = outer_gap + column * (tile_width + column_gap)
            draw.rounded_rectangle(
                (x, y, x + tile_width, y + row_height),
                radius=8,
                fill="white",
                outline=(207, 212, 220),
                width=2,
            )
            draw.text(
                (x + 14, y + 10),
                f"Figure {int(number)}",
                font=title_font,
                fill=(20, 25, 32),
            )
            status = "JAX-Wind overlay" if number in overlaid else "reference only"
            status_color = (
                OVERLAY_COLOR if number in overlaid else REFERENCE_ONLY_COLOR
            )
            bounds = draw.textbbox((0, 0), status, font=status_font)
            draw.text(
                (x + tile_width - (bounds[2] - bounds[0]) - 14, y + 17),
                status,
                font=status_font,
                fill=status_color,
            )
            montage.paste(
                image,
                (x + (tile_width - image.width) // 2, y + title_height),
            )
        y += row_height + row_gap
    return montage


def overlay_results(
    results: Path,
    reference: Path,
    output: Path,
) -> tuple[Path, ...]:
    """Overlay diagnostics available in the active profiles output."""

    manifest = _manifest(reference)
    profile = _profile(results)
    ustar, scalar_flux, coriolis = _normalization(results)
    height = np.asarray(profile["z_f_over_ustar"], dtype=float)
    z = np.asarray(profile["z_m"], dtype=float)
    profile_names = set(profile.dtype.names or ())
    du_dz = np.gradient(np.asarray(profile["mean_u_m_s"], dtype=float), z)
    dv_dz = np.gradient(np.asarray(profile["mean_v_m_s"], dtype=float), z)
    phi_m = 0.4 * z * np.hypot(du_dz, dv_dz) / ustar
    phi_m[0] = 1.0
    concentration_scale = scalar_flux / ustar

    output.mkdir(parents=True, exist_ok=True)
    panel_specs = manifest["panels"]

    def panel(number: int) -> Image.Image:
        specification = panel_specs[f"{number:02d}"]
        with Image.open(reference / specification["file"]) as source:
            return source.convert("RGB")

    overlaid: dict[str, Image.Image] = {}

    history = _csv(results / "history.csv")
    if history is not None:
        history_names = set(history.dtype.names or ())
        if {"time_hours", "integrated_total_tke_m3_s2"}.issubset(history_names):
            figure2 = panel(2)
            _draw_curve(
                figure2,
                _axis(manifest, "figure_02_total_tke"),
                np.asarray(history["time_hours"], dtype=float) * 3600.0 * coriolis,
                coriolis
                * np.asarray(history["integrated_total_tke_m3_s2"], dtype=float)
                / ustar**3,
            )
            overlaid["02"] = figure2
        if {
            "time_hours",
            "momentum_stationarity_cu",
            "momentum_stationarity_cv",
        }.issubset(history_names):
            figure3 = panel(3)
            nondimensional_time = (
                np.asarray(history["time_hours"], dtype=float) * 3600.0 * coriolis
            )
            _draw_curve(
                figure3,
                _axis(manifest, "figure_03_cu"),
                nondimensional_time,
                np.asarray(history["momentum_stationarity_cu"], dtype=float),
            )
            _draw_curve(
                figure3,
                _axis(manifest, "figure_03_cv"),
                nondimensional_time,
                np.asarray(history["momentum_stationarity_cv"], dtype=float),
            )
            overlaid["03"] = figure3

    figure4 = panel(4)
    _draw_curve(
        figure4,
        _axis(manifest, "figure_04_momentum_gradient"),
        phi_m,
        height,
    )
    if concentration_scale != 0.0:
        scalar_gradient = np.gradient(
            np.asarray(profile["mean_scalar_kg_m3"], dtype=float),
            z,
        )
        phi_c = -0.4 * z * scalar_gradient / concentration_scale
        phi_c[0] = 1.0
        _draw_curve(
            figure4,
            _axis(manifest, "figure_04_scalar_gradient"),
            phi_c,
            height,
        )
    overlaid["04"] = figure4

    figure5 = panel(5)
    for axis_name, column in (
        ("figure_05_u_variance", "resolved_u_variance_m2_s2"),
        ("figure_05_v_variance", "resolved_v_variance_m2_s2"),
        ("figure_05_w_variance", "resolved_w_variance_m2_s2"),
    ):
        _draw_curve(
            figure5,
            _axis(manifest, axis_name),
            np.asarray(profile[column], dtype=float) / ustar**2,
            height,
        )
    overlaid["05"] = figure5

    if {
        "total_uw_m2_s2",
        "total_vw_m2_s2",
    }.issubset(profile_names):
        figure6 = panel(6)
        _draw_curve(
            figure6,
            _axis(manifest, "figure_06_uw_flux"),
            np.asarray(profile["total_uw_m2_s2"], dtype=float) / ustar**2,
            height,
        )
        _draw_curve(
            figure6,
            _axis(manifest, "figure_06_vw_flux"),
            np.asarray(profile["total_vw_m2_s2"], dtype=float) / ustar**2,
            height,
        )
        overlaid["06"] = figure6

    if concentration_scale != 0.0:
        figure7 = panel(7)
        _draw_curve(
            figure7,
            _axis(manifest, "figure_07_scalar_variance"),
            np.asarray(profile["resolved_scalar_variance_kg2_m6"], dtype=float)
            / concentration_scale**2,
            height,
        )
        overlaid["07"] = figure7

    if (
        scalar_flux != 0.0
        and {"total_wc_kg_m2_s", "sgs_wc_kg_m2_s"}.issubset(profile_names)
    ):
        figure8 = panel(8)
        _draw_curve(
            figure8,
            _axis(manifest, "figure_08_scalar_flux"),
            np.asarray(profile["total_wc_kg_m2_s"], dtype=float) / scalar_flux,
            height,
        )
        _draw_curve(
            figure8,
            _axis(manifest, "figure_08_scalar_flux"),
            np.asarray(profile["sgs_wc_kg_m2_s"], dtype=float) / scalar_flux,
            height,
            color=SGS_COLOR,
        )
        overlaid["08"] = figure8

    if "resolved_tke_sgs_transfer_m2_s3" in profile_names:
        figure11 = panel(11)
        _draw_curve(
            figure11,
            _axis(manifest, "figure_11_tke_sgs_transfer"),
            np.asarray(
                profile["resolved_tke_sgs_transfer_m2_s3"], dtype=float
            )
            / (coriolis * ustar**2),
            height,
        )
        overlaid["11"] = figure11

    if {
        "momentum_diffusivity_m2_s",
        "scalar_diffusivity_m2_s",
    }.issubset(profile_names):
        figure14 = panel(14)
        _draw_curve(
            figure14,
            _axis(manifest, "figure_14_momentum_diffusivity"),
            np.asarray(profile["momentum_diffusivity_m2_s"], dtype=float),
            height,
        )
        _draw_curve(
            figure14,
            _axis(manifest, "figure_14_scalar_diffusivity"),
            np.asarray(profile["scalar_diffusivity_m2_s"], dtype=float),
            height,
        )
        overlaid["14"] = figure14

    spectra = _csv(results / "spectra.csv")
    if spectra is not None:
        spectrum_names = set(spectra.dtype.names or ())
        columns = (
            ("figure_15_u_spectrum", "kEu_over_ustar2"),
            ("figure_15_v_spectrum", "kEv_over_ustar2"),
            ("figure_15_w_spectrum", "kEw_over_ustar2"),
            ("figure_15_scalar_spectrum", "kEc_over_cstar2"),
        )
        required = {"k_ustar_over_f", *(column for _, column in columns)}
        if required.issubset(spectrum_names):
            figure15 = panel(15)
            for axis_name, column in columns:
                _draw_curve(
                    figure15,
                    _axis(manifest, axis_name),
                    np.asarray(spectra["k_ustar_over_f"], dtype=float),
                    np.asarray(spectra[column], dtype=float),
                )
            overlaid["15"] = figure15

    outputs = []
    for number, image in sorted(overlaid.items()):
        path = output / f"figure_{number}_overlay.png"
        image.save(path, optimize=True)
        outputs.append(path)

    gap = 24
    ordered = sorted(overlaid.items())
    width = max(image.width for _, image in ordered)
    height_total = sum(image.height for _, image in ordered) + gap * (
        len(ordered) - 1
    )
    montage = Image.new("RGB", (width, height_total), "white")
    y = 0
    for _, image in ordered:
        montage.paste(image, ((width - image.width) // 2, y))
        y += image.height + gap
    montage_path = output / "andren1994_profile_overlays.png"
    montage.save(montage_path, optimize=True)
    outputs.append(montage_path)

    complete = _complete_montage(manifest, reference, overlaid)
    complete_path = output / "andren1994_complete_overlay.png"
    complete.save(complete_path, optimize=True)
    outputs.append(complete_path)

    comparison_manifest = {
        "overlaid_figures": sorted(int(number) for number in overlaid),
        "reference_only_figures": [
            int(number)
            for number in manifest["panels"]
            if number not in overlaid
        ],
        "diagnostic_outputs": {
            "history": "history.csv",
            "profiles": "profiles.csv",
            "spectra": "spectra.csv" if spectra is not None else None,
        },
    }
    comparison_path = output / "overlay_manifest.json"
    comparison_path.write_text(
        json.dumps(comparison_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs.append(comparison_path)
    return tuple(outputs)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.extract_from_pdf is not None:
        outputs = extract_reference_panels(args.extract_from_pdf, args.reference)
        print(f"extracted {len(outputs)} reference panels to {args.reference}")
    if args.results is None:
        if args.extract_from_pdf is None:
            raise SystemExit("results is required unless --extract-from-pdf is used")
        return 0
    output = args.output or args.results / "paper_overlays"
    outputs = overlay_results(args.results, args.reference, output)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
