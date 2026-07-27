#!/usr/bin/env python3
"""Overlay WIRE-LES statistics on the original Andrén et al. (1994) figures.

The article PDF is deliberately supplied by the caller and is never copied into the
benchmark tree.  The page/axis registration below targets the institutional scan linked
from ``reference/Andren1994.md``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from numpy.typing import NDArray


REFERENCE_PAGE_SIZE = (1008, 1440)
OVERLAY_COLOR = (210, 24, 72)
SGS_COLOR = (22, 105, 178)
PAPER_ONLY_COLOR = (94, 104, 116)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class FigureSpec:
    number: int
    page: int
    crop: tuple[int, int, int, int]
    comparison: str | None = None


# Pixel registration of the actual Fig. 7 plot frame on rendered PDF page 13.
# Its printed abscissa ends at 8.0; caption/legend whitespace is not axis area.
FIGURE7_AXIS = Axis(324, 832, 736, 1243, 0.0, 8.0, 0.0, 0.35)
FIGURE14_AXES = (
    Axis(332, 138, 740, 551, 0.0, 10.0, 0.0, 0.35),
    Axis(327, 806, 735, 1217, 0.0, 15.0, 0.0, 0.35),
)
FIGURE13_AXES = (
    Axis(190, 138, 483, 436, -40.0, 40.0, 0.0, 0.35),
    Axis(568, 138, 861, 436, -40.0, 40.0, 0.0, 0.35),
    Axis(190, 520, 483, 817, -40.0, 40.0, 0.0, 0.35),
    Axis(568, 520, 861, 817, -40.0, 40.0, 0.0, 0.35),
)
FIGURE15_AXES = (
    Axis(196, 521, 489, 818, 1.0, 1000.0, 0.01, 10.0, True, True),
    Axis(573, 522, 869, 818, 1.0, 1000.0, 0.01, 10.0, True, True),
    Axis(195, 902, 488, 1198, 1.0, 1000.0, 0.01, 10.0, True, True),
    Axis(572, 902, 867, 1199, 1.0, 1000.0, 0.01, 10.0, True, True),
)


# Crop registration for every numbered figure in the institutional article scan.
# Captions are retained wherever they fit on the figure's page so the combined sheet
# remains useful without a separate caption lookup.
FIGURES = (
    FigureSpec(1, 5, (120, 455, 900, 1370)),
    FigureSpec(2, 8, (100, 90, 900, 675), "WIRE-LES total TKE"),
    FigureSpec(3, 9, (205, 90, 830, 1365)),
    FigureSpec(4, 10, (250, 90, 820, 1360), "WIRE-LES on panels (a,b)"),
    FigureSpec(5, 11, (100, 90, 900, 900), "WIRE-LES total variances"),
    FigureSpec(6, 12, (100, 90, 900, 1335), "WIRE-LES total flux"),
    FigureSpec(7, 13, (250, 750, 795, 1370), "WIRE-LES scalar variance"),
    FigureSpec(8, 14, (245, 90, 800, 690), "WIRE-LES scalar flux"),
    FigureSpec(9, 15, (215, 90, 815, 700)),
    FigureSpec(10, 15, (215, 730, 815, 1365)),
    FigureSpec(11, 17, (220, 90, 815, 690)),
    FigureSpec(12, 18, (125, 90, 905, 1370)),
    FigureSpec(13, 19, (115, 90, 905, 985), "WIRE-LES complete scalar-flux budget"),
    FigureSpec(14, 20, (240, 90, 810, 1350), "WIRE-LES LASD diffusivities"),
    FigureSpec(15, 21, (105, 450, 905, 1370), "WIRE-LES spectra"),
    FigureSpec(16, 23, (195, 90, 815, 1340)),
    FigureSpec(17, 24, (205, 90, 815, 700)),
    FigureSpec(18, 24, (205, 720, 815, 1365)),
    FigureSpec(19, 25, (100, 455, 905, 1370)),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-pdf", type=Path, required=True)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmark/Andren1994/results/lasd_40x40x40"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _render_page(pdf: Path, page_number: int) -> Image.Image:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on optional PDF runtime
        rendered = pdf.parent / "rendered" / f"page-{page_number:02d}.png"
        if rendered.exists():
            image = Image.open(rendered).convert("RGB")
            if image.size != REFERENCE_PAGE_SIZE:
                raise ValueError(
                    f"unexpected cached page size {image.size}; "
                    f"expected {REFERENCE_PAGE_SIZE}"
                )
            return image
        raise RuntimeError(
            "PyMuPDF is required unless page PNGs are cached beside the PDF under "
            "rendered/page-NN.png: python -m pip install pymupdf"
        ) from exc
    document = fitz.open(pdf)
    try:
        pixmap = document[page_number - 1].get_pixmap(
            matrix=fitz.Matrix(2.0, 2.0), alpha=False
        )
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    finally:
        document.close()
    if image.size != REFERENCE_PAGE_SIZE:
        raise ValueError(
            f"unexpected rendered page size {image.size}; expected {REFERENCE_PAGE_SIZE} "
            "for the institutional Andrén et al. scan"
        )
    return image


def _finite_pairs(
    x: NDArray[np.floating], y: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    selected = np.isfinite(x) & np.isfinite(y)
    return x[selected], y[selected]


def _pixels(
    axis: Axis, x: NDArray[np.floating], y: NDArray[np.floating]
) -> list[tuple[int, int]]:
    x, y = _finite_pairs(np.asarray(x), np.asarray(y))
    selected = (x >= axis.xmin) & (x <= axis.xmax) & (y >= axis.ymin) & (y <= axis.ymax)
    if axis.xlog:
        selected &= x > 0.0
    if axis.ylog:
        selected &= y > 0.0
    x = x[selected]
    y = y[selected]
    transform_x = np.log10 if axis.xlog else np.asarray
    transform_y = np.log10 if axis.ylog else np.asarray
    tx = transform_x(x)
    ty = transform_y(y)
    xmin, xmax = transform_x(np.asarray((axis.xmin, axis.xmax)))
    ymin, ymax = transform_y(np.asarray((axis.ymin, axis.ymax)))
    px = axis.left + (tx - xmin) * (axis.right - axis.left) / (xmax - xmin)
    py = axis.bottom - (ty - ymin) * (axis.bottom - axis.top) / (ymax - ymin)
    return [(int(round(a)), int(round(b))) for a, b in zip(px, py, strict=True)]


def _draw_curve(
    image: Image.Image,
    axis: Axis,
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    color: tuple[int, int, int] = OVERLAY_COLOR,
    width: int = 5,
    underlay_width: int = 9,
) -> bool:
    points = _pixels(axis, x, y)
    if len(points) < 2:
        warnings.warn(
            "overlay curve has fewer than two points inside its registered paper axis",
            stacklevel=2,
        )
        return False
    draw = ImageDraw.Draw(image)
    draw.line(points, fill="white", width=underlay_width, joint="curve")
    draw.line(points, fill=color, width=width, joint="curve")
    return True


def _label(image: Image.Image, xy: tuple[int, int], text: str, size: int = 22) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(size)
    bounds = draw.textbbox(xy, text, font=font, stroke_width=1)
    pad = 6
    background = (
        bounds[0] - pad,
        bounds[1] - pad,
        bounds[2] + pad,
        bounds[3] + pad,
    )
    draw.rounded_rectangle(background, radius=4, fill=(255, 255, 255, 220))
    draw.text(
        xy,
        text,
        font=font,
        fill=OVERLAY_COLOR + (255,),
        stroke_width=1,
        stroke_fill=(255, 255, 255, 255),
    )


def _crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(box)


def _profile_data(results: Path, statistics_ustar: float) -> dict[str, np.ndarray]:
    profile = np.genfromtxt(
        results / "normalized_profiles.csv", delimiter=",", names=True
    )
    dimensional = np.genfromtxt(results / "profiles.csv", delimiter=",", names=True)
    height = np.asarray(profile["z_f_over_ustar"])
    z = np.asarray(dimensional["z_m"])
    du_dz = np.gradient(dimensional["u_m_s"], z)
    dv_dz = np.gradient(dimensional["v_m_s"], z)
    phi_m = 0.4 * z * np.hypot(du_dz, dv_dz) / statistics_ustar
    # The first cell-centred one-sided difference does not see the wall.  The imposed
    # neutral log wall gives kappa*z*|dU/dz|/u*=1 exactly at that boundary point.
    phi_m[0] = 1.0
    names = set(profile.dtype.names or ())
    has_lasd = "total_u_variance_over_ustar2" in names
    data = {
        "height": height,
        "phi_m": phi_m,
        "phi_c": np.asarray(profile["phi_c"]) if "phi_c" in names else None,
        "u_variance": np.asarray(
            profile[
                "total_u_variance_over_ustar2"
                if has_lasd
                else "resolved_u_variance_over_ustar2"
            ]
        ),
        "v_variance": np.asarray(
            profile[
                "total_v_variance_over_ustar2"
                if has_lasd
                else "resolved_v_variance_over_ustar2"
            ]
        ),
        "w_variance": np.asarray(
            profile[
                "total_w_variance_over_ustar2"
                if has_lasd
                else "resolved_w_variance_over_ustar2"
            ]
        ),
        "uw": np.asarray(profile["resolved_uw_over_ustar2"]),
        "vw": np.asarray(profile["resolved_vw_over_ustar2"]),
        "total_uw": np.asarray(profile["total_uw_over_ustar2"]),
        "total_vw": np.asarray(profile["total_vw_over_ustar2"]),
    }
    optional = (
        "sgs_scalar_variance_over_cstar2",
        "total_scalar_variance_over_cstar2",
        "sgs_wc_over_ustar_cstar",
        "total_wc_over_ustar_cstar",
        "momentum_diffusivity_m2_s",
        "scalar_diffusivity_m2_s",
    )
    for name in optional:
        data[name] = np.asarray(profile[name]) if name in names else None
    return data


def _history_data(results: Path, statistics_ustar: float) -> dict[str, np.ndarray]:
    history = np.genfromtxt(results / "history.csv", delimiter=",", names=True)
    names = set(history.dtype.names or ())
    tke_name = (
        "integrated_total_tke_m3_s2"
        if "integrated_total_tke_m3_s2" in names
        else "integrated_resolved_tke_m3_s2"
    )
    return {
        "tf": np.asarray(history["time_seconds"]) * 1.0e-4,
        "total_tke": 1.0e-4 * np.asarray(history[tke_name]) / statistics_ustar**3,
    }


def _spectra_data(results: Path) -> dict[str, np.ndarray] | None:
    path = results / "spectra.csv"
    if not path.exists():
        return None
    spectra = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(spectra[name]) for name in spectra.dtype.names or ()}


FIGURE13_COLORS = {
    "production": (215, 35, 45),
    "subgrid": (20, 105, 190),
    "transport": (25, 145, 75),
    "pressure": (130, 55, 170),
    "coriolis": (225, 125, 20),
    "tendency": (0, 145, 160),
}


def _budget_data(results: Path) -> dict[str, np.ndarray] | None:
    path = results / "fig13_budget_profiles.csv"
    if not path.exists():
        return None
    budget = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(budget[name]) for name in budget.dtype.names or ()}


def _draw_budget_legend(image: Image.Image) -> None:
    """Place one compact shared legend in the whitespace above Fig. 13."""
    draw = ImageDraw.Draw(image)
    font = _font(13)
    labels = (
        ("production", "Prod."),
        ("subgrid", "SGS"),
        ("transport", "Trans."),
        ("pressure", "Press."),
        ("coriolis", "Cor."),
        ("tendency", "Tend."),
    )
    x = 145
    y = 111
    for name, label in labels:
        color = FIGURE13_COLORS[name]
        draw.line((x, y + 7, x + 25, y + 7), fill="white", width=5)
        draw.line((x, y + 7, x + 25, y + 7), fill=color, width=3)
        draw.text((x + 30, y), label, font=font, fill=color)
        x += 120


def _make_montage(images: list[tuple[str, Image.Image]]) -> Image.Image:
    target_width = 920
    title_font = _font(28)
    title_height = 52
    gap = 28
    scaled: list[tuple[str, Image.Image]] = []
    for title, image in images:
        scale = target_width / image.width
        resized = image.resize(
            (target_width, round(image.height * scale)), Image.Resampling.LANCZOS
        )
        scaled.append((title, resized))
    height = gap + sum(title_height + image.height + gap for _, image in scaled)
    montage = Image.new("RGB", (target_width + 2 * gap, height), "white")
    draw = ImageDraw.Draw(montage)
    y = gap
    for title, image in scaled:
        draw.text((gap, y), title, font=title_font, fill="black")
        y += title_height
        montage.paste(image, (gap, y))
        y += image.height + gap
    return montage


def _make_all_figure_montage(
    images: list[tuple[FigureSpec, Image.Image]],
    *,
    is_lasd: bool,
    active_comparisons: dict[int, str],
) -> Image.Image:
    """Lay all numbered figures out in paper order on one readable sheet."""

    columns = 3
    tile_width = 1000
    max_image_height = 1220
    outer_gap = 34
    column_gap = 28
    row_gap = 34
    title_height = 76
    header_height = 190
    title_font = _font(33)
    status_font = _font(24)
    header_font = _font(46)
    legend_font = _font(27)

    prepared: list[tuple[FigureSpec, Image.Image]] = []
    for spec, image in images:
        scale = min(tile_width / image.width, max_image_height / image.height)
        resized = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
        prepared.append((spec, resized))

    rows = [
        prepared[index : index + columns] for index in range(0, len(prepared), columns)
    ]
    row_heights = [title_height + max(image.height for _, image in row) for row in rows]
    width = 2 * outer_gap + columns * tile_width + (columns - 1) * column_gap
    height = (
        outer_gap
        + header_height
        + sum(row_heights)
        + (len(rows) - 1) * row_gap
        + outer_gap
    )
    montage = Image.new("RGB", (width, height), (242, 244, 247))
    draw = ImageDraw.Draw(montage)
    draw.text(
        (outer_gap, outer_gap),
        (
            "Andrén et al. (1994): paper figures + WIRE-LES "
            + ("LASD" if is_lasd else "static Smagorinsky")
        ),
        font=header_font,
        fill=(20, 25, 32),
    )
    legend_y = outer_gap + 72
    draw.line(
        (outer_gap, legend_y + 16, outer_gap + 90, legend_y + 16),
        fill="white",
        width=10,
    )
    draw.line(
        (outer_gap, legend_y + 16, outer_gap + 90, legend_y + 16),
        fill=OVERLAY_COLOR,
        width=6,
    )
    draw.text(
        (outer_gap + 112, legend_y),
        (
            "WIRE-LES LASD: red = total/resolved; blue = diagnostic SGS contribution"
            if is_lasd
            else "WIRE-LES static Smagorinsky: red = available resolved/total diagnostic"
        ),
        font=legend_font,
        fill=(40, 46, 54),
    )
    draw.text(
        (outer_gap, legend_y + 52),
        "paper only = no directly comparable diagnostic in the current dry-flow run",
        font=legend_font,
        fill=PAPER_ONLY_COLOR,
    )

    y = outer_gap + header_height
    for row, row_height in zip(rows, row_heights, strict=True):
        for column, (spec, image) in enumerate(row):
            x = outer_gap + column * (tile_width + column_gap)
            draw.rounded_rectangle(
                (x, y, x + tile_width, y + row_height),
                radius=10,
                fill="white",
                outline=(207, 212, 220),
                width=2,
            )
            draw.text(
                (x + 18, y + 12),
                f"Figure {spec.number}",
                font=title_font,
                fill=(20, 25, 32),
            )
            status = active_comparisons.get(spec.number, "paper only")
            status_color = (
                OVERLAY_COLOR
                if spec.number in active_comparisons
                else PAPER_ONLY_COLOR
            )
            status_width = draw.textbbox((0, 0), status, font=status_font)[2]
            draw.text(
                (x + tile_width - status_width - 18, y + 21),
                status,
                font=status_font,
                fill=status_color,
            )
            image_x = x + (tile_width - image.width) // 2
            montage.paste(image, (image_x, y + title_height))
        y += row_height + row_gap
    return montage


def main() -> None:
    args = _parse_args()
    output = args.output or args.results / "paper_overlays"
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.results / "summary.json").read_text())
    statistics_ustar = summary["comparison"]["statistics_ustar_m_s"]
    history = _history_data(args.results, statistics_ustar)
    profile = _profile_data(args.results, statistics_ustar)
    spectra = _spectra_data(args.results)
    budget = _budget_data(args.results)
    is_lasd = profile["total_scalar_variance_over_cstar2"] is not None
    active_comparisons = {
        2: "WIRE-LES total TKE" if is_lasd else "WIRE-LES resolved TKE",
        4: "WIRE-LES on panels (a,b)" if is_lasd else "WIRE-LES on panel (a)",
        5: "WIRE-LES total variances" if is_lasd else "WIRE-LES resolved variances",
        6: "WIRE-LES total flux",
    }

    pages = {spec.page: _render_page(args.paper_pdf, spec.page) for spec in FIGURES}

    figure2 = pages[8]
    _draw_curve(
        figure2,
        Axis(316, 128, 724, 541, 0.0, 14.0, 0.0, 1.5),
        history["tf"],
        history["total_tke"],
    )
    _label(
        figure2,
        (510, 505),
        "WIRE-LES LASD total" if is_lasd else "WIRE-LES resolved",
    )
    figure2 = _crop(figure2, (100, 90, 900, 675))

    figure4 = pages[10]
    _draw_curve(
        figure4,
        Axis(335, 136, 744, 549, 0.0, 2.0, 0.0, 0.1),
        profile["phi_m"],
        profile["height"],
    )
    if profile["phi_c"] is not None:
        _draw_curve(
            figure4,
            Axis(335, 806, 744, 1219, 0.0, 2.0, 0.0, 0.1),
            profile["phi_c"],
            profile["height"],
        )
    _label(
        figure4,
        (555, 500),
        "WIRE-LES LASD" if is_lasd else "WIRE-LES static Smag.",
    )
    figure4 = _crop(figure4, (250, 90, 820, 1360))

    figure5 = pages[11]
    axes5 = (
        Axis(186, 132, 478, 428, 0.0, 8.0, 0.0, 0.35),
        Axis(563, 132, 859, 428, 0.0, 4.0, 0.0, 0.35),
        Axis(186, 514, 478, 811, 0.0, 3.0, 0.0, 0.35),
    )
    for axis, quantity in zip(
        axes5,
        (profile["u_variance"], profile["v_variance"], profile["w_variance"]),
        strict=True,
    ):
        _draw_curve(figure5, axis, quantity, profile["height"])
    _label(
        figure5,
        (560, 660),
        "WIRE-LES total (diagnostic SGS)" if is_lasd else "WIRE-LES resolved",
    )
    figure5 = _crop(figure5, (100, 90, 900, 900))

    figure6 = pages[12]
    _draw_curve(
        figure6,
        Axis(310, 137, 718, 550, -1.0, 0.2, 0.0, 0.35),
        profile["total_uw"],
        profile["height"],
    )
    _draw_curve(
        figure6,
        Axis(310, 811, 718, 1223, -0.7, 0.3, 0.0, 0.35),
        profile["total_vw"],
        profile["height"],
    )
    _label(figure6, (390, 690), "WIRE-LES total")
    figure6 = _crop(figure6, (100, 90, 900, 1335))

    scalar_outputs = []
    if profile["total_scalar_variance_over_cstar2"] is not None:
        figure7 = pages[13]
        _draw_curve(
            figure7,
            FIGURE7_AXIS,
            profile["total_scalar_variance_over_cstar2"],
            profile["height"],
        )
        _draw_curve(
            figure7,
            FIGURE7_AXIS,
            profile["sgs_scalar_variance_over_cstar2"],
            profile["height"],
            color=SGS_COLOR,
        )
        _label(figure7, (480, 790), "red total; blue diag. SGS")
        figure7 = _crop(figure7, (250, 750, 795, 1370))
        scalar_outputs.append(
            ("Figure 7 - scalar variance", "fig07_wireles_overlay.png", figure7)
        )

        figure8 = pages[14]
        axis8 = Axis(330, 138, 740, 550, 0.0, 1.0, 0.0, 0.35)
        _draw_curve(
            figure8,
            axis8,
            profile["total_wc_over_ustar_cstar"],
            profile["height"],
        )
        _draw_curve(
            figure8,
            axis8,
            profile["sgs_wc_over_ustar_cstar"],
            profile["height"],
            color=SGS_COLOR,
        )
        _label(figure8, (485, 515), "red total; blue diag. SGS")
        figure8 = _crop(figure8, (245, 90, 800, 690))
        scalar_outputs.append(
            ("Figure 8 - scalar flux", "fig08_wireles_overlay.png", figure8)
        )

        figure14 = pages[20]
        _draw_curve(
            figure14,
            FIGURE14_AXES[0],
            profile["momentum_diffusivity_m2_s"],
            profile["height"],
        )
        _draw_curve(
            figure14,
            FIGURE14_AXES[1],
            profile["scalar_diffusivity_m2_s"],
            profile["height"],
        )
        _label(figure14, (500, 745), "WIRE-LES LASD")
        figure14 = _crop(figure14, (240, 90, 810, 1350))
        scalar_outputs.append(
            ("Figure 14 - SGS diffusivities", "fig14_wireles_overlay.png", figure14)
        )
        active_comparisons.update(
            {
                7: "WIRE-LES scalar variance",
                8: "WIRE-LES scalar flux",
                14: "WIRE-LES LASD diffusivities",
            }
        )

    if budget is not None:
        figure13 = pages[19]
        for axis in FIGURE13_AXES:
            for name, color in FIGURE13_COLORS.items():
                _draw_curve(
                    figure13,
                    axis,
                    budget[name],
                    budget["height"],
                    color=color,
                    width=3,
                    underlay_width=5,
                )
        _draw_budget_legend(figure13)
        figure13 = _crop(figure13, (115, 90, 905, 985))
        scalar_outputs.append(
            (
                "Figure 13 - resolved scalar-flux budget",
                "fig13_wireles_overlay.png",
                figure13,
            )
        )
        active_comparisons[13] = "WIRE-LES complete scalar-flux budget"

    if spectra is not None:
        figure15 = pages[21]
        for axis, name in zip(
            FIGURE15_AXES,
            (
                "kEu_over_ustar2",
                "kEv_over_ustar2",
                "kEw_over_ustar2",
                "kEc_over_cstar2",
            ),
            strict=True,
        ):
            _draw_curve(
                figure15,
                axis,
                spectra["k_ustar_over_f"],
                spectra[name],
            )
        _label(figure15, (640, 470), "WIRE-LES LASD")
        figure15 = _crop(figure15, (105, 450, 905, 1370))
        scalar_outputs.append(
            ("Figure 15 - spectra", "fig15_wireles_overlay.png", figure15)
        )
        active_comparisons[15] = "WIRE-LES spectra"

    outputs = (
        ("Figure 2 - integrated TKE", "fig02_wireles_overlay.png", figure2),
        (
            "Figure 4 - mean velocity/scalar gradients",
            "fig04_wireles_overlay.png",
            figure4,
        ),
        ("Figure 5 - velocity variances", "fig05_wireles_overlay.png", figure5),
        ("Figure 6 - momentum fluxes", "fig06_wireles_overlay.png", figure6),
    ) + tuple(scalar_outputs)
    for _, filename, image in outputs:
        image.save(output / filename, dpi=(180, 180))
    montage = _make_montage([(title, image) for title, _, image in outputs])
    montage.save(output / "andren1994_wireles_paper_overlays.png", dpi=(180, 180))

    all_figures = [(spec, _crop(pages[spec.page], spec.crop)) for spec in FIGURES]
    all_montage = _make_all_figure_montage(
        all_figures,
        is_lasd=is_lasd,
        active_comparisons=active_comparisons,
    )
    all_output = output / "andren1994_all_figures_wireles_overlay.png"
    all_montage.save(all_output, dpi=(180, 180), optimize=True)
    print(all_output)


if __name__ == "__main__":
    main()
