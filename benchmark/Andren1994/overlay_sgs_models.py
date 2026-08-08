#!/usr/bin/env python3
"""Overlay MGM, LASD, and AMD results on Andrén et al. (1994) figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.Andren1994 import run as andren
from benchmark.Andren1994.overlay_paper_figures import (
    FIGURE15_AXES,
    Axis,
    _crop,
    _font,
    _pixels,
    _render_page,
)


MODELS = (
    ("MGM", (211, 47, 47)),
    ("LASD", (25, 103, 210)),
    ("AMD", (22, 145, 85)),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-pdf", type=Path, required=True)
    parser.add_argument("--mgm", type=Path, required=True)
    parser.add_argument("--lasd", type=Path, required=True)
    parser.add_argument("--amd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_result(path: Path) -> dict:
    summary = json.loads((path / "summary.json").read_text())
    profile = np.genfromtxt(path / "normalized_profiles.csv", delimiter=",", names=True)
    dimensional = np.genfromtxt(path / "profiles.csv", delimiter=",", names=True)
    history = np.genfromtxt(path / "history.csv", delimiter=",", names=True)
    spectra = np.genfromtxt(path / "spectra.csv", delimiter=",", names=True)
    ustar = float(summary["comparison"]["statistics_ustar_m_s"])
    z = np.asarray(dimensional["z_m"])
    phi_m = (
        0.4
        * z
        * np.hypot(
            np.gradient(dimensional["u_m_s"], z),
            np.gradient(dimensional["v_m_s"], z),
        )
        / ustar
    )
    phi_m[0] = 1.0
    return {
        "summary": summary,
        "height": np.asarray(profile["z_f_over_ustar"]),
        "phi_m": phi_m,
        "u_variance": np.asarray(profile["resolved_u_variance_over_ustar2"]),
        "v_variance": np.asarray(profile["resolved_v_variance_over_ustar2"]),
        "w_variance": np.asarray(profile["resolved_w_variance_over_ustar2"]),
        "uw": np.asarray(profile["total_uw_over_ustar2"]),
        "vw": np.asarray(profile["total_vw_over_ustar2"]),
        "tf": np.asarray(history["time_seconds"]) * andren.F_CORIOLIS,
        "total_tke": (
            andren.F_CORIOLIS
            * np.asarray(history["integrated_total_tke_m3_s2"])
            / ustar**3
        ),
        "spectra_x": np.asarray(spectra["k_ustar_over_f"]),
        "spectra_u": np.asarray(spectra["kEu_over_ustar2"]),
        "spectra_v": np.asarray(spectra["kEv_over_ustar2"]),
        "spectra_w": np.asarray(spectra["kEw_over_ustar2"]),
    }


def _draw_curves(
    image: Image.Image,
    axis: Axis,
    curves: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]],
) -> None:
    draw = ImageDraw.Draw(image)
    pixel_curves = [(_pixels(axis, x, y), color) for x, y, color in curves]
    for points, _ in pixel_curves:
        if len(points) >= 2:
            draw.line(points, fill="white", width=8, joint="curve")
    for points, color in pixel_curves:
        if len(points) >= 2:
            draw.line(points, fill=color, width=4, joint="curve")


def _montage(
    images: list[tuple[str, Image.Image]], results: dict[str, dict]
) -> Image.Image:
    target_width = 980
    gap = 28
    header_height = 190
    title_height = 48
    scaled = []
    for title, image in images:
        scale = target_width / image.width
        scaled.append(
            (
                title,
                image.resize(
                    (target_width, round(image.height * scale)),
                    Image.Resampling.LANCZOS,
                ),
            )
        )
    height = header_height + sum(
        title_height + image.height + gap for _, image in scaled
    )
    canvas = Image.new("RGB", (target_width + 2 * gap, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 20),
        "Andrén et al. (1994) + JAX-Wind SGS comparison",
        font=_font(35),
        fill=(20, 25, 32),
    )
    x = gap
    for name, color in MODELS:
        ustar = results[name]["summary"]["comparison"]["statistics_ustar_m_s"]
        draw.line((x, 86, x + 55, 86), fill=color, width=7)
        draw.text(
            (x + 68, 67), f"{name}  u*={ustar:.4f} m/s", font=_font(22), fill=color
        )
        x += 315
    draw.text(
        (gap, 119),
        "Same 40³ grid, initial state, dt, conservative form, and 3/2 padding.",
        font=_font(18),
        fill=(65, 70, 78),
    )
    draw.text(
        (gap, 145),
        "Fig. 2: resolved + diagnostic SGS TKE. Fig. 5: resolved variances. "
        "Fig. 6: total flux.",
        font=_font(18),
        fill=(65, 70, 78),
    )
    y = header_height
    for title, image in scaled:
        draw.text((gap, y), title, font=_font(25), fill=(20, 25, 32))
        y += title_height
        canvas.paste(image, (gap, y))
        y += image.height + gap
    return canvas


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {"MGM": args.mgm, "LASD": args.lasd, "AMD": args.amd}
    results = {name: _load_result(path) for name, path in paths.items()}
    for name, _ in MODELS:
        physics = results[name]["summary"]["physics"]
        if physics.get("momentum_advection") != "conservative":
            raise ValueError(f"{name} did not use conservative advection")
        if physics.get("nonlinear_padding_ratio") != 1.5:
            raise ValueError(f"{name} did not use the common 3/2 padding")
        if not results[name]["summary"]["case"].get("diagnostic_sgs_energy"):
            raise ValueError(f"{name} does not include diagnostic SGS energy")

    figure2 = _render_page(args.paper_pdf, 8)
    _draw_curves(
        figure2,
        Axis(316, 128, 724, 541, 0.0, 14.0, 0.0, 1.5),
        [
            (results[name]["tf"], results[name]["total_tke"], color)
            for name, color in MODELS
        ],
    )
    figure2 = _crop(figure2, (100, 90, 900, 675))

    figure4 = _render_page(args.paper_pdf, 10)
    _draw_curves(
        figure4,
        Axis(335, 136, 744, 549, 0.0, 2.0, 0.0, 0.1),
        [
            (results[name]["phi_m"], results[name]["height"], color)
            for name, color in MODELS
        ],
    )
    figure4 = _crop(figure4, (250, 90, 820, 720))

    figure5 = _render_page(args.paper_pdf, 11)
    axes5 = (
        Axis(186, 132, 478, 428, 0.0, 8.0, 0.0, 0.35),
        Axis(563, 132, 859, 428, 0.0, 4.0, 0.0, 0.35),
        Axis(186, 514, 478, 811, 0.0, 3.0, 0.0, 0.35),
    )
    for axis, quantity in zip(
        axes5, ("u_variance", "v_variance", "w_variance"), strict=True
    ):
        _draw_curves(
            figure5,
            axis,
            [
                (results[name][quantity], results[name]["height"], color)
                for name, color in MODELS
            ],
        )
    figure5 = _crop(figure5, (100, 90, 900, 900))

    figure6 = _render_page(args.paper_pdf, 12)
    for axis, quantity in (
        (Axis(310, 137, 718, 550, -1.0, 0.2, 0.0, 0.35), "uw"),
        (Axis(310, 811, 718, 1223, -0.7, 0.3, 0.0, 0.35), "vw"),
    ):
        _draw_curves(
            figure6,
            axis,
            [
                (results[name][quantity], results[name]["height"], color)
                for name, color in MODELS
            ],
        )
    figure6 = _crop(figure6, (100, 90, 900, 1335))

    figure15 = _render_page(args.paper_pdf, 21)
    for axis, quantity in zip(
        FIGURE15_AXES[:3], ("spectra_u", "spectra_v", "spectra_w"), strict=True
    ):
        _draw_curves(
            figure15,
            axis,
            [
                (results[name]["spectra_x"], results[name][quantity], color)
                for name, color in MODELS
            ],
        )
    figure15 = _crop(figure15, (105, 450, 905, 1370))

    figures = (
        (
            "Paper Fig. 2 — integrated resolved + diagnostic SGS TKE",
            "fig02_three_sgs.png",
            figure2,
        ),
        (
            "Paper Fig. 4a — dimensionless velocity gradient",
            "fig04a_three_sgs.png",
            figure4,
        ),
        ("Paper Fig. 5 — resolved velocity variances", "fig05_three_sgs.png", figure5),
        ("Paper Fig. 6 — total momentum flux", "fig06_three_sgs.png", figure6),
        ("Paper Fig. 15 — resolved velocity spectra", "fig15_three_sgs.png", figure15),
    )
    for _, filename, image in figures:
        image.save(args.output / filename, dpi=(180, 180))
    montage = _montage([(title, image) for title, _, image in figures], results)
    montage.save(args.output / "andren1994_three_sgs_paper_overlay.png", dpi=(180, 180))
    manifest = {
        "models": {name: str(paths[name]) for name, _ in MODELS},
        "paper_pdf": str(args.paper_pdf),
        "common_numerics": {
            "advection": "conservative",
            "nonlinear_padding_ratio": 1.5,
            "comparison_tke": "resolved plus diagnostic SGS for all three models",
            "comparison_variances": "resolved for all three models",
            "comparison_momentum_flux": "resolved plus modeled SGS for all three models",
        },
        "output": str(args.output / "andren1994_three_sgs_paper_overlay.png"),
    }
    (args.output / "overlay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
