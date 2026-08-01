#!/usr/bin/env python3
"""Extract and register Nieuwstadt et al. (1993) Figs. 1--17.

The Springer PDF is a scanned document, so the overlay code needs both a crop
and a pixel registration for every plot.  The script renders the embedded page
scan at 200 dpi directly with pypdf when a pre-rendered page PNG is absent,
then maps the scanned plot rectangles onto the coordinates used by
:mod:`overlay_figures`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

from PIL import Image

from overlay_figures import AXES


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BENCHMARK_DIR / "reference" / "Nieuwstadt1993.pdf"


@dataclass(frozen=True)
class ScanFigure:
    page: int
    left: float
    right: float
    top: float
    bottom: float
    output_height: int


# Axis rectangles in this repository's PDF rendered at 200 dpi.  The source is
# a scan, hence the one- or two-pixel differences between otherwise similar
# plots.  Output heights retain enough margin for labels and captions while the
# affine registration places every axis rectangle exactly at AXES[figure].
SCAN_FIGURES = {
    1: ScanFigure(8, 410.0, 934.5, 343.0, 860.0, 817),
    2: ScanFigure(9, 390.5, 920.0, 322.0, 849.0, 816),
    3: ScanFigure(10, 402.5, 930.5, 1298.5, 1812.5, 1081),
    4: ScanFigure(11, 410.5, 937.5, 1121.0, 1637.5, 924),
    5: ScanFigure(12, 404.5, 930.5, 754.5, 1269.0, 844),
    6: ScanFigure(13, 404.5, 931.5, 547.5, 1065.5, 764),
    7: ScanFigure(14, 418.0, 945.0, 334.0, 849.0, 984),
    8: ScanFigure(14, 403.0, 932.0, 1106.0, 1619.0, 838),
    9: ScanFigure(15, 388.0, 916.0, 350.0, 870.0, 807),
    10: ScanFigure(16, 397.5, 926.5, 323.5, 854.0, 890),
    11: ScanFigure(17, 390.0, 917.0, 334.5, 849.5, 732),
    12: ScanFigure(17, 404.5, 931.5, 1057.0, 1564.0, 818),
    13: ScanFigure(18, 394.5, 922.5, 352.0, 870.0, 851),
    14: ScanFigure(18, 405.0, 932.0, 1133.5, 1647.0, 863),
    15: ScanFigure(19, 402.0, 928.5, 346.0, 860.0, 763),
    16: ScanFigure(20, 407.0, 933.5, 343.0, 861.5, 829),
    17: ScanFigure(20, 402.5, 929.0, 1125.0, 1640.0, 798),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract registered paper figures from 200-dpi PNG pages."
    )
    parser.add_argument(
        "--page-dir",
        type=Path,
        default=ROOT / "tmp" / "nieuwstadt_pages",
        help="Optional directory containing page-08.png through page-20.png.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="Scanned paper PDF used when a pre-rendered page PNG is absent.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_DIR / "reference" / "figures",
    )
    parser.add_argument("--width", type=int, default=1250)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def render_scanned_page(pdf: Path, page_number: int, dpi: int) -> Image.Image:
    """Reproduce the PDF page raster without requiring a Poppler install."""
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - depends on local runtime
        raise SystemExit(
            "ERROR: pypdf is required when pre-rendered page PNGs are absent."
        ) from error

    reader = PdfReader(pdf)
    if not 1 <= page_number <= len(reader.pages):
        raise SystemExit(
            f"ERROR: PDF page {page_number} is outside 1..{len(reader.pages)}."
        )
    page = reader.pages[page_number - 1]
    images = list(page.images)
    if len(images) != 1:
        raise SystemExit(
            f"ERROR: expected one embedded scan on page {page_number}, "
            f"found {len(images)}."
        )

    content = page.get_contents().get_data()
    match = re.search(
        rb"([-+0-9.]+)\s+0\s+0\s+([-+0-9.]+)\s+"
        rb"([-+0-9.]+)\s+([-+0-9.]+)\s+cm\s*/Im0\s+Do",
        content,
    )
    if match is None:
        raise SystemExit(
            f"ERROR: could not locate the scan placement matrix on page {page_number}."
        )
    width_pt, height_pt, left_pt, bottom_pt = (
        float(value) for value in match.groups()
    )
    media_width = float(page.mediabox.width)
    media_height = float(page.mediabox.height)
    scale = dpi / 72.0
    canvas = Image.new(
        "L",
        (round(media_width * scale), round(media_height * scale)),
        255,
    )
    scan = images[0].image.convert("L")
    scan = scan.resize(
        (round(width_pt * scale), round(height_pt * scale)),
        Image.Resampling.LANCZOS,
    )
    left = round(left_pt * scale)
    top = round((media_height - bottom_pt - height_pt) * scale)
    canvas.paste(scan, (left, top))
    return canvas.convert("RGB")


def registered_crop(
    page: Image.Image,
    scan: ScanFigure,
    figure: int,
    output_width: int,
) -> Image.Image:
    target = AXES[figure]
    scale_x = (target.right - target.left) / (scan.right - scan.left)
    scale_y = (target.bottom - target.top) / (scan.bottom - scan.top)
    origin_x = scan.left - target.left / scale_x
    origin_y = scan.top - target.top / scale_y

    # PIL's affine tuple maps output pixels back into the scanned page.  Using
    # separate x/y scale factors corrects the small anisotropy of the scan and
    # makes the paper axes coincide exactly with the diagnostic coordinates.
    return page.transform(
        (output_width, scan.output_height),
        Image.Transform.AFFINE,
        (1.0 / scale_x, 0.0, origin_x, 0.0, 1.0 / scale_y, origin_y),
        resample=Image.Resampling.BICUBIC,
        fillcolor="white",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages: dict[int, Image.Image] = {}
    for figure, scan in SCAN_FIGURES.items():
        if scan.page not in pages:
            source = args.page_dir / f"page-{scan.page:02d}.png"
            if source.exists():
                pages[scan.page] = Image.open(source).convert("RGB")
            else:
                pages[scan.page] = render_scanned_page(
                    args.pdf, scan.page, args.dpi
                )
        output = args.output_dir / f"fig{figure}.png"
        registered_crop(pages[scan.page], scan, figure, args.width).save(output)
        print(f"[extract] wrote {output}")


if __name__ == "__main__":
    main()
