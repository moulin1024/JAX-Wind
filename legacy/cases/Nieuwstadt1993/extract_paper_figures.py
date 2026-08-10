#!/usr/bin/env python3
"""Extract and register Nieuwstadt et al. (1993) Figs. 1--17.

The Springer PDF is a scanned document, so the overlay code needs both a crop
and a pixel registration for every plot.  Render PDF pages 8--20 at 200 dpi
with ``pdftoppm`` first; this script then maps the scanned plot rectangles onto
the coordinates used by :mod:`overlay_figures`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from overlay_figures import AXES


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ScanFigure:
    page: int
    left: float
    right: float
    top: float
    bottom: float
    crop_bottom: float


# Pixel coordinates in the official PDF rendered at 200 dpi.  The source is a
# scan, hence the one- or two-pixel differences between otherwise similar plots.
SCAN_FIGURES = {
    1: ScanFigure(8, 291, 759, 194, 659, 800),
    2: ScanFigure(9, 278, 751, 186, 660, 780),
    3: ScanFigure(10, 282, 753, 1059, 1522, 1740),
    4: ScanFigure(11, 293, 763, 891, 1356, 1480),
    5: ScanFigure(12, 287, 757, 571, 1034, 1190),
    6: ScanFigure(13, 293, 763, 388, 854, 950),
    7: ScanFigure(14, 299, 770, 193, 655, 815),
    8: ScanFigure(14, 290, 761, 886, 1349, 1500),
    9: ScanFigure(15, 281, 751, 207, 675, 800),
    10: ScanFigure(16, 281, 753, 186, 663, 850),
    11: ScanFigure(17, 278, 749, 198, 661, 735),
    12: ScanFigure(17, 292, 762, 848, 1304, 1460),
    13: ScanFigure(18, 279, 750, 206, 672, 840),
    14: ScanFigure(18, 295, 765, 909, 1371, 1540),
    15: ScanFigure(19, 286, 756, 204, 666, 770),
    16: ScanFigure(20, 285, 755, 201, 668, 820),
    17: ScanFigure(20, 286, 756, 905, 1368, 1480),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract registered paper figures from 200-dpi PNG pages."
    )
    parser.add_argument(
        "--page-dir",
        type=Path,
        default=ROOT / "tmp" / "nieuwstadt_pages",
        help="Directory containing page-08.png through page-20.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_DIR / "reference" / "figures",
    )
    parser.add_argument("--width", type=int, default=1250)
    return parser.parse_args()


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
    output_height = round((scan.crop_bottom - origin_y) * scale_y)

    # PIL's affine tuple maps output pixels back into the scanned page.  Using
    # separate x/y scale factors corrects the small anisotropy of the scan and
    # makes the paper axes coincide exactly with the diagnostic coordinates.
    return page.transform(
        (output_width, output_height),
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
            if not source.exists():
                raise SystemExit(
                    f"ERROR: missing {source}; render PDF pages 8--20 at 200 dpi first."
                )
            pages[scan.page] = Image.open(source).convert("RGB")
        output = args.output_dir / f"fig{figure}.png"
        registered_crop(pages[scan.page], scan, figure, args.width).save(output)
        print(f"[extract] wrote {output}")


if __name__ == "__main__":
    main()
