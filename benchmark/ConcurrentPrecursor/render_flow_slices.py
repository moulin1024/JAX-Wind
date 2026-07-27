#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "jax"))

from run_single import RUN_DEFAULTS, load_config_file, params_from_settings  # noqa: E402
from run_warmup_diagnostics import make_flow_slices_gif  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a three-plane flow GIF from saved distributed slice samples."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-to", type=Path)
    args = parser.parse_args()

    import jax.numpy as jnp

    settings = dict(RUN_DEFAULTS)
    settings.update(load_config_file(args.config))
    params = params_from_settings(settings, jnp)
    with np.load(args.input) as data:
        xy = np.asarray(data["u_xy"])
        xz = np.asarray(data["u_xz"])
        yz = np.asarray(data["u_yz"])
        elapsed = np.asarray(data["elapsed_seconds"])
        height = float(np.asarray(data["horizontal_height_m"]))
    if not (len(xy) == len(xz) == len(yz) == len(elapsed)):
        raise ValueError("Flow-slice arrays have inconsistent frame counts")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_flow_slices_gif(
        args.output, xy, xz, yz, elapsed, params, height
    )
    if args.copy_to is not None:
        args.copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, args.copy_to)
    print(f"rendered {len(elapsed)} frames: {args.output}")


if __name__ == "__main__":
    main()
