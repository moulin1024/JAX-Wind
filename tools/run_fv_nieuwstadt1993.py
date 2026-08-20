"""Compatibility entry point for the TOML-driven FV ABL core."""

from __future__ import annotations

from pathlib import Path

from applications.fv_abl.__main__ import parser, run
from applications.fv_abl.diagnostics import (
    PROFILE_NAMES,
    ConvectiveAccumulator,
    profile_columns as _profile_columns,
    write_columns as _write_columns,
    write_radial_spectra as _write_radial_spectra,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "cases" / "Nieuwstadt1993" / "config.toml"
DEFAULT_OUTPUT = ROOT / "outputs" / "nieuwstadt1993_fv_fft_40x40x48"


def parse_arguments(argv: list[str] | None = None):
    return parser(DEFAULT_CONFIG).parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_arguments(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
