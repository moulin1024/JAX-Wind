"""Compatibility entry point for the TOML-driven FV ABL core."""

from __future__ import annotations

from pathlib import Path

from applications.fv_abl.__main__ import parser, run
from applications.fv_abl.diagnostics import (
    PROFILE_NAMES,
    ProfileAccumulator,
    initial_fields as _initial_fields,
    steps_to_next_sample as _steps_to_next_sample,
    write_profiles as _write_profiles,
    write_streamwise_spectra as _write_spectra,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "cases" / "Andren1994" / "config.toml"
DEFAULT_OUTPUT = ROOT / "outputs" / "andren1994_fv_fft_40x40x40"


def parse_arguments(argv: list[str] | None = None):
    return parser(DEFAULT_CONFIG).parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_arguments(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
