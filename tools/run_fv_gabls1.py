"""Compatibility entry point for the TOML-driven FV ABL core."""

from __future__ import annotations

from pathlib import Path

from applications.fv_abl.__main__ import parser, run


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "cases" / "GABLS1" / "config.toml"
DEFAULT_OUTPUT = ROOT / "outputs" / "gabls1_fv_fft_32x32x32"


def parse_arguments(argv: list[str] | None = None):
    return parser(DEFAULT_CONFIG).parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_arguments(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
