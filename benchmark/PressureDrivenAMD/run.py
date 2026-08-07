#!/usr/bin/env python3
"""Run the pressure-driven AMD case and plot its neutral log law."""

from __future__ import annotations

from benchmark.PressureDrivenMGM.run import ROOT, run_benchmark


DEFAULT_OUTPUT = ROOT / "outputs" / "pressure_driven_amd_gpu"


def main(argv: list[str] | None = None) -> int:
    return run_benchmark(
        argv,
        sgs_model="amd",
        program="python -m benchmark.PressureDrivenAMD",
        description=__doc__ or "Pressure-driven AMD benchmark",
        default_output=DEFAULT_OUTPUT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
