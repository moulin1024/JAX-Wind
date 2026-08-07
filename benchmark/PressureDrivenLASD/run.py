#!/usr/bin/env python3
"""Run the pressure-driven LASD case and plot its neutral log law."""

from __future__ import annotations

from benchmark.PressureDrivenMGM.run import ROOT, run_benchmark


DEFAULT_OUTPUT = ROOT / "outputs" / "pressure_driven_lasd_gpu"


def main(argv: list[str] | None = None) -> int:
    return run_benchmark(
        argv,
        sgs_model="lasd",
        program="python -m benchmark.PressureDrivenLASD",
        description=__doc__ or "Pressure-driven LASD benchmark",
        default_output=DEFAULT_OUTPUT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
