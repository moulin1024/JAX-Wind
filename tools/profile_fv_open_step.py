#!/usr/bin/env python3
"""Compare steady open FV main-step variants on an existing precursor."""

from __future__ import annotations

import argparse
import dataclasses
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--steps", type=int, default=300)
    arguments = parser.parse_args()

    import jax

    from applications.fv_abl.workflow import load_workflow, run_main

    workflow = load_workflow(arguments.config)
    variants = (
        ("full", workflow),
        ("without_turbine", dataclasses.replace(workflow, turbine=None)),
    )
    for name, variant in variants:
        started = time.perf_counter()
        result = run_main(variant, steps=arguments.steps)
        elapsed = time.perf_counter() - started
        jax.clear_caches()
        print(
            f"{name}: reported={result['elapsed_seconds']:.6f}s "
            f"total={elapsed:.6f}s "
            f"steps_per_second={arguments.steps / result['elapsed_seconds']:.3f} "
            f"steady={result['steady_steps_per_second']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
