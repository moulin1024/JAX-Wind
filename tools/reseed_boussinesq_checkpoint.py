#!/usr/bin/env python3
"""Use checkpoint fields as a cold-start seed for the current physics model."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from applications.pressure_driven_lasd.config import load_case
from applications.pressure_driven_lasd.evaluate import _configure_source_paths
from applications.pressure_driven_lasd.problem import build_pressure_driven_problem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("source_checkpoint", type=Path)
    parser.add_argument("output_checkpoint", type=Path)
    args = parser.parse_args()
    if args.output_checkpoint.exists():
        raise FileExistsError(args.output_checkpoint)

    _configure_source_paths()
    import jax
    import jax.numpy as jnp
    from jaxwind.domain import AcceptedClock
    from jaxwind.effects import (
        JaxRuntime,
        load_boussinesq_checkpoint,
        save_boussinesq_checkpoint,
    )
    from jaxwind.integrators import ColdStart

    runtime = JaxRuntime.from_initialized_jax(jax)
    case = load_case(args.case)
    problem = build_pressure_driven_problem(case, runtime=runtime)
    state = load_boussinesq_checkpoint(
        args.source_checkpoint,
        layout=problem.solver.checkpoint_layout(jnp.asarray),
        config=problem.integrator,
        scale_fingerprint=problem.scales.fingerprint,
        closure_fingerprint=problem.closure_fingerprint,
        # This operation intentionally imports fields from another numerical
        # operator. It is a new initial condition, not an exact restart.
        physics_fingerprint=None,
    )
    seeded = replace(
        state,
        clock=AcceptedClock(0.0, 0),
        history=ColdStart(),
        integrator_fingerprint=problem.integrator.fingerprint,
    )
    save_boussinesq_checkpoint(
        args.output_checkpoint,
        seeded,
        scale_fingerprint=problem.scales.fingerprint,
        physics_fingerprint=problem.physics_fingerprint,
    )
    if runtime.is_primary:
        print(args.output_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
