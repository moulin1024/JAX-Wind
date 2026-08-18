"""Retag a warm AB2 checkpoint for a smaller fixed timestep.

The prognostic fields and previous evaluated tendency are retained.  This is
appropriate for timestep refinement: the first refined AB2 step has one
history sample at the old spacing, after which the history is synchronized to
the new cadence.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from applications.pressure_driven_lasd.config import load_case
from applications.pressure_driven_lasd.evaluate import _configure_source_paths
from applications.pressure_driven_lasd.problem import build_pressure_driven_problem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_case", type=Path)
    parser.add_argument("target_case", type=Path)
    parser.add_argument("source_checkpoint", type=Path)
    parser.add_argument("target_checkpoint", type=Path)
    args = parser.parse_args()

    _configure_source_paths()
    import jax
    import jax.numpy as jnp
    from jaxwind.effects import (
        JaxRuntime,
        load_boussinesq_checkpoint,
        save_boussinesq_checkpoint,
    )

    source_case = load_case(args.source_case)
    target_case = load_case(args.target_case)
    if source_case.domain != target_case.domain:
        raise ValueError("checkpoint retiming requires identical domains")
    runtime = JaxRuntime.from_initialized_jax(jax)
    source = build_pressure_driven_problem(source_case, runtime=runtime)
    target = build_pressure_driven_problem(target_case, runtime=runtime)
    state = load_boussinesq_checkpoint(
        args.source_checkpoint,
        layout=source.solver.checkpoint_layout(jnp.asarray),
        config=source.integrator,
        scale_fingerprint=source.scales.fingerprint,
        closure_fingerprint=source.closure_fingerprint,
        physics_fingerprint=source.physics_fingerprint,
    )
    retimed = replace(
        state,
        integrator_fingerprint=target.integrator.fingerprint,
    )
    save_boussinesq_checkpoint(
        args.target_checkpoint,
        retimed,
        scale_fingerprint=target.scales.fingerprint,
        physics_fingerprint=target.physics_fingerprint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
