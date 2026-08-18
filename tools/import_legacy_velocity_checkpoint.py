#!/usr/bin/env python3
"""Seed a JAX checkpoint with a legacy Fortran velocity restart."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from applications.pressure_driven_lasd.config import load_case
from applications.pressure_driven_lasd.evaluate import _configure_source_paths
from applications.pressure_driven_lasd.problem import build_pressure_driven_problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_case", type=Path)
    parser.add_argument("target_case", type=Path)
    parser.add_argument("template_checkpoint", type=Path)
    parser.add_argument("legacy_directory", type=Path)
    parser.add_argument("output_checkpoint", type=Path)
    args = parser.parse_args()

    _configure_source_paths()
    import jax
    import jax.numpy as jnp
    from jaxwind.effects import JaxRuntime, load_boussinesq_checkpoint, save_boussinesq_checkpoint

    runtime = JaxRuntime.from_initialized_jax(jax)
    source_case = load_case(args.source_case)
    target_case = load_case(args.target_case)
    source = build_pressure_driven_problem(
        source_case,
        runtime=runtime,
    )
    target = build_pressure_driven_problem(target_case, runtime=runtime)
    state = load_boussinesq_checkpoint(
        args.template_checkpoint,
        layout=source.solver.checkpoint_layout(jnp.asarray),
        config=source.integrator,
        scale_fingerprint=source.scales.fingerprint,
        closure_fingerprint=source.closure_fingerprint,
        physics_fingerprint=source.physics_fingerprint,
    )
    shape = (source_case.domain.nx, source_case.domain.ny, source_case.domain.nz)

    def read(name: str) -> np.ndarray:
        values = np.fromfile(args.legacy_directory / f"{name}.bin", np.float32)
        if values.size != np.prod(shape):
            raise ValueError(f"legacy {name} field has {values.size} values")
        physical = values.reshape(shape, order="F").transpose(2, 1, 0)
        return source.scales.to_execution_velocity(physical)[None, ...]

    velocity = state.fields.velocity
    imported = replace(
        velocity,
        x=replace(velocity.x, payload=jnp.asarray(read("u"))),
        y=replace(velocity.y, payload=jnp.asarray(read("v"))),
        z=replace(
            velocity.z,
            owned=replace(velocity.z.owned, payload=jnp.asarray(read("w"))),
            lower_boundary=jnp.zeros_like(velocity.z.lower_boundary),
        ),
    )
    converted = replace(
        state,
        fields=replace(state.fields, velocity=imported),
        integrator_fingerprint=target.integrator.fingerprint,
    )
    save_boussinesq_checkpoint(
        args.output_checkpoint,
        converted,
        scale_fingerprint=target.scales.fingerprint,
        physics_fingerprint=target.physics_fingerprint,
    )
    print(args.output_checkpoint)


if __name__ == "__main__":
    main()
