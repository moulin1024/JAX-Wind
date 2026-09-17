"""Audit a uniformly translating isobaric contact; no fitted reference data."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.domain import UniformGrid
from jaxwind.numerics.poisson import build_pressure_poisson
from jaxwind.physics.moisture import MoistureConfig
from jaxwind.spray_low_mach import build_moist_gas_transport, moist_gas_from_primitive
from jaxwind.state import StaggeredVelocity


def audit():
    jax.config.update("jax_enable_x64", True)
    config = MoistureConfig()
    rows = []
    for kind in ("temperature", "composition"):
        for nx in (8, 16, 32, 64):
            grid = UniformGrid(nx, 4, 4, 1.0, 0.5, 0.5)
            shape = (4, 4, nx)
            left = jnp.broadcast_to(jnp.arange(nx) < nx // 2, shape)
            temperature = (
                jnp.where(left, 290.0, 350.0)
                if kind == "temperature"
                else jnp.full(shape, 310.0)
            )
            vapor = (
                jnp.where(left, 0.005, 0.04)
                if kind == "composition"
                else jnp.full(shape, 0.01)
            )
            gas = moist_gas_from_primitive(temperature, vapor, config)
            velocity = StaggeredVelocity(
                jnp.ones(shape), jnp.zeros(shape), jnp.zeros((5, 4, nx))
            )
            poisson = build_pressure_poisson(
                grid, backend="fft", periodic_x=True, periodic_y=True, dtype="float64"
            )
            ambient = (
                jnp.stack(gas)[..., 0] / (gas.dry_density + gas.vapor_density)[..., 0]
            )
            step = jax.jit(
                build_moist_gas_transport(poisson, config, ambient, tolerance=1e-11)
            )
            dt = 0.25 / nx
            result = step(gas, velocity, jax.tree.map(jnp.zeros_like, gas), dt)
            # Constant-speed donor-cell reference: a convex combination of the
            # original conserved states. Both contacts preserve the EOS exactly.
            q = np.asarray(jnp.stack(gas))
            reference = 0.75 * q + 0.25 * np.roll(q, 1, axis=-1)
            actual = np.asarray(jnp.stack(result.gas))
            scale = np.max(np.abs(q), axis=(1, 2, 3))
            rows.append(
                {
                    "kind": kind,
                    "nx": nx,
                    "cfl": 0.25,
                    "iteration_tolerance": 1e-11,
                    "accepted": bool(result.accepted),
                    "iterations": int(result.iterations),
                    "eos_error": float(result.eos_error),
                    "continuity_error": float(result.continuity_error),
                    "max_velocity_error_m_s": max(
                        float(jnp.max(jnp.abs(a - b)))
                        for a, b in zip(result.velocity, velocity)
                    ),
                    "pressure_span_pa": float(jnp.ptp(result.pressure)),
                    "relative_conserved_contact_error": np.max(
                        np.abs(actual - reference), axis=(1, 2, 3)
                    )
                    .__truediv__(scale)
                    .tolist(),
                    "relative_global_budget_error": (
                        np.abs(np.sum(actual - q, axis=(1, 2, 3)))
                        / np.sum(np.abs(q), axis=(1, 2, 3))
                    ).tolist(),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = audit()
    passed = all(
        r["accepted"]
        and r["max_velocity_error_m_s"] < 1e-7
        and r["pressure_span_pa"] < 1e-7
        and max(r["relative_conserved_contact_error"]) < 1e-7
        for r in rows
    )
    report = {
        "passed": passed,
        "scope": "One-step periodic contact preservation, numerical verification only",
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
