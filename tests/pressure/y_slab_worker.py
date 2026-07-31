from __future__ import annotations

import json

import numpy as np
import jax
import jax.numpy as jnp

from jaxwind.pressure import (
    BoundaryCondition,
    PCGConfig,
    RectilinearGrid,
    PoissonBoundaryConditions,
    YSlabConfig,
    YSlabMACProjector,
    YSlabMatrixFreePoissonSolver,
    gather_y_field,
    shard_y_field,
)


def main() -> None:
    devices = tuple(jax.devices())

    def faces(count: int, exponent: float) -> tuple[float, ...]:
        return tuple(np.linspace(0.0, 1.0, count + 1) ** exponent)

    grid = RectilinearGrid(faces(8, 1.0), faces(8, 1.0), faces(16, 2.0))
    periodic = BoundaryCondition("periodic")
    dirichlet = BoundaryCondition("dirichlet")
    boundaries = PoissonBoundaryConditions(
        periodic,
        periodic,
        periodic,
        periodic,
        dirichlet,
        dirichlet,
    )
    solver = YSlabMatrixFreePoissonSolver(
        grid,
        boundaries,
        devices=devices,
        dtype=jnp.float32,
        krylov=PCGConfig(
            max_iterations=60,
            relative_tolerance=2.0e-6,
        ),
        distribution=YSlabConfig(coarse_cells_per_device=2),
    )
    z_faces = jnp.asarray(grid.z_faces, dtype=jnp.float32)
    y_faces = jnp.asarray(grid.y_faces, dtype=jnp.float32)
    x_faces = jnp.asarray(grid.x_faces, dtype=jnp.float32)
    z = 0.5 * (z_faces[1:] + z_faces[:-1])
    y = 0.5 * (y_faces[1:] + y_faces[:-1])
    x = 0.5 * (x_faces[1:] + x_faces[:-1])
    exact = (
        jnp.sin(jnp.pi * z)[:, None, None]
        * jnp.cos(2.0 * jnp.pi * y)[None, :, None]
        * jnp.cos(2.0 * jnp.pi * x)[None, None, :]
    )
    rhs = solver.operator.apply(exact)
    sharded_exact = shard_y_field(exact, len(devices))
    sharded_rhs = shard_y_field(rhs, len(devices))
    other = jnp.cos(
        0.19 * jnp.arange(grid.cell_count, dtype=jnp.float32)
    ).reshape(grid.shape)
    sharded_other = solver.project_nullspace(
        shard_y_field(other, len(devices))
    )
    preconditioned_exact = solver.precondition(sharded_exact)
    preconditioned_other = solver.precondition(sharded_other)
    adjoint_left = solver.inner(sharded_exact, preconditioned_other)
    adjoint_right = solver.inner(preconditioned_exact, sharded_other)
    adjoint_scale = jnp.maximum(
        jnp.maximum(jnp.abs(adjoint_left), jnp.abs(adjoint_right)),
        1.0,
    )
    preconditioner_symmetry_error = (
        jnp.abs(adjoint_left - adjoint_right) / adjoint_scale
    )
    preconditioner_positive = solver.inner(
        sharded_other,
        preconditioned_other,
    )
    apply_error = jnp.max(
        jnp.abs(gather_y_field(solver.apply(sharded_exact)) - rhs)
    )
    result = solver.solve(sharded_rhs)
    solution = gather_y_field(result.solution)
    relative_error = solver.operator.norm(solution - exact) / (
        solver.operator.norm(exact)
    )
    mac_projector = YSlabMACProjector(solver)
    gradient_velocity = mac_projector.gradient(sharded_exact)
    stage = mac_projector.project(gradient_velocity, timestep=1.0)
    stage_divergence = gather_y_field(stage.divergence_after)
    stage_divergence_norm = solver.operator.norm(stage_divergence)
    print(
        json.dumps(
            {
                "devices": len(devices),
                "apply_error": float(apply_error),
                "preconditioner_symmetry_error": float(
                    preconditioner_symmetry_error
                ),
                "preconditioner_positive": float(preconditioner_positive),
                "converged": result.converged,
                "iterations": result.iterations,
                "relative_residual": result.relative_residual,
                "relative_error": float(relative_error),
                "replication_level": solver.replication_level,
                "replicated_shape": solver.replicated_shape,
                "stage_converged": stage.linear_result.converged,
                "stage_divergence_norm": float(stage_divergence_norm),
            }
        )
    )


if __name__ == "__main__":
    main()
