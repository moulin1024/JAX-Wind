"""Device-resident GMRES construction for weighted finite-volume systems."""

from __future__ import annotations

import math
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres


Array = jax.Array


def build_device_gmres_solver(
    *,
    apply: Callable[[Array], Array],
    precondition: Callable[[Array], Array],
    volume: Array,
    project: Callable[[Array], Array],
    restart: int,
    max_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    solve_method: Literal["incremental", "batched"],
) -> Callable[[Array, Array], Array]:
    """Build a JIT GMRES solve using the FV volume-weighted inner product."""
    square_root_volume = jnp.sqrt(volume)
    restart_cycles = math.ceil(max_iterations / restart)

    def to_euclidean(field: Array) -> Array:
        return square_root_volume * field

    def to_physical(field: Array) -> Array:
        return project(field / square_root_volume)

    def transformed_apply(field: Array) -> Array:
        return to_euclidean(apply(to_physical(field)))

    def transformed_preconditioner(field: Array) -> Array:
        return to_euclidean(precondition(to_physical(field)))

    def solve(effective_rhs: Array, initial: Array) -> Array:
        transformed_solution, _ = gmres(
            transformed_apply,
            to_euclidean(effective_rhs),
            x0=to_euclidean(project(initial)),
            tol=relative_tolerance,
            atol=absolute_tolerance,
            restart=restart,
            maxiter=restart_cycles,
            M=transformed_preconditioner,
            solve_method=solve_method,
        )
        return to_physical(transformed_solution)

    return jax.jit(solve)


__all__ = ["build_device_gmres_solver"]
