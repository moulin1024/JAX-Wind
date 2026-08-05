"""Device-resident PCG construction for weighted finite-volume systems."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


Array = jax.Array


def build_device_pcg_solver(
    *,
    apply: Callable[[Array], Array],
    precondition: Callable[[Array], Array],
    volume: Array,
    project: Callable[[Array], Array],
    max_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> Callable[[Array, Array], Array]:
    """Build a JIT PCG solve using the FV volume-weighted inner product."""
    square_root_volume = jnp.sqrt(volume)

    def to_euclidean(field: Array) -> Array:
        return square_root_volume * field

    def to_physical(field: Array) -> Array:
        # The prepared RHS, initial guess, operator range, and V-cycle output
        # all lie in the compatible subspace.  PCG linear combinations keep
        # that invariant, so projecting at every operator conversion only
        # repeats a global reduction.  The solve boundaries still fix gauge.
        return field / square_root_volume

    def transformed_apply(field: Array) -> Array:
        return to_euclidean(apply(to_physical(field)))

    def transformed_preconditioner(field: Array) -> Array:
        return to_euclidean(precondition(to_physical(field)))

    def solve(effective_rhs: Array, initial: Array) -> Array:
        transformed_solution, _ = cg(
            transformed_apply,
            to_euclidean(effective_rhs),
            x0=to_euclidean(project(initial)),
            tol=relative_tolerance,
            atol=absolute_tolerance,
            maxiter=max_iterations,
            M=transformed_preconditioner,
        )
        return project(to_physical(transformed_solution))

    return jax.jit(solve)


__all__ = ["build_device_pcg_solver"]
