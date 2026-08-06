"""Small shared linear-algebra kernels specialized for JAX-Wind grids."""

from __future__ import annotations

import jax
import jax.numpy as jnp


Array = jax.Array


def pcr_tridiagonal_solve(
    lower: Array,
    diagonal: Array,
    upper: Array,
    rhs: Array,
) -> Array:
    """Solve fixed-length batched tridiagonal systems with parallel PCR.

    The system dimension is the last diagonal axis and the penultimate RHS
    axis.  The Python loop is evaluated while tracing, so XLA sees only
    ``ceil(log2(n))`` vector stages without a vendor custom call or a
    sequential scan.  Callers must provide diagonally dominant systems.
    """
    if lower.shape != diagonal.shape or upper.shape != diagonal.shape:
        raise ValueError("PCR diagonals must have identical shapes")
    if rhs.shape[:-1] != diagonal.shape:
        raise ValueError("PCR right-hand side shape does not match diagonals")

    system_size = diagonal.shape[-1]
    reduced_lower = lower
    reduced_diagonal = diagonal
    reduced_upper = upper
    reduced_rhs = rhs
    stride = 1
    while stride < system_size:
        left_zeros = jnp.zeros_like(reduced_lower[..., :stride])
        right_zeros = jnp.zeros_like(reduced_lower[..., :stride])
        left_diagonal = jnp.concatenate(
            (
                jnp.ones_like(reduced_diagonal[..., :stride]),
                reduced_diagonal[..., :-stride],
            ),
            axis=-1,
        )
        right_diagonal = jnp.concatenate(
            (
                reduced_diagonal[..., stride:],
                jnp.ones_like(reduced_diagonal[..., :stride]),
            ),
            axis=-1,
        )
        left_lower = jnp.concatenate(
            (left_zeros, reduced_lower[..., :-stride]),
            axis=-1,
        )
        left_upper = jnp.concatenate(
            (left_zeros, reduced_upper[..., :-stride]),
            axis=-1,
        )
        right_lower = jnp.concatenate(
            (reduced_lower[..., stride:], right_zeros),
            axis=-1,
        )
        right_upper = jnp.concatenate(
            (reduced_upper[..., stride:], right_zeros),
            axis=-1,
        )
        rhs_zeros = jnp.zeros_like(reduced_rhs[..., :stride, :])
        left_rhs = jnp.concatenate(
            (rhs_zeros, reduced_rhs[..., :-stride, :]),
            axis=-2,
        )
        right_rhs = jnp.concatenate(
            (reduced_rhs[..., stride:, :], rhs_zeros),
            axis=-2,
        )

        eliminate_left = -reduced_lower / left_diagonal
        eliminate_right = -reduced_upper / right_diagonal
        reduced_diagonal = (
            reduced_diagonal
            + eliminate_left * left_upper
            + eliminate_right * right_lower
        )
        reduced_rhs = (
            reduced_rhs
            + eliminate_left[..., None] * left_rhs
            + eliminate_right[..., None] * right_rhs
        )
        reduced_lower = eliminate_left * left_lower
        reduced_upper = eliminate_right * right_upper
        stride *= 2

    return reduced_rhs / reduced_diagonal[..., None]


__all__ = ["pcr_tridiagonal_solve"]
