"""Polynomial utilities shared by the original scale-dependent models."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def largest_positive_real_polynomial_root(coefficients: jax.Array) -> jax.Array:
    """Return the largest positive real root of polynomials up to degree five.

    ``coefficients[..., i]`` multiplies ``beta**i``.  The nominal fifth-order
    LASD equation can become a valid lower-order equation when a plane
    contraction vanishes, so its numerical degree is selected independently
    for every horizontal plane.
    """
    scale = jnp.max(jnp.abs(coefficients), axis=-1)
    tolerance = (
        32.0
        * jnp.asarray(jnp.finfo(coefficients.dtype).eps, dtype=coefficients.dtype)
        * jnp.maximum(scale, jnp.asarray(1.0e-30, dtype=coefficients.dtype))
    )
    powers = jnp.arange(6, dtype=jnp.int32)
    degree = jnp.max(
        jnp.where(jnp.abs(coefficients) > tolerance[..., None], powers, 0),
        axis=-1,
    )

    def no_root(_: jax.Array) -> jax.Array:
        return jnp.asarray(jnp.nan, dtype=coefficients.dtype)

    def solve_degree(n: int):
        def solve(c: jax.Array) -> jax.Array:
            monic = c[:n] / c[n]
            companion = jnp.zeros((n, n), dtype=c.dtype)
            if n > 1:
                companion = companion.at[
                    jnp.arange(1, n), jnp.arange(n - 1)
                ].set(1.0)
            companion = companion.at[:, n - 1].set(-monic)
            roots = jnp.linalg.eigvals(companion)
            real = jnp.real(roots)
            nearly_real = jnp.abs(jnp.imag(roots)) <= 1.0e-5 * (
                1.0 + jnp.abs(real)
            )
            positive = nearly_real & (real > 0.0)
            root = jnp.max(jnp.where(positive, real, -jnp.inf))
            return jnp.where(jnp.isfinite(root), root, jnp.nan).astype(c.dtype)

        return solve

    branches = (no_root,) + tuple(solve_degree(n) for n in range(1, 6))
    flat_coefficients = coefficients.reshape((-1, 6))
    flat_degree = degree.reshape((-1,))
    roots = jax.vmap(lambda d, c: jax.lax.switch(d, branches, c))(
        flat_degree, flat_coefficients
    )
    return roots.reshape(coefficients.shape[:-1])


def porte_agel_polynomial(
    p: jax.Array,
    q: jax.Array,
    r: jax.Array,
    s: jax.Array,
    t: jax.Array,
    p2: jax.Array,
    q2: jax.Array,
    r2: jax.Array,
    s2: jax.Array,
    t2: jax.Array,
) -> jax.Array:
    """Return coefficients of Porté-Agel's fifth-order beta equation."""
    return jnp.stack(
        (
            p * r2 - p2 * r,
            -4.0 * q * r2 + 8.0 * p2 * t,
            -32.0 * p * t2 - 16.0 * p2 * s + 16.0 * q2 * r,
            128.0 * q * t2 - 128.0 * q2 * t,
            256.0 * p * s2 + 256.0 * q2 * s,
            -1024.0 * q * s2,
        ),
        axis=-1,
    )
