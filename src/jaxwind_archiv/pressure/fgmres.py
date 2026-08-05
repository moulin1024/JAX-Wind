"""Restarted flexible GMRES for JAX array-valued linear systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp


Array = jax.Array


@dataclass(frozen=True, slots=True)
class FGMRESConfig:
    """Restarted flexible-GMRES controls."""

    restart: int = 30
    max_iterations: int = 120
    relative_tolerance: float = 1.0e-8
    absolute_tolerance: float = 0.0
    reorthogonalize: bool = True
    jit_kernels: bool = True
    execution: Literal["python", "jax"] = "python"
    jax_solve_method: Literal["incremental", "batched"] = "incremental"

    def __post_init__(self) -> None:
        if self.restart <= 0 or self.max_iterations <= 0:
            raise ValueError("restart and max_iterations must be positive")
        if self.relative_tolerance < 0.0 or self.absolute_tolerance < 0.0:
            raise ValueError("solver tolerances must be nonnegative")
        if self.execution not in {"python", "jax"}:
            raise ValueError("FGMRES execution must be 'python' or 'jax'")
        if self.jax_solve_method not in {"incremental", "batched"}:
            raise ValueError(
                "JAX GMRES solve method must be 'incremental' or 'batched'"
            )


@dataclass(frozen=True, slots=True)
class FGMRESResult:
    solution: Array
    converged: bool
    iterations: int
    residual_norm: float
    relative_residual: float
    residual_history: tuple[float, ...]
    compatibility_shift: float = 0.0


def fgmres(
    operator: Callable[[Array], Array],
    rhs: Array,
    *,
    preconditioner: Callable[[Array], Array] | None = None,
    initial: Array | None = None,
    inner: Callable[[Array, Array], Array] | None = None,
    project: Callable[[Array], Array] | None = None,
    config: FGMRESConfig = FGMRESConfig(),
) -> FGMRESResult:
    """Solve a real linear system with restarted right-preconditioned FGMRES."""
    if initial is not None and tuple(initial.shape) != tuple(rhs.shape):
        raise ValueError("initial guess and RHS shapes must match")
    dot = inner or (lambda left, right: jnp.vdot(left, right).real)
    projection = project or (lambda value: value)
    precondition = preconditioner or (lambda value: value)
    apply = jax.jit(operator) if config.jit_kernels else operator
    apply_preconditioner = (
        jax.jit(precondition) if config.jit_kernels else precondition
    )

    def norm(value: Array) -> Array:
        return jnp.sqrt(jnp.maximum(dot(value, value), 0.0))

    x = jnp.zeros_like(rhs) if initial is None else jnp.asarray(initial)
    x = projection(x)
    b = projection(rhs)
    rhs_norm = float(norm(b))
    target = max(
        config.absolute_tolerance,
        config.relative_tolerance * rhs_norm,
    )
    residual = projection(b - apply(x))
    beta = float(norm(residual))
    history = [beta]
    if beta <= target:
        relative = 0.0 if rhs_norm == 0.0 else beta / rhs_norm
        return FGMRESResult(x, True, 0, beta, relative, tuple(history))

    total_iterations = 0
    converged = False
    epsilon = float(jnp.finfo(rhs.dtype).eps)

    while total_iterations < config.max_iterations:
        cycle_size = min(
            config.restart,
            config.max_iterations - total_iterations,
        )
        basis = [residual / beta]
        preconditioned_basis: list[Array] = []
        hessenberg = jnp.zeros(
            (cycle_size + 1, cycle_size),
            dtype=rhs.dtype,
        )
        cosines = jnp.zeros((cycle_size,), dtype=rhs.dtype)
        sines = jnp.zeros((cycle_size,), dtype=rhs.dtype)
        transformed_rhs = jnp.zeros((cycle_size + 1,), dtype=rhs.dtype)
        transformed_rhs = transformed_rhs.at[0].set(beta)
        used = 0

        for column in range(cycle_size):
            preconditioned = projection(
                apply_preconditioner(basis[column])
            )
            preconditioned_basis.append(preconditioned)
            work = projection(apply(preconditioned))

            for row in range(column + 1):
                coefficient = dot(basis[row], work)
                hessenberg = hessenberg.at[row, column].add(coefficient)
                work = work - coefficient * basis[row]
            if config.reorthogonalize:
                for row in range(column + 1):
                    correction = dot(basis[row], work)
                    hessenberg = hessenberg.at[row, column].add(correction)
                    work = work - correction * basis[row]

            next_norm = norm(work)
            hessenberg = hessenberg.at[column + 1, column].set(next_norm)
            safe_norm = jnp.where(next_norm > 0.0, next_norm, 1.0)
            basis.append(projection(work / safe_norm))

            for row in range(column):
                upper = hessenberg[row, column]
                lower = hessenberg[row + 1, column]
                hessenberg = hessenberg.at[row, column].set(
                    cosines[row] * upper + sines[row] * lower
                )
                hessenberg = hessenberg.at[row + 1, column].set(
                    -sines[row] * upper + cosines[row] * lower
                )

            upper = hessenberg[column, column]
            lower = hessenberg[column + 1, column]
            magnitude = jnp.hypot(upper, lower)
            safe_magnitude = jnp.where(magnitude > 0.0, magnitude, 1.0)
            cosine = upper / safe_magnitude
            sine = lower / safe_magnitude
            cosines = cosines.at[column].set(cosine)
            sines = sines.at[column].set(sine)
            hessenberg = hessenberg.at[column, column].set(
                cosine * upper + sine * lower
            )
            hessenberg = hessenberg.at[column + 1, column].set(0.0)

            old_value = transformed_rhs[column]
            transformed_rhs = transformed_rhs.at[column].set(
                cosine * old_value
            )
            transformed_rhs = transformed_rhs.at[column + 1].set(
                -sine * old_value
            )
            estimated_residual = float(
                jnp.abs(transformed_rhs[column + 1])
            )
            history.append(estimated_residual)
            used = column + 1
            total_iterations += 1
            if estimated_residual <= target or float(next_norm) <= epsilon:
                break

        triangular = hessenberg[:used, :used]
        coefficients = jnp.linalg.solve(
            triangular,
            transformed_rhs[:used],
        )
        correction = jnp.zeros_like(x)
        for index in range(used):
            correction = correction + coefficients[index] * preconditioned_basis[
                index
            ]
        x = projection(x + correction)
        residual = projection(b - apply(x))
        beta = float(norm(residual))
        if beta <= target:
            converged = True
            break
        if used == 0:
            break

    relative = 0.0 if rhs_norm == 0.0 else beta / rhs_norm
    return FGMRESResult(
        x,
        converged,
        total_iterations,
        beta,
        relative,
        tuple(history),
    )


__all__ = ["FGMRESConfig", "FGMRESResult", "fgmres"]
