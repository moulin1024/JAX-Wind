"""Preconditioned conjugate gradients for weighted JAX array systems."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import jax
import jax.numpy as jnp


Array = jax.Array


@dataclass(frozen=True, slots=True)
class PCGConfig:
    """Preconditioned conjugate-gradient controls."""

    max_iterations: int = 120
    relative_tolerance: float = 1.0e-8
    absolute_tolerance: float = 0.0
    jit_kernels: bool = True
    execution: Literal["python", "jax"] = "python"

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.relative_tolerance < 0.0 or self.absolute_tolerance < 0.0:
            raise ValueError("solver tolerances must be nonnegative")
        if self.execution not in {"python", "jax"}:
            raise ValueError("PCG execution must be 'python' or 'jax'")


@dataclass(frozen=True, slots=True)
class PCGResult:
    """Result and convergence diagnostics from a PCG solve."""

    solution: Array
    converged: bool
    iterations: int
    residual_norm: float
    relative_residual: float
    residual_history: tuple[float, ...]
    compatibility_shift: float = 0.0


def pcg(
    operator: Callable[[Array], Array],
    rhs: Array,
    *,
    preconditioner: Callable[[Array], Array] | None = None,
    initial: Array | None = None,
    inner: Callable[[Array, Array], Array] | None = None,
    project: Callable[[Array], Array] | None = None,
    config: PCGConfig = PCGConfig(),
) -> PCGResult:
    """Solve an SPD system with a fixed SPD left preconditioner."""
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
    residual_norm = float(norm(residual))
    history = [residual_norm]
    if residual_norm <= target:
        relative = 0.0 if rhs_norm == 0.0 else residual_norm / rhs_norm
        return PCGResult(x, True, 0, residual_norm, relative, tuple(history))

    preconditioned = projection(apply_preconditioner(residual))
    residual_dot_preconditioned = float(dot(residual, preconditioned))
    if (
        not math.isfinite(residual_dot_preconditioned)
        or residual_dot_preconditioned <= 0.0
    ):
        relative = (
            0.0 if rhs_norm == 0.0 else residual_norm / rhs_norm
        )
        return PCGResult(
            x,
            False,
            0,
            residual_norm,
            relative,
            tuple(history),
        )
    direction = preconditioned
    converged = False
    iterations = 0

    for iteration in range(1, config.max_iterations + 1):
        action = projection(apply(direction))
        curvature = float(dot(direction, action))
        if not math.isfinite(curvature) or curvature <= 0.0:
            break
        step = residual_dot_preconditioned / curvature
        x = projection(x + step * direction)
        residual = projection(residual - step * action)
        residual_norm = float(norm(residual))
        history.append(residual_norm)
        iterations = iteration
        if not math.isfinite(residual_norm):
            break
        if residual_norm <= target:
            converged = True
            break

        preconditioned = projection(apply_preconditioner(residual))
        next_dot = float(dot(residual, preconditioned))
        if not math.isfinite(next_dot) or next_dot <= 0.0:
            break
        direction = projection(
            preconditioned
            + (next_dot / residual_dot_preconditioned) * direction
        )
        residual_dot_preconditioned = next_dot

    relative = 0.0 if rhs_norm == 0.0 else residual_norm / rhs_norm
    return PCGResult(
        x,
        converged,
        iterations,
        residual_norm,
        relative,
        tuple(history),
    )


__all__ = ["PCGConfig", "PCGResult", "pcg"]
