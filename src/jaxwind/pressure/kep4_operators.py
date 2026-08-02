"""Fourth-order compatible staggered derivatives for uniform MAC grids.

The face-to-cell divergence is defined as the negative Euclidean transpose
of the cell-to-face pressure gradient. Periodic face endpoints carry one half
weight each because they represent the same physical degree of freedom.
Homogeneous Neumann walls use even pressure reflection and odd normal-velocity
reflection.  Those closures impose zero wall-normal pressure gradient while
preserving the negative-transpose relation and fourth-order accuracy.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


Array = jax.Array


def uniform_spacing(faces: tuple[float, ...], name: str) -> float:
    """Return a uniform spacing or reject a grid incompatible with KEP4."""
    spacing = (faces[-1] - faces[0]) / (len(faces) - 1)
    tolerance = 1.0e-12 * max(1.0, abs(spacing))
    if not all(
        math.isclose(
            right - left,
            spacing,
            rel_tol=1.0e-12,
            abs_tol=tolerance,
        )
        for left, right in zip(faces[:-1], faces[1:], strict=True)
    ):
        raise ValueError(f"KEP4 pressure projection requires uniform {name}")
    return spacing


def validate_kep4_pressure_grid(grid, boundaries) -> tuple[float, float, float]:
    """Validate the boundary subset supported by the compatible operator."""
    spacings = (
        uniform_spacing(grid.x_faces, "x spacing"),
        uniform_spacing(grid.y_faces, "y spacing"),
        uniform_spacing(grid.z_faces, "z spacing"),
    )
    for count, (lower, upper, name) in zip(
        (grid.shape[2], grid.shape[1], grid.shape[0]),
        boundaries.axis_pairs(),
        strict=True,
    ):
        if lower.kind == "periodic" and upper.kind == "periodic":
            if count < 4:
                raise ValueError(f"periodic KEP4 {name} requires at least four cells")
            continue
        if (
            lower.kind != "neumann"
            or upper.kind != "neumann"
            or lower.value != 0.0
            or upper.value != 0.0
        ):
            raise ValueError(
                "KEP4 pressure projection supports periodic or homogeneous "
                f"Neumann {name} boundaries"
            )
        if count < 4:
            raise ValueError(
                f"homogeneous-Neumann KEP4 {name} requires at least four cells"
            )
    return spacings


def _move_last(field: Array, axis: int) -> Array:
    return jnp.moveaxis(field, axis, -1)


def periodic_gradient_axis(field: Array, spacing: float, axis: int) -> Array:
    """Map periodic cell values to duplicated periodic face gradients."""
    values = _move_last(field, axis)
    gradient = (
        jnp.roll(values, 2, axis=-1)
        - 27.0 * jnp.roll(values, 1, axis=-1)
        + 27.0 * values
        - jnp.roll(values, -1, axis=-1)
    ) / (24.0 * spacing)
    gradient = jnp.concatenate((gradient, gradient[..., :1]), axis=-1)
    return jnp.moveaxis(gradient, -1, axis)


def periodic_divergence_axis(field: Array, spacing: float, axis: int) -> Array:
    """Map duplicated periodic faces to cells using ``-G.T`` exactly."""
    faces = _move_last(field, axis)
    unique = faces[..., :-1]
    shared = 0.5 * (faces[..., 0] + faces[..., -1])
    unique = unique.at[..., 0].set(shared)
    divergence = (
        jnp.roll(unique, 1, axis=-1)
        - 27.0 * unique
        + 27.0 * jnp.roll(unique, -1, axis=-1)
        - jnp.roll(unique, -2, axis=-1)
    ) / (24.0 * spacing)
    return jnp.moveaxis(divergence, -1, axis)


def neumann_gradient_axis(field: Array, spacing: float, axis: int) -> Array:
    """Fourth-order cell-to-face gradient using even pressure reflection."""
    values = _move_last(field, axis)
    count = values.shape[-1]
    if count < 4:
        raise ValueError("homogeneous-Neumann KEP4 requires at least four cells")
    extended = jnp.concatenate(
        (
            values[..., 1:2],
            values[..., :1],
            values,
            values[..., -1:],
            values[..., -2:-1],
        ),
        axis=-1,
    )
    gradient = (
        extended[..., : count + 1]
        - 27.0 * extended[..., 1 : count + 2]
        + 27.0 * extended[..., 2 : count + 3]
        - extended[..., 3 : count + 4]
    ) / (24.0 * spacing)
    return jnp.moveaxis(gradient, -1, axis)


def neumann_divergence_axis(field: Array, spacing: float, axis: int) -> Array:
    """Fourth-order face divergence using odd normal-velocity reflection."""
    faces = _move_last(field, axis)
    count = faces.shape[-1] - 1
    if count < 4:
        raise ValueError("homogeneous-Neumann KEP4 requires at least four cells")
    physical = faces.at[..., 0].set(0.0).at[..., -1].set(0.0)
    extended = jnp.concatenate(
        (-physical[..., 1:2], physical, -physical[..., -2:-1]),
        axis=-1,
    )
    divergence = (
        extended[..., :count]
        - 27.0 * extended[..., 1 : count + 1]
        + 27.0 * extended[..., 2 : count + 2]
        - extended[..., 3 : count + 3]
    ) / (24.0 * spacing)
    return jnp.moveaxis(divergence, -1, axis)


def gradient_axis(
    field: Array,
    *,
    spacing: float,
    axis: int,
    lower_kind: str,
    upper_kind: str,
) -> Array:
    if lower_kind == "periodic" and upper_kind == "periodic":
        return periodic_gradient_axis(field, spacing, axis)
    return neumann_gradient_axis(field, spacing, axis)


def divergence_axis(
    field: Array,
    *,
    spacing: float,
    axis: int,
    lower_kind: str,
    upper_kind: str,
) -> Array:
    if lower_kind == "periodic" and upper_kind == "periodic":
        return periodic_divergence_axis(field, spacing, axis)
    return neumann_divergence_axis(field, spacing, axis)


def poisson_axis(
    field: Array,
    *,
    spacing: float,
    axis: int,
    lower_kind: str,
    upper_kind: str,
) -> Array:
    gradient = gradient_axis(
        field,
        spacing=spacing,
        axis=axis,
        lower_kind=lower_kind,
        upper_kind=upper_kind,
    )
    return -divergence_axis(
        gradient,
        spacing=spacing,
        axis=axis,
        lower_kind=lower_kind,
        upper_kind=upper_kind,
    )


def poisson_diagonal_axis(
    count: int,
    spacing: float,
    *,
    periodic: bool,
    dtype: jnp.dtype,
) -> Array:
    """Return the exact diagonal of one positive ``-D4 G4`` axis block."""
    if periodic:
        return jnp.full(
            (count,),
            1460.0 / (576.0 * spacing * spacing),
            dtype=dtype,
        )
    diagonal = jnp.zeros((count,), dtype=dtype)
    scale = 1.0 / (24.0 * spacing)
    for index, weight in enumerate((-26.0, 27.0, -1.0)):
        diagonal = diagonal.at[index].add((weight * scale) ** 2)
    for face in range(2, count - 1):
        for offset, weight in enumerate((1.0, -27.0, 27.0, -1.0)):
            diagonal = diagonal.at[face - 2 + offset].add((weight * scale) ** 2)
    for offset, weight in enumerate((1.0, -27.0, 26.0)):
        diagonal = diagonal.at[count - 3 + offset].add((weight * scale) ** 2)
    return diagonal


_PERIODIC_POISSON_COEFFICIENTS = (-1.0, 54.0, -783.0, 1460.0, -783.0, 54.0, -1.0)


def periodic_poisson_axis_from_halo(
    padded_field: Array,
    *,
    spacing: float,
    axis: int,
    local_count: int,
) -> Array:
    """Apply the periodic seven-point ``-D4 G4`` stencil to a 3-cell halo."""
    values = _move_last(padded_field, axis)
    if values.shape[-1] != local_count + 6:
        raise ValueError("KEP4 Poisson halo must contain three cells per side")
    result = sum(
        coefficient * values[..., offset : offset + local_count]
        for offset, coefficient in enumerate(_PERIODIC_POISSON_COEFFICIENTS)
    ) / (576.0 * spacing * spacing)
    return jnp.moveaxis(result, -1, axis)


__all__ = [
    "divergence_axis",
    "gradient_axis",
    "neumann_divergence_axis",
    "neumann_gradient_axis",
    "periodic_divergence_axis",
    "periodic_gradient_axis",
    "periodic_poisson_axis_from_halo",
    "poisson_axis",
    "poisson_diagonal_axis",
    "uniform_spacing",
    "validate_kep4_pressure_grid",
]
