"""Compact physical-space filters for finite-volume LES closures."""

from __future__ import annotations

import math
from typing import Literal

import jax
from jax import lax
import jax.numpy as jnp


Array = jax.Array
FilterBoundary = Literal["periodic", "reflect", "reflect_odd"]


def top_hat_stencil(filter_width: float) -> tuple[tuple[int, float], ...]:
    """Return cell-average weights for a centered top-hat in grid units."""
    if not math.isfinite(filter_width) or filter_width <= 0.0:
        raise ValueError("filter width must be positive and finite")
    radius = int(math.ceil(0.5 * filter_width + 0.5))
    half_width = 0.5 * filter_width
    stencil = []
    for offset in range(-radius, radius + 1):
        overlap = max(
            0.0,
            min(offset + 0.5, half_width)
            - max(offset - 0.5, -half_width),
        )
        if overlap > 0.0:
            stencil.append((offset, overlap / filter_width))
    total = sum(weight for _, weight in stencil)
    return tuple((offset, weight / total) for offset, weight in stencil)


def _sample(
    values: Array,
    offset: int,
    *,
    axis: int,
    boundary: FilterBoundary,
    odd_reflect_components: tuple[int, ...] = (),
) -> Array:
    if boundary == "periodic":
        return jnp.roll(values, -offset, axis=axis)
    if boundary not in {"reflect", "reflect_odd"}:
        raise ValueError(f"unsupported filter boundary: {boundary!r}")
    size = values.shape[axis]
    if size == 1:
        return values
    indices = jnp.arange(size) + offset
    folded = jnp.mod(indices, 2 * size)
    reflected = jnp.where(folded < size, folded, 2 * size - 1 - folded)
    sampled = jnp.take(values, reflected, axis=axis)
    if boundary == "reflect_odd" or odd_reflect_components:
        blocks = jnp.floor_divide(indices, size)
        sign = jnp.where(jnp.mod(blocks, 2) == 0, 1.0, -1.0)
        shape = [1] * values.ndim
        shape[axis] = size
        sign = jnp.reshape(sign, shape)
        if boundary == "reflect_odd":
            sampled = sampled * sign
        else:
            mask_shape = [1] * values.ndim
            mask_shape[-1] = values.shape[-1]
            mask = jnp.zeros((values.shape[-1],), dtype=bool)
            mask = mask.at[jnp.asarray(odd_reflect_components)].set(True)
            sampled = sampled * jnp.where(
                jnp.reshape(mask, mask_shape),
                sign,
                1.0,
            )
    return sampled


def _padded_axis(
    values: Array,
    *,
    axis: int,
    radius: int,
    boundary: FilterBoundary,
    odd_reflect_components: tuple[int, ...],
) -> Array:
    """Pad one axis without communicating with the opposite physical wall."""
    size = values.shape[axis]
    coordinates = jnp.arange(-radius, size + radius)
    if boundary == "periodic":
        return jnp.take(values, jnp.mod(coordinates, size), axis=axis)
    folded = jnp.mod(coordinates, 2 * size)
    reflected = jnp.where(folded < size, folded, 2 * size - 1 - folded)
    padded = jnp.take(values, reflected, axis=axis)
    if boundary == "reflect_odd" or odd_reflect_components:
        blocks = jnp.floor_divide(coordinates, size)
        sign = jnp.where(jnp.mod(blocks, 2) == 0, 1.0, -1.0)
        sign_shape = [1] * values.ndim
        sign_shape[axis] = size + 2 * radius
        sign = jnp.reshape(sign, sign_shape)
        if boundary == "reflect_odd":
            padded = padded * sign
        else:
            mask_shape = [1] * values.ndim
            mask_shape[-1] = values.shape[-1]
            mask = jnp.zeros((values.shape[-1],), dtype=bool)
            mask = mask.at[jnp.asarray(odd_reflect_components)].set(True)
            padded = padded * jnp.where(
                jnp.reshape(mask, mask_shape),
                sign,
                1.0,
            )
    return padded


def _compact_even_width_filter(
    values: Array,
    filter_width: float,
    *,
    axis: int,
    boundary: FilterBoundary,
    odd_reflect_components: tuple[int, ...],
) -> Array:
    """Fast exact forms of the width-two and width-four overlap stencils."""
    if values.shape[axis] == 1:
        return values
    radius = 1 if filter_width == 2.0 else 2
    padded = _padded_axis(
        values,
        axis=axis,
        radius=radius,
        boundary=boundary,
        odd_reflect_components=odd_reflect_components,
    )
    window = [1] * values.ndim
    window[axis] = 2 * radius + 1
    wide_sum = lax.reduce_window(
        padded,
        jnp.asarray(0.0, dtype=values.dtype),
        lax.add,
        tuple(window),
        (1,) * values.ndim,
        "VALID",
    )
    if filter_width == 2.0:
        # [1/4, 1/2, 1/4] = (box_3 + identity) / 4.
        return 0.25 * (wide_sum + values)

    # [1/8, 1/4, 1/4, 1/4, 1/8] = (box_5 + box_3) / 8.
    window[axis] = 3
    narrow_sum = lax.reduce_window(
        padded,
        jnp.asarray(0.0, dtype=values.dtype),
        lax.add,
        tuple(window),
        (1,) * values.ndim,
        "VALID",
    )
    interior = [slice(None)] * values.ndim
    interior[axis] = slice(1, -1)
    return 0.125 * (wide_sum + narrow_sum[tuple(interior)])


def _compact_even_width_filter_pair(
    values: Array,
    *,
    axis: int,
    boundary: FilterBoundary,
    odd_reflect_components: tuple[int, ...],
) -> tuple[Array, Array]:
    """Apply the exact width-two and width-four stencils from one padding.

    Both compact filters need the centered three-point box sum.  Computing
    them together also lets the width-two path reuse the radius-two padded
    input required by width four.
    """
    if values.shape[axis] == 1:
        return values, values
    padded = _padded_axis(
        values,
        axis=axis,
        radius=2,
        boundary=boundary,
        odd_reflect_components=odd_reflect_components,
    )
    window = [1] * values.ndim
    window[axis] = 5
    box_five = lax.reduce_window(
        padded,
        jnp.asarray(0.0, dtype=values.dtype),
        lax.add,
        tuple(window),
        (1,) * values.ndim,
        "VALID",
    )
    window[axis] = 3
    box_three_padded = lax.reduce_window(
        padded,
        jnp.asarray(0.0, dtype=values.dtype),
        lax.add,
        tuple(window),
        (1,) * values.ndim,
        "VALID",
    )
    interior = [slice(None)] * values.ndim
    interior[axis] = slice(1, -1)
    box_three = box_three_padded[tuple(interior)]
    return (
        0.25 * (box_three + values),
        0.125 * (box_five + box_three),
    )


def physical_top_hat_filter(
    values: Array,
    filter_width: float,
    *,
    axes: tuple[int, ...] = (-2, -1),
    boundaries: tuple[FilterBoundary, ...] = ("periodic", "periodic"),
    odd_reflect_components: tuple[int, ...] = (),
) -> Array:
    """Apply a separable compact top-hat without any spectral transform.

    ``reflect`` is an even, cell-centred extension and ``reflect_odd`` is its
    antisymmetric counterpart.  Neither samples the opposite side of the
    domain.
    """
    if len(axes) != len(boundaries):
        raise ValueError("one filter boundary is required for every axis")
    canonical_axes = tuple(axis % values.ndim for axis in axes)
    if odd_reflect_components:
        if values.ndim < 2 or values.ndim - 1 in canonical_axes:
            raise ValueError(
                "odd reflection components require a separate trailing "
                "component axis"
            )
        if min(odd_reflect_components) < 0 or max(
            odd_reflect_components
        ) >= values.shape[-1]:
            raise ValueError("odd reflection component index is out of range")
    stencil = top_hat_stencil(filter_width)
    filtered = values
    for axis, boundary in zip(canonical_axes, boundaries, strict=True):
        if boundary not in {"periodic", "reflect", "reflect_odd"}:
            raise ValueError(f"unsupported filter boundary: {boundary!r}")
        component_parity = (
            odd_reflect_components if boundary == "reflect" else ()
        )
        if filter_width in {2.0, 4.0}:
            filtered = _compact_even_width_filter(
                filtered,
                filter_width,
                axis=axis,
                boundary=boundary,
                odd_reflect_components=component_parity,
            )
            continue
        accumulated = jnp.zeros_like(filtered)
        for offset, weight in stencil:
            accumulated += weight * _sample(
                filtered,
                offset,
                axis=axis,
                boundary=boundary,
                odd_reflect_components=component_parity,
            )
        filtered = accumulated
    return filtered.astype(values.dtype)


def physical_top_hat_filter_pair(
    values: Array,
    *,
    axes: tuple[int, ...] = (-2, -1),
    boundaries: tuple[FilterBoundary, ...] = ("periodic", "periodic"),
    odd_reflect_components: tuple[int, ...] = (),
) -> tuple[Array, Array]:
    """Apply width-two and width-four compact top-hats together.

    The first separable axis shares padding, input reads, and its three-point
    box sum.  Later axes remain two deliberately bounded branches because
    their inputs have already diverged.  This keeps the optimization local to
    filtering instead of constructing one oversized fused timestep graph.
    """
    if len(axes) != len(boundaries):
        raise ValueError("one filter boundary is required for every axis")
    if not axes:
        return values, values
    canonical_axes = tuple(axis % values.ndim for axis in axes)
    if odd_reflect_components:
        if values.ndim < 2 or values.ndim - 1 in canonical_axes:
            raise ValueError(
                "odd reflection components require a separate trailing "
                "component axis"
            )
        if min(odd_reflect_components) < 0 or max(
            odd_reflect_components
        ) >= values.shape[-1]:
            raise ValueError("odd reflection component index is out of range")
    for boundary in boundaries:
        if boundary not in {"periodic", "reflect", "reflect_odd"}:
            raise ValueError(f"unsupported filter boundary: {boundary!r}")

    first_axis = canonical_axes[0]
    first_boundary = boundaries[0]
    component_parity = (
        odd_reflect_components if first_boundary == "reflect" else ()
    )
    filtered_two, filtered_four = _compact_even_width_filter_pair(
        values,
        axis=first_axis,
        boundary=first_boundary,
        odd_reflect_components=component_parity,
    )
    for axis, boundary in zip(
        canonical_axes[1:],
        boundaries[1:],
        strict=True,
    ):
        component_parity = (
            odd_reflect_components if boundary == "reflect" else ()
        )
        filtered_two = _compact_even_width_filter(
            filtered_two,
            2.0,
            axis=axis,
            boundary=boundary,
            odd_reflect_components=component_parity,
        )
        filtered_four = _compact_even_width_filter(
            filtered_four,
            4.0,
            axis=axis,
            boundary=boundary,
            odd_reflect_components=component_parity,
        )
    return (
        filtered_two.astype(values.dtype),
        filtered_four.astype(values.dtype),
    )


__all__ = [
    "FilterBoundary",
    "physical_top_hat_filter",
    "physical_top_hat_filter_pair",
    "top_hat_stencil",
]
