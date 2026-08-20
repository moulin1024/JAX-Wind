"""Rayleigh damping of the upper domain, to absorb waves at the lid.

A rigid, impermeable top wall reflects the internal waves and turbulent
fluctuations that reach it instead of letting them radiate away, and those
reflections contaminate the flow below.  The standard remedy is a sponge
layer: an extra relaxation term, active only near the top, that nudges the
velocity toward a target state on a time scale much shorter than the flow's
own, so energy is absorbed before it can reflect.

The tendency plugs into :attr:`~jaxwind.fv.integrate.FlowModel.forcing`, so it
is integrated by the same explicit scheme as everything else and needs no
change to the stepper.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from jaxwind.domain.grid import UniformGrid

from .state import StaggeredVelocity, cell_coordinates, face_coordinates


PLANE_MEAN = "plane-mean"
REST = "rest"


def _ramp(height: jnp.ndarray, start_height: float, top: float, power: float) -> jnp.ndarray:
    """The ``[0, 1]`` damping profile: flat below the start height, ramping up to
    the lid."""
    depth = max(top - start_height, 1.0e-12)
    eta = jnp.clip((height - start_height) / depth, 0.0, 1.0)
    return eta**power


def rayleigh_sponge_tendency(
    grid: UniformGrid,
    *,
    start_height: float,
    timescale: float,
    power: float = 2.0,
    target: str = PLANE_MEAN,
) -> Callable[[StaggeredVelocity, jnp.ndarray], StaggeredVelocity]:
    """Return a tendency that relaxes the layer above ``start_height``.

    ``target="plane-mean"`` relaxes ``u`` and ``v`` toward their own
    horizontal mean at each height, so the sponge removes perturbations
    without imposing a wind the pressure-driven balance did not ask for;
    ``target="rest"`` relaxes them toward zero, which is appropriate when the
    mean wind is externally prescribed (e.g. by a geostrophic forcing) rather
    than left free.  Either way ``w`` is relaxed toward zero, since a
    vertical velocity at the lid can only be numerical noise or a reflected
    wave.

    The strength ramps from zero at ``start_height`` to ``1 / timescale`` at
    the lid as ``((z - start_height) / (lz - start_height)) ** power``, the
    same shape used to damp sponge layers in the literature: gentle enough at
    its foot not to disturb the flow it borders, sharp enough at the lid to
    absorb what reaches it.
    """
    if target not in (PLANE_MEAN, REST):
        raise ValueError(f"unsupported sponge target: {target!r}")
    if timescale <= 0.0:
        raise ValueError("the sponge timescale must be positive")
    if not 0.0 <= start_height < grid.lz:
        raise ValueError("the sponge must start inside the domain")

    _, _, center_height = cell_coordinates(grid)
    _, _, face_height = face_coordinates(grid)
    center_rate = (_ramp(center_height, start_height, grid.lz, power) / timescale)[
        :, None, None
    ]
    face_rate = (_ramp(face_height, start_height, grid.lz, power) / timescale)[:, None, None]

    def tendency(velocity: StaggeredVelocity, time: jnp.ndarray) -> StaggeredVelocity:
        del time
        if target == PLANE_MEAN:
            target_x = jnp.mean(velocity.x, axis=(1, 2), keepdims=True)
            target_y = jnp.mean(velocity.y, axis=(1, 2), keepdims=True)
        else:
            target_x = jnp.zeros((), velocity.x.dtype)
            target_y = jnp.zeros((), velocity.y.dtype)
        return StaggeredVelocity(
            -center_rate.astype(velocity.x.dtype) * (velocity.x - target_x),
            -center_rate.astype(velocity.y.dtype) * (velocity.y - target_y),
            -face_rate.astype(velocity.z.dtype) * velocity.z,
        )

    return tendency


__all__ = [
    "PLANE_MEAN",
    "REST",
    "rayleigh_sponge_tendency",
]
