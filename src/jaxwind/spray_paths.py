"""Uniform-grid residence intervals for frozen-velocity parcel trajectories."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class ParcelPath(NamedTuple):
    cells: jax.Array
    durations: jax.Array
    position: jax.Array
    exited: jax.Array
    accepted: jax.Array
    segments: jax.Array
    elapsed: jax.Array
    trapped: jax.Array


def build_parcel_path(grid, *, periodic_x, periodic_y, max_segments=64, wall_policy="reject"):
    """Trace one straight path, with xyz positions/velocities and zyx cell IDs.

    step(position, velocity, dt). Segment residence times sum to dt or the
    open-x exit time. Periodic x/y crossings wrap; z and nonperiodic y are
    impermeable, with no impaction model. Exceeding capacity or reaching such a
    wall rejects the entire path. No epsilon displacement changes residence.
    Negative motion from an exact face starts in the cell on its negative side.
    Dynamics along the path must be supplied separately; this is geometry only.
    """
    if wall_policy not in ("reject", "trap"):
        raise ValueError("wall_policy must be reject or trap")
    if not grid.is_uniform or max_segments < 1:
        raise ValueError("uniform grid and positive path capacity required")
    widths = jnp.array([grid.dx, grid.dy, grid.dz])
    counts = jnp.array([grid.nx, grid.ny, grid.nz], dtype=jnp.int32)
    periodic = jnp.array([periodic_x, periodic_y, False])
    lengths = widths * counts

    def step(position, velocity, dt):
        position, velocity = jnp.asarray(position), jnp.asarray(velocity)
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("path position and velocity must be xyz vectors")
        dtype = position.dtype
        finite = jnp.all(jnp.isfinite(position)) & jnp.all(jnp.isfinite(velocity))
        valid = finite & jnp.isfinite(dt) & (dt > 0)
        valid &= jnp.all(periodic | ((position >= 0) & (position <= lengths)))
        pos = jnp.where(periodic, jnp.mod(position, lengths), position)
        coordinate = pos / widths
        base = jnp.floor(coordinate).astype(jnp.int32)
        base -= ((velocity < 0) & (coordinate == jnp.floor(coordinate))).astype(
            jnp.int32
        )
        # Stationary endpoint belongs to the adjacent physical cell.
        base = jnp.where((velocity == 0) & (base == counts), counts - 1, base)
        direction = jnp.sign(velocity).astype(jnp.int32)
        face = (base + (velocity > 0)) * widths
        next_time = jnp.where(
            velocity != 0, (face - pos) / jnp.where(velocity != 0, velocity, 1), jnp.inf
        )
        stride = jnp.where(
            velocity != 0,
            widths / jnp.where(velocity != 0, jnp.abs(velocity), 1),
            jnp.inf,
        )
        cells = jnp.zeros((max_segments, 3), jnp.int32)
        durations = jnp.zeros((max_segments,), dtype)

        # Unwrapped integer cells and global crossing times avoid coordinate
        # nudges and allow repeated periodic crossings without losing time.
        def classify_raw(index):
            outside = (index < 0) | (index >= counts)
            wall = outside[2] | (outside[1] & jnp.logical_not(periodic_y))
            out = outside[0] & jnp.logical_not(periodic_x)
            return out, wall

        def classify(index):
            out, wall = classify_raw(index)
            return (out | wall, jnp.asarray(False)) if wall_policy == "trap" else (out, wall)

        out, wall = classify(base)
        initial = (
            base,
            next_time,
            jnp.asarray(0.0, dtype),
            cells,
            durations,
            jnp.asarray(0, jnp.int32),
            out,
            valid & ~wall,
        )

        def condition(state):
            _, _, elapsed, _, _, n, exited, ok = state
            return ok & ~exited & (elapsed < dt) & (n < max_segments)

        def advance(state):
            index, times, elapsed, cells, durations, n, _, ok = state
            crossing = jnp.min(times)
            end = jnp.minimum(dt, crossing)
            duration = end - elapsed
            cells = cells.at[n].set(jnp.mod(index, counts)[::-1])
            durations = durations.at[n].set(duration)
            hit = (times == crossing) & (crossing <= dt)
            index = index + direction * hit
            times = jnp.where(hit, times + stride, times)
            exited, wall = classify(index)
            return (
                index,
                times,
                end,
                cells,
                durations,
                n + 1,
                exited,
                ok & ~wall & (duration > 0),
            )

        _index, _times, elapsed, cells, durations, n, exited, ok = jax.lax.while_loop(
            condition, advance, initial
        )
        ok &= exited | (elapsed >= dt)
        end = pos + velocity * elapsed
        end = jnp.where(periodic, jnp.mod(end, lengths), end)
        return ParcelPath(
            jnp.where(ok, cells, 0),
            jnp.where(ok, durations, 0),
            jnp.where(ok, end, position),
            ok & exited,
            ok,
            n,
            elapsed,
            ok & classify_raw(_index)[1] & (wall_policy == "trap"),
        )

    return step
