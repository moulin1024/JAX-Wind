from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .config import Params
from .sharding import _shard_map, adjoint_z_slab_spec
from .timestep_sharded import (
    ShardedFlowState,
    ShardedOperators,
    make_step_ab2_sharded,
)


FIELD_COUNT = 5


def fringe_start_index(params: Params) -> int:
    x = (np.arange(params.nx) + 0.5) * params.dx * params.z_i
    start = int(np.searchsorted(x, params.fringe_start_x))
    if start >= params.nx:
        raise ValueError("Concurrent fringe contains no cell centres")
    return start


def adjoint_fringe_chunk_spec(
    adjoint_axis_name: str = "adjoint", z_axis_name: str = "z"
) -> P:
    """Partition `(time, adjoint, field, x, y, z)` chunks."""

    return P(None, adjoint_axis_name, None, None, None, z_axis_name)


def duplicate_state_for_adjoint(
    state: ShardedFlowState,
    mesh: Mesh,
    *,
    adjoint_axis_name: str = "adjoint",
    z_axis_name: str = "z",
) -> ShardedFlowState:
    """Duplicate a warm-up state using device-to-device JAX resharding.

    The logical leading axis is `(precursor, turbine)`.  No host materializes
    either global field while changing from the warm-up z mesh to the 2-D mesh.
    """

    field_sharding = NamedSharding(
        mesh, adjoint_z_slab_spec(adjoint_axis_name, z_axis_name)
    )
    scalar_sharding = NamedSharding(
        mesh, P(adjoint_axis_name, None, None, z_axis_name, None)
    )
    step_sharding = NamedSharding(mesh, P(adjoint_axis_name))

    def duplicate(x: jax.Array, sharding: NamedSharding) -> jax.Array:
        return jax.jit(
            lambda q: jnp.stack((q, q), axis=0), out_shardings=sharding
        )(x)

    values = []
    for name, value in zip(state._fields, state, strict=True):
        if name == "step":
            values.append(duplicate(value, step_sharding))
        elif name == "scalar_c":
            values.append(duplicate(value, scalar_sharding))
        else:
            values.append(duplicate(value, field_sharding))
    return ShardedFlowState(*values)


def make_empty_fringe_chunk(
    params: Params,
    mesh: Mesh,
    chunk_steps: int,
    *,
    adjoint_axis_name: str = "adjoint",
    z_axis_name: str = "z",
) -> jax.Array:
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")
    nx_fringe = params.nx - fringe_start_index(params)
    shape = (chunk_steps, 2, FIELD_COUNT, nx_fringe, params.ny, params.nz)
    sharding = NamedSharding(
        mesh, adjoint_fringe_chunk_spec(adjoint_axis_name, z_axis_name)
    )
    return jax.jit(
        lambda: jnp.zeros(shape, dtype=params.dtype), out_shardings=sharding
    )()


def _fringe_snapshot(state: ShardedFlowState, start: int) -> jax.Array:
    return jnp.stack(
        (
            state.u[:, start:],
            state.v[:, start:],
            state.w[:, start:],
            state.theta[:, start:],
            state.qv[:, start:],
        ),
        axis=1,
    )


def _select_domains(
    old: ShardedFlowState,
    new: ShardedFlowState,
    *,
    advance_turbine: bool,
) -> ShardedFlowState:
    if advance_turbine:
        return new
    selected = []
    for old_value, new_value in zip(old, new, strict=True):
        precursor = jnp.arange(old_value.shape[0]) == 0
        mask = precursor.reshape(
            precursor.shape + (1,) * (old_value.ndim - precursor.ndim)
        )
        selected.append(jnp.where(mask, new_value, old_value))
    return ShardedFlowState(*selected)


def make_adjoint_chunk_step(
    params: Params,
    ops: ShardedOperators,
    mesh: Mesh,
    *,
    chunk_steps: int,
    advance_turbine: bool = True,
    adjoint_axis_name: str = "adjoint",
    z_axis_name: str = "z",
) -> Callable[..., tuple[ShardedFlowState, jax.Array]]:
    """Advance one fixed-size chunk and record precursor fringe snapshots."""

    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")
    start = fringe_start_index(params)
    step = make_step_ab2_sharded(
        params,
        ops,
        mesh,
        z_axis_name,
        concurrent_fringe=True,
        adjoint_axis_name=adjoint_axis_name,
    )

    def chunk(
        state: ShardedFlowState,
        targets: jax.Array,
        runtime_pressure_ops,
        runtime_spike_ops,
    ) -> tuple[ShardedFlowState, jax.Array]:
        if targets.shape[0] != chunk_steps:
            raise ValueError(
                f"Expected {chunk_steps} target snapshots, got {targets.shape[0]}"
            )

        def body(carry: ShardedFlowState, target: jax.Array):
            snapshot = _fringe_snapshot(carry, start)
            fringe_target = tuple(target[:, field] for field in range(FIELD_COUNT))
            proposed = step(
                carry,
                runtime_pressure_ops,
                runtime_spike_ops,
                fringe_target,
            )
            return (
                _select_domains(
                    carry, proposed, advance_turbine=advance_turbine
                ),
                snapshot,
            )

        return lax.scan(body, state, targets)

    return chunk


def make_exchange_precursor_chunk(
    mesh: Mesh,
    *,
    adjoint_axis_name: str = "adjoint",
    z_axis_name: str = "z",
) -> Callable[[jax.Array], jax.Array]:
    """Send one packed precursor chunk to the turbine domain with ppermute."""

    spec = adjoint_fringe_chunk_spec(adjoint_axis_name, z_axis_name)

    def local_exchange(chunk_local: jax.Array) -> jax.Array:
        # The adjoint shard has local extent one.  CollectivePermute sends the
        # packed time/field block once per z pair and fills non-destinations.
        packed = jnp.squeeze(chunk_local, axis=1)
        received = lax.ppermute(
            packed, axis_name=adjoint_axis_name, perm=((0, 1),)
        )
        return jnp.expand_dims(received, axis=1)

    return _shard_map(
        local_exchange,
        mesh=mesh,
        in_specs=spec,
        out_specs=spec,
        axis_name=z_axis_name,
        additional_axis_names=(adjoint_axis_name,),
    )


def make_adjoint_pipeline_prime(
    params: Params,
    ops: ShardedOperators,
    mesh: Mesh,
    *,
    chunk_steps: int,
    adjoint_axis_name: str = "adjoint",
    z_axis_name: str = "z",
) -> Callable[..., tuple[ShardedFlowState, jax.Array]]:
    """Advance the precursor look-ahead and exchange its first packed chunk.

    Keeping the initial exchange inside the compiled function avoids a host
    synchronization between pipeline initialization and the first batch.
    """

    prime_chunk = make_adjoint_chunk_step(
        params,
        ops,
        mesh,
        chunk_steps=chunk_steps,
        advance_turbine=False,
        adjoint_axis_name=adjoint_axis_name,
        z_axis_name=z_axis_name,
    )
    exchange = make_exchange_precursor_chunk(
        mesh,
        adjoint_axis_name=adjoint_axis_name,
        z_axis_name=z_axis_name,
    )

    def prime(
        state: ShardedFlowState,
        empty_targets: jax.Array,
        runtime_pressure_ops,
        runtime_spike_ops,
    ) -> tuple[ShardedFlowState, jax.Array]:
        state, produced = prime_chunk(
            state,
            empty_targets,
            runtime_pressure_ops,
            runtime_spike_ops,
        )
        return state, exchange(produced)

    return prime


def make_adjoint_pipeline_batch(
    params: Params,
    ops: ShardedOperators,
    mesh: Mesh,
    *,
    chunk_steps: int,
    chunks_per_launch: int,
    adjoint_axis_name: str = "adjoint",
    z_axis_name: str = "z",
) -> Callable[..., tuple[ShardedFlowState, jax.Array]]:
    """Build one device-resident batch of concurrent precursor chunks.

    The outer scan fuses flow advancement and packed precursor exchange.  The
    host therefore dispatches once per batch rather than once per chunk while
    the state and mailbox remain distributed on the two-dimensional mesh.
    """

    if chunks_per_launch <= 0:
        raise ValueError("chunks_per_launch must be positive")
    advance_chunk = make_adjoint_chunk_step(
        params,
        ops,
        mesh,
        chunk_steps=chunk_steps,
        advance_turbine=True,
        adjoint_axis_name=adjoint_axis_name,
        z_axis_name=z_axis_name,
    )
    exchange = make_exchange_precursor_chunk(
        mesh,
        adjoint_axis_name=adjoint_axis_name,
        z_axis_name=z_axis_name,
    )

    def batch(
        state: ShardedFlowState,
        targets: jax.Array,
        runtime_pressure_ops,
        runtime_spike_ops,
    ) -> tuple[ShardedFlowState, jax.Array]:
        def body(carry, _):
            current_state, current_targets = carry
            next_state, produced = advance_chunk(
                current_state,
                current_targets,
                runtime_pressure_ops,
                runtime_spike_ops,
            )
            return (next_state, exchange(produced)), None

        (state, targets), _ = lax.scan(
            body,
            (state, targets),
            xs=None,
            length=chunks_per_launch,
        )
        return state, targets

    return batch
