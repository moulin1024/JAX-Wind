from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _params():
    import jax.numpy as jnp

    from wireles_jax import Params

    return Params(
        nx=8,
        ny=4,
        nz=4,
        lx=4.0,
        ly=2.0,
        lz=2.0,
        z_i=1.0,
        dt=1.0e-4,
        nsteps=4,
        momentum_wall_model="free_slip",
        initial_velocity_noise=0.01,
        sgs_model="smagorinsky",
        scalar_sgs_model="fixed_prandtl",
        fringe_enabled=True,
        fringe_start_x=3.0,
        fringe_timescale=0.25,
        dtype=jnp.float32,
    )


def test_manifest_and_rank_local_batches_round_trip(tmp_path: Path) -> None:
    from wireles_jax.inflow_batches import (
        build_inflow_manifest,
        load_local_inflow_batch,
        read_inflow_manifest,
        write_inflow_manifest,
        write_local_inflow_batch,
    )

    params = _params()
    manifest = build_inflow_manifest(
        params,
        source_parts=2,
        start_step=17,
        total_steps=4,
        batch_steps=3,
        compress=False,
    )
    assert manifest["global_shape_per_step"] == [5, 2, 4, 4]
    assert [batch["step_count"] for batch in manifest["batches"]] == [3, 1]
    assert (
        manifest["batches"][0]["uncompressed_bytes_per_rank"]
        == 3 * 5 * 2 * 4 * 2 * np.dtype(np.float32).itemsize
    )

    for batch in manifest["batches"]:
        batch_id = int(batch["batch_id"])
        for rank in range(2):
            shape = (int(batch["step_count"]), 5, 2, 4, 2)
            packed = np.full(shape, 10 * batch_id + rank, dtype=np.float32)
            write_local_inflow_batch(
                tmp_path,
                packed,
                batch_id=batch_id,
                rank=rank,
                global_start_step=int(batch["global_start_step"]),
                z_start=2 * rank,
                z_stop=2 * (rank + 1),
            )

    write_inflow_manifest(tmp_path, manifest)
    restored_manifest = read_inflow_manifest(tmp_path)
    for batch_id in range(2):
        for rank in range(2):
            restored = load_local_inflow_batch(
                tmp_path,
                restored_manifest,
                batch_id=batch_id,
                rank=rank,
            )
            np.testing.assert_array_equal(restored, 10 * batch_id + rank)


def test_precursor_batch_snapshots_pre_step_fields() -> None:
    import jax

    from wireles_jax.inflow_batches import make_precursor_inflow_batch
    from wireles_jax.sharding import make_single_node_mesh
    from wireles_jax.timestep_sharded import (
        initial_sharded_state,
        make_sharded_operators,
    )

    params = _params()
    mesh = make_single_node_mesh(1)
    state = initial_sharded_state(params, mesh, seed=9)
    expected = np.stack(
        [
            np.asarray(jax.device_get(getattr(state, name)))[6:]
            for name in ("u", "v", "w", "theta", "qv")
        ],
        axis=0,
    )
    ops = make_sharded_operators(params, mesh)
    advance = jax.jit(
        make_precursor_inflow_batch(
            params,
            ops,
            mesh,
            batch_steps=2,
        ),
        donate_argnums=(0,),
    )

    final, snapshots = jax.block_until_ready(
        advance(state, ops.pressure, ops.pressure_spike)
    )
    snapshots = np.asarray(jax.device_get(snapshots))
    assert snapshots.shape == (2, 5, 2, 4, 4)
    np.testing.assert_array_equal(snapshots[0], expected)
    assert int(jax.device_get(final.step)) == 2
    assert np.isfinite(snapshots).all()
