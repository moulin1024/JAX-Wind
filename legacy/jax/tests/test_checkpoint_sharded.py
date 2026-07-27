from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wireles_jax.checkpoint_sharded import load_repartitioned_local_slabs
from wireles_jax.timestep_sharded import ShardedFlowState


def test_four_slabs_repartition_into_two_without_global_assembly(tmp_path: Path) -> None:
    nx, ny, nz = 3, 2, 8
    array_fields = [name for name in ShardedFlowState._fields if name != "step"]
    for rank in range(4):
        payload = {}
        for field in array_fields:
            shape = (nx, ny, nz // 4, 2) if field == "scalar_c" else (nx, ny, nz // 4)
            payload[field] = np.full(shape, rank, dtype=np.float32)
        payload["step"] = np.asarray(17, dtype=np.int32)
        np.savez(tmp_path / f"rank_{rank:05d}.npz", **payload)
    manifest = {
        "format": "wireles-jax-zslab-v1",
        "source_parts": 4,
        "global_shape": [nx, ny, nz],
        "fields": array_fields,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    lower, lower_step, _ = load_repartitioned_local_slabs(
        tmp_path, target_rank=0, target_parts=2
    )
    upper, upper_step, _ = load_repartitioned_local_slabs(
        tmp_path, target_rank=1, target_parts=2
    )
    assert lower_step == upper_step == 17
    np.testing.assert_array_equal(lower["u"][:, :, :2], 0.0)
    np.testing.assert_array_equal(lower["u"][:, :, 2:], 1.0)
    np.testing.assert_array_equal(upper["u"][:, :, :2], 2.0)
    np.testing.assert_array_equal(upper["u"][:, :, 2:], 3.0)
    assert lower["u"].shape == (nx, ny, nz // 2)
    assert lower["scalar_c"].shape == (nx, ny, nz // 2, 2)


def test_checkpoint_restart_is_bitwise_identical_to_continuous_ab2(
    tmp_path: Path,
) -> None:
    import jax
    import jax.numpy as jnp

    from wireles_jax import Params
    from wireles_jax.checkpoint_sharded import (
        load_sharded_checkpoint,
        save_sharded_checkpoint,
    )
    from wireles_jax.sharding import make_single_node_mesh
    from wireles_jax.timestep_sharded import (
        initial_sharded_state,
        make_project_velocity_sharded,
        make_sharded_operators,
        make_step_ab2_sharded,
    )

    params = Params(
        nx=8,
        ny=8,
        nz=8,
        lx=2.0,
        ly=2.0,
        lz=1.0,
        dt=1.0e-4,
        nsteps=6,
        time_scheme="ab2",
        momentum_wall_model="free_slip",
        initial_velocity_noise=0.01,
        sgs_model="lasd",
        cs_count=2,
        dtype=jnp.float32,
    )
    mesh = make_single_node_mesh(1)
    ops = make_sharded_operators(params, mesh)
    initial = initial_sharded_state(params, mesh, seed=31)
    project = jax.jit(
        make_project_velocity_sharded(
            params, ops.pressure, mesh, spike_ops=ops.pressure_spike
        )
    )
    u, v, w, p = project(
        initial.u,
        initial.v,
        initial.w,
        ops.pressure,
        ops.pressure_spike,
    )
    initial = initial._replace(u=u, v=v, w=w, p=p)
    advance = jax.jit(make_step_ab2_sharded(params, ops, mesh))

    continuous = initial
    for _ in range(6):
        continuous = advance(
            continuous, ops.pressure, ops.pressure_spike
        )

    split = initial
    for _ in range(3):
        split = advance(split, ops.pressure, ops.pressure_spike)
    save_sharded_checkpoint(tmp_path, split, params, mesh, rank=0)
    resumed = load_sharded_checkpoint(
        tmp_path, replace(params, nsteps=3), mesh, rank=0
    )
    for _ in range(3):
        resumed = advance(resumed, ops.pressure, ops.pressure_spike)

    continuous, resumed = jax.block_until_ready((continuous, resumed))
    for name in continuous._fields:
        expected = np.asarray(jax.device_get(getattr(continuous, name)))
        actual = np.asarray(jax.device_get(getattr(resumed, name)))
        np.testing.assert_array_equal(actual, expected, err_msg=name)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["complete_restart_state"] is True
    assert manifest["restart_signature_sha256"]
    assert set(manifest["fields"]) == {
        name for name in ShardedFlowState._fields if name != "step"
    }
