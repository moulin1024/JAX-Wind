from pathlib import Path

import pytest

from jaxwind.effects import JaxRuntime


def runtime(*, processes: int, process: int, local_devices: int) -> JaxRuntime:
    return JaxRuntime(
        jax=object(),
        jnp=object(),
        lax=object(),
        global_devices=processes * local_devices,
        local_devices=local_devices,
        process_count=processes,
        process_index=process,
        backend="test",
    )


def test_one_process_is_the_degenerate_runtime_topology() -> None:
    job = runtime(processes=1, process=0, local_devices=1)
    value = object()

    assert job.is_primary
    assert job.addressable_partitions == (0,)
    assert job.global_array(value) is value
    assert job.checkpoint_path(Path("state.npz")) == Path("state.npz")


def test_process_placement_and_checkpoint_names_are_runtime_effects() -> None:
    job = runtime(processes=4, process=2, local_devices=2)

    assert not job.is_primary
    assert job.addressable_partitions == (4, 5)
    assert job.checkpoint_path(Path("state.npz")) == Path(
        "state.process-00002.npz"
    )


def test_heterogeneous_process_device_counts_fail_at_the_effect_boundary() -> None:
    with pytest.raises(ValueError, match="same number"):
        JaxRuntime(
            jax=object(),
            jnp=object(),
            lax=object(),
            global_devices=3,
            local_devices=2,
            process_count=2,
            process_index=0,
            backend="test",
        )
