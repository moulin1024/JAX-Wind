from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from jaxwind.domain import (
    Accepted,
    AcceptedClock,
    AddressableField,
    Cell,
    DistributionSpec,
    EqualVerticalPartition,
    MeshAxis,
    MeshTopology,
    PassiveScalarConcentration,
    Projected,
    VerticalFaceField,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
    UniformGrid,
)
from jaxwind.effects import (
    HDF5PrecursorPlayback,
    HDF5PrecursorRecorder,
    JaxRuntime,
    PrecursorPlaybackConfig,
    PrecursorRecordingConfig,
    finalize_precursor_recording,
    run_main_with_precursor,
    run_precursor,
)
from jaxwind.integrators import AB2BoussinesqState, ColdStart, PreviousTendency
from jaxwind.operators import VelocityVector
from jaxwind.physics import BoussinesqFields


def _runtime(*, processes: int, process: int) -> JaxRuntime:
    return JaxRuntime(
        jax=SimpleNamespace(device_get=np.asarray),
        jnp=np,
        lax=object(),
        global_devices=processes,
        local_devices=1,
        process_count=processes,
        process_index=process,
        backend="test",
    )


def _global_value(grid: UniformGrid, step: int, offset: float) -> np.ndarray:
    coordinates = np.arange(grid.cell_count, dtype=np.float64).reshape(
        grid.nz, grid.ny, grid.nx
    )
    return coordinates + step * 10_000.0 + offset


def _state(runtime: JaxRuntime, step: int) -> AB2BoussinesqState:
    grid = UniformGrid(4, 3, 4, 8.0, 6.0, 4.0)
    decomposition = EqualVerticalPartition(
        grid,
        MeshTopology((MeshAxis("z", runtime.global_devices),)),
        DistributionSpec.vertical(),
    )
    partition = runtime.addressable_partitions[0]
    cell_region = (decomposition.regions(Cell)[partition],)
    face_region = (decomposition.regions(ZFace)[partition],)
    local_nz = decomposition.cells_per_partition

    def local(offset: float) -> np.ndarray:
        values = _global_value(grid, step, offset)
        return values.reshape(
            runtime.global_devices, local_nz, grid.ny, grid.nx
        )[partition : partition + 1]

    velocity = VelocityVector(
        AddressableField(XVelocity, Cell, cell_region, Projected, local(0.0)),
        AddressableField(YVelocity, Cell, cell_region, Projected, local(1_000.0)),
        VerticalFaceField(
            AddressableField(
                VerticalVelocity,
                ZFace,
                face_region,
                Projected,
                local(2_000.0),
            ),
            np.zeros((grid.ny, grid.nx)),
        ),
    )
    scalar = AddressableField(
        PassiveScalarConcentration,
        Cell,
        cell_region,
        Accepted,
        local(3_000.0),
    )
    return AB2BoussinesqState(
        BoussinesqFields(velocity, scalar),
        AcceptedClock(step * 0.5, step),
        PreviousTendency(object()),
        "test-integrator",
    )


def test_run_precursor_buffers_pre_step_sections_in_one_hdf5_file(tmp_path) -> None:
    runtime = _runtime(processes=1, process=0)
    path = tmp_path / "precursor.h5"

    def advance(state, *, compute_projection_residual):
        assert not compute_projection_residual
        return SimpleNamespace(state=_state(runtime, state.clock.step + 1))

    final = run_precursor(
        _state(runtime, 4),
        steps=3,
        advance=advance,
        path=path,
        runtime=runtime,
        recording=PrecursorRecordingConfig(sample_every=2, buffer_samples=2),
    )

    assert final.clock.step == 7
    with h5py.File(path, "r") as recording:
        assert recording.attrs["schema"] == "jaxwind.precursor-sections.v2"
        assert recording.attrs["storage"] == "single-file"
        assert bool(recording.attrs["complete"])
        np.testing.assert_array_equal(recording["step"], [4, 6])
        np.testing.assert_allclose(recording["time"], [2.0, 3.0])
        assert recording["velocity"].shape == (2, 2, 3, 4, 3, 1)
        assert recording["scalar"].shape == (2, 2, 4, 3, 1)
        grid = _state(runtime, 4).fields.velocity.x.regions[0].grid
        initial_u = _global_value(grid, 4, 0.0)
        np.testing.assert_array_equal(
            recording["velocity"][0, 0, 0, ..., 0], initial_u[:, :, 0]
        )
        np.testing.assert_array_equal(
            recording["velocity"][0, 1, 0, ..., 0], initial_u[:, :, -1]
        )
        initial_scalar = _global_value(grid, 4, 3_000.0)
        np.testing.assert_array_equal(
            recording["scalar"][0, 0, ..., 0], initial_scalar[:, :, 0]
        )


def test_run_precursor_requires_warm_ab2_history(tmp_path) -> None:
    runtime = _runtime(processes=1, process=0)
    state = _state(runtime, 4)
    cold = AB2BoussinesqState(
        state.fields,
        state.clock,
        ColdStart(),
        state.integrator_fingerprint,
    )

    with pytest.raises(ValueError, match="developed warm AB2"):
        run_precursor(
            cold,
            steps=1,
            advance=lambda _state, **_kwargs: None,
            path=tmp_path / "cold.h5",
            runtime=runtime,
        )


def test_rank_shards_publish_a_global_virtual_dataset_catalog(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "distributed.h5"
    runtimes = tuple(_runtime(processes=2, process=rank) for rank in range(2))
    for runtime in runtimes:
        with HDF5PrecursorRecorder(
            path,
            runtime=runtime,
            config=PrecursorRecordingConfig(buffer_samples=4),
        ) as recorder:
            recorder.record(_state(runtime, 8))

    monkeypatch.setattr(JaxRuntime, "synchronize", lambda _self, _name: None)
    finalize_precursor_recording(path, runtime=runtimes[0])

    assert (tmp_path / "distributed.process-00000.h5").exists()
    assert (tmp_path / "distributed.process-00001.h5").exists()
    with h5py.File(path, "r") as catalog:
        assert catalog.attrs["storage"] == "virtual-dataset-catalog"
        assert catalog["velocity"].is_virtual
        assert catalog["velocity"].shape == (1, 2, 3, 4, 3, 1)
        grid = _state(runtimes[0], 8).fields.velocity.x.regions[0].grid
        expected = _global_value(grid, 8, 0.0)
        np.testing.assert_array_equal(
            catalog["velocity"][0, 0, 0, ..., 0], expected[:, :, 0]
        )
        np.testing.assert_array_equal(
            catalog["velocity"][0, 1, 0, ..., 0], expected[:, :, -1]
        )
        assert sorted(catalog["shards"]) == [
            "process_00000",
            "process_00001",
        ]

    for rank, runtime in enumerate(runtimes):
        state = _state(runtime, 8)
        with HDF5PrecursorPlayback(
            path,
            runtime=runtime,
            state=state,
        ) as playback:
            target = playback.environment(state).velocity.x.payload
        grid = state.fields.velocity.x.regions[0].grid
        expected = _global_value(grid, 8, 0.0).reshape(2, 2, 3, 4)[rank, ..., 0]
        np.testing.assert_array_equal(target[..., 0], expected[None, ...])


def test_playback_reads_a_clock_matched_plane_and_broadcasts_it_on_device(
    tmp_path,
) -> None:
    runtime = _runtime(processes=1, process=0)
    path = tmp_path / "playback.h5"
    with HDF5PrecursorRecorder(path, runtime=runtime) as recorder:
        recorder.record(_state(runtime, 4))
        recorder.record(_state(runtime, 5))

    main = _state(runtime, 4)
    with HDF5PrecursorPlayback(
        path,
        runtime=runtime,
        state=main,
        config=PrecursorPlaybackConfig(section="inflow", buffer_samples=2),
    ) as playback:
        environment = playback.environment(main)

    target = environment.velocity
    expected = _global_value(target.x.regions[0].grid, 4, 0.0)[:, :, 0]
    np.testing.assert_array_equal(target.x.payload[..., 0], expected[None, ...])
    for x_index in range(target.x.regions[0].grid.nx):
        np.testing.assert_array_equal(
            target.x.payload[..., x_index],
            target.x.payload[..., 0],
        )


def test_playback_applies_a_periodic_spanwise_shift(tmp_path) -> None:
    runtime = _runtime(processes=1, process=0)
    path = tmp_path / "shifted-playback.h5"
    with HDF5PrecursorRecorder(path, runtime=runtime) as recorder:
        recorder.record(_state(runtime, 4))

    main = _state(runtime, 4)
    with HDF5PrecursorPlayback(
        path,
        runtime=runtime,
        state=main,
        config=PrecursorPlaybackConfig(spanwise_shift_cells=1),
    ) as playback:
        target = playback.environment(main).velocity.x.payload[..., 0]

    grid = main.fields.velocity.x.regions[0].grid
    expected = _global_value(grid, 4, 0.0)[:, :, 0]
    np.testing.assert_array_equal(target, np.roll(expected[None, ...], 1, axis=-1))


def test_multiplane_sections_preserve_distinct_x_values_and_sample_cadence(
    tmp_path,
) -> None:
    runtime = _runtime(processes=1, process=0)
    path = tmp_path / "slab.h5"
    config = PrecursorRecordingConfig(
        sample_every=2,
        section_width=2,
        inflow_start_index=1,
    )
    with HDF5PrecursorRecorder(path, runtime=runtime, config=config) as recorder:
        for step in range(4, 8):
            recorder.record(_state(runtime, step))

    with h5py.File(path, "r") as recording:
        assert int(recording.attrs["sample_every"]) == 2
        assert int(recording.attrs["section_width"]) == 2
        np.testing.assert_array_equal(recording["step"], [4, 6])
        np.testing.assert_array_equal(recording["sections/x_index"], [[1, 2], [2, 3]])
        grid = _state(runtime, 4).fields.velocity.x.regions[0].grid
        expected = _global_value(grid, 4, 0.0)
        np.testing.assert_array_equal(
            recording["velocity"][0, 0, 0], expected[:, :, 1:3]
        )

    main = _state(runtime, 5)
    with HDF5PrecursorPlayback(path, runtime=runtime, state=main) as playback:
        assert playback.covered_steps == 4
        target = playback.environment(main).velocity.x.payload
    expected = _global_value(grid, 4, 0.0)[:, :, 1:3]
    np.testing.assert_array_equal(target[..., :2], expected[None, ...])
    np.testing.assert_array_equal(target[..., 2:], 0.0)


def test_compiled_style_batch_accepts_nonunit_recording_cadence(tmp_path) -> None:
    runtime = _runtime(processes=1, process=0)
    path = tmp_path / "compiled-cadence.h5"
    config = PrecursorRecordingConfig(sample_every=2, buffer_samples=2)
    state = _state(runtime, 4)
    with HDF5PrecursorRecorder(path, runtime=runtime, config=config) as recorder:
        velocity, scalar = recorder._initialize(state)
        first = recorder._extract_sections(velocity, scalar)
        velocity_6, scalar_6 = recorder._extract_sections(
            _state(runtime, 6).fields.velocity,
            _state(runtime, 6).fields.potential_temperature,
        )
        recorder.record_batch(
            state,
            np.stack((first[0], velocity_6)),
            np.stack((first[1], scalar_6)),
            steps=np.asarray((4, 6)),
            times=np.asarray((2.0, 3.0)),
        )

    with h5py.File(path, "r") as recording:
        np.testing.assert_array_equal(recording["step"], [4, 6])


def test_main_runner_supplies_each_clock_matched_environment(tmp_path) -> None:
    runtime = _runtime(processes=1, process=0)
    path = tmp_path / "main-playback.h5"
    with HDF5PrecursorRecorder(path, runtime=runtime) as recorder:
        recorder.record(_state(runtime, 4))
        recorder.record(_state(runtime, 5))

    seen = []

    def advance(state, *, environment, compute_projection_residual):
        assert not compute_projection_residual
        seen.append((state.clock.step, environment.velocity.x.payload[..., 0].copy()))
        return SimpleNamespace(state=_state(runtime, state.clock.step + 1))

    main = _state(runtime, 4)
    with HDF5PrecursorPlayback(
        path,
        runtime=runtime,
        state=main,
    ) as playback:
        final = run_main_with_precursor(
            main,
            steps=2,
            advance=advance,
            playback=playback,
        )

    assert final.clock.step == 6
    assert [step for step, _target in seen] == [4, 5]
    assert not np.array_equal(seen[0][1], seen[1][1])
