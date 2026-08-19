from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from applications.windfarm_precursor.visualization import (
    evenly_spaced_frame_offsets,
    save_flow_frames,
    write_flow_gif,
)
from jaxwind.domain import UniformGrid


def test_frame_schedule_has_exact_endpoints_and_count() -> None:
    offsets = evenly_spaced_frame_offsets(36_000, 100)

    assert len(offsets) == 100
    assert offsets[0] == 0
    assert offsets[-1] == 36_000
    assert all(right > left for left, right in zip(offsets, offsets[1:]))


def test_two_frames_are_saved_and_rendered(tmp_path) -> None:
    grid = UniformGrid(4, 3, 2, 40.0, 30.0, 20.0)
    first = np.arange(8, dtype=np.float32).reshape(2, 4)
    frames = [first, first + 1.0]
    times = [0.0, 1.0]

    archive = save_flow_frames(tmp_path / "frames.npz", frames, times)
    gif = write_flow_gif(
        tmp_path / "flow.gif",
        frames,
        times,
        grid=grid,
        inlet_end_x_m=10.0,
        fps=2,
        turbine=SimpleNamespace(
            x_m=20.0,
            hub_height_m=10.0,
            rotor_diameter_m=8.0,
        ),
        equal_physical_scale=True,
    )

    with np.load(archive) as values:
        assert values["u_xz_m_s"].shape == (2, 2, 4)
    assert gif.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
