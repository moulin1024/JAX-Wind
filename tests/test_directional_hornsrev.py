"""Direction geometry and shared-precursor amplitude-scaling contracts."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from jaxwind.config.document import load_case
from jaxwind.io.recording import write_chunk, write_manifest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hornsrev_generator", ROOT / "tools/create_hornsrev_case.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


@pytest.mark.parametrize("direction", [0., 30., 60., 90., 120., 150., 180., 210., 240., 270., 300., 330., 275.])
def test_rotation_preserves_distances_and_center(direction):
    source = load_case(ROOT / "cases/HornsRev1/fv_hornsrev1_80_uniform10_open_gmg.toml")
    layout = source.document["physics"]["wind_farm"]["layout"]
    result = generator.rotate_layout(layout, direction, [8192., 8192., 1024.])
    original = np.array([(r["x_m"], r["y_m"]) for r in layout])
    rotated = np.array([(r["x_m"], r["y_m"]) for r in result])
    np.testing.assert_allclose(rotated.mean(0), [4096., 4096.], atol=1.e-10)
    np.testing.assert_allclose(np.linalg.norm(original[:, None]-original[None], axis=-1),
                               np.linalg.norm(rotated[:, None]-rotated[None], axis=-1), atol=1.e-9)
    assert [r["id"] for r in result] == [r["id"] for r in layout]
    if direction == 270.:
        np.testing.assert_array_equal(rotated, original)
    if direction == 0.:
        np.testing.assert_array_equal(rotated, np.column_stack((8192-original[:, 1], original[:, 0])))


def test_directions_share_reference_and_no_warmup_in_main(tmp_path):
    from jaxwind.workflows.engine import check_workflow
    reference = None
    for angle in (270., 0., 30., 275.):
        result = generator.generate(angle, directory=tmp_path / "cases", run_root=tmp_path / "runs")
        if reference is None:
            reference = result["reference_workflow"]
        assert result["reference_workflow"] == reference
        main = check_workflow(result["main_workflow"])
        assert main["order"] == ["main"]
        assert main["stages"]["main"]["time"]["steps"] * .25 == 3600.
    ref = check_workflow(reference)
    assert ref["stages"]["warmup"]["time"]["steps"] * ref["stages"]["warmup"]["time"]["dt_seconds"] == 36000.
    assert ref["stages"]["precursor"]["time"]["steps"] * .25 == 3600.
    assert ref["stages"]["warmup"]["time"]["cfl"] == .9
    assert ref["stages"]["warmup"]["time"]["checkpoint_every_seconds"] == 3600.
    assert "cfl" not in ref["stages"]["precursor"]["time"]
    assert "cfl" not in main["stages"]["main"]["time"]
    assert result["selected_wind_rose_sector_deg"] == 270.
    # Regeneration is idempotent, but differing assets must not be overwritten.
    profile = Path(reference).parent / "initial_profile.csv"
    profile.write_text("changed by user")
    with pytest.raises(FileExistsError):
        generator.generate(270., directory=tmp_path / "cases", run_root=tmp_path / "runs")


def test_recorded_mean_is_time_weighted(tmp_path):
    from jaxwind import UniformGrid
    grid = UniformGrid(4, 3, 2, 4., 3., 2.)
    outputs = {"x_velocity": np.stack((np.full((2, 3), 2.), np.full((2, 3), 6.))),
        "y_velocity": np.zeros((2, 2, 3)), "z_velocity": np.zeros((2, 3, 3)),
        "scalar": np.zeros((2, 2, 3)), "time_seconds": np.array([0., .25]),
        "dt_seconds": np.array([.25, .75])}
    chunk = write_chunk(tmp_path, 0, outputs)
    write_manifest(tmp_path, grid, [chunk])
    meta = json.loads((tmp_path / "metadata.json").read_text())
    np.testing.assert_allclose(meta["mean_x_velocity_profile_m_s"], [5., 5.])
    assert meta["duration_seconds"] == 1.


def test_replay_scales_all_velocities_not_scalar_or_time():
    import jax.numpy as jnp
    from jaxwind import InflowPlane
    from jaxwind.simulation.recorded_farm import open_plane
    plane = InflowPlane(jnp.ones((2, 3, 4)), jnp.full((2, 3, 4), 2.),
                        jnp.full((2, 4, 4), 3.), jnp.full((2, 3, 4), 280.))
    result = open_plane(plane, .75)
    np.testing.assert_allclose(result.x_velocity, .75)
    np.testing.assert_allclose(result.y_velocity, 1.5)
    np.testing.assert_allclose(result.z_velocity, 2.25)
    np.testing.assert_array_equal(result.scalar, plane.scalar)
    assert result.y_velocity.shape == (2, 3, 5)


def test_muscl_is_now_default():
    from jaxwind import FlowModel
    from jaxwind.config.abl import load_fv_abl
    assert FlowModel().momentum_advection_scheme == "muscl-mc"
    case = load_case(ROOT / "cases/HornsRev1/fv_000deg_precursor.toml")
    case.document["numerics"].pop("momentum_advection_scheme")
    assert load_fv_abl(case).options.momentum_advection_scheme == "muscl-mc"
