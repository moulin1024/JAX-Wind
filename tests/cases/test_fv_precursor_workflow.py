from __future__ import annotations

from pathlib import Path

import pytest

from applications.fv_abl.workflow import load_workflow


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "cases" / "Andren1994" / "config.toml"
DTU_CONFIG = ROOT / "cases" / "DTU10MWPrecursor" / "fv_workflow.toml"
HITSZ_CONFIG = ROOT / "cases" / "HITSZWindTunnel" / "fv_workflow.toml"


def test_one_toml_resolves_the_three_distinct_fv_stages() -> None:
    workflow = load_workflow(CONFIG)
    result = workflow.resolved()

    assert result["warmup"]["pressure_backend"] == "fft"
    assert result["warmup"]["periodic_x"] is True
    assert result["precursor"]["pressure_backend"] == "fft"
    assert result["precursor"]["stored_x_layers_per_sample"] == 1
    assert result["precursor"]["sample_every_steps"] == 1
    assert result["main"]["pressure_backend"] == "gmg"
    assert result["main"]["periodic_x"] is False
    assert result["main"]["x_velocity_faces"] == 41
    assert result["main"]["outflow"].startswith("second-order")


def test_dtu_configuration_adds_adbem_only_to_the_open_main_stage() -> None:
    workflow = load_workflow(DTU_CONFIG)
    result = workflow.resolved()

    assert result["warmup"]["pressure_backend"] == "fft"
    assert result["precursor"]["stored_x_layers_per_sample"] == 1
    assert result["main"]["pressure_backend"] == "gmg"
    assert result["main"]["pressure_force"] is False
    assert result["main"]["x_velocity_faces"] == 129
    assert result["turbine"]["model"] == "openfast-ad-bem"
    assert result["turbine"]["rotor_speed_rpm"] == 9.6


def test_main_cannot_outlive_the_recorded_precursor(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "main_steps = 4500",
            "main_steps = 4501",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="main_steps cannot exceed"):
        load_workflow(invalid)


def test_hitsz_main_is_fixed_fast_rk3_with_native_turbine_and_frames() -> None:
    result = load_workflow(HITSZ_CONFIG).resolved()

    assert result["main"]["dt_seconds"] == 0.01125
    assert result["main"]["duration_seconds"] == 90.0
    assert result["main"]["time_integration"] == "fast-rk3"
    assert result["main"]["frame_count"] == 100
    assert result["turbine"]["model"] == "hitsz-r9-ad-bem"
    assert result["turbine"]["rotor_speed_rpm"] == 480.0


def test_main_frame_capture_matches_full_field_host_interpolation() -> None:
    from types import SimpleNamespace

    import jax.numpy as jnp
    import numpy as np

    from applications.fv_abl.workflow import _capture_main_frame
    from jaxwind.domain import UniformGrid

    grid = UniformGrid(8, 4, 6, 8.0, 4.0, 3.0)
    x_faces = jnp.arange(
        grid.nz * grid.ny * (grid.nx + 1), dtype=jnp.float32
    ).reshape(grid.nz, grid.ny, grid.nx + 1)
    solution = SimpleNamespace(
        velocity=SimpleNamespace(x=x_faces),
        time=jnp.asarray(1.25),
        step=jnp.asarray(5),
    )
    y_m = 1.7
    z_m = 1.1
    result = _capture_main_frame(solution, grid, y_m=y_m, z_m=z_m)

    host_faces = np.asarray(x_faces)
    u_cell = 0.5 * (host_faces[..., :-1] + host_faces[..., 1:])
    z_index = np.clip(z_m / grid.dz - 0.5, 0.0, grid.nz - 1.0)
    z_lower = int(np.floor(z_index))
    z_upper = min(z_lower + 1, grid.nz - 1)
    z_weight = z_index - z_lower
    expected_hub = (
        (1.0 - z_weight) * u_cell[z_lower] + z_weight * u_cell[z_upper]
    )
    y_index = y_m / grid.dy - 0.5
    y_floor = np.floor(y_index)
    y_lower = int(y_floor) % grid.ny
    y_upper = (y_lower + 1) % grid.ny
    y_weight = y_index - y_floor
    expected_centre = (
        (1.0 - y_weight) * u_cell[:, y_lower]
        + y_weight * u_cell[:, y_upper]
    )

    np.testing.assert_allclose(result["u_hub_yx"], expected_hub, rtol=1e-6)
    np.testing.assert_allclose(
        result["u_center_zx"], expected_centre, rtol=1e-6
    )
    assert result["time_seconds"] == 1.25
    assert result["step"] == 5
