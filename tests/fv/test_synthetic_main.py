"""Measured-profile calibration and direct turbine runtime coverage."""

from copy import deepcopy

import jax
import numpy as np
import pytest

from jaxwind import UniformGrid, build_mann_inflow, generate_mann_box
from jaxwind.config.document import ResolvedCase, load_case
from jaxwind.config.synthetic_inflow import reference_profile
from jaxwind.runtime.engine import resume, run

CASE = "cases/HITSZWindTunnel/fv_mann_512x128x256_adbem_90s.toml"


def test_calibrated_profile_over_complete_box():
    grid = UniformGrid(nx=8, ny=8, nz=8, lx=24.0, ly=6.0, lz=3.6)
    box = generate_mann_box(
        shape=(16, 8, 8), lengths=(24.0, 6.0, 3.6), length_scale=0.5
    )
    mean = np.linspace(2.0, 4.0, 8)
    sigma = mean * np.linspace(0.08, 0.12, 8)
    inflow = build_mann_inflow(
        box, grid, mean_speed=3.0, mean_profile=mean, sigma_u_profile=sigma
    )
    times = np.arange(16 * 16) * (24 / 3) / (16 * 16)
    planes = np.asarray(jax.jit(jax.vmap(inflow))(times).x_velocity)
    np.testing.assert_allclose(planes.mean(axis=(0, 2)), mean, atol=2e-6)
    np.testing.assert_allclose(planes.std(axis=(0, 2)), sigma, rtol=0.003)


def test_measured_hub_values():
    case = load_case(CASE)
    z, u, ti = reference_profile(
        case.document["physics"]["inflow"]["reference_profile"]
    )
    assert np.interp(0.876, z, u) == pytest.approx(3.3108)
    assert np.interp(0.876, z, ti) == pytest.approx(0.093492)


def test_direct_adbem_run_and_resume(tmp_path):
    case = load_case(CASE)
    doc = deepcopy(case.document)
    doc["case"]["profile_resampling"] = "linear"
    doc["mesh"]["cells"] = [16, 8, 16]
    doc["physics"]["inflow"]["box_cells"] = [32, 8, 16]
    doc["physics"]["inflow"]["box_lengths_m"] = [40.0, 6.0, 3.6]
    doc["time"].update(steps=4, frame_count=0, chunk_steps=2)
    doc["diagnostics"].update(sample_start_step=0, sample_every_steps=1)
    case = ResolvedCase(case.source, doc)
    first = run(case, output=tmp_path / "run", max_steps=2)
    assert first.summary["step"] == 2
    last = resume(tmp_path / "run")
    assert last.summary["status"] == "complete"
    assert last.summary["step"] == 4
    assert np.isfinite(last.summary["final_cfl"])
