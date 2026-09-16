"""An internal nitrogen source can be carried by a prescribed ambient inlet."""
from dataclasses import replace
from pathlib import Path

import numpy as np

from jaxwind.config.jet import load_case
from jaxwind.simulation.jet import build_simulation

CASE = Path(__file__).resolve().parents[2] / 'cases/HITSZWindTunnel/fv_512x128x256_l24_jet_inflow5_adaptive_cfl0p6_1s.toml'


def test_uniform_initial_airflow_and_inlet_survive_projection():
    case = load_case(CASE)
    assert case.ambient_streamwise_velocity == 5.0
    assert case.streamwise_boundaries == 'inflow-outflow'
    assert case.source_mode == 'volume'
    assert case.cfl == .6
    assert case.steps * case.dt == 1.0
    small = replace(case, cells=(16, 8, 8), dt=.001, cfl=None, steps=2)
    _, _, _, state, advance, _ = build_simulation(small)
    np.testing.assert_allclose(state.velocity.x, 5.0)
    np.testing.assert_allclose(state.velocity.y, 0.0)
    np.testing.assert_allclose(state.velocity.z, 0.0)
    result = advance(state, 2)
    np.testing.assert_allclose(result.velocity.x[..., 0], 5.0, atol=1e-6)
    assert np.all(np.asarray(result.velocity.x[..., -1]) > 0.0)
    np.testing.assert_allclose(result.velocity.y[:, (0, -1), :], 0.0, atol=1e-6)
    assert np.isfinite(np.asarray(result.temperature)).all()
    assert float(result.nitrogen_density.sum()) > 0.0
    assert float(result.continuity_error) < 1e-3
