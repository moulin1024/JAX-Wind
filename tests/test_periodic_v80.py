"""Direct periodic turbine runs must apply rotor forcing, not silently omit it."""
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.config.document import ResolvedCase, load_case
from jaxwind.simulation.api import RunControls, build_simulation
from jaxwind.turbine import _x_faces
from jaxwind.domain.grid import UniformGrid

ROOT = Path(__file__).resolve().parents[1]


def test_periodic_force_mapping_conserves_sum():
    grid = UniformGrid(8, 4, 4, 128., 64., 64.)
    values = jnp.arange(128, dtype=jnp.float32).reshape(4, 4, 8)
    faces = _x_faces(values, grid, periodic=True)
    assert faces.shape == values.shape
    np.testing.assert_allclose(faces.sum(), values.sum())
    np.testing.assert_allclose(faces[..., 0], .5 * (values[..., -1] + values[..., 0]))


def test_direct_periodic_v80_changes_momentum(monkeypatch):
    monkeypatch.setenv("JAXWIND_V80_FAST", str(ROOT / "cases/HornsRev1/turbines/V80/CustomRotor.fst"))
    source = load_case(ROOT / "cases/HornsRev1/fv_v80_periodic_smoke_512x512x256_1h.toml")
    doc = deepcopy(source.document)
    doc["mesh"] = {"cells": [16, 16, 256], "lengths_m": [256., 256., 1024.]}
    doc["physics"]["turbine"].update(x_m=128., y_m=128.)
    doc["time"].update(dt_seconds=.01, steps=1, frame_count=0)
    doc["diagnostics"].update(sample_start_step=0, sample_every_steps=1)
    active = build_simulation(ResolvedCase(source.source, doc))
    passive_doc = deepcopy(doc)
    del passive_doc["physics"]["turbine"]
    passive = build_simulation(ResolvedCase(source.source, passive_doc))
    controls = RunControls(1, .01)
    a = active.advance(active.initial_state, controls)
    b = passive.advance(passive.initial_state, controls)
    jax.block_until_ready((a, b))
    assert a.velocity.x.shape == (256, 16, 16)
    assert np.isfinite(np.asarray(a.velocity.x)).all()
    assert float(jnp.sum(a.velocity.x - b.velocity.x)) < -1e-3
