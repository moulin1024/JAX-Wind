"""Check the interactive runner across repeated donation of evolving state."""
from copy import deepcopy
from pathlib import Path

import jax
import numpy as np

from jaxwind.config.document import ResolvedCase, load_case
from jaxwind.simulation.api import RunControls, build_simulation
from tools.run_jet_interactive import with_donated_state


def test_reused_buffers_match_ordinary_consecutive_blocks():
    root = Path(__file__).resolve().parents[2]
    case = load_case(root / "cases/HITSZWindTunnel/fv_1024x256x512_l24_jet_only_two_outlets_1s.toml")
    document = deepcopy(case.document)
    document["mesh"]["cells"] = [16, 8, 8]
    case = ResolvedCase(case.source, document)
    ordinary = build_simulation(case)
    donated = with_donated_state(build_simulation(case))
    expected, actual = ordinary.initialize(), donated.initialize()
    for _ in range(2):
        expected = ordinary.advance(expected, RunControls(2, 1.0))
        actual = donated.advance(actual, RunControls(2, 1.0))
        jax.block_until_ready(actual)
        for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-5, atol=1e-6)
