"""Exercise transported carrier and conservative inertial parcels together."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.config.document import load_case
from jaxwind.io.checkpoint import load_checkpoint, save_checkpoint
from jaxwind.rans_kepsilon import RANSInertialSolution
from jaxwind.simulation.api import RunControls
from jaxwind.simulation.water_spray_benchmark import build_simulation


@pytest.mark.parametrize("transport", ["upwind", "muscl-mc"])
@pytest.mark.parametrize("model", ["standard-k-epsilon", "realizable-k-epsilon"])
def test_rans_coupling_advances_and_checkpoint_restores(tmp_path, model, transport):
    root = Path(__file__).resolve().parents[2]
    case = load_case(
        root
        / "cases/WaterSprayMontazeri2015/centre_investigation/standard_kepsilon_case3.toml"
    )
    doc = deepcopy(case.document)
    doc["case"]["carrier_turbulence_model"] = model
    doc["numerics"]["turbulence_advection_scheme"] = transport
    doc["mesh"]["cells"] = [8, 8, 8]
    doc["case"]["parcel_capacity"] = 32
    doc["case"]["parcels_per_step"] = 4
    doc["case"]["parcel_substeps"] = 1
    simulation = build_simulation(replace(case, document=doc))
    initial = simulation.initial_state
    assert isinstance(initial, RANSInertialSolution)
    result = simulation.advance(initial, RunControls(count=2, target_time=0.0005))
    assert int(result.step) == 2
    assert float(result.time) == 0.0005
    assert float(result.parcels.injected_mass) > 0
    assert bool(jnp.all(jnp.isfinite(jnp.stack(result.turbulence))))
    assert bool(jnp.all(jnp.stack(result.turbulence) > 0))
    # Total prescribed RANS energy is not duplicated as stochastic inlet energy.
    diagnostic = simulation.state_diagnostics(result)
    assert float(diagnostic["inlet_resolved_k_m2_s2"]) == 0
    assert abs(float(diagnostic["parcel_enthalpy_balance_error_j"])) < 1e-9
    assert abs(float(diagnostic["parcel_mass_balance_error_kg"])) < 1e-14
    checkpoint = tmp_path / "rans.npz"
    save_checkpoint(checkpoint, result, metadata={"fingerprint": "audit"})
    restored, _, _ = load_checkpoint(checkpoint, initial, fingerprint="audit")
    for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(a, b)
