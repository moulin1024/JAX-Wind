from copy import deepcopy
from pathlib import Path

import pytest

from jaxwind.config.document import ResolvedCase, load_case
from jaxwind.config.moisture import load_moisture
from jaxwind.config.stages import load_workflow

CASE = (
    Path(__file__).resolve().parents[1] / "cases/HITSZWindTunnel/fv_far_wake_water.toml"
)


def test_water_case_resolves_and_ln2_default_is_unchanged():
    workflow = load_workflow(CASE)
    assert workflow.water_spray.mass_flow_rate_kg_s == 0.001
    assert workflow.moisture.ambient_relative_humidity == 0.5
    assert workflow.cooling is None
    assert workflow.resolved()["water_spray"]["droplet_diameter_m"] == 20.0e-6
    ln2 = load_workflow(CASE.with_name("fv_far_wake_cooled.toml"))
    assert ln2.moisture is None
    assert ln2.cooling.cooling_power_w > 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("ambient_relative_humidity", 1.1),
        ("pressure_pa", 0),
        ("temperature_offset_k", float("nan")),
        ("unknown", 1),
    ],
)
def test_invalid_moisture_settings_are_rejected(key, value):
    with pytest.raises(ValueError):
        load_moisture({"moisture": {key: value}})


@pytest.mark.parametrize(
    "key,value",
    [
        ("mass_flow_rate_kg_s", -1),
        ("droplet_diameter_m", 0),
        ("standard_deviation_m", [1, float("nan"), 1]),
        ("ramp_time_s", -1),
        ("streamwise_offset_m", 0),
        ("unknown", 1),
    ],
)
def test_invalid_spray_settings_are_rejected(key, value):
    table = {
        "mass_flow_rate_kg_s": 0.01,
        "streamwise_offset_m": 0.1,
        "standard_deviation_m": [0.1, 0.1, 0.1],
        key: value,
    }
    with pytest.raises(ValueError):
        load_moisture({"moisture": {}, "water_spray": table})


def test_spray_requires_humidity():
    with pytest.raises(ValueError, match="requires"):
        load_moisture({"water_spray": {}})


@pytest.mark.parametrize("change", ["scalar", "turbine", "offset", "ab2", "cooling"])
def test_invalid_coupling_fails_early(change):
    base = load_case(CASE)
    doc = deepcopy(base.document)
    if change == "scalar":
        doc["workflow"]["evolve_scalar"] = False
    elif change == "turbine":
        del doc["physics"]["turbine"]
    elif change == "offset":
        doc["physics"]["water_spray"]["streamwise_offset_m"] = 1000
    elif change == "ab2":
        doc["numerics"]["time_integration"] = "ab2"
    else:
        doc["physics"]["cooling"] = load_case(
            CASE.with_name("fv_far_wake_cooled.toml")
        ).document["physics"]["cooling"]
    with pytest.raises(ValueError):
        load_workflow(ResolvedCase(base.source, doc))
