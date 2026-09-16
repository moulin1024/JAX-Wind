"""Guard reference reconstruction and wall-bounded benchmark integration."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jaxwind.config.document import load_case
from jaxwind.config.moisture import load_moisture
from jaxwind.simulation.water_spray_benchmark import (
    equivalent_diameter,
    inlet_mixing_ratio,
)

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "cases/WaterSprayMontazeri2015/coarse.toml"


def test_mass_distribution_preserves_area_per_liquid_mass():
    doc = load_case(CASE).document
    r = doc["case"]["reference"]
    d = np.linspace(r["minimum_diameter_m"], r["maximum_diameter_m"], 100001)
    f = 1 - np.exp(-((d / r["rosin_rammler_scale_m"]) ** r["rosin_rammler_spread"]))
    weights = np.diff(f)
    weights /= weights.sum()
    direct = 1 / np.sum(weights / ((d[1:] + d[:-1]) / 2))
    np.testing.assert_allclose(equivalent_diameter(r), direct, rtol=1.0e-8)
    assert abs(direct * 1.0e6 - 293) < 1


def test_psychrometric_boundary_is_dry_and_finite():
    doc = load_case(CASE).document
    moist, _ = load_moisture(doc["physics"])
    q = inlet_mixing_ratio(doc["case"]["reference"], moist.thermodynamics)
    assert 0.004 < q < 0.006


def test_buoyancy_supports_distinct_sidewall_faces():
    from jaxwind.buoyancy import LinearBoussinesqBuoyancy, boussinesq_tendency

    field = jnp.arange(4 * 6 * 8, dtype=jnp.float32).reshape(4, 6, 8)
    force = boussinesq_tendency(
        field, LinearBoussinesqBuoyancy(0.03), x_face_count=9, y_face_count=7
    )
    assert force.x.shape == (4, 6, 9)
    assert force.y.shape == (4, 7, 8)
    assert force.z.shape == (5, 6, 8)
    assert jnp.all(force.y == 0)
    np.testing.assert_allclose(jnp.mean(force.z[1:-1], axis=(1, 2)), 0, atol=1.0e-7)


def test_turbulent_inlet_preserves_bulk_flux_and_prescribed_energy():
    import jax

    from jaxwind import InflowPlane
    from jaxwind.domain import UniformGrid
    from jaxwind.simulation.water_spray_benchmark import (
        build_benchmark_inlet,
        inlet_kinetic_energy,
    )

    jax.config.update("jax_enable_x64", True)
    grid = UniformGrid(8, 8, 8, 1.9, 0.585, 0.585)
    plane = InflowPlane(
        jnp.full((8, 8), 3.0), jnp.zeros((8, 9)), jnp.zeros((9, 8)), jnp.zeros((8, 8))
    )
    settings = {
        "normalization": "total_energy",
        "intensity": 0.1,
        "length_scale_m": 0.04095,
        "box_cells": [32, 12, 12],
        "box_lengths_m": [1.2, 0.585, 0.585],
        "seed": 12,
    }
    inlet = build_benchmark_inlet(grid, 3.0, plane, settings, 0.1)
    times = jnp.arange(2048) * 0.4 / 2048
    energies = jax.jit(jax.vmap(lambda t: inlet_kinetic_energy(inlet(t), 3.0)))(times)
    np.testing.assert_allclose(
        float(jnp.mean(jnp.sum(energies, axis=1))), 0.09, rtol=0.001
    )
    for t in [0.0, 0.071, 0.173]:
        value = inlet(t)
        np.testing.assert_allclose(jnp.mean(value.x_velocity), 3.0, atol=1e-14)
        np.testing.assert_array_equal(value.y_velocity[:, [0, -1]], 0.0)
        np.testing.assert_array_equal(np.asarray(value.z_velocity)[[0, -1]], 0.0)
    assert not np.allclose(inlet(0).x_velocity, inlet(0.071).x_velocity)


def test_original_sensor_table_agrees_with_independent_figure_digitization():
    import json

    directory = CASE.parent
    table = json.loads((directory / "reference_table.json").read_text())
    figure = json.loads((directory / "reference.json").read_text())
    x0, x1 = figure["axis_x_pixels"]
    t0, t1 = figure["axis_temperature_c"]
    digitized = t0 + (np.asarray(figure["marker_x_pixels"]) - x0) / (x1 - x0) * (
        t1 - t0
    )
    np.testing.assert_allclose(
        sorted(table["experimental_dbt_c"]), sorted(digitized), atol=0.1, rtol=0
    )
    assert len(table["experimental_dbt_c"]) == 9
    assert table["experimental_dbt_c"][4] == 31.4  # Original middle-center DBT.
