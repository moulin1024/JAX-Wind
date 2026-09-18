"""Physical invariants for the coupled spray on a centrally clustered mesh."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.config.document import load_case
from jaxwind.config.spray_mesh import build_spray_grid
from jaxwind.cryogenic import _cic_coordinates, _cic_sample
from jaxwind.domain import AnalyticalGrid, UniformGrid
from jaxwind.open_boundary import InflowPlane
from jaxwind.runtime.frames import build_frame_capture
from jaxwind.simulation.api import RunControls
from jaxwind.simulation.water_spray_benchmark import (
    build_benchmark_inlet,
    build_simulation,
)
from jaxwind.state import StaggeredVelocity

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[2]


def mesh(strength=2.0):
    return {
        "cells": [8, 8, 8],
        "lengths_m": [1.9, 0.585, 0.585],
        "mapping": {
            "types": ["sinh"] * 3,
            "focus_m": [0, 0.2925, 0.2925],
            "strength": [strength] * 3,
        },
    }


def test_mapped_faces_volume_coarsening_and_uniform_compatibility():
    grid = build_spray_grid(mesh())
    assert isinstance(grid, AnalyticalGrid)
    assert grid.x_widths[0] < grid.x_widths[-1]
    assert grid.y_widths[3] < grid.y_widths[0]
    np.testing.assert_allclose(grid.y_widths, grid.y_widths[::-1], rtol=1e-13)
    assert np.sum(grid.cell_volumes) == pytest.approx(1.9 * 0.585**2)
    for a in "xyz":
        np.testing.assert_array_equal(
            getattr(grid.coarsen((2, 2, 2)), a + "_faces"),
            getattr(grid, a + "_faces")[::2],
        )
    assert isinstance(build_spray_grid(mesh(0)), UniformGrid)
    with pytest.raises(ValueError, match="spray_model=inertial"):
        build_spray_grid(mesh(), spray_model="fluent-dpm")
    bad = mesh()
    bad["mapping"]["focus_m"][0] = -1
    with pytest.raises(ValueError):
        build_spray_grid(bad)


def test_physical_particle_and_frame_interpolation_of_affine_fields():
    grid = build_spray_grid(mesh())
    z, y, x = np.meshgrid(grid.z_centers, grid.y_centers, grid.x_centers, indexing="ij")
    field = jnp.asarray(2 * x - 3 * y + 4 * z)
    px = jnp.array([0.3, 0.7, 1.2])
    py = jnp.array([0.12, 0.31, 0.45])
    pz = jnp.array([0.18, 0.34, 0.42])
    np.testing.assert_allclose(
        _cic_sample(field, _cic_coordinates(px, py, pz, grid)),
        2 * px - 3 * py + 4 * pz,
        atol=1e-13,
    )
    # Off-centre plane intentionally distinguishes physical from nominal indices.
    yp, zp = 0.22, 0.27
    uf = jnp.ones((grid.nz, grid.ny, grid.nx + 1))
    _, _, hub, centre = build_frame_capture(grid, y_m=yp, z_m=zp)(uf, field)
    np.testing.assert_allclose(
        centre,
        2 * grid.x_centers[None, :] - 3 * yp + 4 * grid.z_centers[:, None],
        atol=1e-13,
    )
    np.testing.assert_allclose(
        hub,
        2 * grid.x_centers[None, :] - 3 * grid.y_centers[:, None] + 4 * zp,
        atol=1e-13,
    )


def test_stretched_inlet_preserves_bulk_flux_and_total_turbulent_energy():
    grid = build_spray_grid(mesh())
    uniform = InflowPlane(
        jnp.full((8, 8), 3.0), jnp.zeros((8, 9)), jnp.zeros((9, 8)), jnp.zeros((8, 8))
    )
    settings = {
        "intensity": 0.1,
        "length_scale_m": 0.04095,
        "box_cells": [16, 8, 8],
        "box_lengths_m": [2.0, 0.585, 0.585],
        "seed": 7,
        "normalization": "total_energy",
    }
    inlet = build_benchmark_inlet(grid, 3.0, uniform, settings, 0.01)
    area = grid.z_widths[:, None] * grid.y_widths[None, :]
    times = (
        (np.arange(16)[:, None] + [0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)])
        * (2 / 3)
        / 16
    ).ravel()

    def energy(t):
        p = inlet(t)
        v = 0.5 * (p.y_velocity[:, :-1] + p.y_velocity[:, 1:])
        w = 0.5 * (p.z_velocity[:-1] + p.z_velocity[1:])
        return (
            jnp.stack(
                [
                    jnp.sum((p.x_velocity - 3) ** 2 * area),
                    jnp.sum((v * v + w * w) * area),
                ]
            )
            / area.sum()
            / 2
        )

    k = np.asarray(jax.jit(jax.vmap(energy))(jnp.asarray(times)))[:, 0].mean()
    k += np.asarray(
        jax.jit(jax.vmap(energy))(jnp.asarray(times) + grid.x_centers[0] / 3)
    )[:, 1].mean()
    assert k == pytest.approx(0.09, abs=1e-12)
    for t in [0.013, 0.073, 0.219]:
        p = inlet(t)
        assert float(jnp.sum(p.x_velocity * area) / area.sum()) == pytest.approx(
            3, abs=1e-12
        )
    assert abs(float(jnp.mean(inlet(0.073).x_velocity)) - 3) > 1e-5


def test_stretched_coupled_parcels_conserve_water_and_liquid_energy(tmp_path):
    case = load_case(
        ROOT
        / "cases/WaterSprayMontazeri2015/centre_investigation/central_clustered_case3.toml"
    )
    doc = deepcopy(case.document)
    doc["mesh"] = mesh(1.5)
    doc["case"].update(parcel_capacity=64, parcels_per_step=4, parcel_substeps=2)
    doc["case"]["inlet_turbulence"].update(
        box_cells=[16, 8, 8], box_lengths_m=[15.0, 0.585, 0.585]
    )
    sim = build_simulation(replace(case, document=doc))
    result = sim.advance(
        sim.initial_state,
        RunControls(count=4, target_time=4 * doc["time"]["dt_seconds"]),
    )
    d = sim.state_diagnostics(result)
    assert float(result.parcels.injected_mass) > 0
    assert float(d["inlet_bulk_u_m_s"]) == pytest.approx(3, abs=1e-12)
    assert abs(float(d["parcel_mass_balance_error_kg"])) < 1e-13
    assert abs(float(d["parcel_enthalpy_balance_error_j"])) < 1e-9
    assert float(sim.courant(result)) < 0.5
    assert bool(jnp.all(jnp.isfinite(result.scalar)))
    from jaxwind.io.checkpoint import load_checkpoint, save_checkpoint

    path = tmp_path / "mapped.npz"
    save_checkpoint(path, result, metadata={"fingerprint": "mapped"})
    restored, _, _ = load_checkpoint(path, sim.initial_state, fingerprint="mapped")
    for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(a, b)


def test_rans_viscous_wall_uses_actual_boundary_cell_width():
    from jaxwind.domain import SinhMapping
    from jaxwind.rans_kepsilon import KEpsilonState, wall_terms

    grid = AnalyticalGrid(8, 8, 8, 1.9, 0.585, 0.585, z_mapping=SinhMapping(0.0, 2.0))
    velocity = StaggeredVelocity(
        jnp.full((8, 8, 9), 3.0), jnp.zeros((8, 9, 8)), jnp.zeros((9, 8, 8))
    )
    k = jnp.full((8, 8, 8), 1e-12)
    force, pk, epsilon, _ = wall_terms(velocity, KEpsilonState(k, k), grid, 1.7e-5)
    for index in (0, -1):
        width = grid.z_widths[index]
        distance = width / 2
        assert float(epsilon[index, 4, 4]) == pytest.approx(
            2 * 1.7e-5 * 1e-12 / distance**2, rel=1e-12
        )
        assert float(force.x[index, 4, 4]) == pytest.approx(
            -1.7e-5 * 3 / distance / width, rel=1e-12
        )
        assert float(pk[index, 4, 4]) == 0


def test_stretched_streamwise_rms_uses_spanwise_area_weights():
    from jaxwind.inflow import build_mann_inflow, generate_mann_box

    grid = build_spray_grid(mesh())
    box = generate_mann_box(
        shape=(16, 8, 8),
        lengths=(2.0, 0.585, 0.585),
        length_scale=0.04095,
        gamma=0.0,
        seed=11,
    )
    inlet = build_mann_inflow(
        box, grid, mean_speed=3.0, wall_y=True, sigma_u_profile=0.3
    )
    times = (
        (np.arange(16)[:, None] + [0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)])
        * (2 / 3)
        / 16
    ).ravel()
    values = np.asarray(
        jax.jit(jax.vmap(lambda t: inlet(t).x_velocity - 3))(jnp.asarray(times))
    )
    variance = np.average((values**2).mean(axis=0), axis=1, weights=grid.y_widths)
    np.testing.assert_allclose(variance, 0.3**2, atol=1e-12)
