from __future__ import annotations

import math

import numpy as np

from benchmark.Andren1994 import run
from benchmark.Andren1994.overlay_paper_figures import (
    _history_data,
    _profile_data,
)


def test_andren_runner_defaults_to_filter_free_amd() -> None:
    args = run.parse_args([])

    assert args.sgs == "amd"
    assert args.amd_coefficient == 0.212
    assert args.end_ft == 0.1
    assert args.sample_start_ft == 0.05


def test_andren_runner_accepts_lasd_and_canonical_amd_controls() -> None:
    lasd = run.parse_args(["--sgs", "lasd"])
    amd = run.parse_args(
        [
            "--sgs",
            "amd",
            "--amd-coefficient",
            "0.3",
            "--end-ft",
            "10",
            "--sample-start-ft",
            "7",
        ]
    )

    assert lasd.sgs == "lasd"
    assert amd.sgs == "amd"
    assert amd.amd_coefficient == 0.3
    assert amd.end_ft == 10.0
    assert amd.sample_start_ft == 7.0


def test_paper_overlay_reads_current_amd_profile_schema(tmp_path) -> None:
    header = (
        "z_m,zf_over_ustar,mean_u_m_s,mean_v_m_s,"
        "var_u_m2_s2,var_v_m2_s2,var_w_m2_s2,"
        "resolved_uw_m2_s2,resolved_vw_m2_s2,"
        "sgs_uw_m2_s2,sgs_vw_m2_s2,"
        "total_uw_m2_s2,total_vw_m2_s2,phi_m\n"
    )
    rows = (
        "10,0.1,1,0,4,8,12,-1,-2,-0.5,-0.25,-1.5,-2.25,0.9\n"
        "20,0.2,2,0,8,12,16,-2,-3,-0.5,-0.25,-2.5,-3.25,1.1\n"
    )
    (tmp_path / "andren1994_profiles.csv").write_text(header + rows)

    profile = _profile_data(tmp_path, statistics_ustar=2.0)

    assert np.allclose(profile["height"], (0.1, 0.2))
    assert np.allclose(profile["u_variance"], (1.0, 2.0))
    assert np.allclose(profile["total_uw"], (-0.375, -0.625))
    assert np.allclose(profile["phi_m"], (0.9, 1.1))
    assert profile["phi_c"] is None
    assert _history_data(tmp_path, statistics_ustar=2.0) is None


def test_logarithmic_shear_is_exact_on_a_log_profile_at_every_level() -> None:
    """A centred difference is not, and the bias is largest where it matters.

    Monin-Obukhov similarity is read off the first few levels, where a centred
    difference of a ``1/z`` shear is biased by tens of per cent purely from the
    grid.  This estimator has to be exact there or the near-wall part of the
    plot describes the operator instead of the flow.
    """

    kappa, ustar, roughness = 0.4, 0.35, 0.1
    for faces in (
        np.linspace(0.0, 1500.0, 41),
        1500.0 * np.expm1(2.5 * np.linspace(0.0, 1.0, 41)) / np.expm1(2.5),
    ):
        z = 0.5 * (faces[:-1] + faces[1:])
        u = (ustar / kappa) * np.log(z / roughness)
        v = np.zeros_like(u)

        du_dz, dv_dz = run._logarithmic_shear(u, v, z)
        phi_m = kappa * z * np.hypot(du_dz, dv_dz) / ustar

        assert np.allclose(phi_m, 1.0, rtol=1.0e-12)
        assert np.allclose(dv_dz, 0.0)

    # The estimator that was replaced, for the record: it reads 0.549 at the
    # first level and 1.207 at the second on a uniform mesh.
    z = 0.5 * (np.linspace(0.0, 1500.0, 41)[:-1] + np.linspace(0.0, 1500.0, 41)[1:])
    u = (ustar / kappa) * np.log(z / roughness)
    centred = kappa * z * np.gradient(u, z) / ustar
    assert not np.isclose(centred[0], 1.0, rtol=0.1)
    assert not np.isclose(centred[1], 1.0, rtol=0.1)


def test_cells_from_faces_averages_onto_the_enclosed_cells() -> None:
    faces = np.array((1.0, 3.0, 5.0, 9.0))

    (cells,) = run._cells_from_faces(faces)

    assert np.allclose(cells, (2.0, 4.0, 7.0))


def test_initial_tables_are_recovered_exactly_on_the_published_mesh() -> None:
    from jaxwind.pressure import RectilinearGrid

    grid = RectilinearGrid.uniform(40, 40, 40, lx=4000.0, ly=2000.0, lz=1500.0)

    initial_u, initial_v, initial_tke = run._initial_tables_on_grid(
        grid,
        1500.0,
        0.1,
    )

    assert np.allclose(initial_u, run.INITIAL_U)
    assert np.allclose(initial_v, run.INITIAL_V)
    assert np.allclose(initial_tke, run.INITIAL_TKE)


def test_initial_tables_follow_a_wall_refined_mesh_in_height() -> None:
    """A stretched mesh must read the tables by height, not by index."""

    from jaxwind.pressure import RectilinearGrid

    strength = 2.283724
    parameter = np.linspace(0.0, 1.0, 41)
    faces = 1500.0 * np.expm1(strength * parameter) / np.expm1(strength)
    grid = RectilinearGrid(
        tuple(np.linspace(0.0, 4000.0, 41)),
        tuple(np.linspace(0.0, 2000.0, 41)),
        tuple(float(value) for value in faces),
    )

    initial_u, _, initial_tke = run._initial_tables_on_grid(grid, 1500.0, 0.1)
    centers = np.asarray(grid.z_centers)

    # The first centre drops from 18.75 m to 5 m, inside the surface layer, so
    # the wind is continued logarithmically instead of held at its 18.75 m value.
    assert np.isclose(centers[0], 5.0, atol=1.0e-6)
    expected = (
        run.INITIAL_U[0] * math.log(centers[0] / 0.1) / math.log(18.75 / 0.1)
    )
    assert np.isclose(initial_u[0], expected, rtol=1.0e-12)
    assert initial_u[0] < run.INITIAL_U[0]
    # The perturbation amplitude is only held constant there.
    assert np.isclose(initial_tke[0], run.INITIAL_TKE[0])
    # Above the first published level nothing is extrapolated.
    assert initial_u.max() <= max(run.INITIAL_U) + 1.0e-12


def test_andren_runner_accepts_a_mesh_artifact() -> None:
    assert run.parse_args([]).mesh is None
    assert run.parse_args(["--mesh", "m.json"]).mesh.name == "m.json"
