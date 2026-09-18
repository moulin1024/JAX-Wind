"""Independent pressure-operator checks for a physical transverse inlet face."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.domain import AnalyticalGrid, TanhMapping, UniformGrid
from jaxwind.numerics.discretization import divergence, pressure_gradient
from jaxwind.numerics.poisson import (
    _apply_laplacian,
    _build_gmg_levels,
    _diagonal_stencil,
    build_gmg_solver,
    build_pressure_poisson,
    project,
)
from jaxwind.state import StaggeredVelocity

jax.config.update("jax_enable_x64", True)


def _independent_matrix(grid, periodic_y=False):
    """Assemble face energy contributions without calling production D or G.

    Inlet x-face has zero normal correction. Every transverse interior face
    in the first column is active, while those in the last column are fixed.
    The pressure-open high-x face contributes its half-cell conductance.
    """
    matrix = np.zeros((grid.cell_count, grid.cell_count))

    def index(k, j, i):
        return (k * grid.ny + j) * grid.nx + i

    def connect(a, b, weight):
        matrix[a, a] += weight
        matrix[b, b] += weight
        matrix[a, b] -= weight
        matrix[b, a] -= weight

    for k in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx):
                row = index(k, j, i)
                if i + 1 < grid.nx:
                    connect(row, index(k, j, i + 1), 1 / grid.dx**2)
                else:
                    matrix[row, row] += 2 / grid.dx**2
                if i == grid.nx - 1:
                    continue
                if periodic_y:
                    connect(row, index(k, (j + 1) % grid.ny, i), 1 / grid.dy**2)
                elif j + 1 < grid.ny:
                    connect(row, index(k, j + 1, i), 1 / grid.dy**2)
                if k + 1 < grid.nz:
                    connect(row, index(k + 1, j, i), 1 / grid.dz**2)
    return matrix


def _gradient(p, grid, periodic_y=False, **kwargs):
    return pressure_gradient(
        p,
        grid,
        periodic_x=False,
        periodic_y=periodic_y,
        physical_transverse_inlet=True,
        **kwargs,
    )


def _poisson(grid, **kwargs):
    return build_pressure_poisson(
        grid,
        backend="gmg",
        periodic_x=False,
        periodic_y=False,
        physical_transverse_inlet=True,
        config={"tolerance": 1e-11, "max_iterations": 400},
        **kwargs,
    )


def test_releases_only_first_column_transverse_pressure_gradient():
    grid = UniformGrid(4, 3, 2, 1.4, 0.9, 0.8)
    p = jnp.broadcast_to(
        jnp.asarray(grid.y_centers)[None, :, None]
        + 2 * jnp.asarray(grid.z_centers)[:, None, None],
        (grid.nz, grid.ny, grid.nx),
    )
    old = pressure_gradient(p, grid, periodic_x=False, periodic_y=False)
    explicit_default = pressure_gradient(
        p,
        grid,
        periodic_x=False,
        periodic_y=False,
        physical_transverse_inlet=False,
    )
    new = _gradient(p, grid)
    for implicit, explicit in zip(old, explicit_default):
        np.testing.assert_array_equal(implicit, explicit)
    np.testing.assert_array_equal(new.x, old.x)
    np.testing.assert_array_equal(new.x[..., 0], 0)
    np.testing.assert_allclose(new.y[:, 1:-1, 0], 1, rtol=0, atol=1e-14)
    np.testing.assert_allclose(new.z[1:-1, :, 0], 2, rtol=0, atol=1e-14)
    for before, after in zip(old[1:], new[1:]):
        np.testing.assert_array_equal(before[..., 1:], after[..., 1:])
        np.testing.assert_array_equal(after[..., -1], 0)
    np.testing.assert_array_equal(new.y[:, [0, -1], :], 0)
    np.testing.assert_array_equal(new.z[[0, -1], :, :], 0)


@pytest.mark.parametrize("periodic_y", [False, True])
def test_every_multigrid_level_matches_independent_operator_and_diagonal(periodic_y):
    grid = UniformGrid(8, 6, 4, 1.4, 0.9, 0.8)
    levels, _ = _build_gmg_levels(grid)
    # Include the unit periodic-y direction encountered in other hierarchies.
    levels += [UniformGrid(4, 1, 2, 1.4, 0.9, 0.8)]
    for level in levels:
        matrix = _independent_matrix(level, periodic_y)
        p = np.sin(np.arange(level.cell_count) + 0.31).reshape(
            (level.nz, level.ny, level.nx)
        )
        applied = _apply_laplacian(
            jnp.asarray(p),
            level,
            periodic_x=False,
            periodic_y=periodic_y,
            physical_transverse_inlet=True,
        )
        np.testing.assert_allclose(
            applied.ravel(), matrix @ p.ravel(), rtol=2e-14, atol=2e-12
        )
        diagonal = _diagonal_stencil(
            level,
            np.dtype("float64"),
            periodic_x=False,
            periodic_y=periodic_y,
            physical_transverse_inlet=True,
        )
        np.testing.assert_allclose(
            diagonal.ravel(), np.diag(matrix), rtol=1e-14, atol=1e-12
        )
        np.testing.assert_array_equal(matrix, matrix.T)
        assert np.linalg.eigvalsh(matrix).min() > 0


def test_gmg_recovers_independent_pressure_and_projection_at_inlet_column():
    grid = UniformGrid(8, 6, 4, 1.4, 0.9, 0.8)
    shape = (grid.nz, grid.ny, grid.nx)
    exact = np.random.default_rng(713).normal(size=shape)
    rhs = jnp.asarray(-(_independent_matrix(grid) @ exact.ravel()).reshape(shape))
    poisson = _poisson(grid)
    assert poisson.physical_transverse_inlet
    solved = jax.jit(poisson.solve)(rhs)
    np.testing.assert_allclose(solved, exact, rtol=0, atol=2e-9)
    assert float(poisson.residual_norm(solved, rhs)) < 1e-7
    dt = 0.017
    gradient = _gradient(jnp.asarray(exact), grid)
    base = StaggeredVelocity(
        jnp.full((grid.nz, grid.ny, grid.nx + 1), 2.0),
        jnp.zeros((grid.nz, grid.ny + 1, grid.nx)),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx)),
    )
    predictor = jax.tree.map(lambda u, g: u + dt * g, base, gradient)
    projected, _ = jax.jit(lambda u: project(u, poisson, dt))(predictor)
    for corrected, expected in zip(projected, base):
        np.testing.assert_allclose(corrected, expected, rtol=0, atol=2e-9)
    assert float(jnp.max(jnp.abs(predictor.y[:, 1:-1, 0]))) > 0.01
    np.testing.assert_array_equal(projected.x[..., 0], base.x[..., 0])
    np.testing.assert_array_equal(projected.y[..., -1], predictor.y[..., -1])
    np.testing.assert_array_equal(projected.z[..., -1], predictor.z[..., -1])


def test_target_divergence_preserves_inlet_and_integrated_outlet_flux():
    grid = UniformGrid(8, 6, 4, 1.4, 0.9, 0.8)
    poisson = _poisson(grid)
    target = jnp.zeros((grid.nz, grid.ny, grid.nx)).at[1:3, 1:5, 0].set(0.7)
    zero = StaggeredVelocity(
        jnp.zeros((grid.nz, grid.ny, grid.nx + 1)),
        jnp.zeros((grid.nz, grid.ny + 1, grid.nx)),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx)),
    )
    corrected, _ = jax.jit(
        lambda v: project(v, poisson, 0.01, target_divergence=target)
    )(zero)
    np.testing.assert_allclose(divergence(corrected, grid), target, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(corrected.x[..., 0], 0)
    np.testing.assert_array_equal(corrected.y[:, [0, -1], :], 0)
    np.testing.assert_array_equal(corrected.z[[0, -1], :, :], 0)
    outlet = np.asarray(corrected.x[..., -1]).sum() * grid.dy * grid.dz
    required = np.asarray(target).sum() * grid.dx * grid.dy * grid.dz
    assert abs(outlet - required) < 1e-10


@pytest.mark.parametrize("backend", ["amg", "fft"])
def test_explicitly_rejects_unsupported_pressure_backends(backend):
    grid = UniformGrid(4, 4, 4, 1, 1, 1)
    with pytest.raises(ValueError, match="GMG"):
        build_pressure_poisson(
            grid,
            backend=backend,
            periodic_x=False,
            physical_transverse_inlet=True,
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"periodic_x": True},
        {"periodic_x": False, "open_x_low": True},
        {"periodic_x": False, "periodic_y": False, "open_y": True},
    ],
)
def test_rejects_incompatible_inlet_topologies(settings):
    grid = UniformGrid(4, 4, 4, 1, 1, 1)
    with pytest.raises(ValueError, match="fixed-flux"):
        build_pressure_poisson(
            grid, backend="gmg", physical_transverse_inlet=True, **settings
        )
    with pytest.raises(ValueError, match="fixed-flux"):
        build_gmg_solver(grid, physical_transverse_inlet=True, **settings)


def test_rejects_unsupported_mapped_pressure_grid():
    grid = AnalyticalGrid(
        4, 4, 4, 1, 1, 1, TanhMapping(1.0), TanhMapping(0.0), TanhMapping(0.0)
    )
    with pytest.raises(ValueError, match="uniform"):
        _poisson(grid)
