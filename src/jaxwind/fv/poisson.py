"""The pressure Poisson system of the staggered projection, solved by JAX-AMG.

On the MAC arrangement the composition of the discrete divergence with the
discrete gradient is the compact seven-point Laplacian: periodic in x and y,
homogeneous Neumann on both walls.  Assembling exactly that operator as a
sparse matrix is what makes an algebraic solve equivalent to the projection --
the corrected velocity is divergence-free to round-off rather than to solver
tolerance in the discretisation sense.

The operator is negated on assembly so the matrix is positive definite, which
is what the Krylov and multigrid solvers expect.  Periodic-plus-Neumann leaves
a constant null space, handled the standard way: the right-hand side is
projected onto the range of the operator and one cell is pinned by symmetric
elimination, then the solution is shifted back to zero mean.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import jax.numpy as jnp
import numpy as np

from jaxwind.domain.grid import UniformGrid

from .operators import divergence, pressure_gradient
from .state import StaggeredVelocity


# Preconditioned conjugate gradients around a classical algebraic multigrid
# V-cycle.  The matrix is symmetric positive definite once the gauge cell is
# pinned, so PCG is the right Krylov method, and classical AMG (PMIS coarsening
# with distance-two interpolation) is the standard choice for a scalar
# seven-point Laplacian: its convergence rate is mesh independent, which is what
# keeps the cost per step flat as the mesh is refined.
CLASSICAL_AMG_PCG: Mapping[str, Any] = {
    "solver": "PCG",
    "preconditioner": {
        "solver": "AMG",
        "algorithm": "CLASSICAL",
        "selector": "PMIS",
        "interpolator": "D2",
        "smoother": {"solver": "BLOCK_JACOBI", "relaxation_factor": 0.9},
        "presweeps": 1,
        "postsweeps": 1,
        "cycle": "V",
        "max_iters": 1,
        "max_levels": 100,
        "strength_threshold": 0.5,
        "coarse_solver": "DENSE_LU_SOLVER",
        "dense_lu_num_rows": 1,
    },
    "convergence": "RELATIVE_INI",
    "tolerance": 1.0e-10,
    "max_iters": 200,
    "norm": "L2",
}


@dataclass(frozen=True, slots=True)
class SparseMatrix:
    """A CSR matrix with the pinned reference row recorded alongside it."""

    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    row_count: int
    reference_cell: int | None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.row_count, self.row_count)


class LinearSolver(Protocol):
    def __call__(self, right_hand_side: jnp.ndarray) -> jnp.ndarray: ...


def default_tolerance(dtype) -> float:
    """Relative residual a given precision can actually reach.

    Single precision carries about seven decimal digits, so asking for the
    double-precision tolerance would simply run the Krylov solver to its
    iteration cap on every solve -- slower than double precision rather than
    faster, and with no better answer.
    """
    return 1.0e-6 if np.dtype(dtype).itemsize <= 4 else 1.0e-10


def _coalesce_to_csr(
    rows: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    row_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum duplicate entries and emit sorted CSR arrays."""
    order = np.lexsort((columns, rows))
    rows, columns, values = rows[order], columns[order], values[order]
    starts = np.ones(rows.size, dtype=bool)
    starts[1:] = (rows[1:] != rows[:-1]) | (columns[1:] != columns[:-1])
    offsets = np.flatnonzero(starts)
    rows, columns = rows[offsets], columns[offsets]
    values = np.add.reduceat(values, offsets)
    keep = (values != 0.0) | (rows == columns)
    rows, columns, values = rows[keep], columns[keep], values[keep]
    indptr = np.zeros(row_count + 1, dtype=np.int32)
    np.cumsum(np.bincount(rows, minlength=row_count), out=indptr[1:])
    return values, columns.astype(np.int32), indptr


def assemble_pressure_matrix(
    grid: UniformGrid,
    *,
    dtype: str = "float64",
    reference_cell: int | None = 0,
) -> SparseMatrix:
    """Assemble ``-D G`` for the staggered mesh, optionally pinning the gauge.

    Row and column ``r`` of the matrix is cell ``(k, j, i)`` with
    ``r = k * ny * nx + j * nx + i``, the C ordering of a ``(nz, ny, nx)``
    cell-centred array.  With ``reference_cell=None`` the matrix is exactly
    ``-D G`` and therefore singular; pinning a cell makes it definite.
    """
    if reference_cell is not None and not 0 <= reference_cell < grid.cell_count:
        raise ValueError("the pinned reference cell is outside the mesh")
    resolved = np.dtype(dtype)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    plane = ny * nx
    k, j, i = (
        index.ravel()
        for index in np.meshgrid(
            np.arange(nz),
            np.arange(ny),
            np.arange(nx),
            indexing="ij",
        )
    )
    rows = (k * plane + j * nx + i).astype(np.int64)
    inverse_dx2 = 1.0 / grid.dx**2
    inverse_dy2 = 1.0 / grid.dy**2
    inverse_dz2 = 1.0 / grid.dz**2

    has_lower = k > 0
    has_upper = k < nz - 1
    diagonal = np.full(rows.size, 2.0 * (inverse_dx2 + inverse_dy2))
    diagonal += inverse_dz2 * (has_lower.astype(np.float64) + has_upper)

    row_blocks = [rows]
    column_blocks = [rows]
    value_blocks = [diagonal]

    for shift in (-1, 1):
        row_blocks.append(rows)
        column_blocks.append(k * plane + j * nx + (i + shift) % nx)
        value_blocks.append(np.full(rows.size, -inverse_dx2))
        row_blocks.append(rows)
        column_blocks.append(k * plane + ((j + shift) % ny) * nx + i)
        value_blocks.append(np.full(rows.size, -inverse_dy2))

    for mask, shift in ((has_lower, -1), (has_upper, 1)):
        row_blocks.append(rows[mask])
        column_blocks.append(rows[mask] + shift * plane)
        value_blocks.append(np.full(int(mask.sum()), -inverse_dz2))

    all_rows = np.concatenate(row_blocks)
    all_columns = np.concatenate(column_blocks).astype(np.int64)
    all_values = np.concatenate(value_blocks)

    if reference_cell is not None:
        # Symmetric elimination of the pinned cell: its equation is the
        # redundant one, and pinning it to zero means the eliminated column
        # contributes nothing to the remaining right-hand sides.
        interior = (all_rows != reference_cell) & (all_columns != reference_cell)
        pinned_diagonal = float(diagonal[reference_cell])
        all_rows = np.append(all_rows[interior], reference_cell)
        all_columns = np.append(all_columns[interior], reference_cell)
        all_values = np.append(all_values[interior], pinned_diagonal)

    values, columns, indptr = _coalesce_to_csr(
        all_rows,
        all_columns,
        all_values,
        grid.cell_count,
    )
    return SparseMatrix(
        values.astype(resolved),
        columns,
        indptr,
        grid.cell_count,
        reference_cell,
    )


def matrix_vector_product(matrix: SparseMatrix, values: jnp.ndarray) -> jnp.ndarray:
    """Apply the assembled matrix without materialising a dense operator.

    The row index of every stored entry is expanded on the host, where the
    sparsity pattern is already a concrete array; expanding it inside the traced
    computation would leave XLA constant-folding it on every compilation.
    """
    row_index = jnp.asarray(
        np.repeat(np.arange(matrix.row_count), np.diff(matrix.indptr))
    )
    products = jnp.asarray(matrix.data, values.dtype) * values[
        jnp.asarray(matrix.indices)
    ]
    return jnp.zeros(matrix.row_count, values.dtype).at[row_index].add(products)


def build_amg_solver(
    matrix: SparseMatrix,
    *,
    config: Mapping[str, Any] | None = None,
    reuse_setup: bool = True,
) -> LinearSolver:
    """Solve the pinned system with JAX-AMG on the GPU.

    Defaults to :data:`CLASSICAL_AMG_PCG`; ``config`` replaces it wholesale and
    is passed straight to :func:`jaxamg.solve`.

    Two environment settings are needed at run time.  AmgX allocates on the
    device through its own allocator rather than the JAX memory pool, so JAX
    must be told not to preallocate the GPU::

        export XLA_PYTHON_CLIENT_PREALLOCATE=false

    and the AmgX shared library must be loadable::

        export LD_LIBRARY_PATH=$AMGX_BUILD:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    Without the first, AmgX fails its allocations and reports the misleading
    "Incorrect parameters for amgx call".
    """
    try:
        import jaxamg
        from jax.experimental import sparse as jax_sparse
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the AMG pressure solver requires jaxamg; initialise the "
            "external/jax-amg submodule and install it with "
            "'pip install -e external/jax-amg' (AmgX and a CUDA jaxlib are "
            "required at run time)"
        ) from exc
    operator = jax_sparse.BCSR(
        (
            jnp.asarray(matrix.data),
            jnp.asarray(matrix.indices),
            jnp.asarray(matrix.indptr),
        ),
        shape=matrix.shape,
    )
    operator = jaxamg.with_cache(operator, is_symmetric=True)
    if config is None:
        settings = dict(CLASSICAL_AMG_PCG)
        settings["tolerance"] = default_tolerance(matrix.data.dtype)
    else:
        settings = dict(config)

    def solve(right_hand_side: jnp.ndarray) -> jnp.ndarray:
        solution, _ = jaxamg.solve(
            operator,
            right_hand_side,
            config=settings,
            reuse_setup=reuse_setup,
        )
        return solution

    return solve


def build_cg_solver(
    matrix: SparseMatrix,
    *,
    tolerance: float | None = None,
    max_iterations: int = 2000,
) -> LinearSolver:
    """Solve the pinned system with Jacobi-preconditioned conjugate gradients.

    This backend solves the same assembled matrix as :func:`build_amg_solver`
    and runs on any JAX platform, which is what makes the finite-volume solver
    testable and runnable without AmgX.
    """
    from jax.scipy.sparse.linalg import cg

    if tolerance is None:
        tolerance = default_tolerance(matrix.data.dtype)
    diagonal = np.zeros(matrix.row_count, dtype=matrix.data.dtype)
    for row in range(matrix.row_count):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        columns = matrix.indices[start:stop]
        diagonal[row] = matrix.data[start:stop][columns == row].sum()
    inverse_diagonal = jnp.asarray(np.where(diagonal != 0.0, 1.0 / diagonal, 1.0))

    def solve(right_hand_side: jnp.ndarray) -> jnp.ndarray:
        solution, _ = cg(
            lambda values: matrix_vector_product(matrix, values),
            right_hand_side,
            M=lambda values: inverse_diagonal.astype(values.dtype) * values,
            tol=tolerance,
            atol=0.0,
            maxiter=max_iterations,
        )
        return solution

    return solve


@dataclass(frozen=True, slots=True)
class PressurePoisson:
    """The assembled pressure operator together with its linear solver."""

    grid: UniformGrid
    matrix: SparseMatrix
    linear_solver: LinearSolver

    def solve(self, right_hand_side: jnp.ndarray) -> jnp.ndarray:
        """Return the zero-mean cell-centred solution of ``D G p = rhs``."""
        if right_hand_side.shape != (self.grid.nz, self.grid.ny, self.grid.nx):
            raise ValueError("the pressure right-hand side must be cell centred")
        flat = self._prepare(right_hand_side)
        solution = self.linear_solver(flat)
        solution = solution - jnp.mean(solution)
        return solution.reshape(right_hand_side.shape)

    def residual_norm(
        self,
        pressure: jnp.ndarray,
        right_hand_side: jnp.ndarray,
    ) -> jnp.ndarray:
        """Norm of ``D G p - rhs`` for the physical, unpinned operator.

        The constant part of the residual is removed because it is the part no
        pressure can reproduce: it is the component of the right-hand side
        outside the range of a periodic-plus-Neumann Laplacian, and it is zero
        whenever the right-hand side comes from a divergence.
        """
        applied = divergence(pressure_gradient(pressure, self.grid), self.grid)
        error = applied - right_hand_side
        return jnp.linalg.norm(error - jnp.mean(error))

    def _prepare(self, right_hand_side: jnp.ndarray) -> jnp.ndarray:
        """Negate, make compatible with the null space, and drop the gauge row."""
        flat = -right_hand_side.reshape(-1)
        flat = flat - jnp.mean(flat)
        if self.matrix.reference_cell is None:
            return flat
        return flat.at[self.matrix.reference_cell].set(0.0)


def build_pressure_poisson(
    grid: UniformGrid,
    *,
    backend: str = "amg",
    dtype: str = "float64",
    reference_cell: int | None = 0,
    config: Mapping[str, Any] | None = None,
) -> PressurePoisson:
    """Assemble the pressure operator and attach the requested solver."""
    matrix = assemble_pressure_matrix(
        grid,
        dtype=dtype,
        reference_cell=reference_cell,
    )
    if backend == "amg":
        solver = build_amg_solver(matrix, config=config)
    elif backend == "cg":
        solver = build_cg_solver(matrix, **dict(config or {}))
    else:
        raise ValueError(f"unsupported pressure backend: {backend!r}")
    return PressurePoisson(grid, matrix, solver)


def project(
    velocity: StaggeredVelocity,
    poisson: PressurePoisson,
    dt: float,
) -> tuple[StaggeredVelocity, jnp.ndarray]:
    """Remove the divergent part of a candidate velocity."""
    grid = poisson.grid
    pressure = poisson.solve(divergence(velocity, grid) / dt)
    gradient = pressure_gradient(pressure, grid)
    corrected = StaggeredVelocity(
        velocity.x - dt * gradient.x,
        velocity.y - dt * gradient.y,
        velocity.z - dt * gradient.z,
    )
    return corrected, pressure


__all__ = [
    "CLASSICAL_AMG_PCG",
    "LinearSolver",
    "PressurePoisson",
    "SparseMatrix",
    "assemble_pressure_matrix",
    "build_amg_solver",
    "build_cg_solver",
    "build_pressure_poisson",
    "default_tolerance",
    "matrix_vector_product",
    "project",
]
