"""The pressure Poisson system of the staggered projection.

Three backends solve the pressure Poisson system: ``amg`` hands the assembled
operator to JAX-AMG (:mod:`jaxamg`) on the GPU, which needs AmgX and a CUDA
build; ``gmg`` runs conjugate gradients preconditioned by a geometric
multigrid V-cycle built directly from the mesh (see :func:`build_gmg_solver`)
-- pure JAX and matrix-free, with a mesh-independent iteration count and, being
built from local stencils rather than a global FFT, friendlier to a
domain-decomposed, multi-GPU mesh; and ``fft`` solves it exactly, in one shot,
by diagonalising the operator with a 2-D FFT and a one-time eigendecomposition
-- valid only because the mesh is periodic in x and y (see below), but then
exact to floating point with no iteration at all.

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

import jax
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


def build_fft_solver(
    grid: UniformGrid,
    *,
    dtype: str = "float64",
) -> LinearSolver:
    """Solve the pressure system exactly by diagonalising the operator.

    The mesh is always periodic in x and y and Neumann in z (see the module
    docstring), so the seven-point Laplacian is the Kronecker sum of a
    periodic horizontal operator and a Neumann vertical one, and the two
    commute.  A real 2-D FFT diagonalises the horizontal part exactly; the
    vertical operator is a fixed ``nz x nz`` tridiagonal matrix, diagonalised
    once by a dense eigendecomposition.  The 3-D solve then reduces to one
    elementwise division per mode -- exact up to floating point, with no
    iteration and no setup-time tuning.

    This is only valid because of that periodicity: unlike ``amg``, which solves whatever sparse matrix it is handed, this
    diagonalisation stops being correct the moment the horizontal boundary
    is not periodic.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    resolved = np.dtype(dtype)
    inverse_dx2 = 1.0 / grid.dx**2
    inverse_dy2 = 1.0 / grid.dy**2
    inverse_dz2 = 1.0 / grid.dz**2

    # rfft2 keeps the full range of ky but only the non-redundant half of kx;
    # cos(2 pi k / n) is symmetric under k -> n - k, so no wrapping is needed.
    kx = np.arange(nx // 2 + 1)
    ky = np.arange(ny)
    lambda_x = 2.0 * (1.0 - np.cos(2.0 * np.pi * kx / nx)) * inverse_dx2
    lambda_y = 2.0 * (1.0 - np.cos(2.0 * np.pi * ky / ny)) * inverse_dy2
    horizontal = lambda_y[:, None] + lambda_x[None, :]

    # The dense vertical operator, matching assemble_pressure_matrix's
    # one-sided stencil at the walls exactly.
    diag_index = np.arange(nz)
    vertical = np.zeros((nz, nz))
    vertical[diag_index, diag_index] = inverse_dz2 * (
        (diag_index > 0).astype(np.float64) + (diag_index < nz - 1)
    )
    if nz > 1:
        off_diagonal = -inverse_dz2 * np.ones(nz - 1)
        vertical[diag_index[:-1], diag_index[:-1] + 1] = off_diagonal
        vertical[diag_index[:-1] + 1, diag_index[:-1]] = off_diagonal

    # eigh is ascending, so mode 0 is the constant, null-space eigenvector;
    # clip its eigenvalue to exactly zero to avoid dividing by round-off.
    eigenvalues, eigenvectors = np.linalg.eigh(vertical)
    eigenvalues[0] = 0.0

    eigenvalues = jnp.asarray(eigenvalues, resolved)
    eigenvectors = jnp.asarray(eigenvectors, resolved)
    horizontal = jnp.asarray(horizontal, resolved)

    def solve(right_hand_side: jnp.ndarray) -> jnp.ndarray:
        field = right_hand_side.reshape(nz, ny, nx)
        spectrum = jnp.fft.rfft2(field, axes=(1, 2))
        modal = jnp.einsum("mz,zyx->myx", eigenvectors.T, spectrum)
        denominator = eigenvalues[:, None, None] + horizontal[None, :, :]
        # The single all-zero mode is the null space; the caller has already
        # made the right-hand side compatible (zero mean), so its coefficient
        # is already zero and dividing by one there just leaves it alone.
        safe = jnp.where(denominator == 0.0, 1.0, denominator)
        modal = jnp.where(denominator == 0.0, 0.0, modal / safe)
        spectrum = jnp.einsum("zm,myx->zyx", eigenvectors, modal)
        solution = jnp.fft.irfft2(spectrum, s=(ny, nx), axes=(1, 2))
        return solution.reshape(-1)

    return solve


def _apply_laplacian(pressure: jnp.ndarray, grid: UniformGrid) -> jnp.ndarray:
    """Matrix-free ``-D G p`` on ``grid``, matching :func:`assemble_pressure_matrix`."""
    return -divergence(pressure_gradient(pressure, grid), grid)


def _diagonal_stencil(grid: UniformGrid, dtype: np.dtype) -> jnp.ndarray:
    """The diagonal of ``-D G`` on ``grid``, broadcastable over ``(nz, ny, nx)``.

    Independent of x and y (the horizontal directions are periodic, so every
    cell sees the same two neighbours); only the wall-adjacent z-layers differ,
    carrying one vertical neighbour instead of two.
    """
    inverse_dx2 = 1.0 / grid.dx**2
    inverse_dy2 = 1.0 / grid.dy**2
    inverse_dz2 = 1.0 / grid.dz**2
    k = np.arange(grid.nz)
    has_lower = k > 0
    has_upper = k < grid.nz - 1
    diagonal = 2.0 * (inverse_dx2 + inverse_dy2) + inverse_dz2 * (
        has_lower.astype(np.float64) + has_upper.astype(np.float64)
    )
    return jnp.asarray(diagonal, dtype).reshape(grid.nz, 1, 1)


def _coarsening_factors(grid: UniformGrid) -> tuple[int, int, int]:
    """Per-axis factor-two agglomeration, one axis at a time as it allows it.

    An axis stops coarsening as soon as its cell count is odd or one, which is
    what lets a mesh with mixed factors (say ``12 = 4 * 3``) coarsen as far as
    each direction supports rather than stalling the whole hierarchy on the one
    direction that cannot be halved evenly.
    """
    return tuple(2 if n > 1 and n % 2 == 0 else 1 for n in (grid.nx, grid.ny, grid.nz))


def _coarsen_grid(grid: UniformGrid, factors: tuple[int, int, int]) -> UniformGrid:
    factor_x, factor_y, factor_z = factors
    return UniformGrid(
        grid.nx // factor_x,
        grid.ny // factor_y,
        grid.nz // factor_z,
        grid.lx,
        grid.ly,
        grid.lz,
    )


def _build_gmg_levels(grid: UniformGrid) -> tuple[list[UniformGrid], list[tuple[int, int, int]]]:
    """Coarsen by cell-agglomeration until no axis can be halved further."""
    levels = [grid]
    factors = []
    current = grid
    while True:
        current_factors = _coarsening_factors(current)
        if current_factors == (1, 1, 1):
            return levels, factors
        current = _coarsen_grid(current, current_factors)
        levels.append(current)
        factors.append(current_factors)


def _neighbor(
    values: jnp.ndarray,
    axis: int,
    offset: int,
    *,
    periodic: bool,
) -> jnp.ndarray:
    """Shift one cell, extending a non-periodic axis with its edge value."""
    if periodic:
        return jnp.roll(values, offset, axis=axis)
    edge = [slice(None)] * values.ndim
    interior = [slice(None)] * values.ndim
    if offset == 1:
        edge[axis] = slice(0, 1)
        interior[axis] = slice(0, -1)
        return jnp.concatenate((values[tuple(edge)], values[tuple(interior)]), axis=axis)
    edge[axis] = slice(-1, None)
    interior[axis] = slice(1, None)
    return jnp.concatenate((values[tuple(interior)], values[tuple(edge)]), axis=axis)


def _restrict_axis(
    values: jnp.ndarray,
    axis: int,
    *,
    periodic: bool,
) -> jnp.ndarray:
    """Apply the scaled adjoint of cell-centred linear interpolation."""
    lower_index = [slice(None)] * values.ndim
    upper_index = [slice(None)] * values.ndim
    lower_index[axis] = slice(0, None, 2)
    upper_index[axis] = slice(1, None, 2)
    lower = values[tuple(lower_index)]
    upper = values[tuple(upper_index)]
    if periodic:
        previous_upper = jnp.roll(upper, 1, axis=axis)
        next_lower = jnp.roll(lower, -1, axis=axis)
    else:
        first = [slice(None)] * values.ndim
        before_last = [slice(None)] * values.ndim
        after_first = [slice(None)] * values.ndim
        last = [slice(None)] * values.ndim
        first[axis] = slice(0, 1)
        before_last[axis] = slice(0, -1)
        after_first[axis] = slice(1, None)
        last[axis] = slice(-1, None)
        previous_upper = jnp.concatenate(
            (lower[tuple(first)], upper[tuple(before_last)]), axis=axis
        )
        next_lower = jnp.concatenate(
            (lower[tuple(after_first)], upper[tuple(last)]), axis=axis
        )
    return 0.375 * (lower + upper) + 0.125 * (previous_upper + next_lower)


def _restrict(residual: jnp.ndarray, factors: tuple[int, int, int]) -> jnp.ndarray:
    """Cell-centred full weighting, scaled-adjoint to :func:`_prolong`."""
    factor_x, factor_y, factor_z = factors
    if factor_z == 2:
        residual = _restrict_axis(residual, 0, periodic=False)
    if factor_y == 2:
        residual = _restrict_axis(residual, 1, periodic=True)
    if factor_x == 2:
        residual = _restrict_axis(residual, 2, periodic=True)
    return residual


def _prolong_axis(
    values: jnp.ndarray,
    axis: int,
    *,
    periodic: bool,
) -> jnp.ndarray:
    """Linearly interpolate coarse cell centres to their two fine children."""
    previous = _neighbor(values, axis, 1, periodic=periodic)
    following = _neighbor(values, axis, -1, periodic=periodic)
    lower = 0.75 * values + 0.25 * previous
    upper = 0.75 * values + 0.25 * following
    shape = list(values.shape)
    shape[axis] *= 2
    return jnp.stack((lower, upper), axis=axis + 1).reshape(shape)


def _prolong(correction: jnp.ndarray, factors: tuple[int, int, int]) -> jnp.ndarray:
    """Cell-centred trilinear interpolation with Neumann extension in z."""
    factor_x, factor_y, factor_z = factors
    if factor_z == 2:
        correction = _prolong_axis(correction, 0, periodic=False)
    if factor_y == 2:
        correction = _prolong_axis(correction, 1, periodic=True)
    if factor_x == 2:
        correction = _prolong_axis(correction, 2, periodic=True)
    return correction


def build_gmg_solver(
    grid: UniformGrid,
    *,
    dtype: str = "float64",
    tolerance: float | None = None,
    max_iterations: int = 200,
    presweeps: int = 2,
    postsweeps: int = 2,
    omega: float = 0.8,
    cycles_per_precondition: int = 1,
) -> LinearSolver:
    """Solve the pressure system with PCG preconditioned by a geometric V-cycle.

    Every level is a coarser :class:`UniformGrid` over the same physical box,
    obtained by agglomerating cells two at a time along each axis that still
    allows it (see :func:`_coarsening_factors`); the operator at every level is
    then just :func:`_apply_laplacian` rediscretised on that coarser mesh --
    the same seven-point stencil :func:`assemble_pressure_matrix` assembles,
    applied matrix-free with :func:`~jaxwind.fv.operators.divergence` and
    :func:`~jaxwind.fv.operators.pressure_gradient` instead of a sparse
    matrix-vector product. No level ever materialises a matrix, which is what
    makes this backend "matrix-free": setup only builds the grid hierarchy and
    the (diagonal) Jacobi weights, and every solve is stencils and reductions.

    The coarsest level -- typically tiny once every factor of two has been
    agglomerated out -- is approximated by five matrix-free CG iterations.
    This keeps the complete V-cycle in stencil-and-reduction operations and
    avoids introducing an FFT solely for the bottom solve.

    The operator is left unpinned, like ``fft``: the periodic-plus-Neumann
    null space is one constant vector, and because ``-D G`` is symmetric its
    range is exactly the orthogonal complement of that constant, so a
    compatible (zero-mean) right-hand side keeps every PCG residual zero-mean
    automatically, with no explicit projection needed inside the iteration.
    """
    resolved = np.dtype(dtype)
    if tolerance is None:
        tolerance = default_tolerance(resolved)
    from jax.scipy.sparse.linalg import cg

    levels, factors = _build_gmg_levels(grid)
    diagonals = [_diagonal_stencil(level, resolved) for level in levels[:-1]]
    coarse_grid = levels[-1]
    coarse_shape = (coarse_grid.nz, coarse_grid.ny, coarse_grid.nx)
    shape = (grid.nz, grid.ny, grid.nx)

    def coarse_solve(rhs: jnp.ndarray) -> jnp.ndarray:
        """Approximately invert the coarsest operator with five CG iterations."""
        # A one-cell Neumann grid contains only the constant null mode. Its
        # compatible right-hand side and correction are identically zero, so
        # tracing five vacuous CG passes would only add launch overhead.
        if coarse_grid.cell_count == 1:
            return jnp.zeros_like(rhs)
        # Restriction preserves compatibility analytically; remove reduction
        # round-off from the constant null mode before applying CG.
        compatible = (rhs - jnp.mean(rhs)).reshape(-1)

        def matvec(flat: jnp.ndarray) -> jnp.ndarray:
            return _apply_laplacian(flat.reshape(coarse_shape), coarse_grid).reshape(-1)

        def iteration(_, state):
            solution, residual, direction, residual_norm = state
            applied = matvec(direction)
            denominator = jnp.vdot(direction, applied).real
            active = denominator > 0.0
            safe_denominator = jnp.where(active, denominator, 1.0)
            step = jnp.where(active, residual_norm / safe_denominator, 0.0)
            solution = solution + step * direction
            next_residual = residual - step * applied
            next_residual = next_residual - jnp.mean(next_residual)
            next_norm = jnp.vdot(next_residual, next_residual).real
            safe_norm = jnp.where(residual_norm > 0.0, residual_norm, 1.0)
            beta = jnp.where(residual_norm > 0.0, next_norm / safe_norm, 0.0)
            direction = next_residual + beta * direction
            return solution, next_residual, direction, next_norm

        zeros = jnp.zeros_like(compatible)
        initial_norm = jnp.vdot(compatible, compatible).real
        solution, _, _, _ = jax.lax.fori_loop(
            0,
            5,
            iteration,
            (zeros, compatible, compatible, initial_norm),
        )
        return (solution - jnp.mean(solution)).reshape(coarse_shape)

    def smooth(pressure: jnp.ndarray, rhs: jnp.ndarray, level: int, sweeps: int) -> jnp.ndarray:
        for _ in range(sweeps):
            residual = rhs - _apply_laplacian(pressure, levels[level])
            pressure = pressure + omega * residual / diagonals[level]
        return pressure

    def v_cycle(rhs: jnp.ndarray, level: int) -> jnp.ndarray:
        if level == len(levels) - 1:
            return coarse_solve(rhs)
        pressure = smooth(jnp.zeros_like(rhs), rhs, level, presweeps)
        residual = rhs - _apply_laplacian(pressure, levels[level])
        coarse_correction = v_cycle(_restrict(residual, factors[level]), level + 1)
        pressure = pressure + _prolong(coarse_correction, factors[level])
        return smooth(pressure, rhs, level, postsweeps)

    def precondition(flat: jnp.ndarray) -> jnp.ndarray:
        rhs = flat.reshape(shape)
        pressure = v_cycle(rhs, 0)
        for _ in range(cycles_per_precondition - 1):
            residual = rhs - _apply_laplacian(pressure, grid)
            pressure = pressure + v_cycle(residual, 0)
        return (pressure - jnp.mean(pressure)).reshape(-1)

    def solve(right_hand_side: jnp.ndarray) -> jnp.ndarray:
        def matvec(flat: jnp.ndarray) -> jnp.ndarray:
            return _apply_laplacian(flat.reshape(shape), grid).reshape(-1)

        solution, _ = cg(
            matvec,
            right_hand_side,
            M=precondition,
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
    if backend in ("fft", "gmg"):
        # Both handle the null space themselves (an explicit eigenmode for
        # ``fft``, symmetry of the unpinned operator for ``gmg``), so the
        # assembled matrix kept for bookkeeping stays unpinned.
        matrix = assemble_pressure_matrix(grid, dtype=dtype, reference_cell=None)
        if backend == "fft":
            solver = build_fft_solver(grid, dtype=dtype, **dict(config or {}))
        else:
            solver = build_gmg_solver(grid, dtype=dtype, **dict(config or {}))
        return PressurePoisson(grid, matrix, solver)
    matrix = assemble_pressure_matrix(
        grid,
        dtype=dtype,
        reference_cell=reference_cell,
    )
    if backend == "amg":
        solver = build_amg_solver(matrix, config=config)
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
    "build_fft_solver",
    "build_gmg_solver",
    "build_pressure_poisson",
    "default_tolerance",
    "matrix_vector_product",
    "project",
]
