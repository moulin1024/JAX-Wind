"""The pressure Poisson system of the staggered projection.

Three backends solve the pressure Poisson system: ``amg`` hands the assembled
operator to JAX-AMG (:mod:`jaxamg`) on the GPU, which needs AmgX and a CUDA
build; ``gmg`` runs conjugate gradients preconditioned by a geometric
multigrid V-cycle built directly from the mesh (see :func:`build_gmg_solver`)
-- pure JAX and matrix-free, with a mesh-independent iteration count and, being
built from local stencils rather than a global FFT, friendlier to a
domain-decomposed, multi-GPU mesh; and ``fft`` transforms the periodic
horizontal directions and solves the remaining Neumann tridiagonal
systems directly in one shot -- valid only because the mesh is periodic in x
and y (see below), but then exact to floating point with no iteration at all.

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

from jaxwind.domain.grid import AnalyticalGrid, Grid, UniformGrid

from jaxwind.metrics import cell_volumes
from jaxwind.numerics.discretization import divergence, pressure_gradient
from jaxwind.state import StaggeredVelocity


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


def _assemble_mapped_pressure_matrix(
    grid: Grid,
    *,
    dtype: str,
    periodic_x: bool,
    periodic_y: bool,
    reference_cell: int | None,
) -> SparseMatrix:
    """Assemble the symmetric, volume-integrated mapped pressure operator."""
    resolved = np.dtype(dtype)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    plane = ny * nx
    k, j, i = (
        index.ravel()
        for index in np.meshgrid(
            np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
        )
    )
    rows = (k * plane + j * nx + i).astype(np.int64)
    hx, hy, hz = grid.x_widths, grid.y_widths, grid.z_widths
    dx_periodic = 0.5 * (hx + np.roll(hx, 1))
    dy_periodic = 0.5 * (hy + np.roll(hy, 1))
    dx_faces = np.concatenate(
        ((0.5 * hx[0],), 0.5 * (hx[:-1] + hx[1:]), (0.5 * hx[-1],))
    )
    dy_faces = np.concatenate(
        ((0.5 * hy[0],), 0.5 * (hy[:-1] + hy[1:]), (0.5 * hy[-1],))
    )
    dz_faces = np.concatenate(
        ((0.5 * hz[0],), 0.5 * (hz[:-1] + hz[1:]), (0.5 * hz[-1],))
    )
    diagonal = np.zeros(rows.size, dtype=np.float64)
    row_blocks: list[np.ndarray] = []
    column_blocks: list[np.ndarray] = []
    value_blocks: list[np.ndarray] = []

    def connect(
        mask: np.ndarray, neighbor: np.ndarray, conductance: np.ndarray
    ) -> None:
        diagonal[mask] += conductance[mask]
        row_blocks.append(rows[mask])
        column_blocks.append(neighbor[mask].astype(np.int64))
        value_blocks.append(-conductance[mask])

    area_x = hz[k] * hy[j]
    if periodic_x:
        left_x = area_x / dx_periodic[i]
        right_x = area_x / dx_periodic[(i + 1) % nx]
        all_cells = np.ones(rows.size, dtype=bool)
        connect(all_cells, k * plane + j * nx + (i - 1) % nx, left_x)
        connect(all_cells, k * plane + j * nx + (i + 1) % nx, right_x)
        transverse = all_cells
    else:
        lower = i > 0
        upper = i < nx - 1
        left_x = area_x / dx_faces[i]
        right_x = area_x / dx_faces[i + 1]
        connect(lower, rows - 1, left_x)
        connect(upper, rows + 1, right_x)
        outlet = i == nx - 1
        diagonal[outlet] += area_x[outlet] / dx_faces[-1]
        transverse = (i > 0) & (i < nx - 1)

    area_y = hz[k] * hx[i]
    if periodic_y:
        lower_y = area_y / dy_periodic[j]
        upper_y = area_y / dy_periodic[(j + 1) % ny]
        connect(
            transverse,
            k * plane + ((j - 1) % ny) * nx + i,
            lower_y,
        )
        connect(
            transverse,
            k * plane + ((j + 1) % ny) * nx + i,
            upper_y,
        )
    else:
        lower = transverse & (j > 0)
        upper = transverse & (j < ny - 1)
        lower_y = area_y / dy_faces[j]
        upper_y = area_y / dy_faces[j + 1]
        connect(lower, rows - nx, lower_y)
        connect(upper, rows + nx, upper_y)

    area_z = hy[j] * hx[i]
    lower = transverse & (k > 0)
    upper = transverse & (k < nz - 1)
    lower_z = area_z / dz_faces[k]
    upper_z = area_z / dz_faces[k + 1]
    connect(lower, rows - plane, lower_z)
    connect(upper, rows + plane, upper_z)

    row_blocks.insert(0, rows)
    column_blocks.insert(0, rows)
    value_blocks.insert(0, diagonal)
    all_rows = np.concatenate(row_blocks)
    all_columns = np.concatenate(column_blocks)
    all_values = np.concatenate(value_blocks)
    if reference_cell is not None:
        interior = (all_rows != reference_cell) & (all_columns != reference_cell)
        pinned_diagonal = float(diagonal[reference_cell])
        all_rows = np.append(all_rows[interior], reference_cell)
        all_columns = np.append(all_columns[interior], reference_cell)
        all_values = np.append(all_values[interior], pinned_diagonal)
    values, columns, indptr = _coalesce_to_csr(
        all_rows, all_columns, all_values, grid.cell_count
    )
    return SparseMatrix(
        values.astype(resolved), columns, indptr, grid.cell_count, reference_cell
    )


def assemble_pressure_matrix(
    grid: Grid,
    *,
    dtype: str = "float64",
    periodic_x: bool = True,
    periodic_y: bool = True,
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
    if not grid.is_uniform:
        return _assemble_mapped_pressure_matrix(
            grid,
            dtype=dtype,
            periodic_x=periodic_x,
            periodic_y=periodic_y,
            reference_cell=reference_cell,
        )
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
    if periodic_x:
        x_diagonal = np.full(rows.size, 2.0 * inverse_dx2)
    else:
        x_diagonal = np.full(rows.size, 2.0 * inverse_dx2)
        x_diagonal[i == 0] = inverse_dx2
        x_diagonal[i == nx - 1] = 3.0 * inverse_dx2
    transverse = np.ones(rows.size, dtype=bool) if periodic_x else (
        (i > 0) & (i < nx - 1)
    )
    if periodic_y:
        y_diagonal = np.full(rows.size, 2.0 * inverse_dy2)
    else:
        y_diagonal = inverse_dy2 * (
            (j > 0).astype(np.float64)
            + (j < ny - 1).astype(np.float64)
        )
    diagonal = x_diagonal + y_diagonal * transverse
    diagonal += inverse_dz2 * (
        has_lower.astype(np.float64) + has_upper
    ) * transverse

    row_blocks = [rows]
    column_blocks = [rows]
    value_blocks = [diagonal]

    if periodic_x:
        for shift in (-1, 1):
            row_blocks.append(rows)
            column_blocks.append(k * plane + j * nx + (i + shift) % nx)
            value_blocks.append(np.full(rows.size, -inverse_dx2))
    else:
        for mask, shift in ((i > 0, -1), (i < nx - 1, 1)):
            row_blocks.append(rows[mask])
            column_blocks.append(rows[mask] + shift)
            value_blocks.append(np.full(int(mask.sum()), -inverse_dx2))
    for side_mask, shift in ((j > 0, -1), (j < ny - 1, 1)):
        mask = transverse if periodic_y else (transverse & side_mask)
        row_blocks.append(rows[mask])
        neighbor_j = (j + shift) % ny if periodic_y else j + shift
        column_blocks.append((k * plane + neighbor_j * nx + i)[mask])
        value_blocks.append(np.full(int(mask.sum()), -inverse_dy2))

    for vertical_mask, shift in ((has_lower, -1), (has_upper, 1)):
        mask = vertical_mask & transverse
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


def _thomas_factor_arrays(lower, diagonal, upper):
    """Precompute the modified diagonal and upper factors for Thomas solves."""
    inverse_first = 1.0 / diagonal[0]
    gamma_first = jnp.zeros_like(inverse_first)
    if diagonal.shape[0] == 1:
        return inverse_first[None], gamma_first[None]

    def factor_row(carry, rows):
        inverse_previous, upper_previous = carry
        lower_row, diagonal_row, upper_row = rows
        gamma_row = upper_previous * inverse_previous
        inverse_row = 1.0 / (diagonal_row - lower_row * gamma_row)
        return (inverse_row, upper_row), (inverse_row, gamma_row)

    _, (inverse_tail, gamma_tail) = jax.lax.scan(
        factor_row,
        (inverse_first, upper[0]),
        (lower[1:], diagonal[1:], upper[1:]),
    )
    return (
        jnp.concatenate((inverse_first[None], inverse_tail), axis=0),
        jnp.concatenate((gamma_first[None], gamma_tail), axis=0),
    )


def _chunked_scan_rows(initial, row_arrays, step, chunk: int):
    """Scan dependent rows while statically unrolling rows within each chunk."""
    row_count = row_arrays[0].shape[0]
    resolved_chunk = min(chunk, row_count)
    chunk_count = row_count // resolved_chunk
    full_count = chunk_count * resolved_chunk
    carry = initial
    pieces = []

    if chunk_count:
        chunked = tuple(
            values[:full_count].reshape(
                (chunk_count, resolved_chunk) + values.shape[1:]
            )
            for values in row_arrays
        )

        def scan_chunk(previous, rows):
            current = previous
            outputs = []
            for index in range(resolved_chunk):
                current = step(
                    current,
                    tuple(values[index] for values in rows),
                )
                outputs.append(current)
            return current, jnp.stack(outputs)

        carry, output = jax.lax.scan(scan_chunk, carry, chunked)
        pieces.append(output.reshape((full_count,) + output.shape[2:]))

    if full_count < row_count:
        tail = []
        for index in range(full_count, row_count):
            carry = step(
                carry,
                tuple(values[index] for values in row_arrays),
            )
            tail.append(carry)
        pieces.append(jnp.stack(tail))

    return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=0)


def _chunked_thomas_solve(
    lower,
    inverse_diagonal,
    gamma,
    right_hand_side,
    *,
    chunk: int,
):
    """Solve z-first batches with the reference solver's chunked Thomas scan."""
    zero = jnp.zeros_like(right_hand_side[0])

    def forward(previous, rows):
        lower_row, inverse_row, rhs_row = rows
        return (rhs_row - lower_row * previous) * inverse_row

    forward_values = _chunked_scan_rows(
        zero,
        (lower, inverse_diagonal, right_hand_side),
        forward,
        chunk,
    )
    gamma_next = jnp.concatenate(
        (gamma[1:], jnp.zeros_like(gamma[:1])),
        axis=0,
    )

    def backward(next_value, rows):
        forward_value, gamma_value = rows
        return forward_value - gamma_value * next_value

    reversed_solution = _chunked_scan_rows(
        zero,
        (forward_values[::-1], gamma_next[::-1]),
        backward,
        chunk,
    )
    return reversed_solution[::-1]


def _build_spike_factors(
    lower,
    diagonal,
    upper,
    *,
    block_size: int,
    thomas_chunk: int,
):
    """Factor independent blocks and their reduced SPIKE interface system."""
    vertical_size = diagonal.shape[0]
    block_count = vertical_size // block_size

    def blocked(values):
        return values.reshape(
            (block_count, block_size) + values.shape[1:]
        ).swapaxes(0, 1)

    blocked_lower = blocked(lower)
    blocked_diagonal = blocked(diagonal)
    blocked_upper = blocked(upper)
    local_lower = blocked_lower.at[0].set(0.0)
    local_upper = blocked_upper.at[-1].set(0.0)
    inverse_diagonal, gamma = _thomas_factor_arrays(
        local_lower,
        blocked_diagonal,
        local_upper,
    )

    left_basis = jnp.zeros_like(blocked_diagonal).at[0].set(
        blocked_lower[0]
    )
    right_basis = jnp.zeros_like(blocked_diagonal).at[-1].set(
        blocked_upper[-1]
    )
    left_spike = _chunked_thomas_solve(
        local_lower,
        inverse_diagonal,
        gamma,
        left_basis,
        chunk=thomas_chunk,
    )
    right_spike = _chunked_thomas_solve(
        local_lower,
        inverse_diagonal,
        gamma,
        right_basis,
        chunk=thomas_chunk,
    )

    # The reduced interface matrix has two unknowns per block: the first and
    # last vertical values. Its off-diagonal 2x2 blocks contain only one
    # nonzero column, so retain the reference solver's six scalar factors
    # instead of forming a dense batched matrix.
    left_first, left_last = left_spike[0], left_spike[-1]
    right_first, right_last = right_spike[0], right_spike[-1]
    zero = jnp.zeros_like(left_first[0])

    def factor_interface(previous_c1, spike_rows):
        w_first, w_last, v_first, v_last = spike_rows
        inverse_pivot = 1.0 / (1.0 - w_first * previous_c1)
        g10 = w_last * previous_c1 * inverse_pivot
        a0 = inverse_pivot * w_first
        a1 = g10 * w_first + w_last
        c0 = inverse_pivot * v_first
        c1 = g10 * v_first + v_last
        return c1, (inverse_pivot, g10, a0, a1, c0, c1)

    _, interface_factors = jax.lax.scan(
        factor_interface,
        zero,
        (left_first, left_last, right_first, right_last),
    )
    return (
        local_lower,
        inverse_diagonal,
        gamma,
        left_spike,
        right_spike,
        interface_factors,
    )


def _spike_solve(
    factors,
    right_hand_side,
    *,
    block_size: int,
    thomas_chunk: int,
):
    """Apply a pre-factored SPIKE solve to a z-first batch of systems."""
    (
        local_lower,
        inverse_diagonal,
        gamma,
        left_spike,
        right_spike,
        interface_factors,
    ) = factors
    block_count = right_hand_side.shape[0] // block_size
    blocked_rhs = right_hand_side.reshape(
        (block_count, block_size) + right_hand_side.shape[1:]
    ).swapaxes(0, 1)
    local_solution = _chunked_thomas_solve(
        local_lower,
        inverse_diagonal,
        gamma,
        blocked_rhs,
        chunk=thomas_chunk,
    )
    g00, g10, a0, a1, c0, c1 = interface_factors
    interface_rhs = (local_solution[0], local_solution[-1])
    zero = jnp.zeros_like(interface_rhs[0][0])

    def forward(previous_last, rows):
        first_rhs, last_rhs, row_g00, row_g10, row_a0, row_a1 = rows
        first = row_g00 * first_rhs - row_a0 * previous_last
        last = (
            row_g10 * first_rhs + last_rhs - row_a1 * previous_last
        )
        return last, (first, last)

    _, (forward_first, forward_last) = jax.lax.scan(
        forward,
        zero,
        (*interface_rhs, g00, g10, a0, a1),
    )

    def backward(next_first, rows):
        first_value, last_value, row_c0, row_c1 = rows
        first = first_value - row_c0 * next_first
        last = last_value - row_c1 * next_first
        return first, (first, last)

    _, reversed_interface = jax.lax.scan(
        backward,
        zero,
        (
            forward_first[::-1],
            forward_last[::-1],
            c0[::-1],
            c1[::-1],
        ),
    )
    interface_first, interface_last = (
        values[::-1] for values in reversed_interface
    )
    previous_last = jnp.concatenate(
        (jnp.zeros_like(interface_last[:1]), interface_last[:-1]), axis=0
    )
    next_first = jnp.concatenate(
        (interface_first[1:], jnp.zeros_like(interface_first[:1])), axis=0
    )
    blocked_solution = (
        local_solution
        - left_spike * previous_last[None]
        - right_spike * next_first[None]
    )
    return blocked_solution.swapaxes(0, 1).reshape(right_hand_side.shape)


def build_fft_solver(
    grid: Grid,
    *,
    dtype: str = "float64",
    method: str = "thomas",
    thomas_chunk: int = 16,
    spike_block_size: int = 32,
) -> LinearSolver:
    """Solve with a horizontal FFT and batched vertical tridiagonal solves.

    The mesh is periodic in x and y and Neumann in z, so a real 2-D FFT
    diagonalises the uniform horizontal part exactly. Each horizontal mode
    leaves one ``nz x nz`` Neumann tridiagonal system. ``method="thomas"``
    uses a pre-factored chunked Thomas sweep; ``method="spike"`` splits that
    sweep into independent vertical blocks and reconnects their two endpoint
    values with a structured reduced solve. Both paths use the custom Thomas
    kernel and never call JAX's built-in tridiagonal solver.

    Horizontal stretching is incompatible with Fourier diagonalisation.
    Vertical stretching is supported because it only changes the coefficients
    of the independent tridiagonal system for each horizontal mode.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    resolved = np.dtype(dtype)
    if method not in {"thomas", "spike"}:
        raise ValueError("FFT method must be 'thomas' or 'spike'")
    if thomas_chunk <= 0:
        raise ValueError("thomas_chunk must be positive")
    if spike_block_size < 2:
        raise ValueError("spike_block_size must be at least two")
    if method == "spike" and nz % spike_block_size:
        raise ValueError(
            "spike_block_size must divide the number of vertical cells"
        )
    uniform_x = np.allclose(
        grid.x_widths, grid.x_widths[0], rtol=1.0e-13, atol=0.0
    )
    uniform_y = np.allclose(
        grid.y_widths, grid.y_widths[0], rtol=1.0e-13, atol=0.0
    )
    if not uniform_x or not uniform_y:
        raise ValueError(
            "the FFT pressure backend requires uniform x and y spacing"
        )
    inverse_dx2 = 1.0 / grid.dx**2
    inverse_dy2 = 1.0 / grid.dy**2

    # rfft2 keeps the full range of ky but only the non-redundant half of kx;
    # cos(2 pi k / n) is symmetric under k -> n - k.
    kx = np.arange(nx // 2 + 1)
    ky = np.arange(ny)
    lambda_x = 2.0 * (1.0 - np.cos(2.0 * np.pi * kx / nx)) * inverse_dx2
    lambda_y = 2.0 * (1.0 - np.cos(2.0 * np.pi * ky / ny)) * inverse_dy2
    horizontal = lambda_y[:, None] + lambda_x[None, :]

    # Uniform meshes use the pointwise -D G system. Mapped meshes use the
    # symmetric volume-integrated system prepared by PressurePoisson: the
    # conductance through an interior z face is its horizontal area divided
    # by the distance between adjacent cell centres.
    if grid.is_uniform:
        inverse_dz2 = 1.0 / grid.dz**2
        vertical_diagonal = np.full(nz, 2.0 * inverse_dz2, dtype=resolved)
        if nz == 1:
            vertical_diagonal[0] = 0.0
        else:
            vertical_diagonal[0] = inverse_dz2
            vertical_diagonal[-1] = inverse_dz2
        lower_z = np.full(nz, -inverse_dz2, dtype=resolved)
        upper_z = np.full(nz, -inverse_dz2, dtype=resolved)
        lower_z[0] = 0.0
        upper_z[-1] = 0.0
        horizontal_weight = np.ones(nz, dtype=resolved)
    else:
        area = grid.dx * grid.dy
        widths_z = np.asarray(grid.z_widths, dtype=resolved)
        lower_z = np.zeros(nz, dtype=resolved)
        upper_z = np.zeros(nz, dtype=resolved)
        if nz > 1:
            centre_distance = 0.5 * (widths_z[:-1] + widths_z[1:])
            conductance = area / centre_distance
            lower_z[1:] = -conductance
            upper_z[:-1] = -conductance
        vertical_diagonal = -(lower_z + upper_z)
        horizontal_weight = area * widths_z

    vertical_diagonal = jnp.asarray(vertical_diagonal)
    lower_z = jnp.asarray(lower_z)
    upper_z = jnp.asarray(upper_z)
    horizontal_weight = jnp.asarray(horizontal_weight)
    horizontal = jnp.asarray(horizontal, resolved)

    # Use the reference solver's z-first representation. Factors depend only
    # on the mesh and horizontal wave number, so construct them once rather
    # than rebuilding the diagonal inside every pressure solve.
    diagonal = (
        vertical_diagonal[:, None, None]
        + horizontal_weight[:, None, None] * horizontal[None, :, :]
    )
    lower = jnp.broadcast_to(lower_z[:, None, None], diagonal.shape)
    upper = jnp.broadcast_to(upper_z[:, None, None], diagonal.shape)

    # The sole singular system is the horizontally constant mode. Pin its
    # first vertical unknown; compatibility makes the omitted equation
    # redundant, and PressurePoisson restores the zero-mean gauge.
    diagonal = diagonal.at[0, 0, 0].set(1.0)
    upper = upper.at[0, 0, 0].set(0.0)
    if method == "thomas":
        inverse_diagonal, gamma = _thomas_factor_arrays(
            lower,
            diagonal,
            upper,
        )
        factors = None
    else:
        inverse_diagonal = gamma = None
        factors = _build_spike_factors(
            lower,
            diagonal,
            upper,
            block_size=spike_block_size,
            thomas_chunk=thomas_chunk,
        )

    def solve(right_hand_side: jnp.ndarray) -> jnp.ndarray:
        field = right_hand_side.reshape(nz, ny, nx)
        spectrum = jnp.fft.rfft2(field, axes=(1, 2))
        spectrum = spectrum.at[0, 0, 0].set(0.0)
        if method == "thomas":
            solution_spectrum = _chunked_thomas_solve(
                lower,
                inverse_diagonal,
                gamma,
                spectrum,
                chunk=thomas_chunk,
            )
        else:
            solution_spectrum = _spike_solve(
                factors,
                spectrum,
                block_size=spike_block_size,
                thomas_chunk=thomas_chunk,
            )
        solution = jnp.fft.irfft2(
            solution_spectrum, s=(ny, nx), axes=(1, 2)
        )
        return solution.reshape(-1)

    return solve


def _apply_laplacian(
    pressure: jnp.ndarray,
    grid: Grid,
    *,
    periodic_x: bool = True,
    periodic_y: bool = True,
    open_x_low: bool = False,
    open_y: bool = False,
    volume_integrated: bool | None = None,
) -> jnp.ndarray:
    """Apply the matrix-free negative pressure Laplacian."""
    applied = -divergence(
        pressure_gradient(
            pressure,
            grid,
            periodic_x=periodic_x,
            periodic_y=periodic_y,
            open_x_low=open_x_low, open_y=open_y,
        ),
        grid,
    )
    integrated = (
        not grid.is_uniform
        if volume_integrated is None
        else volume_integrated
    )
    if not integrated:
        return applied
    return cell_volumes(grid, pressure.dtype) * applied


def _mapped_diagonal_stencil(
    grid: Grid,
    dtype: np.dtype,
    *,
    periodic_x: bool,
    periodic_y: bool,
    open_x_low: bool = False,
    open_y: bool = False,
) -> jnp.ndarray:
    """Diagonal of the symmetric volume-integrated mapped operator."""
    hx, hy, hz = grid.x_widths, grid.y_widths, grid.z_widths
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx_periodic = 0.5 * (hx + np.roll(hx, 1))
    dy_periodic = 0.5 * (hy + np.roll(hy, 1))
    dx_faces = np.concatenate(
        ((0.5 * hx[0],), 0.5 * (hx[:-1] + hx[1:]), (0.5 * hx[-1],))
    )
    dy_faces = np.concatenate(
        ((0.5 * hy[0],), 0.5 * (hy[:-1] + hy[1:]), (0.5 * hy[-1],))
    )
    dz_faces = np.concatenate(
        ((0.5 * hz[0],), 0.5 * (hz[:-1] + hz[1:]), (0.5 * hz[-1],))
    )
    if periodic_x:
        x_factor = 1.0 / dx_periodic + 1.0 / np.roll(dx_periodic, -1)
        transverse = np.ones(nx, dtype=np.float64)
    else:
        x_factor = np.zeros(nx, dtype=np.float64)
        if nx > 1:
            x_factor[1:] += 1.0 / dx_faces[1:-1]
            x_factor[:-1] += 1.0 / dx_faces[1:-1]
        x_factor[-1] += 1.0 / dx_faces[-1]
        if open_x_low:
            x_factor[0] += 1.0 / dx_faces[0]
        transverse = ((np.arange(nx) > 0) & (np.arange(nx) < nx - 1)).astype(
            np.float64
        )
    x_diagonal = hz[:, None, None] * hy[None, :, None] * x_factor[None, None, :]
    if periodic_y:
        y_factor = 1.0 / dy_periodic + 1.0 / np.roll(dy_periodic, -1)
    else:
        y_factor = np.zeros(ny, dtype=np.float64)
        if ny > 1:
            y_factor[1:] += 1.0 / dy_faces[1:-1]
            y_factor[:-1] += 1.0 / dy_faces[1:-1]
    y_diagonal = (
        hz[:, None, None]
        * hx[None, None, :]
        * y_factor[None, :, None]
        * transverse[None, None, :]
    )
    z_factor = np.zeros(nz, dtype=np.float64)
    if nz > 1:
        z_factor[1:] += 1.0 / dz_faces[1:-1]
        z_factor[:-1] += 1.0 / dz_faces[1:-1]
    z_diagonal = (
        hy[None, :, None]
        * hx[None, None, :]
        * z_factor[:, None, None]
        * transverse[None, None, :]
    )
    return jnp.asarray(x_diagonal + y_diagonal + z_diagonal, dtype)


def _diagonal_stencil(
    grid: Grid,
    dtype: np.dtype,
    *,
    periodic_x: bool = True,
    periodic_y: bool = True,
    open_x_low: bool = False,
    open_y: bool = False,
    volume_integrated: bool | None = None,
) -> jnp.ndarray:
    """Diagonal of the periodic or mixed-boundary negative Laplacian."""
    integrated = (
        not grid.is_uniform
        if volume_integrated is None
        else volume_integrated
    )
    if integrated:
        return _mapped_diagonal_stencil(
            grid, dtype, periodic_x=periodic_x, periodic_y=periodic_y,
            open_x_low=open_x_low, open_y=open_y,
        )
    inverse_dx2 = 1.0 / grid.dx**2
    inverse_dy2 = 1.0 / grid.dy**2
    inverse_dz2 = 1.0 / grid.dz**2
    k = np.arange(grid.nz)
    has_lower = k > 0
    has_upper = k < grid.nz - 1
    vertical = inverse_dz2 * (
        has_lower.astype(np.float64) + has_upper.astype(np.float64)
    )
    if periodic_x:
        horizontal_x = np.full(grid.nx, 2.0 * inverse_dx2)
    else:
        horizontal_x = np.full(grid.nx, 2.0 * inverse_dx2)
        horizontal_x[0] = inverse_dx2
        horizontal_x[-1] = 3.0 * inverse_dx2
        if open_x_low:
            horizontal_x[0] = 3.0 * inverse_dx2
            if grid.nx == 1:
                horizontal_x[0] = 4.0 * inverse_dx2
    transverse = (
        np.ones(grid.nx)
        if periodic_x
        else ((np.arange(grid.nx) > 0) & (np.arange(grid.nx) < grid.nx - 1))
    )
    horizontal_y = (
        np.full(grid.ny, 2.0 * inverse_dy2)
        if periodic_y
        else inverse_dy2
        * (
            (np.arange(grid.ny) > 0).astype(np.float64)
            + (np.arange(grid.ny) < grid.ny - 1).astype(np.float64)
        )
    )
    if open_y:
        horizontal_y[0] += 2.0 * inverse_dy2
        horizontal_y[-1] += 2.0 * inverse_dy2
    diagonal = horizontal_x[None, None, :] + transverse[None, None, :] * (
        vertical[:, None, None] + horizontal_y[None, :, None]
    )
    return jnp.asarray(diagonal, dtype)


def _coarsening_factors(grid: Grid) -> tuple[int, int, int]:
    """Per-axis factor-two agglomeration, one axis at a time as it allows it.

    An axis stops coarsening as soon as its cell count is odd or one, which is
    what lets a mesh with mixed factors (say ``12 = 4 * 3``) coarsen as far as
    each direction supports rather than stalling the whole hierarchy on the one
    direction that cannot be halved evenly.
    """
    return tuple(2 if n > 1 and n % 2 == 0 else 1 for n in (grid.nx, grid.ny, grid.nz))


def _anisotropy_aware_coarsening_factors(
    grid: Grid,
) -> tuple[int, int, int]:
    """Coarsen the finest physical directions before the wider ones."""
    counts = (grid.nx, grid.ny, grid.nz)
    spacings = (
        float(np.min(grid.x_widths)),
        float(np.min(grid.y_widths)),
        float(np.min(grid.z_widths)),
    )
    eligible = tuple(n > 1 and n % 2 == 0 for n in counts)
    if not any(eligible):
        return (1, 1, 1)
    finest = min(h for h, can_coarsen in zip(spacings, eligible) if can_coarsen)
    threshold = finest * (1.0 + 32.0 * np.finfo(float).eps)
    return tuple(
        2 if can_coarsen and h <= threshold else 1
        for h, can_coarsen in zip(spacings, eligible)
    )


def _coarsen_grid(grid: Grid, factors: tuple[int, int, int]) -> Grid:
    factor_x, factor_y, factor_z = factors
    if isinstance(grid, AnalyticalGrid):
        return grid.coarsen(factors)
    return UniformGrid(
        grid.nx // factor_x,
        grid.ny // factor_y,
        grid.nz // factor_z,
        grid.lx,
        grid.ly,
        grid.lz,
    )


def _build_gmg_levels(
    grid: Grid,
    *,
    anisotropy_aware: bool = True,
) -> tuple[list[Grid], list[tuple[int, int, int]]]:
    """Coarsen by cell-agglomeration until no axis can be halved further."""
    levels = [grid]
    factors = []
    current = grid
    while True:
        current_factors = (
            _anisotropy_aware_coarsening_factors(current)
            if anisotropy_aware
            else _coarsening_factors(current)
        )
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
        return jnp.concatenate(
            (values[tuple(edge)], values[tuple(interior)]), axis=axis
        )
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


def _axis_geometry(grid: Grid, axis: int) -> tuple[np.ndarray, float]:
    if axis == 0:
        return grid.z_centers, grid.lz
    if axis == 1:
        return grid.y_centers, grid.ly
    if axis == 2:
        return grid.x_centers, grid.lx
    raise ValueError("transfer axis must be zero, one, or two")


def _metric_prolongation_weights(
    fine_grid: Grid,
    coarse_grid: Grid,
    axis: int,
    *,
    periodic: bool,
    dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Weights of each coarse value in its lower and upper fine children."""
    fine_centers, length = _axis_geometry(fine_grid, axis)
    coarse_centers, _ = _axis_geometry(coarse_grid, axis)
    if fine_centers.size != 2 * coarse_centers.size:
        raise ValueError("metric transfer requires factor-two coarsening")

    lower_fine = fine_centers[::2]
    upper_fine = fine_centers[1::2]
    lower_current = np.ones(coarse_centers.size, dtype=np.float64)
    upper_current = np.ones(coarse_centers.size, dtype=np.float64)
    if periodic:
        previous = np.roll(coarse_centers, 1)
        previous[0] -= length
        following = np.roll(coarse_centers, -1)
        following[-1] += length
        lower_current = (lower_fine - previous) / (
            coarse_centers - previous
        )
        upper_current = (following - upper_fine) / (
            following - coarse_centers
        )
    else:
        lower_current[1:] = (
            lower_fine[1:] - coarse_centers[:-1]
        ) / (coarse_centers[1:] - coarse_centers[:-1])
        upper_current[:-1] = (
            coarse_centers[1:] - upper_fine[:-1]
        ) / (coarse_centers[1:] - coarse_centers[:-1])
    tolerance = 128.0 * np.finfo(np.float64).eps
    if (
        np.any(lower_current < -tolerance)
        or np.any(lower_current > 1.0 + tolerance)
        or np.any(upper_current < -tolerance)
        or np.any(upper_current > 1.0 + tolerance)
    ):
        raise ValueError("fine centers are not nested between coarse centers")
    return (
        jnp.asarray(np.clip(lower_current, 0.0, 1.0), dtype),
        jnp.asarray(np.clip(upper_current, 0.0, 1.0), dtype),
    )


def _metric_prolong_axis(
    values: jnp.ndarray,
    fine_grid: Grid,
    coarse_grid: Grid,
    axis: int,
    *,
    periodic: bool,
) -> jnp.ndarray:
    """Interpolate coarse values to fine centers in physical coordinates."""
    lower_current, upper_current = _metric_prolongation_weights(
        fine_grid,
        coarse_grid,
        axis,
        periodic=periodic,
        dtype=values.dtype,
    )
    weight_shape = [1] * values.ndim
    weight_shape[axis] = lower_current.size
    lower_current = lower_current.reshape(weight_shape)
    upper_current = upper_current.reshape(weight_shape)
    previous = _neighbor(values, axis, 1, periodic=periodic)
    following = _neighbor(values, axis, -1, periodic=periodic)
    lower = lower_current * values + (1.0 - lower_current) * previous
    upper = upper_current * values + (1.0 - upper_current) * following
    shape = list(values.shape)
    shape[axis] *= 2
    return jnp.stack((lower, upper), axis=axis + 1).reshape(shape)


def _metric_restrict_axis(
    integrated_residual: jnp.ndarray,
    fine_grid: Grid,
    coarse_grid: Grid,
    axis: int,
    *,
    periodic: bool,
) -> jnp.ndarray:
    """Restrict mapped residuals with the volume-weighted adjoint.

    The mapped operator carries ``V_f r_f`` rather than the intensive
    residual ``r_f``. Applying ``P.T`` here is therefore equivalent to the
    volume-weighted restriction ``V_c**-1 P.T V_f`` in intensive variables.
    It also keeps the multigrid preconditioner symmetric for outer PCG.
    """
    lower_index = [slice(None)] * integrated_residual.ndim
    upper_index = [slice(None)] * integrated_residual.ndim
    lower_index[axis] = slice(0, None, 2)
    upper_index[axis] = slice(1, None, 2)
    lower = integrated_residual[tuple(lower_index)]
    upper = integrated_residual[tuple(upper_index)]
    lower_current, upper_current = _metric_prolongation_weights(
        fine_grid,
        coarse_grid,
        axis,
        periodic=periodic,
        dtype=integrated_residual.dtype,
    )
    weight_shape = [1] * integrated_residual.ndim
    weight_shape[axis] = lower_current.size
    lower_current = lower_current.reshape(weight_shape)
    upper_current = upper_current.reshape(weight_shape)
    restricted = lower_current * lower + upper_current * upper
    to_previous = (1.0 - lower_current) * lower
    to_following = (1.0 - upper_current) * upper
    if periodic:
        return (
            restricted
            + jnp.roll(to_previous, -1, axis=axis)
            + jnp.roll(to_following, 1, axis=axis)
        )
    zero_shape = list(restricted.shape)
    zero_shape[axis] = 1
    zero = jnp.zeros(zero_shape, integrated_residual.dtype)
    previous_source = [slice(None)] * restricted.ndim
    previous_source[axis] = slice(1, None)
    following_source = [slice(None)] * restricted.ndim
    following_source[axis] = slice(0, -1)
    return (
        restricted
        + jnp.concatenate(
            (to_previous[tuple(previous_source)], zero), axis=axis
        )
        + jnp.concatenate(
            (zero, to_following[tuple(following_source)]), axis=axis
        )
    )


def _restrict(
    residual: jnp.ndarray,
    factors: tuple[int, int, int],
    *,
    periodic_x: bool = True,
    periodic_y: bool = True,
    fine_grid: Grid | None = None,
    coarse_grid: Grid | None = None,
) -> jnp.ndarray:
    """Restrict an intensive uniform or volume-integrated mapped residual."""
    if (fine_grid is None) != (coarse_grid is None):
        raise ValueError("both transfer grids must be supplied together")
    factor_x, factor_y, factor_z = factors
    if factor_z == 2:
        residual = (
            _restrict_axis(residual, 0, periodic=False)
            if fine_grid is None
            else _metric_restrict_axis(
                residual, fine_grid, coarse_grid, 0, periodic=False
            )
        )
    if factor_y == 2:
        residual = (
            _restrict_axis(residual, 1, periodic=periodic_y)
            if fine_grid is None
            else _metric_restrict_axis(
                residual, fine_grid, coarse_grid, 1, periodic=periodic_y
            )
        )
    if factor_x == 2:
        residual = (
            _restrict_axis(residual, 2, periodic=periodic_x)
            if fine_grid is None
            else _metric_restrict_axis(
                residual, fine_grid, coarse_grid, 2, periodic=periodic_x
            )
        )
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


def _prolong(
    correction: jnp.ndarray,
    factors: tuple[int, int, int],
    *,
    periodic_x: bool = True,
    periodic_y: bool = True,
    fine_grid: Grid | None = None,
    coarse_grid: Grid | None = None,
) -> jnp.ndarray:
    """Prolong in computational or mapped physical coordinates."""
    if (fine_grid is None) != (coarse_grid is None):
        raise ValueError("both transfer grids must be supplied together")
    factor_x, factor_y, factor_z = factors
    if factor_z == 2:
        correction = (
            _prolong_axis(correction, 0, periodic=False)
            if fine_grid is None
            else _metric_prolong_axis(
                correction, fine_grid, coarse_grid, 0, periodic=False
            )
        )
    if factor_y == 2:
        correction = (
            _prolong_axis(correction, 1, periodic=periodic_y)
            if fine_grid is None
            else _metric_prolong_axis(
                correction, fine_grid, coarse_grid, 1, periodic=periodic_y
            )
        )
    if factor_x == 2:
        correction = (
            _prolong_axis(correction, 2, periodic=periodic_x)
            if fine_grid is None
            else _metric_prolong_axis(
                correction, fine_grid, coarse_grid, 2, periodic=periodic_x
            )
        )
    return correction


def build_gmg_solver(
    grid: Grid,
    *,
    dtype: str = "float64",
    periodic_x: bool = True,
    periodic_y: bool = True,
    open_x_low: bool = False,
    open_y: bool = False,
    tolerance: float | None = None,
    max_iterations: int = 200,
    presweeps: int = 2,
    postsweeps: int = 2,
    omega: float = 0.8,
    cycles_per_precondition: int = 1,
    anisotropy_aware: bool = True,
) -> LinearSolver:
    """Solve the pressure system with PCG preconditioned by a geometric V-cycle.

    Every level is a coarser :class:`UniformGrid` over the same physical box,
    obtained by agglomerating cells two at a time along each axis that still
    allows it (see :func:`_coarsening_factors`); the operator at every level is
    then just :func:`_apply_laplacian` rediscretised on that coarser mesh --
    the same seven-point stencil :func:`assemble_pressure_matrix` assembles,
    applied matrix-free with :func:`~jaxwind.numerics.discretization.divergence` and
    :func:`~jaxwind.numerics.discretization.pressure_gradient` instead of a sparse
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
    if open_x_low and periodic_x:
        raise ValueError("an upstream pressure outlet requires nonperiodic x")
    if open_y and (periodic_x or periodic_y or not grid.is_uniform):
        raise ValueError("lateral pressure outlets require nonperiodic x/y on a uniform mesh")
    resolved = np.dtype(dtype)
    if tolerance is None:
        tolerance = default_tolerance(resolved)
    from jax.scipy.sparse.linalg import cg

    levels, factors = _build_gmg_levels(
        grid, anisotropy_aware=anisotropy_aware
    )
    # Once the finest operator is volume integrated, retain that
    # normalization even if a mapped coarse level happens to have uniform
    # widths (for example after clustered z coarsens to one cell).
    volume_integrated = not grid.is_uniform
    diagonals = [
        _diagonal_stencil(
            level,
            resolved,
            periodic_x=periodic_x,
            periodic_y=periodic_y,
            open_x_low=open_x_low, open_y=open_y,
            volume_integrated=volume_integrated,
        )
        for level in levels[:-1]
    ]
    coarse_grid = levels[-1]
    coarse_shape = (coarse_grid.nz, coarse_grid.ny, coarse_grid.nx)
    shape = (grid.nz, grid.ny, grid.nx)

    def coarse_solve(rhs: jnp.ndarray) -> jnp.ndarray:
        """Approximately invert the coarsest operator with five CG iterations."""
        # A one-cell Neumann grid contains only the constant null mode. Its
        # compatible right-hand side and correction are identically zero, so
        # tracing five vacuous CG passes would only add launch overhead.
        if periodic_x and coarse_grid.cell_count == 1:
            return jnp.zeros_like(rhs)
        # Restriction preserves compatibility analytically; remove reduction
        # round-off from the constant null mode before applying CG.
        compatible = (
            rhs - jnp.mean(rhs) if periodic_x else rhs
        ).reshape(-1)

        def matvec(flat: jnp.ndarray) -> jnp.ndarray:
            return _apply_laplacian(
                flat.reshape(coarse_shape),
                coarse_grid,
                periodic_x=periodic_x,
                periodic_y=periodic_y,
                open_x_low=open_x_low, open_y=open_y,
                volume_integrated=volume_integrated,
            ).reshape(-1)

        def iteration(_, state):
            solution, residual, direction, residual_norm = state
            applied = matvec(direction)
            denominator = jnp.vdot(direction, applied).real
            active = denominator > 0.0
            safe_denominator = jnp.where(active, denominator, 1.0)
            step = jnp.where(active, residual_norm / safe_denominator, 0.0)
            solution = solution + step * direction
            next_residual = residual - step * applied
            if periodic_x:
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
        if periodic_x:
            solution = solution - jnp.mean(solution)
        return solution.reshape(coarse_shape)

    def smooth(
        pressure: jnp.ndarray, rhs: jnp.ndarray, level: int, sweeps: int
    ) -> jnp.ndarray:
        for _ in range(sweeps):
            residual = rhs - _apply_laplacian(
                pressure,
                levels[level],
                periodic_x=periodic_x,
                periodic_y=periodic_y,
                open_x_low=open_x_low, open_y=open_y,
                volume_integrated=volume_integrated,
            )
            pressure = pressure + omega * residual / diagonals[level]
        return pressure

    def v_cycle(rhs: jnp.ndarray, level: int) -> jnp.ndarray:
        if level == len(levels) - 1:
            return coarse_solve(rhs)
        pressure = smooth(jnp.zeros_like(rhs), rhs, level, presweeps)
        residual = rhs - _apply_laplacian(
            pressure,
            levels[level],
            periodic_x=periodic_x,
            periodic_y=periodic_y,
            open_x_low=open_x_low, open_y=open_y,
            volume_integrated=volume_integrated,
        )
        coarse_rhs = _restrict(
            residual,
            factors[level],
            periodic_x=periodic_x,
            periodic_y=periodic_y,
            fine_grid=levels[level] if volume_integrated else None,
            coarse_grid=(
                levels[level + 1] if volume_integrated else None
            ),
        )
        coarse_correction = v_cycle(coarse_rhs, level + 1)
        pressure = pressure + _prolong(
            coarse_correction,
            factors[level],
            periodic_x=periodic_x,
            periodic_y=periodic_y,
            fine_grid=levels[level] if volume_integrated else None,
            coarse_grid=(
                levels[level + 1] if volume_integrated else None
            ),
        )
        return smooth(pressure, rhs, level, postsweeps)

    def precondition(flat: jnp.ndarray) -> jnp.ndarray:
        rhs = flat.reshape(shape)
        pressure = v_cycle(rhs, 0)
        for _ in range(cycles_per_precondition - 1):
            residual = rhs - _apply_laplacian(
                pressure,
                grid,
                periodic_x=periodic_x,
                periodic_y=periodic_y,
                open_x_low=open_x_low, open_y=open_y,
                volume_integrated=volume_integrated,
            )
            pressure = pressure + v_cycle(residual, 0)
        if periodic_x:
            pressure = pressure - jnp.mean(pressure)
        return pressure.reshape(-1)

    def solve(
        right_hand_side: jnp.ndarray,
        initial_guess: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        def matvec(flat: jnp.ndarray) -> jnp.ndarray:
            return _apply_laplacian(
                flat.reshape(shape),
                grid,
                periodic_x=periodic_x,
                periodic_y=periodic_y,
                open_x_low=open_x_low, open_y=open_y,
                volume_integrated=volume_integrated,
            ).reshape(-1)

        solution, _ = cg(
            matvec,
            right_hand_side,
            M=precondition,
            x0=initial_guess,
            tol=tolerance,
            atol=0.0,
            maxiter=max_iterations,
        )
        return solution

    return solve


@dataclass(frozen=True, slots=True)
class PressurePoisson:
    """The assembled pressure operator together with its linear solver."""

    grid: Grid
    matrix: SparseMatrix
    linear_solver: LinearSolver
    periodic_x: bool = True
    periodic_y: bool = True
    open_x_low: bool = False
    open_y: bool = False

    def solve(
        self,
        right_hand_side: jnp.ndarray,
        initial_pressure: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Return the cell-centred solution of ``D G p = rhs``."""
        if right_hand_side.shape != (self.grid.nz, self.grid.ny, self.grid.nx):
            raise ValueError("the pressure right-hand side must be cell centred")
        flat = self._prepare(right_hand_side)
        if initial_pressure is None:
            solution = self.linear_solver(flat)
        else:
            initial = initial_pressure.reshape(-1)
            if self.periodic_x:
                initial = initial - jnp.mean(initial)
            solution = self.linear_solver(flat, initial)
        if self.periodic_x:
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
        applied = divergence(pressure_gradient(
            pressure, self.grid, periodic_x=self.periodic_x, periodic_y=self.periodic_y,
            open_x_low=self.open_x_low, open_y=self.open_y,
        ), self.grid)
        error = applied - right_hand_side
        if self.periodic_x:
            if self.grid.is_uniform:
                error = error - jnp.mean(error)
            else:
                volumes = cell_volumes(self.grid, error.dtype)
                error = error - jnp.sum(volumes * error) / jnp.sum(volumes)
        return jnp.linalg.norm(error)

    def _prepare(self, right_hand_side: jnp.ndarray) -> jnp.ndarray:
        """Negate, make compatible with the null space, and drop the gauge row."""
        if self.grid.is_uniform:
            flat = -right_hand_side.reshape(-1)
        else:
            flat = -(right_hand_side * cell_volumes(
                self.grid, right_hand_side.dtype
            )).reshape(-1)
        if self.periodic_x:
            flat = flat - jnp.mean(flat)
        if self.matrix.reference_cell is None:
            return flat
        return flat.at[self.matrix.reference_cell].set(0.0)


def build_pressure_poisson(
    grid: Grid,
    *,
    backend: str = "amg",
    periodic_x: bool = True,
    periodic_y: bool = True,
    open_x_low: bool = False,
    open_y: bool = False,
    dtype: str = "float64",
    reference_cell: int | None = 0,
    config: Mapping[str, Any] | None = None,
) -> PressurePoisson:
    """Assemble the pressure operator and attach the requested solver."""
    if open_x_low and (periodic_x or backend != "gmg"):
        raise ValueError("two pressure outlets require nonperiodic x and GMG")
    if open_y and (backend != "gmg" or periodic_x or periodic_y or not grid.is_uniform):
        raise ValueError("lateral pressure outlets require GMG and nonperiodic uniform x/y")
    if backend == "fft" and (not periodic_x or not periodic_y):
        raise ValueError("the FFT pressure backend requires periodic x and y")
    if backend == "fft":
        uniform_x = np.allclose(
            grid.x_widths, grid.x_widths[0], rtol=1.0e-13, atol=0.0
        )
        uniform_y = np.allclose(
            grid.y_widths, grid.y_widths[0], rtol=1.0e-13, atol=0.0
        )
        if not uniform_x or not uniform_y:
            raise ValueError(
                "the FFT pressure backend requires uniform x and y spacing"
            )
    if backend in ("fft", "gmg"):
        # Both handle the null space themselves (an explicit eigenmode for
        # ``fft``, symmetry of the unpinned operator for ``gmg``), so the
        # assembled matrix kept for bookkeeping stays unpinned.
        if backend == "fft":
            matrix = assemble_pressure_matrix(
                grid,
                dtype=dtype,
                periodic_x=periodic_x,
                periodic_y=periodic_y,
                reference_cell=None,
            )
            solver = build_fft_solver(grid, dtype=dtype, **dict(config or {}))
        else:
            # GMG is matrix-free; avoid an unused multi-gigabyte CSR allocation.
            resolved = np.dtype(dtype)
            matrix = SparseMatrix(
                np.empty((0,), resolved),
                np.empty((0,), np.int32),
                np.zeros((1,), np.int32),
                grid.cell_count,
                None,
            )
            solver = build_gmg_solver(
                grid,
                dtype=dtype,
                periodic_x=periodic_x,
                periodic_y=periodic_y,
                open_x_low=open_x_low, open_y=open_y,
                **dict(config or {}),
            )
        return PressurePoisson(grid, matrix, solver, periodic_x, periodic_y, open_x_low, open_y)
    matrix = assemble_pressure_matrix(
        grid,
        dtype=dtype,
        periodic_x=periodic_x,
        periodic_y=periodic_y,
        reference_cell=reference_cell,
    )
    if backend == "amg":
        solver = build_amg_solver(matrix, config=config)
    else:
        raise ValueError(f"unsupported pressure backend: {backend!r}")
    return PressurePoisson(grid, matrix, solver, periodic_x, periodic_y)


def project(
    velocity: StaggeredVelocity,
    poisson: PressurePoisson,
    dt: float,
    initial_pressure: jnp.ndarray | None = None,
    target_divergence: jnp.ndarray | None = None,
) -> tuple[StaggeredVelocity, jnp.ndarray]:
    """Remove the divergent part of a candidate velocity."""
    grid = poisson.grid
    current_divergence = divergence(velocity, grid)
    target = (
        jnp.zeros_like(current_divergence)
        if target_divergence is None
        else jnp.asarray(target_divergence, current_divergence.dtype)
    )
    if target.shape != current_divergence.shape:
        raise ValueError("target divergence must be cell centred")
    pressure = poisson.solve(
        (current_divergence - target) / dt,
        initial_pressure,
    )
    gradient = pressure_gradient(
        pressure, grid, periodic_x=poisson.periodic_x, periodic_y=poisson.periodic_y,
        open_x_low=poisson.open_x_low, open_y=poisson.open_y,
    )
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
