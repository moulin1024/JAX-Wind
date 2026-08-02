"""Compatible fourth-order Poisson operator built from staggered D4/G4."""

from __future__ import annotations

import jax.numpy as jnp

from .kep4_operators import (
    poisson_axis,
    poisson_diagonal_axis,
    validate_kep4_pressure_grid,
)
from .matrix_free_gmg import (
    MatrixFreePoissonOperator,
    PoissonBoundaryConditions,
    RectilinearGrid,
)


class KEP4PoissonOperator(MatrixFreePoissonOperator):
    """Uniform-grid positive ``-D4 G4`` operator for compatible projection."""

    def __init__(
        self,
        grid: RectilinearGrid,
        boundaries: PoissonBoundaryConditions,
        *,
        dtype: jnp.dtype = jnp.float64,
    ) -> None:
        super().__init__(grid, boundaries, dtype=dtype)
        self.spacings = validate_kep4_pressure_grid(grid, boundaries)
        dx = poisson_diagonal_axis(
            grid.shape[2],
            self.spacings[0],
            periodic=boundaries.x_lower.kind == "periodic",
            dtype=self.dtype,
        )
        dy = poisson_diagonal_axis(
            grid.shape[1],
            self.spacings[1],
            periodic=boundaries.y_lower.kind == "periodic",
            dtype=self.dtype,
        )
        dz = poisson_diagonal_axis(
            grid.shape[0],
            self.spacings[2],
            periodic=boundaries.z_lower.kind == "periodic",
            dtype=self.dtype,
        )
        self._kep4_diagonal = dz[:, None, None] + dy[None, :, None] + dx[None, None, :]

    @property
    def diagonal(self):
        return self._kep4_diagonal

    def apply(self, pressure):
        self._check_shape(pressure)
        result = jnp.zeros_like(pressure)
        axis_data = (
            (-1, self.spacings[0], self.boundaries.x_lower, self.boundaries.x_upper),
            (-2, self.spacings[1], self.boundaries.y_lower, self.boundaries.y_upper),
            (-3, self.spacings[2], self.boundaries.z_lower, self.boundaries.z_upper),
        )
        for axis, spacing, lower, upper in axis_data:
            result = result + poisson_axis(
                pressure,
                spacing=spacing,
                axis=axis,
                lower_kind=lower.kind,
                upper_kind=upper.kind,
            )
        return result

    def boundary_rhs(self):
        return jnp.zeros(self.shape, dtype=self.dtype)


__all__ = ["KEP4PoissonOperator"]
