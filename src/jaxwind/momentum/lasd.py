"""Multilevel Lagrangian scale-dependent dynamic SGS closure."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.momentum.metrics import AxisMetric
from jaxwind.pressure.matrix_free_gmg import MatrixFreeGMG


Array = jax.Array
FilterBoundary = str


@dataclass(frozen=True, slots=True)
class LASDModel:
    """Momentum LASD controls for the MAC finite-volume solver.

    The stored coefficient is :math:`C_s^2`.  The two Germano test scales are
    the first two grids of the pressure multigrid hierarchy.
    """

    filter_grid_ratio: float = 1.0
    update_interval: int = 1
    timescale_coefficient: float = 1.5
    initial_coefficient: float = 0.03
    minimum_coefficient: float = 1.0e-6
    maximum_coefficient: float = 0.81
    scale_dependent: bool = True
    x_boundary: FilterBoundary = "periodic"
    y_boundary: FilterBoundary = "periodic"
    z_boundary: FilterBoundary = "reflect"
    molecular_viscosity: float = 0.0
    sgs_delta_scale: float | None = None

    def __post_init__(self) -> None:
        if self.filter_grid_ratio <= 0.0:
            raise ValueError("LASD grid-filter ratio must be positive")
        if self.update_interval <= 0:
            raise ValueError("LASD update interval must be positive")
        if self.timescale_coefficient <= 0.0:
            raise ValueError("LASD timescale coefficient must be positive")
        if not (
            0.0
            <= self.minimum_coefficient
            <= self.initial_coefficient
            <= self.maximum_coefficient
        ):
            raise ValueError("LASD coefficient bounds are invalid")
        if self.x_boundary not in {"periodic", "reflect"}:
            raise ValueError("unsupported x filter boundary")
        if self.y_boundary not in {"periodic", "reflect"}:
            raise ValueError("unsupported y filter boundary")
        if self.z_boundary not in {"periodic", "reflect"}:
            raise ValueError("unsupported z filter boundary")
        if (
            not math.isfinite(self.molecular_viscosity)
            or self.molecular_viscosity < 0.0
        ):
            raise ValueError("molecular viscosity must be finite and nonnegative")
        if self.sgs_delta_scale is not None and (
            not math.isfinite(self.sgs_delta_scale)
            or self.sgs_delta_scale <= 0.0
        ):
            raise ValueError("SGS delta scale must be positive and finite")

    @property
    def effective_delta_scale(self) -> float:
        """Scale the geometric cell width for a three-dimensional grid filter."""
        if self.sgs_delta_scale is not None:
            return self.sgs_delta_scale
        return self.filter_grid_ratio


class LASDState(NamedTuple):
    """Accepted-step Lagrangian memory on cell centres."""

    coefficient: Array
    lm: Array
    mm: Array
    qn: Array
    nn: Array
    trajectory_x: Array
    trajectory_y: Array
    trajectory_z: Array


def _symmetric_dot(left: Array, right: Array) -> Array:
    return (
        left[..., 0] * right[..., 0]
        + 2.0 * left[..., 1] * right[..., 1]
        + 2.0 * left[..., 2] * right[..., 2]
        + left[..., 3] * right[..., 3]
        + 2.0 * left[..., 4] * right[..., 4]
        + left[..., 5] * right[..., 5]
    )


def _strain_tensor(gradient: Array) -> Array:
    strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
    return jnp.stack(
        (
            strain[..., 0, 0],
            strain[..., 0, 1],
            strain[..., 0, 2],
            strain[..., 1, 1],
            strain[..., 1, 2],
            strain[..., 2, 2],
        ),
        axis=-1,
    )


def _tensor_magnitude(tensor: Array) -> Array:
    return jnp.sqrt(jnp.maximum(2.0 * _symmetric_dot(tensor, tensor), 0.0))


def _safe_divide(numerator: Array, denominator: Array) -> Array:
    valid = jnp.abs(denominator) > 1.0e-30
    return jnp.where(
        valid,
        numerator / jnp.where(valid, denominator, 1.0),
        0.0,
    )


class MultilevelLASD:
    """LASD evaluated on the pressure multigrid hierarchy.

    Germano statistics and Lagrangian memory live on level one.  Level-two
    statistics are interpolated to level one for the scale-dependent solve,
    and only the resulting coefficient is interpolated back to the LES grid.
    """

    def __init__(
        self,
        *,
        multigrid: MatrixFreeGMG,
        model: LASDModel,
    ) -> None:
        if len(multigrid.hierarchy.grids) < 3:
            raise ValueError("multilevel LASD requires two multigrid transfers")
        if not math.isclose(model.filter_grid_ratio, 1.0):
            raise ValueError(
                "multilevel LASD fixes the grid-filter ratio at one; "
                "use sgs_delta_scale only for an explicit model calibration"
            )
        first_factors = multigrid.hierarchy.coarsening_factors[0]
        second_factors = multigrid.hierarchy.coarsening_factors[1]
        if first_factors != second_factors:
            raise ValueError(
                "the first two LASD multigrid levels must use equal coarsening"
            )
        fine_grid = multigrid.hierarchy.grids[0]
        test_grid = multigrid.hierarchy.grids[1]
        fine_axis_widths = tuple(
            np.diff(np.asarray(faces, dtype=float))
            for faces in (fine_grid.x_faces, fine_grid.y_faces, fine_grid.z_faces)
        )
        self.metric_aware = not all(
            np.allclose(widths, widths[0], rtol=1.0e-12, atol=1.0e-12)
            for widths in fine_axis_widths
        )
        self.multigrid = multigrid
        self.hierarchy = multigrid.hierarchy
        self.model = model
        self.dx = test_grid.x_faces[1] - test_grid.x_faces[0]
        self.dy = test_grid.y_faces[1] - test_grid.y_faces[0]
        self.dz = test_grid.z_faces[1] - test_grid.z_faces[0]
        self.test_z_centers = jnp.asarray(
            0.5
            * (
                np.asarray(test_grid.z_faces[1:], dtype=float)
                + np.asarray(test_grid.z_faces[:-1], dtype=float)
            )
        )
        if self.metric_aware:
            dx, dy, dz = (
                jnp.asarray(widths) for widths in fine_axis_widths
            )
            self.delta = model.effective_delta_scale * (
                dz[:, None, None] * dy[None, :, None] * dx[None, None, :]
            ) ** (1.0 / 3.0)
            self.memory_delta = self.hierarchy.restrict(self.delta, fine_level=0)
            second_memory_delta = self.hierarchy.restrict(
                self.memory_delta,
                fine_level=1,
            )
            self.test_delta = self._grid_delta(test_grid)
            self.second_test_delta = self._grid_delta(
                multigrid.hierarchy.grids[2]
            )
            self.beta_test_ratio = self.test_delta / self.memory_delta
            self.beta_second_ratio = self.multigrid.prolong(
                self.second_test_delta / second_memory_delta,
                fine_level=1,
            )
        else:
            fine_widths = tuple(widths[0] for widths in fine_axis_widths)
            self.delta = (
                model.effective_delta_scale
                * math.prod(fine_widths) ** (1.0 / 3.0)
            )
            self.memory_delta = self.delta
            self.test_delta = None
            self.second_test_delta = None
            self.beta_test_ratio = math.prod(first_factors) ** (1.0 / 3.0)
            self.beta_second_ratio = self.beta_test_ratio**2
        # Restriction is the test filter, so its volume-equivalent width is
        # determined by the actual GMG coarsening rather than by a separately
        # imposed nominal filter ratio.  On metric-aware grids the beta solve
        # above retains the corresponding local physical ratios.
        self.test_ratio = math.prod(first_factors) ** (1.0 / 3.0)
        self.second_test_ratio = self.test_ratio**2
        if self.metric_aware:
            self.level_deltas = (
                self.delta,
                self.test_delta,
                self.second_test_delta,
            )
        else:
            self.level_deltas = (
                self.delta,
                self.delta * self.test_ratio,
                self.delta * self.second_test_ratio,
            )
        self.level_metrics = tuple(
            self._grid_metrics(grid, multigrid.dtype)
            for grid in self.hierarchy.grids[:3]
        )

    @staticmethod
    def _grid_metrics(
        grid: object,
        dtype: object,
    ) -> tuple[AxisMetric, AxisMetric, AxisMetric]:
        """Build the same finite-volume derivative metrics on one GMG grid."""
        return (
            AxisMetric(grid.x_faces, axis=2, periodic=True, dtype=dtype),
            AxisMetric(grid.y_faces, axis=1, periodic=True, dtype=dtype),
            AxisMetric(
                grid.z_faces,
                axis=0,
                periodic=False,
                dtype=dtype,
                derivative_width=3,
            ),
        )

    def _grid_delta(self, grid: object) -> Array:
        dx, dy, dz = (
            jnp.diff(jnp.asarray(faces))
            for faces in (grid.x_faces, grid.y_faces, grid.z_faces)
        )
        return self.model.effective_delta_scale * (
            dz[:, None, None] * dy[None, :, None] * dx[None, None, :]
        ) ** (1.0 / 3.0)

    def initialize(self, cell_velocity: Array) -> LASDState:
        fine_shape = cell_velocity.shape[:-1]
        if fine_shape != self.hierarchy.grids[0].shape:
            raise ValueError("LASD velocity shape does not match the hierarchy")
        shape = self.hierarchy.grids[1].shape
        coefficient = jnp.full(
            fine_shape,
            self.model.initial_coefficient,
            dtype=cell_velocity.dtype,
        )
        zero = jnp.zeros(shape, dtype=cell_velocity.dtype)
        return LASDState(coefficient, zero, zero, zero, zero, zero, zero, zero)

    @staticmethod
    def _velocity_products(cell_velocity: Array) -> Array:
        """Return the six independent components of ``u_i u_j``."""
        return jnp.stack(
            (
                cell_velocity[..., 0] ** 2,
                cell_velocity[..., 0] * cell_velocity[..., 1],
                cell_velocity[..., 0] * cell_velocity[..., 2],
                cell_velocity[..., 1] ** 2,
                cell_velocity[..., 1] * cell_velocity[..., 2],
                cell_velocity[..., 2] ** 2,
            ),
            axis=-1,
        )

    def _velocity_gradient(self, cell_velocity: Array, *, level: int) -> Array:
        """Differentiate a cell-average velocity on its own FV grid level."""
        metrics = self.level_metrics[level]
        derivatives = []
        for component in range(3):
            value = cell_velocity[..., component]
            derivatives.append(
                jnp.stack(
                    tuple(metric.derivative(value) for metric in metrics),
                    axis=-1,
                )
            )
        return jnp.stack(derivatives, axis=-2)

    def _model_tensor(self, cell_velocity: Array, *, level: int) -> Array:
        """Return ``Delta^2 |S| S`` from the level-native FV gradient."""
        tensor = _strain_tensor(
            self._velocity_gradient(cell_velocity, level=level)
        )
        delta = jnp.asarray(self.level_deltas[level], dtype=cell_velocity.dtype)
        return delta[..., None] ** 2 * _tensor_magnitude(tensor)[..., None] * tensor

    def _fine_filter_inputs(self, cell_velocity: Array) -> Array:
        """Pack conservative quantities filtered from the LES grid."""
        return jnp.concatenate(
            (
                cell_velocity,
                self._velocity_products(cell_velocity),
                self._model_tensor(cell_velocity, level=0),
            ),
            axis=-1,
        )

    def _germano_tensors_from_filtered(
        self,
        filtered: Array,
        *,
        level: int,
    ) -> tuple[Array, Array]:
        """Build one discrete FV Germano pair on a test grid.

        Restriction supplies conservative test-filtered cell averages.  The
        test-grid strain is deliberately recomputed as ``D_H(R u)``; filtering
        the fine-grid strain as ``R(D_h u)`` is not a valid identity for the
        finite-volume operators, especially on a stretched grid.
        """
        velocity_hat = filtered[..., :3]
        products_hat = filtered[..., 3:9]
        model_tensor_hat = filtered[..., 9:15]
        resolved = jnp.stack(
            (
                products_hat[..., 0] - velocity_hat[..., 0] ** 2,
                products_hat[..., 1]
                - velocity_hat[..., 0] * velocity_hat[..., 1],
                products_hat[..., 2]
                - velocity_hat[..., 0] * velocity_hat[..., 2],
                products_hat[..., 3] - velocity_hat[..., 1] ** 2,
                products_hat[..., 4]
                - velocity_hat[..., 1] * velocity_hat[..., 2],
                products_hat[..., 5] - velocity_hat[..., 2] ** 2,
            ),
            axis=-1,
        )
        model = 2.0 * (
            model_tensor_hat
            - self._model_tensor(velocity_hat, level=level)
        )
        return resolved, model

    def germano_tensors(
        self,
        cell_velocity: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Return the two nested, level-native finite-volume Germano pairs."""
        if cell_velocity.shape[:-1] != self.hierarchy.grids[0].shape:
            raise ValueError("LASD velocity shape does not match the hierarchy")
        filtered_one = self.hierarchy.restrict(
            self._fine_filter_inputs(cell_velocity),
            fine_level=0,
        )
        filtered_two = self.hierarchy.restrict(filtered_one, fine_level=1)
        resolved_one, model_one = self._germano_tensors_from_filtered(
            filtered_one,
            level=1,
        )
        resolved_two, model_two = self._germano_tensors_from_filtered(
            filtered_two,
            level=2,
        )
        return resolved_one, model_one, resolved_two, model_two

    @staticmethod
    def _contractions_from_tensors(
        resolved: Array,
        model: Array,
    ) -> tuple[Array, Array]:
        return _symmetric_dot(resolved, model), _symmetric_dot(model, model)

    def contractions(
        self,
        cell_velocity: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Return contractions from the nested FV Germano identities."""
        resolved_one, model_one, resolved_two, model_two = self.germano_tensors(
            cell_velocity
        )
        lm, mm = self._contractions_from_tensors(resolved_one, model_one)
        qn, nn = self._contractions_from_tensors(resolved_two, model_two)
        return (
            lm,
            mm,
            self.multigrid.prolong(qn, fine_level=1),
            self.multigrid.prolong(nn, fine_level=1),
        )

    @staticmethod
    def _fold_indices(
        indices: Array,
        size: int,
        boundary: FilterBoundary,
    ) -> Array:
        if boundary == "periodic":
            return jnp.mod(indices, size)
        return jnp.clip(indices, 0, size - 1)

    def _departure(
        self,
        values: Array,
        state: LASDState,
        interval_dt: float,
    ) -> Array:
        if values.ndim not in {3, 4}:
            raise ValueError("departure fields must be scalar or field-batched")
        nz, ny, nx = values.shape[:3]
        x_index = jnp.arange(nx, dtype=values.dtype)[None, None, :]
        y_index = jnp.arange(ny, dtype=values.dtype)[None, :, None]
        z_index = jnp.arange(nz, dtype=values.dtype)[:, None, None]
        xi = x_index - state.trajectory_x * interval_dt / self.dx
        eta = y_index - state.trajectory_y * interval_dt / self.dy
        if self.metric_aware:
            z_centers = self.test_z_centers.astype(values.dtype)
            departure_z = jnp.clip(
                z_centers[:, None, None] - state.trajectory_z * interval_dt,
                z_centers[0],
                z_centers[-1],
            )
            k1 = jnp.clip(
                jnp.searchsorted(z_centers, departure_z, side="right"),
                1,
                nz - 1,
            )
            k0 = k1 - 1
            lower_z = z_centers[k0]
            upper_z = z_centers[k1]
            fz = _safe_divide(departure_z - lower_z, upper_z - lower_z)
        else:
            zeta = jnp.clip(
                z_index - state.trajectory_z * interval_dt / self.dz,
                0.0,
                float(nz - 1),
            )
            k0 = jnp.floor(zeta).astype(jnp.int32)
            k1 = jnp.minimum(k0 + 1, nz - 1)
            fz = zeta - jnp.floor(zeta)
        i0_raw = jnp.floor(xi).astype(jnp.int32)
        j0_raw = jnp.floor(eta).astype(jnp.int32)
        i0 = self._fold_indices(i0_raw, nx, self.model.x_boundary)
        i1 = self._fold_indices(i0_raw + 1, nx, self.model.x_boundary)
        j0 = self._fold_indices(j0_raw, ny, self.model.y_boundary)
        j1 = self._fold_indices(j0_raw + 1, ny, self.model.y_boundary)
        fx = xi - jnp.floor(xi)
        fy = eta - jnp.floor(eta)
        if values.ndim == 4:
            fx = fx[..., None]
            fy = fy[..., None]
            fz = fz[..., None]
        q000 = values[k0, j0, i0]
        q100 = values[k0, j0, i1]
        q010 = values[k0, j1, i0]
        q110 = values[k0, j1, i1]
        q001 = values[k1, j0, i0]
        q101 = values[k1, j0, i1]
        q011 = values[k1, j1, i0]
        q111 = values[k1, j1, i1]
        q00 = (1.0 - fx) * q000 + fx * q100
        q10 = (1.0 - fx) * q010 + fx * q110
        q01 = (1.0 - fx) * q001 + fx * q101
        q11 = (1.0 - fx) * q011 + fx * q111
        return (1.0 - fz) * (
            (1.0 - fy) * q00 + fy * q10
        ) + fz * ((1.0 - fy) * q01 + fy * q11)

    def _lagrangian_average(
        self,
        current_a: Array,
        current_b: Array,
        old_a: Array,
        old_b: Array,
        state: LASDState,
        interval_dt: float,
        departures: tuple[Array, Array] | None = None,
    ) -> tuple[Array, Array]:
        product = old_a * old_b
        valid = (old_a > 0.0) & (old_b >= 0.0) & (product > 0.0)
        timescale = (
            self.model.timescale_coefficient
            * self.memory_delta
            * jnp.where(valid, product ** (-0.125), 1.0)
        )
        weight = jnp.where(
            valid,
            (interval_dt / timescale) / (1.0 + interval_dt / timescale),
            0.0,
        )
        if departures is None:
            departure_a = self._departure(old_a, state, interval_dt)
            departure_b = self._departure(old_b, state, interval_dt)
        else:
            departure_a, departure_b = departures
        return (
            weight * current_a + (1.0 - weight) * departure_a,
            jnp.maximum(
                weight * current_b + (1.0 - weight) * departure_b,
                0.0,
            ),
        )

    @staticmethod
    def _physical_z_boundary(values: Array) -> Array:
        if values.shape[0] < 2:
            return values
        return values.at[0].set(values[1]).at[-1].set(values[-2])

    def accumulate(
        self,
        state: LASDState,
        cell_velocity: Array,
    ) -> LASDState:
        interval = self.model.update_interval
        coarse_velocity = self.hierarchy.restrict(cell_velocity, fine_level=0)
        return state._replace(
            trajectory_x=state.trajectory_x
            + coarse_velocity[..., 0] / interval,
            trajectory_y=state.trajectory_y
            + coarse_velocity[..., 1] / interval,
            trajectory_z=state.trajectory_z
            + coarse_velocity[..., 2] / interval,
        )

    def update(
        self,
        state: LASDState,
        cell_velocity: Array,
        *,
        interval_dt: float,
        first_update: bool,
    ) -> LASDState:
        return self.update_from_contractions(
            state,
            *self.contractions(cell_velocity),
            interval_dt=interval_dt,
            first_update=first_update,
        )

    def update_from_contractions(
        self,
        state: LASDState,
        lm: Array,
        mm: Array,
        qn: Array,
        nn: Array,
        *,
        interval_dt: float,
        first_update: bool,
    ) -> LASDState:
        """Advance Lagrangian memory from precomputed dual-scale statistics."""
        histories = (
            jnp.where(first_update, self.model.initial_coefficient * mm, state.lm),
            jnp.where(first_update, mm, state.mm),
            jnp.where(first_update, self.model.initial_coefficient * nn, state.qn),
            jnp.where(first_update, nn, state.nn),
        )
        histories = tuple(self._physical_z_boundary(value) for value in histories)
        seeded = state._replace(
            lm=histories[0],
            mm=histories[1],
            qn=histories[2],
            nn=histories[3],
        )
        departures = self._departure(
            jnp.stack(histories, axis=-1),
            seeded,
            interval_dt,
        )
        lm_avg, mm_avg = self._lagrangian_average(
            lm,
            mm,
            histories[0],
            histories[1],
            seeded,
            interval_dt,
            (departures[..., 0], departures[..., 1]),
        )
        qn_avg, nn_avg = self._lagrangian_average(
            qn,
            nn,
            histories[2],
            histories[3],
            seeded,
            interval_dt,
            (departures[..., 2], departures[..., 3]),
        )
        coefficient_2d = jnp.maximum(_safe_divide(lm_avg, mm_avg), 0.0)
        coefficient_4d = jnp.maximum(_safe_divide(qn_avg, nn_avg), 0.0)
        first_ratio = jnp.asarray(
            self.beta_test_ratio,
            dtype=coefficient_2d.dtype,
        )
        second_ratio = jnp.asarray(
            self.beta_second_ratio,
            dtype=coefficient_2d.dtype,
        )
        exponent = jnp.log(first_ratio) / (
            jnp.log(second_ratio) - jnp.log(first_ratio)
        )
        raw_beta = jnp.maximum(
            _safe_divide(coefficient_4d, coefficient_2d),
            0.0,
        ) ** exponent
        beta_floor = 1.0 / first_ratio**3
        beta = jnp.maximum(raw_beta, beta_floor)
        if not self.model.scale_dependent:
            beta = jnp.ones_like(beta)
        coarse_coefficient = jnp.clip(
            _safe_divide(coefficient_2d, beta),
            self.model.minimum_coefficient,
            self.model.maximum_coefficient,
        )
        coefficient = jnp.clip(
            self.multigrid.prolong(coarse_coefficient, fine_level=0),
            self.model.minimum_coefficient,
            self.model.maximum_coefficient,
        )
        zero = jnp.zeros_like(state.trajectory_x)
        return LASDState(
            coefficient,
            lm_avg,
            mm_avg,
            qn_avg,
            nn_avg,
            zero,
            zero,
            zero,
        )

    def viscosity(self, coefficient: Array, gradient: Array) -> Array:
        tensor = _strain_tensor(gradient)
        return (
            coefficient * self.delta**2 * _tensor_magnitude(tensor)
            + self.model.molecular_viscosity
        )

    def stress(self, coefficient: Array, gradient: Array) -> Array:
        strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
        return 2.0 * self.viscosity(
            coefficient,
            gradient,
        )[..., None, None] * strain


__all__ = [
    "LASDModel",
    "LASDState",
    "MultilevelLASD",
]
