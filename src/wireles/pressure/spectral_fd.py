"""Semantic adapter around the external ``spectral-fd`` solver facade."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from wireles.domain import (
    AddressableField,
    Cell,
    EqualZSlab,
    Evaluated,
    PressureCorrection,
    PressureRhs,
)


@dataclass(frozen=True, slots=True)
class SpectralFDPressureAdapter:
    """Keep solver workspaces behind an owned-cell semantic boundary."""

    decomposition: EqualZSlab
    addressable_shards: tuple[int, ...]
    solver: Any

    def __post_init__(self) -> None:
        config = self.solver.config
        grid = self.decomposition.grid
        if config.discretization != "cell-centered-compatible":
            raise ValueError("pressure facade must use cell-centered-compatible")
        if config.data_layout != "z-first":
            raise ValueError("pressure facade must use z-first local arrays")
        if (config.nx, config.ny, config.nz) != (grid.nx, grid.ny, grid.nz):
            raise ValueError("pressure facade grid shape does not match ownership")
        for actual, expected in (
            (config.lx, grid.lx),
            (config.ly, grid.ly),
            (config.lz, grid.lz),
        ):
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("pressure facade lengths do not match ownership")
        if self.solver.global_devices != self.decomposition.shard_count:
            raise ValueError("pressure facade device mesh does not match z ownership")
        if self.solver.local_devices != len(self.addressable_shards):
            raise ValueError("pressure facade local devices do not match addressable slabs")
        first = self.solver.process_index * self.solver.local_devices
        expected_shards = tuple(range(first, first + self.solver.local_devices))
        if self.addressable_shards != expected_shards:
            raise ValueError(
                "addressable slabs must follow the JAX process-local device order"
            )
        expected_shape = (
            len(self.addressable_shards),
            self.decomposition.cells_per_shard,
            grid.ny,
            grid.nx,
        )
        if self.solver.local_input_shape != expected_shape:
            raise ValueError("pressure facade local shape does not match owned cells")

    def _expected_regions(self) -> tuple:
        regions = self.decomposition.regions(Cell)
        return tuple(regions[index] for index in self.addressable_shards)

    def solve(self, rhs: AddressableField) -> AddressableField:
        """Solve ``D G phi = rhs`` without gathering or changing ownership."""
        if rhs.quantity is not PressureRhs or rhs.location is not Cell:
            raise TypeError("pressure facade requires a cell-centred PressureRhs")
        if rhs.regions != self._expected_regions():
            raise ValueError("pressure RHS ownership does not match the facade")
        pressure = self.solver.solve(rhs.payload)
        return AddressableField(
            PressureCorrection,
            Cell,
            rhs.regions,
            Evaluated,
            pressure,
        )


def build_spectral_fd_pressure_adapter(
    decomposition: EqualZSlab,
    *,
    addressable_shards: tuple[int, ...],
    runtime: Any,
    dtype: str,
    method: str = "transpose",
    tridiag: str = "thomas",
    spike_interface_collective: str = "allgather",
    spike_interface_solver: str = "selected-rows",
) -> SpectralFDPressureAdapter:
    """Construct the external facade from an application-owned JAX runtime."""
    try:
        from spectral_fd import Poisson3DConfig, Poisson3DSolver
    except ImportError as exc:
        raise ImportError(
            "spectral-fd is required for the production pressure adapter; "
            "install the pressure extra or an editable spectral-fd checkout"
        ) from exc
    grid = decomposition.grid
    config = Poisson3DConfig(
        nx=grid.nx,
        ny=grid.ny,
        nz=grid.nz,
        lx=grid.lx,
        ly=grid.ly,
        lz=grid.lz,
        dtype=dtype,
        method=method,
        tridiag=tridiag,
        data_layout="z-first",
        discretization="cell-centered-compatible",
        nyquist_filter=True,
        spike_interface_collective=spike_interface_collective,
        spike_interface_solver=spike_interface_solver,
    )
    solver = Poisson3DSolver(config, runtime=runtime)
    return SpectralFDPressureAdapter(decomposition, addressable_shards, solver)
