"""Uniform physical grid metadata in canonical SI units."""

from __future__ import annotations

from dataclasses import dataclass

from .locations import Cell, Location, ZFace


@dataclass(frozen=True, slots=True)
class UniformGrid:
    """A uniform Cartesian cell grid with SI lengths."""

    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.nz) <= 0:
            raise ValueError("grid cell counts must be positive")
        if min(self.lx, self.ly, self.lz) <= 0.0:
            raise ValueError("grid lengths must be positive")

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def dz(self) -> float:
        return self.lz / self.nz

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny * self.nz

    def global_z_first_shape(self, location: type[Location]) -> tuple[int, int, int]:
        if location is Cell:
            return (self.nz, self.ny, self.nx)
        if location is ZFace:
            return (self.nz + 1, self.ny, self.nx)
        raise ValueError(f"unsupported location: {location!r}")
