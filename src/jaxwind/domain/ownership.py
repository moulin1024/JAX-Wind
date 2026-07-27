"""Mesh-general distribution metadata and the first equal z-slab realization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Generic, TypeAlias, TypeVar

from .axes import DomainAxis, MeshCoordinate, MeshTopology
from .grid import UniformGrid
from .locations import Cell, Location, ZFace


L = TypeVar("L", bound=Location)


@dataclass(frozen=True, slots=True)
class Replicated:
    """A logical domain axis replicated over the process mesh."""


@dataclass(frozen=True, slots=True)
class Partitioned:
    """A logical domain axis partitioned over one named mesh axis."""

    mesh_axis: str

    def __post_init__(self) -> None:
        if not self.mesh_axis:
            raise ValueError("partitioned mesh-axis name must be non-empty")


AxisPlacement: TypeAlias = Replicated | Partitioned


@dataclass(frozen=True, slots=True)
class DistributionSpec:
    """Total mapping from logical domain axes to mesh placement."""

    x: AxisPlacement
    y: AxisPlacement
    z: AxisPlacement

    @classmethod
    def z_slab(cls, mesh_axis: str = "z") -> "DistributionSpec":
        return cls(Replicated(), Replicated(), Partitioned(mesh_axis))

    def placement(self, axis: DomainAxis) -> AxisPlacement:
        return {
            DomainAxis.X: self.x,
            DomainAxis.Y: self.y,
            DomainAxis.Z: self.z,
        }[axis]

    def validate(self, topology: MeshTopology) -> None:
        used = []
        for axis in DomainAxis:
            placement = self.placement(axis)
            if isinstance(placement, Partitioned):
                topology.axis(placement.mesh_axis)
                used.append(placement.mesh_axis)
        if len(set(used)) != len(used):
            raise ValueError(
                "one process-mesh axis cannot partition multiple domain axes"
            )


@dataclass(frozen=True, slots=True, order=True)
class OwnedInterval:
    """A half-open interval of logical integer coordinates."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("owned interval must be non-empty and nonnegative")

    @property
    def size(self) -> int:
        return self.stop - self.start

    def coordinates(self) -> range:
        return range(self.start, self.stop)


@dataclass(frozen=True, slots=True)
class OwnedRegion(Generic[L]):
    """One realized z-slab region for a semantic location."""

    grid: UniformGrid
    topology: MeshTopology
    distribution: DistributionSpec
    coordinate: MeshCoordinate
    location: type[L]
    cell_z: OwnedInterval
    stored_z: OwnedInterval
    lower_physical: bool
    upper_physical: bool

    @property
    def storage_shape(self) -> tuple[int, int, int]:
        return (self.stored_z.size, self.grid.ny, self.grid.nx)

    def at_location(self, location: type[Location]) -> "OwnedRegion":
        if location is Cell:
            stored = self.cell_z
        elif location is ZFace:
            stored = OwnedInterval(self.cell_z.start + 1, self.cell_z.stop + 1)
        else:
            raise ValueError(f"unsupported location: {location!r}")
        return replace(self, location=location, stored_z=stored)


@dataclass(frozen=True, slots=True)
class GlobalTestRegion(Generic[L]):
    """Global storage permitted only to a bounded reference interpretation."""

    grid: UniformGrid
    location: type[L]

    @property
    def storage_shape(self) -> tuple[int, int, int]:
        return self.grid.global_z_first_shape(self.location)

    def at_location(self, location: type[Location]) -> "GlobalTestRegion":
        return GlobalTestRegion(self.grid, location)


@dataclass(frozen=True, slots=True)
class EqualZSlab:
    """First supported realization of a mesh-general distribution spec."""

    grid: UniformGrid
    topology: MeshTopology
    distribution: DistributionSpec

    def __post_init__(self) -> None:
        self.distribution.validate(self.topology)
        if not isinstance(self.distribution.x, Replicated):
            raise ValueError("the first interpreter does not support x partitioning")
        if not isinstance(self.distribution.y, Replicated):
            raise ValueError("the first interpreter does not support y partitioning")
        if not isinstance(self.distribution.z, Partitioned):
            raise ValueError("the first interpreter requires z partitioning")
        if len(self.topology.axes) != 1:
            raise ValueError("the first interpreter supports exactly one mesh axis")
        if self.topology.axes[0].name != self.distribution.z.mesh_axis:
            raise ValueError("the z partition must use the sole process-mesh axis")
        if self.grid.nz % self.shard_count:
            raise ValueError("nz must be divisible by the equal z-slab count")

    @property
    def mesh_axis(self) -> str:
        assert isinstance(self.distribution.z, Partitioned)
        return self.distribution.z.mesh_axis

    @property
    def shard_count(self) -> int:
        return self.topology.axis(self.mesh_axis).size

    @property
    def cells_per_shard(self) -> int:
        return self.grid.nz // self.shard_count

    def region(
        self,
        location: type[L],
        coordinate: MeshCoordinate,
    ) -> OwnedRegion[L]:
        coordinate.validate(self.topology)
        shard = coordinate.index(self.topology, self.mesh_axis)
        start = shard * self.cells_per_shard
        cell_z = OwnedInterval(start, start + self.cells_per_shard)
        if location is Cell:
            stored_z = cell_z
        elif location is ZFace:
            stored_z = OwnedInterval(cell_z.start + 1, cell_z.stop + 1)
        else:
            raise ValueError(f"unsupported location: {location!r}")
        return OwnedRegion(
            grid=self.grid,
            topology=self.topology,
            distribution=self.distribution,
            coordinate=coordinate,
            location=location,
            cell_z=cell_z,
            stored_z=stored_z,
            lower_physical=shard == 0,
            upper_physical=shard == self.shard_count - 1,
        )

    def regions(self, location: type[L]) -> tuple[OwnedRegion[L], ...]:
        return tuple(
            self.region(location, MeshCoordinate((shard,)))
            for shard in range(self.shard_count)
        )

