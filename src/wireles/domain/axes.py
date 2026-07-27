"""Logical-domain and process-mesh axes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod


class DomainAxis(Enum):
    """Named logical axes; these names do not prescribe storage order."""

    X = "x"
    Y = "y"
    Z = "z"


@dataclass(frozen=True, slots=True)
class MeshAxis:
    """One named axis of a logical process mesh."""

    name: str
    size: int

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError("mesh-axis name must be a non-empty Python identifier")
        if self.size <= 0:
            raise ValueError("mesh-axis size must be positive")


@dataclass(frozen=True, slots=True)
class MeshTopology:
    """Array-independent logical process-mesh metadata."""

    axes: tuple[MeshAxis, ...]

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("a process mesh must contain at least one axis")
        names = tuple(axis.name for axis in self.axes)
        if len(set(names)) != len(names):
            raise ValueError("process-mesh axis names must be unique")

    @property
    def size(self) -> int:
        return prod(axis.size for axis in self.axes)

    def axis(self, name: str) -> MeshAxis:
        for axis in self.axes:
            if axis.name == name:
                return axis
        raise ValueError(f"mesh axis {name!r} is not present")

    def axis_position(self, name: str) -> int:
        for position, axis in enumerate(self.axes):
            if axis.name == name:
                return position
        raise ValueError(f"mesh axis {name!r} is not present")


@dataclass(frozen=True, slots=True)
class MeshCoordinate:
    """One coordinate in a :class:`MeshTopology`."""

    indices: tuple[int, ...]

    def validate(self, topology: MeshTopology) -> None:
        if len(self.indices) != len(topology.axes):
            raise ValueError("mesh coordinate rank does not match topology")
        for index, axis in zip(self.indices, topology.axes, strict=True):
            if not 0 <= index < axis.size:
                raise ValueError(
                    f"mesh coordinate {index} is outside axis {axis.name!r}"
                )

    def index(self, topology: MeshTopology, axis_name: str) -> int:
        self.validate(topology)
        return self.indices[topology.axis_position(axis_name)]

