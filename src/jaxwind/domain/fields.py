"""Immutable semantic field wrappers with cheap construction-time validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from .locations import Location
from .markers import Phase, Quantity
from .ownership import GlobalTestRegion, OwnedRegion


Q = TypeVar("Q", bound=Quantity)
L = TypeVar("L", bound=Location)
OwnershipT = TypeVar("OwnershipT")
P = TypeVar("P", bound=Phase)
A = TypeVar("A")


class Shaped(Protocol):
    shape: tuple[int, ...]


def _payload_shape(payload: Any) -> tuple[int, ...]:
    try:
        return tuple(int(extent) for extent in payload.shape)
    except AttributeError as exc:
        raise TypeError("field payload must expose a shape") from exc


@dataclass(frozen=True, slots=True)
class Field(Generic[Q, L, OwnershipT, P, A]):
    """One semantic field with one global-test or owned region."""

    quantity: type[Q]
    location: type[L]
    ownership: OwnershipT
    phase: type[P]
    payload: A

    def __post_init__(self) -> None:
        if not issubclass(self.quantity, Quantity):
            raise TypeError("quantity must be a Quantity marker type")
        if not issubclass(self.location, Location):
            raise TypeError("location must be a Location marker type")
        if not issubclass(self.phase, Phase):
            raise TypeError("phase must be a Phase marker type")
        if not isinstance(self.ownership, (OwnedRegion, GlobalTestRegion)):
            raise TypeError("unsupported field ownership value")
        if self.ownership.location is not self.location:
            raise ValueError("field location does not match ownership location")
        if _payload_shape(self.payload) != self.ownership.storage_shape:
            raise ValueError(
                "payload shape does not match ownership storage shape: "
                f"{_payload_shape(self.payload)} != {self.ownership.storage_shape}"
            )


@dataclass(frozen=True, slots=True)
class AddressableField(Generic[Q, L, P, A]):
    """One process's local-device batch of uniquely owned regions."""

    quantity: type[Q]
    location: type[L]
    regions: tuple[OwnedRegion[L], ...]
    phase: type[P]
    payload: A

    def __post_init__(self) -> None:
        if not self.regions:
            raise ValueError("an addressable field must contain at least one region")
        if not issubclass(self.quantity, Quantity):
            raise TypeError("quantity must be a Quantity marker type")
        if not issubclass(self.location, Location):
            raise TypeError("location must be a Location marker type")
        if not issubclass(self.phase, Phase):
            raise TypeError("phase must be a Phase marker type")
        first_shape = self.regions[0].storage_shape
        coordinates = set()
        for region in self.regions:
            if region.location is not self.location:
                raise ValueError("region location does not match field location")
            if region.storage_shape != first_shape:
                raise ValueError("addressable equal slabs must have identical shapes")
            if region.coordinate.indices in coordinates:
                raise ValueError("addressable regions must have unique coordinates")
            coordinates.add(region.coordinate.indices)
        expected = (len(self.regions),) + first_shape
        if _payload_shape(self.payload) != expected:
            raise ValueError(
                "addressable payload shape does not match owned regions: "
                f"{_payload_shape(self.payload)} != {expected}"
            )


@dataclass(frozen=True, slots=True)
class VerticalFaceField:
    """Owned upper faces paired with the physical lower-boundary face.

    This is a semantic staggered-field product.  How the owned values are
    partitioned and how neighboring faces are reconstructed are backend
    concerns, so the type deliberately contains no topology-specific name.
    """

    owned: AddressableField
    lower_boundary: Any

    def extract_owned(self) -> AddressableField:
        return self.owned
