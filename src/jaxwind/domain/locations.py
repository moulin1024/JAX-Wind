"""Geometric field locations independent of backend storage."""

from __future__ import annotations


class Location:
    __slots__ = ()


class Cell(Location):
    __slots__ = ()


class ZFace(Location):
    __slots__ = ()


CELL = Cell()
ZFACE = ZFace()

