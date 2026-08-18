"""I/O policies for offline precursor recording and playback."""

from __future__ import annotations

from dataclasses import dataclass


SECTION_NAMES = ("inflow", "outflow")


@dataclass(frozen=True, slots=True)
class PrecursorRecordingConfig:
    """Host-I/O policy for an offline precursor section recording.

    ``sample_every`` is measured from the first supplied accepted state.  A
    buffer is copied to HDF5 as one time-contiguous hyperslab.  Compression is
    deliberately opt-in because uncompressed writes are usually preferable on
    parallel filesystems; ``lzf`` is the low-overhead compressed choice.
    """

    sample_every: int = 1
    buffer_samples: int = 8
    compression: str | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.sample_every, "sample interval"),
            (self.buffer_samples, "sample buffer"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"precursor {name} must be a positive integer")
        if self.compression not in (None, "lzf", "gzip"):
            raise ValueError("precursor compression must be None, 'lzf', or 'gzip'")
        if not isinstance(self.overwrite, bool):
            raise TypeError("precursor overwrite must be boolean")


@dataclass(frozen=True, slots=True)
class PrecursorPlaybackConfig:
    """Rank-local read policy for replaying a recorded boundary plane."""

    section: str = "inflow"
    buffer_samples: int = 16
    spanwise_shift_cells: int = 0

    def __post_init__(self) -> None:
        if self.section not in SECTION_NAMES:
            raise ValueError("precursor playback section must be inflow or outflow")
        if (
            isinstance(self.buffer_samples, bool)
            or not isinstance(self.buffer_samples, int)
            or self.buffer_samples <= 0
        ):
            raise ValueError("precursor playback buffer must be a positive integer")
        if (
            isinstance(self.spanwise_shift_cells, bool)
            or not isinstance(self.spanwise_shift_cells, int)
        ):
            raise TypeError("precursor spanwise shift must be an integer cell count")


__all__ = ["PrecursorPlaybackConfig", "PrecursorRecordingConfig", "SECTION_NAMES"]
