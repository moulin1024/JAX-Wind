"""Effect-shell adapters around the pure numerical core."""

from .checkpoint import (
    ReferenceCheckpointLayout,
    DistributedCheckpointLayout,
    load_ab2_checkpoint,
    load_boussinesq_checkpoint,
    save_ab2_checkpoint,
    save_boussinesq_checkpoint,
)
from .side_by_side import SideBySideStreamLauncher
from .runtime import JaxRuntime
from .precursor import (
    HDF5PrecursorPlayback,
    HDF5PrecursorRecorder,
    finalize_precursor_recording,
)
from .precursor_config import PrecursorPlaybackConfig, PrecursorRecordingConfig
from .precursor_run import run_main_with_precursor, run_precursor

__all__ = [
    "ReferenceCheckpointLayout",
    "JaxRuntime",
    "HDF5PrecursorPlayback",
    "HDF5PrecursorRecorder",
    "PrecursorPlaybackConfig",
    "PrecursorRecordingConfig",
    "SideBySideStreamLauncher",
    "DistributedCheckpointLayout",
    "load_ab2_checkpoint",
    "load_boussinesq_checkpoint",
    "finalize_precursor_recording",
    "run_main_with_precursor",
    "run_precursor",
    "save_ab2_checkpoint",
    "save_boussinesq_checkpoint",
]
