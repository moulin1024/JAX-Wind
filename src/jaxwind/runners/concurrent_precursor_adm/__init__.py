"""Concurrent-precursor runner with a uniform pure-thrust actuator disk."""

from .config import CaseConfig, ConfigError, load_case
from .runner import run_case

__all__ = ["CaseConfig", "ConfigError", "load_case", "run_case"]
