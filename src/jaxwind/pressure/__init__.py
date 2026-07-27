"""Interpreter-only pressure solver adapters."""

from .spectral_fd import SpectralFDPressureAdapter, build_spectral_fd_pressure_adapter

__all__ = [
    "SpectralFDPressureAdapter",
    "build_spectral_fd_pressure_adapter",
]
