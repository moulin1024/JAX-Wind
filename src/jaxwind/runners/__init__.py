"""Package-owned case runners selected by declarative case configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class RunnerDefinition:
    """Lazy CLI-facing hooks for one built-in runner kind."""

    load_case: Callable[[str | Path], Any]
    run_case: Callable[..., dict[str, Any]]


def get_runner(name: str) -> RunnerDefinition:
    """Resolve a configured runner name without importing JAX."""

    if name == "concurrent_precursor_adm":
        from .concurrent_precursor_adm import load_case, run_case

        return RunnerDefinition(load_case=load_case, run_case=run_case)
    if name == "pressure_driven_warmup":
        from .pressure_driven_warmup import load_case, run_case

        return RunnerDefinition(load_case=load_case, run_case=run_case)
    if name == "gabls1":
        from .gabls1 import load_case, run_case

        return RunnerDefinition(load_case=load_case, run_case=run_case)
    if name in (
        "direct_rigid_alm",
        "direct_aeroelastic_alm",
    ):
        from .direct_rigid_alm import load_case, run_case

        return RunnerDefinition(load_case=load_case, run_case=run_case)
    available = (
        "concurrent_precursor_adm, direct_aeroelastic_alm, "
        "direct_rigid_alm, gabls1, pressure_driven_warmup"
    )
    raise ValueError(f"unknown runner {name!r}; available runners: {available}")


__all__ = ["RunnerDefinition", "get_runner"]
