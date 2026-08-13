"""JAX process discovery and host-side distributed effects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class JaxRuntime:
    """Description of one initialized JAX job, including the one-process case.

    Numerical code receives this value instead of inspecting global JAX state.
    Host-only distribution effects such as diagnostic gathering, barriers, and
    rank-aware checkpoint paths remain here rather than entering the physics
    model or compiled solver step.
    """

    jax: Any
    jnp: Any
    lax: Any
    global_devices: int
    local_devices: int
    process_count: int
    process_index: int
    backend: str

    def __post_init__(self) -> None:
        counts = (self.global_devices, self.local_devices, self.process_count)
        if any(isinstance(value, bool) or value <= 0 for value in counts):
            raise ValueError("JAX device and process counts must be positive")
        if not 0 <= self.process_index < self.process_count:
            raise ValueError("JAX process index is outside the process topology")
        if self.global_devices != self.local_devices * self.process_count:
            raise ValueError(
                "the unified solver requires the same number of JAX devices "
                "on every process"
            )

    @classmethod
    def from_initialized_jax(cls, jax_module: Any) -> "JaxRuntime":
        """Capture an application-owned JAX runtime without initializing it."""

        import jax.numpy as jnp
        from jax import lax

        return cls(
            jax=jax_module,
            jnp=jnp,
            lax=lax,
            global_devices=jax_module.device_count(),
            local_devices=jax_module.local_device_count(),
            process_count=jax_module.process_count(),
            process_index=jax_module.process_index(),
            backend=jax_module.default_backend(),
        )

    @property
    def is_primary(self) -> bool:
        return self.process_index == 0

    @property
    def addressable_partitions(self) -> tuple[int, ...]:
        """Global partition ids owned by this process in JAX device order."""

        first = self.process_index * self.local_devices
        return tuple(range(first, first + self.local_devices))

    def global_array(self, local_values: Any) -> Any:
        """Gather equal process-local leading-axis batches for diagnostics.

        This is deliberately a host/effect operation and must not be called
        from a compiled physical transition.  Every process in the job must
        call it in the same order.
        """

        if self.process_count == 1:
            return local_values
        from jax.experimental import multihost_utils

        return multihost_utils.process_allgather(local_values, tiled=True)

    def checkpoint_path(self, path: str | Path) -> Path:
        """Return this process's owned checkpoint path without a shared-file race."""

        target = Path(path)
        if self.process_count == 1:
            return target
        return target.with_name(
            f"{target.stem}.process-{self.process_index:05d}{target.suffix}"
        )

    def synchronize(self, name: str) -> None:
        """Barrier all processes at an application effect boundary."""

        if self.process_count == 1:
            return
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(name)


__all__ = ["JaxRuntime"]
