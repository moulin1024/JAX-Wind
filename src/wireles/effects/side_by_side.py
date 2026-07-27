"""Runtime launch policies for independent side-by-side domain advances."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class SideBySideStreamLauncher:
    """Dispatch two independent JAX programs from dedicated host threads.

    With ``execution_streams=True`` each thread receives a distinct PJRT
    execution-stream id.  This is deliberately isolated as a runtime effect:
    numerical coupling and restart state do not depend on stream scheduling.
    The caller must still benchmark overlap on its installed JAX/XLA build.
    """

    execution_streams: bool = True
    _pool: ThreadPoolExecutor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="wireles-side-by-side",
        )

    def _submit(self, stream_id: int, operation: Callable[[], Any]) -> Any:
        context = nullcontext()
        if self.execution_streams:
            try:
                from jaxlib import xla_client
            except ImportError as exc:
                raise RuntimeError(
                    "this jaxlib does not expose execution stream selection"
                ) from exc
            context = xla_client.execution_stream_id(stream_id)
        with context:
            return operation()

    def __call__(
        self,
        left: Callable[[], Any],
        right: Callable[[], Any],
    ) -> tuple[Any, Any]:
        left_future = self._pool.submit(self._submit, 1, left)
        right_future = self._pool.submit(self._submit, 2, right)
        return left_future.result(), right_future.result()

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self) -> SideBySideStreamLauncher:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()
