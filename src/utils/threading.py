"""Threading helpers for background execution."""
from __future__ import annotations

import threading
from queue import Queue
from typing import Any, Callable


class WorkerThread(threading.Thread):
    """A daemonised thread that executes a callable and pushes exceptions to a queue."""

    def __init__(self, target: Callable[..., Any], args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None, *, queue: Queue | None = None, name: str | None = None) -> None:
        super().__init__(target=target, args=args, kwargs=kwargs or {}, name=name, daemon=True)
        self._exception_queue = queue

    def run(self) -> None:
        try:
            super().run()
        except Exception as exc:  # noqa: BLE001 - propagate to queue
            if self._exception_queue is not None:
                self._exception_queue.put(exc)
            else:
                raise
