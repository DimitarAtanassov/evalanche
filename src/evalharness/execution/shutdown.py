"""Graceful shutdown signal handling for in-flight runs."""

from __future__ import annotations

import asyncio
import signal

from evalharness.observability import get_logger

logger = get_logger(__name__)


class GracefulShutdown:
    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.reason: str | None = None
        self._installed = False

    @property
    def requested(self) -> bool:
        return self.event.is_set()

    def install(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)
        self._installed = False

    def _handle(self) -> None:
        self.request("signal")

    def request(self, reason: str) -> None:
        if self.event.is_set():
            return
        self.reason = reason
        self.event.set()
        logger.info("shutdown_requested")
