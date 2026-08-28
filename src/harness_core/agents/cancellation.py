"""
Cancellation support for agent execution.

Provides:
  - Clean Ctrl+C handling
  - Subprocess cancellation
  - Session checkpoint on interrupt
  - Resource cleanup
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from typing import Any, Optional

import ctypes


class CancellationHandler:
    """Handle clean cancellation of agent execution.

    When the user presses Ctrl+C:
      1. Stop current operation safely
      2. Cancel model request where possible
      3. Cancel subprocess
      4. Stop agents
      5. Persist session/checkpoint
      6. Release locks
      7. Exit cleanly
    """

    def __init__(self) -> None:
        self._cancelled = False
        self._cancel_reason = ""
        self._callbacks: list = []
        self._original_handler = None

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def cancel_reason(self) -> str:
        return self._cancel_reason

    def cancel(self, reason: str = "User cancelled") -> None:
        """Request cancellation."""
        self._cancelled = True
        self._cancel_reason = reason

        # Run registered callbacks
        for cb in self._callbacks:
            try:
                cb()
            except Exception:
                pass

    def register_cleanup(self, callback) -> None:
        """Register a cleanup callback."""
        self._callbacks.append(callback)

    def check(self) -> bool:
        """Check if cancelled. Returns True if cancelled."""
        return self._cancelled

    def reset(self) -> None:
        """Reset cancellation state."""
        self._cancelled = False
        self._cancel_reason = ""
        self._callbacks.clear()


class GracefulShutdown:
    """Context manager for graceful shutdown on Ctrl+C."""

    def __init__(self, handler: CancellationHandler | None = None):
        self.handler = handler or CancellationHandler()
        self._processes: list[int] = []

    def __enter__(self):
        # Install signal handler
        if os.name != "nt":  # Not Windows
            self._original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._signal_handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original handler
        if os.name != "nt" and hasattr(self, "_original_handler"):
            signal.signal(signal.SIGINT, self._original_handler)

        # Cancel any remaining
        if not self.handler.is_cancelled:
            self.handler.cancel("Shutdown")

        # Kill tracked processes
        self._cleanup_processes()

    def _signal_handler(self, signum, frame):
        """Handle SIGINT."""
        self.handler.cancel("User pressed Ctrl+C")

    def track_process(self, pid: int) -> None:
        """Track a subprocess for cleanup."""
        self._processes.append(pid)

    def _cleanup_processes(self) -> None:
        """Kill tracked processes."""
        for pid in self._processes:
            try:
                if os.name == "nt":
                    os.kill(pid, signal.SIGTERM)
                else:
                    os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
        self._processes.clear()


class OperationTimeout:
    """Timeout wrapper for operations."""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def expired(self) -> bool:
        """Check if timeout has expired."""
        return (time.perf_counter() - self.start_time) >= self.timeout_seconds

    def remaining(self) -> float:
        """Get remaining time in seconds."""
        elapsed = time.perf_counter() - self.start_time
        return max(0.0, self.timeout_seconds - elapsed)


async def with_timeout(
    coro,
    timeout_seconds: float,
    cancel_handler: CancellationHandler | None = None,
    default=None,
):
    """Run a coroutine with timeout and cancellation support.

    Args:
        coro: Coroutine to run
        timeout_seconds: Maximum seconds to wait
        cancel_handler: Optional cancellation handler
        default: Default value on timeout/cancel

    Returns:
        Result of coroutine, or default on timeout/cancel
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        if cancel_handler:
            cancel_handler.cancel("Operation timed out")
        return default
    except asyncio.CancelledError:
        if cancel_handler:
            cancel_handler.cancel("Operation cancelled")
        return default
