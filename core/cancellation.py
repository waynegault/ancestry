#!/usr/bin/env python3
"""
core.cancellation - Cooperative cancellation signaling for long-running actions.

Provides a process-wide threading.Event-based flag that can be set by error
handlers (e.g., timeouts) to request graceful shutdown of in-progress work
running in other threads. Actions should periodically check is_cancel_requested()
inside their main loops and exit cleanly when True.
"""
from __future__ import annotations

import threading


class _CancellationState:
    """Manages cancellation state for cooperative shutdown."""
    event = threading.Event()
    scope: str | None = None


def request_cancel(scope: str | None = None) -> None:
    """Signal that current long-running operation should cancel ASAP."""
    _CancellationState.scope = scope
    _CancellationState.event.set()


def clear_cancel() -> None:
    """Clear any prior cancellation request before starting a new operation."""
    _CancellationState.scope = None
    _CancellationState.event.clear()


def is_cancel_requested() -> bool:
    """Return True if a cancellation has been requested."""
    return _CancellationState.event.is_set()


def cancel_scope() -> str | None:
    """Optional string describing who requested cancel (for diagnostics)."""
    return _CancellationState.scope

