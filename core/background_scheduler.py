#!/usr/bin/env python3

"""
core/background_scheduler.py - Background Task Scheduler

Phase 10.2 & 10.3: Implements periodic background tasks for hands-off operation.

Features:
- Inbox polling at configurable intervals (15-30 minutes)
- Session keepalive for long-running operations
- Graceful shutdown support
- Configurable task intervals
"""

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# Ensure project root is on sys.path when running as a script
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.registry_utils import auto_register_module
from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)
auto_register_module(globals(), __name__)


@dataclass
class TaskConfig:
    """Configuration for a scheduled task."""

    name: str
    interval_seconds: float
    callback: Callable[[], None]
    enabled: bool = True
    run_on_start: bool = False
    last_run: float = field(default=0.0)
    run_count: int = field(default=0)
    error_count: int = field(default=0)


class BackgroundScheduler:
    """
    Manages periodic background tasks with configurable intervals.

    Tasks run on separate threads and can be stopped gracefully.
    """

    def __init__(self) -> None:
        """Initialize the scheduler."""
        self._tasks: dict[str, TaskConfig] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._running = False
        self._lock = threading.RLock()

    def register_task(
        self,
        name: str,
        interval_seconds: float,
        callback: Callable[[], None],
        enabled: bool = True,
        run_on_start: bool = False,
    ) -> None:
        """
        Register a periodic task.

        Args:
            name: Unique task identifier
            interval_seconds: Time between task runs
            callback: Function to call (must be thread-safe)
            enabled: Whether task is initially enabled
            run_on_start: Whether to run immediately when scheduler starts
        """
        with self._lock:
            self._tasks[name] = TaskConfig(
                name=name,
                interval_seconds=interval_seconds,
                callback=callback,
                enabled=enabled,
                run_on_start=run_on_start,
            )
            logger.debug(f"Registered task: {name} (interval={interval_seconds}s)")

    def start(self) -> None:
        """Start all enabled tasks."""
        with self._lock:
            if self._running:
                logger.warning("Scheduler already running")
                return

            self._running = True
            logger.info("🕐 Starting background scheduler...")

            for name, task in self._tasks.items():
                if task.enabled:
                    self._start_task(name, task)

            active_count = len([t for t in self._tasks.values() if t.enabled])
            logger.info(f"✅ Background scheduler started with {active_count} task(s)")

    def _start_task(self, name: str, task: TaskConfig) -> None:
        """Start a single task thread."""
        stop_event = threading.Event()
        self._stop_events[name] = stop_event

        thread = threading.Thread(
            target=self._run_task_loop,
            args=(task, stop_event),
            name=f"bg-{name}",
            daemon=True,
        )
        self._threads[name] = thread
        thread.start()
        logger.debug(f"Started task thread: {name}")

    def _run_task_loop(self, task: TaskConfig, stop_event: threading.Event) -> None:
        """Main loop for a scheduled task."""
        # Run immediately if configured
        if task.run_on_start:
            self._execute_task(task)

        while not stop_event.is_set():
            # Wait for interval or stop signal
            if stop_event.wait(task.interval_seconds):
                break  # Stop signal received

            if not task.enabled:
                continue  # Task was disabled

            self._execute_task(task)

    @staticmethod
    def _execute_task(task: TaskConfig) -> None:
        """Execute a single task with error handling."""
        try:
            logger.debug(f"Running task: {task.name}")
            start_time = time.time()
            task.callback()
            elapsed = time.time() - start_time
            task.last_run = start_time
            task.run_count += 1
            logger.debug(f"Task {task.name} completed in {elapsed:.2f}s (run #{task.run_count})")
        except Exception as e:
            task.error_count += 1
            logger.warning(f"Task {task.name} failed (error #{task.error_count}): {e}")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop all tasks gracefully."""
        with self._lock:
            if not self._running:
                return

            logger.info("🛑 Stopping background scheduler...")

            # Signal all tasks to stop
            for _name, stop_event in self._stop_events.items():
                stop_event.set()

            # Wait for threads to finish
            for name, thread in self._threads.items():
                if thread.is_alive():
                    thread.join(timeout=timeout)
                    if thread.is_alive():
                        logger.warning(f"Task {name} did not stop within timeout")

            self._threads.clear()
            self._stop_events.clear()
            self._running = False
            logger.info("✅ Background scheduler stopped")

    def enable_task(self, name: str) -> bool:
        """Enable a task. Returns True if successful."""
        with self._lock:
            if name not in self._tasks:
                return False
            self._tasks[name].enabled = True
            logger.debug(f"Enabled task: {name}")
            return True

    def disable_task(self, name: str) -> bool:
        """Disable a task (will complete current run). Returns True if successful."""
        with self._lock:
            if name not in self._tasks:
                return False
            self._tasks[name].enabled = False
            logger.debug(f"Disabled task: {name}")
            return True

    def get_task_status(self, name: str) -> Optional[dict[str, Any]]:
        """Get status information for a task."""
        with self._lock:
            if name not in self._tasks:
                return None

            task = self._tasks[name]
            return {
                "name": task.name,
                "enabled": task.enabled,
                "interval_seconds": task.interval_seconds,
                "last_run": task.last_run,
                "run_count": task.run_count,
                "error_count": task.error_count,
                "running": name in self._threads and self._threads[name].is_alive(),
            }

    def get_all_status(self) -> list[dict[str, Any]]:
        """Get status for all registered tasks."""
        return [status for name in self._tasks if (status := self.get_task_status(name)) is not None]

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


# Module-level scheduler instance
_scheduler: Optional[BackgroundScheduler] = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> BackgroundScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler  # noqa: PLW0603
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = BackgroundScheduler()
        return _scheduler


def setup_inbox_polling(
    session_manager: Any,
    interval_minutes: float = 15.0,
    enabled: bool = True,
) -> None:
    """
    Set up periodic inbox polling task.

    Phase 10.2: Automatically checks inbox for new messages
    and processes them through InboundOrchestrator.

    Args:
        session_manager: Session manager with browser/DB access
        interval_minutes: Time between inbox checks (default 15 minutes)
        enabled: Whether to enable polling immediately
    """

    def _poll_inbox() -> None:
        """Inbox polling callback."""
        try:
            # Check if automation is enabled
            from config import config_schema

            if getattr(config_schema, "emergency_stop_enabled", False):
                logger.debug("Inbox polling skipped: emergency stop enabled")
                return

            # Ensure session is ready
            if not session_manager.ensure_db_ready():
                logger.warning("Inbox polling skipped: DB not ready")
                return

            # Get inbox messages via Action 7 InboxProcessor
            from actions.action7_inbox import InboxProcessor

            # Process with InboxProcessor - search_inbox is the main entry point
            processor = InboxProcessor(session_manager)
            result = processor.search_inbox()

            if result:
                logger.info("📬 Inbox polling completed: new messages processed")
            else:
                logger.debug("📬 Inbox polling: no new messages")

        except ImportError as e:
            logger.debug(f"Inbox polling dependencies not available: {e}")
        except Exception as e:
            logger.warning(f"Inbox polling failed: {e}")

    scheduler = get_scheduler()
    scheduler.register_task(
        name="inbox_polling",
        interval_seconds=interval_minutes * 60,
        callback=_poll_inbox,
        enabled=enabled,
        run_on_start=False,  # Don't poll immediately on start
    )
    logger.info(f"📬 Inbox polling configured (every {interval_minutes} minutes)")


def setup_session_keepalive(
    session_manager: Any,
    interval_minutes: float = 10.0,
    enabled: bool = True,
) -> None:
    """
    Set up periodic session keepalive task.

    Phase 10.3: Prevents session timeout during long-running operations
    by refreshing cookies and checking session validity.

    Args:
        session_manager: Session manager with browser access
        interval_minutes: Time between keepalive checks (default 10 minutes)
        enabled: Whether to enable keepalive immediately
    """

    def _keepalive() -> None:
        """Session keepalive callback."""
        try:
            # Check session age
            age_seconds = session_manager.session_age_seconds()
            if age_seconds < 0:
                logger.debug("Session keepalive skipped: no active session")
                return

            # Session lifetime is typically 40 minutes, refresh at 25 minutes
            refresh_threshold = 25 * 60  # 25 minutes in seconds

            if age_seconds >= refresh_threshold:
                logger.info(f"🔄 Session keepalive: refreshing (age={age_seconds / 60:.1f}min)")

                # Refresh browser cookies
                if hasattr(session_manager, "refresh_browser_cookies"):
                    session_manager.refresh_browser_cookies()

                # Sync cookies to API session
                if hasattr(session_manager, "sync_cookies_from_browser"):
                    session_manager.sync_cookies_from_browser()

                logger.info("✅ Session keepalive: cookies refreshed")
            else:
                logger.debug(f"Session keepalive: session healthy (age={age_seconds / 60:.1f}min)")

        except Exception as e:
            logger.warning(f"Session keepalive failed: {e}")

    scheduler = get_scheduler()
    scheduler.register_task(
        name="session_keepalive",
        interval_seconds=interval_minutes * 60,
        callback=_keepalive,
        enabled=enabled,
        run_on_start=False,
    )
    logger.info(f"🔄 Session keepalive configured (every {interval_minutes} minutes)")


def start_background_tasks() -> None:
    """Start all configured background tasks."""
    get_scheduler().start()


def stop_background_tasks(timeout: float = 5.0) -> None:
    """Stop all background tasks."""
    get_scheduler().stop(timeout=timeout)


# -----------------------------------------------------------------------------
# Module Tests
# -----------------------------------------------------------------------------


def module_tests() -> bool:
    """Run module tests for BackgroundScheduler."""
    suite = TestSuite("Background Scheduler", "core/background_scheduler.py")

    # Test: BackgroundScheduler initialization
    def test_scheduler_init() -> None:
        scheduler = BackgroundScheduler()
        assert not scheduler.is_running
        assert len(scheduler.get_all_status()) == 0

    suite.run_test(
        "BackgroundScheduler initialization",
        test_scheduler_init,
        test_summary="Verify scheduler initializes correctly",
        functions_tested="BackgroundScheduler.__init__",
        method_description="Check initial state",
    )

    # Test: Task registration
    def test_task_registration() -> None:
        scheduler = BackgroundScheduler()
        callback = lambda: None  # noqa: E731
        scheduler.register_task(
            name="test_task",
            interval_seconds=60.0,
            callback=callback,
            enabled=True,
        )
        status = scheduler.get_task_status("test_task")
        assert status is not None
        assert status["name"] == "test_task"
        assert status["interval_seconds"] == 60.0
        assert status["enabled"] is True
        assert status["run_count"] == 0

    suite.run_test(
        "Task registration",
        test_task_registration,
        test_summary="Verify task can be registered",
        functions_tested="BackgroundScheduler.register_task",
        method_description="Register and verify task config",
    )

    # Test: Enable/disable task
    def test_enable_disable_task() -> None:
        scheduler = BackgroundScheduler()
        scheduler.register_task("toggle_task", 60.0, lambda: None, enabled=True)

        assert scheduler.disable_task("toggle_task")
        status = scheduler.get_task_status("toggle_task")
        assert status is not None
        assert status["enabled"] is False

        assert scheduler.enable_task("toggle_task")
        status = scheduler.get_task_status("toggle_task")
        assert status is not None
        assert status["enabled"] is True

        # Non-existent task
        assert not scheduler.enable_task("nonexistent")
        assert not scheduler.disable_task("nonexistent")

    suite.run_test(
        "Enable/disable task",
        test_enable_disable_task,
        test_summary="Verify task enable/disable logic",
        functions_tested="BackgroundScheduler.enable_task, BackgroundScheduler.disable_task",
        method_description="Toggle task enabled state",
    )

    # Test: Start and stop scheduler
    def test_start_stop_scheduler() -> None:
        scheduler = BackgroundScheduler()
        run_counter = {"count": 0}

        def increment_counter() -> None:
            run_counter["count"] += 1

        scheduler.register_task("quick_task", 0.1, increment_counter, enabled=True, run_on_start=True)

        scheduler.start()
        assert scheduler.is_running

        # Wait for at least one run
        time.sleep(0.2)

        scheduler.stop(timeout=1.0)
        assert not scheduler.is_running
        assert run_counter["count"] >= 1

    suite.run_test(
        "Start and stop scheduler",
        test_start_stop_scheduler,
        test_summary="Verify scheduler lifecycle",
        functions_tested="BackgroundScheduler.start, BackgroundScheduler.stop",
        method_description="Start, wait for task run, stop",
    )

    # Test: Task execution tracking
    def test_task_execution_tracking() -> None:
        scheduler = BackgroundScheduler()
        run_times: list[float] = []

        def track_run() -> None:
            run_times.append(time.time())

        scheduler.register_task("tracked_task", 0.1, track_run, enabled=True)

        scheduler.start()
        time.sleep(0.35)  # Allow 2-3 runs
        scheduler.stop(timeout=1.0)

        status = scheduler.get_task_status("tracked_task")
        assert status is not None
        assert status["run_count"] >= 2
        assert status["error_count"] == 0

    suite.run_test(
        "Task execution tracking",
        test_task_execution_tracking,
        test_summary="Verify task run tracking",
        functions_tested="BackgroundScheduler._execute_task",
        method_description="Track multiple task executions",
    )

    # Test: Task error handling
    def test_task_error_handling() -> None:
        scheduler = BackgroundScheduler()

        def failing_task() -> None:
            raise ValueError("Intentional test error")

        scheduler.register_task("failing_task", 0.1, failing_task, enabled=True)

        scheduler.start()
        time.sleep(0.25)  # Allow 1-2 runs
        scheduler.stop(timeout=1.0)

        status = scheduler.get_task_status("failing_task")
        assert status is not None
        assert status["error_count"] >= 1

    suite.run_test(
        "Task error handling",
        test_task_error_handling,
        test_summary="Verify task errors are tracked",
        functions_tested="BackgroundScheduler._execute_task",
        method_description="Check error count increases on failure",
    )

    # Test: get_scheduler singleton
    def test_get_scheduler_singleton() -> None:
        global _scheduler  # noqa: PLW0603
        _scheduler = None  # Reset for test

        sched1 = get_scheduler()
        sched2 = get_scheduler()
        assert sched1 is sched2

    suite.run_test(
        "get_scheduler singleton",
        test_get_scheduler_singleton,
        test_summary="Verify singleton pattern",
        functions_tested="get_scheduler",
        method_description="Check same instance returned",
    )

    # Test: get_all_status
    def test_get_all_status() -> None:
        scheduler = BackgroundScheduler()
        scheduler.register_task("task_a", 60.0, lambda: None)
        scheduler.register_task("task_b", 120.0, lambda: None)

        statuses = scheduler.get_all_status()
        assert len(statuses) == 2
        names = {s["name"] for s in statuses}
        assert names == {"task_a", "task_b"}

    suite.run_test(
        "get_all_status returns all tasks",
        test_get_all_status,
        test_summary="Verify all task statuses returned",
        functions_tested="BackgroundScheduler.get_all_status",
        method_description="Check status list completeness",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
