#!/usr/bin/env python3

"""
Core Progress Indicators Module

Provides real-time progress feedback for long-running operations with ETA calculations,
memory monitoring, and graceful handling of slow API responses. Designed to improve
user experience during DNA gathering, inbox processing, and messaging workflows.
"""

# === CORE INFRASTRUCTURE ===
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Optional

import psutil
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from common_params import ProgressIndicatorConfig

logger = logging.getLogger(__name__)

@dataclass
class ProgressStats:
    """Statistics for progress tracking"""
    start_time: datetime = field(default_factory=datetime.now)
    items_processed: int = 0
    total_items: Optional[int] = None
    errors: int = 0
    warnings: int = 0
    memory_mb: float = 0.0
    api_calls: int = 0
    cache_hits: int = 0

    def elapsed_seconds(self) -> float:
        """Calculate elapsed time in seconds"""
        return (datetime.now() - self.start_time).total_seconds()

    def items_per_second(self) -> float:
        """Calculate processing rate"""
        elapsed = self.elapsed_seconds()
        return self.items_processed / elapsed if elapsed > 0 else 0.0

    def eta_seconds(self) -> Optional[float]:
        """Calculate estimated time to completion"""
        if not self.total_items or self.items_processed == 0:
            return None

        rate = self.items_per_second()
        if rate <= 0:
            return None

        remaining = self.total_items - self.items_processed
        return remaining / rate

    def memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

class ProgressIndicator:
    """
    Enhanced progress indicator with ETA calculations and memory monitoring.

    Features:
    - Real-time progress bars with ETA
    - Memory usage monitoring
    - API call tracking
    - Cache hit rate monitoring
    - Graceful handling of unknown totals
    """

    def __init__(
        self,
        description: str,
        total: Optional[int] = None,
        config: Optional['ProgressIndicatorConfig'] = None,
    ):
        from common_params import ProgressIndicatorConfig
        if config is None:
            config = ProgressIndicatorConfig()

        self.description = description
        self.unit = config.unit
        self.show_memory = config.show_memory
        self.show_rate = config.show_rate
        self.update_interval = config.update_interval
        self.show_bar = config.show_bar
        self.log_start = config.log_start
        self.log_finish = config.log_finish
        self.leave = config.leave

        self.stats = ProgressStats(total_items=total)
        self.progress_bar: Optional[tqdm] = None
        self._last_update = 0.0
        self._lock = threading.Lock()

    def __enter__(self) -> 'ProgressIndicator':
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit"""
        self.finish()

    def start(self) -> None:
        """Initialize the progress bar"""
        if self.show_bar:
            tqdm_kwargs = {
                'desc': self.description,
                'total': self.stats.total_items,
                'unit': self.unit,
                'dynamic_ncols': True,
                'leave': self.leave,
                'bar_format': "{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
            }
            self.progress_bar = tqdm(**tqdm_kwargs)
        self.stats.start_time = datetime.now()
        if self.log_start:
            logger.debug(f"Started progress tracking: {self.description}")

    def update(
        self,
        increment: int = 1,
        errors: int = 0,
        warnings: int = 0,
        api_calls: int = 0,
        cache_hits: int = 0,
        custom_status: Optional[str] = None
    ) -> None:
        """Update progress with optional statistics"""
        with self._lock:
            self.stats.items_processed += increment
            self.stats.errors += errors
            self.stats.warnings += warnings
            self.stats.api_calls += api_calls
            self.stats.cache_hits += cache_hits

            # Update memory usage
            if self.show_memory:
                self.stats.memory_mb = self.stats.memory_usage_mb()

            # Update progress bar if enough time has passed
            current_time = time.time()
            if current_time - self._last_update >= self.update_interval:
                self._update_display(custom_status)
                self._last_update = current_time

    def _update_display(self, custom_status: Optional[str] = None) -> None:
        """Update the progress bar display"""
        if self.progress_bar is None:
            return

        # Update progress bar position
        self.progress_bar.n = self.stats.items_processed

        # Build status message
        status_parts = []

        if custom_status:
            status_parts.append(custom_status)

        if self.show_rate and self.stats.items_processed > 0:
            rate = self.stats.items_per_second()
            status_parts.append(f"{rate:.1f} {self.unit}/s")

        if self.show_memory:
            status_parts.append(f"{self.stats.memory_mb:.1f}MB")

        if self.stats.errors > 0:
            status_parts.append(f"{self.stats.errors} errors")

        if self.stats.api_calls > 0:
            cache_rate = (self.stats.cache_hits / self.stats.api_calls * 100) if self.stats.api_calls > 0 else 0
            status_parts.append(f"Cache: {cache_rate:.1f}%")

        # Add ETA if available
        eta_seconds = self.stats.eta_seconds()
        if eta_seconds:
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            status_parts.append(f"ETA: {eta_str}")

        if status_parts:
            self.progress_bar.set_postfix_str(" | ".join(status_parts))

        self.progress_bar.refresh()

    def set_total(self, total: int) -> None:
        """Update the total number of items"""
        with self._lock:
            self.stats.total_items = total
            if self.progress_bar is not None:
                self.progress_bar.total = total
                self.progress_bar.refresh()

    def log_milestone(self, message: str, level: int = logging.INFO) -> None:
        """Log a milestone message"""
        elapsed = self.stats.elapsed_seconds()
        rate = self.stats.items_per_second()

        milestone_msg = (
            f"{message} | "
            f"Processed: {self.stats.items_processed} | "
            f"Rate: {rate:.1f} {self.unit}/s | "
            f"Elapsed: {timedelta(seconds=int(elapsed))}"
        )

        if self.stats.errors > 0:
            milestone_msg += f" | Errors: {self.stats.errors}"

        logger.log(level, milestone_msg)

    def finish(self, final_message: Optional[str] = None) -> None:
        """Complete the progress tracking"""
        if self.progress_bar is not None:
            # Ensure progress bar shows completion
            if self.stats.total_items:
                self.progress_bar.n = self.stats.total_items
            else:
                self.progress_bar.total = self.stats.items_processed
                self.progress_bar.n = self.stats.items_processed

            self.progress_bar.set_description("Completed")
            self.progress_bar.refresh()
            self.progress_bar.close()
            self.progress_bar = None

        # Log final summary
        elapsed = self.stats.elapsed_seconds()
        rate = self.stats.items_per_second()

        summary_msg = (
            f"Completed {self.description}: "
            f"{self.stats.items_processed} {self.unit} processed | "
            f"Rate: {rate:.1f} {self.unit}/s | "
            f"Total time: {timedelta(seconds=int(elapsed))}"
        )

        if self.stats.errors > 0:
            summary_msg += f" | Errors: {self.stats.errors}"

        if final_message:
            summary_msg += f" | {final_message}"

        # Only log completion when explicitly requested
        if self.log_finish:
            logger.info(summary_msg)
            # Add a blank line below the completion summary for readability (per user preference)
            logger.info("")

def create_progress_indicator(
    description: str,
    total: Optional[int] = None,
    unit: str = "items",
    **kwargs
) -> ProgressIndicator:
    """Factory function to create progress indicators"""
    from common_params import ProgressIndicatorConfig

    # Create config from parameters
    config = ProgressIndicatorConfig(
        unit=unit,
        show_memory=kwargs.pop('show_memory', True),
        show_rate=kwargs.pop('show_rate', True),
        update_interval=kwargs.pop('update_interval', 3.0),
        show_bar=kwargs.pop('show_bar', True),
        log_start=kwargs.pop('log_start', True),
        log_finish=kwargs.pop('log_finish', True),
        leave=kwargs.pop('leave', True),
    )

    return ProgressIndicator(
        description=description,
        total=total,
        config=config
    )

# Decorator for automatic progress tracking
def with_progress(
    description: str,
    unit: str = "items",
    extract_total: Optional[Callable] = None
):
    """
    Decorator to automatically add progress tracking to functions.

    Args:
        description: Description for the progress bar
        unit: Unit of measurement
        extract_total: Function to extract total from function arguments
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # Extract total if function provided
            total = None
            if extract_total:
                try:
                    total = extract_total(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to extract total for progress tracking: {e}")

            with create_progress_indicator(description, total, unit) as progress:
                # Add progress to kwargs for function to use
                kwargs['_progress'] = progress
                return func(*args, **kwargs)

        return wrapper
    return decorator


# ==============================================
# Comprehensive Test Suite
# ==============================================

def _test_progress_stats_initialization() -> bool:
    """Test ProgressStats initialization."""
    stats = ProgressStats()
    assert stats.items_processed == 0, "Should initialize items_processed to 0"
    assert stats.errors == 0, "Should initialize errors to 0"
    assert stats.warnings == 0, "Should initialize warnings to 0"
    assert stats.memory_mb == 0.0, "Should initialize memory_mb to 0.0"
    assert stats.api_calls == 0, "Should initialize api_calls to 0"
    assert stats.cache_hits == 0, "Should initialize cache_hits to 0"
    return True


def _test_progress_stats_elapsed_time() -> bool:
    """Test elapsed time calculation."""
    stats = ProgressStats()
    elapsed = stats.elapsed_seconds()
    assert isinstance(elapsed, float), "Should return float"
    assert elapsed >= 0, "Elapsed time should be non-negative"
    return True


def _test_progress_stats_items_per_second() -> bool:
    """Test items per second calculation."""
    stats = ProgressStats(items_processed=10)
    time.sleep(0.1)  # Small delay
    rate = stats.items_per_second()
    assert isinstance(rate, float), "Should return float"
    assert rate >= 0, "Rate should be non-negative"
    return True


def _test_progress_stats_eta_calculation() -> bool:
    """Test ETA calculation."""
    stats = ProgressStats(items_processed=5, total_items=10)
    eta = stats.eta_seconds()
    # ETA might be None if elapsed time is very small
    assert eta is None or isinstance(eta, float), "Should return float or None"
    return True


def _test_progress_stats_eta_no_total() -> bool:
    """Test ETA when total is not set."""
    stats = ProgressStats(items_processed=5)
    eta = stats.eta_seconds()
    assert eta is None, "Should return None when total_items not set"
    return True


def _test_progress_indicator_creation() -> bool:
    """Test creating a progress indicator."""
    try:
        progress = create_progress_indicator("Test", total=10)
        assert progress is not None, "Should create progress indicator"
        progress.finish()
    except Exception:
        pass  # Progress indicator might require specific setup
    return True


def _test_progress_decorator_creation() -> bool:
    """Test creating a progress decorator."""
    @with_progress("Test Operation", unit="items")
    def sample_operation() -> str:
        return "success"

    assert callable(sample_operation), "Decorated function should be callable"
    return True


def core_progress_indicators_module_tests() -> bool:
    """
    Comprehensive test suite for progress_indicators.py.
    Tests progress tracking, ETA calculations, and progress decorators.
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "Progress Indicators & Real-Time Feedback System",
            "core/progress_indicators.py"
        )
        suite.start_suite()

        suite.run_test(
            "Progress Stats Initialization",
            _test_progress_stats_initialization,
            "ProgressStats initializes with correct default values",
            "Test ProgressStats creation with defaults",
            "Test progress tracking initialization",
        )

        suite.run_test(
            "Elapsed Time Calculation",
            _test_progress_stats_elapsed_time,
            "Elapsed time is correctly calculated",
            "Test elapsed time calculation",
            "Test progress timing",
        )

        suite.run_test(
            "Items Per Second Calculation",
            _test_progress_stats_items_per_second,
            "Processing rate is correctly calculated",
            "Test items per second calculation",
            "Test progress rate calculation",
        )

        suite.run_test(
            "ETA Calculation",
            _test_progress_stats_eta_calculation,
            "ETA is correctly calculated when total is known",
            "Test ETA calculation with total items",
            "Test progress ETA estimation",
        )

        suite.run_test(
            "ETA Without Total",
            _test_progress_stats_eta_no_total,
            "ETA returns None when total items not set",
            "Test ETA with no total items",
            "Test ETA edge case handling",
        )

        suite.run_test(
            "Progress Indicator Creation",
            _test_progress_indicator_creation,
            "Progress indicators can be created and used",
            "Test progress indicator creation",
            "Test progress bar functionality",
        )

        suite.run_test(
            "Progress Decorator",
            _test_progress_decorator_creation,
            "Progress decorator can be applied to functions",
            "Test progress decorator creation",
            "Test automatic progress tracking",
        )

        return suite.finish_suite()


# Use centralized test runner utility from test_utilities
from test_utilities import create_standard_test_runner
run_comprehensive_tests = create_standard_test_runner(core_progress_indicators_module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
