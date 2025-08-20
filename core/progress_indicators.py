#!/usr/bin/env python3

"""
Core Progress Indicators Module

Provides real-time progress feedback for long-running operations with ETA calculations,
memory monitoring, and graceful handling of slow API responses. Designed to improve
user experience during DNA gathering, inbox processing, and messaging workflows.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

import psutil
from tqdm.auto import tqdm

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
        unit: str = "items",
        show_memory: bool = True,
        show_rate: bool = True,
        update_interval: float = 3.0,
        show_bar: bool = True,
        log_start: bool = True,
        log_finish: bool = True,
        leave: bool = True,
    ):
        self.description = description
        self.unit = unit
        self.show_memory = show_memory
        self.show_rate = show_rate
        self.update_interval = update_interval
        self.show_bar = show_bar
        self.log_start = log_start
        self.log_finish = log_finish
        self.leave = leave

        self.stats = ProgressStats(total_items=total)
        self.progress_bar: Optional[tqdm] = None
        self._last_update = 0.0
        self._lock = threading.Lock()

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.finish()

    def start(self):
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
            logger.info(f"Started progress tracking: {self.description}")

    def update(
        self,
        increment: int = 1,
        errors: int = 0,
        warnings: int = 0,
        api_calls: int = 0,
        cache_hits: int = 0,
        custom_status: Optional[str] = None
    ):
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

    def _update_display(self, custom_status: Optional[str] = None):
        """Update the progress bar display"""
        if not self.progress_bar:
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

    def set_total(self, total: int):
        """Update the total number of items"""
        with self._lock:
            self.stats.total_items = total
            if self.progress_bar:
                self.progress_bar.total = total
                self.progress_bar.refresh()

    def log_milestone(self, message: str, level: int = logging.INFO):
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

    def finish(self, final_message: Optional[str] = None):
        """Complete the progress tracking"""
        if self.progress_bar:
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
    return ProgressIndicator(
        description=description,
        total=total,
        unit=unit,
        **kwargs
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
    def decorator(func):
        def wrapper(*args, **kwargs):
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
