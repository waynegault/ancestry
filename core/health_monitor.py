#!/usr/bin/env python3
"""
Enhanced health monitoring system for Action 6.
Provides real-time health metrics and early warning detection.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

@dataclass
class HealthMetrics:
    """Health metrics for monitoring Action 6 performance."""
    api_success_rate: float
    avg_response_time: float
    cascade_count: int
    pages_processed: int
    matches_processed: int
    errors_per_minute: float
    session_age_minutes: float
    last_successful_api_call: Optional[datetime]

class HealthMonitor:
    """Enhanced health monitoring for Action 6."""

    def __init__(self):
        self.start_time = datetime.now()
        self.api_calls_total = 0
        self.api_calls_successful = 0
        self.api_response_times = []
        self.error_timestamps = []
        self.last_health_check = datetime.now()

    def record_api_call(self, success: bool, response_time: float):
        """Record API call metrics."""
        self.api_calls_total += 1
        if success:
            self.api_calls_successful += 1
            self.api_response_times.append(response_time)
        else:
            self.error_timestamps.append(datetime.now())

    def get_health_metrics(self, cascade_count: int, pages_processed: int, matches_processed: int) -> HealthMetrics:
        """Get current health metrics."""
        now = datetime.now()

        # Calculate success rate
        success_rate = (self.api_calls_successful / max(1, self.api_calls_total)) * 100

        # Calculate average response time
        avg_response_time = sum(self.api_response_times[-100:]) / max(1, len(self.api_response_times[-100:]))

        # Calculate errors per minute
        recent_errors = [ts for ts in self.error_timestamps if now - ts < timedelta(minutes=1)]
        errors_per_minute = len(recent_errors)

        # Session age
        session_age = (now - self.start_time).total_seconds() / 60

        # Last successful API call
        last_success = now if self.api_calls_successful > 0 else None

        return HealthMetrics(
            api_success_rate=success_rate,
            avg_response_time=avg_response_time,
            cascade_count=cascade_count,
            pages_processed=pages_processed,
            matches_processed=matches_processed,
            errors_per_minute=errors_per_minute,
            session_age_minutes=session_age,
            last_successful_api_call=last_success
        )

    def check_health_warnings(self, metrics: HealthMetrics) -> List[str]:
        """Check for health warning conditions."""
        warnings = []

        if metrics.api_success_rate < 50:
            warnings.append(f"🚨 LOW SUCCESS RATE: {metrics.api_success_rate:.1f}% (threshold: 50%)")

        if metrics.avg_response_time > 10.0:
            warnings.append(f"⚠️ SLOW RESPONSES: {metrics.avg_response_time:.1f}s avg (threshold: 10s)")

        if metrics.errors_per_minute > 10:
            warnings.append(f"🚨 HIGH ERROR RATE: {metrics.errors_per_minute} errors/min (threshold: 10)")

        if metrics.session_age_minutes > 120:  # 2 hours
            warnings.append(f"⚠️ LONG SESSION: {metrics.session_age_minutes:.1f} min (threshold: 120 min)")

        if metrics.last_successful_api_call and (datetime.now() - metrics.last_successful_api_call).total_seconds() > 300:
            warnings.append("🚨 NO RECENT SUCCESS: No successful API calls in 5+ minutes")

        return warnings

    def should_recommend_restart(self, metrics: HealthMetrics) -> bool:
        """Determine if a restart is recommended."""
        critical_conditions = [
            metrics.api_success_rate < 20,  # Very low success rate
            metrics.errors_per_minute > 20,  # Very high error rate
            metrics.cascade_count > 1,  # Any cascades detected
            metrics.session_age_minutes > 180,  # Very long session
        ]

        return any(critical_conditions)
