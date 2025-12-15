#!/usr/bin/env python3
"""
Startup Health Checks and Runtime Monitoring.

This module provides systematic health checks for verifying system readiness
before operations. It implements the HealthCheck protocol with checks for:
- Database connectivity and schema
- API availability and authentication
- File system paths and permissions
- Cache systems

Phase 1 Implementation (Nov 2025):
- HealthCheck protocol for consistent check interface
- DatabaseHealthCheck, FileSystemHealthCheck, CacheHealthCheck implementations
- HealthCheckRunner for aggregated startup validation
- Integration with main menu via 'health' action
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from sqlalchemy import text as sa_text

if __package__ in {None, ""}:
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.session_manager import SessionManager


class HealthStatus(Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    name: str
    status: HealthStatus
    message: str
    duration_ms: float = 0.0
    details: dict[str, object] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Return True if status is HEALTHY."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_critical(self) -> bool:
        """Return True if status is UNHEALTHY."""
        return self.status == HealthStatus.UNHEALTHY

    def __str__(self) -> str:
        icons = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.DEGRADED: "⚠️",
            HealthStatus.UNHEALTHY: "❌",
            HealthStatus.UNKNOWN: "❓",
        }
        icon = icons.get(self.status, "❓")
        return f"{icon} {self.name}: {self.message} ({self.duration_ms:.1f}ms)"


@dataclass
class HealthReport:
    """Aggregated health check results."""

    results: list[HealthCheckResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    @property
    def overall_status(self) -> HealthStatus:
        """Return overall health status based on all check results."""
        if not self.results:
            return HealthStatus.UNKNOWN

        has_unhealthy = any(r.status == HealthStatus.UNHEALTHY for r in self.results)
        has_degraded = any(r.status == HealthStatus.DEGRADED for r in self.results)

        if has_unhealthy:
            return HealthStatus.UNHEALTHY
        if has_degraded:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    @property
    def is_healthy(self) -> bool:
        """Return True if overall status is HEALTHY."""
        return self.overall_status == HealthStatus.HEALTHY

    @property
    def total_duration_ms(self) -> float:
        """Return total duration of all health checks in milliseconds."""
        return (self.end_time - self.start_time) * 1000 if self.end_time > 0 else 0.0

    def add(self, result: HealthCheckResult) -> None:
        """Add a health check result."""
        self.results.append(result)

    @staticmethod
    def _print_result_group(header: str, results: list[HealthCheckResult], *, show_details: bool = False) -> None:
        """Print a group of health check results."""
        if not results:
            return
        print(f"\n{header}")
        for r in results:
            print(f"   {r}")
            if show_details and r.details:
                for key, value in r.details.items():
                    print(f"      • {key}: {value}")

    @staticmethod
    def _get_status_icon(status: HealthStatus) -> str:
        """Get emoji icon for a health status."""
        return {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.DEGRADED: "⚠️",
            HealthStatus.UNHEALTHY: "❌",
            HealthStatus.UNKNOWN: "❓",
        }[status]

    def print_report(self) -> None:
        """Print formatted health report."""
        print("\n" + "=" * 60)
        print(" SYSTEM HEALTH CHECK ".center(60, "="))
        print("=" * 60)

        if not self.results:
            print("\n❓ No health checks executed")
            return

        # Group by status
        healthy = [r for r in self.results if r.status == HealthStatus.HEALTHY]
        degraded = [r for r in self.results if r.status == HealthStatus.DEGRADED]
        unhealthy = [r for r in self.results if r.status == HealthStatus.UNHEALTHY]
        unknown = [r for r in self.results if r.status == HealthStatus.UNKNOWN]

        # Print each group
        self._print_result_group("❌ CRITICAL ISSUES:", unhealthy, show_details=True)
        self._print_result_group("⚠️  WARNINGS:", degraded)
        self._print_result_group("✅ HEALTHY:", healthy)
        self._print_result_group("❓ UNKNOWN:", unknown)

        # Summary
        total = len(self.results)
        print("\n" + "-" * 60)
        status_icon = self._get_status_icon(self.overall_status)
        print(
            f"{status_icon} Overall: {self.overall_status.value.upper()} "
            f"({len(healthy)}/{total} healthy, {self.total_duration_ms:.0f}ms total)"
        )
        print("=" * 60 + "\n")


class HealthCheck(ABC):
    """Abstract base class for health checks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this health check."""
        ...

    @abstractmethod
    def check(self) -> HealthCheckResult:
        """Execute the health check and return result."""
        ...

    def _timed_check(self, check_func: Callable[[], HealthCheckResult]) -> HealthCheckResult:
        """Execute a check function and measure duration."""
        start = time.time()
        try:
            result = check_func()
            duration_ms = (time.time() - start) * 1000
            result.duration_ms = duration_ms
            return result
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed with error: {e}",
                duration_ms=duration_ms,
            )


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity and schema."""

    @property
    def name(self) -> str:
        return "Database"

    def check(self) -> HealthCheckResult:
        """Check database connectivity and basic schema."""
        return self._timed_check(self._do_check)

    def _do_check(self) -> HealthCheckResult:
        """Execute the actual database check."""
        try:
            from core.database_manager import DatabaseManager

            db_manager = DatabaseManager()

            # Check if database is ready
            if not db_manager.ensure_ready():
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Database failed to initialize",
                    details={"suggestion": "Check DATABASE_FILE path and permissions"},
                )

            # Try a simple query
            engine = getattr(db_manager, "engine", None)
            if engine is None:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Database engine not available",
                )

            # Execute simple query to verify connectivity
            text_func: Callable[[str], object] | None = getattr(db_manager, "_text_func", None)
            query = text_func("SELECT 1") if callable(text_func) else sa_text("SELECT 1")

            with engine.connect() as conn:
                result = conn.execute(query)
                _ = result.fetchone()

            # Clean up
            db_manager.close_connections(dispose_engine=True)

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Database connected and responsive",
            )

        except ImportError as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database module not available: {e}",
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {e}",
                details={"error_type": type(e).__name__},
            )


class FileSystemHealthCheck(HealthCheck):
    """Health check for required file system paths."""

    @property
    def name(self) -> str:
        return "File System"

    def check(self) -> HealthCheckResult:
        """Check required directories exist and are writable."""
        return self._timed_check(self._do_check)

    def _do_check(self) -> HealthCheckResult:
        """Execute the actual file system check."""
        required_dirs = [
            ("Data", Path("Data")),
            ("Logs", Path("Logs")),
            ("Cache", Path("Cache")),
        ]

        missing_dirs: list[str] = []
        not_writable: list[str] = []

        for name, path in required_dirs:
            if not path.exists():
                missing_dirs.append(name)
            elif not self._is_writable(path):
                not_writable.append(name)

        if missing_dirs:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message=f"Missing directories: {', '.join(missing_dirs)}",
                details={"missing": missing_dirs, "suggestion": "Create missing directories"},
            )

        if not_writable:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Directories not writable: {', '.join(not_writable)}",
                details={"not_writable": not_writable},
            )

        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.HEALTHY,
            message="All required directories present and writable",
        )

    @staticmethod
    def _is_writable(path: Path) -> bool:
        """Check if a directory is writable."""
        try:
            test_file = path / ".health_check_test"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception:
            return False


class CacheHealthCheck(HealthCheck):
    """Health check for cache systems."""

    @property
    def name(self) -> str:
        return "Cache"

    def check(self) -> HealthCheckResult:
        """Check cache systems are operational."""
        return self._timed_check(self._do_check)

    def _do_check(self) -> HealthCheckResult:
        """Execute the actual cache check."""
        cache_dir = Path("Cache")

        if not cache_dir.exists():
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message="Cache directory does not exist",
                details={"suggestion": "Create Cache directory for optimal performance"},
            )

        # Check cache directory size
        try:
            total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            size_mb = total_size / (1024 * 1024)

            if size_mb > 500:  # Warning if cache > 500MB
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"Cache size is large: {size_mb:.1f}MB",
                    details={"size_mb": size_mb, "suggestion": "Consider clearing old cache files"},
                )

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=f"Cache operational ({size_mb:.1f}MB used)",
                details={"size_mb": size_mb},
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message=f"Could not check cache size: {e}",
            )


class ConfigurationHealthCheck(HealthCheck):
    """Health check for configuration validity."""

    @property
    def name(self) -> str:
        return "Configuration"

    def check(self) -> HealthCheckResult:
        """Check configuration is valid and complete."""
        return self._timed_check(self._do_check)

    def _do_check(self) -> HealthCheckResult:
        """Execute the actual configuration check."""
        try:
            from config.validator import ConfigurationValidator

            validator = ConfigurationValidator()
            report = validator.validate_all()

            if not report.config_loaded:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Configuration failed to load",
                    details={"suggestion": "Check .env file exists and is properly formatted"},
                )

            error_count = len(report.errors)
            warning_count = len(report.warnings)

            if error_count > 0:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Configuration has {error_count} error(s)",
                    details={"errors": error_count, "warnings": warning_count},
                )

            if warning_count > 0:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"Configuration has {warning_count} warning(s)",
                    details={"warnings": warning_count},
                )

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Configuration valid",
            )

        except ImportError as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Configuration validator not available: {e}",
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Configuration check failed: {e}",
            )


class SessionHealthCheck(HealthCheck):
    """Health check for session manager (optional - requires session_manager)."""

    def __init__(self, session_manager: Optional[SessionManager] = None):
        self._session_manager = session_manager

    @property
    def name(self) -> str:
        return "Session"

    def check(self) -> HealthCheckResult:
        """Check session manager status."""
        return self._timed_check(self._do_check)

    def _do_check(self) -> HealthCheckResult:
        """Execute the actual session check."""
        if self._session_manager is None:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="Session manager not provided",
            )

        try:
            # Check if session is valid
            is_valid = self._session_manager.is_sess_valid()
            session_age = self._session_manager.session_age_seconds()
            age_value = session_age if session_age is not None else 0.0

            if is_valid:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"Session valid (age: {age_value:.0f}s)",
                    details={"age_seconds": age_value},
                )

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message="Session not valid or not established",
                details={"suggestion": "Session will be established when needed"},
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message=f"Could not check session: {e}",
            )


class HealthCheckRunner:
    """Runs multiple health checks and aggregates results."""

    def __init__(self, session_manager: Optional[SessionManager] = None):
        self._session_manager = session_manager
        self._checks: list[HealthCheck] = []
        self._setup_default_checks()

    def _setup_default_checks(self) -> None:
        """Setup default health checks."""
        self._checks = [
            ConfigurationHealthCheck(),
            FileSystemHealthCheck(),
            DatabaseHealthCheck(),
            CacheHealthCheck(),
        ]

        # Only add session check if browser has been initialized
        # (session check is only meaningful after browser actions have run)
        if self._session_manager is not None and self._session_manager.is_sess_valid():
            self._checks.append(SessionHealthCheck(self._session_manager))

    def add_check(self, check: HealthCheck) -> None:
        """Add a custom health check."""
        self._checks.append(check)

    def run_all(self) -> HealthReport:
        """Run all health checks and return aggregated report."""
        report = HealthReport()

        for check in self._checks:
            try:
                result = check.check()
                report.add(result)
            except Exception as e:
                report.add(
                    HealthCheckResult(
                        name=check.name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed unexpectedly: {e}",
                    )
                )

        report.end_time = time.time()
        return report

    @staticmethod
    def run_quick() -> HealthReport:
        """Run only quick health checks (no database)."""
        report = HealthReport()

        quick_checks: list[HealthCheck] = [
            ConfigurationHealthCheck(),
            FileSystemHealthCheck(),
            CacheHealthCheck(),
        ]

        for check in quick_checks:
            try:
                result = check.check()
                report.add(result)
            except Exception as e:
                report.add(
                    HealthCheckResult(
                        name=check.name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {e}",
                    )
                )

        report.end_time = time.time()
        return report

    def get_check_count(self) -> int:
        """Get the number of registered health checks."""
        return len(self._checks)


def run_startup_health_checks(session_manager: Optional[SessionManager] = None) -> bool:
    """
    Run startup health checks and return whether system is healthy.

    This should be called early in main() before any actions.

    Returns:
        True if all critical checks passed, False otherwise
    """
    runner = HealthCheckRunner(session_manager)
    report = runner.run_all()

    if report.overall_status == HealthStatus.UNHEALTHY:
        report.print_report()
        return False

    # Log success without full report
    healthy_count = sum(1 for r in report.results if r.is_healthy)
    logger.info(f"✅ Health checks passed: {healthy_count}/{len(report.results)} healthy")
    return True


def run_interactive_health_check(session_manager: Optional[SessionManager] = None) -> HealthReport:
    """
    Run interactive health check from main menu.

    Returns:
        HealthReport with all check results
    """
    runner = HealthCheckRunner(session_manager)
    report = runner.run_all()
    report.print_report()
    return report


# === Module Tests ===


def _test_health_status_enum() -> bool:
    """Test HealthStatus enum values."""
    assert HealthStatus.HEALTHY.value == "healthy"
    assert HealthStatus.DEGRADED.value == "degraded"
    assert HealthStatus.UNHEALTHY.value == "unhealthy"
    assert HealthStatus.UNKNOWN.value == "unknown"
    return True


def _test_health_check_result_properties() -> bool:
    """Test HealthCheckResult properties."""
    healthy = HealthCheckResult(name="Test", status=HealthStatus.HEALTHY, message="OK")
    assert healthy.is_healthy is True
    assert healthy.is_critical is False

    unhealthy = HealthCheckResult(name="Test", status=HealthStatus.UNHEALTHY, message="Failed")
    assert unhealthy.is_healthy is False
    assert unhealthy.is_critical is True

    return True


def _test_health_check_result_str() -> bool:
    """Test HealthCheckResult string formatting."""
    result = HealthCheckResult(name="Test", status=HealthStatus.HEALTHY, message="OK", duration_ms=50.5)
    result_str = str(result)

    assert "✅" in result_str, "Should have checkmark for healthy"
    assert "Test" in result_str, "Should include name"
    assert "OK" in result_str, "Should include message"
    assert "50.5ms" in result_str, "Should include duration"

    return True


def _test_health_report_overall_status() -> bool:
    """Test HealthReport overall status calculation."""
    report = HealthReport()

    # Empty report
    assert report.overall_status == HealthStatus.UNKNOWN

    # All healthy
    report.add(HealthCheckResult(name="A", status=HealthStatus.HEALTHY, message="OK"))
    report.add(HealthCheckResult(name="B", status=HealthStatus.HEALTHY, message="OK"))
    assert report.overall_status == HealthStatus.HEALTHY

    # One degraded
    report.add(HealthCheckResult(name="C", status=HealthStatus.DEGRADED, message="Warn"))
    assert report.overall_status == HealthStatus.DEGRADED

    # One unhealthy
    report.add(HealthCheckResult(name="D", status=HealthStatus.UNHEALTHY, message="Fail"))
    assert report.overall_status == HealthStatus.UNHEALTHY

    return True


def _test_file_system_health_check() -> bool:
    """Test FileSystemHealthCheck."""
    check = FileSystemHealthCheck()
    assert check.name == "File System"

    result = check.check()
    assert isinstance(result, HealthCheckResult)
    assert result.duration_ms >= 0
    # Result status depends on actual filesystem

    return True


def _test_cache_health_check() -> bool:
    """Test CacheHealthCheck."""
    check = CacheHealthCheck()
    assert check.name == "Cache"

    result = check.check()
    assert isinstance(result, HealthCheckResult)
    # Result status depends on actual cache directory

    return True


def _test_configuration_health_check() -> bool:
    """Test ConfigurationHealthCheck."""
    check = ConfigurationHealthCheck()
    assert check.name == "Configuration"

    result = check.check()
    assert isinstance(result, HealthCheckResult)
    # Result depends on actual config state

    return True


def _test_health_check_runner() -> bool:
    """Test HealthCheckRunner."""
    runner = HealthCheckRunner()

    # Should have default checks
    assert runner.get_check_count() >= 3

    # Run quick checks (no database)
    report = runner.run_quick()
    assert isinstance(report, HealthReport)
    assert len(report.results) >= 3

    return True


def _test_run_startup_health_checks_returns_bool() -> bool:
    """Test that run_startup_health_checks returns a boolean."""
    from unittest.mock import patch

    # Mock print to avoid output
    with patch("builtins.print"):
        result = run_startup_health_checks()
        assert isinstance(result, bool)

    return True


def _test_run_interactive_health_check_returns_report() -> bool:
    """Test that run_interactive_health_check returns a HealthReport."""
    from unittest.mock import patch

    with patch("builtins.print"):
        report = run_interactive_health_check()
        assert isinstance(report, HealthReport)

    return True


def module_tests() -> bool:
    """Run module tests for core.health_check."""
    from testing.test_framework import TestSuite

    suite = TestSuite("core.health_check", "core/health_check.py")

    suite.run_test(
        "HealthStatus enum values",
        _test_health_status_enum,
        "Ensures HealthStatus enum has expected values.",
    )

    suite.run_test(
        "HealthCheckResult properties",
        _test_health_check_result_properties,
        "Ensures HealthCheckResult has correct is_healthy and is_critical.",
    )

    suite.run_test(
        "HealthCheckResult string formatting",
        _test_health_check_result_str,
        "Ensures HealthCheckResult formats correctly.",
    )

    suite.run_test(
        "HealthReport overall status",
        _test_health_report_overall_status,
        "Ensures HealthReport calculates overall status correctly.",
    )

    suite.run_test(
        "FileSystemHealthCheck",
        _test_file_system_health_check,
        "Ensures FileSystemHealthCheck executes.",
    )

    suite.run_test(
        "CacheHealthCheck",
        _test_cache_health_check,
        "Ensures CacheHealthCheck executes.",
    )

    suite.run_test(
        "ConfigurationHealthCheck",
        _test_configuration_health_check,
        "Ensures ConfigurationHealthCheck executes.",
    )

    suite.run_test(
        "HealthCheckRunner",
        _test_health_check_runner,
        "Ensures HealthCheckRunner runs checks.",
    )

    suite.run_test(
        "run_startup_health_checks returns bool",
        _test_run_startup_health_checks_returns_bool,
        "Ensures startup health checks return boolean.",
    )

    suite.run_test(
        "run_interactive_health_check returns report",
        _test_run_interactive_health_check_returns_report,
        "Ensures interactive health check returns HealthReport.",
    )

    return suite.finish_suite()


if __name__ == "__main__":
    from testing.test_utilities import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
