#!/usr/bin/env python3

"""
Performance Profiling Utilities

Provides reusable cProfile/timing helpers with decorators and CLI integration
for capturing detailed performance data from long-running actions without manual scripts.

Key Features:
• cProfile integration via @profile_with_cprofile decorator
• Lightweight timing via @time_function decorator
• CLI switches for enabling profiling (--profile, --profile-output)
• Automatic profile data export (stats file + human-readable report)
• Integration-ready for main.py exec_actn() wrapper

Usage Examples:

    # Decorator on action functions
    @profile_with_cprofile(output_file="action6_profile.stats")
    def coord(session_manager, start=None):
        # ... action implementation
        pass

    # CLI invocation
    python main.py --profile --profile-output=custom_profile.stats

    # Lightweight timing (no cProfile overhead)
    @time_function
    def quick_helper():
        pass

Design Philosophy:
• Zero configuration required for basic profiling
• Opt-in via CLI flags (no performance impact when disabled)
• Generates both machine-readable (.stats) and human-readable (.txt) reports
• Compatible with existing action module signatures
"""

# === CORE INFRASTRUCTURE ===
import logging

logger = logging.getLogger(__name__)

# === STANDARD LIBRARY IMPORTS ===
import cProfile
import io
import pstats
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

# Type variables for generic decorator support
FuncType = TypeVar('FuncType', bound=Callable[..., Any])
R = TypeVar('R')


# === CONFIGURATION ===


@dataclass
class ProfileConfig:
    """Configuration for profiling operations."""

    enabled: bool = False
    output_dir: Path = Path("Logs/profiles")
    default_filename: str = "profile.stats"
    sort_by: str = "cumulative"  # Sort stats by cumulative time
    top_n_functions: int = 50  # Show top 50 functions in reports
    strip_dirs: bool = True  # Strip directory paths for cleaner output
    generate_txt_report: bool = True  # Generate human-readable .txt alongside .stats


# Global profiling configuration
_profile_config = ProfileConfig()


def configure_profiling(
    enabled: bool = False,
    output_dir: Path | None = None,
    sort_by: str = "cumulative",
    top_n_functions: int = 50,
) -> None:
    """
    Configure global profiling settings.

    Args:
        enabled: Enable profiling globally
        output_dir: Directory for profile output files
        sort_by: Stat sorting key (cumulative, time, calls, etc.)
        top_n_functions: Number of functions to show in reports
    """
    _profile_config.enabled = enabled
    if output_dir:
        _profile_config.output_dir = output_dir
    _profile_config.sort_by = sort_by
    _profile_config.top_n_functions = top_n_functions

    # Ensure output directory exists
    _profile_config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Profiling configured: enabled={enabled}, output_dir={_profile_config.output_dir}")


def enable_profiling_from_cli() -> None:
    """
    Enable profiling based on CLI arguments.

    Checks for --profile and --profile-output flags in sys.argv.
    Should be called early in main.py before action execution.

    Example:
        python main.py --profile
        python main.py --profile --profile-output=custom_profile.stats
    """
    import sys

    if "--profile" in sys.argv:
        # Remove the flag so it doesn't interfere with other arg parsing
        sys.argv.remove("--profile")

        # Check for custom output filename
        output_file = None
        for arg in sys.argv[:]:
            if arg.startswith("--profile-output="):
                output_file = arg.split("=", 1)[1]
                sys.argv.remove(arg)
                break

        configure_profiling(enabled=True)

        if output_file:
            _profile_config.default_filename = output_file

        logger.info(f"CLI profiling enabled. Output: {_profile_config.default_filename}")


def is_profiling_enabled() -> bool:
    """Check if profiling is currently enabled."""
    return _profile_config.enabled


# === PROFILING DECORATORS ===


def profile_with_cprofile(
    output_file: str | None = None,
    enabled: bool | None = None,
    sort_by: str | None = None,
) -> Callable[[FuncType], FuncType]:
    """
    Decorator to profile a function with cProfile.

    Generates both a .stats file (for programmatic analysis) and a .txt file
    (for human review). Stats are sorted by cumulative time by default.

    Args:
        output_file: Custom output filename (without directory)
        enabled: Override global profiling config (None = use global)
        sort_by: Custom sort key (cumulative, time, calls, etc.)

    Returns:
        Decorated function that profiles execution when enabled

    Example:
        @profile_with_cprofile(output_file="action6_profile.stats")
        def coord(session_manager, start=None):
            # ... action implementation
            pass
    """

    def decorator(func: FuncType) -> FuncType:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if profiling is enabled (decorator arg overrides global)
            profiling_active = enabled if enabled is not None else _profile_config.enabled

            if not profiling_active:
                # Profiling disabled - execute normally with no overhead
                return func(*args, **kwargs)

            # Determine output filename
            filename = output_file or _profile_config.default_filename
            stats_path = _profile_config.output_dir / filename

            # Ensure .stats extension
            if stats_path.suffix != '.stats':
                stats_path = stats_path.with_suffix(".stats")

            logger.info(f"📊 Profiling {func.__name__} → {stats_path}")

            # Run function with cProfile
            profiler = cProfile.Profile()
            try:
                profiler.enable()
                result = func(*args, **kwargs)
                profiler.disable()

                # Save stats file
                profiler.dump_stats(str(stats_path))
                logger.info(f"✅ Profile stats saved: {stats_path}")

                # Generate human-readable report
                if _profile_config.generate_txt_report:
                    _generate_txt_report(stats_path, sort_by or _profile_config.sort_by)

                return result

            except Exception as e:
                # Still save profile data even if function failed
                profiler.disable()
                profiler.dump_stats(str(stats_path))
                logger.warning(f"⚠️  Function failed but profile saved: {stats_path}")
                raise e

        return cast(FuncType, wrapper)

    return decorator


def time_function[R](func: Callable[..., R]) -> Callable[..., R]:
    """
    Lightweight timing decorator with no cProfile overhead.

    Logs function execution time at INFO level. Useful for quick performance
    checks without the overhead of full profiling.

    Args:
        func: Function to time

    Returns:
        Decorated function that logs execution time

    Example:
        @time_function
        def process_batch(items):
            # ... processing logic
            pass

        # Output: "⏱️  process_batch completed in 2.34s"
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            logger.info(f"⏱️  {func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"⏱️  {func.__name__} failed after {duration:.2f}s: {e}")
            raise

    return wrapper


# === REPORT GENERATION ===


def _generate_txt_report(stats_path: Path, sort_by: str = "cumulative") -> None:
    """
    Generate human-readable text report from cProfile stats file.

    Creates a .txt file alongside the .stats file with top functions,
    call counts, and timing information.

    Args:
        stats_path: Path to .stats file
        sort_by: pstats sorting key
    """
    txt_path = stats_path.with_suffix(".txt")

    try:
        # Capture pstats output to string
        stream = io.StringIO()
        stats = pstats.Stats(str(stats_path), stream=stream)

        if _profile_config.strip_dirs:
            stats.strip_dirs()

        stats.sort_stats(sort_by)
        stats.print_stats(_profile_config.top_n_functions)

        # Write to file
        txt_path.write_text(stream.getvalue(), encoding="utf-8")
        logger.info(f"📄 Profile report saved: {txt_path}")

    except Exception as e:
        logger.error(f"Failed to generate text report: {e}")


def generate_report_from_stats(
    stats_file: Path, output_file: Path | None = None, sort_by: str = "cumulative"
) -> str:
    """
    Generate human-readable report from existing .stats file.

    Useful for post-processing profile data or generating custom reports.

    Args:
        stats_file: Path to existing .stats file
        output_file: Optional output path (if None, returns string)
        sort_by: pstats sorting key

    Returns:
        Report content as string

    Example:
        report = generate_report_from_stats(
            Path("Logs/profiles/action6_profile.stats"),
            sort_by="time"
        )
        print(report)
    """
    stream = io.StringIO()
    stats = pstats.Stats(str(stats_file), stream=stream)

    if _profile_config.strip_dirs:
        stats.strip_dirs()

    stats.sort_stats(sort_by)
    stats.print_stats(_profile_config.top_n_functions)

    report = stream.getvalue()

    if output_file:
        output_file.write_text(report, encoding="utf-8")
        logger.info(f"Report saved: {output_file}")

    return report


# === INTEGRATION HELPERS ===


def get_profile_output_path(action_name: str) -> Path:
    """
    Get standardized output path for action profiling.

    Args:
        action_name: Name of action (e.g., "action6_gather")

    Returns:
        Path for profile output

    Example:
        path = get_profile_output_path("action6_gather")
        # Returns: Logs/profiles/action6_gather_20231117_143022.stats
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{action_name}_{timestamp}.stats"
    return _profile_config.output_dir / filename


# === MODULE TESTS ===


def module_tests() -> bool:
    """Module-specific test implementation."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Performance Profiling Utilities", "performance_profiling.py")
    suite.start_suite()

    # Test 1: ProfileConfig dataclass
    def test_profile_config():
        config = ProfileConfig()
        assert config.enabled is False, "ProfileConfig should default to disabled"
        assert config.output_dir == Path("Logs/profiles"), "ProfileConfig should have correct default output_dir"

    suite.run_test("ProfileConfig Defaults", test_profile_config, "Test ProfileConfig dataclass default values")

    # Test 2: Configuration
    def test_configuration():
        configure_profiling(enabled=True, sort_by="time", top_n_functions=25)
        assert _profile_config.enabled is True, "configure_profiling should set enabled flag"
        assert _profile_config.sort_by == "time", "configure_profiling should set sort_by"
        assert _profile_config.top_n_functions == 25, "configure_profiling should set top_n_functions"
        # Reset for remaining tests
        configure_profiling(enabled=False)

    suite.run_test("Configuration Settings", test_configuration, "Test configure_profiling function")

    # Test 3: time_function decorator (no profiling overhead)
    def test_time_decorator():
        call_count = [0]  # Use list for mutable closure variable

        @time_function
        def test_func():
            call_count[0] += 1
            return 42

        result = test_func()
        assert result == 42, "time_function decorator should preserve return value"
        assert call_count[0] == 1, "time_function decorator should call function once"

    suite.run_test("Time Function Decorator", test_time_decorator, "Test lightweight timing decorator")

    # Test 4: profile_with_cprofile decorator (disabled)
    def test_profile_disabled():
        configure_profiling(enabled=False)

        @profile_with_cprofile(output_file="test_profile.stats")
        def profiled_func(x: int) -> int:
            return x * 2

        result = profiled_func(21)
        assert result == 42, "profile_with_cprofile (disabled) should preserve return value"

    suite.run_test(
        "Profile Decorator (Disabled)", test_profile_disabled, "Test cProfile decorator when profiling is disabled"
    )

    # Test 5: profile_with_cprofile decorator (enabled)
    def test_profile_enabled():
        configure_profiling(enabled=True)
        test_stats_path = _profile_config.output_dir / "test_profile.stats"

        # Clean up any existing test file
        if test_stats_path.exists():
            test_stats_path.unlink()

        @profile_with_cprofile(output_file="test_profile.stats")
        def profiled_func_enabled(x: int) -> int:
            # Add some measurable work
            total = 0
            for i in range(1000):
                total += i
            return x * 2 + total

        result = profiled_func_enabled(21)
        expected = 42 + 499500
        assert result == expected, (
            f"profile_with_cprofile (enabled) should preserve return value (got {result}, expected {expected})"
        )
        assert test_stats_path.exists(), "profile_with_cprofile should generate .stats file"

        # Test 6: Text report generation
        txt_path = test_stats_path.with_suffix(".txt")
        assert txt_path.exists(), "profile_with_cprofile should generate .txt report"

        # Clean up
        configure_profiling(enabled=False)
        if test_stats_path.exists():
            test_stats_path.unlink()
        if txt_path.exists():
            txt_path.unlink()

    suite.run_test(
        "Profile Decorator (Enabled)", test_profile_enabled, "Test cProfile decorator when profiling is enabled"
    )

    # Test 7: get_profile_output_path
    def test_output_path():
        path = get_profile_output_path("test_action")
        assert "test_action" in str(path), "get_profile_output_path should include action name"
        assert path.suffix == ".stats", "get_profile_output_path should use .stats extension"

    suite.run_test("Profile Output Path", test_output_path, "Test get_profile_output_path function")

    # Test 8: generate_report_from_stats
    def test_generate_report():
        # Create a test profile first
        configure_profiling(enabled=True)
        test_stats_path = _profile_config.output_dir / "test_report.stats"

        @profile_with_cprofile(output_file="test_report.stats")
        def dummy_func() -> int:
            return sum(range(100))

        dummy_func()

        if test_stats_path.exists():
            report = generate_report_from_stats(test_stats_path, sort_by="time")
            assert len(report) > 0, "generate_report_from_stats should produce non-empty report"
            assert "function calls" in report.lower() or "ncalls" in report.lower(), (
                "generate_report_from_stats should contain function call statistics"
            )

        # Clean up
        configure_profiling(enabled=False)
        if test_stats_path.exists():
            test_stats_path.unlink()
        txt_path = test_stats_path.with_suffix(".txt")
        if txt_path.exists():
            txt_path.unlink()

    suite.run_test("Generate Report from Stats", test_generate_report, "Test generate_report_from_stats function")

    return suite.finish_suite()


# === STANDARD TEST RUNNER ===

from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
