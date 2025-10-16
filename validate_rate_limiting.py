#!/usr/bin/env python3
"""
Quick validation script to verify the optimized rate limiting configuration.
Tests that all new parameters are loaded correctly.
"""

import sys
from pathlib import Path
from typing import Any

# Add parent directory to path
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import config_schema


def _get_expected_values() -> dict[str, Any]:
    """Get expected configuration values."""
    return {
        'thread_pool_workers': 2,
        'requests_per_second': 0.9,
        'initial_delay': 1.0,
        'max_delay': 15.0,
        'backoff_factor': 1.5,
        'decrease_factor': 0.95,
        'token_bucket_capacity': 10.0,
        'token_bucket_fill_rate': 2.0,
    }


def _validate_config_values(expected: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Validate configuration values and return errors and warnings."""
    errors = []
    warnings = []

    print("📊 Configuration Values:")
    print("-" * 60)

    for key, expected_value in expected.items():
        actual_value = getattr(config_schema.api, key, None)
        status = "✅" if actual_value == expected_value else "⚠️"

        print(f"{status} {key:.<40} {actual_value}")

        if actual_value != expected_value:
            if actual_value is None:
                errors.append(f"Missing config: {key}")
            else:
                warnings.append(f"Unexpected value for {key}: {actual_value} (expected: {expected_value})")

    print("-" * 60)
    return errors, warnings


def _calculate_performance_metrics() -> None:
    """Calculate and display performance metrics."""
    print("\n📈 Expected Performance:")
    print("-" * 60)

    workers = config_schema.api.thread_pool_workers
    rps = config_schema.api.requests_per_second
    per_worker_rps = rps / workers if workers > 0 else 0
    delay_between_requests = 1 / per_worker_rps if per_worker_rps > 0 else 0

    print(f"Total Workers:            {workers}")
    print(f"Total RPS:                {rps:.2f} requests/second")
    print(f"Per-Worker RPS:           {per_worker_rps:.2f} requests/second")
    print(f"Delay Between Requests:   {delay_between_requests:.2f} seconds")
    print(f"Expected Time (20 items): {20 / rps:.1f} seconds")
    print(f"Expected Time (40 items): {40 / rps:.1f} seconds")

    print("-" * 60)


def _perform_safety_checks() -> list[str]:
    """Perform safety checks and return warnings."""
    warnings = []

    print("\n🛡️ Safety Checks:")
    print("-" * 60)

    # Check 1: RPS per worker should be <= 0.5 for safety
    workers = getattr(config_schema.api, 'thread_pool_workers', 2)
    per_worker_rps = config_schema.api.requests_per_second / workers
    if per_worker_rps > 0.5:
        print("⚠️  Warning: Per-worker RPS > 0.5 may risk rate limiting")
        warnings.append(f"High per-worker RPS: {per_worker_rps:.2f}")
    else:
        print(f"✅ Per-worker RPS ({per_worker_rps:.2f}) is safe (≤ 0.5)")

    # Check 2: Initial delay should be reasonable
    if config_schema.api.initial_delay < 0.5:
        print(f"⚠️  Warning: Initial delay ({config_schema.api.initial_delay}s) may be too aggressive")
        warnings.append(f"Low initial delay: {config_schema.api.initial_delay}s")
    else:
        print(f"✅ Initial delay ({config_schema.api.initial_delay}s) is reasonable")

    # Check 3: Max delay should allow recovery
    if config_schema.api.max_delay < 10:
        print(f"⚠️  Warning: Max delay ({config_schema.api.max_delay}s) may be too short")
        warnings.append(f"Low max delay: {config_schema.api.max_delay}s")
    else:
        print(f"✅ Max delay ({config_schema.api.max_delay}s) allows proper recovery")

    # Check 4: Token bucket should be configured
    if config_schema.api.token_bucket_capacity <= 0:
        print("❌ Error: Token bucket capacity must be positive")
        return ["Invalid token bucket capacity"]
    print(f"✅ Token bucket capacity ({config_schema.api.token_bucket_capacity}) is configured")

    print("-" * 60)
    return warnings


def _print_validation_summary(errors: list[str], warnings: list[str]) -> bool:
    """Print validation summary and return success status."""
    print("\n📝 Validation Summary:")
    print("=" * 60)

    if errors:
        print(f"\n❌ {len(errors)} Critical Error(s):")
        for error in errors:
            print(f"   • {error}")

    if warnings:
        print(f"\n⚠️  {len(warnings)} Warning(s):")
        for warning in warnings:
            print(f"   • {warning}")

    if not errors and not warnings:
        print("✅ All validations passed! Configuration is optimal.")
        print("\n🚀 Ready to run with 2 workers and improved rate limiting!")
        return True
    if not errors:
        print("\n✅ Configuration loaded successfully with minor warnings.")
        print("   Review warnings above, but system should run properly.")
        return True
    print("\n❌ Configuration has critical errors. Please fix before running.")
    return False


def validate_rate_limiting_config() -> bool:
    """Validate rate limiting configuration."""
    print("🔍 Validating Rate Limiting Configuration...\n")

    expected = _get_expected_values()
    errors, warnings = _validate_config_values(expected)
    _calculate_performance_metrics()
    safety_warnings = _perform_safety_checks()
    warnings.extend(safety_warnings)
    return _print_validation_summary(errors, warnings)


# ==============================================
# Comprehensive Test Suite
# ==============================================

def _test_get_expected_values() -> bool:
    """Test that expected values are properly defined."""
    expected = _get_expected_values()
    assert isinstance(expected, dict), "Should return dict"
    assert "thread_pool_workers" in expected, "Should have thread_pool_workers"
    assert "requests_per_second" in expected, "Should have requests_per_second"
    assert "initial_delay" in expected, "Should have initial_delay"
    assert "max_delay" in expected, "Should have max_delay"
    assert "backoff_factor" in expected, "Should have backoff_factor"
    assert "token_bucket_capacity" in expected, "Should have token_bucket_capacity"
    return True


def _test_validate_config_values_returns_tuple() -> bool:
    """Test that validate_config_values returns tuple of lists."""
    expected = _get_expected_values()
    errors, warnings = _validate_config_values(expected)
    assert isinstance(errors, list), "Should return list of errors"
    assert isinstance(warnings, list), "Should return list of warnings"
    return True


def _test_calculate_performance_metrics() -> bool:
    """Test that performance metrics calculation works."""
    try:
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            _calculate_performance_metrics()

        output = f.getvalue()
        assert "Workers" in output or "RPS" in output, "Should calculate metrics"
        return True
    except Exception:
        return False


def _test_perform_safety_checks_returns_list() -> bool:
    """Test that safety checks return list of warnings."""
    try:
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            warnings = _perform_safety_checks()

        assert isinstance(warnings, list), "Should return list"
        return True
    except Exception:
        return False


def _test_print_validation_summary_returns_bool() -> bool:
    """Test that validation summary returns boolean."""
    try:
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = _print_validation_summary([], [])

        assert isinstance(result, bool), "Should return bool"
        return True
    except Exception:
        return False


def _test_validate_rate_limiting_config_returns_bool() -> bool:
    """Test that main validation function returns boolean."""
    try:
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = validate_rate_limiting_config()

        assert isinstance(result, bool), "Should return bool"
        return True
    except Exception:
        return False


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for validate_rate_limiting.py.
    Tests rate limiting configuration validation functions.
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "Rate Limiting Configuration Validator",
            "validate_rate_limiting.py"
        )
        suite.start_suite()

        suite.run_test(
            "Expected Values Definition",
            _test_get_expected_values,
            "Expected values are properly defined",
            "Test expected configuration values",
            "Test rate limiting configuration defaults",
        )

        suite.run_test(
            "Config Validation Returns Tuple",
            _test_validate_config_values_returns_tuple,
            "Config validation returns tuple of lists",
            "Test validation function return type",
            "Test error and warning collection",
        )

        suite.run_test(
            "Performance Metrics Calculation",
            _test_calculate_performance_metrics,
            "Performance metrics are calculated",
            "Test metrics calculation",
            "Test performance analysis",
        )

        suite.run_test(
            "Safety Checks Return List",
            _test_perform_safety_checks_returns_list,
            "Safety checks return list of warnings",
            "Test safety check function",
            "Test configuration safety validation",
        )

        suite.run_test(
            "Validation Summary Returns Bool",
            _test_print_validation_summary_returns_bool,
            "Validation summary returns boolean",
            "Test summary function",
            "Test validation result reporting",
        )

        suite.run_test(
            "Main Validation Function",
            _test_validate_rate_limiting_config_returns_bool,
            "Main validation function returns boolean",
            "Test main validation function",
            "Test complete validation workflow",
        )

        return suite.finish_suite()


if __name__ == "__main__":
    success = run_comprehensive_tests()

    if success:
        print("\n" + "=" * 60)
        print("RATE LIMITING CONFIGURATION VALIDATOR")
        print("=" * 60)
        print()

        try:
            validation_success = validate_rate_limiting_config()
            sys.exit(0 if validation_success else 1)
        except Exception as e:
            print(f"\n❌ Validation failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        sys.exit(1)
