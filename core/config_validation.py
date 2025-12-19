import builtins
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

import logging

from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


def check_processing_limits(config: Any) -> None:
    """Check essential processing limits and log warnings."""
    # Note: MAX_PAGES=0 is valid (means unlimited), so no warning needed
    if config.batch_size <= 0:
        logger.warning("BATCH_SIZE not set or invalid - actions may use large batches")
    # Note: MAX_PRODUCTIVE_TO_PROCESS=0 and MAX_INBOX=0 are valid (means unlimited)


def check_rate_limiting_settings(config: Any) -> None:
    """Check rate limiting settings and log warnings."""
    # Note: Rate limiting settings are user preferences - no warnings needed
    # Users can adjust REQUESTS_PER_SECOND, INITIAL_DELAY, and BACKOFF_FACTOR as needed
    _ = config  # Parameter kept for API compatibility but not currently used


def log_basic_configuration_values(config: Any) -> None:
    """Log the primary configuration values shown at startup."""

    logger.info(f"  MAX_PAGES: {config.api.max_pages}")
    logger.info(f"  BATCH_SIZE: {config.batch_size}")
    logger.info(f"  MAX_PRODUCTIVE_TO_PROCESS: {config.max_productive_to_process}")
    logger.info(f"  MAX_INBOX: {config.max_inbox}")
    logger.info(f"  PARALLEL_WORKERS: {config.parallel_workers}")
    logger.info(f"  Rate Limiting - RPS: {config.api.requests_per_second}, Delay: {config.api.initial_delay}s")

    match_throughput = getattr(config.api, "target_match_throughput", 0.0)
    if match_throughput > 0:
        logger.info("  Match Throughput Target: %.2f match/s", match_throughput)
    else:
        logger.info("  Match Throughput Target: disabled")

    logger.info(
        "  Max Pacing Delay/Page: %.2fs",
        getattr(config.api, "max_throughput_catchup_delay", 0.0),
    )


def should_suppress_config_warnings() -> bool:
    """Return True when runtime context indicates configuration warnings should be muted."""

    if os.environ.get("PYTEST_CURRENT_TEST") is not None:
        return True
    return any("test" in arg.lower() for arg in sys.argv)


def log_persisted_rate_state(persisted_state: dict[str, Any]) -> None:
    """Log persisted rate limiter metadata from previous runs."""

    saved_rate = persisted_state.get("fill_rate")
    saved_requests = persisted_state.get("total_requests", "n/a")
    timestamp_value = persisted_state.get("timestamp")
    if isinstance(timestamp_value, (int, float)):
        timestamp_str = datetime.fromtimestamp(timestamp_value).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timestamp_str = "unknown"

    if isinstance(saved_rate, (int, float)):
        logger.info(
            "    Last run: %.3f req/s | saved at %s | total_requests=%s",
            float(saved_rate),
            timestamp_str,
            saved_requests,
        )


def log_rate_limiter_summary(config: Any) -> None:
    """Log the adaptive rate limiter plan without instantiating it early."""

    try:
        from core.rate_limiter import get_persisted_rate_state
    except ImportError:
        logger.debug("Rate limiter module unavailable during configuration summary")
        return

    persisted_state = get_persisted_rate_state()
    batch_threshold = max(getattr(config, "batch_size", 50) or 50, 1)
    configured_threshold = getattr(config.api, "token_bucket_success_threshold", None)
    if isinstance(configured_threshold, int) and configured_threshold > 0:
        success_threshold = configured_threshold
    else:
        success_threshold = max(batch_threshold, 10)
    configured_rps = getattr(config.api, "requests_per_second", 0.3) or 0.3
    min_fill_rate = getattr(config.api, "rate_limiter_min_rate", 0.5) or 0.5
    max_fill_rate = getattr(config.api, "rate_limiter_max_rate", configured_rps)
    max_fill_rate = max(max_fill_rate, min_fill_rate)
    bucket_capacity = getattr(config.api, "token_bucket_capacity", 10.0)

    logger.info(
        "  Rate Limiter (planned): target=%.3f req/s | success_threshold=%d | bounds=%.3f-%.3f | capacity=%.1f",
        configured_rps,
        success_threshold,
        min_fill_rate,
        max_fill_rate,
        bucket_capacity,
    )

    if persisted_state:
        log_persisted_rate_state(persisted_state)


def log_configuration_summary(config: Any) -> None:
    """Log current configuration for transparency."""
    # Clear screen at startup (temporarily disabled for debugging global session complaints)
    # import os
    # os.system('cls' if os.name == 'nt' else 'clear')

    print(" CONFIG ".center(80, "="))
    log_basic_configuration_values(config)
    log_rate_limiter_summary(config)
    print("")  # Blank line after configuration


def load_and_validate_config_schema() -> Any | None:
    """Load and validate configuration schema."""
    try:
        from config import config_schema

        logger.debug("Configuration loaded successfully")
        return config_schema
    except ImportError as e:
        logger.error(f"Could not import config_schema from config package: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        return None


def validate_action_config() -> bool:
    """
    Validate that all actions respect .env configuration limits.
    Prevents Action 6-style failures by ensuring conservative settings are applied.
    """
    try:
        # Load and validate configuration
        config = load_and_validate_config_schema()
        if config is None:
            return False

        # Check processing limits
        check_processing_limits(config)

        # Check rate limiting settings
        check_rate_limiting_settings(config)

        # Log configuration summary
        log_configuration_summary(config)

        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def print_config_error_message() -> None:
    """Print detailed configuration error message and exit."""
    logger.critical("Configuration validation failed - unable to proceed")
    print("\n❌ CONFIGURATION ERROR:")
    print("   Critical configuration validation failed.")
    print("   This usually means missing credentials or configuration files.")
    print("")
    print("💡 SOLUTION:")
    print("   1. Copy .env.example to .env and add your credentials")
    print("   2. Ensure all required environment variables are set")

    print("\n📚 For detailed instructions:")
    print("   See ENV_IMPORT_GUIDE.md or readme.md")

    print("\nExiting application...")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Module Tests
# ---------------------------------------------------------------------------


def _make_base_config() -> SimpleNamespace:
    api_defaults = SimpleNamespace(
        max_pages=1,
        requests_per_second=0.3,
        initial_delay=1.0,
        target_match_throughput=0.0,
        max_throughput_catchup_delay=0.0,
        token_bucket_success_threshold=None,
        token_bucket_capacity=10.0,
        rate_limiter_max_rate=5.0,
    )
    return SimpleNamespace(
        batch_size=50,
        max_productive_to_process=25,
        max_inbox=100,
        parallel_workers=1,
        api=api_defaults,
    )


def _test_check_processing_limits_warns_on_invalid_batch() -> bool:
    config = SimpleNamespace(batch_size=0)
    with patch.object(logger, "warning") as mock_warn:
        check_processing_limits(config)
    mock_warn.assert_called_once()
    return True


def _test_check_processing_limits_allows_valid_batches() -> bool:
    config = SimpleNamespace(batch_size=10)
    with patch.object(logger, "warning") as mock_warn:
        check_processing_limits(config)
    mock_warn.assert_not_called()
    return True


def _test_log_persisted_rate_state_formats_message() -> bool:
    persisted = {"fill_rate": 0.45, "total_requests": 500, "timestamp": 1_725_000_000}
    with patch.object(logger, "info") as mock_info:
        log_persisted_rate_state(persisted)
    mock_info.assert_called_once()
    args, _ = mock_info.call_args
    assert abs(float(args[1]) - 0.45) < 1e-6, "Formatted rate should be forwarded to logger"
    return True


def _test_should_suppress_config_warnings_detects_pytest_env() -> bool:
    with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "core/tests"}, clear=False):
        assert should_suppress_config_warnings() is True
    return True


def _test_should_suppress_config_warnings_detects_cli_flag() -> bool:
    original_argv = sys.argv
    sys.argv = ["prog", "--run-tests"]
    try:
        assert should_suppress_config_warnings() is True
    finally:
        sys.argv = original_argv
    return True


def _test_validate_action_config_handles_missing_schema() -> bool:
    module_ref = sys.modules[__name__]
    with patch.object(module_ref, "load_and_validate_config_schema", return_value=None):
        assert validate_action_config() is False
    return True


def _test_validate_action_config_success_path() -> bool:
    config = _make_base_config()
    module_ref = sys.modules[__name__]

    with (
        patch.object(module_ref, "load_and_validate_config_schema", return_value=config) as mock_loader,
        patch.object(module_ref, "check_processing_limits") as mock_limits,
        patch.object(module_ref, "check_rate_limiting_settings") as mock_rate,
        patch.object(module_ref, "log_configuration_summary") as mock_summary,
    ):
        result = validate_action_config()

    assert result is True
    mock_loader.assert_called_once()
    mock_limits.assert_called_once_with(config)
    mock_rate.assert_called_once_with(config)
    mock_summary.assert_called_once_with(config)
    return True


def _test_log_rate_limiter_summary_handles_import_error() -> bool:
    config = _make_base_config()

    original_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any):
        if name == "rate_limiter":
            raise ImportError("rate limiter unavailable")
        return original_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=_fake_import):
        log_rate_limiter_summary(config)
    # Should not raise
    return True


def module_tests() -> bool:
    suite = TestSuite("core.config_validation", "core/config_validation.py")

    suite.run_test(
        "Processing limits warning",
        _test_check_processing_limits_warns_on_invalid_batch,
        "Ensures invalid batch sizes emit a warning.",
    )

    suite.run_test(
        "Processing limits valid",
        _test_check_processing_limits_allows_valid_batches,
        "Ensures valid batch sizes do not warn.",
    )

    suite.run_test(
        "Persisted rate state logging",
        _test_log_persisted_rate_state_formats_message,
        "Ensures persisted rate metadata is logged when available.",
    )

    suite.run_test(
        "Suppress detection (env)",
        _test_should_suppress_config_warnings_detects_pytest_env,
        "Ensures env var causes warnings to suppress.",
    )

    suite.run_test(
        "Suppress detection (argv)",
        _test_should_suppress_config_warnings_detects_cli_flag,
        "Ensures CLI arguments with 'test' suppress warnings.",
    )

    suite.run_test(
        "validate_action_config missing schema",
        _test_validate_action_config_handles_missing_schema,
        "Ensures validation fails when schema cannot load.",
    )

    suite.run_test(
        "validate_action_config success",
        _test_validate_action_config_success_path,
        "Ensures validation orchestrates helper calls on success.",
    )

    suite.run_test(
        "Rate limiter summary import handling",
        _test_log_rate_limiter_summary_handles_import_error,
        "Ensures rate limiter summary tolerates missing module.",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = module_tests()
    sys.exit(0 if success else 1)
