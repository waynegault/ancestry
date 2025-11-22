import os
import sys
from datetime import datetime
from typing import Any

from standard_imports import setup_module

logger = setup_module(globals(), __name__)


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


def warn_if_unsafe_profile(speed_profile: str, allow_unsafe: bool, suppress_warnings: bool) -> None:
    """Emit warning when unsafe API profiles are active."""

    if suppress_warnings:
        return

    if not (allow_unsafe or speed_profile in {"max", "aggressive", "experimental"}):
        return

    # "baseline" profile is now considered a safe baseline (validated 2025-11-22)
    if speed_profile == "baseline":
        return

    profile_label = speed_profile or "custom"
    logger.warning(
        "  Unsafe API speed profile '%s' active; safety clamps relaxed. Monitor for 429 errors.",
        profile_label,
    )


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


def log_rate_limiter_summary(config: Any, allow_unsafe: bool, speed_profile: str) -> None:
    """Log the adaptive rate limiter plan without instantiating it early."""

    try:
        from rate_limiter import get_persisted_rate_state
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
    safe_rps = getattr(config.api, "requests_per_second", 0.3) or 0.3
    desired_rate = getattr(config.api, "token_bucket_fill_rate", None) or safe_rps
    allow_aggressive = allow_unsafe or speed_profile in {"max", "aggressive", "experimental"}
    min_fill_rate = max(0.05, safe_rps * 0.25)
    max_fill_rate = desired_rate if allow_aggressive else safe_rps
    max_fill_rate = max(max_fill_rate, min_fill_rate)
    bucket_capacity = getattr(config.api, "token_bucket_capacity", 10.0)

    logger.info(
        "  Rate Limiter (planned): target=%.3f req/s | success_threshold=%d | bounds=%.3f-%.3f | capacity=%.1f",
        desired_rate,
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
    speed_profile = str(getattr(config.api, "speed_profile", "safe")).lower()
    allow_unsafe = bool(getattr(config.api, "allow_unsafe_rate_limit", False))
    log_basic_configuration_values(config)
    suppress_warnings = should_suppress_config_warnings()
    warn_if_unsafe_profile(speed_profile, allow_unsafe, suppress_warnings)
    log_rate_limiter_summary(config, allow_unsafe, speed_profile)
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
