#!/usr/bin/env python3

"""
constants.py - Centralized Application Constants

Single source of truth for magic numbers, strings, and configuration values
used throughout the codebase. This reduces duplication and makes it easier
to update values consistently.

Usage:
    from core.constants import (
        DB_FILE_DEFAULT,
        COOKIE_FILE_NAME,
        DEFAULT_RATE_LIMIT,
        ...
    )
"""

from pathlib import Path

# ============================================================================
# FILE & PATH CONSTANTS
# ============================================================================

# Database and session files
DB_FILE_DEFAULT = "Data/ancestry.db"
DB_BACKUP_NAME = "ancestry_backup.db"
COOKIE_FILE_NAME = "ancestry_cookies.json"
COOKIE_PICKLE_NAME = "ancestry_cookies.pkl"
SESSION_STATE_FILE = "ancestry_session.json"

# GEDCOM files
GEDCOM_FILE_DEFAULT = "Data/family_tree.ged"

# Cache and data directories
CACHE_DIR_DEFAULT = "Cache"
DATA_DIR_DEFAULT = "Data"
LOG_DIR_DEFAULT = "Logs"
CHECKPOINT_FILE_DEFAULT = "Cache/action6_checkpoint.json"

# ============================================================================
# RATE LIMITING & PERFORMANCE
# ============================================================================

# Default rate limit (requests per second)
DEFAULT_RATE_LIMIT = 2.0

# Rate limiter bounds
RATE_LIMITER_MIN_RATE = 0.1
RATE_LIMITER_MAX_RATE = 20.0

# Adaptive rate limiter factors
RATE_429_BACKOFF_FACTOR = 0.80  # 20% reduction on 429 error
RATE_SUCCESS_FACTOR = 1.05  # 5% increase on success

# Token bucket settings
TOKEN_BUCKET_CAPACITY = 20.0

# Concurrency settings (1 = sequential, safest)
DEFAULT_MAX_CONCURRENCY = 1

# Target throughput
DEFAULT_TARGET_THROUGHPUT = 1.0  # matches per second

# ============================================================================
# TIMEOUTS
# ============================================================================

# Request timeouts (seconds)
DEFAULT_REQUEST_TIMEOUT = 60
API_TIMEOUT = 30
PAGE_LOAD_TIMEOUT = 45
IMPLICIT_WAIT = 10
EXPLICIT_WAIT = 20
TWO_FA_TIMEOUT = 180  # 3 minutes for 2FA code entry

# Session timeouts
SESSION_TIMEOUT_MINUTES = 60

# Startup timeouts
LM_STUDIO_STARTUP_TIMEOUT = 60

# ============================================================================
# RETRY & BACKOFF SETTINGS
# ============================================================================

# Retry settings
DEFAULT_MAX_RETRIES = 5
RETRY_BACKOFF_FACTOR = 6.0
CHROME_MAX_RETRIES = 3
CHROME_RETRY_DELAY = 5

# Retryable HTTP status codes
RETRYABLE_STATUS_CODES = [429, 500, 502, 503, 504]

# Circuit breaker settings
CIRCUIT_BREAKER_THRESHOLD = 5  # failures before tripping

# ============================================================================
# DNA MATCH THRESHOLDS
# ============================================================================

# DNA match probability threshold (centimorgans)
DNA_MATCH_PROBABILITY_THRESHOLD_CM = 10

# Relationship probability categories (cM ranges)
CM_PARENT_CHILD = (3300, 3700)
CM_FULL_SIBLING = (2200, 3400)
CM_HALF_SIBLING = (1160, 2430)
CM_GRANDPARENT = (1150, 2300)
CM_AUNT_UNCLE = (1100, 2500)
CM_FIRST_COUSIN = (396, 1397)

# ============================================================================
# AI PROVIDER DEFAULTS
# ============================================================================

# Default AI provider
DEFAULT_AI_PROVIDER = "gemini"

# Model defaults
GEMINI_MODEL_DEFAULT = "gemini-1.5-flash-latest"
DEEPSEEK_MODEL_DEFAULT = "deepseek-chat"
MOONSHOT_MODEL_DEFAULT = "kimi-k2-thinking"
XAI_MODEL_DEFAULT = "grok-4-fast-non-reasoning"
TETRATE_MODEL_DEFAULT = "xai/grok-code-fast-1"
LOCAL_LLM_MODEL_DEFAULT = "qwen2.5-14b-instruct"
INCEPTION_MODEL_DEFAULT = "mercury"

# API base URLs
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
XAI_API_HOST = "api.x.ai"
TETRATE_BASE_URL = "https://api.router.tetrate.ai/v1"
LOCAL_LLM_BASE_URL = "http://localhost:1234/v1"
INCEPTION_BASE_URL = "https://api.inceptionlabs.ai/v1"

# ============================================================================
# LOGGING
# ============================================================================

# Default log settings
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = "Logs/app.log"
DEFAULT_MAX_LOG_SIZE_MB = 10
DEFAULT_LOG_BACKUP_COUNT = 5

# Log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# DATABASE SETTINGS
# ============================================================================

# SQLite settings
DEFAULT_JOURNAL_MODE = "WAL"
DEFAULT_SYNCHRONOUS = "NORMAL"
DEFAULT_CACHE_SIZE_MB = 256
DEFAULT_PAGE_SIZE = 4096
DEFAULT_POOL_SIZE = 10
DEFAULT_MAX_OVERFLOW = 5
DEFAULT_POOL_TIMEOUT = 30

# Backup settings
DEFAULT_BACKUP_INTERVAL_HOURS = 24
DEFAULT_MAX_BACKUPS = 7

# ============================================================================
# CHROME/SELENIUM
# ============================================================================

# Default Chrome settings
DEFAULT_DEBUG_PORT = 9222
DEFAULT_WINDOW_SIZE = "1920,1080"
DEFAULT_PROFILE_DIR = "Default"

# User agent string
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# ============================================================================
# MONITORING
# ============================================================================

# Prometheus settings
PROMETHEUS_DEFAULT_PORT = 9001
PROMETHEUS_DEFAULT_HOST = "0.0.0.0"
PROMETHEUS_NAMESPACE = "ancestry"

# Grafana settings
GRAFANA_DEFAULT_URL = "http://localhost:3000"
GRAFANA_DEFAULT_PORT = 3000
GRAFANA_DASHBOARD_PORT = 3300  # Mapped from 3000 in Docker

# ============================================================================
# APPLICATION MODES
# ============================================================================

# Environment types
ENV_DEVELOPMENT = "development"
ENV_TESTING = "testing"
ENV_PRODUCTION = "production"

# App modes
APP_MODE_DEVELOPMENT = "development"
APP_MODE_PRODUCTION = "production"

# Test modes
TEST_MODE_MOCK = "mock"
TEST_MODE_REAL = "real"
TEST_MODE_INTEGRATION = "integration"

# ============================================================================
# SAFETY & GUARDRAILS
# ============================================================================

# Auto-approval thresholds
MIN_HUMAN_REVIEWS_FOR_AUTO_APPROVAL = 100
MIN_ACCEPTANCE_RATE_FOR_AUTO_APPROVAL = 0.95

# Quality scoring thresholds
QUALITY_SCORE_EXCELLENT_MIN = 85
QUALITY_SCORE_GOOD_MIN = 70
QUALITY_SCORE_ACCEPTABLE_MIN = 50

# ============================================================================
# MODULE TESTS
# ============================================================================


def constants_module_tests() -> bool:
    """
    Run basic tests for constants module.

    Returns:
        True if all tests pass
    """
    print("🔧 Testing Constants Module...")
    print()

    success = True

    # Test 1: All path constants are Path objects or strings
    print("Test 1: Validate path constants")
    path_constants = [
        "DB_FILE_DEFAULT",
        "GEDCOM_FILE_DEFAULT",
        "CACHE_DIR_DEFAULT",
        "DATA_DIR_DEFAULT",
        "LOG_DIR_DEFAULT",
    ]

    for const_name in path_constants:
        value = globals().get(const_name)
        if value and not isinstance(value, (str, Path)):
            print(f"❌ {const_name} should be str or Path, got {type(value)}")
            success = False
        else:
            print(f"✅ {const_name} = {value}")

    print("✅ PASSED: Path constants valid\n")

    # Test 2: Numeric constants are positive
    print("Test 2: Validate numeric constants")
    numeric_constants = [
        "DEFAULT_RATE_LIMIT",
        "DEFAULT_REQUEST_TIMEOUT",
        "DEFAULT_MAX_RETRIES",
        "CIRCUIT_BREAKER_THRESHOLD",
    ]

    for const_name in numeric_constants:
        value = globals().get(const_name)
        if value and isinstance(value, (int, float)) and value > 0:
            print(f"✅ {const_name} = {value}")
        else:
            print(f"❌ {const_name} should be positive number, got {value}")
            success = False

    print("✅ PASSED: Numeric constants valid\n")

    if success:
        print("🎉 All constants tests PASSED")
    else:
        print("❌ Some constants tests FAILED")

    return success


# Use centralized test runner utility from test_utilities
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(constants_module_tests)


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
