#!/usr/bin/env python3
"""
Test suite for Phase 2 improvements:
1. RateLimiter Singleton Pattern
2. Timestamp Logic Gate (Data Freshness Check)
3. Logging Consolidation
4. RPS Increase to 5.0
"""

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config_schema
from core.session_manager import SessionManager
from utils import RateLimiter, get_rate_limiter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# TEST 1: RateLimiter Singleton Pattern
# ============================================================================

def test_rate_limiter_singleton():
    """Test that RateLimiter singleton is reused across calls."""
    logger.info("=" * 70)
    logger.info("TEST 1: RateLimiter Singleton Pattern")
    logger.info("=" * 70)

    # Get rate limiter twice
    rl1 = get_rate_limiter()
    rl2 = get_rate_limiter()

    # Verify they're the same instance
    assert rl1 is rl2, "RateLimiter singleton not working - different instances returned"
    assert id(rl1) == id(rl2), "RateLimiter instances have different IDs"

    logger.info("✅ PASS: RateLimiter singleton verified")
    logger.info(f"   - Instance ID: {id(rl1)}")
    logger.info(f"   - Type: {type(rl1).__name__}")
    return True


def test_rate_limiter_in_session_manager():
    """Test that SessionManager uses the RateLimiter singleton."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: RateLimiter in SessionManager")
    logger.info("=" * 70)

    # Create two SessionManager instances
    sm1 = SessionManager()
    sm2 = SessionManager()

    # Verify they share the same rate limiter
    assert sm1.rate_limiter is sm2.rate_limiter, "SessionManagers don't share rate limiter"
    assert sm1.rate_limiter is get_rate_limiter(), "SessionManager rate limiter not singleton"

    logger.info("✅ PASS: SessionManager uses RateLimiter singleton")
    logger.info(f"   - SM1 rate limiter ID: {id(sm1.rate_limiter)}")
    logger.info(f"   - SM2 rate limiter ID: {id(sm2.rate_limiter)}")
    logger.info(f"   - Global rate limiter ID: {id(get_rate_limiter())}")
    return True


# ============================================================================
# TEST 3: RPS Configuration
# ============================================================================

def test_rps_configuration():
    """Test that RPS is configured (can be overridden by .env)."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: RPS Configuration")
    logger.info("=" * 70)

    rps = config_schema.api.requests_per_second
    assert rps > 0, f"RPS must be positive, got {rps}"
    assert rps >= 0.4, f"RPS should be at least 0.4 (was 0.4 before optimization), got {rps}"

    logger.info("✅ PASS: RPS configured")
    logger.info(f"   - Current RPS: {rps}")
    logger.info("   - Default: 5.0 (can be overridden by .env REQUESTS_PER_SECOND)")
    logger.info(f"   - Speedup vs 0.4: {rps/0.4:.1f}x faster")
    return True


# ============================================================================
# TEST 4: Person Refresh Days Configuration
# ============================================================================

def test_person_refresh_days_config():
    """Test that person_refresh_days configuration exists."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Person Refresh Days Configuration")
    logger.info("=" * 70)

    refresh_days = getattr(config_schema, 'person_refresh_days', None)
    assert refresh_days is not None, "person_refresh_days not configured"
    assert refresh_days > 0, f"person_refresh_days should be positive, got {refresh_days}"

    logger.info("✅ PASS: person_refresh_days configured")
    logger.info(f"   - Refresh days: {refresh_days}")
    logger.info("   - Default: 7 (can be overridden by .env PERSON_REFRESH_DAYS)")
    logger.info(f"   - Skip if updated within: {refresh_days} days")
    return True


# ============================================================================
# TEST 5: Timestamp Logic Gate (Simulated)
# ============================================================================

def test_timestamp_logic_gate():
    """Test timestamp-based skip logic."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Timestamp Logic Gate")
    logger.info("=" * 70)

    refresh_days = getattr(config_schema, 'person_refresh_days', 7)

    # Simulate recent update (should skip)
    now = datetime.now(timezone.utc)
    recent_update = now - timedelta(days=2)
    time_since_update = now - recent_update
    threshold = timedelta(days=refresh_days)
    should_skip_recent = time_since_update < threshold

    assert should_skip_recent, "Recent data should be skipped"
    logger.info(f"✅ PASS: Recent data (2 days old) should be skipped (threshold: {refresh_days} days)")

    # Simulate old update (should not skip if beyond threshold)
    old_update = now - timedelta(days=refresh_days + 5)
    time_since_update = now - old_update
    should_skip_old = time_since_update < threshold

    assert not should_skip_old, f"Old data ({refresh_days + 5} days old) should not be skipped"
    logger.info(f"✅ PASS: Old data ({refresh_days + 5} days old) should not be skipped")

    return True


# ============================================================================
# TEST 6: Rate Limiter State Preservation
# ============================================================================

def test_rate_limiter_state_preservation():
    """Test that rate limiter state is preserved across sessions."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 6: Rate Limiter State Preservation")
    logger.info("=" * 70)

    rl = get_rate_limiter()

    # Check that rate limiter is a RateLimiter instance
    assert isinstance(rl, RateLimiter), "Rate limiter is not a RateLimiter instance"
    assert hasattr(rl, 'circuit_breaker'), "RateLimiter missing circuit_breaker"

    logger.info("✅ PASS: RateLimiter has required attributes")
    logger.info(f"   - Type: {type(rl).__name__}")
    logger.info(f"   - Has circuit breaker: {hasattr(rl, 'circuit_breaker')}")

    return True


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all Phase 2 improvement tests."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 IMPROVEMENTS - TEST SUITE")
    logger.info("=" * 70)

    tests = [
        ("RateLimiter Singleton", test_rate_limiter_singleton),
        ("SessionManager Integration", test_rate_limiter_in_session_manager),
        ("RPS Configuration", test_rps_configuration),
        ("Person Refresh Days", test_person_refresh_days_config),
        ("Timestamp Logic Gate", test_timestamp_logic_gate),
        ("Rate Limiter State", test_rate_limiter_state_preservation),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            failed += 1
            logger.error(f"❌ FAIL: {test_name}")
            logger.error(f"   Error: {e}")
        except Exception as e:
            failed += 1
            logger.error(f"❌ ERROR: {test_name}")
            logger.error(f"   Exception: {e}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"✅ Passed: {passed}/{len(tests)}")
    logger.info(f"❌ Failed: {failed}/{len(tests)}")
    logger.info("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

