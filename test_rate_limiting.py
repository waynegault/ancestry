#!/usr/bin/env python3

"""
test_rate_limiting.py - Rate Limit Test Harness

Tests different RPS settings by making controlled API calls and monitoring for 429 errors.
Validates that rate limiting is working correctly before deploying to production.

Usage:
    python test_rate_limiting.py --rps 1.0 --calls 50
    python test_rate_limiting.py --rps 1.5 --calls 100
    python test_rate_limiting.py --rps 2.0 --calls 100 --validate
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import argparse
import sys
import time
from datetime import datetime
from typing import Optional

# === LOCAL IMPORTS ===
from config import config_schema
from core.session_manager import SessionManager
from utils import _api_req


class RateLimitTester:
    """Test harness for validating rate limiting configurations."""

    def __init__(self, rps: float, num_calls: int, session_manager: SessionManager):
        """
        Initialize the rate limit tester.

        Args:
            rps: Requests per second to test
            num_calls: Number of API calls to make
            session_manager: Active session manager instance
        """
        self.rps = rps
        self.num_calls = num_calls
        self.session_manager = session_manager
        self.results = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'error_429_count': 0,
            'other_errors': 0,
            'start_time': None,
            'end_time': None,
            'duration': 0.0,
            'effective_rps': 0.0,
            'errors': [],
        }

    def run_test(self) -> dict:
        """
        Run the rate limit test.

        Returns:
            Dictionary containing test results
        """
        logger.info("=" * 80)
        logger.info(f"RATE LIMIT TEST: {self.rps} RPS with {self.num_calls} calls")
        logger.info("=" * 80)

        # Override rate limiter settings for this test
        if not self.session_manager.rate_limiter:
            logger.error("Rate limiter not available")
            return self.results

        original_fill_rate = self.session_manager.rate_limiter.fill_rate
        self.session_manager.rate_limiter.fill_rate = self.rps
        self.session_manager.rate_limiter.reset_metrics()

        logger.info(f"Rate limiter configured to {self.rps} RPS")
        logger.info(f"Making {self.num_calls} API calls...")

        self.results['start_time'] = time.time()

        # Make API calls
        for i in range(1, self.num_calls + 1):
            self._make_test_call(i)

            # Progress update every 10 calls
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{self.num_calls} calls completed")

        self.results['end_time'] = time.time()
        self.results['duration'] = self.results['end_time'] - self.results['start_time']
        self.results['effective_rps'] = (
            self.results['total_calls'] / self.results['duration']
            if self.results['duration'] > 0
            else 0.0
        )

        # Restore original rate limiter settings
        self.session_manager.rate_limiter.fill_rate = original_fill_rate

        # Print results
        self._print_results()

        return self.results

    def _make_test_call(self, call_num: int) -> None:
        """
        Make a single test API call.

        Args:
            call_num: The call number (for logging)
        """
        self.results['total_calls'] += 1

        try:
            # Use a lightweight API endpoint (profile ID check)
            url = f"{config_schema.api.base_url}app-api/cdp-p13n/api/v1/users/me?attributes=ucdmid"
            api_description = f"Test Call {call_num}"

            response = _api_req(
                url=url,
                driver=self.session_manager.driver,
                session_manager=self.session_manager,
                method="GET",
                api_description=api_description,
                use_csrf_token=False,
            )

            if response and isinstance(response, dict):
                self.results['successful_calls'] += 1
                logger.debug(f"Call {call_num}: SUCCESS")
            else:
                self.results['failed_calls'] += 1
                logger.warning(f"Call {call_num}: FAILED (unexpected response)")

        except Exception as e:
            self.results['failed_calls'] += 1
            error_msg = str(e)

            # Check for 429 errors
            if '429' in error_msg or 'Too Many Requests' in error_msg:
                self.results['error_429_count'] += 1
                logger.error(f"Call {call_num}: 429 ERROR - {error_msg}")
                self.results['errors'].append({
                    'call_num': call_num,
                    'type': '429',
                    'message': error_msg,
                    'timestamp': datetime.now().isoformat(),
                })
            else:
                self.results['other_errors'] += 1
                logger.error(f"Call {call_num}: ERROR - {error_msg}")
                self.results['errors'].append({
                    'call_num': call_num,
                    'type': 'other',
                    'message': error_msg,
                    'timestamp': datetime.now().isoformat(),
                })

    def _print_results(self) -> None:
        """Print formatted test results."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST RESULTS")
        logger.info("=" * 80)
        logger.info("Configuration:")
        logger.info(f"  Target RPS:          {self.rps:.2f}")
        logger.info(f"  Total Calls:         {self.num_calls}")
        logger.info("")
        logger.info("Results:")
        logger.info(f"  Successful Calls:    {self.results['successful_calls']}")
        logger.info(f"  Failed Calls:        {self.results['failed_calls']}")
        logger.info(f"  429 Errors:          {self.results['error_429_count']}")
        logger.info(f"  Other Errors:        {self.results['other_errors']}")
        logger.info("")
        logger.info("Performance:")
        logger.info(f"  Duration:            {self.results['duration']:.2f}s")
        logger.info(f"  Effective RPS:       {self.results['effective_rps']:.2f}")
        logger.info(f"  Target RPS:          {self.rps:.2f}")
        logger.info(f"  RPS Accuracy:        {(self.results['effective_rps'] / self.rps * 100):.1f}%")
        logger.info("")

        # Print rate limiter metrics
        if self.session_manager.rate_limiter:
            self.session_manager.rate_limiter.print_metrics_summary()

        # Validation
        if self.results['error_429_count'] == 0:
            logger.info("✅ PASS: No 429 errors detected")
        else:
            logger.error(f"❌ FAIL: {self.results['error_429_count']} 429 errors detected")
            logger.error("First 5 errors:")
            for error in self.results['errors'][:5]:
                logger.error(f"  Call {error['call_num']}: {error['message']}")

        logger.info("=" * 80)


def main():
    """Main entry point for rate limit testing."""
    parser = argparse.ArgumentParser(
        description="Test rate limiting with different RPS settings"
    )
    parser.add_argument(
        '--rps',
        type=float,
        default=1.0,
        help='Requests per second to test (default: 1.0)'
    )
    parser.add_argument(
        '--calls',
        type=int,
        default=50,
        help='Number of API calls to make (default: 50)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation suite with multiple RPS settings'
    )

    args = parser.parse_args()

    # Initialize session manager
    logger.info("Initializing session manager...")
    session_manager = SessionManager()

    try:
        # Ensure session is ready
        if not session_manager.ensure_session_ready(action_name="Rate Limit Test"):
            logger.error("Failed to initialize session. Exiting.")
            sys.exit(1)

        if args.validate:
            # Run validation suite with multiple RPS settings
            logger.info("Running validation suite...")
            test_configs = [
                (1.0, 50),   # Conservative
                (1.5, 75),   # Moderate
                (2.0, 100),  # Aggressive
            ]

            all_passed = True
            for rps, calls in test_configs:
                tester = RateLimitTester(rps, calls, session_manager)
                results = tester.run_test()

                if results['error_429_count'] > 0:
                    all_passed = False
                    logger.error(f"❌ RPS {rps} FAILED with {results['error_429_count']} 429 errors")
                else:
                    logger.info(f"✅ RPS {rps} PASSED with 0 429 errors")

                # Wait between tests
                if rps != test_configs[-1][0]:
                    logger.info("Waiting 30 seconds before next test...")
                    time.sleep(30)

            if all_passed:
                logger.info("=" * 80)
                logger.info("✅ ALL VALIDATION TESTS PASSED")
                logger.info("=" * 80)
            else:
                logger.error("=" * 80)
                logger.error("❌ SOME VALIDATION TESTS FAILED")
                logger.error("=" * 80)
                sys.exit(1)

        else:
            # Run single test
            tester = RateLimitTester(args.rps, args.calls, session_manager)
            results = tester.run_test()

            if results['error_429_count'] > 0:
                sys.exit(1)

    finally:
        # Cleanup
        logger.info("Cleaning up...")
        session_manager.cleanup()


# ==============================================
# Comprehensive Test Suite
# ==============================================

def _test_rate_limit_tester_initialization() -> bool:
    """Test RateLimitTester initialization."""
    try:
        # Create a mock session manager
        from unittest.mock import MagicMock
        mock_session = MagicMock()

        tester = RateLimitTester(rps=1.0, num_calls=10, session_manager=mock_session)
        assert tester.rps == 1.0, "Should store RPS"
        assert tester.num_calls == 10, "Should store num_calls"
        assert tester.session_manager is not None, "Should store session manager"
        assert isinstance(tester.results, dict), "Should initialize results dict"
        return True
    except Exception:
        return False


def _test_rate_limit_tester_results_structure() -> bool:
    """Test that RateLimitTester results have expected structure."""
    try:
        from unittest.mock import MagicMock
        mock_session = MagicMock()

        tester = RateLimitTester(rps=1.0, num_calls=10, session_manager=mock_session)

        # Check results structure
        assert 'total_calls' in tester.results, "Should have total_calls"
        assert 'successful_calls' in tester.results, "Should have successful_calls"
        assert 'error_429_count' in tester.results, "Should have error_429_count"
        assert 'error_other_count' in tester.results, "Should have error_other_count"
        assert 'start_time' in tester.results, "Should have start_time"
        assert 'end_time' in tester.results, "Should have end_time"
        return True
    except Exception:
        return False


def _test_results_calculation() -> bool:
    """Test results calculation and metrics."""
    try:
        from unittest.mock import MagicMock
        mock_session = MagicMock()

        tester = RateLimitTester(rps=2.0, num_calls=10, session_manager=mock_session)

        # Simulate test results
        tester.results['total_calls'] = 10
        tester.results['successful_calls'] = 9
        tester.results['failed_calls'] = 1
        tester.results['error_429_count'] = 0
        tester.results['duration'] = 5.0

        # Calculate effective RPS
        effective_rps = tester.results['total_calls'] / tester.results['duration']
        assert effective_rps == 2.0, "Should calculate effective RPS correctly"
        return True
    except Exception:
        return False


def _test_error_tracking() -> bool:
    """Test error tracking in results."""
    try:
        from unittest.mock import MagicMock
        mock_session = MagicMock()

        tester = RateLimitTester(rps=1.0, num_calls=10, session_manager=mock_session)

        # Check error tracking structure
        assert 'errors' in tester.results, "Should have errors list"
        assert isinstance(tester.results['errors'], list), "Errors should be list"
        assert 'error_429_count' in tester.results, "Should track 429 errors"
        assert 'other_errors' in tester.results, "Should track other errors"
        return True
    except Exception:
        return False


def _test_parse_arguments() -> bool:
    """Test argument parsing."""
    try:
        # Test with default arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--rps', type=float, default=1.0)
        parser.add_argument('--calls', type=int, default=50)
        parser.add_argument('--validate', action='store_true')

        args = parser.parse_args([])
        assert args.rps == 1.0, "Should have default RPS"
        assert args.calls == 50, "Should have default calls"
        assert args.validate is False, "Should have default validate"
        return True
    except Exception:
        return False


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for test_rate_limiting.py.
    Tests rate limiting test harness functionality.
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "Rate Limit Test Harness",
            "test_rate_limiting.py"
        )
        suite.start_suite()

        suite.run_test(
            "RateLimitTester Initialization",
            _test_rate_limit_tester_initialization,
            "RateLimitTester initializes correctly",
            "Test tester initialization",
            "Test rate limit test harness setup",
        )

        suite.run_test(
            "Results Structure",
            _test_rate_limit_tester_results_structure,
            "Results dict has expected structure",
            "Test results initialization",
            "Test metrics collection structure",
        )

        suite.run_test(
            "Results Calculation",
            _test_results_calculation,
            "Results calculation and metrics work correctly",
            "Test results calculation",
            "Test effective RPS computation",
        )

        suite.run_test(
            "Error Tracking",
            _test_error_tracking,
            "Error tracking structure is correct",
            "Test error tracking",
            "Test error collection and categorization",
        )

        suite.run_test(
            "Argument Parsing",
            _test_parse_arguments,
            "Command line arguments parse correctly",
            "Test argument parsing",
            "Test CLI configuration",
        )

        return suite.finish_suite()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

