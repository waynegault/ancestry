#!/usr/bin/env python3

"""
Action 6 Reliable Integration Demo

This module demonstrates how the new ReliableSessionManager would integrate
with the existing Action 6 DNA match gathering system to provide improved
reliability and eliminate the race conditions that caused cascade failures.

Key Integration Points:
1. Replace existing SessionManager with ReliableSessionManager
2. Integrate critical error detection with Action 6 processing
3. Add resource monitoring to prevent memory exhaustion
4. Implement proactive browser restart strategy
5. Provide comprehensive monitoring and alerting

This is a demonstration/prototype showing the integration approach.
For production deployment, this would replace the existing action6_gather.py
coordination logic.
"""

import logging
import time
from typing import Any, Dict, Optional

from core.session_manager import (
    CriticalError,
    SessionManager as ReliableSessionManager,
    SystemHealthError,
)

logger = logging.getLogger(__name__)


class Action6ReliableCoordinator:
    """
    Reliable coordinator for Action 6 DNA match gathering.

    Integrates ReliableSessionManager with Action 6 processing logic
    to provide improved reliability and eliminate race conditions.
    """

    def __init__(self):
        self.session_manager = ReliableSessionManager()
        self.processing_stats = {
            'pages_completed': 0,
            'matches_processed': 0,
            'errors_encountered': 0,
            'restarts_performed': 0,
            'start_time': time.time()
        }

    def process_dna_matches(self, start_page: int = 1, end_page: int = 724) -> Dict[str, Any]:
        """
        Process DNA matches with improved reliability.

        Args:
            start_page: Starting page number
            end_page: Ending page number (724 for full workload)

        Returns:
            Dict containing processing results and statistics
        """
        logger.info(f"🚀 Starting reliable DNA match processing: pages {start_page} to {end_page}")

        try:
            # Override the _process_single_page method to use Action 6 logic
            self.session_manager._process_single_page = self._process_action6_page

            # Process pages using reliable session manager
            success = self.session_manager.process_pages(start_page, end_page)

            # Get final statistics
            session_summary = self.session_manager.get_session_summary()

            return {
                'success': success,
                'pages_processed': session_summary['session_state']['pages_processed'],
                'errors_encountered': session_summary['session_state']['error_count'],
                'restarts_performed': session_summary['session_state']['restart_count'],
                'session_duration_hours': session_summary['session_state']['session_duration_hours'],
                'final_system_health': session_summary['system_health'],
                'error_summary': session_summary['error_summary'],
                'processing_stats': self.processing_stats
            }

        except CriticalError as e:
            logger.critical(f"🚨 Critical error halted processing: {e}")
            return self._create_failure_result(f"Critical error: {e}")

        except SystemHealthError as e:
            logger.critical(f"🚨 System health error halted processing: {e}")
            return self._create_failure_result(f"System health error: {e}")

        except Exception as e:
            logger.error(f"❌ Unexpected error in DNA match processing: {e}")
            return self._create_failure_result(f"Unexpected error: {e}")

        finally:
            # Always clean up resources
            self.session_manager.cleanup()

    def _process_action6_page(self, page_num: int) -> Dict[str, Any]:
        """
        Process a single Action 6 page with DNA match gathering logic.

        This method integrates the existing Action 6 page processing logic
        with the new reliable session management framework.
        """
        logger.debug(f"🔄 Processing Action 6 page {page_num}")

        # Simulate Action 6 page processing logic
        # In production, this would contain the actual DNA match gathering code

        try:
            # Check browser health before processing
            if not self.session_manager._quick_browser_health_check():
                raise RuntimeError("Browser health check failed before page processing")

            # Simulate navigation to page
            page_url = f"https://www.ancestry.com/dna/matches/page/{page_num}"
            logger.debug(f"🌐 Navigating to page: {page_url}")

            # Simulate cookie access (the originally failing operation)
            cookies = self.session_manager.browser_manager.driver.get_cookies()
            if not isinstance(cookies, list):
                raise RuntimeError("Failed to access cookies - potential WebDriver death")

            # Simulate DNA match extraction
            matches_found = self._extract_dna_matches_simulation(page_num)

            # Update processing statistics
            self.processing_stats['pages_completed'] += 1
            self.processing_stats['matches_processed'] += matches_found

            return {
                'page_num': page_num,
                'matches_found': matches_found,
                'cookies_count': len(cookies),
                'processing_time': time.time(),
                'success': True
            }

        except Exception as e:
            logger.error(f"❌ Error processing Action 6 page {page_num}: {e}")
            self.processing_stats['errors_encountered'] += 1
            raise

    def _extract_dna_matches_simulation(self, page_num: int) -> int:
        """
        Simulate DNA match extraction for demonstration.

        In production, this would contain the actual match extraction logic
        from the existing Action 6 system.
        """
        # Simulate variable number of matches per page (realistic for DNA matching)
        import random
        matches_per_page = random.randint(15, 25)  # Typical range for DNA match pages

        # Simulate some processing time
        time.sleep(0.1)  # Simulate network and processing time

        return matches_per_page

    def _create_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized failure result."""
        session_summary = self.session_manager.get_session_summary()

        return {
            'success': False,
            'error_message': error_message,
            'pages_processed': session_summary['session_state']['pages_processed'],
            'errors_encountered': session_summary['session_state']['error_count'],
            'restarts_performed': session_summary['session_state']['restart_count'],
            'session_duration_hours': session_summary['session_state']['session_duration_hours'],
            'final_system_health': session_summary['system_health'],
            'processing_stats': self.processing_stats
        }

    def get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time processing status for monitoring."""
        session_summary = self.session_manager.get_session_summary()

        return {
            'current_page': session_summary['session_state']['current_page'],
            'pages_processed': session_summary['session_state']['pages_processed'],
            'processing_rate_pages_per_hour': self._calculate_processing_rate(),
            'estimated_completion_time': self._estimate_completion_time(),
            'system_health': session_summary['system_health'],
            'recent_errors': session_summary['error_summary'],
            'browser_status': session_summary['browser_status'],
            'processing_stats': self.processing_stats
        }

    def _calculate_processing_rate(self) -> float:
        """Calculate current processing rate in pages per hour."""
        elapsed_hours = (time.time() - self.processing_stats['start_time']) / 3600
        if elapsed_hours > 0:
            return self.processing_stats['pages_completed'] / elapsed_hours
        return 0.0

    def _estimate_completion_time(self) -> Optional[str]:
        """Estimate completion time based on current processing rate."""
        rate = self._calculate_processing_rate()
        if rate > 0:
            remaining_pages = 724 - self.processing_stats['pages_completed']
            remaining_hours = remaining_pages / rate
            completion_time = time.time() + (remaining_hours * 3600)
            return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(completion_time))
        return None


def demo_reliable_processing():
    """
    Demonstrate reliable Action 6 processing with the new architecture.

    This demo shows how the system would handle a small subset of pages
    to validate the integration and reliability improvements.
    """
    print("🚀 Action 6 Reliable Integration Demo")
    print("=" * 60)

    # Create reliable coordinator
    coordinator = Action6ReliableCoordinator()

    # Process a small subset of pages for demonstration (5 pages)
    print("📊 Processing 5 pages to demonstrate reliability improvements...")

    start_time = time.time()
    result = coordinator.process_dna_matches(start_page=1, end_page=5)
    duration = time.time() - start_time

    # Display results
    print("\n📈 Processing Results:")
    print(f"   Success: {result['success']}")
    print(f"   Pages Processed: {result['pages_processed']}")
    print(f"   Matches Found: {result['processing_stats']['matches_processed']}")
    print(f"   Errors Encountered: {result['errors_encountered']}")
    print(f"   Browser Restarts: {result['restarts_performed']}")
    print(f"   Duration: {duration:.2f} seconds")

    if result['success']:
        print(f"   Processing Rate: {result['processing_stats']['matches_processed'] / duration:.1f} matches/second")

    # Display system health
    health = result['final_system_health']
    print("\n🏥 Final System Health:")
    print(f"   Overall: {health['overall']}")
    print(f"   Memory: {health['memory']['status']} ({health['memory']['available_mb']:.1f}MB available)")
    print(f"   Processes: {health['processes']['status']} ({health['processes']['process_count']} browser processes)")
    print(f"   Network: {health['network']['status']}")

    # Show error summary if any
    if result['error_summary']:
        print(f"\n⚠️ Error Summary: {result['error_summary']}")

    print("\n✅ Demo completed successfully!")
    return result['success']


# ============================================================================
# EMBEDDED TESTS - Following user preference for tests in same file
# ============================================================================

def test_action6_coordinator_initialization():
    """Test Action6ReliableCoordinator initialization."""
    print("🧪 Testing Action6ReliableCoordinator initialization...")

    coordinator = Action6ReliableCoordinator()

    # Test initialization
    assert coordinator.session_manager is not None, "Session manager should be initialized"
    assert coordinator.processing_stats is not None, "Processing stats should be initialized"
    assert 'pages_completed' in coordinator.processing_stats, "Stats should include pages completed"
    assert 'start_time' in coordinator.processing_stats, "Stats should include start time"

    print("   ✅ Action6ReliableCoordinator initialization working correctly")
    return True


def test_processing_rate_calculation():
    """Test processing rate and completion time estimation."""
    print("🧪 Testing processing rate calculation...")

    coordinator = Action6ReliableCoordinator()

    # Simulate some processing
    coordinator.processing_stats['pages_completed'] = 10
    coordinator.processing_stats['start_time'] = time.time() - 3600  # 1 hour ago

    # Test rate calculation
    rate = coordinator._calculate_processing_rate()
    assert rate > 0, "Processing rate should be positive"
    assert rate == 10.0, f"Expected rate 10.0 pages/hour, got {rate}"

    # Test completion time estimation
    completion_time = coordinator._estimate_completion_time()
    assert completion_time is not None, "Completion time should be estimated"

    print("   ✅ Processing rate calculation working correctly")
    return True


def test_real_time_status():
    """Test real-time status reporting."""
    print("🧪 Testing real-time status reporting...")

    coordinator = Action6ReliableCoordinator()

    # Get status
    status = coordinator.get_real_time_status()

    # Verify status structure
    assert 'current_page' in status, "Status should include current page"
    assert 'pages_processed' in status, "Status should include pages processed"
    assert 'processing_rate_pages_per_hour' in status, "Status should include processing rate"
    assert 'system_health' in status, "Status should include system health"
    assert 'browser_status' in status, "Status should include browser status"
    assert 'processing_stats' in status, "Status should include processing stats"

    print("   ✅ Real-time status reporting working correctly")
    return True


def test_dna_match_extraction_simulation():
    """Test DNA match extraction simulation."""
    print("🧪 Testing DNA match extraction simulation...")

    coordinator = Action6ReliableCoordinator()

    # Test simulation
    matches = coordinator._extract_dna_matches_simulation(1)

    # Verify realistic range
    assert isinstance(matches, int), "Matches should be integer"
    assert 15 <= matches <= 25, f"Matches should be in realistic range 15-25, got {matches}"

    print("   ✅ DNA match extraction simulation working correctly")
    return True


def test_failure_result_creation():
    """Test failure result creation."""
    print("🧪 Testing failure result creation...")

    coordinator = Action6ReliableCoordinator()

    # Test failure result
    result = coordinator._create_failure_result("Test error message")

    # Verify structure
    assert not result['success'], "Failure result should have success=False"
    assert 'error_message' in result, "Failure result should include error message"
    assert 'pages_processed' in result, "Failure result should include pages processed"
    assert 'final_system_health' in result, "Failure result should include final system health"
    assert result['error_message'] == "Test error message", "Error message should match"

    print("   ✅ Failure result creation working correctly")
    return True


def run_integration_tests():
    """Run all embedded tests for Action 6 integration."""
    print("🚀 Running Embedded Tests for Action 6 Integration...")
    print("=" * 60)

    tests = [
        ("Action6 Coordinator Initialization", test_action6_coordinator_initialization),
        ("Processing Rate Calculation", test_processing_rate_calculation),
        ("Real-time Status Reporting", test_real_time_status),
        ("DNA Match Extraction Simulation", test_dna_match_extraction_simulation),
        ("Failure Result Creation", test_failure_result_creation),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            result = test_func()
            if result:
                passed += 1
                print(f"✅ PASSED: {test_name}")
            else:
                failed += 1
                print(f"❌ FAILED: {test_name}")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED: {test_name} - {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All integration tests passed!")
        return True
    else:
        print(f"❌ {failed} tests failed!")
        return False


if __name__ == "__main__":
    import sys

    # Check if user wants to run tests or demo
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run embedded tests
        success = run_integration_tests()
        sys.exit(0 if success else 1)
    else:
        # Run demo
        try:
            success = demo_reliable_processing()
            if success:
                print("\n🎉 Reliable Action 6 integration demo completed successfully!")
                print("\n💡 Tip: Run with --test flag to execute embedded tests")
            else:
                print("\n❌ Demo encountered issues - see logs for details")
        except Exception as e:
            print(f"\n💥 Demo failed with exception: {e}")
            import traceback
            traceback.print_exc()
