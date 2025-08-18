#!/usr/bin/env python3
"""
Comprehensive testing framework for Action 6.
Tests all aspects of the enhanced cascade prevention system.
"""

import sys
import time
import logging
import unittest
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime, timedelta

# Import the modules we want to test
from core.session_manager import SessionManager
from core.health_monitor import HealthMonitor, HealthMetrics
from core.circuit_breaker import CircuitBreaker, CircuitState
from action6_gather import MaxApiFailuresExceededError

class TestCascadePrevention(unittest.TestCase):
    """Test cascade prevention mechanisms."""
    
    def setUp(self):
        self.session_manager = SessionManager()
        self.session_manager.session_health_monitor = {
            'death_detected': MagicMock(),
            'death_cascade_count': 0,
            'emergency_shutdown': False,
            'is_alive': MagicMock(),
        }
    
    def test_immediate_halt_on_first_cascade(self):
        """Test that first cascade triggers immediate halt."""
        self.session_manager.session_health_monitor['death_detected'].is_set.return_value = True
        
        result = self.session_manager.should_halt_operations()
        self.assertTrue(result, "Should halt immediately on first cascade")
        self.assertEqual(self.session_manager.session_health_monitor['death_cascade_count'], 1)
    
    def test_emergency_shutdown_at_threshold(self):
        """Test emergency shutdown at cascade threshold."""
        self.session_manager.session_health_monitor['death_detected'].is_set.return_value = True
        self.session_manager.session_health_monitor['death_cascade_count'] = 2
        
        with patch.object(self.session_manager, 'emergency_shutdown') as mock_shutdown:
            result = self.session_manager.should_halt_operations()
            self.assertTrue(result)
            mock_shutdown.assert_called_once()
    
    def test_no_recovery_attempts(self):
        """Test that no recovery attempts are made."""
        self.session_manager.session_health_monitor['death_detected'].is_set.return_value = True
        
        with patch.object(self.session_manager, 'attempt_cascade_recovery') as mock_recovery:
            self.session_manager.should_halt_operations()
            mock_recovery.assert_not_called()

class TestHealthMonitoring(unittest.TestCase):
    """Test health monitoring system."""
    
    def setUp(self):
        self.health_monitor = HealthMonitor()
    
    def test_api_call_recording(self):
        """Test API call metrics recording."""
        self.health_monitor.record_api_call(True, 1.5)
        self.health_monitor.record_api_call(False, 0.0)
        
        self.assertEqual(self.health_monitor.api_calls_total, 2)
        self.assertEqual(self.health_monitor.api_calls_successful, 1)
        self.assertEqual(len(self.health_monitor.error_timestamps), 1)
    
    def test_health_metrics_calculation(self):
        """Test health metrics calculation."""
        # Record some API calls
        for i in range(10):
            self.health_monitor.record_api_call(i < 8, 1.0)  # 80% success rate
        
        metrics = self.health_monitor.get_health_metrics(0, 5, 100)
        
        self.assertAlmostEqual(metrics.api_success_rate, 80.0, places=1)
        self.assertEqual(metrics.pages_processed, 5)
        self.assertEqual(metrics.matches_processed, 100)
    
    def test_health_warnings(self):
        """Test health warning detection."""
        # Create metrics with warning conditions
        metrics = HealthMetrics(
            api_success_rate=30.0,  # Below 50% threshold
            avg_response_time=15.0,  # Above 10s threshold
            cascade_count=0,
            pages_processed=5,
            matches_processed=100,
            errors_per_minute=15,  # Above 10/min threshold
            session_age_minutes=150,  # Above 120min threshold
            last_successful_api_call=datetime.now() - timedelta(minutes=10)  # Above 5min threshold
        )
        
        warnings = self.health_monitor.check_health_warnings(metrics)
        self.assertGreaterEqual(len(warnings), 4)  # Should have multiple warnings
    
    def test_restart_recommendation(self):
        """Test restart recommendation logic."""
        # Create metrics that should trigger restart recommendation
        metrics = HealthMetrics(
            api_success_rate=15.0,  # Very low
            avg_response_time=5.0,
            cascade_count=2,  # Cascades detected
            pages_processed=5,
            matches_processed=100,
            errors_per_minute=25,  # Very high
            session_age_minutes=200,  # Very long
            last_successful_api_call=datetime.now()
        )
        
        should_restart = self.health_monitor.should_recommend_restart(metrics)
        self.assertTrue(should_restart)

class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality."""
    
    def setUp(self):
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
    
    def test_normal_operation(self):
        """Test normal operation when circuit is closed."""
        def test_func():
            return "success"
        
        result = self.circuit_breaker.call(test_func)
        self.assertEqual(result, "success")
        self.assertEqual(self.circuit_breaker.get_state(), "closed")
    
    def test_circuit_opens_on_failures(self):
        """Test that circuit opens after threshold failures."""
        def failing_func():
            raise Exception("API failure")
        
        # Trigger failures to open circuit
        for i in range(3):
            with self.assertRaises(Exception):
                self.circuit_breaker.call(failing_func)
        
        self.assertEqual(self.circuit_breaker.get_state(), "open")
    
    def test_circuit_blocks_when_open(self):
        """Test that circuit blocks calls when open."""
        # Force circuit open
        self.circuit_breaker.force_open()
        
        def test_func():
            return "should not execute"
        
        with self.assertRaises(Exception) as context:
            self.circuit_breaker.call(test_func)
        
        self.assertIn("Circuit breaker OPEN", str(context.exception))
    
    def test_circuit_recovery(self):
        """Test circuit recovery after timeout."""
        # Force circuit open
        self.circuit_breaker.force_open()
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        def success_func():
            return "recovered"
        
        # Should attempt reset and succeed
        result = self.circuit_breaker.call(success_func)
        self.assertEqual(result, "recovered")
        self.assertEqual(self.circuit_breaker.get_state(), "closed")

class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def test_cascade_triggers_circuit_breaker(self):
        """Test that cascades can trigger circuit breaker."""
        circuit_breaker = CircuitBreaker(failure_threshold=2)
        
        def api_call_with_cascade():
            raise MaxApiFailuresExceededError("Session death cascade")
        
        # Trigger failures
        for i in range(2):
            with self.assertRaises(MaxApiFailuresExceededError):
                circuit_breaker.call(api_call_with_cascade)
        
        self.assertEqual(circuit_breaker.get_state(), "open")
    
    def test_health_monitor_with_circuit_breaker(self):
        """Test health monitoring with circuit breaker integration."""
        health_monitor = HealthMonitor()
        circuit_breaker = CircuitBreaker(failure_threshold=3)
        
        # Record failures that would trigger circuit breaker
        for i in range(5):
            health_monitor.record_api_call(False, 0.0)
        
        metrics = health_monitor.get_health_metrics(0, 1, 20)
        warnings = health_monitor.check_health_warnings(metrics)
        
        # Should have warnings about low success rate
        self.assertTrue(any("LOW SUCCESS RATE" in warning for warning in warnings))

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🧪 RUNNING COMPREHENSIVE ACTION 6 TESTS")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCascadePrevention))
    suite.addTests(loader.loadTestsFromTestCase(TestHealthMonitoring))
    suite.addTests(loader.loadTestsFromTestCase(TestCircuitBreaker))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print("\n🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("✅ Cascade Prevention: WORKING")
        print("✅ Health Monitoring: WORKING")
        print("✅ Circuit Breaker: WORKING")
        print("✅ Integration: WORKING")
        return True
    else:
        print(f"\n❌ TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
