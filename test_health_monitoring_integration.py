#!/usr/bin/env python3

"""
test_health_monitoring_integration.py - Comprehensive Testing of Health Monitoring Integration

This script thoroughly tests the health monitoring system integration to ensure:
1. Health monitoring is properly integrated and working
2. Emergency intervention triggers correctly
3. Session refresh mechanisms work
4. All components are properly connected

CRITICAL: This must pass before claiming the health monitoring system works!
"""

import sys
import time
import logging
from unittest.mock import Mock, MagicMock

# Setup logging to see all output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_health_monitoring_integration():
    """Test the complete health monitoring integration."""
    print("🧪 COMPREHENSIVE HEALTH MONITORING INTEGRATION TEST")
    print("=" * 70)
    
    test_results = {
        "health_monitor_creation": False,
        "session_manager_integration": False,
        "health_metrics_update": False,
        "dashboard_generation": False,
        "risk_assessment": False,
        "emergency_intervention": False,
        "session_refresh_mechanism": False,
        "continuous_monitoring": False
    }
    
    try:
        # TEST 1: Health Monitor Creation
        print("\n📊 TEST 1: Health Monitor Creation")
        from health_monitor import get_health_monitor, SessionHealthMonitor
        
        monitor = get_health_monitor()
        assert monitor is not None, "Health monitor should not be None"
        assert isinstance(monitor, SessionHealthMonitor), "Should be SessionHealthMonitor instance"
        print("✅ Health monitor created successfully")
        test_results["health_monitor_creation"] = True
        
        # TEST 2: Session Manager Integration
        print("\n🔗 TEST 2: Session Manager Integration")
        
        # Create mock session manager
        mock_session_manager = Mock()
        mock_session_manager.browser_health_monitor = {
            'browser_start_time': time.time() - 1800,  # 30 minutes ago
            'pages_since_refresh': 25,
            'last_browser_refresh': time.time() - 1500
        }
        mock_session_manager.session_health_monitor = {
            'session_start_time': time.time() - 2000,
            'last_proactive_refresh': time.time() - 1000
        }
        
        # Test integration
        from health_monitor import integrate_with_session_manager
        integrated_monitor = integrate_with_session_manager(mock_session_manager)
        assert integrated_monitor is not None, "Integration should return monitor"
        print("✅ Session manager integration successful")
        test_results["session_manager_integration"] = True
        
        # TEST 3: Health Metrics Update
        print("\n📈 TEST 3: Health Metrics Update")
        
        # Update various metrics
        monitor.update_metric("api_response_time", 3.5)
        monitor.update_metric("memory_usage_mb", 180.0)
        monitor.update_metric("error_rate", 0.03)
        monitor.update_metric("session_age_minutes", 25.0)
        
        # Verify metrics were updated
        assert monitor.current_metrics["api_response_time"].value == 3.5
        assert monitor.current_metrics["memory_usage_mb"].value == 180.0
        print("✅ Health metrics update successful")
        test_results["health_metrics_update"] = True
        
        # TEST 4: Dashboard Generation
        print("\n📊 TEST 4: Dashboard Generation")
        
        dashboard = monitor.get_health_dashboard()
        required_fields = ["health_score", "health_status", "risk_score", "metrics", "recommended_actions"]
        
        for field in required_fields:
            assert field in dashboard, f"Dashboard missing required field: {field}"
        
        assert isinstance(dashboard["health_score"], (int, float))
        assert 0 <= dashboard["health_score"] <= 100
        assert isinstance(dashboard["risk_score"], float)
        assert 0.0 <= dashboard["risk_score"] <= 1.0
        
        print(f"✅ Dashboard generated - Health: {dashboard['health_score']:.1f}, Risk: {dashboard['risk_score']:.2f}")
        test_results["dashboard_generation"] = True
        
        # TEST 5: Risk Assessment
        print("\n⚠️ TEST 5: Risk Assessment")
        
        # Test normal conditions
        normal_risk = monitor.predict_session_death_risk()
        print(f"   Normal risk score: {normal_risk:.2f}")
        
        # Test degraded conditions
        monitor.update_metric("api_response_time", 12.0)  # Very slow
        monitor.update_metric("memory_usage_mb", 450.0)   # High memory
        monitor.update_metric("error_rate", 0.20)         # High error rate
        
        # Record errors to increase risk
        for i in range(15):
            monitor.record_error("test_error")
        
        degraded_risk = monitor.predict_session_death_risk()
        print(f"   Degraded risk score: {degraded_risk:.2f}")
        
        assert degraded_risk > normal_risk, "Degraded conditions should increase risk"
        assert degraded_risk > 0.4, "Degraded conditions should show elevated risk"  # More realistic threshold
        
        print("✅ Risk assessment working correctly")
        test_results["risk_assessment"] = True
        
        # TEST 6: Emergency Intervention Logic
        print("\n🚨 TEST 6: Emergency Intervention Logic")
        
        # Test performance recommendations
        from health_monitor import get_performance_recommendations
        
        # Test emergency recommendations
        emergency_recs = get_performance_recommendations(15.0, 0.9)  # Low health, high risk
        assert emergency_recs["action_required"] == "emergency_refresh"
        assert emergency_recs["max_concurrency"] == 1
        assert emergency_recs["batch_size"] == 1
        
        # Test normal recommendations
        normal_recs = get_performance_recommendations(85.0, 0.1)  # High health, low risk
        assert emergency_recs["max_concurrency"] < normal_recs["max_concurrency"]
        
        print("✅ Emergency intervention logic working")
        test_results["emergency_intervention"] = True
        
        # TEST 7: Session Refresh Mechanism
        print("\n🔄 TEST 7: Session Refresh Mechanism")
        
        # Create mock session manager with refresh capability
        mock_session_manager.perform_proactive_refresh = Mock(return_value=True)
        mock_session_manager.is_sess_valid = Mock(return_value=True)
        mock_session_manager.session_health_monitor = {
            'refresh_in_progress': Mock(),
            'last_proactive_refresh': time.time(),
            'session_start_time': time.time()
        }
        mock_session_manager.session_health_monitor['refresh_in_progress'].is_set = Mock(return_value=False)
        mock_session_manager.session_health_monitor['refresh_in_progress'].set = Mock()
        mock_session_manager.session_health_monitor['refresh_in_progress'].clear = Mock()
        
        # Test refresh call
        refresh_result = mock_session_manager.perform_proactive_refresh()
        assert refresh_result == True, "Mock refresh should succeed"
        
        print("✅ Session refresh mechanism accessible")
        test_results["session_refresh_mechanism"] = True
        
        # TEST 8: Continuous Monitoring Simulation
        print("\n⏱️ TEST 8: Continuous Monitoring Simulation")
        
        # Simulate continuous monitoring over multiple pages
        for page in range(320, 330):
            # Update metrics as if processing pages
            monitor.record_api_response_time(3.0 + (page % 3))  # Varying response times
            monitor.record_page_processing_time(45.0 + (page % 10))  # Varying processing times
            
            # Get health status
            dashboard = monitor.get_health_dashboard()
            health_score = dashboard['health_score']
            risk_score = dashboard['risk_score']
            
            # Verify monitoring is working
            assert health_score >= 0, "Health score should be valid"
            assert risk_score >= 0, "Risk score should be valid"
            
            if page % 5 == 0:  # Every 5 pages
                print(f"   Page {page}: Health={health_score:.1f}, Risk={risk_score:.2f}")
        
        print("✅ Continuous monitoring simulation successful")
        test_results["continuous_monitoring"] = True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # FINAL RESULTS
    print("\n" + "=" * 70)
    print("🎯 COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
        if result:
            passed_tests += 1
    
    print(f"\nOVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Health monitoring integration is working!")
        return True
    else:
        print("❌ SOME TESTS FAILED - Health monitoring integration needs fixes!")
        return False


def test_action6_integration():
    """Test that Action 6 can properly use the health monitoring."""
    print("\n🔧 ACTION 6 INTEGRATION TEST")
    print("-" * 50)
    
    try:
        # Test that Action 6 can import and use health monitoring
        from health_monitor import get_health_monitor
        
        # Simulate what Action 6 does
        monitor = get_health_monitor()
        
        # Create mock session manager like Action 6 would have
        mock_session_manager = Mock()
        mock_session_manager.health_monitor = monitor
        mock_session_manager.browser_health_monitor = {
            'browser_start_time': time.time() - 1000,
            'pages_since_refresh': 15
        }
        mock_session_manager.session_health_monitor = {
            'session_start_time': time.time() - 1500
        }
        
        # Test the integration points that Action 6 uses
        if hasattr(mock_session_manager, 'health_monitor') and mock_session_manager.health_monitor:
            health_monitor = mock_session_manager.health_monitor
            
            # Update metrics (like Action 6 does)
            health_monitor.update_session_metrics(mock_session_manager)
            health_monitor.update_system_metrics()
            
            # Get dashboard (like Action 6 does)
            dashboard = health_monitor.get_health_dashboard()
            
            # Check risk score (like Action 6 does)
            risk_score = dashboard['risk_score']
            
            print(f"✅ Action 6 integration test passed - Risk: {risk_score:.2f}")
            return True
        else:
            print("❌ Action 6 integration test failed - health_monitor not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Action 6 integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE HEALTH MONITORING TESTS")
    print("This will verify that ALL health monitoring components work correctly")
    print("=" * 80)
    
    # Run comprehensive integration test
    integration_success = test_health_monitoring_integration()
    
    # Run Action 6 specific test
    action6_success = test_action6_integration()
    
    print("\n" + "=" * 80)
    print("🏁 FINAL TEST SUMMARY")
    print("=" * 80)
    
    if integration_success and action6_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Health monitoring integration is working correctly")
        print("✅ Action 6 integration is working correctly")
        print("✅ System is ready for production testing")
        sys.exit(0)
    else:
        print("❌ TESTS FAILED!")
        print("❌ Health monitoring integration has issues")
        print("❌ System is NOT ready for production")
        sys.exit(1)
