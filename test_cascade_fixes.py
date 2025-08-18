#!/usr/bin/env python3
"""
Test script to validate the session death cascade fixes.
This tests the critical fixes implemented to prevent infinite cascade loops.
"""

import sys
import time
from unittest.mock import MagicMock, patch
from core.session_manager import SessionManager

def test_cascade_hard_stop():
    """Test that cascade count hard stop works at 20."""
    print("🧪 Testing cascade hard stop mechanism...")
    
    # Create a mock session manager
    session_manager = SessionManager()
    
    # Initialize health monitor
    session_manager.session_health_monitor = {
        'death_detected': MagicMock(),
        'death_cascade_count': 0,
        'emergency_shutdown': False,
        'is_alive': MagicMock(),
    }
    
    # Mock the death cascade detection
    session_manager.session_health_monitor['death_detected'].is_set.return_value = True
    
    # Test normal operation (should try recovery)
    session_manager.session_health_monitor['death_cascade_count'] = 1
    with patch.object(session_manager, 'attempt_cascade_recovery', return_value=True):
        result = session_manager.should_halt_operations()
        assert not result, "Should not halt on first cascade with successful recovery"
    
    # Test hard stop at 20 cascades
    session_manager.session_health_monitor['death_cascade_count'] = 19  # Will increment to 20
    with patch.object(session_manager, 'emergency_shutdown') as mock_shutdown:
        result = session_manager.should_halt_operations()
        assert result, "Should halt at cascade count 20"
        mock_shutdown.assert_called_once()
    
    print("   ✅ Hard stop mechanism works correctly")

def test_emergency_shutdown():
    """Test emergency shutdown functionality."""
    print("🧪 Testing emergency shutdown mechanism...")
    
    session_manager = SessionManager()
    session_manager.session_health_monitor = {
        'emergency_shutdown': False,
        'death_detected': MagicMock(),
        'is_alive': MagicMock(),
        'death_cascade_count': 0,
    }
    
    # Mock driver property
    with patch.object(type(session_manager), 'driver', new_callable=lambda: MagicMock()) as mock_driver:
    
        # Test emergency shutdown
        session_manager.emergency_shutdown("Test shutdown")

        assert session_manager.session_health_monitor['emergency_shutdown'] == True
        assert session_manager.session_health_monitor['death_cascade_count'] == 9999
        mock_driver.quit.assert_called_once()

        # Test emergency shutdown detection
        assert session_manager.is_emergency_shutdown() == True
    
    print("   ✅ Emergency shutdown mechanism works correctly")

def test_session_health_cascade_detection():
    """Test that session health check detects cascades."""
    print("🧪 Testing session health cascade detection...")
    
    session_manager = SessionManager()
    session_manager.session_health_monitor = {
        'death_detected': MagicMock(),
        'death_cascade_count': 6,  # Above threshold of 5
        'is_alive': MagicMock(),
        'last_heartbeat': time.time(),
    }
    
    # Mock cascade detection
    session_manager.session_health_monitor['death_detected'].is_set.return_value = True
    
    # Mock is_sess_valid to return True (driver exists)
    with patch.object(session_manager, 'is_sess_valid', return_value=True):
        result = session_manager.check_session_health()
        assert not result, "Should return False when cascade count exceeds threshold"
    
    print("   ✅ Session health cascade detection works correctly")

def main():
    """Run all cascade fix validation tests."""
    print("🔧 VALIDATING SESSION DEATH CASCADE FIXES")
    print("=" * 50)
    
    try:
        test_cascade_hard_stop()
        test_emergency_shutdown()
        test_session_health_cascade_detection()
        
        print("\n🎉 ALL CASCADE FIX TESTS PASSED!")
        print("✅ Hard stop mechanism: WORKING")
        print("✅ Emergency shutdown: WORKING") 
        print("✅ Health check cascade detection: WORKING")
        print("\n🛡️ The session death cascade infinite loop issue has been FIXED!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
