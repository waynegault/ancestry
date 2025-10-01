#!/usr/bin/env python3
"""
Test script to run actions with automated login.
This script will:
1. Start a browser session
2. Log in to Ancestry.com using credentials from .env
3. Test Action 11 (API Report) with Fraser Gault data
4. Clean up and close browser
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load .env file
from dotenv import load_dotenv
load_dotenv()

from core.session_manager import SessionManager
from utils import log_in, login_status

def test_action_11_with_login():
    """Test Action 11 with automated login."""
    print("=" * 60)
    print("Testing Action 11 with Automated Login")
    print("=" * 60)
    
    session_manager = None
    
    try:
        # Step 1: Create session manager
        print("\n[1/5] Creating session manager...")
        session_manager = SessionManager()
        print("✅ Session manager created")
        
        # Step 2: Start browser
        print("\n[2/5] Starting browser...")
        if not session_manager.browser_manager.ensure_driver_live("Test Login"):
            print("❌ Failed to start browser")
            return False
        print("✅ Browser started")
        
        # Step 3: Check login status
        print("\n[3/5] Checking login status...")
        status = login_status(session_manager, disable_ui_fallback=False)
        
        if status is True:
            print("✅ Already logged in")
        else:
            print("⚠️ Not logged in - attempting login...")
            
            # Step 4: Log in
            print("\n[4/5] Logging in to Ancestry.com...")
            print(f"Username: {os.getenv('ANCESTRY_USERNAME', 'Not set')}")
            
            login_result = log_in(session_manager)
            
            if login_result:
                print("✅ Login successful!")
                
                # Verify login
                final_status = login_status(session_manager, disable_ui_fallback=False)
                if final_status is True:
                    print("✅ Login verified")
                    if session_manager.my_profile_id:
                        print(f"   Profile ID: {session_manager.my_profile_id}")
                else:
                    print("❌ Login verification failed")
                    return False
            else:
                print("❌ Login failed")
                return False
        
        # Step 5: Test Action 11
        print("\n[5/5] Testing Action 11 (API Report)...")
        print("-" * 60)
        
        # Import action11 module
        import action11
        
        # Run Action 11 tests
        print("\nRunning Action 11 test suite...")
        result = action11.run_tests()
        
        if result:
            print("\n✅ Action 11 tests PASSED")
        else:
            print("\n❌ Action 11 tests FAILED")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if session_manager:
            print("\n[Cleanup] Closing browser session...")
            try:
                session_manager.close_sess()
                print("✅ Session closed")
            except Exception as e:
                print(f"⚠️ Error during cleanup: {e}")

def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("Automated Testing with Login")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Start a browser session")
    print("2. Log in to Ancestry.com")
    print("3. Test Action 11 (API Report)")
    print("4. Clean up and close browser")
    print("\nPress Ctrl+C to cancel...")
    print("=" * 60)
    
    try:
        import time
        time.sleep(2)  # Give user time to read
        
        result = test_action_11_with_login()
        
        print("\n" + "=" * 60)
        if result:
            print("✅ TESTING COMPLETE - ALL TESTS PASSED")
        else:
            print("❌ TESTING COMPLETE - SOME TESTS FAILED")
        print("=" * 60)
        
        return 0 if result else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

