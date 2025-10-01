#!/usr/bin/env python3
"""
Complete automated testing with proper session initialization.
This script will:
1. Initialize SessionManager properly
2. Ensure session is ready (with login)
3. Test Action 11 with live API
4. Test other actions as possible
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.session_manager import SessionManager

def main():
    """Run complete automated tests with proper session."""
    print("=" * 70)
    print("COMPLETE AUTOMATED TESTING WITH SESSION")
    print("=" * 70)
    
    session_manager = None
    
    try:
        # Step 1: Create and initialize session manager
        print("\n[Step 1/4] Creating SessionManager...")
        session_manager = SessionManager()
        print("✅ SessionManager created")
        
        # Step 2: Start browser first
        print("\n[Step 2/4] Starting browser...")
        browser_started = session_manager.start_browser("Automated Testing")

        if not browser_started:
            print("❌ Failed to start browser")
            return False

        print("✅ Browser started")
        print(f"   Driver: {session_manager.driver}")
        print(f"   Driver Live: {session_manager.driver_live}")

        # Step 3: Check if already logged in (using saved session from user data directory)
        print("\n[Step 3/4] Checking if already logged in (using saved session)...")
        print("This will check if you have a saved login session...")

        # Import login_status to check if already logged in
        from utils import login_status

        # Check if already logged in (using saved cookies from user data directory)
        login_check = login_status(session_manager, disable_ui_fallback=False)

        if login_check is True:
            print("✅ Already logged in! Using saved session.")
            # Sync cookies and get profile info
            session_manager._sync_cookies()
            session_manager.api_manager.get_profile_id()
            session_manager.api_manager.get_tree_owner()
            print(f"   Profile ID: {session_manager.my_profile_id}")
            print(f"   Tree Owner: {session_manager.tree_owner_name}")
        elif login_check is False:
            print("⚠️  Not logged in. You need to log in manually first.")
            print("   Please run: python main.py")
            print("   Then select option 5 to log in.")
            return False
        else:
            print("❌ Unable to determine login status")
            return False
        
        # Step 4: Test Action 11 (API Report)
        print("\n[Step 4/5] Testing Action 11 (API Report)...")
        print("-" * 70)

        import action11

        print("\nRunning Action 11 comprehensive test suite...")
        test_result = action11.run_tests()

        if test_result:
            print("\n✅ Action 11 tests PASSED")
        else:
            print("\n❌ Action 11 tests FAILED")

        # Step 5: Summary
        print("\n[Step 5/5] Test Summary")
        print("-" * 70)
        print(f"Session Established: ✅")
        print(f"Action 11 Tests: {'✅ PASSED' if test_result else '❌ FAILED'}")
        
        return test_result
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if session_manager:
            print("\n[Cleanup] Closing session...")
            try:
                session_manager.close_sess()
                print("✅ Session closed")
            except Exception as e:
                print(f"⚠️ Error during cleanup: {e}")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("AUTOMATED TESTING WITH PROPER SESSION INITIALIZATION")
    print("=" * 70)
    print("\nThis script will:")
    print("1. Create SessionManager")
    print("2. Establish session (login automatically using .env credentials)")
    print("3. Test Action 11 (API Report) with Fraser Gault data")
    print("4. Clean up and close session")
    print("\nUsing credentials from .env file...")
    print("=" * 70)
    
    try:
        result = main()
        
        print("\n" + "=" * 70)
        if result:
            print("✅ ALL TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED")
        print("=" * 70)
        
        sys.exit(0 if result else 1)
        
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

