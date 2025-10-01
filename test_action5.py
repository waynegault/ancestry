"""Test Action 5 (Check Login) directly."""

import sys
sys.path.insert(0, '.')

from core.session_manager import SessionManager
import main

print("=" * 70)
print("ACTION 5 TEST - CHECK LOGIN")
print("=" * 70)

# Create session manager
print("\n[1/3] Creating SessionManager...")
session_manager = SessionManager()
print("✅ SessionManager created")

# Start browser
print("\n[2/3] Starting browser...")
browser_started = session_manager.start_browser("Action 5 Test")

if browser_started:
    print("✅ Browser started successfully")
    print(f"   Driver: {session_manager.driver}")
    print(f"   Driver Live: {session_manager.driver_live}")
    
    # Run Action 5
    print("\n[3/3] Running Action 5 (Check Login)...")
    print("-" * 70)
    result = main.check_login_actn(session_manager)
    print("-" * 70)
    print(f"\nAction 5 Result: {result}")
    
    if result:
        print("\n✅ Action 5 SUCCEEDED!")
        print(f"   Profile ID: {session_manager.my_profile_id}")
        print(f"   Tree Owner: {session_manager.tree_owner_name}")
    else:
        print("\n❌ Action 5 FAILED!")
    
    # Keep browser open for manual inspection
    print("\n" + "=" * 70)
    print("Browser will remain open for 60 seconds for inspection...")
    print("Check the browser window to see what happened.")
    print("=" * 70)
    
    import time
    time.sleep(60)
    
    # Cleanup
    print("\n[Cleanup] Closing session...")
    session_manager.close_sess()
    print("✅ Session closed")
else:
    print("❌ Failed to start browser")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

