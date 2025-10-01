"""Detailed test of login process with 2FA."""

import sys
sys.path.insert(0, '.')

from core.session_manager import SessionManager
from utils import log_in, login_status
import time

print("=" * 70)
print("DETAILED LOGIN TEST WITH 2FA (120 second timeout)")
print("=" * 70)

# Create session manager
print("\n[1/4] Creating SessionManager...")
session_manager = SessionManager()
print("✅ SessionManager created")

# Start browser
print("\n[2/4] Starting browser...")
browser_started = session_manager.start_browser("Login Test")

if browser_started:
    print("✅ Browser started successfully")
    
    # Check initial login status
    print("\n[3/4] Checking initial login status...")
    initial_status = login_status(session_manager, disable_ui_fallback=False)
    print(f"   Initial status: {initial_status}")
    
    if initial_status is not True:
        print("\n[4/4] Attempting login...")
        print("   This will:")
        print("   - Navigate to login page")
        print("   - Accept cookies")
        print("   - Enter credentials")
        print("   - Detect and handle 2FA (120 second timeout)")
        print("\n" + "=" * 70)
        print("PLEASE WATCH THE BROWSER WINDOW!")
        print("If 2FA appears, you have 120 seconds to enter the code.")
        print("=" * 70)
        
        login_result = log_in(session_manager)
        
        print("\n" + "=" * 70)
        print(f"Login result: {login_result}")
        print("=" * 70)
        
        if login_result == "LOGIN_SUCCEEDED":
            print("\n✅ LOGIN SUCCEEDED!")
            
            # Verify with login_status
            print("\nVerifying login status...")
            final_status = login_status(session_manager, disable_ui_fallback=False)
            print(f"   Final status: {final_status}")
            
            if final_status is True:
                print("\n✅ Login verified successfully!")
                print(f"   Profile ID: {session_manager.my_profile_id}")
                print(f"   Tree Owner: {session_manager.tree_owner_name}")
            else:
                print("\n⚠️  Login succeeded but verification failed")
                print("   This might mean cookies need to sync")
        else:
            print(f"\n❌ LOGIN FAILED: {login_result}")
            print("\nPossible reasons:")
            print("   - Incorrect credentials")
            print("   - 2FA timeout (didn't enter code in 120 seconds)")
            print("   - Ancestry.com detected automation")
            print("   - Network issues")
    else:
        print("\n✅ Already logged in!")
        print(f"   Profile ID: {session_manager.my_profile_id}")
        print(f"   Tree Owner: {session_manager.tree_owner_name}")
    
    # Keep browser open for inspection
    print("\n" + "=" * 70)
    print("Browser will remain open for 30 seconds for inspection...")
    print("=" * 70)
    time.sleep(30)
    
    # Cleanup
    print("\n[Cleanup] Closing session...")
    session_manager.close_sess()
    print("✅ Session closed")
else:
    print("❌ Failed to start browser")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

