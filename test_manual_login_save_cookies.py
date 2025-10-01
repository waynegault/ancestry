#!/usr/bin/env python3

"""
Manual login test - user completes login manually, we save cookies
"""

import time
from core.session_manager import SessionManager
from utils import _save_login_cookies, login_status

def test_manual_login_and_save_cookies():
    """Test manual login and cookie saving"""
    print("=" * 70)
    print("MANUAL LOGIN AND COOKIE SAVE TEST")
    print("=" * 70)
    
    print("\n[1/4] Creating SessionManager...")
    session_manager = SessionManager()
    print("✅ SessionManager created")
    
    print("\n[2/4] Starting browser...")
    if not session_manager.start_browser():
        print("❌ Failed to start browser")
        return False
    print("✅ Browser started successfully")
    
    print("\n[3/4] Navigating to Ancestry login page...")
    try:
        session_manager.driver.get("https://www.ancestry.co.uk/account/signin")
        time.sleep(3)
        print("✅ Navigated to login page")
    except Exception as e:
        print(f"❌ Failed to navigate: {e}")
        return False
    
    print("\n" + "="*70)
    print("🎯 PLEASE COMPLETE LOGIN MANUALLY IN THE BROWSER")
    print("   1. Accept cookies")
    print("   2. Enter your username and click Next")
    print("   3. Enter your password and click Sign In")
    print("   4. Complete 2FA if required")
    print("   5. Wait for the home page to load")
    print("="*70)
    
    # Wait for user to complete login
    print("\n⏳ Waiting for you to complete login...")
    print("   Checking login status every 10 seconds...")
    
    max_wait_time = 300  # 5 minutes
    check_interval = 10  # Check every 10 seconds
    checks = max_wait_time // check_interval
    
    for i in range(checks):
        time.sleep(check_interval)
        
        print(f"\n[Check {i+1}/{checks}] Checking login status...")
        status = login_status(session_manager, disable_ui_fallback=True)
        
        if status is True:
            print("🎉 LOGIN DETECTED!")
            
            print("\n[4/4] Saving cookies...")
            if _save_login_cookies(session_manager):
                print("✅ Cookies saved successfully!")
                
                # Verify cookie file was created
                from utils import _get_cookie_file_path
                import os
                cookie_file = _get_cookie_file_path()
                if os.path.exists(cookie_file):
                    file_size = os.path.getsize(cookie_file)
                    print(f"   Cookie file: {cookie_file}")
                    print(f"   File size: {file_size} bytes")
                    return True
                else:
                    print("❌ Cookie file was not created")
                    return False
            else:
                print("❌ Failed to save cookies")
                return False
        elif status is False:
            print(f"   Still not logged in... ({(i+1)*check_interval}s elapsed)")
        else:
            print(f"   Login status check failed... ({(i+1)*check_interval}s elapsed)")
    
    print(f"\n❌ Timeout after {max_wait_time} seconds - login not detected")
    return False

if __name__ == "__main__":
    try:
        result = test_manual_login_and_save_cookies()
        print(f"\n{'='*70}")
        print(f"TEST RESULT: {'SUCCESS' if result else 'FAILED'}")
        print(f"{'='*70}")
        
        if result:
            print("\n🎉 SUCCESS! Cookies have been saved.")
            print("   Now you can test cookie persistence by running:")
            print("   python test_cookie_persistence.py")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
