#!/usr/bin/env python3

"""
Test cookie persistence functionality
"""

import os
import time
from core.session_manager import SessionManager
from utils import _save_login_cookies, _load_login_cookies, _get_cookie_file_path

def test_cookie_persistence():
    """Test that we can save and load cookies"""
    print("=" * 70)
    print("COOKIE PERSISTENCE TEST")
    print("=" * 70)
    
    # Check if cookie file exists from previous login
    cookie_file = _get_cookie_file_path()
    print(f"Cookie file path: {cookie_file}")
    
    if os.path.exists(cookie_file):
        print(f"✅ Cookie file exists: {cookie_file}")
        
        # Check file size
        file_size = os.path.getsize(cookie_file)
        print(f"   File size: {file_size} bytes")
        
        # Show first few lines
        try:
            with open(cookie_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')[:10]  # First 10 lines
                print(f"   First few lines:")
                for i, line in enumerate(lines):
                    print(f"   {i+1}: {line[:80]}...")  # First 80 chars
        except Exception as e:
            print(f"   Error reading file: {e}")
    else:
        print(f"❌ No cookie file found at: {cookie_file}")
    
    print("\n[1/3] Creating SessionManager...")
    session_manager = SessionManager()
    print("✅ SessionManager created")
    
    print("\n[2/3] Starting browser...")
    if not session_manager.start_browser():
        print("❌ Failed to start browser")
        return False
    print("✅ Browser started successfully")
    
    # Test cookie loading
    print("\n[3/3] Testing cookie loading...")
    if _load_login_cookies(session_manager):
        print("✅ Cookies loaded successfully")
    else:
        print("⚠️  No cookies to load or loading failed")
    
    # Check login status after loading cookies
    print("\n[4/4] Checking login status after cookie loading...")
    from utils import login_status
    status = login_status(session_manager, disable_ui_fallback=True)
    
    if status is True:
        print("🎉 ALREADY LOGGED IN! Cookie persistence working!")
        return True
    elif status is False:
        print("❌ Not logged in - cookies didn't work or expired")
        return False
    else:
        print("⚠️  Login status check failed")
        return False

if __name__ == "__main__":
    try:
        result = test_cookie_persistence()
        print(f"\n{'='*70}")
        print(f"TEST RESULT: {'SUCCESS' if result else 'FAILED'}")
        print(f"{'='*70}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
