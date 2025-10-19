#!/usr/bin/env python3
"""
Refresh Session Cookies for Phase 2 Testing

This script establishes a fresh Ancestry session and saves cookies for Phase 2 testing.
It will:
1. Start a browser session
2. Log in to Ancestry
3. Save cookies to ancestry_cookies.json
4. Verify the session is valid

Usage:
    python refresh_session_cookies.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.session_manager import SessionManager
from utils import _save_login_cookies, log_in
from config import config_schema
import logging

logger = logging.getLogger(__name__)

def refresh_cookies() -> bool:
    """Refresh session cookies by establishing a fresh login."""
    print("\n" + "=" * 80)
    print("REFRESHING SESSION COOKIES FOR PHASE 2 TESTING")
    print("=" * 80)
    
    try:
        # Step 1: Initialize SessionManager
        print("\n✓ Step 1: Initializing SessionManager...")
        sm = SessionManager()
        print("  ✅ SessionManager created")

        # Step 1.5: Mark browser as needed
        print("\n✓ Step 1.5: Marking browser as needed...")
        sm.browser_manager.browser_needed = True
        print("  ✅ Browser marked as needed")

        # Step 2: Start session (database + browser)
        print("\n✓ Step 2: Starting session (database + browser)...")
        if not sm.start_sess("Cookie Refresh"):
            print("  ❌ Failed to start session")
            return False
        print("  ✅ Session started")
        
        # Step 3: Force a fresh login to get all cookies
        print("\n✓ Step 3: Forcing fresh login to get all cookies...")

        # Step 4: Log in
        print("\n✓ Step 4: Logging in...")
        username = config_schema.api.username
        password = config_schema.api.password

        if not username or not password:
            print("  ❌ Missing ANCESTRY_USERNAME or ANCESTRY_PASSWORD in .env")
            return False

        login_result = log_in(sm)
        if login_result != "LOGIN_SUCCEEDED":
            print(f"  ❌ Login failed: {login_result}")
            return False
        print("  ✅ Login successful")

        # Wait for cookies to be set
        print("\n✓ Step 5: Waiting for cookies to be set...")
        time.sleep(5)

        # Step 5.5: Navigate to main site to ensure all cookies are set
        print("\n✓ Step 5.5: Navigating to main site to ensure all cookies are set...")
        try:
            from utils import nav_to_page
            if sm.driver:
                nav_to_page(sm.driver, config_schema.api.base_url, selector="body", session_manager=sm)
                time.sleep(3)
                print("  ✅ Navigation complete")
        except Exception as nav_err:
            print(f"  ⚠️  Navigation error (non-critical): {nav_err}")

        # Step 6: Save cookies
        print("\n✓ Step 6: Saving cookies to ancestry_cookies.json...")
        if not _save_login_cookies(sm):
            print("  ❌ Failed to save cookies")
            return False
        print("  ✅ Cookies saved successfully")
        
        # Step 7: Verify cookies file
        cookies_file = Path("ancestry_cookies.json")
        if cookies_file.exists():
            file_size = cookies_file.stat().st_size
            print(f"  ✅ Cookies file verified: {file_size} bytes")
        else:
            print("  ❌ Cookies file not found after save")
            return False
        
        # Step 8: Close session
        print("\n✓ Step 8: Closing session...")
        sm.close_sess(keep_db=False)
        print("  ✅ Session closed")
        
        print("\n" + "=" * 80)
        print("✅ SESSION COOKIES REFRESHED SUCCESSFULLY")
        print("=" * 80)
        print("\nYou can now run Phase 2 tests with valid cookies:")
        print("  python phase2_test_runner.py")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = refresh_cookies()
    sys.exit(0 if success else 1)

