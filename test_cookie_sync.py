"""Test cookie synchronization in the wider codebase."""
from core.session_manager import SessionManager
from utils import nav_to_page

def test_cookie_sync():
    """Test that cookies are properly synced after navigation."""
    print("\n" + "="*60)
    print("TESTING COOKIE SYNCHRONIZATION")
    print("="*60)
    
    # Initialize session
    sm = SessionManager()
    success = sm.ensure_session_ready(action_name="action6b_gather - Setup", skip_csrf=True)
    
    if not success:
        print("✗ Session initialization failed")
        return
    
    print("✓ Session initialized")

    # Navigate to match list page
    my_uuid = sm.get_my_uuid()
    match_list_url = f"https://www.ancestry.co.uk/discoveryui-matches/list/{my_uuid}"
    
    print(f"\nNavigating to: {match_list_url}")
    nav_success = nav_to_page(
        sm.driver,
        match_list_url,
        selector="ui-custom[type='match-entry']",
        session_manager=sm,
    )
    
    if not nav_success:
        print("✗ Navigation failed")
        sm.cleanup()
        return
    
    print("✓ Navigation successful")
    
    # Check browser cookies AFTER navigation
    print("\n--- AFTER NAVIGATION ---")
    browser_cookies = sm.driver.get_cookies()
    print(f"Browser cookies: {len(browser_cookies)}")
    browser_cookie_names = [c['name'] for c in browser_cookies]
    print(f"Browser cookie names: {browser_cookie_names[:10]}")
    
    # Check if CSRF cookie exists in browser
    csrf_cookie = next((c for c in browser_cookies if c['name'] == '_dnamatches-matchlistui-x-csrf-token'), None)
    if csrf_cookie:
        print(f"✓ CSRF cookie found in browser: {csrf_cookie['value'][:20]}...")
    else:
        print("✗ CSRF cookie NOT found in browser")
    
    # Skip checking requests cookies before sync (may have duplicates from initialization)
    
    # Manually trigger cookie sync (like _api_req does)
    print("\n--- MANUAL COOKIE SYNC ---")
    sync_success = sm.api_manager.sync_cookies_from_browser(sm.browser_manager)
    print(f"Cookie sync result: {sync_success}")
    
    # Check cookies in requests session AFTER sync
    req_cookies_after_sync = list(sm.api_manager.requests_session.cookies)
    print(f"Requests session cookies (after sync): {len(req_cookies_after_sync)}")

    # Check for duplicates AFTER sync
    cookie_names_after = [c.name for c in req_cookies_after_sync]
    duplicates_after = [name for name in set(cookie_names_after) if cookie_names_after.count(name) > 1]
    if duplicates_after:
        print(f"⚠ DUPLICATE COOKIES FOUND AFTER SYNC: {duplicates_after}")
        for dup_name in duplicates_after:
            dup_cookies = [c for c in req_cookies_after_sync if c.name == dup_name]
            print(f"  {dup_name}: {len(dup_cookies)} copies")
            for c in dup_cookies:
                print(f"    - domain={c.domain}, path={c.path}, value={c.value[:20]}...")

    # Check if CSRF cookie exists in requests session
    csrf_cookies = [c for c in req_cookies_after_sync if c.name == '_dnamatches-matchlistui-x-csrf-token']
    if csrf_cookies:
        print(f"✓ CSRF cookie found in requests session: {csrf_cookies[0].value[:20]}...")
    else:
        print("✗ CSRF cookie NOT found in requests session")
    
    # Cleanup
    sm.cleanup()
    print("\n" + "="*60)

if __name__ == "__main__":
    test_cookie_sync()

