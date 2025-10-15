"""Test what _api_req returns for Match List API."""
import sys
from core.session_manager import SessionManager
from utils import _api_req
from urllib.parse import urljoin
from config import config_schema

def test_api_req_response():
    """Test what _api_req returns."""
    print("\n" + "="*60)
    print("TESTING _api_req RESPONSE TYPE")
    print("="*60)
    
    # Initialize session
    sm = SessionManager()
    success = sm.ensure_session_ready(action_name="action6b_gather - Setup", skip_csrf=True)
    
    if not success:
        print("✗ Session initialization failed")
        return
    
    print("✓ Session initialized")
    
    # Navigate to match list page
    from utils import nav_to_page
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
    
    # Get CSRF token
    csrf_token = sm.api_manager.get_csrf_token()
    print(f"\n✓ CSRF token: {csrf_token[:20]}...")
    
    # Build API URL
    url = urljoin(
        config_schema.api.base_url,
        f"discoveryui-matches/parents/list/api/matchList/{my_uuid}?currentPage=1",
    )
    
    # Build headers
    headers = {
        "X-CSRF-Token": csrf_token,
        "Accept": "application/json",
        "Referer": "https://www.ancestry.co.uk/discoveryui-matches/list/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "priority": "u=1, i",
    }
    
    print(f"\nCalling _api_req...")
    print(f"URL: {url}")

    # Manually sync cookies (like Action 6 does)
    print("\nManually syncing cookies...")
    browser_cookies = sm.driver.get_cookies()
    print(f"Browser cookies count: {len(browser_cookies)}")
    for cookie in browser_cookies:
        sm.api_manager.requests_session.cookies.set(
            cookie['name'],
            cookie['value'],
            domain=cookie.get('domain'),
            path=cookie.get('path')
        )
    print(f"✓ Cookies synced to requests session")

    # Call _api_req
    response = _api_req(
        url=url,
        driver=sm.driver,
        session_manager=sm,
        method="GET",
        headers=headers,
        use_csrf_token=False,
        api_description="Match List API",
        allow_redirects=True,
    )
    
    print(f"\n" + "="*60)
    print("RESPONSE ANALYSIS")
    print("="*60)
    print(f"Response type: {type(response)}")
    print(f"Response is None: {response is None}")
    print(f"Response is dict: {isinstance(response, dict)}")
    print(f"Response has .json(): {hasattr(response, 'json')}")
    
    if response is None:
        print("\n✗ Response is None!")
    elif isinstance(response, dict):
        print(f"\n✓ Response is dict")
        print(f"Keys: {list(response.keys())}")
        match_groups = response.get("matchGroups", [])
        print(f"matchGroups count: {len(match_groups)}")
        if match_groups:
            matches = match_groups[0].get("matches", [])
            print(f"Matches count: {len(matches)}")
    elif hasattr(response, 'json'):
        print(f"\n⚠ Response is Response object")
        print(f"Status code: {response.status_code}")
        print(f"Reason: {response.reason}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Location header: {response.headers.get('Location', 'N/A')}")
        print(f"Content length: {len(response.content)}")
        print(f"Content (first 200 chars): {response.text[:200]}")
        try:
            data = response.json()
            print(f"✓ Successfully parsed JSON")
            print(f"Type after .json(): {type(data)}")
            print(f"Keys: {list(data.keys())}")
            match_groups = data.get("matchGroups", [])
            print(f"matchGroups count: {len(match_groups)}")
            if match_groups:
                matches = match_groups[0].get("matches", [])
                print(f"Matches count: {len(matches)}")
        except Exception as e:
            print(f"✗ Failed to parse JSON: {e}")
    else:
        print(f"\n✗ Unknown response type: {type(response)}")
        print(f"Response: {str(response)[:200]}")
    
    # Cleanup
    sm.cleanup()
    print("\n" + "="*60)

if __name__ == "__main__":
    test_api_req_response()

