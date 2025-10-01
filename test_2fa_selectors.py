"""Test to check 2FA page selectors."""

import sys
sys.path.insert(0, '.')

from core.session_manager import SessionManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from my_selectors import *
from utils import nav_to_page, consent, enter_creds
from config import config_schema
from urllib.parse import urljoin
import time

print("=" * 70)
print("2FA SELECTORS TEST")
print("=" * 70)

# Create session manager
print("\n[1/4] Creating SessionManager...")
session_manager = SessionManager()
print("✅ SessionManager created")

# Start browser
print("\n[2/4] Starting browser...")
browser_started = session_manager.start_browser("2FA Selectors Test")

if browser_started:
    driver = session_manager.driver
    print("✅ Browser started successfully")
    
    # Navigate to login page
    print("\n[3/4] Navigating to login page...")
    signin_url = urljoin(config_schema.api.base_url, "account/signin")
    if nav_to_page(driver, signin_url, USERNAME_INPUT_SELECTOR, session_manager):
        print("✅ Navigated to login page")
        
        # Accept cookies
        print("\n[4/4] Accepting cookies...")
        if consent(driver):
            print("✅ Cookies accepted")
        else:
            print("⚠️  Cookie consent failed, continuing anyway")
        
        # Enter credentials to get to 2FA page
        print("\n[5/5] Entering credentials to reach 2FA page...")
        if enter_creds(driver):
            print("✅ Credentials entered successfully")
            
            # Wait for 2FA page
            print("\nWaiting 5 seconds for 2FA page to load...")
            time.sleep(5)
            
            print("\n" + "=" * 70)
            print("TESTING 2FA PAGE SELECTORS")
            print("=" * 70)
            
            # Test different selectors for 2FA detection
            selectors_to_test = [
                ("body.mfaPage h2.conTitle", "Original selector"),
                ("h1", "Any h1"),
                ("h2", "Any h2"),
                ("[data-method='sms']", "SMS button"),
                ("[data-method='email']", "Email button"),
                ("body.mfaPage", "Body with mfaPage class"),
                ("div:contains('Two-step verification')", "Div containing text"),
                ("*:contains('Two-step verification')", "Any element containing text"),
            ]
            
            for selector, description in selectors_to_test:
                print(f"\nTesting: {description}")
                print(f"  Selector: {selector}")
                try:
                    if ":contains(" in selector:
                        # Skip contains selectors as they're not standard CSS
                        print(f"  ⚠️  Skipping :contains selector (not standard CSS)")
                        continue
                    
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        print(f"  ✅ Found {len(elements)} element(s)")
                        for i, elem in enumerate(elements[:3]):  # Show first 3
                            try:
                                text = elem.text.strip()[:50]  # First 50 chars
                                print(f"    [{i+1}] Text: '{text}'")
                                print(f"    [{i+1}] Tag: {elem.tag_name}")
                                if elem.get_attribute("class"):
                                    print(f"    [{i+1}] Classes: {elem.get_attribute('class')}")
                            except:
                                print(f"    [{i+1}] Could not get element details")
                    else:
                        print(f"  ❌ Not found")
                except Exception as e:
                    print(f"  ❌ Error: {e}")
            
            # Check page source for debugging
            print(f"\n" + "=" * 70)
            print("PAGE INFO")
            print("=" * 70)
            print(f"Current URL: {driver.current_url}")
            print(f"Page title: {driver.title}")
            
            # Check body classes
            try:
                body = driver.find_element(By.TAG_NAME, "body")
                body_classes = body.get_attribute("class")
                print(f"Body classes: '{body_classes}'")
            except:
                print("Could not get body classes")
            
            print("\n" + "=" * 70)
            print("Browser will remain open for 60 seconds for inspection...")
            print("=" * 70)
            time.sleep(60)
            
        else:
            print("❌ Failed to enter credentials")
    else:
        print("❌ Failed to navigate to login page")
    
    # Cleanup
    print("\n[Cleanup] Closing session...")
    session_manager.close_sess()
    print("✅ Session closed")
else:
    print("❌ Failed to start browser")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
