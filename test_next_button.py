"""Debug test for Next button clicking."""

import sys
sys.path.insert(0, '.')

from core.session_manager import SessionManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from my_selectors import SIGN_IN_BUTTON_SELECTOR, USERNAME_INPUT_SELECTOR, CONSENT_ACCEPT_BUTTON_SELECTOR
from utils import nav_to_page, consent
from config import config_schema
from urllib.parse import urljoin
import time

print("=" * 70)
print("NEXT BUTTON DEBUG TEST")
print("=" * 70)

# Create session manager
print("\n[1/5] Creating SessionManager...")
session_manager = SessionManager()
print("✅ SessionManager created")

# Start browser
print("\n[2/5] Starting browser...")
browser_started = session_manager.start_browser("Next Button Test")

if browser_started:
    driver = session_manager.driver
    print("✅ Browser started successfully")
    
    # Navigate to login page
    print("\n[3/5] Navigating to login page...")
    signin_url = urljoin(config_schema.api.base_url, "account/signin")
    if nav_to_page(driver, signin_url, USERNAME_INPUT_SELECTOR, session_manager):
        print("✅ Navigated to login page")
        
        # Accept cookies
        print("\n[4/5] Accepting cookies...")
        if consent(driver):
            print("✅ Cookies accepted")
        else:
            print("⚠️  Cookie consent failed, continuing anyway")
        
        # Enter username
        print("\n[5/5] Entering username and testing Next button...")
        try:
            username_input = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR))
            )
            print(f"✅ Username field found")
            
            username_input.clear()
            username_input.send_keys(config_schema.api.username)
            print(f"✅ Username entered: {config_schema.api.username}")
            time.sleep(1)
            
            # Try to find the Next button with multiple selectors
            print("\n" + "=" * 70)
            print("TESTING DIFFERENT SELECTORS FOR NEXT BUTTON")
            print("=" * 70)
            
            selectors_to_try = [
                ("#signInBtn", "ID: signInBtn"),
                ("button#signInBtn", "button#signInBtn"),
                ("button[type='submit']", "button[type='submit']"),
                ("button:contains('Next')", "button containing 'Next'"),
                ("//button[text()='Next']", "XPath: button with text 'Next'"),
            ]
            
            for selector, description in selectors_to_try:
                print(f"\nTrying: {description}")
                print(f"  Selector: {selector}")
                try:
                    if selector.startswith("//"):
                        # XPath
                        button = WebDriverWait(driver, 3).until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                    else:
                        # CSS
                        button = WebDriverWait(driver, 3).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                    print(f"  ✅ Found! Button text: '{button.text}'")
                    print(f"  Attempting to click...")
                    
                    # Try JavaScript click
                    driver.execute_script("arguments[0].click();", button)
                    print(f"  ✅ CLICKED via JavaScript!")
                    
                    # Wait to see if password field appears
                    print(f"  Waiting 3 seconds to see if password field appears...")
                    time.sleep(3)
                    
                    try:
                        password_field = driver.find_element(By.CSS_SELECTOR, "input#password")
                        if password_field.is_displayed():
                            print(f"  ✅✅✅ SUCCESS! Password field appeared!")
                            break
                        else:
                            print(f"  ⚠️  Password field exists but not visible")
                    except:
                        print(f"  ❌ Password field did not appear")
                        print(f"  Current URL: {driver.current_url}")
                    
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
            
            print("\n" + "=" * 70)
            print("Browser will remain open for 60 seconds for inspection...")
            print("=" * 70)
            time.sleep(60)
            
        except Exception as e:
            print(f"❌ Error: {e}")
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

