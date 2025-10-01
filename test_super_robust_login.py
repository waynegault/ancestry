"""Super robust login test with multiple Next button strategies."""

import sys
sys.path.insert(0, '.')

from core.session_manager import SessionManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.keys import Keys
from my_selectors import *
from utils import nav_to_page, consent, login_status
from config import config_schema
from urllib.parse import urljoin
import time
import random

print("=" * 70)
print("SUPER ROBUST LOGIN TEST - MULTIPLE NEXT BUTTON STRATEGIES")
print("=" * 70)

# Create session manager
print("\n[1/6] Creating SessionManager...")
session_manager = SessionManager()
print("✅ SessionManager created")

# Start browser
print("\n[2/6] Starting browser...")
browser_started = session_manager.start_browser("Super Robust Login Test")

if browser_started:
    driver = session_manager.driver
    print("✅ Browser started successfully")
    
    # Navigate to login page
    print("\n[3/6] Navigating to login page...")
    signin_url = urljoin(config_schema.api.base_url, "account/signin")
    if nav_to_page(driver, signin_url, USERNAME_INPUT_SELECTOR, session_manager):
        print("✅ Navigated to login page")
        
        # Accept cookies
        print("\n[4/6] Accepting cookies...")
        if consent(driver):
            print("✅ Cookies accepted")
        else:
            print("⚠️  Cookie consent failed, continuing anyway")
        
        # Wait extra time for page to fully load
        print("\n[5/6] Waiting for page to fully stabilize...")
        time.sleep(3)
        
        try:
            # Step 1: Find and enter username
            print("\n  Step 1: Finding username field...")
            username_input = WebDriverWait(driver, 15).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR))
            )
            print("  ✅ Username field found")
            
            print("  Step 2: Clearing and entering username...")
            username_input.clear()
            time.sleep(1)  # Longer wait
            username_input.send_keys(config_schema.api.username)
            print(f"  ✅ Username entered: {config_schema.api.username}")
            time.sleep(2)  # Wait longer after entering username
            
            # Step 2: Multiple strategies for Next button
            print("\n  Step 3: MULTIPLE STRATEGIES FOR NEXT BUTTON...")
            next_clicked = False
            
            # Strategy 1: Standard approach
            print("    Strategy 1: Standard Next button click...")
            try:
                next_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR))
                )
                print(f"    ✅ Next button found: '{next_button.text}'")
                driver.execute_script("arguments[0].click();", next_button)
                print("    ✅ Next button clicked via JavaScript")
                next_clicked = True
            except Exception as e:
                print(f"    ❌ Strategy 1 failed: {e}")
            
            # Strategy 2: Try Enter key on username field
            if not next_clicked:
                print("    Strategy 2: Pressing Enter on username field...")
                try:
                    username_input.send_keys(Keys.RETURN)
                    print("    ✅ Enter key sent to username field")
                    next_clicked = True
                except Exception as e:
                    print(f"    ❌ Strategy 2 failed: {e}")
            
            # Strategy 3: Look for different button selectors
            if not next_clicked:
                print("    Strategy 3: Trying alternative button selectors...")
                alt_selectors = [
                    "button[type='submit']",
                    "input[type='submit']", 
                    "button:contains('Next')",
                    ".btn-primary",
                    "[data-testid='signin-button']"
                ]
                
                for selector in alt_selectors:
                    try:
                        if ":contains(" in selector:
                            continue  # Skip contains selectors
                        button = WebDriverWait(driver, 3).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        print(f"    ✅ Alternative button found: {selector}")
                        driver.execute_script("arguments[0].click();", button)
                        print(f"    ✅ Alternative button clicked")
                        next_clicked = True
                        break
                    except:
                        continue
            
            # Strategy 4: Tab and Enter
            if not next_clicked:
                print("    Strategy 4: Tab to next element and press Enter...")
                try:
                    username_input.send_keys(Keys.TAB)
                    time.sleep(0.5)
                    driver.switch_to.active_element.send_keys(Keys.RETURN)
                    print("    ✅ Tab + Enter executed")
                    next_clicked = True
                except Exception as e:
                    print(f"    ❌ Strategy 4 failed: {e}")
            
            if next_clicked:
                print("  ✅ Next button successfully activated!")
                
                # Wait longer for password field
                print("  Step 4: Waiting for password field to appear...")
                time.sleep(5)  # Longer wait
                
                password_found = False
                for attempt in range(5):  # More attempts
                    try:
                        password_input = WebDriverWait(driver, 8).until(
                            EC.visibility_of_element_located((By.CSS_SELECTOR, PASSWORD_INPUT_SELECTOR))
                        )
                        print("  ✅ Password field found!")
                        password_found = True
                        break
                    except TimeoutException:
                        print(f"    Attempt {attempt + 1}/5 failed, waiting 3 more seconds...")
                        time.sleep(3)
                
                if password_found:
                    # Continue with password entry...
                    print("\n  Step 5: Entering password...")
                    password_input.clear()
                    time.sleep(0.5)
                    password_input.send_keys(config_schema.api.password)
                    print("  ✅ Password entered")
                    time.sleep(1)
                    
                    # Click Sign In
                    print("\n  Step 6: Clicking Sign In...")
                    signin_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR))
                    )
                    driver.execute_script("arguments[0].click();", signin_button)
                    print("  ✅ Sign In clicked")
                    
                    # Wait and check for 2FA
                    time.sleep(5)
                    try:
                        sms_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, TWO_FA_SMS_SELECTOR))
                        )
                        print("  ✅ 2FA page reached! SMS button found")
                        driver.execute_script("arguments[0].click();", sms_button)
                        print("  ✅ SMS button clicked!")
                        
                        print("\n🎉 SUCCESS! Please enter 2FA code in browser...")
                        print("Waiting 120 seconds for completion...")
                        
                        # Check for login success
                        for i in range(24):
                            time.sleep(5)
                            status = login_status(session_manager, disable_ui_fallback=True)
                            if status is True:
                                print(f"\n🎉 LOGIN COMPLETE! (after {(i+1)*5} seconds)")
                                break
                            elif i % 4 == 0:
                                print(f"  Still waiting... ({(i+1)*5}s elapsed)")
                        
                    except TimeoutException:
                        print("  ❌ 2FA page not found")
                        
                else:
                    print("  ❌ Password field never appeared")
            else:
                print("  ❌ ALL NEXT BUTTON STRATEGIES FAILED")
                print("  Current URL:", driver.current_url)
                print("  Taking screenshot...")
                driver.save_screenshot("next_button_failure.png")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            
        print("\n[6/6] Browser open for 30 seconds...")
        time.sleep(30)
        
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
