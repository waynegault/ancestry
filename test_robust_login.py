"""Robust login test with detailed step-by-step debugging."""

import sys
sys.path.insert(0, '.')

from core.session_manager import SessionManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from my_selectors import *
from utils import nav_to_page, consent
from config import config_schema
from urllib.parse import urljoin
import time
import random

print("=" * 70)
print("ROBUST LOGIN TEST WITH STEP-BY-STEP DEBUGGING")
print("=" * 70)

# Create session manager
print("\n[1/6] Creating SessionManager...")
session_manager = SessionManager()
print("✅ SessionManager created")

# Start browser
print("\n[2/6] Starting browser...")
browser_started = session_manager.start_browser("Robust Login Test")

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
        
        # Step-by-step credential entry
        print("\n[5/6] Step-by-step credential entry...")
        
        try:
            # Step 1: Find and enter username
            print("\n  Step 1: Finding username field...")
            username_input = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR))
            )
            print("  ✅ Username field found")
            
            print("  Step 2: Clearing and entering username...")
            username_input.clear()
            time.sleep(0.5)
            username_input.send_keys(config_schema.api.username)
            print(f"  ✅ Username entered: {config_schema.api.username}")
            time.sleep(1)
            
            # Step 2: Click Next button
            print("\n  Step 3: Looking for Next button...")
            try:
                next_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR))
                )
                print(f"  ✅ Next button found: '{next_button.text}'")
                
                print("  Step 4: Clicking Next button...")
                driver.execute_script("arguments[0].click();", next_button)
                print("  ✅ Next button clicked via JavaScript")
                
                # Wait longer for password field
                print("  Step 5: Waiting for password field to appear...")
                time.sleep(3)  # Fixed 3-second wait
                
                # Try to find password field with multiple attempts
                password_found = False
                for attempt in range(3):
                    print(f"    Attempt {attempt + 1}/3 to find password field...")
                    try:
                        password_input = WebDriverWait(driver, 10).until(
                            EC.visibility_of_element_located((By.CSS_SELECTOR, PASSWORD_INPUT_SELECTOR))
                        )
                        print("  ✅ Password field found!")
                        password_found = True
                        break
                    except TimeoutException:
                        print(f"    ❌ Attempt {attempt + 1} failed, waiting 2 more seconds...")
                        time.sleep(2)
                
                if not password_found:
                    print("  ❌ Password field never appeared after 3 attempts")
                    print(f"  Current URL: {driver.current_url}")
                    print("  Taking screenshot for debugging...")
                    driver.save_screenshot("password_field_timeout.png")
                    print("  Screenshot saved as password_field_timeout.png")
                else:
                    # Step 3: Enter password
                    print("\n  Step 6: Entering password...")
                    password_input.clear()
                    time.sleep(0.5)
                    password_input.send_keys(config_schema.api.password)
                    print("  ✅ Password entered")
                    time.sleep(1)
                    
                    # Step 4: Click Sign In
                    print("\n  Step 7: Looking for Sign In button...")
                    try:
                        signin_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR))
                        )
                        print(f"  ✅ Sign In button found: '{signin_button.text}'")
                        
                        print("  Step 8: Clicking Sign In button...")
                        driver.execute_script("arguments[0].click();", signin_button)
                        print("  ✅ Sign In button clicked")
                        
                        # Wait for page change
                        print("\n  Step 9: Waiting for page change...")
                        time.sleep(5)
                        
                        # Check for 2FA page
                        print("  Step 10: Checking for 2FA page...")
                        try:
                            sms_button = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, TWO_FA_SMS_SELECTOR))
                            )
                            print("  ✅ 2FA page detected! SMS button found")
                            
                            print("  Step 11: Clicking SMS button...")
                            driver.execute_script("arguments[0].click();", sms_button)
                            print("  ✅ SMS button clicked!")
                            
                            print("\n" + "=" * 70)
                            print("🎉 SUCCESS! SMS should be sent to your phone!")
                            print("You have 120 seconds to enter the code...")
                            print("=" * 70)
                            
                            # Wait for user to enter 2FA code
                            time.sleep(120)
                            
                        except TimeoutException:
                            print("  ❌ 2FA page not detected")
                            print(f"  Current URL: {driver.current_url}")
                            print("  Page title:", driver.title)
                            
                    except TimeoutException:
                        print("  ❌ Sign In button not found")
                        
            except TimeoutException:
                print("  ❌ Next button not found")
                
        except Exception as e:
            print(f"❌ Error during credential entry: {e}")
            
        print("\n[6/6] Browser will remain open for 30 seconds for inspection...")
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
