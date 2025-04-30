#!/usr/bin/env python3

# chromedriver.py

"""
chromedriver.py - ChromeDriver management and configuration utility

Features:
- Safe Chrome process cleanup and initialization
- Chrome preferences management
- Connection pool configuration for Selenium
- Retry logic for driver initialization
- Window management integration
- Enhanced logging and error handling
- Headless mode support via .env configuration
- User-Agent spoofing for enhanced privacy
- Robust process cleanup for Chrome and ChromeDriver
- Chrome profile management via config
- Custom retry and timeout settings for Selenium connections

https://googlechromelabs.github.io/chrome-for-testing/#stable
"""

import os
import json
import time
import subprocess
from tkinter import N
import urllib3
import psutil
from urllib3.util import Retry
import logging
import random
import undetected_chromedriver as uc
from selenium import webdriver  # Standard Selenium WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.remote_connection import RemoteConnection
from selenium.common.exceptions import (
    WebDriverException,
    NoSuchElementException,
    TimeoutException,
    NoSuchWindowException,
)
from dotenv import load_dotenv
from my_selectors import CONFIRMED_LOGGED_IN_SELECTOR
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webdriver import WebDriver
from typing import Optional
import subprocess
from config import config_instance, selenium_config
from logging_config import setup_logging, logger

logger = logging.getLogger("logger")

# Define constants dependent on the CHROME_CONFIG values
CHROME_USER_DATA_DIR = selenium_config.CHROME_USER_DATA_DIR  # Use selenium_config
DEFAULT_PROFILE_PATH = os.path.join(CHROME_USER_DATA_DIR, "Default")
PREFERENCES_FILE = os.path.join(DEFAULT_PROFILE_PATH, "Preferences")

# --------------------------
# Chrome Configuration
# --------------------------


def reset_preferences_file():
    """Replace Chrome Preferences file with controlled configuration."""
    try:
        # Create the directory if it does not exist
        os.makedirs(DEFAULT_PROFILE_PATH, exist_ok=True)
        minimal_preferences = {
            "profile": {"exit_type": "Normal", "exited_cleanly": True},
            "browser": {
                "has_seen_welcome_page": True,
                "window_placement": {
                    "bottom": 1192,
                    "left": 1409,
                    "maximized": False,
                    "right": 2200,
                    "top": 0,
                    "work_area_bottom": 1192,
                    "work_area_left": 1409,
                    "work_area_right": 2200,
                    "work_area_top": 0,
                },
            },
            "privacy_sandbox": {
                "m1": {
                    "ad_measurement_enabled": False,
                    "consent_decision_made": True,
                    "eea_notice_acknowledged": True,
                    "fledge_enabled": False,
                    "topics_enabled": False,
                }
            },
            "sync": {"allowed": False},
            "extensions": {"alerts": {"initialized": True}},
            "session": {"restore_on_startup": 4, "startup_urls": []},
        }
        try:
            with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
                json.dump(minimal_preferences, f, indent=2)
        except IOError as e:
            logger.error(f"IOError writing Preferences file: {e}", exc_info=True)
            raise
    except OSError as e:
        logger.error(f"OSError creating directory: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in reset_preferences_file: {e}", exc_info=True)
        raise
# End of reset_preferences_file


def set_win_size(driver):
    """Set the window size and position to the right half of the screen, 95% height."""
    try:
        # Get screen dimensions
        screen_width = driver.execute_script("return screen.width;")
        screen_height = driver.execute_script("return screen.height;")

        # Calculate window size and position for the right half of the screen
        window_width = screen_width // 2
        window_height = int(
            screen_height * 0.965
        )  # % of screen height (using 96.5% as per prev version)
        window_x = screen_width // 2  # Position on the right
        window_y = 0  # Position at the top

        driver.set_window_rect(
            x=window_x, y=window_y, width=window_width, height=window_height
        )
    except Exception as e:
        logger.error(f"Failed to set window size and position: {e}", exc_info=True)
# End of set_win_size


def close_tabs(driver):
    """Closes all but the first tab in the given driver."""
    logger.debug("Closing extra tabs...")
    try:
        while len(driver.window_handles) > 1:
            driver.switch_to.window(driver.window_handles[-1])
            driver.close()
        driver.switch_to.window(
            driver.window_handles[0]
        )  # Switch back to the first tab
        logger.debug("Switched back to the original tab.")
    except NoSuchWindowException:
        logger.warning("Attempted to close or switch to a tab that no longer exists.")
    except Exception as e:
        logger.error(f"Error in close_tabs: {e}", exc_info=True)
# end close_tabs


def init_webdvr(attach_attempt=False) -> Optional[WebDriver]:
    """
    V1.3 REVISED: Added specific TypeError handling around uc.Chrome call.
    Initializes undetected_chromedriver and minimizes the window if not headless.
    """
    config = selenium_config  # Use selenium_config instance

    # --- 1. Pre-Initialization Cleanup ---
    cleanup_webdrv()  # Kill existing processes
    reset_preferences_file()  # Reset Chrome preferences

    # --- Retry Loop ---
    max_init_retries = config.CHROME_MAX_RETRIES
    retry_delay = config.CHROME_RETRY_DELAY
    driver = None

    for attempt_num in range(1, max_init_retries + 1):
        logger.debug(
            f"WebDriver initialization attempt {attempt_num}/{max_init_retries}..."
        )

        # --- Create FRESH Options object INSIDE the loop ---
        options = uc.ChromeOptions()

        # --- Configure Options ---
        if config.HEADLESS_MODE:
            logger.info("Configuring headless mode.")
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
        user_data_dir_path = config.CHROME_USER_DATA_DIR
        if user_data_dir_path:
            user_data_dir_str = str(user_data_dir_path.resolve())
            options.add_argument(f"--user-data-dir={user_data_dir_str}")
            logger.info(f"User data directory (no --profile-directory):\n{user_data_dir_str}")
        # Removed --profile-directory option for correct Chrome profile persistence
        # profile_dir_str = config.PROFILE_DIR
        # if profile_dir_str:
        #     options.add_argument(f"--profile-directory={profile_dir_str}")
        #     logger.debug(f"Using profile directory: {profile_dir_str}")
        browser_path_obj = config.CHROME_BROWSER_PATH
        if browser_path_obj:
            browser_path_str = str(browser_path_obj.resolve())
            if os.path.exists(browser_path_str):
                options.binary_location = browser_path_str
                logger.debug(f"Using browser executable:\n{browser_path_str}")
            else:
                logger.warning(
                    f"Specified browser path not found: {browser_path_str}. Relying on system default."
                )
        else:
            logger.debug("No explicit browser path specified, using system default.")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins-discovery")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-infobars")
        if not config.HEADLESS_MODE:
            options.add_argument("--start-maximized")
        user_agent = random.choice(config_instance.USER_AGENTS)
        options.add_argument(f"--user-agent={user_agent}")
        logger.debug(f"Setting User-Agent:\n{user_agent}")
        options.add_argument("--disable-popup-blocking")
        # --- End Configure Options ---

        # --- Attempt Driver Initialization ---
        try:
            # Self-patching: Let undetected_chromedriver (uc) auto-manage ChromeDriver version.
            # Do not pass Service or executable_path unless overriding auto-management.
            chrome_kwargs = {"options": options}
            logger.debug("Letting undetected_chromedriver auto-manage ChromeDriver version (self-patching mode).")
            try:
                logger.info(f"[init_webdvr] Attempting uc.Chrome() self-patching (attempt {attempt_num})...")
                start_time = time.time()
                driver = uc.Chrome(**chrome_kwargs)
                logger.info(f"[init_webdvr] uc.Chrome() self-patching succeeded in {time.time() - start_time:.2f}s (attempt {attempt_num})")
                logger.debug(
                    f"WebDriver instance object potentially created (attempt {attempt_num})."
                )  # Changed log slightly
            except Exception as uc_exc:
                logger.error(f"[init_webdvr] uc.Chrome() self-patching failed on attempt {attempt_num}: {uc_exc}", exc_info=True)
                if "cannot connect to chrome" in str(uc_exc).lower() or "chrome not reachable" in str(uc_exc).lower():
                    logger.warning("[init_webdvr] 'cannot connect to chrome':\n- Check for antivirus/firewall blocking Chrome or ChromeDriver.\n- Ensure Chrome is not crashing on startup (try launching manually with the same user data directory).\n- Check permissions for user data/profile directory.\n- Reinstall Chrome if necessary.")
                # Fallback: Try manual path if available
                driver_path_obj = config.CHROME_DRIVER_PATH
                if driver_path_obj:
                    driver_path_str = str(driver_path_obj.resolve())
                    if os.path.exists(driver_path_str):
                        logger.info(f"[init_webdvr] Falling back to manual ChromeDriver path: {driver_path_str}")
                        try:
                            from selenium.webdriver.chrome.service import Service
                            # FRESH ChromeOptions for fallback!
                            fallback_options = uc.ChromeOptions()
                            # Copy all arguments and settings from original options
                            for arg in options.arguments:
                                fallback_options.add_argument(arg)
                            fallback_options.binary_location = options.binary_location
                            chrome_kwargs_fallback = {"options": fallback_options, "service": Service(executable_path=driver_path_str)}
                            start_time_fallback = time.time()
                            driver = uc.Chrome(**chrome_kwargs_fallback)
                            logger.info(f"[init_webdvr] Fallback uc.Chrome() with manual path succeeded in {time.time() - start_time_fallback:.2f}s (attempt {attempt_num})")
                        except Exception as fallback_exc:
                            logger.error(f"[init_webdvr] Fallback uc.Chrome() with manual path also failed: {fallback_exc}", exc_info=True)
                            driver = None
                    else:
                        logger.error(f"[init_webdvr] Manual ChromeDriver path not found: {driver_path_str}. No further fallback possible.")
                        driver = None
                else:
                    logger.error("[init_webdvr] No manual ChromeDriver path configured. No further fallback possible.")
                    driver = None

            # Only proceed with driver setup if driver is not None
            if driver is not None:
                # Post-Initialization Settings (still within the inner try)
                try:
                    driver.execute_cdp_cmd(
                        "Network.setUserAgentOverride", {"userAgent": user_agent}
                    )
                    logger.debug("User-Agent override applied via CDP.")
                except Exception as cdp_exc:
                    logger.warning(f"CDP command failed: {cdp_exc}")

                if not config.HEADLESS_MODE:
                    logger.debug("Attempting to minimize window (non-headless mode)...")
                    try:
                        driver.minimize_window()
                        logger.debug("Browser window minimized.")
                    except WebDriverException as win_e:
                        logger.warning(f"Could not minimize window: {win_e}")
                    except Exception as min_e:
                        logger.error(
                            f"Unexpected error minimizing window: {min_e}",
                            exc_info=True,
                        )

                driver.set_page_load_timeout(config.PAGE_TIMEOUT)
                driver.set_script_timeout(config.ASYNC_SCRIPT_TIMEOUT)

                # Check for extra tabs immediately after init
                try:
                    if len(driver.window_handles) > 1:
                        logger.debug(
                            f"Multiple tabs ({len(driver.window_handles)}) detected immediately after init. Closing extras."
                        )
                        close_tabs(driver)
                except Exception as tab_check_e:
                    logger.warning(
                        f"Error checking/closing tabs immediately after init: {tab_check_e}"
                    )

                logger.debug(
                    f"WebDriver instance fully configured successfully (attempt {attempt_num})."
                )
                return driver  # SUCCESS!


        # --- Handle Specific Exceptions During Outer Initialization Attempt ---
        except TimeoutException as e:
            logger.warning(f"Timeout during WebDriver init attempt {attempt_num}: {e}")
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass
            driver = None
        except WebDriverException as e:  # Catches errors before/during uc.Chrome call
            err_str = str(e).lower()
            if "cannot connect to chrome" in err_str or "failed to start" in err_str:
                logger.error(
                    f"Failed to connect/start Chrome (attempt {attempt_num}): {e}"
                )
            elif "version mismatch" in err_str:
                logger.error(
                    f"ChromeDriver/Chrome version mismatch (attempt {attempt_num}): {e}."
                )
            elif "service" in err_str and "exited" in err_str:
                logger.error(
                    f"Service executable exited unexpectedly (Attempt {attempt_num}): {e}"
                )
            else:
                logger.warning(
                    f"WebDriverException during init attempt {attempt_num}: {e}"
                )
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass
            driver = None
        except Exception as e:  # Catch-all for other unexpected errors
            logger.error(
                f"Unexpected error during WebDriver init attempt {attempt_num}: {e}",
                exc_info=True,
            )
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass
            driver = None

        # --- Wait Before Retrying ---
        if attempt_num < max_init_retries:
            logger.info(
                f"Waiting {retry_delay} seconds before retrying initialization..."
            )
            time.sleep(retry_delay)
        else:
            logger.critical(
                f"Failed to initialize WebDriver after {max_init_retries} attempts."
            )
            return None

    logger.error("Exited WebDriver initialization loop unexpectedly.")
    return None
# End of init_webdvr


def cleanup_webdrv():
    """
    Cleans up any leftover chromedriver processes.  Important for preventing
    orphaned processes and port conflicts.
    """
    try:
        # Kill all Chrome processes.
        if os.name == "nt":  # Windows
            subprocess.run(
                ["taskkill", "/F", "/IM", "chromedriver.exe", "/T"],
                check=False,
                capture_output=True,
            )
            process = subprocess.run(
                ["taskkill", "/f", "/im", "chrome.exe"], capture_output=True, text=True
            )
            if process.returncode == 0:
                logger.debug(
                    f"Cleaned {process.stdout.count('SUCCESS')} chrome processes."
                )
        else:  # Linux/macOS
            # pkill is more reliable than killall on some systems.
            subprocess.run(["pkill", "-f", "chromedriver"], check=False)
            subprocess.run(["pkill", "-f", "chrome"], check=False)  # Kill Chrome itself

    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)
# end of cleanup_webdr


# ------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------


def main():
    """Main function for standalone use (debugging)."""
    setup_logging(log_level="DEBUG")
    try:
        driver = init_webdvr()
        if driver:
            logger.info("WebDriver initialized successfully.")
            driver.get(config_instance.BASE_URL)
            input("Press Enter to close the browser...")
            driver.quit()
    except Exception as e:
        print(f"An error occurred: {e}")


# end main

if __name__ == "__main__":
    main()
