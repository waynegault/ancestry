#!/usr/bin/env python3

# chromedriver.py - Updated Version

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
from selenium import webdriver
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
CHROME_USER_DATA_DIR = selenium_config.CHROME_USER_DATA_DIR
DEFAULT_PROFILE_PATH = os.path.join(CHROME_USER_DATA_DIR, "Default")
PREFERENCES_FILE = os.path.join(DEFAULT_PROFILE_PATH, "Preferences")

# --------------------------
# Chrome Configuration
# --------------------------

def reset_preferences_file():
    """Replace Chrome Preferences file with controlled configuration."""
    try:
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
        with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
            json.dump(minimal_preferences, f, indent=2)
    except Exception as e:
        logger.error(f"Unexpected error in reset_preferences_file: {e}", exc_info=True)
        raise

def set_win_size(driver):
    """Set the window size and position to the right half of the screen, 95% height."""
    try:
        screen_width = driver.execute_script("return screen.width;")
        screen_height = driver.execute_script("return screen.height;")
        window_width = screen_width // 2
        window_height = int(screen_height * 0.965)
        window_x = screen_width // 2
        window_y = 0
        driver.set_window_rect(
            x=window_x, y=window_y, width=window_width, height=window_height
        )
    except Exception as e:
        logger.error(f"Failed to set window size and position: {e}", exc_info=True)

def close_tabs(driver):
    """Closes all but the first tab in the given driver."""
    logger.debug("Closing extra tabs...")
    try:
        while len(driver.window_handles) > 1:
            driver.switch_to.window(driver.window_handles[-1])
            driver.close()
        driver.switch_to.window(driver.window_handles[0])
    except NoSuchWindowException:
        logger.warning("Attempted to close or switch to a tab that no longer exists.")
    except Exception as e:
        logger.error(f"Error in close_tabs: {e}", exc_info=True)

def init_webdvr(attach_attempt=False) -> Optional[WebDriver]:
    """
    V1.3 FIXED: Automatic ChromeDriver version management
    """
    config = selenium_config
    cleanup_webdrv()
    reset_preferences_file()

    max_init_retries = config.CHROME_MAX_RETRIES
    retry_delay = config.CHROME_RETRY_DELAY
    driver = None

    for attempt_num in range(1, max_init_retries + 1):
        logger.debug(f"WebDriver initialization attempt {attempt_num}/{max_init_retries}...")

        options = uc.ChromeOptions()

        if config.HEADLESS_MODE:
            logger.info("Configuring headless mode.")
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")

        # Updated profile handling
        if config.CHROME_USER_DATA_DIR is None:
            raise ValueError("CHROME_USER_DATA_DIR is not set in selenium_config.")
        user_data_dir_path = str(config.CHROME_USER_DATA_DIR.resolve())
        profile_dir_str = config.PROFILE_DIR

        # Browser executable
        if browser_path_obj := config.CHROME_BROWSER_PATH:
            browser_path_str = str(browser_path_obj.resolve())
            if os.path.exists(browser_path_str):
                options.binary_location = browser_path_str

        # Stability & Stealth Options (preserved original arguments)
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins-discovery")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-infobars")
        if not config.HEADLESS_MODE:
            options.add_argument("--start-maximized")

        # User-Agent handling
        user_agent = random.choice(config_instance.USER_AGENTS)
        options.add_argument(f"--user-agent={user_agent}")

        try:
            # Updated driver initialization
            driver = uc.Chrome(
                user_data_dir=str(config.CHROME_USER_DATA_DIR.resolve()),
                profile_directory=config.PROFILE_DIR,
                options=options,
                version_main=135,  
                use_subprocess=True,
                headless=config.HEADLESS_MODE
            )

            # Enhanced User-Agent override
            driver.execute_cdp_cmd("Network.setUserAgentOverride", {
                "userAgent": user_agent,
                "userAgentMetadata": {
                    "brands": [
                        {"brand": "Not.A/Brand", "version": "99"},
                        {"brand": "Chromium", "version": "135"},
                        {"brand": "Google Chrome", "version": "135"}
                    ],
                    "platform": "Windows",
                    "platformVersion": "10.0.0",
                    "architecture": "x86",
                    "model": "",
                    "mobile": False
                },
            })

            if not config.HEADLESS_MODE:
                set_win_size(driver)
            driver.set_page_load_timeout(config.PAGE_TIMEOUT)
            driver.set_script_timeout(config.ASYNC_SCRIPT_TIMEOUT)
            
            if len(driver.window_handles) > 1:
                close_tabs(driver)
                
            return driver

        except Exception as e:
            logger.error(f"Attempt {attempt_num} failed: {e}", exc_info=True)
            if driver:
                driver.quit()
            if attempt_num < max_init_retries:
                time.sleep(retry_delay)
            else:
                logger.critical("Post-initialization failed on final attempt.")
                return None

def cleanup_webdrv():
    """Improved process cleanup"""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/IM", "chromedriver*", "/T"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True
            )
            subprocess.run(
                ["taskkill", "/F", "/IM", "chrome.exe", "/T"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True
            )
        else:
            subprocess.run(["pkill", "-f", "chromedriver"], check=False)
            subprocess.run(["pkill", "-f", "chrome"], check=False)
        time.sleep(1)
    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)

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

if __name__ == "__main__":
    main()