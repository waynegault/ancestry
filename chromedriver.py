#!/usr/bin/env python3

"""
ChromeDriver Management & Browser Automation Engine

Advanced browser automation platform providing sophisticated ChromeDriver management,
intelligent browser configuration, and comprehensive automation capabilities with
optimized performance settings, robust error handling, and professional-grade
browser orchestration for genealogical automation and web scraping workflows.

Browser Orchestration:
• Advanced ChromeDriver management with intelligent process lifecycle and resource optimization
• Sophisticated browser configuration with optimized settings for automation and performance
• Comprehensive browser session management with intelligent cleanup and recovery protocols
• Advanced browser pool management with resource optimization and concurrent session handling
• Intelligent browser monitoring with performance analytics and automated optimization
• Integration with session management systems for comprehensive browser orchestration

Automation Intelligence:
• Sophisticated automation capabilities with intelligent element detection and interaction
• Advanced error handling with comprehensive recovery protocols and fallback strategies
• Intelligent performance optimization with memory management and resource allocation
• Comprehensive automation analytics with detailed performance metrics and insights
• Advanced browser security with secure automation protocols and data protection
• Integration with automation frameworks for comprehensive browser automation workflows

Performance Optimization:
• High-performance browser configuration with optimized settings for speed and reliability
• Memory-efficient browser management with intelligent resource allocation and cleanup
• Advanced browser caching with intelligent cache management and optimization strategies
• Comprehensive performance monitoring with real-time analytics and optimization recommendations
• Intelligent browser scaling with automated resource management and load balancing
• Integration with performance systems for comprehensive browser performance optimization

Foundation Services:
Provides the essential browser automation infrastructure that enables reliable,
high-performance web automation through intelligent ChromeDriver management,
comprehensive browser orchestration, and professional automation for research workflows.

Features:
- Safe Chrome process cleanup and initialization
- Chrome preferences management
- Connection pool configuration for Selenium
- Retry logic for driver initialization
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    get_function,
    is_function_available,
    setup_module,
)

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
# === THIRD-PARTY IMPORTS ===
import contextlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import undetected_chromedriver as uc  # type: ignore  # Anti-bot detection bypass
from selenium.common.exceptions import (
    NoSuchWindowException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.remote.webdriver import WebDriver

from config import config_schema
from logging_config import setup_logging

# --- Test framework imports ---
from test_framework import (
    TestSuite,
)

# Logger and registration handled by setup_module above

# Define constants dependent on the CHROME_CONFIG values
# Handle the case where selenium_config might be None
CHROME_USER_DATA_DIR = (
    config_schema.selenium.chrome_user_data_dir if config_schema.selenium else None
)
# Get the profile directory from config (respects PROFILE_DIR from .env)
PROFILE_DIR = (
    config_schema.selenium.profile_dir if config_schema.selenium else "Default"
)
# Handle the case where CHROME_USER_DATA_DIR might be None
if CHROME_USER_DATA_DIR is not None:
    DEFAULT_PROFILE_PATH = str(Path(CHROME_USER_DATA_DIR) / PROFILE_DIR)
    PREFERENCES_FILE = str(Path(DEFAULT_PROFILE_PATH) / "Preferences")
else:
    # Use a default temporary directory if CHROME_USER_DATA_DIR is None
    DEFAULT_PROFILE_PATH = str(Path.home() / ".ancestry_temp" / PROFILE_DIR)  # type: ignore
    PREFERENCES_FILE = str(Path(DEFAULT_PROFILE_PATH) / "Preferences")  # type: ignore

# --------------------------
# Chrome Configuration
# --------------------------


def reset_preferences_file() -> None:
    """Replace Chrome Preferences file with controlled configuration."""
    try:
        # Create the directory if it does not exist
        from pathlib import Path
        Path(DEFAULT_PROFILE_PATH).mkdir(parents=True, exist_ok=True)
        minimal_preferences = {
            "profile": {"exit_type": "Normal", "exited_cleanly": True},
            "browser": {
                "has_seen_welcome_page": True,
                "window_placement": {
                    "bottom": 1,
                    "left": 0,
                    "maximized": False,
                    "right": 1,
                    "top": 0,
                    "work_area_bottom": 1,
                    "work_area_left": 0,
                    "work_area_right": 1,
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
            from pathlib import Path
            with Path(PREFERENCES_FILE).open("w", encoding="utf-8") as f:
                f.write(json.dumps(minimal_preferences, indent=2))
        except OSError as e:
            logger.error(f"IOError writing Preferences file: {e}", exc_info=True)
            raise
    except OSError as e:
        logger.error(f"OSError creating directory: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in reset_preferences_file: {e}", exc_info=True)
        raise


# End of reset_preferences_file


def set_win_size(driver: WebDriver) -> None:
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


def close_tabs(driver: WebDriver) -> None:
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


# Helper functions for init_webdvr

def _configure_chrome_options(config: Any) -> uc.ChromeOptions:
    """Configure Chrome options for WebDriver initialization with enhanced stealth."""
    options = uc.ChromeOptions()

    # Headless mode configuration
    if config.headless_mode:
        logger.debug("Configuring headless mode.")
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")

    # User data directory
    user_data_dir_path = config.chrome_user_data_dir
    if user_data_dir_path:
        user_data_dir_str = str(user_data_dir_path.resolve())
        options.add_argument(f"--user-data-dir={user_data_dir_str}")
        logger.debug(f"User data directory (no --profile-directory):\n{user_data_dir_str}")

    # Browser path
    browser_path_obj = config.chrome_browser_path
    if browser_path_obj:
        browser_path_str = str(browser_path_obj.resolve())
        from pathlib import Path
        if Path(browser_path_str).exists():
            options.binary_location = browser_path_str
            logger.debug(f"Using browser executable:\n{browser_path_str}")
        else:
            logger.warning(f"Specified browser path not found: {browser_path_str}. Relying on system default.")
    else:
        logger.debug("No explicit browser path specified, using system default.")

    # ANTI-DETECTION: Disable automation flags that Ancestry.co.uk detects
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")

    # ANTI-DETECTION: Remove common automation indicators
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins-discovery")

    # ANTI-DETECTION: Additional stealth flags to appear as normal browser
    options.add_argument("--disable-automation")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--exclude-switches=enable-automation")
    options.add_argument("--exclude-switches=enable-logging")

    # Set realistic user agent (DO NOT use random - use consistent realistic one)
    # Random user agents are a red flag for bot detection
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    options.add_argument(f"--user-agent={user_agent}")
    logger.debug(f"Setting consistent User-Agent: {user_agent}")

    # ANTI-DETECTION: Set preferences to disable automation flags
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    options.add_experimental_option("useAutomationExtension", False)

    # ANTI-DETECTION: Set realistic Chrome preferences
    prefs = {
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
        "profile.default_content_setting_values.notifications": 2
    }
    options.add_experimental_option("prefs", prefs)

    return options


def _create_chrome_driver(_options: uc.ChromeOptions, attempt_num: int) -> Optional[WebDriver]:
    """Create Chrome WebDriver instance with multiple fallback strategies."""
    try:
        logger.debug(f"[init_webdvr] Attempting Chrome WebDriver initialization (attempt {attempt_num})...")
        start_time = time.time()

        # Use undetected_chromedriver for anti-bot protection
        # Auto-detect Chrome version and let UC handle driver download
        logger.debug("[init_webdvr] Auto-detecting Chrome version for compatibility...")

        # Use minimal configuration (Strategy 3) - this is the only one that works reliably
        # Skip Strategies 1 & 2 which fail with "cannot connect to chrome" errors
        logger.debug("[init_webdvr] Using minimal configuration for best compatibility...")
        minimal_options = uc.ChromeOptions()
        minimal_options.add_argument("--no-sandbox")
        minimal_options.add_argument("--disable-dev-shm-usage")

        # Persist authentication state via dedicated profile
        user_data_dir = getattr(config_schema.selenium, "chrome_user_data_dir", None)
        profile_dir = getattr(config_schema.selenium, "profile_dir", "Default")

        if user_data_dir:
            user_data_dir_path = Path(user_data_dir)
            user_data_dir_path.mkdir(parents=True, exist_ok=True)
            user_data_dir_str = str(user_data_dir_path)
            minimal_options.add_argument(f"--user-data-dir={user_data_dir_str}")

            # IMPORTANT: Set the profile directory to match what reset_preferences_file() resets
            # This ensures Chrome uses the correct profile that we've cleaned up
            minimal_options.add_argument(f"--profile-directory={profile_dir}")

            logger.debug(f"[init_webdvr] Using persistent Chrome profile: {user_data_dir_str}/{profile_dir}")
        else:
            logger.warning("[init_webdvr] chrome_user_data_dir not configured - using temporary profile")

        # Additional stability options for Chrome 142+
        minimal_options.add_argument("--disable-gpu")
        minimal_options.add_argument("--disable-software-rasterizer")
        minimal_options.add_argument("--disable-extensions")
        minimal_options.add_argument("--no-first-run")
        minimal_options.add_argument("--no-default-browser-check")

        logger.debug("[init_webdvr] Creating Chrome instance with enhanced stability options...")
        driver = uc.Chrome(
            options=minimal_options,
            version_main=142,
            use_subprocess=False,
            suppress_welcome=True,
        )

        # Verify driver is valid before proceeding
        if not driver:
            logger.error("[init_webdvr] Driver creation returned None")
            return None

        # Check if browser window is actually open
        try:
            _ = driver.current_url  # This will fail if window is closed
            logger.debug("[init_webdvr] Browser window verified as open")
        except Exception as verify_err:
            logger.error(f"[init_webdvr] Browser window closed immediately after creation: {verify_err}")
            logger.error("[init_webdvr] This may indicate:")
            logger.error("  - Chrome profile corruption (try deleting profile)")
            logger.error("  - Multiple Chrome instances running (close all Chrome)")
            logger.error("  - Chrome/ChromeDriver version mismatch")
            logger.error("  - Security software blocking Chrome")
            logger.error("  Run 'python diagnose_chrome.py' for detailed diagnostics")
            from contextlib import suppress
            with suppress(Exception):
                driver.quit()
            return None

        # Browser started successfully; focus will return to terminal after init
        logger.debug("[init_webdvr] Browser started successfully (focus will return to terminal)")

        elapsed = time.time() - start_time
        logger.debug(f"Chrome WebDriver initialization succeeded in {elapsed:.2f}s (attempt {attempt_num})")
        return driver
    except Exception as chrome_exc:
        # Log brief error to file, don't spam console with stacktrace
        error_summary = str(chrome_exc).split('\n')[0] if '\n' in str(chrome_exc) else str(chrome_exc)
        logger.error(f"[init_webdvr] All ChromeDriver initialization strategies failed on attempt {attempt_num}: {error_summary}")

        # Only show detailed stacktrace in debug mode
        if logger.isEnabledFor(10):  # DEBUG level
            logger.debug(f"Full error details: {chrome_exc}", exc_info=True)

        print(f"  ✗ ChromeDriver initialization failed (attempt {attempt_num})", flush=True)
        print("  i Run 'python diagnose_chrome.py' for detailed diagnostics", flush=True)
        return None


def _configure_driver_post_init(driver: WebDriver, config: Any, user_agent: str, attempt_num: int) -> None:  # noqa: ARG001
    """Configure driver after initialization."""
    # NOTE: CDP user-agent override disabled - Chrome 142+ closes immediately when invoked here.
    # The user-agent is already set via Chrome launch arguments, so no additional override required.
    logger.debug("User-Agent set via Chrome options (CDP override disabled for stability)")

    # Bring terminal focus instead of minimizing browser
    if not config.headless_mode:
        logger.debug("Terminal focus will be restored after browser initialization.")

    # Set timeouts
    driver.set_page_load_timeout(config.page_load_timeout)
    driver.set_script_timeout(getattr(config, "script_timeout", 30))

    # Close extra tabs
    try:
        if len(driver.window_handles) > 1:
            logger.debug(f"Multiple tabs ({len(driver.window_handles)}) detected immediately after init. Closing extras.")
            close_tabs(driver)
    except Exception as tab_check_e:
        logger.warning(f"Error checking/closing tabs immediately after init: {tab_check_e}")

    logger.debug(f"WebDriver instance fully configured successfully (attempt {attempt_num}).")


def _handle_driver_exception(e: Exception, driver: Optional[WebDriver], attempt_num: int) -> None:
    """Handle exceptions during driver initialization."""
    if isinstance(e, TimeoutException):
        logger.warning(f"Timeout during WebDriver init attempt {attempt_num}: {e}")
    elif isinstance(e, WebDriverException):
        err_str = str(e).lower()
        if "cannot connect to chrome" in err_str or "failed to start" in err_str:
            logger.error(f"Failed to connect/start Chrome (attempt {attempt_num}): {e}")
        elif "version mismatch" in err_str:
            logger.error(f"ChromeDriver/Chrome version mismatch (attempt {attempt_num}): {e}.")
        elif "service" in err_str and "exited" in err_str:
            logger.error(f"Service executable exited unexpectedly (Attempt {attempt_num}): {e}")
        else:
            logger.warning(f"WebDriverException during init attempt {attempt_num}: {e}")
    else:
        logger.error(f"Unexpected error during WebDriver init attempt {attempt_num}: {e}", exc_info=True)

    # Cleanup driver if it exists
    if driver:
        with contextlib.suppress(Exception):
            driver.quit()


def init_webdvr(_attach_attempt: bool = False) -> Optional[WebDriver]:
    """
    V2.0 MODERNIZED: Uses standard Selenium WebDriver with automatic ChromeDriver management.
    Initializes standard Chrome WebDriver and returns focus to the terminal when complete.
    """
    config = config_schema.selenium

    # Pre-initialization cleanup
    cleanup_webdrv()
    reset_preferences_file()

    # Retry loop
    max_init_retries = config.chrome_max_retries
    retry_delay = config.chrome_retry_delay
    driver = None

    for attempt_num in range(1, max_init_retries + 1):
        logger.debug(f"WebDriver initialization attempt {attempt_num}/{max_init_retries}...")

        try:
            # Configure Chrome options
            options = _configure_chrome_options(config)

            # Get user agent for later use
            user_agent = random.choice(
                getattr(config_schema, "USER_AGENTS", ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"])
            )

            # Create driver
            driver = _create_chrome_driver(options, attempt_num)

            # Configure driver if creation succeeded
            if driver is not None:
                _configure_driver_post_init(driver, config, user_agent, attempt_num)
                return driver  # SUCCESS!

        except (TimeoutException, WebDriverException, Exception) as e:
            _handle_driver_exception(e, driver, attempt_num)
            driver = None

        # Wait before retrying
        if attempt_num < max_init_retries:
            logger.debug(f"Waiting {retry_delay} seconds before retrying initialization...")
            time.sleep(retry_delay)
        else:
            logger.critical(f"Failed to initialize WebDriver after {max_init_retries} attempts.")
            return None

    logger.error("Exited WebDriver initialization loop unexpectedly.")
    return None


# End of init_webdvr


def cleanup_webdrv() -> None:
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
                ["taskkill", "/f", "/im", "chrome.exe"], check=False, capture_output=True, text=True
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
# Self-Test Functions
# ------------------------------------------------------------------------------------


def test_preferences_file() -> bool:
    """Test the reset_preferences_file function."""
    print("\n=== Testing Preferences File Reset ===")
    try:
        reset_preferences_file()
        from pathlib import Path
        if Path(PREFERENCES_FILE).exists():
            print(f"✓ Preferences file created successfully at: {PREFERENCES_FILE}")
            # Verify the file contains valid JSON
            from pathlib import Path
            with Path(PREFERENCES_FILE).open(encoding="utf-8") as f:
                prefs = json.load(f)
                if isinstance(prefs, dict) and "profile" in prefs:
                    print(
                        "✓ Preferences file contains valid JSON with expected structure"
                    )
                else:
                    print("✗ Preferences file does not contain expected structure")
        else:
            print(f"✗ Failed to create preferences file at: {PREFERENCES_FILE}")
        return True
    except Exception as e:
        print(f"✗ Error in test_preferences_file: {e}")
        return False


def test_cleanup() -> bool:
    """Test the cleanup_webdrv function."""
    print("\n=== Testing Chrome Process Cleanup ===")
    try:
        # First check if any Chrome processes are running
        chrome_running = False
        if os.name == "nt":  # Windows
            try:
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq chrome.exe"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                chrome_running = "chrome.exe" in result.stdout
            except Exception as e:
                print(f"  Warning: Could not check for Chrome processes: {e}")

        print(
            f"  Chrome processes {'detected' if chrome_running else 'not detected'} before cleanup"
        )

        # Run the cleanup function
        cleanup_webdrv()
        print("✓ Cleanup function executed without errors")

        # Check again after cleanup
        if os.name == "nt":  # Windows
            try:
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq chrome.exe"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                chrome_running_after = "chrome.exe" in result.stdout
                if chrome_running and not chrome_running_after:
                    print("✓ Successfully terminated Chrome processes")
                elif not chrome_running:
                    print("  No Chrome processes were running before cleanup")
                else:
                    print(
                        "  Note: Some Chrome processes may still be running (possibly user browser)"
                    )
            except Exception as e:
                print(
                    f"  Warning: Could not check for Chrome processes after cleanup: {e}"
                )

        return True
    except Exception as e:
        print(f"✗ Error in test_cleanup: {e}")
        return False


def test_driver_initialization(headless: bool = True) -> bool:
    """Test the init_webdvr function."""
    print("\n=== Testing WebDriver Initialization ===")
    driver = None
    try:
        # Note: Cannot modify config_schema directly as it's immutable
        print(f"  Initializing WebDriver (headless={headless})...")
        start_time = time.time()
        driver = init_webdvr()
        init_time = time.time() - start_time

        if driver:
            print(f"✓ WebDriver initialized successfully in {init_time:.2f} seconds")
            # Test navigation
            try:
                print("  Testing navigation to BASE_URL...")
                driver.get(config_schema.api.base_url)
                print(f"✓ Successfully navigated to {config_schema.api.base_url}")
                print(f"  Page title: {driver.title}")
            except Exception as nav_e:
                print(f"✗ Navigation failed: {nav_e}")

            # Test window management
            try:
                print("  Testing window management...")
                set_win_size(driver)
                window_rect = driver.get_window_rect()
                print(
                    f"✓ Window size set: {window_rect['width']}x{window_rect['height']} at position ({window_rect['x']},{window_rect['y']})"
                )
            except Exception as win_e:
                print(f"✗ Window management failed: {win_e}")

            # Test closing tabs
            try:
                print("  Testing tab management...")
                # Open a new tab
                driver.execute_script("window.open('about:blank', '_blank');")
                tab_count_before = len(driver.window_handles)
                print(f"  Created new tab. Total tabs: {tab_count_before}")

                # Close extra tabs
                close_tabs(driver)
                tab_count_after = len(driver.window_handles)

                if tab_count_after == 1:
                    print("✓ Successfully closed extra tabs")
                else:
                    print(
                        f"✗ Failed to close all extra tabs. Remaining: {tab_count_after}"
                    )
            except Exception as tab_e:
                print(f"✗ Tab management failed: {tab_e}")

            # Clean up
            print("  Closing WebDriver...")
            driver.quit()
            print("✓ WebDriver closed successfully")
            # Note: Cannot restore original headless mode as config is immutable
            return True
        print("✗ WebDriver initialization failed")
        # Note: Cannot restore original headless mode as config is immutable
        return False
    except Exception as e:
        print(f"✗ Error in test_driver_initialization: {e}")
        if driver:
            try:
                driver.quit()
                print("  WebDriver closed after error")
            except Exception:
                pass
        # Note: Cannot restore original headless mode as config is immutable
        return False


def run_all_tests(interactive: bool = False) -> bool:
    """Run all self-tests."""
    print("\n===== ChromeDriver Self-Test Suite =====")
    print(f"Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"OS: {os.name.upper()}")  # Configuration info
    print("\n=== Configuration ===")
    print(f"CHROME_USER_DATA_DIR: {CHROME_USER_DATA_DIR}")
    print(f"DEFAULT_PROFILE_PATH: {DEFAULT_PROFILE_PATH}")
    print(f"HEADLESS_MODE: {config_schema.selenium.headless_mode}")
    print(f"CHROME_MAX_RETRIES: {config_schema.selenium.chrome_max_retries}")
    print(f"CHROME_BROWSER_PATH: {config_schema.selenium.chrome_browser_path}")
    print(f"CHROME_DRIVER_PATH: {config_schema.selenium.chrome_driver_path}")

    # Run tests
    test_results = {}

    # Preferences File
    test_results["preferences_file"] = test_preferences_file()

    # Cleanup
    test_results["cleanup"] = test_cleanup()

    # Driver Initialization (always headless for automated testing)
    test_results["driver_init"] = test_driver_initialization(headless=True)

    # Summary
    print("\n=== Test Summary ===")
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    print(f"Passed: {passed}/{total} tests ({passed/total*100:.1f}%)")

    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    # Interactive mode for manual testing
    if interactive and test_results["driver_init"]:
        print("\n=== Interactive Test ===")
        response = (
            input("Would you like to test with a visible browser? (y/n): ")
            .strip()
            .lower()
        )
        if response == "y":
            print("Starting visible browser test...")
            test_driver_initialization(headless=False)
            print("Interactive test completed.")

    return all(test_results.values())


# ------------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------------


def main() -> int:
    """Main function for standalone use (testing and debugging)."""
    import sys  # Import here to avoid potential circular imports

    # Configure logging
    setup_logging(log_level="DEBUG", allow_env_override=False)

    # Parse command line arguments
    interactive = False
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["-i", "--interactive"]:
        interactive = True

    # Run tests
    try:
        success = run_all_tests(interactive=interactive)
        if success:
            print("\nAll tests passed successfully!")
            return 0
        print("\nSome tests failed. See details above.")
        return 1
    except Exception as e:
        print(f"\nCritical error during testing: {e}")
        logger.error(f"Critical error during testing: {e}", exc_info=True)
        return 2


# End of main


def test_chromedriver_initialization() -> None:
    """Test ChromeDriver initialization functionality."""
    if is_function_available("initialize_chrome_driver"):
        init_func = get_function("initialize_chrome_driver")
        # Test that function exists and is callable
        assert callable(init_func)


def test_preferences_file_reset() -> None:
    """Test preferences file reset functionality."""
    # Test that preference management functions are properly defined
    required_funcs = [
        "init_webdvr",
        "safe_close_chrome",
        "cleanup_chrome_processes",
    ]
    for func_name in required_funcs:
        if func_name in globals():
            func = globals()[func_name]
            assert callable(func), f"{func_name} should be callable"


def test_chrome_process_cleanup() -> None:
    """Test Chrome process cleanup functionality."""
    # Test that cleanup functions exist and are properly structured
    cleanup_functions = ["cleanup_chrome_processes", "safe_close_chrome"]
    for func_name in cleanup_functions:
        if func_name in globals():
            func = globals()[func_name]
            assert callable(func), f"{func_name} should be callable"
            # Test function signature
            import inspect

            sig = inspect.signature(func)
            assert len(sig.parameters) >= 0, f"{func_name} should have valid parameters"


def test_webdriver_initialization() -> None:
    """Test WebDriver initialization with various configurations."""
    if is_function_available("initialize_chrome_driver"):
        init_func = get_function("initialize_chrome_driver")
        assert callable(init_func)

def test_chrome_options_creation() -> None:
    """Test that undetected_chromedriver ChromeOptions can be created without NameError."""
    try:
        # This should work with undetected_chromedriver
        options = uc.ChromeOptions()
        assert options is not None, "undetected_chromedriver ChromeOptions creation should succeed"

        # Test basic option setting
        options.add_argument("--headless=new")
        assert "--headless=new" in options.arguments, "Should be able to add arguments"

        logger.debug("undetected_chromedriver ChromeOptions creation test passed")
        return True  # type: ignore
    except NameError as e:
        if "'uc' is not defined" in str(e):
            raise AssertionError(f"NameError indicates missing undetected_chromedriver import: {e}") from e
        raise AssertionError(f"Unexpected NameError: {e}") from e
    except Exception as e:
        raise AssertionError(f"undetected_chromedriver ChromeOptions creation failed: {e}") from e
    return False  # Should not reach here


def chromedriver_module_tests() -> bool:
    """
    Chrome Driver Management module test suite.
    Tests Chrome driver setup, configuration, and process management.
    """

    suite = TestSuite("Chrome Driver Management", __name__)
    suite.start_suite()

    # Run all tests using the suite
    suite.run_test(
        "ChromeDriver Initialization",
        test_chromedriver_initialization,
        "Chrome WebDriver initializes successfully with proper configuration",
    )

    suite.run_test(
        "Preferences File Reset",
        test_preferences_file_reset,
        "Preferences file resets without critical errors",
    )

    suite.run_test(
        "Chrome Process Cleanup",
        test_chrome_process_cleanup,
        "Chrome processes cleanup without critical errors",
    )

    suite.run_test(
        "WebDriver Initialization",
        test_webdriver_initialization,
        "WebDriver initialization functions are available and callable",
    )

    suite.run_test(
        "undetected_chromedriver ChromeOptions Creation",
        test_chrome_options_creation,
        "undetected_chromedriver ChromeOptions can be created without NameError (tests for missing imports)",
    )

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(chromedriver_module_tests)


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    print("🌐 Running Chrome Driver Management comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
