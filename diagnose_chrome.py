#!/usr/bin/env python3
"""
Chrome/ChromeDriver Diagnostic Tool

Diagnoses common Chrome/ChromeDriver issues including:
- Chrome installation and version
- ChromeDriver installation and version
- Chrome profile corruption
- Running Chrome processes
- Chrome user data directory issues
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def check_chrome_installation() -> bool:
    """Check if Chrome is installed and get version."""
    print_header("Chrome Installation Check")

    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]

    for chrome_path in chrome_paths:
        if Path(chrome_path).exists():
            print(f"✓ Chrome found at: {chrome_path}")

            # Try to get Chrome version
            try:
                result = subprocess.run(
                    [chrome_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5, check=False
                )
                if result.returncode == 0:
                    print(f"✓ Chrome version: {result.stdout.strip()}")
                    return True
            except Exception as e:
                print(f"⚠ Could not get Chrome version: {e}")
                return True  # Chrome exists even if we can't get version

    print("✗ Chrome not found in standard locations")
    print("  Please install Google Chrome from https://www.google.com/chrome/")
    return False


def check_running_processes() -> dict:
    """Check for running Chrome/ChromeDriver processes."""
    print_header("Running Process Check")

    processes = {"chrome": [], "chromedriver": []}

    try:
        # Check for Chrome processes
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq chrome.exe", "/FO", "CSV"],
            capture_output=True,
            text=True,
            check=False
        )
        if "chrome.exe" in result.stdout:
            chrome_count = result.stdout.count("chrome.exe")
            print(f"⚠ Found {chrome_count} Chrome process(es) running")
            print("  Recommendation: Close all Chrome windows before running Action 6")
            processes["chrome"] = [f"chrome.exe ({chrome_count} instances)"]
        else:
            print("✓ No Chrome processes running")

        # Check for ChromeDriver processes
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq chromedriver.exe", "/FO", "CSV"],
            capture_output=True,
            text=True,
            check=False
        )
        if "chromedriver.exe" in result.stdout:
            driver_count = result.stdout.count("chromedriver.exe")
            print(f"⚠ Found {driver_count} ChromeDriver process(es) running")
            print("  Recommendation: Kill orphaned ChromeDriver processes")
            processes["chromedriver"] = [f"chromedriver.exe ({driver_count} instances)"]
        else:
            print("✓ No ChromeDriver processes running")

    except Exception as e:
        print(f"✗ Error checking processes: {e}")

    return processes


def check_chrome_profile() -> bool:
    """Check Chrome user data directory and profile."""
    print_header("Chrome Profile Check")

    # Load from .env
    from dotenv import load_dotenv
    load_dotenv()

    user_data_dir = os.getenv("CHROME_USER_DATA_DIR")
    if not user_data_dir:
        print("⚠ CHROME_USER_DATA_DIR not set in .env")
        return False

    user_data_path = Path(user_data_dir)
    if not user_data_path.exists():
        print(f"✗ Chrome user data directory not found: {user_data_dir}")
        return False

    print(f"✓ Chrome user data directory exists: {user_data_dir}")

    # Check Default profile
    default_profile = user_data_path / "Default"
    if not default_profile.exists():
        print("⚠ Default profile not found")
        return False

    print(f"✓ Default profile exists: {default_profile}")

    # Check Preferences file
    prefs_file = default_profile / "Preferences"
    if not prefs_file.exists():
        print("⚠ Preferences file not found")
        return False

    print(f"✓ Preferences file exists: {prefs_file}")

    # Try to read Preferences file
    try:
        with open(prefs_file, encoding='utf-8') as f:
            prefs = json.load(f)

        # Check for corruption indicators
        if "profile" in prefs:
            exit_type = prefs.get("profile", {}).get("exit_type", "Unknown")
            exited_cleanly = prefs.get("profile", {}).get("exited_cleanly", False)

            print(f"  Exit type: {exit_type}")
            print(f"  Exited cleanly: {exited_cleanly}")

            if exit_type != "Normal" or not exited_cleanly:
                print("⚠ Profile may be corrupted (abnormal exit detected)")
                print("  Recommendation: Delete/rename the Default profile folder")
                return False

        print("✓ Preferences file is valid JSON")
        return True

    except json.JSONDecodeError:
        print("✗ Preferences file is corrupted (invalid JSON)")
        print("  Recommendation: Delete/rename the Default profile folder")
        return False
    except Exception as e:
        print(f"✗ Error reading Preferences file: {e}")
        return False


def check_chromedriver() -> bool:
    """Check ChromeDriver installation."""
    print_header("ChromeDriver Check")

    from dotenv import load_dotenv
    load_dotenv()

    driver_path = os.getenv("CHROME_DRIVER_PATH")
    if driver_path and Path(driver_path).exists():
        print(f"✓ ChromeDriver found at: {driver_path}")
        return True

    print("⚠ ChromeDriver not found at configured path")
    print("  undetected-chromedriver will auto-download if needed")
    return True  # Not critical since UC auto-downloads


def provide_recommendations(processes: dict, profile_ok: bool) -> None:
    """Provide recommendations based on diagnostic results."""
    print_header("Recommendations")

    issues_found = False

    if processes["chrome"]:
        print("1. Close all Chrome browser windows:")
        print("   - Close Chrome manually, or")
        print("   - Run: taskkill /F /IM chrome.exe")
        issues_found = True

    if processes["chromedriver"]:
        print("2. Kill orphaned ChromeDriver processes:")
        print("   - Run: taskkill /F /IM chromedriver.exe")
        issues_found = True

    if not profile_ok:
        print("3. Fix Chrome profile corruption:")
        print("   - Close all Chrome windows")
        print("   - Rename the Default profile folder:")
        from dotenv import load_dotenv
        load_dotenv()
        user_data_dir = os.getenv("CHROME_USER_DATA_DIR", "")
        if user_data_dir:
            default_path = Path(user_data_dir) / "Default"
            backup_path = Path(user_data_dir) / "Default.backup"
            print(f"     From: {default_path}")
            print(f"     To:   {backup_path}")
        print("   - Chrome will create a fresh profile on next run")
        issues_found = True

    if not issues_found:
        print("✓ No obvious issues detected")
        print("\nIf Action 6 still fails, try:")
        print("1. Update Chrome to the latest version")
        print("2. Check Windows Defender/antivirus isn't blocking Chrome")
        print("3. Run main.py with elevated privileges (Run as Administrator)")


def main() -> int:
    """Run all diagnostic checks."""
    print("\n" + "=" * 80)
    print("  Chrome/ChromeDriver Diagnostic Tool")
    print("=" * 80)

    chrome_ok = check_chrome_installation()
    processes = check_running_processes()
    profile_ok = check_chrome_profile()
    chromedriver_ok = check_chromedriver()

    provide_recommendations(processes, profile_ok)

    print("\n" + "=" * 80)
    print("  Diagnostic Complete")
    print("=" * 80 + "\n")

    return 0 if (chrome_ok and profile_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

