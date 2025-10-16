#!/usr/bin/env python3
"""
Chrome/ChromeDriver Diagnostic Tool

Checks Chrome installation, ChromeDriver compatibility, profile status,
and running processes to diagnose browser initialization issues.
"""

import os
import subprocess
import sys
from pathlib import Path

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === IMPORTS ===
from config.config_manager import ConfigManager

config_manager = ConfigManager()
config_schema = config_manager.get_config()


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_chrome_installation() -> dict:
    """Check if Chrome is installed and get version."""
    print_section("Chrome Installation Check")

    result = {
        "installed": False,
        "version": None,
        "path": None,
        "issues": []
    }

    # Check configured Chrome path
    chrome_path = config_schema.selenium.chrome_browser_path
    print(f"Configured Chrome path: {chrome_path}")

    if chrome_path and os.path.exists(chrome_path):
        result["installed"] = True
        result["path"] = chrome_path
        print(f"✅ Chrome found at: {chrome_path}")

        # Try to get Chrome version
        try:
            version_result = subprocess.run(
                [chrome_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5, check=False
            )
            if version_result.returncode == 0:
                version = version_result.stdout.strip()
                result["version"] = version
                print(f"✅ Chrome version: {version}")
            else:
                result["issues"].append("Could not get Chrome version")
                print("⚠️  Could not get Chrome version")
        except Exception as e:
            result["issues"].append(f"Error getting Chrome version: {e}")
            print(f"⚠️  Error getting Chrome version: {e}")
    else:
        result["issues"].append(f"Chrome not found at configured path: {chrome_path}")
        print(f"❌ Chrome not found at: {chrome_path}")

    return result


def check_chrome_processes() -> dict:
    """Check for running Chrome/ChromeDriver processes."""
    print_section("Running Chrome Processes")

    result = {
        "chrome_running": False,
        "chromedriver_running": False,
        "chrome_count": 0,
        "chromedriver_count": 0,
        "issues": []
    }

    if os.name == "nt":  # Windows
        # Check Chrome processes
        try:
            chrome_result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq chrome.exe"],
                capture_output=True,
                text=True,
                timeout=5, check=False
            )
            if "chrome.exe" in chrome_result.stdout:
                result["chrome_running"] = True
                # Count processes
                count = chrome_result.stdout.count("chrome.exe")
                result["chrome_count"] = count
                print(f"⚠️  Chrome is running: {count} processes")
                result["issues"].append(f"Chrome already running ({count} processes) - may cause conflicts")
            else:
                print("✅ No Chrome processes running")
        except Exception as e:
            print(f"⚠️  Error checking Chrome processes: {e}")

        # Check ChromeDriver processes
        try:
            driver_result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq chromedriver.exe"],
                capture_output=True,
                text=True,
                timeout=5, check=False
            )
            if "chromedriver.exe" in driver_result.stdout:
                result["chromedriver_running"] = True
                count = driver_result.stdout.count("chromedriver.exe")
                result["chromedriver_count"] = count
                print(f"⚠️  ChromeDriver is running: {count} processes")
                result["issues"].append(f"ChromeDriver already running ({count} processes) - may cause conflicts")
            else:
                print("✅ No ChromeDriver processes running")
        except Exception as e:
            print(f"⚠️  Error checking ChromeDriver processes: {e}")

    return result


def check_chrome_profile() -> dict:
    """Check Chrome user data directory and profile status."""
    print_section("Chrome Profile Check")

    result = {
        "profile_exists": False,
        "profile_path": None,
        "profile_size_mb": 0,
        "issues": []
    }

    # Get user data directory from config
    user_data_dir = config_schema.selenium.chrome_user_data_dir
    print(f"User data directory: {user_data_dir}")

    if user_data_dir and os.path.exists(user_data_dir):
        result["profile_exists"] = True
        result["profile_path"] = user_data_dir
        print(f"✅ Profile directory exists: {user_data_dir}")

        # Check profile size
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(user_data_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        pass

            size_mb = total_size / (1024 * 1024)
            result["profile_size_mb"] = round(size_mb, 2)
            print(f"✅ Profile size: {size_mb:.2f} MB")

            if size_mb > 1000:
                result["issues"].append(f"Large profile size ({size_mb:.2f} MB) may cause issues")
                print("⚠️  Large profile size may cause performance issues")
        except Exception as e:
            print(f"⚠️  Error calculating profile size: {e}")

        # Check for lock files (indicates Chrome is running)
        lock_files = ["lockfile", "SingletonLock", "SingletonSocket", "SingletonCookie"]
        for lock_file in lock_files:
            lock_path = os.path.join(user_data_dir, lock_file)
            if os.path.exists(lock_path):
                result["issues"].append(f"Lock file exists: {lock_file} (Chrome may be running)")
                print(f"⚠️  Lock file found: {lock_file} (Chrome may be running)")
    else:
        result["issues"].append(f"Profile directory not found: {user_data_dir}")
        print(f"❌ Profile directory not found: {user_data_dir}")

    return result


def check_chromedriver_compatibility() -> dict:
    """Check ChromeDriver configuration and compatibility."""
    print_section("ChromeDriver Configuration")

    result = {
        "configured": False,
        "path": None,
        "issues": []
    }

    # Check configured ChromeDriver path
    driver_path = config_schema.selenium.chrome_driver_path
    print(f"Configured ChromeDriver path: {driver_path}")

    if driver_path:
        # Convert to string for comparison
        driver_path_str = str(driver_path)

        if driver_path_str.lower() == "auto":
            result["configured"] = True
            result["path"] = "auto"
            print("✅ ChromeDriver set to AUTO (automatic version management)")
        elif os.path.exists(driver_path):
            result["configured"] = True
            result["path"] = str(driver_path)
            print(f"✅ ChromeDriver found at: {driver_path}")
        else:
            result["issues"].append(f"ChromeDriver not found at: {driver_path}")
            print(f"❌ ChromeDriver not found at: {driver_path}")
    else:
        result["issues"].append("ChromeDriver path not configured")
        print("❌ ChromeDriver path not configured")

    return result


def check_selenium_config() -> dict:
    """Check Selenium configuration settings."""
    print_section("Selenium Configuration")

    result = {
        "headless_mode": config_schema.selenium.headless_mode,
        "page_load_timeout": config_schema.selenium.page_load_timeout,
        "max_retries": config_schema.selenium.chrome_max_retries,
        "issues": []
    }

    print(f"Headless mode: {result['headless_mode']}")
    print(f"Page load timeout: {result['page_load_timeout']}s")
    print(f"Max retries: {result['max_retries']}")

    if not result['headless_mode']:
        print("⚠️  Running in windowed mode (may have display issues)")
        result["issues"].append("Windowed mode enabled - consider headless for stability")
    else:
        print("✅ Running in headless mode")

    return result


def generate_recommendations(diagnostics: dict) -> list:
    """Generate recommendations based on diagnostic results."""
    print_section("Recommendations")

    recommendations = []

    # Check for running processes
    if diagnostics["processes"]["chrome_running"]:
        recommendations.append({
            "priority": "HIGH",
            "issue": f"Chrome is already running ({diagnostics['processes']['chrome_count']} processes)",
            "solution": "Close all Chrome instances before running Action 6",
            "command": "taskkill /F /IM chrome.exe /T"
        })

    if diagnostics["processes"]["chromedriver_running"]:
        recommendations.append({
            "priority": "HIGH",
            "issue": f"ChromeDriver is already running ({diagnostics['processes']['chromedriver_count']} processes)",
            "solution": "Kill all ChromeDriver processes",
            "command": "taskkill /F /IM chromedriver.exe /T"
        })

    # Check profile issues
    if diagnostics["profile"]["issues"]:
        for issue in diagnostics["profile"]["issues"]:
            if "lock file" in issue.lower():
                recommendations.append({
                    "priority": "HIGH",
                    "issue": issue,
                    "solution": "Close Chrome and delete lock files, or use a different profile",
                    "command": None
                })

    # Check headless mode
    if not diagnostics["config"]["headless_mode"]:
        recommendations.append({
            "priority": "MEDIUM",
            "issue": "Running in windowed mode",
            "solution": "Try headless mode for better stability",
            "command": "Set HEADLESS_MODE=True in .env"
        })

    # Print recommendations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority']}] {rec['issue']}")
            print(f"   Solution: {rec['solution']}")
            if rec['command']:
                print(f"   Command: {rec['command']}")
    else:
        print("\n✅ No critical issues found!")

    return recommendations


def main() -> bool:
    """Run comprehensive Chrome diagnostics."""
    print("\n" + "=" * 80)
    print("  CHROME/CHROMEDRIVER DIAGNOSTIC TOOL")
    print("=" * 80)

    diagnostics = {
        "chrome": check_chrome_installation(),
        "processes": check_chrome_processes(),
        "profile": check_chrome_profile(),
        "chromedriver": check_chromedriver_compatibility(),
        "config": check_selenium_config()
    }

    # Generate recommendations
    recommendations = generate_recommendations(diagnostics)

    # Summary
    print_section("Summary")

    total_issues = sum(len(d.get("issues", [])) for d in diagnostics.values())

    if total_issues == 0:
        print("✅ No issues detected - Chrome should initialize successfully")
        return True
    print(f"⚠️  {total_issues} potential issue(s) detected")
    print(f"   {len(recommendations)} recommendation(s) provided")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

