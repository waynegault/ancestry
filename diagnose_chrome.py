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
import winreg
from pathlib import Path


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def get_chrome_version_from_registry() -> str | None:
    """Get Chrome version from Windows Registry."""
    try:
        # Try HKEY_CURRENT_USER first (user-specific installation)
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Google\Chrome\BLBeacon") as key:
            version, _ = winreg.QueryValueEx(key, "version")
            return version
    except (FileNotFoundError, OSError):
        pass
    
    try:
        # Try HKEY_LOCAL_MACHINE (system-wide installation)
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"Software\Google\Chrome\BLBeacon") as key:
            version, _ = winreg.QueryValueEx(key, "version")
            return version
    except (FileNotFoundError, OSError):
        pass
    
    try:
        # Try alternate registry path
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"Software\Wow6432Node\Google\Update\Clients\{8A69D345-D564-463c-AFF1-A69D9E530F96}") as key:
            version, _ = winreg.QueryValueEx(key, "pv")
            return version
    except (FileNotFoundError, OSError):
        pass
    
    return None


def check_chrome_installation() -> bool:
    """Check if Chrome is installed and get version."""
    print_header("Chrome Installation Check")

    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]

    chrome_found = False
    chrome_path_found = None

    for chrome_path in chrome_paths:
        if Path(chrome_path).exists():
            chrome_found = True
            chrome_path_found = chrome_path
            print(f"✓ Chrome found at: {chrome_path}")
            break

    if not chrome_found:
        print("✗ Chrome not found in standard locations")
        print("  Please install Google Chrome from https://www.google.com/chrome/")
        return False

    # Try to get Chrome version from registry first (most reliable)
    version = get_chrome_version_from_registry()
    if version:
        print(f"✓ Chrome version (from registry): {version}")
        return True

    # Fallback: Try to get version from executable
    if chrome_path_found:
        try:
            result = subprocess.run(
                [chrome_path_found, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"✓ Chrome version (from executable): {result.stdout.strip()}")
                return True
        except Exception as e:
            print(f"⚠ Could not get Chrome version from executable: {e}")

        # Fallback: Try to get version from version folder
        try:
            chrome_dir = Path(chrome_path_found).parent
            version_dirs = [d for d in chrome_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
            if version_dirs:
                latest_version = sorted(version_dirs, key=lambda x: [int(p) for p in x.name.split('.')])[-1]
                print(f"✓ Chrome version (from directory): {latest_version.name}")
                return True
        except Exception as e:
            print(f"⚠ Could not get Chrome version from directory: {e}")

    print("⚠ Chrome found but version could not be determined")
    return True  # Chrome exists even if we can't get version


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

    # Validate user data directory
    user_data_dir = os.getenv("CHROME_USER_DATA_DIR")
    if not user_data_dir:
        print("⚠ CHROME_USER_DATA_DIR not set in .env")
        return False

    user_data_path = Path(user_data_dir)
    if not user_data_path.exists():
        print(f"✗ Chrome user data directory not found: {user_data_dir}")
        return False

    print(f"✓ Chrome user data directory exists: {user_data_dir}")

    # Validate Default profile and Preferences file
    default_profile = user_data_path / "Default"
    prefs_file = default_profile / "Preferences"

    if not default_profile.exists():
        print("⚠ Default profile not found")
        return False
    print(f"✓ Default profile exists: {default_profile}")

    if not prefs_file.exists():
        print("⚠ Preferences file not found")
        return False
    print(f"✓ Preferences file exists: {prefs_file}")

    # Validate Preferences file content
    return _validate_preferences_file(prefs_file)


def _validate_preferences_file(prefs_file: Path) -> bool:
    """Validate Chrome Preferences file content."""
    try:
        with prefs_file.open(encoding='utf-8') as f:
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
    """Check ChromeDriver installation and version."""
    print_header("ChromeDriver Check")

    from dotenv import load_dotenv
    load_dotenv()

    driver_path = os.getenv("CHROME_DRIVER_PATH")
    if driver_path and Path(driver_path).exists():
        print(f"✓ ChromeDriver found at: {driver_path}")
        
        # Try to get ChromeDriver version
        try:
            result = subprocess.run(
                [driver_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"✓ ChromeDriver version: {result.stdout.strip()}")
        except Exception as e:
            print(f"⚠ Could not get ChromeDriver version: {e}")
        
        return True

    print("⚠ ChromeDriver not found at configured path")
    print("  undetected-chromedriver will auto-download if needed")
    return True  # Not critical since UC auto-downloads


def check_disk_space() -> bool:
    """Check available disk space for Chrome cache and logs."""
    print_header("Disk Space Check")
    
    try:
        import shutil
        
        # Check system drive
        system_drive = os.getenv("SystemDrive", "C:")
        total, used, free = shutil.disk_usage(system_drive)
        
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        used_percent = (used / total) * 100
        
        print(f"System drive ({system_drive}): {free_gb:.2f} GB free of {total_gb:.2f} GB ({used_percent:.1f}% used)")
        
        if free_gb < 1.0:
            print("✗ CRITICAL: Less than 1 GB free space!")
            print("  Recommendation: Free up disk space immediately")
            return False
        elif free_gb < 5.0:
            print("⚠ WARNING: Less than 5 GB free space")
            print("  Recommendation: Free up disk space soon")
            return False
        else:
            print("✓ Sufficient disk space available")
            return True
            
    except Exception as e:
        print(f"⚠ Could not check disk space: {e}")
        return True  # Don't fail on this check


def check_cache_directory() -> bool:
    """Check if Cache directory exists and is accessible."""
    print_header("Cache Directory Check")
    
    cache_dir = Path("Cache")
    
    if not cache_dir.exists():
        print(f"⚠ Cache directory not found: {cache_dir.absolute()}")
        print("  It will be created automatically when needed")
        return True
    
    print(f"✓ Cache directory exists: {cache_dir.absolute()}")
    
    # Check if writable
    try:
        test_file = cache_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        print("✓ Cache directory is writable")
        return True
    except Exception as e:
        print(f"✗ Cache directory is not writable: {e}")
        print("  Recommendation: Check file permissions")
        return False


def provide_recommendations(processes: dict, profile_ok: bool, disk_ok: bool, cache_ok: bool) -> None:
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

    if not disk_ok:
        print("4. Free up disk space:")
        print("   - Delete temporary files")
        print("   - Clear browser cache")
        print("   - Remove unnecessary programs")
        print("   - Chrome needs space for cache and logs")
        issues_found = True

    if not cache_ok:
        print("5. Fix Cache directory permissions:")
        print("   - Check folder permissions")
        print("   - Run: icacls Cache /grant %USERNAME%:F")
        issues_found = True

    if not issues_found:
        print("✓ No obvious issues detected")
        print("\nIf Action 6 still fails, try:")
        print("1. Update Chrome to the latest version")
        print("2. Check Windows Defender/antivirus isn't blocking Chrome")
        print("3. Run main.py with elevated privileges (Run as Administrator)")
        print("4. Check Logs\\app.log for detailed error messages")


def main() -> int:
    """Run all diagnostic checks."""
    print("\n" + "=" * 80)
    print("  Chrome/ChromeDriver Diagnostic Tool")
    print("=" * 80)

    chrome_ok = check_chrome_installation()
    processes = check_running_processes()
    profile_ok = check_chrome_profile()
    check_chromedriver()  # Check chromedriver but don't block on result
    disk_ok = check_disk_space()
    cache_ok = check_cache_directory()

    provide_recommendations(processes, profile_ok, disk_ok, cache_ok)

    print("\n" + "=" * 80)
    print("  Diagnostic Complete")
    print("=" * 80 + "\n")

    return 0 if (chrome_ok and profile_ok and disk_ok and cache_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

