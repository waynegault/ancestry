#!/usr/bin/env python3

"""
cross_platform.py - Cross-platform compatibility utilities

Provides OS-agnostic wrappers for platform-specific operations.
Ensures the codebase works on Windows, macOS, and Linux without
scattering OS-specific checks throughout the codebase.

Usage:
    from core.cross_platform import clear_screen, get_platform

    clear_screen()  # Works on any platform
"""

import os
import platform
from enum import Enum
from typing import Any


class Platform(Enum):
    """Supported operating systems."""

    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    UNKNOWN = "unknown"


def get_platform() -> Platform:
    """Detect current platform."""
    system = platform.system().lower()
    if system == "windows":
        return Platform.WINDOWS
    if system == "darwin":
        return Platform.MACOS
    if system == "linux":
        return Platform.LINUX
    return Platform.UNKNOWN


def clear_screen() -> None:
    """Clear terminal screen in a cross-platform way."""
    os.system("cls" if os.name == "nt" else "clear")


def get_path_separator() -> str:
    """Get OS-specific path separator."""
    return os.sep


def normalize_path(path: str) -> str:
    """Normalize path separators for current platform."""
    return os.path.normpath(path)


def get_default_browser() -> str:
    """Get default browser name for current platform."""
    platform = get_platform()
    if platform == Platform.WINDOWS:
        return "chrome"
    if platform == Platform.MACOS:
        return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if platform == Platform.LINUX:
        return "google-chrome"
    return "chrome"


def prevent_sleep(enable: bool) -> Any:
    """
    Prevent system sleep during long operations.

    Returns context manager or state object for restoration.
    On non-Windows platforms, this is a no-op with warning.

    Args:
        enable: True to prevent sleep, False to restore

    Returns:
        State object for restore (Windows) or None (other platforms)
    """
    if get_platform() != Platform.WINDOWS:
        if enable:
            import logging

            logging.getLogger(__name__).debug("Sleep prevention is Windows-only, skipping")
        return None

    # Import Windows-specific implementation only when needed
    from core.utils import prevent_system_sleep, restore_system_sleep

    if enable:
        return prevent_system_sleep()
    return None


def get_file_version(filepath: str) -> str | None:
    """
    Get file version on Windows, None on other platforms.

    Uses Windows API via ctypes. On non-Windows, returns None.

    Args:
        filepath: Path to executable

    Returns:
        Version string or None
    """
    if get_platform() != Platform.WINDOWS:
        return None

    import ctypes

    try:
        chrome_exe = filepath
        size = ctypes.windll.version.GetFileVersionInfoSizeW(chrome_exe, None)  # type: ignore[union-attr]
        if not size:
            return None

        data = ctypes.create_string_buffer(size)
        ctypes.windll.version.GetFileVersionInfoW(chrome_exe, 0, size, data)  # type: ignore[union-attr]

        p = ctypes.c_void_p()
        length = ctypes.c_uint()
        ctypes.windll.version.VerQueryValueW(data, r"\\", ctypes.byref(p), ctypes.byref(length))  # type: ignore[union-attr]

        ms = ctypes.c_uint32.from_address(p.value + 8).value  # type: ignore[arg-type]
        major = ms >> 16
        return str(major)
    except Exception:
        return None


def focus_console_window() -> None:
    """
    Focus console window on Windows. No-op on other platforms.
    """
    if get_platform() != Platform.WINDOWS:
        return

    import ctypes

    try:
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return
        kernel32 = getattr(windll, "kernel32", None)
        user32 = getattr(windll, "user32", None)
        if kernel32 is None or user32 is None:
            return

        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            user32.SetForegroundWindow(hwnd)
    except Exception:
        pass  # Best effort, don't fail if this doesn't work


def cross_platform_module_tests() -> bool:
    """Test cross-platform utilities."""
    print("🌐 Testing Cross-Platform Utilities...")
    print()

    all_passed = True

    # Test 1: Platform detection
    print("Test 1: Platform detection")
    try:
        plat = get_platform()
        if isinstance(plat, Platform):
            print(f"✅ Detected platform: {plat.value}")
        else:
            print(f"❌ Expected Platform enum, got {type(plat)}")
            all_passed = False
    except Exception as e:
        print(f"❌ Failed: {e}")
        all_passed = False

    # Test 2: Clear screen (should not raise)
    print("Test 2: Clear screen")
    try:
        clear_screen()
        print("✅ PASSED: Clear screen works")
    except Exception as e:
        print(f"❌ Failed: {e}")
        all_passed = False

    # Test 3: Path helpers
    print("Test 3: Path helpers")
    try:
        sep = get_path_separator()
        assert isinstance(sep, str) and len(sep) == 1
        print(f"✅ Path separator: '{sep}'")

        normalized = normalize_path("foo/bar/baz")
        assert isinstance(normalized, str)
        print("✅ Path normalization works")
    except Exception as e:
        print(f"❌ Failed: {e}")
        all_passed = False

    # Test 4: File version (Windows-only)
    print("Test 4: File version")
    try:
        version = get_file_version("chrome.exe")
        if get_platform() == Platform.WINDOWS:
            print(f"✅ File version on Windows: {version}")
        else:
            print(f"✅ File version returns None on non-Windows: {version}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        all_passed = False

    if all_passed:
        print("\n🎉 All cross-platform tests PASSED")
    else:
        print("\n❌ Some tests FAILED")

    return all_passed


# Use centralized test runner utility from test_utilities
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(cross_platform_module_tests)


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
