#!/usr/bin/env python3

"""
Color Utilities Module

Provides standardized ANSI color codes and formatting utilities for consistent
terminal output across the entire Ancestry project. This module consolidates
color functionality that was previously duplicated across multiple modules.

Features:
- ANSI color constants for all standard colors
- Static methods for easy text formatting
- Consistent color naming and behavior
- Support for both END and RESET naming conventions
"""

from typing import Any


class Colors:
    """ANSI color codes for terminal output with formatting utilities."""
    
    # Standard ANSI color codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'  # Reset to default
    RESET = '\033[0m'  # Alternative name for compatibility
    
    @staticmethod
    def green(text: str) -> str:
        """Return text in green color."""
        return f"{Colors.GREEN}{text}{Colors.END}"
    
    @staticmethod
    def red(text: str) -> str:
        """Return text in red color."""
        return f"{Colors.RED}{text}{Colors.END}"
    
    @staticmethod
    def yellow(text: str) -> str:
        """Return text in yellow color."""
        return f"{Colors.YELLOW}{text}{Colors.END}"
    
    @staticmethod
    def blue(text: str) -> str:
        """Return text in blue color."""
        return f"{Colors.BLUE}{text}{Colors.END}"
    
    @staticmethod
    def magenta(text: str) -> str:
        """Return text in magenta color."""
        return f"{Colors.MAGENTA}{text}{Colors.END}"
    
    @staticmethod
    def cyan(text: str) -> str:
        """Return text in cyan color."""
        return f"{Colors.CYAN}{text}{Colors.END}"
    
    @staticmethod
    def white(text: str) -> str:
        """Return text in white color."""
        return f"{Colors.WHITE}{text}{Colors.END}"
    
    @staticmethod
    def gray(text: str) -> str:
        """Return text in gray color."""
        return f"{Colors.GRAY}{text}{Colors.END}"
    
    @staticmethod
    def bold(text: str) -> str:
        """Return text in bold formatting."""
        return f"{Colors.BOLD}{text}{Colors.END}"
    
    @staticmethod
    def underline(text: str) -> str:
        """Return text with underline formatting."""
        return f"{Colors.UNDERLINE}{text}{Colors.END}"


def has_ansi_codes(text: Any) -> bool:
    """Check if text already contains ANSI color codes."""
    return '\033[' in str(text)


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text."""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


# Test functions for quality validation
def test_color_constants() -> bool:
    """Test that all color constants are properly defined."""
    assert Colors.RED == "\033[91m"
    assert Colors.GREEN == "\033[92m"
    assert Colors.YELLOW == "\033[93m"
    assert Colors.BLUE == "\033[94m"
    assert Colors.MAGENTA == "\033[95m"
    assert Colors.CYAN == "\033[96m"
    assert Colors.WHITE == "\033[97m"
    assert Colors.GRAY == "\033[90m"
    assert Colors.BOLD == "\033[1m"
    assert Colors.UNDERLINE == "\033[4m"
    assert Colors.END == "\033[0m"
    assert Colors.RESET == "\033[0m"
    return True


def test_color_methods() -> bool:
    """Test that color formatting methods work correctly."""
    test_text = "test"
    assert Colors.green(test_text) == f"\033[92m{test_text}\033[0m"
    assert Colors.red(test_text) == f"\033[91m{test_text}\033[0m"
    assert Colors.yellow(test_text) == f"\033[93m{test_text}\033[0m"
    assert Colors.blue(test_text) == f"\033[94m{test_text}\033[0m"
    assert Colors.bold(test_text) == f"\033[1m{test_text}\033[0m"
    return True


def test_utility_functions() -> bool:
    """Test utility functions for ANSI code detection and stripping."""
    plain_text = "Hello World"
    colored_text = Colors.red("Hello World")
    
    assert not has_ansi_codes(plain_text)
    assert has_ansi_codes(colored_text)
    assert strip_ansi_codes(colored_text) == plain_text
    return True


def run_comprehensive_tests() -> bool:
    """Run all color utility tests."""
    try:
        test_color_constants()
        test_color_methods()
        test_utility_functions()
        return True
    except AssertionError:
        return False


if __name__ == "__main__":
    # Demo the color utilities
    print("Color Utilities Demo:")
    print(Colors.red("This is red text"))
    print(Colors.green("This is green text"))
    print(Colors.yellow("This is yellow text"))
    print(Colors.blue("This is blue text"))
    print(Colors.bold("This is bold text"))
    print(Colors.underline("This is underlined text"))
    
    # Run tests
    if run_comprehensive_tests():
        print(Colors.green("✅ All tests passed!"))
    else:
        print(Colors.red("❌ Some tests failed!"))
