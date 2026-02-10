#!/usr/bin/env python3

"""String formatting utilities for name processing, ordinal formatting, and text helpers."""

import logging
import re

from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


# === ORDINAL FORMATTING HELPER FUNCTIONS ===


def _get_ordinal_suffix(num: int) -> str:
    """Get the ordinal suffix (st, nd, rd, th) for a number."""
    # Special case for 11-13 (always 'th')
    if 11 <= (num % 100) <= 13:
        return "th"

    # Check last digit
    last_digit = num % 10
    if last_digit == 1:
        return "st"
    if last_digit == 2:
        return "nd"
    if last_digit == 3:
        return "rd"
    return "th"


def _format_number_as_ordinal(num: int) -> str:
    """Format a number as an ordinal string (e.g., 1 -> '1st', 2 -> '2nd')."""
    return str(num) + _get_ordinal_suffix(num)


def _title_case_with_lowercase_particles(text: str) -> str:
    """Apply title case but keep certain particles lowercase (of, the, a, etc.)."""
    words = text.title().split()
    lowercase_particles = {"Of", "The", "A", "An", "In", "On", "At", "For", "To", "With"}

    for i, word in enumerate(words):
        # Keep particles lowercase except at start
        if i > 0 and word in lowercase_particles:
            words[i] = word.lower()

    return " ".join(words)


def ordinal_case(text: str | int) -> str:
    """
    Corrects ordinal suffixes (1st, 2nd, 3rd, 4th) to lowercase within a string,
    often used after applying title casing. Handles relationship terms simply.
    Accepts string or integer input for numbers.
    """
    # Handle empty/None input
    if not text and text != 0:
        return str(text)

    # Try to convert to number and format as ordinal
    try:
        num = int(text)
        return _format_number_as_ordinal(num)
    except (ValueError, TypeError):
        # Not a number - apply title case with lowercase particles
        if isinstance(text, str):
            return _title_case_with_lowercase_particles(text)
        return str(text)


# End of ordinal_case


# === NAME FORMATTING HELPER FUNCTIONS ===


def _remove_gedcom_slashes(name: str) -> str:
    """Remove GEDCOM-style slashes around surnames."""
    name = re.sub(r"\s*/([^/]+)/\s*", r" \1 ", name)  # Middle
    name = re.sub(r"^/([^/]+)/\s*", r"\1 ", name)  # Start
    name = re.sub(r"\s*/([^/]+)/$", r" \1", name)  # End
    name = re.sub(r"^/([^/]+)/$", r"\1", name)  # Only
    return re.sub(r"\s+", " ", name).strip()


def _format_quoted_nickname(part: str) -> str | None:
    """Format quoted nicknames like 'Betty' or 'Bo'."""
    if part.startswith("'") and part.endswith("'") and len(part) > 2:
        inner_content = part[1:-1]
        return "'" + inner_content.capitalize() + "'"
    return None


def _format_hyphenated_name(part: str, lowercase_particles: set[str]) -> str:
    """Format hyphenated names like Smith-Jones or van-der-Berg."""
    hyphenated_elements: list[str] = []
    sub_parts = part.split("-")
    for idx, sub_part in enumerate(sub_parts):
        if idx > 0 and sub_part.lower() in lowercase_particles:
            hyphenated_elements.append(sub_part.lower())
        elif sub_part:
            hyphenated_elements.append(sub_part.capitalize())
    return "-".join(filter(None, hyphenated_elements))


def _format_apostrophe_name(part: str) -> str | None:
    """Format names with internal apostrophes like O'Malley or D'Angelo."""
    if "'" in part and len(part) > 1 and not (part.startswith("'") or part.endswith("'")):
        name_pieces = part.split("'")
        return "'".join(p.capitalize() for p in name_pieces)
    return None


def _format_mc_mac_prefix(part: str) -> str | None:
    """Format Mc/Mac prefixes like McDonald or MacGregor."""
    part_lower = part.lower()
    if part_lower.startswith("mc") and len(part) > 2:
        return "Mc" + part[2:].capitalize()
    if part_lower.startswith("mac") and len(part) > 3:
        if part_lower == "mac":
            return "Mac"
        return "Mac" + part[3:].capitalize()
    return None


def _format_initial(part: str) -> str | None:
    """Format initials like J. or J."""
    if len(part) == 2 and part.endswith(".") and part[0].isalpha():
        return part[0].upper() + "."
    if len(part) == 1 and part.isalpha():
        return part.upper()
    return None


def _format_name_part(part: str, index: int, lowercase_particles: set[str], uppercase_exceptions: set[str]) -> str:
    """Format a single part of a name with all special case handling."""
    part_lower = part.lower()
    result = None

    # Lowercase particles (but not at start)
    if index > 0 and part_lower in lowercase_particles:
        result = part_lower
    # Uppercase exceptions (II, III, SR, JR)
    elif part.upper() in uppercase_exceptions:
        result = part.upper()
    else:
        # Quoted nicknames
        quoted = _format_quoted_nickname(part)
        if quoted:
            result = quoted
        # Hyphenated names
        elif "-" in part:
            result = _format_hyphenated_name(part, lowercase_particles)
        else:
            # Apostrophe names (O'Malley)
            apostrophe = _format_apostrophe_name(part)
            if apostrophe:
                result = apostrophe
            else:
                # Mc/Mac prefixes
                mc_mac = _format_mc_mac_prefix(part)
                if mc_mac:
                    result = mc_mac
                else:
                    # Initials
                    initial = _format_initial(part)
                    result = initial or part.capitalize()

    return result


def format_name(name: str | None) -> str:
    """
    Formats a person's name string to title case, preserving uppercase components
    (like initials or acronyms) and handling None/empty input gracefully.
    Also removes GEDCOM-style slashes around surnames anywhere in the string.
    Handles common name particles and prefixes like Mc/Mac/O' and quoted nicknames.
    """
    # Validate input
    if not name:
        return "Valued Relative"

    # Handle non-alphabetic input
    if name.isdigit() or re.fullmatch(r"[^a-zA-Z]+", name):
        logger.debug(f"Formatting name: Input '{name}' appears non-alphabetic, returning as is.")
        stripped_name = name.strip()
        return stripped_name if stripped_name else "Valued Relative"

    try:
        # Clean and prepare name
        cleaned_name = _remove_gedcom_slashes(name.strip())

        # Define special case sets
        lowercase_particles = {"van", "von", "der", "den", "de", "di", "da", "do", "la", "le", "el"}
        uppercase_exceptions = {"II", "III", "IV", "SR", "JR"}

        # Format each part
        parts = cleaned_name.split()
        formatted_parts = [
            _format_name_part(part, i, lowercase_particles, uppercase_exceptions) for i, part in enumerate(parts)
        ]

        # Join and clean up
        final_name = " ".join(formatted_parts)
        final_name = re.sub(r"\s+", " ", final_name).strip()
        return final_name if final_name else "Valued Relative"

    except Exception as e:
        logger.error(f"Error formatting name '{name}': {e}", exc_info=False)
        return name.title()


# End of format_name


# === TEST FUNCTIONS ===


def _test_ordinal_case() -> None:
    """Test ordinal number formatting with various input types"""
    test_cases = [
        (1, "1st", "First ordinal"),
        (2, "2nd", "Second ordinal"),
        (3, "3rd", "Third ordinal"),
        (4, "4th", "Fourth ordinal"),
        (11, "11th", "Eleventh (special case)"),
        (12, "12th", "Twelfth (special case)"),
        (13, "13th", "Thirteenth (special case)"),
        (21, "21st", "Twenty-first ordinal"),
        (22, "22nd", "Twenty-second ordinal"),
        (23, "23rd", "Twenty-third ordinal"),
        (101, "101st", "One hundred first"),
        ("Great Uncle", "Great Uncle", "Text input"),
    ]

    print("📋 Testing ordinal number formatting:")
    results: list[bool] = []

    for input_val, expected, description in test_cases:
        try:
            result = ordinal_case(input_val)
            matches_expected = result == expected

            status = "✅" if matches_expected else "❌"
            print(f"   {status} {description}")
            print(f"      Input: {input_val} (Type: {type(input_val).__name__})")
            print(f"      Output: '{result}' (Expected: '{expected}')")

            results.append(matches_expected)
            assert matches_expected, f"Failed for {input_val}: expected '{expected}', got '{result}'"

        except Exception as e:
            print(f"   ❌ {description}: Exception {e}")
            results.append(False)

    print(f"📊 Results: {sum(results)}/{len(results)} ordinal formatting tests passed")


def _test_format_name() -> None:
    """Test name formatting with various input types and edge cases"""
    test_cases = [
        ("john doe", "John Doe", "Basic name formatting"),
        (None, "Valued Relative", "None input handling"),
        ("", "Valued Relative", "Empty string handling"),
        ("ALLCAPS NAME", "Allcaps Name", "All caps input"),
        ("  spaces  everywhere  ", "Spaces Everywhere", "Extra whitespace"),
        ("o'connor", "O'Connor", "Apostrophe handling"),
    ]

    print("📋 Testing name formatting with various cases:")
    results: list[bool] = []

    for input_val, expected, description in test_cases:
        try:
            result = format_name(input_val)
            matches_expected = result == expected

            status = "✅" if matches_expected else "❌"
            print(f"   {status} {description}")
            print(f"      Input: {input_val!r} → Output: '{result}'")
            print(f"      Expected: '{expected}'")

            results.append(matches_expected)
            assert matches_expected, f"Failed for {input_val!r}: expected '{expected}', got '{result}'"

        except Exception as e:
            print(f"   ❌ {description}: Exception {e}")
            results.append(False)

    print(f"📊 Results: {sum(results)}/{len(results)} name formatting tests passed")


# === TEST SUITE ===


def module_tests() -> bool:
    """Run string formatting tests using standardized TestSuite format."""
    suite = TestSuite("String Formatting Utilities", "string_utils.py")

    suite.run_test("Ordinal Formatting", _test_ordinal_case, "Format numbers with ordinal suffixes (1st, 2nd, etc.)")

    suite.run_test("Name Formatting", _test_format_name, "Format names with proper capitalization")

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
