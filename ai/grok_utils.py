#!/usr/bin/env python3

"""
grok_utils.py - Utilities for normalizing and extracting content from Grok (xAI) SDK responses

Consolidates duplicate Grok normalization functions that were scattered across:
- ai_api_test.py (3 functions)
- ai/ai_interface.py (3 functions)

All Grok response normalization now uses these centralized utilities.
"""

from collections.abc import Iterable
from typing import Any


def normalize_grok_entry(entry: Any) -> str | None:
    """
    Normalize a single Grok SDK entry to text.

    Handles:
    - Plain strings (stripped)
    - Objects with .text attribute
    - Objects with .content attribute

    Args:
        entry: Grok SDK entry (string or object)

    Returns:
        Stripped text or None
    """
    if isinstance(entry, str):
        return entry.strip()
    # Try common attribute names for text content
    for attr in ("text", "content"):
        value = getattr(entry, attr, None)
        if isinstance(value, str):
            return value.strip()
    return None


def normalize_grok_sequence(entries: Iterable[Any]) -> str | None:
    """
    Join multiple Grok entries into a single string.

    Args:
        entries: List of Grok entries (strings or objects)

    Returns:
        Joined text or None
    """
    parts = [part for part in (normalize_grok_entry(item) for item in entries) if part]
    return "\n".join(parts) if parts else None


def normalize_grok_value(value: Any) -> str | None:
    """
    Normalize Grok value which could be string, list, or object.

    Args:
        value: Grok SDK value (string, list, or object)

    Returns:
        Normalized text or None
    """
    if value is None:
        return None
    if isinstance(value, list):
        return normalize_grok_sequence(value)
    return normalize_grok_entry(value)


def normalize_grok_payload(payload: Any) -> str | None:
    """
    Normalize Grok payload (string or list of parts).

    Args:
        payload: Grok payload (string or list)

    Returns:
        Normalized text or None
    """
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        return normalize_grok_sequence(payload)
    return None


def normalize_grok_message(message: Any) -> str | None:
    """
    Normalize Grok message object.

    Args:
        message: Grok message object with .content attribute

    Returns:
        Normalized text or None
    """
    if message is None:
        return None
    message_content = getattr(message, "content", None)
    return normalize_grok_payload(message_content)


def extract_grok_content(response: Any | None) -> str | None:
    """
    Extract text content from Grok (xAI) SDK responses.

    Tries multiple strategies in order:
    1. response.content
    2. response.message.content
    3. response itself
    4. Fallback to str(response)

    Args:
        response: Grok SDK response object

    Returns:
        Extracted text or None
    """
    if response is None:
        return None

    try:
        primary_content = getattr(response, "content", None)

        # Try primary content first
        normalized = normalize_grok_value(primary_content)
        if normalized:
            return normalized

        # Try message content
        message = getattr(response, "message", None)
        if message is not None:
            message_content = getattr(message, "content", None)
            normalized = normalize_grok_value(message_content)
            if normalized:
                return normalized

        # Try response itself
        normalized = normalize_grok_value(response)
        if normalized:
            return normalized

        # Last resort: convert to string
        fallback_source = primary_content if primary_content is not None else response
        fallback_str = str(fallback_source).strip()
        return fallback_str or None
    except Exception:
        # SDK objects may have unexpected attributes
        return None


# ============================================================================
# MODULE TESTS
# ============================================================================


def grok_utils_module_tests() -> bool:
    """Test Grok utilities with mock SDK objects."""
    print("🤖 Testing Grok Utils Module...")
    print()

    all_passed = True

    # Test 1: normalize_grok_entry with string
    print("Test 1: normalize_grok_entry with string")
    try:
        from . import normalize_grok_entry

        result = normalize_grok_entry("  hello world  ")
        assert result == "hello world", f"Expected 'hello world', got {result!r}"
        print("✅ PASSED: String normalization")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 2: normalize_grok_entry with object
    print("Test 2: normalize_grok_entry with object")
    try:
        from types import SimpleNamespace

        obj = SimpleNamespace(text="test content")
        result = normalize_grok_entry(obj)
        assert result == "test content", f"Expected 'test content', got {result!r}"
        print("✅ PASSED: Object with .text")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 3: normalize_grok_sequence
    print("Test 3: normalize_grok_sequence")
    try:
        entries = ["part1", "  part2  ", None, "part3"]
        result = normalize_grok_sequence(entries)
        assert result == "part1\npart2\npart3", f"Got {result!r}"
        print("✅ PASSED: Sequence joining")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 4: extract_grok_content with simple response
    print("Test 4: extract_grok_content with simple response")
    try:
        from types import SimpleNamespace

        response = SimpleNamespace(content="simple response")
        result = extract_grok_content(response)
        assert result == "simple response", f"Got {result!r}"
        print("✅ PASSED: Simple content extraction")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 5: extract_grok_content with None
    print("Test 5: extract_grok_content with None")
    try:
        result = extract_grok_content(None)
        assert result is None, f"Expected None, got {result!r}"
        print("✅ PASSED: None handling")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    if all_passed:
        print("\n🎉 All Grok utils tests PASSED")
    else:
        print("\n❌ Some tests FAILED")

    return all_passed


# Use centralized test runner utility from test_utilities
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(grok_utils_module_tests)


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
