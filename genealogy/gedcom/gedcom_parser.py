#!/usr/bin/env python3

"""GEDCOM name extraction and validation helpers.

Provides functions to extract, validate, and format individual names from
GEDCOM records. This module handles multiple name extraction strategies
(format method, sub-tag format, manual GIVN/SURN combination, sub-tag value)
and delegates final cleanup to the shared ``format_name`` utility.
"""

# === PATH SETUP FOR PACKAGE IMPORTS ===
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# === CORE INFRASTRUCTURE ===
import logging
import os
from typing import Any, cast

from core.utils import format_name
from genealogy.gedcom.gedcom_utils import (
    TAG_GIVN,
    TAG_INDI,
    TAG_NAME,
    TAG_SURN,
    GedcomIndividualType,
    _is_individual,
    extract_and_fix_id,
)
from testing.test_framework import TestSuite, create_standard_test_runner

logger = logging.getLogger(__name__)


# ==============================================
# Helper functions for get_full_name
# ==============================================


def _validate_individual_type(indi: GedcomIndividualType) -> tuple[GedcomIndividualType | None, str]:
    """Validate and extract individual from input, handling wrapped values."""
    if not _is_individual(indi):
        wrapped_value = getattr(indi, "value", None)
        if _is_individual(wrapped_value):
            return cast(GedcomIndividualType, wrapped_value), ""
        logger.warning(f"get_full_name called with non-Individual type: {type(indi)}")
        return None, "Unknown (Invalid Type)"

    if indi is None:
        return None, "Unknown (None)"

    return indi, ""


def _try_name_format_method(indi: GedcomIndividualType, indi_id_log: str) -> str | None:
    """Try to get name using indi.name.format() method."""
    if not indi or not hasattr(indi, "name"):
        return None

    name_rec = indi.name
    if not name_rec or not hasattr(name_rec, "format") or not callable(name_rec.format):
        return None

    try:
        return name_rec.format()
    except Exception as fmt_err:
        logger.warning(f"Error calling indi.name.format() for {indi_id_log}: {fmt_err}")
        return None


def _try_sub_tag_format_method(indi: GedcomIndividualType, indi_id_log: str) -> str | None:
    """Try to get name using indi.sub_tag(TAG_NAME).format() method."""
    if not indi or not hasattr(indi, "sub_tag"):
        return None

    name_tag = indi.sub_tag(TAG_NAME)
    if not name_tag or not hasattr(name_tag, "format") or not callable(getattr(name_tag, "format", None)):
        return None

    try:
        format_method = getattr(name_tag, "format", None)
        if callable(format_method):
            formatted = format_method()
            return formatted if isinstance(formatted, str) else None
        return None
    except Exception as fmt_err:
        logger.warning(f"Error calling indi.sub_tag(TAG_NAME).format() for {indi_id_log}: {fmt_err}")
        return None


def _try_manual_name_combination(indi: GedcomIndividualType, indi_id_log: str) -> str | None:
    """Try to manually combine GIVN and SURN tags."""
    _ = indi_id_log  # Unused but kept for API consistency
    if not indi or not hasattr(indi, "sub_tag"):
        return None

    name_tag = indi.sub_tag(TAG_NAME)
    if not name_tag:
        return None

    givn = name_tag.sub_tag_value(TAG_GIVN) if hasattr(name_tag, "sub_tag_value") else None
    surn = name_tag.sub_tag_value(TAG_SURN) if hasattr(name_tag, "sub_tag_value") else None

    # Combine, prioritizing surname placement
    if givn and surn:
        formatted_name = f"{givn} {surn}"
    elif givn:
        formatted_name = givn
    elif surn:
        formatted_name = surn
    else:
        return None

    return formatted_name


def _try_sub_tag_value_method(indi: GedcomIndividualType, indi_id_log: str) -> str | None:
    """Try to get name using indi.sub_tag_value(TAG_NAME) as last resort."""
    _ = indi_id_log  # Unused but kept for API consistency
    if not indi or not hasattr(indi, "sub_tag_value"):
        return None

    name_val = indi.sub_tag_value(TAG_NAME)
    if not isinstance(name_val, str) or not name_val.strip() or name_val == "/":
        return None

    return name_val


def _clean_and_format_name(formatted_name: str | None, name_source: str) -> str:
    """Clean and format the extracted name."""
    if not formatted_name:
        return "Unknown (No Name Found)"

    cleaned_name = format_name(formatted_name)
    if cleaned_name and cleaned_name != "Unknown":
        return cleaned_name

    return f"Unknown ({name_source} Error)"


def get_full_name(indi: GedcomIndividualType) -> str:
    """Safely gets formatted name, checking for .format method. V3"""
    # Validate individual type
    indi, error_msg = _validate_individual_type(indi)
    if error_msg:
        return error_msg

    indi_id_log = extract_and_fix_id(indi) or "Unknown ID"

    try:
        # Try multiple methods to extract name, in order of preference
        name_extraction_methods = [
            (_try_name_format_method, "indi.name.format()"),
            (_try_sub_tag_format_method, "indi.sub_tag(TAG_NAME).format()"),
            (_try_manual_name_combination, "manual GIVN/SURN combination"),
            (_try_sub_tag_value_method, "indi.sub_tag_value(TAG_NAME)"),
        ]

        for method, source in name_extraction_methods:
            formatted_name = method(indi, indi_id_log)
            if formatted_name:
                return _clean_and_format_name(formatted_name, source)

        # No method succeeded
        return _clean_and_format_name(None, "Unknown")

    except Exception as e:
        logger.error(f"Unexpected error in get_full_name for @{indi_id_log}@: {e}", exc_info=True)
        return "Unknown (Error)"


# ==============================================
# Test Helpers
# ==============================================

_PATCH_GEDCOM_SENTINEL = object()


from contextlib import contextmanager


@contextmanager
def _temporary_globals(overrides: dict[str, Any]):
    previous: dict[str, Any] = {}
    try:
        for key, value in overrides.items():
            previous[key] = globals().get(key, _PATCH_GEDCOM_SENTINEL)
            globals()[key] = value
        yield
    finally:
        for key, original in previous.items():
            if original is _PATCH_GEDCOM_SENTINEL:
                globals().pop(key, None)
            else:
                globals()[key] = original


class _FakeNameTag:
    def __init__(self, givn: str | None = None, surn: str | None = None) -> None:
        self._givn = givn
        self._surn = surn

    def sub_tag_value(self, tag: str) -> str | None:
        if tag == TAG_GIVN:
            return self._givn
        if tag == TAG_SURN:
            return self._surn
        return None


class _FakeIndividualRecord:
    def __init__(
        self,
        *,
        xref_id: str = "@I1@",
        name_obj: Any = None,
        name_tag: Any = None,
        sub_tag_values: dict[str, Any] | None = None,
    ) -> None:
        self.tag = TAG_INDI
        self.xref_id = xref_id
        self.name = name_obj
        self._name_tag = name_tag
        self._sub_tag_values = sub_tag_values or {}

    def sub_tag(self, tag: str) -> Any:
        if tag == TAG_NAME:
            return self._name_tag
        return None

    def sub_tag_value(self, tag: str) -> Any:
        return self._sub_tag_values.get(tag)


class _FakeNameObject:
    def __init__(self, formatted: str) -> None:
        self._formatted = formatted

    def format(self) -> str:
        return self._formatted


# ==============================================
# Tests
# ==============================================


def _test_get_full_name_prefers_name_format_and_manual_fallback() -> bool:
    def _format_name_without_slashes(value: str) -> str:
        return " ".join(value.replace("/", " ").split())

    indi_with_format = cast(GedcomIndividualType, _FakeIndividualRecord(name_obj=_FakeNameObject("Ada /Lovelace/")))
    with _temporary_globals({"format_name": _format_name_without_slashes}):
        assert get_full_name(indi_with_format) == "Ada Lovelace"

    indi_manual = cast(GedcomIndividualType, _FakeIndividualRecord(name_tag=_FakeNameTag("Ada", "Lovelace")))

    def _identity(value: str) -> str:
        return value

    with _temporary_globals({"format_name": _identity}):
        assert get_full_name(indi_manual) == "Ada Lovelace"

    bad_value = cast(GedcomIndividualType, object())
    assert get_full_name(bad_value) == "Unknown (Invalid Type)"
    return True


def module_tests() -> bool:
    suite = TestSuite("gedcom_parser", "gedcom_parser.py")
    suite.run_test(
        "Name resolution",
        _test_get_full_name_prefers_name_format_and_manual_fallback,
        "Ensures name formatting prefers indi.name and falls back to manual combination.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    if os.environ.get("RUN_MODULE_TESTS") == "1":
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    print("gedcom_parser provides name extraction helpers; no standalone CLI entry point.")
    print("Set RUN_MODULE_TESTS=1 before execution to run the embedded regression tests.")
    sys.exit(0)
