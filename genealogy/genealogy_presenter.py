#!/usr/bin/env python3
"""
Genealogy Presenter

Shared presentation helpers to unify the post-selection flow for both
GEDCOM (Action 10) and API (search core) paths:
  - Header: "===Name (years) ==="
  - Family sections: Parents, Siblings (GEDCOM only), Spouses, Children
  - Relationship path: formatted text at the end

This module centralizes the UI output so both sources produce identical
user experience. Keep api_search_core focused on retrieval.
"""


# === PATH SETUP FOR PACKAGE IMPORTS ===
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import contextlib
import io
import logging
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

logger = logging.getLogger(__name__)

# Canonical implementation lives in research.relationship_formatting
from research.relationship_formatting import format_years_display


def _print_section_header(label: str, is_first: bool) -> None:
    """Print section header with appropriate spacing."""
    if is_first:
        print(f"{label}:")
    else:
        print(f"\n{label}:")


def _print_family_member(member: dict[str, Any]) -> None:
    """Print a single family member's information."""
    name = member.get("name", "Unknown")
    birth_year = member.get("birth_year")
    death_year = member.get("death_year")
    years_display = format_years_display(birth_year, death_year)
    print(f"   - {name}{years_display}")


def _deduplicate_members(members: list[Any]) -> list[dict[str, Any]]:
    """Deduplicate family members by (name, birth_year, death_year)."""
    seen: set[tuple[str, Any, Any]] = set()
    deduped: list[dict[str, Any]] = []
    for m in members:
        if not isinstance(m, dict):
            continue
        m_dict = cast(dict[str, Any], m)
        key = (str(m_dict.get("name", "")).strip().lower(), m_dict.get("birth_year"), m_dict.get("death_year"))
        if key not in seen:
            seen.add(key)
            deduped.append(m_dict)
    return deduped


def _filter_valid_members(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter out placeholder/blank names."""
    valid: list[dict[str, Any]] = []
    for m in members:
        name = str(m.get("name", "")).strip()
        if name and name.lower() not in {"unknown", "-", "n/a", "none", "?", "null"}:
            valid.append(m)
    return valid


def display_family_members(
    family_data: dict[str, list[dict[str, Any]]],
    relation_labels: dict[str, str] | None = None,
) -> None:
    """
    Display family members in a consistent format for both Action 10 and Action 11.

    Args:
        family_data: Dict with keys like 'parents', 'siblings', 'spouses', 'children'
        relation_labels: Optional custom labels for each relation type
    """
    if relation_labels is None:
        relation_labels = {
            "parents": "Parents",
            "siblings": "Siblings",
            "spouses": "Spouses",
            "children": "Children",
        }

    first_section = True
    for relation_key, label in relation_labels.items():
        members = family_data.get(relation_key, [])
        deduped_members = _deduplicate_members(members)
        valid_members = _filter_valid_members(deduped_members)

        _print_section_header(label, first_section)
        first_section = False

        if not valid_members:
            print("   - None recorded")
            continue

        for member in valid_members:
            _print_family_member(member)


def present_post_selection(
    display_name: str,
    birth_year: int | None,
    death_year: int | None,
    family_data: dict[str, list[dict[str, Any]]],
    owner_name: str,
    relation_labels: dict[str, str] | None = None,
    unified_path: list[dict[str, Any]] | None = None,
    formatted_path: str | None = None,
) -> None:
    """
    Unified presenter for post-selection info: match header, family, and relationship.
    Ensures identical output order and spacing for Actions 10 and 11.

    Args:
        display_name: Person's display name
        birth_year: Birth year if known
        death_year: Death year if known
        family_data: Dict with keys like 'parents', 'siblings', 'spouses', 'children'
        owner_name: Tree owner's display name
        relation_labels: Optional custom labels for family sections
        unified_path: Optional unified path list to format via relationship_utils
        formatted_path: Optional preformatted relationship text (takes precedence)
    """
    years_display = format_years_display(birth_year, death_year)

    # 1) Top match header
    print(f"=== {display_name}{years_display} ===\n")

    # 2) Family sections
    display_family_members(family_data, relation_labels)

    # 3) Relationship (blank line before)
    relation_text: str | None = None
    if formatted_path:
        relation_text = formatted_path
    elif unified_path:
        try:
            from research.relationship_utils import format_relationship_path_unified  # late import to avoid cycles

            relation_text = format_relationship_path_unified(unified_path, display_name, owner_name, None)
        except Exception as e:
            logger.error(f"Error formatting relationship path: {e}", exc_info=True)
            relation_text = None

    if relation_text:
        print("")
        print(relation_text)


def _capture_output(func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()


def _test_deduplicate_members_prefers_first_occurrence() -> None:
    members: list[Any] = [
        {"name": "Jane Doe", "birth_year": 1900, "death_year": 1970},
        {"name": "jane doe", "birth_year": 1900, "death_year": 1970},
        {"name": "Unique Person", "birth_year": 1905, "death_year": 1982},
        "invalid entry",
    ]

    deduped = _deduplicate_members(members)
    assert len(deduped) == 2
    assert deduped[0]["name"] == "Jane Doe"
    assert deduped[1]["name"] == "Unique Person"


def _test_filter_valid_members_skips_placeholders() -> None:
    members = [
        {"name": "Unknown"},
        {"name": " "},
        {"name": "-"},
        {"name": "Valid Name"},
    ]

    filtered = _filter_valid_members(members)
    assert filtered == [{"name": "Valid Name"}]


def _test_display_family_members_outputs_sections() -> None:
    family_data = {
        "parents": [{"name": "Parent One"}],
        "siblings": [],
        "spouses": [{"name": "Spouse"}, {"name": "spouse"}],
        "children": [{"name": "Unknown"}],
    }

    output = _capture_output(display_family_members, family_data)
    assert "Parents:" in output
    assert "Parent One" in output
    assert "Spouses:" in output
    assert "   - None recorded" in output  # siblings empty
    assert output.count("   - Spouse") == 1  # deduplicated spouse entry
    assert "Unknown" not in output  # filtered child entry


def _test_present_post_selection_formats_relationship_path() -> None:
    family_data = {relation: [] for relation in ("parents", "siblings", "spouses", "children")}
    fake_relationship_module = SimpleNamespace(
        format_relationship_path_unified=lambda _path, display_name, _owner, _unused: f"Path for {display_name}"
    )

    with mock.patch.dict(sys.modules, {"research.relationship_utils": fake_relationship_module}):
        output = _capture_output(
            present_post_selection,
            display_name="Sample Person",
            birth_year=None,
            death_year=None,
            family_data=family_data,
            owner_name="Owner",
            unified_path=[{"id": 1}],
        )

    assert "=== Sample Person" in output
    assert "Parents:" in output
    assert "Path for Sample Person" in output


# --- Internal Tests ---
try:  # pragma: no cover - optional dependency for runtime testing
    from testing.test_framework import TestSuite as _RealTestSuite
except Exception:

    @dataclass
    class _FallbackTestSuite:
        name: str
        module: str

        def start_suite(self) -> None:
            logger.info("Starting test suite: %s", self.name)

        @staticmethod
        def run_test(*args: Any, **kwargs: Any) -> None:
            func: Callable[..., Any] | None = None
            if "test_func" in kwargs and callable(kwargs["test_func"]):
                func = kwargs["test_func"]
            elif len(args) > 1 and callable(args[1]):
                func = args[1]

            if func is not None:
                func()

        def finish_suite(self) -> bool:
            logger.info("Finished test suite: %s", self.name)
            return True

    TestSuite = _FallbackTestSuite
else:
    TestSuite = _RealTestSuite


def genealogy_presenter_module_tests() -> bool:
    """Basic tests for genealogy_presenter core functions."""
    suite = TestSuite("genealogy_presenter", __name__)
    suite.start_suite()

    def _test_presenter_runs() -> None:
        family = {"parents": [], "siblings": [], "spouses": [], "children": []}
        present_post_selection(
            display_name="Test Person",
            birth_year=1900,
            death_year=1970,
            family_data=family,
            owner_name="Owner Person",
        )

    suite.run_test(
        test_name="Presenter prints header and sections",
        test_func=_test_presenter_runs,
        functions_tested="present_post_selection",
        test_summary="Ensure presenter runs without exceptions on minimal input",
        expected_outcome="No exceptions; output printed",
    )

    suite.run_test(
        test_name="Deduplicate members",
        test_func=_test_deduplicate_members_prefers_first_occurrence,
        functions_tested="_deduplicate_members",
        test_summary="Ensure duplicates are removed case-insensitively while preserving first entry",
        expected_outcome="Only unique member dictionaries remain in their original order.",
    )

    suite.run_test(
        test_name="Filter placeholder names",
        test_func=_test_filter_valid_members_skips_placeholders,
        functions_tested="_filter_valid_members",
        test_summary="Ensure placeholder values are dropped.",
        expected_outcome="Only meaningful names remain after filtering.",
    )

    suite.run_test(
        test_name="Display family members output",
        test_func=_test_display_family_members_outputs_sections,
        functions_tested="display_family_members",
        test_summary="Ensure sections print with deduplication and empty placeholders",
        expected_outcome="Sections print once and empty groups display 'None recorded'.",
    )

    suite.run_test(
        test_name="Relationship path formatting",
        test_func=_test_present_post_selection_formats_relationship_path,
        functions_tested="present_post_selection",
        test_summary="Ensure unified relationship path rendering is used when provided",
        expected_outcome="present_post_selection prints the relationship text from relationship_utils.",
    )

    return suite.finish_suite()


# Use centralized test runner utility
try:  # pragma: no cover - optional dependency for runtime testing
    from testing.test_utilities import create_standard_test_runner as _real_create_runner
except Exception:

    def _fallback_create_runner(test_func: Callable[[], bool]) -> Callable[[], bool]:
        def _runner() -> bool:
            return test_func()

        return _runner

    create_standard_test_runner = _fallback_create_runner
else:
    create_standard_test_runner = _real_create_runner

run_comprehensive_tests = create_standard_test_runner(genealogy_presenter_module_tests)


if __name__ == "__main__":
    run_comprehensive_tests()
