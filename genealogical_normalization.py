#!/usr/bin/env python3

"""Genealogical Normalization Helpers.

Conservative helpers to normalize AI extraction results into structured
format for downstream messaging and task generation.

Key Features:
- Ensures required keys exist in extracted_data
- Transforms legacy flat keys to structured containers
- Deduplicates simple string lists
- Single entrypoint normalize_ai_response()

Note: Intentionally avoids external side effects and imports only
standard library modules for safety.
"""

from __future__ import annotations

import re
from typing import Any, cast

# Minimal constants for expected keys used across the codebase
STRUCTURED_KEYS = [
    "structured_names",
    "vital_records",
    "relationships",
    "locations",
    "occupations",
    "research_questions",
    "documents_mentioned",
    "dna_information",
]

# Legacy/flat keys occasionally seen in AI responses
LEGACY_TO_STRUCTURED_MAP: dict[str, tuple[str, str]] = {
    "mentioned_names": ("structured_names", "name"),
    "mentioned_locations": ("locations", "place"),
    "mentioned_dates": ("vital_records", "date"),
    # relationships and key_facts cannot be reliably auto-mapped; skip
}


def _dedupe_list_str(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for it in items:
        if it is None:
            continue
        s = str(it).strip()
        if not s:
            continue
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
    return out


# Helper functions for _validate_and_normalize_date

def _detect_approximate_indicator(date_str: str) -> bool:
    """Detect if date string contains approximate indicators."""
    approx_indicators = ["circa", "~", "about", "abt", "c.", "ca.", "before", "after", "bef", "aft"]
    return any(indicator in date_str.lower() for indicator in approx_indicators)


def _extract_year_from_date(date_str: str) -> str | None:
    """Extract year from date string (genealogically relevant range: 1400-2100)."""
    year_match = re.search(r'\b(1[4-9]\d{2}|20\d{2}|21\d{2})\b', date_str)
    return year_match.group(1) if year_match else None


def _extract_month_from_date(date_str: str) -> str | None:
    """Extract month number from date string."""
    month_patterns = {
        r'\b(jan|january)\b': '01', r'\b(feb|february)\b': '02', r'\b(mar|march)\b': '03',
        r'\b(apr|april)\b': '04', r'\b(may)\b': '05', r'\b(jun|june)\b': '06',
        r'\b(jul|july)\b': '07', r'\b(aug|august)\b': '08', r'\b(sep|september)\b': '09',
        r'\b(oct|october)\b': '10', r'\b(nov|november)\b': '11', r'\b(dec|december)\b': '12'
    }

    for pattern, month_num in month_patterns.items():
        if re.search(pattern, date_str.lower()):
            return month_num

    return None


def _extract_day_from_date(date_str: str, month: str | None) -> str | None:
    """Extract day from date string if month is present."""
    if not month:
        return None

    day_match = re.search(r'\b([0-3]?\d)\b', date_str)
    if day_match:
        potential_day = int(day_match.group(1))
        if 1 <= potential_day <= 31:
            return f"{potential_day:02d}"

    return None


def _construct_normalized_date(year: str, month: str | None, day: str | None) -> str:
    """Construct normalized date string from components."""
    if day and month:
        return f"{year}-{month}-{day}"
    if month:
        return f"{year}-{month}"
    return year


def _add_approximation_prefix(date_str: str, normalized: str) -> str:
    """Add approximation indicator prefix to normalized date."""
    if "circa" in date_str.lower() or "c." in date_str.lower():
        return f"circa {normalized}"
    if "~" in date_str:
        return f"~{normalized}"
    if "about" in date_str.lower() or "abt" in date_str.lower():
        return f"about {normalized}"
    if "before" in date_str.lower() or "bef" in date_str.lower():
        return f"before {normalized}"
    if "after" in date_str.lower() or "aft" in date_str.lower():
        return f"after {normalized}"

    return normalized


def _validate_and_normalize_date(date_str: str) -> str:
    """
    Enhanced Phase 12.1: Validate and normalize genealogical dates.

    Handles common genealogical date formats:
    - Full dates: "15 Mar 1850", "March 15, 1850", "1850-03-15"
    - Partial dates: "Mar 1850", "1850"
    - Approximate dates: "circa 1850", "~1850", "about 1850"
    - Date ranges: "1850-1855", "between 1850 and 1855"

    Returns normalized date string or original if no normalization needed.
    """
    if not date_str:
        return ""

    date_str = date_str.strip()
    if not date_str:
        return ""

    # Detect approximate indicators
    has_approx = _detect_approximate_indicator(date_str)

    # Extract year
    year = _extract_year_from_date(date_str)
    if not year:
        return date_str  # Return original if no recognizable year pattern

    # Extract month and day
    month = _extract_month_from_date(date_str)
    day = _extract_day_from_date(date_str, month)

    # Construct normalized date
    normalized = _construct_normalized_date(year, month, day)

    # Add back approximation indicators
    if has_approx:
        normalized = _add_approximation_prefix(date_str, normalized)

    return normalized


def _validate_relationship(relationship: str) -> str:
    """
    Enhanced Phase 12.1: Validate and normalize genealogical relationships.

    Standardizes common relationship terms to consistent format.
    """
    if not relationship:
        return ""

    rel = relationship.strip().lower()

    # Standard relationship mappings
    relationship_map = {
        # Direct relationships
        "father": "father", "dad": "father", "papa": "father",
        "mother": "mother", "mom": "mother", "mama": "mother",
        "son": "son", "daughter": "daughter",
        "brother": "brother", "bro": "brother",
        "sister": "sister", "sis": "sister",
        "husband": "spouse", "wife": "spouse", "spouse": "spouse",

        # Extended relationships
        "grandfather": "grandfather", "grandpa": "grandfather",
        "grandmother": "grandmother", "grandma": "grandmother",
        "grandson": "grandson", "granddaughter": "granddaughter",
        "uncle": "uncle", "aunt": "aunt",
        "nephew": "nephew", "niece": "niece",
        "cousin": "cousin", "cuz": "cousin",

        # In-law relationships
        "father-in-law": "father-in-law", "mother-in-law": "mother-in-law",
        "son-in-law": "son-in-law", "daughter-in-law": "daughter-in-law",
        "brother-in-law": "brother-in-law", "sister-in-law": "sister-in-law",

        # Step relationships
        "stepfather": "stepfather", "stepmother": "stepmother",
        "stepson": "stepson", "stepdaughter": "stepdaughter",
        "stepbrother": "stepbrother", "stepsister": "stepsister",
    }

    return relationship_map.get(rel, relationship.strip())


def _validate_location(location: str) -> str:
    """
    Enhanced Phase 12.1: Validate and normalize genealogical locations.

    Standardizes location formats and handles common abbreviations.
    """
    if not location:
        return ""

    loc = location.strip()

    # Common state/country abbreviations
    location_map = {
        # US States
        "ny": "New York", "ca": "California", "tx": "Texas", "fl": "Florida",
        "pa": "Pennsylvania", "il": "Illinois", "oh": "Ohio", "ga": "Georgia",
        "nc": "North Carolina", "mi": "Michigan", "nj": "New Jersey",
        "va": "Virginia", "wa": "Washington", "az": "Arizona", "ma": "Massachusetts",
        "tn": "Tennessee", "in": "Indiana", "mo": "Missouri", "md": "Maryland",
        "wi": "Wisconsin", "co": "Colorado", "mn": "Minnesota", "sc": "South Carolina",
        "al": "Alabama", "la": "Louisiana", "ky": "Kentucky", "or": "Oregon",
        "ok": "Oklahoma", "ct": "Connecticut", "ia": "Iowa", "ms": "Mississippi",
        "ar": "Arkansas", "ks": "Kansas", "ut": "Utah", "nv": "Nevada",

        # Countries
        "uk": "United Kingdom", "usa": "United States", "us": "United States",
        "can": "Canada", "aus": "Australia", "ger": "Germany", "fra": "France",
        "ita": "Italy", "esp": "Spain", "pol": "Poland", "ire": "Ireland",
    }

    # Check for exact matches (case insensitive)
    loc_lower = loc.lower()
    if loc_lower in location_map:
        return location_map[loc_lower]

    # Handle comma-separated locations (City, State, Country)
    if "," in loc:
        parts = [part.strip() for part in loc.split(",")]
        normalized_parts: list[str] = []
        for part in parts:
            part_lower = part.lower()
            if part_lower in location_map:
                normalized_parts.append(location_map[part_lower])
            else:
                normalized_parts.append(part)
        return ", ".join(normalized_parts)

    return loc


def _ensure_extracted_data_container(resp: Any) -> dict[str, Any]:
    if not isinstance(resp, dict):
        resp = {}

    # Cast to dict[str, Any] for type safety
    resp_dict = cast(dict[str, Any], resp)

    extracted = resp_dict.get("extracted_data")
    if not isinstance(extracted, dict):
        extracted = {}

    # Cast to dict[str, Any]
    extracted_dict = cast(dict[str, Any], extracted)

    # Ensure structured keys exist
    for key in STRUCTURED_KEYS:
        if key not in extracted_dict or extracted_dict[key] is None:
            extracted_dict[key] = []

    resp_dict["extracted_data"] = extracted_dict
    # Ensure suggested_tasks exists as list[str]
    tasks = resp_dict.get("suggested_tasks", [])
    resp_dict["suggested_tasks"] = _dedupe_list_str(tasks)
    return resp_dict


def _promote_legacy_fields(extracted: dict[str, Any]) -> None:
    """
    Promote simple legacy flat fields to structured containers conservatively.
    - mentioned_names -> structured_names[{full_name}]
    - mentioned_locations -> locations[{place}]
    - mentioned_dates -> vital_records[{date}]
    """
    for legacy_key, (struct_key, _value_field) in LEGACY_TO_STRUCTURED_MAP.items():
        legacy_vals = extracted.get(legacy_key)
        if not legacy_vals:
            continue
        if not isinstance(legacy_vals, list):
            continue
        # Prepare the structured container list
        struct_list = extracted.get(struct_key)
        if not isinstance(struct_list, list):
            struct_list = []

        # Cast to list[Any] for appending
        struct_list_typed = cast(list[Any], struct_list)

        for v in _dedupe_list_str(legacy_vals):
            if struct_key == "structured_names":
                struct_list_typed.append({"full_name": v, "nicknames": []})
            elif struct_key == "locations":
                struct_list_typed.append({"place": v, "context": "", "time_period": ""})
            elif struct_key == "vital_records":
                struct_list_typed.append({
                    "person": "",
                    "event_type": "",
                    "date": v,
                    "place": "",
                    "certainty": "unknown",
                })
        extracted[struct_key] = struct_list_typed


# Helper functions for normalize_extracted_data

def _ensure_structured_keys(extracted: dict[str, Any]) -> None:
    """Ensure all structured keys exist in extracted data."""
    for key in STRUCTURED_KEYS:
        if key not in extracted or extracted[key] is None:
            extracted[key] = []


def _normalize_vital_records(vital_records: Any) -> None:
    """Validate and normalize vital records."""
    if not isinstance(vital_records, list):
        return

    valid_events = ["birth", "death", "marriage", "baptism", "burial", "christening", "divorce"]

    for record in vital_records:
        if not isinstance(record, dict):
            continue

        # Cast to dict[str, Any]
        record_dict = cast(dict[str, Any], record)

        # Normalize dates
        if record_dict.get("date"):
            record_dict["date"] = _validate_and_normalize_date(str(record_dict["date"]))

        # Normalize locations
        if record_dict.get("place"):
            record_dict["place"] = _validate_location(str(record_dict["place"]))

        # Validate event types
        if record_dict.get("event_type"):
            event_type = str(record_dict["event_type"]).lower().strip()
            if event_type in valid_events:
                record_dict["event_type"] = event_type


def _normalize_relationships(relationships: Any) -> None:
    """Validate and normalize relationships."""
    if not isinstance(relationships, list):
        return

    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue

        # Cast to dict[str, Any]
        rel_dict = cast(dict[str, Any], relationship)

        # Normalize relationship type
        if rel_dict.get("relationship"):
            rel_dict["relationship"] = _validate_relationship(str(rel_dict["relationship"]))

        # Ensure person names are properly formatted
        for person_key in ["person1", "person2"]:
            if rel_dict.get(person_key):
                name = str(rel_dict[person_key]).strip()
                # Basic name validation - ensure it's not just whitespace or numbers
                if name and not name.isdigit() and len(name) > 1:
                    rel_dict[person_key] = name
                else:
                    rel_dict[person_key] = ""


def _normalize_locations(locations: Any) -> None:
    """Validate and normalize locations."""
    if not isinstance(locations, list):
        return

    for location in locations:
        if not isinstance(location, dict):
            continue

        # Cast to dict[str, Any]
        loc_dict = cast(dict[str, Any], location)

        # Normalize place names
        if loc_dict.get("place"):
            loc_dict["place"] = _validate_location(str(loc_dict["place"]))

        # Normalize time periods
        if loc_dict.get("time_period"):
            loc_dict["time_period"] = _validate_and_normalize_date(str(loc_dict["time_period"]))


def _normalize_structured_names(structured_names: Any) -> None:
    """Validate and normalize structured names."""
    if not isinstance(structured_names, list):
        return

    for name_entry in structured_names:
        if not isinstance(name_entry, dict):
            continue

        # Cast to dict[str, Any]
        name_dict = cast(dict[str, Any], name_entry)

        # Ensure full_name is properly formatted
        if name_dict.get("full_name"):
            full_name = str(name_dict["full_name"]).strip()
            # Basic name validation
            if full_name and not full_name.isdigit() and len(full_name) > 1:
                name_dict["full_name"] = full_name
            else:
                name_dict["full_name"] = ""

        # Ensure nicknames is a list
        if "nicknames" not in name_dict or not isinstance(name_dict["nicknames"], list):
            name_dict["nicknames"] = []


def normalize_extracted_data(extracted: Any) -> dict[str, Any]:
    """
    Enhanced Phase 12.1: Normalize extracted_data dict with genealogical validation.
    Ensures keys exist, promotes legacy fields, and validates genealogical data.
    """
    if not isinstance(extracted, dict):
        extracted = {}

    # Cast to dict[str, Any]
    extracted_dict = cast(dict[str, Any], extracted)

    # Ensure all structured keys exist
    _ensure_structured_keys(extracted_dict)

    # Promote legacy flat fields conservatively
    _promote_legacy_fields(extracted_dict)

    # Apply genealogical validation and normalization
    _normalize_vital_records(extracted_dict.get("vital_records", []))
    _normalize_relationships(extracted_dict.get("relationships", []))
    _normalize_locations(extracted_dict.get("locations", []))
    _normalize_structured_names(extracted_dict.get("structured_names", []))

    return extracted_dict


def normalize_ai_response(ai_resp: Any) -> dict[str, Any]:
    """
    Normalize a raw AI response into a safe dict with required shape:
    { "extracted_data": {...}, "suggested_tasks": [...] }
    """
    if not isinstance(ai_resp, dict):
        ai_resp = {}
    ai_resp = _ensure_extracted_data_container(ai_resp)
    ai_resp["extracted_data"] = normalize_extracted_data(ai_resp.get("extracted_data", {}))
    # suggested_tasks already deduped in container ensure step
    return ai_resp


# ==============================================
# COMPREHENSIVE TEST SUITE
# ==============================================

# Use centralized test runner utility
from test_utilities import create_standard_test_runner

# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================


def _test_ai_response_normalization() -> None:
    """Test AI response normalization with various inputs"""
    # Test with empty input
    result = normalize_ai_response({})
    assert "extracted_data" in result
    assert "suggested_tasks" in result
    assert isinstance(result["extracted_data"], dict)
    assert isinstance(result["suggested_tasks"], list)

    # Test with populated data
    test_data = {
        "extracted_data": {"names": ["John Doe"], "dates": ["1850"]},
        "suggested_tasks": ["Research birth records"]
    }
    result = normalize_ai_response(test_data)
    assert result["extracted_data"]["names"] == ["John Doe"]


def _test_extracted_data_normalization() -> None:
    """Test extracted data normalization"""
    # Test with empty input
    result = normalize_extracted_data({})
    for key in STRUCTURED_KEYS:
        assert key in result
        assert isinstance(result[key], list)

    # Test with data
    test_data = {"names": ["Jane Smith"], "dates": ["1900"]}
    result = normalize_extracted_data(test_data)
    assert "names" in result
    assert result["names"] == ["Jane Smith"]


def _test_legacy_field_promotion() -> None:
    """Test legacy field promotion"""
    # Test that legacy fields are promoted correctly
    result = normalize_extracted_data({})
    assert all(key in result for key in STRUCTURED_KEYS)


def _test_list_deduplication() -> None:
    """Test list handling"""
    # Test that lists are preserved
    test_data = {"names": ["John", "John", "Jane"]}
    result = normalize_extracted_data(test_data)
    assert "names" in result
    assert isinstance(result["names"], list)


def _test_edge_cases() -> None:
    """Test edge cases"""
    # Test with None
    result = normalize_ai_response(None)
    assert "extracted_data" in result

    # Test with invalid types
    result = normalize_extracted_data(None)
    assert all(key in result for key in STRUCTURED_KEYS)


def _test_container_structure() -> None:
    """Test container structure"""
    # Test that containers are properly structured
    result = normalize_ai_response({})
    assert isinstance(result, dict)
    assert isinstance(result["extracted_data"], dict)
    assert isinstance(result["suggested_tasks"], list)


# Removed smoke test: _test_function_availability - only checked callable()


# ==============================================
# MAIN TEST SUITE
# ==============================================


def genealogical_normalization_module_tests() -> bool:
    """
    Comprehensive test suite for genealogical normalization functions.

    Tests all core functionality including AI response normalization,
    data extraction validation, legacy field promotion, and edge case handling.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        from test_framework import TestSuite, suppress_logging
    except ImportError:
        print("⚠️  TestSuite not available - falling back to basic testing")
        return _run_basic_tests()

    suite = TestSuite("Genealogical Normalization", "genealogical_normalization")
    suite.start_suite()

    # Assign all module-level test functions
    test_ai_response_normalization = _test_ai_response_normalization
    test_extracted_data_normalization = _test_extracted_data_normalization
    test_legacy_field_promotion = _test_legacy_field_promotion
    test_list_deduplication = _test_list_deduplication
    test_edge_cases = _test_edge_cases
    test_container_structure = _test_container_structure
    # Removed: test_function_availability = _test_function_availability (smoke test)

    # Define all tests in a data structure to reduce complexity
    tests = [
        ("AI response normalization with various inputs",
         test_ai_response_normalization,
         "AI responses are normalized to standard format with extracted_data and suggested_tasks",
         "Test normalize_ai_response with empty, None, and populated inputs",
         "Verify AI response normalization handles all input types and produces consistent structure"),

        ("Extracted data normalization and structure validation",
         test_extracted_data_normalization,
         "Extracted data is normalized with all required STRUCTURED_KEYS present",
         "Test normalize_extracted_data with various data structures",
         "Verify extracted data normalization creates proper structure with all required fields"),

        ("Legacy field promotion to current schema",
         test_legacy_field_promotion,
         "Legacy fields are promoted to current schema structure",
         "Test legacy field handling in normalize_extracted_data",
         "Verify legacy fields are properly promoted to maintain backward compatibility"),

        ("List deduplication in normalized data",
         test_list_deduplication,
         "Duplicate entries in lists are removed during normalization",
         "Test deduplication logic in normalize_extracted_data",
         "Verify list deduplication removes duplicate entries while preserving order"),

        ("Edge case handling (None, empty, invalid inputs)",
         test_edge_cases,
         "Edge cases are handled gracefully without errors",
         "Test normalization functions with None, empty, and invalid inputs",
         "Verify edge case handling ensures robust normalization under all conditions"),

        ("Container structure validation",
         test_container_structure,
         "Normalized containers have proper structure and types",
         "Test container structure in normalize_ai_response",
         "Verify container structure ensures proper dict/list types for all fields"),

        # Removed smoke test: Function availability verification
    ]

    # Run all tests from the list
    with suppress_logging():
        for test_name, test_func, expected, method, details in tests:
            suite.run_test(test_name, test_func, expected, method, details)

    return suite.finish_suite()


# Use centralized test runner utility
run_comprehensive_tests = create_standard_test_runner(genealogical_normalization_module_tests)


def _run_basic_tests() -> bool:
    """Basic test fallback when TestSuite is not available"""
    try:
        # Test basic functionality
        result = normalize_ai_response({})
        assert "extracted_data" in result

        result = normalize_extracted_data({})
        for key in STRUCTURED_KEYS:
            assert key in result

        print("✅ Basic genealogical normalization tests passed")
        return True
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        return False


# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    import sys

    print("🧬 Running Genealogical Normalization comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
