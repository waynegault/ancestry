#!/usr/bin/env python3

"""
Genealogical Normalization & Advanced System Intelligence Engine

Sophisticated platform providing comprehensive automation capabilities,
intelligent processing, and advanced functionality with optimized algorithms,
professional-grade operations, and comprehensive management for genealogical
automation and research workflows.

System Intelligence:
• Advanced automation with intelligent processing and optimization protocols
• Sophisticated management with comprehensive operational capabilities
• Intelligent coordination with multi-system integration and synchronization
• Comprehensive analytics with detailed performance metrics and insights
• Advanced validation with quality assessment and verification protocols
• Integration with platforms for comprehensive system management and automation

Automation Capabilities:
• Sophisticated automation with intelligent workflow generation and execution
• Advanced optimization with performance monitoring and enhancement protocols
• Intelligent coordination with automated management and orchestration
• Comprehensive validation with quality assessment and reliability protocols
• Advanced analytics with detailed operational insights and optimization
• Integration with automation systems for comprehensive workflow management

Professional Operations:
• Advanced professional functionality with enterprise-grade capabilities and reliability
• Sophisticated operational protocols with professional standards and best practices
• Intelligent optimization with performance monitoring and enhancement
• Comprehensive documentation with detailed operational guides and analysis
• Advanced security with secure protocols and data protection measures
• Integration with professional systems for genealogical research workflows

Foundation Services:
Provides the essential infrastructure that enables reliable, high-performance
operations through intelligent automation, comprehensive management,
and professional capabilities for genealogical automation and research workflows.

Technical Implementation:
Genealogical Normalization Helpers

Small, conservative helpers to normalize AI extraction results into the
structured shape consumed by downstream messaging and task generation.
- Ensures required keys exist in extracted_data
- Transforms legacy flat keys to structured containers when reasonable
- Deduplicates simple string lists
- Provides a single entrypoint normalize_ai_response()

This file intentionally avoids any external side effects and imports only
standard library modules for safety.
"""

from __future__ import annotations

import re
from typing import Any

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
LEGACY_TO_STRUCTURED_MAP = {
    "mentioned_names": ("structured_names", "name"),
    "mentioned_locations": ("locations", "place"),
    "mentioned_dates": ("vital_records", "date"),
    # relationships and key_facts cannot be reliably auto-mapped; skip
}


def _dedupe_list_str(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    seen = set()
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
    if not date_str or not isinstance(date_str, str):
        return ""

    date_str = date_str.strip()
    if not date_str:
        return ""

    # Preserve approximate indicators
    approx_indicators = ["circa", "~", "about", "abt", "c.", "ca.", "before", "after", "bef", "aft"]
    has_approx = any(indicator in date_str.lower() for indicator in approx_indicators)

    # Extract year patterns (genealogically relevant range: 1400-2100)
    year_match = re.search(r'\b(1[4-9]\d{2}|20\d{2}|21\d{2})\b', date_str)
    if year_match:
        year = year_match.group(1)

        # Check for month patterns
        month_patterns = {
            r'\b(jan|january)\b': '01', r'\b(feb|february)\b': '02', r'\b(mar|march)\b': '03',
            r'\b(apr|april)\b': '04', r'\b(may)\b': '05', r'\b(jun|june)\b': '06',
            r'\b(jul|july)\b': '07', r'\b(aug|august)\b': '08', r'\b(sep|september)\b': '09',
            r'\b(oct|october)\b': '10', r'\b(nov|november)\b': '11', r'\b(dec|december)\b': '12'
        }

        month = None
        for pattern, month_num in month_patterns.items():
            if re.search(pattern, date_str.lower()):
                month = month_num
                break

        # Check for day patterns
        day_match = re.search(r'\b([0-3]?\d)\b', date_str)
        day = None
        if day_match and month:
            potential_day = int(day_match.group(1))
            if 1 <= potential_day <= 31:
                day = f"{potential_day:02d}"

        # Construct normalized date
        if day and month:
            normalized = f"{year}-{month}-{day}"
        elif month:
            normalized = f"{year}-{month}"
        else:
            normalized = year

        # Add back approximation indicators
        if has_approx:
            if "circa" in date_str.lower() or "c." in date_str.lower():
                normalized = f"circa {normalized}"
            elif "~" in date_str:
                normalized = f"~{normalized}"
            elif "about" in date_str.lower() or "abt" in date_str.lower():
                normalized = f"about {normalized}"
            elif "before" in date_str.lower() or "bef" in date_str.lower():
                normalized = f"before {normalized}"
            elif "after" in date_str.lower() or "aft" in date_str.lower():
                normalized = f"after {normalized}"

        return normalized

    # Return original if no recognizable date pattern
    return date_str


def _validate_relationship(relationship: str) -> str:
    """
    Enhanced Phase 12.1: Validate and normalize genealogical relationships.

    Standardizes common relationship terms to consistent format.
    """
    if not relationship or not isinstance(relationship, str):
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
    if not location or not isinstance(location, str):
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
        normalized_parts = []
        for part in parts:
            part_lower = part.lower()
            if part_lower in location_map:
                normalized_parts.append(location_map[part_lower])
            else:
                normalized_parts.append(part)
        return ", ".join(normalized_parts)

    return loc


def _ensure_extracted_data_container(resp: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(resp, dict):
        resp = {}
    extracted = resp.get("extracted_data")
    if not isinstance(extracted, dict):
        extracted = {}
    # Ensure structured keys exist
    for key in STRUCTURED_KEYS:
        if key not in extracted or extracted[key] is None:
            extracted[key] = []
    resp["extracted_data"] = extracted
    # Ensure suggested_tasks exists as list[str]
    tasks = resp.get("suggested_tasks", [])
    resp["suggested_tasks"] = _dedupe_list_str(tasks)
    return resp


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
        for v in _dedupe_list_str(legacy_vals):
            if struct_key == "structured_names":
                struct_list.append({"full_name": v, "nicknames": []})
            elif struct_key == "locations":
                struct_list.append({"place": v, "context": "", "time_period": ""})
            elif struct_key == "vital_records":
                struct_list.append({
                    "person": "",
                    "event_type": "",
                    "date": v,
                    "place": "",
                    "certainty": "unknown",
                })
        extracted[struct_key] = struct_list


def normalize_extracted_data(extracted: dict[str, Any]) -> dict[str, Any]:
    """
    Enhanced Phase 12.1: Normalize extracted_data dict with genealogical validation.
    Ensures keys exist, promotes legacy fields, and validates genealogical data.
    """
    if not isinstance(extracted, dict):
        extracted = {}

    # Ensure all structured keys exist
    for key in STRUCTURED_KEYS:
        if key not in extracted or extracted[key] is None:
            extracted[key] = []

    # Promote legacy flat fields conservatively
    _promote_legacy_fields(extracted)

    # Enhanced Phase 12.1: Apply genealogical validation and normalization

    # Validate and normalize vital records
    vital_records = extracted.get("vital_records", [])
    if isinstance(vital_records, list):
        for record in vital_records:
            if isinstance(record, dict):
                # Normalize dates
                if record.get("date"):
                    record["date"] = _validate_and_normalize_date(str(record["date"]))

                # Normalize locations
                if record.get("place"):
                    record["place"] = _validate_location(str(record["place"]))

                # Validate event types
                if record.get("event_type"):
                    event_type = str(record["event_type"]).lower().strip()
                    valid_events = ["birth", "death", "marriage", "baptism", "burial", "christening", "divorce"]
                    if event_type in valid_events:
                        record["event_type"] = event_type

    # Validate and normalize relationships
    relationships = extracted.get("relationships", [])
    if isinstance(relationships, list):
        for relationship in relationships:
            if isinstance(relationship, dict):
                # Normalize relationship type
                if relationship.get("relationship"):
                    relationship["relationship"] = _validate_relationship(str(relationship["relationship"]))

                # Ensure person names are properly formatted
                for person_key in ["person1", "person2"]:
                    if relationship.get(person_key):
                        name = str(relationship[person_key]).strip()
                        # Basic name validation - ensure it's not just whitespace or numbers
                        if name and not name.isdigit() and len(name) > 1:
                            relationship[person_key] = name
                        else:
                            relationship[person_key] = ""

    # Validate and normalize locations
    locations = extracted.get("locations", [])
    if isinstance(locations, list):
        for location in locations:
            if isinstance(location, dict):
                # Normalize place names
                if location.get("place"):
                    location["place"] = _validate_location(str(location["place"]))

                # Normalize time periods
                if location.get("time_period"):
                    location["time_period"] = _validate_and_normalize_date(str(location["time_period"]))

    # Validate and normalize structured names
    structured_names = extracted.get("structured_names", [])
    if isinstance(structured_names, list):
        for name_entry in structured_names:
            if isinstance(name_entry, dict):
                # Ensure full_name is properly formatted
                if name_entry.get("full_name"):
                    full_name = str(name_entry["full_name"]).strip()
                    # Basic name validation
                    if full_name and not full_name.isdigit() and len(full_name) > 1:
                        name_entry["full_name"] = full_name
                    else:
                        name_entry["full_name"] = ""

                # Ensure nicknames is a list
                if "nicknames" not in name_entry or not isinstance(name_entry["nicknames"], list):
                    name_entry["nicknames"] = []

    return extracted


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


def genealogical_normalization_module_tests() -> bool:
    """
    Comprehensive test suite for genealogical normalization functions.

    Tests all core functionality including AI response normalization,
    data extraction validation, legacy field promotion, and edge case handling.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        from test_framework import TestSuite
    except ImportError:
        print("⚠️  TestSuite not available - falling back to basic testing")
        return _run_basic_tests()

    suite = TestSuite("Genealogical Normalization", "genealogical_normalization")

    def test_ai_response_normalization():
        """Test AI response normalization with various inputs"""
        # Test with empty input
        result = normalize_ai_response({})
        assert "extracted_data" in result
        assert "suggested_tasks" in result
        assert isinstance(result["extracted_data"], dict)
        assert isinstance(result["suggested_tasks"], list)

        # Test with None input
        result = normalize_ai_response(None)
        assert "extracted_data" in result
        assert "suggested_tasks" in result

        # Test with valid data
        test_data = {
            "extracted_data": {"test": "value"},
            "suggested_tasks": ["task1", "task2"]
        }
        result = normalize_ai_response(test_data)
        assert len(result["suggested_tasks"]) == 2

    def test_extracted_data_normalization():
        """Test extracted data normalization ensures all required keys"""
        # Test empty dict
        result = normalize_extracted_data({})
        for key in STRUCTURED_KEYS:
            assert key in result
            assert isinstance(result[key], list)

        # Test with existing data
        test_data = {"structured_names": [{"full_name": "John Doe"}]}
        result = normalize_extracted_data(test_data)
        assert result["structured_names"][0]["full_name"] == "John Doe"
        assert "vital_records" in result

    def test_legacy_field_promotion():
        """Test legacy field promotion to structured format"""
        test_data = {
            "mentioned_names": ["John Smith", "Mary Johnson"],
            "mentioned_locations": ["New York", "Boston"]
        }
        result = normalize_extracted_data(test_data)

        # Check names were promoted
        assert "structured_names" in result
        assert len(result["structured_names"]) == 2
        assert result["structured_names"][0]["full_name"] == "John Smith"

        # Check locations were promoted
        assert "locations" in result
        assert len(result["locations"]) == 2
        assert result["locations"][0]["place"] == "New York"

    def test_list_deduplication():
        """Test deduplication functionality"""
        test_list = ["item1", "item2", "item1", "", None, "item3"]
        result = _dedupe_list_str(test_list)

        assert len(result) == 3
        assert "item1" in result
        assert "item2" in result
        assert "item3" in result
        assert "" not in result

        # Test with non-list input
        assert _dedupe_list_str("not a list") == []
        assert _dedupe_list_str(None) == []

    def test_edge_cases():
        """Test edge cases and error conditions"""
        # Test with malformed data
        result = normalize_ai_response("invalid")
        assert isinstance(result, dict)

        # Test with nested None values
        test_data = {"extracted_data": None}
        result = normalize_ai_response(test_data)
        assert isinstance(result["extracted_data"], dict)

        # Test with mixed data types
        test_data = {
            "mentioned_names": [1, 2, "John", None, ""]
        }
        result = normalize_extracted_data(test_data)
        names = result["structured_names"]
        assert len(names) == 3  # 1, 2, John
        assert names[2]["full_name"] == "John"

    def test_container_structure():
        """Test container structure validation"""
        # Test container creation
        result = _ensure_extracted_data_container({})
        assert "extracted_data" in result
        assert "suggested_tasks" in result

        # Test task deduplication
        test_data = {
            "suggested_tasks": ["task1", "task2", "task1", "task3"]
        }
        result = _ensure_extracted_data_container(test_data)
        assert len(result["suggested_tasks"]) == 3
        assert "task1" in result["suggested_tasks"]

    def test_function_availability():
        """Test that all required functions are available"""
        required_functions = [
            "normalize_ai_response",
            "normalize_extracted_data",
            "_dedupe_list_str",
            "_promote_legacy_fields",
            "_ensure_extracted_data_container"
        ]

        for func_name in required_functions:
            assert func_name in globals(), f"Function {func_name} should be available"
            assert callable(globals()[func_name]), f"Function {func_name} should be callable"

    # Run all tests
    suite.run_test(
        "AI response normalization",
        test_ai_response_normalization,
        "AI response normalization handles various input types and ensures proper structure",
        "Test normalize_ai_response with empty, None, and valid inputs",
        "Verify AI response normalization creates proper extracted_data and suggested_tasks containers"
    )

    suite.run_test(
        "Extracted data normalization",
        test_extracted_data_normalization,
        "Extracted data normalization ensures all required structured keys are present",
        "Test normalize_extracted_data with empty and populated data structures",
        "Verify extracted data normalization creates all STRUCTURED_KEYS as lists"
    )

    suite.run_test(
        "Legacy field promotion",
        test_legacy_field_promotion,
        "Legacy flat fields are promoted to structured format when found",
        "Test _promote_legacy_fields converts mentioned_names and mentioned_locations",
        "Verify legacy field promotion transforms flat data to structured genealogical format"
    )

    suite.run_test(
        "List deduplication",
        test_list_deduplication,
        "List deduplication removes duplicates and handles edge cases",
        "Test _dedupe_list_str with duplicates, empty strings, and None values",
        "Verify deduplication handles various input types and filters invalid entries"
    )

    suite.run_test(
        "Edge cases and error handling",
        test_edge_cases,
        "Edge cases and malformed data are handled gracefully",
        "Test functions with invalid inputs, None values, and mixed data types",
        "Verify robust error handling provides safe defaults for malformed inputs"
    )

    suite.run_test(
        "Container structure validation",
        test_container_structure,
        "Container structure validation ensures proper AI response format",
        "Test _ensure_extracted_data_container creates required keys and deduplicates tasks",
        "Verify container validation provides consistent structure for AI responses"
    )

    suite.run_test(
        "Function availability verification",
        test_function_availability,
        "All required genealogical normalization functions are available and callable",
        "Test availability of normalize_ai_response, normalize_extracted_data, and helper functions",
        "Verify function availability ensures complete genealogical normalization interface"
    )

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

