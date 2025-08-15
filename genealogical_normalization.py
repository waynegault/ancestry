#!/usr/bin/env python3

"""
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

from typing import Any, Dict, List

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


def _dedupe_list_str(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
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


def _ensure_extracted_data_container(resp: Dict[str, Any]) -> Dict[str, Any]:
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


def _promote_legacy_fields(extracted: Dict[str, Any]) -> None:
    """
    Promote simple legacy flat fields to structured containers conservatively.
    - mentioned_names -> structured_names[{full_name}]
    - mentioned_locations -> locations[{place}]
    - mentioned_dates -> vital_records[{date}]
    """
    for legacy_key, (struct_key, value_field) in LEGACY_TO_STRUCTURED_MAP.items():
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


def normalize_extracted_data(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize extracted_data dict in-place-like and return it.
    Ensures keys exist and promotes simple legacy fields when present.
    """
    if not isinstance(extracted, dict):
        extracted = {}
    # Ensure all structured keys exist
    for key in STRUCTURED_KEYS:
        if key not in extracted or extracted[key] is None:
            extracted[key] = []
    # Promote legacy flat fields conservatively
    _promote_legacy_fields(extracted)
    return extracted


def normalize_ai_response(ai_resp: Any) -> Dict[str, Any]:
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

def run_comprehensive_tests() -> bool:
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

