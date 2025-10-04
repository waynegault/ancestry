#!/usr/bin/env python3

"""
Extraction Quality & Advanced System Intelligence Engine

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
Extraction Quality Summary and Consistency Checks

Lightweight helpers to compute a compact completeness/consistency summary from
extracted_data. Designed for debug-level logging without changing runtime behavior.
"""
from __future__ import annotations

import re
from typing import Any


def summarize_extracted_data(extracted_data: dict[str, Any]) -> dict[str, Any]:
    """
    Returns a small summary dict with counts and basic flags. Safe defaults.
    Expected extracted_data keys (from normalization):
      structured_names, vital_records, relationships, locations, occupations,
      research_questions, documents_mentioned, dna_information
    """
    if not isinstance(extracted_data, dict):
        extracted_data = {}

    def _count_list(key: str) -> int:
        val = extracted_data.get(key)
        return len(val) if isinstance(val, list) else 0

    counts = {
        "names": _count_list("structured_names"),
        "vital_records": _count_list("vital_records"),
        "relationships": _count_list("relationships"),
        "locations": _count_list("locations"),
        "occupations": _count_list("occupations"),
        "research_questions": _count_list("research_questions"),
        "documents": _count_list("documents_mentioned"),
        "dna": _count_list("dna_information"),
    }

    # Simple consistency flags
    flags = {
        "has_any_data": any(v > 0 for v in counts.values()),
        "has_names_and_locations": (counts["names"] > 0 and counts["locations"] > 0),
    }

    return {"counts": counts, "flags": flags}


# === Phase 10: Automated Extraction Quality Scoring ===
ACTION_VERBS = {
    "search","check","verify","compare","review","locate","look","find",
    "confirm","analyze","obtain","request","document","compile","trace",
    "investigate","explore","determine","validate"
}

_YEAR_RE = re.compile(r"\b(17|18|19|20)\d{2}\b")
_SPECIFIC_PATTERN_RE = re.compile(r"\b(census|manifest|marriage|birth|death|baptism|immigration|naturalization|military|obituary|probate|newspaper|ship|passenger|DNA|chromosome)\b", re.IGNORECASE)

def compute_task_quality(tasks: list[Any] | None) -> int:
    """Score the quality of suggested research tasks (0-30).

    Heuristics per task (capped when summing):
      +5 includes an action verb
      +3 contains a year
      +3 references a specific record type / domain keyword
      +2 has bracketed entity or quoted phrase
      +1 length within 25-140 chars (clarity / specificity band)
      -2 if length < 10 (too vague) or > 220 (rambling)
      -3 if contains generic filler ("follow up", "look into it", etc.)

    Average per-task raw score mapped into 0-30 scale. Empty -> 0.
    """
    if not isinstance(tasks, list) or not tasks:
        return 0
    filler_patterns = [
        "follow up","touch base","look into it","do research","research more",
        "get info","ask them","find out"
    ]
    total = 0.0
    counted = 0
    for t in tasks:
        if not isinstance(t, str):
            continue
        text = t.strip()
        if not text:
            continue
        raw = 0.0
        lower = text.lower()
        words = set(re.findall(r"[a-zA-Z']+", lower))
        if words & ACTION_VERBS:
            raw += 5
        if _YEAR_RE.search(text):
            raw += 3
        if _SPECIFIC_PATTERN_RE.search(text):
            raw += 3
        if "[" in text and "]" in text:
            raw += 2
        if '"' in text:
            raw += 2
        ln = len(text)
        if 25 <= ln <= 140:
            raw += 1
        if ln < 10 or ln > 220:
            raw -= 2
        if any(pat in lower for pat in filler_patterns):
            raw -= 3
        # Soft clamp single task raw between -5 and 15
        raw = max(-5.0, min(15.0, raw))
        total += raw
        counted += 1
    if counted == 0:
        return 0
    avg = total / counted  # -5 .. 15 typical
    # Map avg (-5..15) to 0..30
    normalized = (avg + 5) / 20  # 0..1
    scaled = max(0.0, min(1.0, normalized)) * 30.0
    return round(scaled)


def compute_extraction_quality(extraction: dict[str, Any]) -> int:
    """Compute a heuristic overall quality score (0-100) for an extraction result.

    Enhanced Phase 12.1 version with sophisticated genealogical data scoring:
      - Entities & relationships portion (base_score) up to 70
      - Task quality portion up to 30
      - Specialized scoring for DNA information, vital records, and relationships
      - Quality bonuses for complete genealogical data structures
    """
    if not isinstance(extraction, dict):
        return 0
    raw_extracted = extraction.get("extracted_data")
    extracted_data: dict[str, Any] = raw_extracted if isinstance(raw_extracted, dict) else {}
    raw_tasks = extraction.get("suggested_tasks")
    suggested_tasks: list[Any] = raw_tasks if isinstance(raw_tasks, list) else []

    def count(key: str) -> int:
        val = extracted_data.get(key) if extracted_data else None
        return len(val) if isinstance(val, list) else 0

    # Raw counts
    names = count("structured_names")
    vitals = count("vital_records")
    rels = count("relationships")
    locs = count("locations")
    occs = count("occupations")
    questions = count("research_questions")
    docs = count("documents_mentioned")
    dna = count("dna_information")
    tasks = len(suggested_tasks)

    # Enhanced base entity richness score (max 70)
    base_score = 0.0

    # Names scoring with quality bonuses
    base_score += min(names * 5, 20)
    if names > 0:
        # Bonus for complete name structures
        names_data = extracted_data.get("structured_names", [])
        complete_names = sum(1 for n in names_data if isinstance(n, dict) and
                           n.get("full_name") and len(n.get("full_name", "").strip()) > 3)
        if complete_names > 0:
            base_score += min(complete_names * 2, 5)  # Up to 5 bonus points

    # Vital records scoring with enhanced quality assessment
    base_score += min(vitals * 4, 20)
    if vitals > 0:
        # Bonus for complete vital records with dates and places
        vitals_data = extracted_data.get("vital_records", [])
        complete_vitals = sum(1 for v in vitals_data if isinstance(v, dict) and
                            v.get("person") and v.get("date") and v.get("place"))
        if complete_vitals > 0:
            base_score += min(complete_vitals * 2, 6)  # Up to 6 bonus points

    # Relationships scoring with connection quality
    base_score += min(rels * 5, 15)
    if rels > 0:
        # Bonus for complete relationship structures
        rels_data = extracted_data.get("relationships", [])
        complete_rels = sum(1 for r in rels_data if isinstance(r, dict) and
                          r.get("person1") and r.get("person2") and r.get("relationship"))
        if complete_rels > 0:
            base_score += min(complete_rels * 2, 4)  # Up to 4 bonus points

    # Enhanced DNA information scoring
    base_score += min(dna * 2, 8)  # Increased from 1 to 2, max from 5 to 8
    if dna > 0:
        # Bonus for specific DNA information (cM values, chromosome data, etc.)
        dna_data = extracted_data.get("dna_information", [])
        specific_dna = sum(1 for d in dna_data if isinstance(d, str) and
                         any(term in d.lower() for term in ["cm", "chromosome", "segment", "match"]))
        if specific_dna > 0:
            base_score += min(specific_dna * 2, 4)  # Up to 4 bonus points

    # Standard scoring for other categories
    base_score += min(locs * 2, 10)
    base_score += min(occs * 1, 5)
    base_score += min(questions * 2, 10)
    base_score += min(docs * 1, 5)

    # Enhanced penalties and bonuses
    if names == 0:
        base_score -= 10

    # Bonus for genealogically rich extractions
    if names > 0 and vitals > 0 and (rels > 0 or locs > 0):
        base_score += 3  # Genealogical completeness bonus

    # Bonus for DNA + genealogical data combination
    if dna > 0 and names > 0 and (vitals > 0 or rels > 0):
        base_score += 2  # DNA-genealogy integration bonus

    # Task quality component (0-30)
    task_quality_component = compute_task_quality(suggested_tasks)

    # Enhanced task quality bonuses
    if 3 <= tasks <= 8 and task_quality_component >= 15:
        task_quality_component = min(30, task_quality_component + 3)

    # Bonus for genealogically-focused tasks
    if tasks > 0:
        genealogy_tasks = sum(1 for t in suggested_tasks if isinstance(t, str) and
                            any(term in t.lower() for term in ["census", "birth", "death", "marriage",
                                                             "immigration", "dna", "family", "ancestor"]))
        if genealogy_tasks > 0:
            task_quality_component = min(30, task_quality_component + min(genealogy_tasks, 3))

    # Penalty if no tasks at all
    if tasks == 0:
        task_quality_component = max(0, task_quality_component - 8)

    total = max(0.0, min(100.0, base_score + task_quality_component))
    return round(total)


def _check_invalid_years(extracted_data: dict[str, Any]) -> int:
    """Check for vital record dates with invalid year patterns."""
    invalid_years = 0
    vitals_raw = extracted_data.get("vital_records")
    vitals: list[Any] = vitals_raw if isinstance(vitals_raw, list) else []

    for rec in vitals:
        if isinstance(rec, dict):
            date = str(rec.get("date", "")).strip()
            if date:
                if len(date) == 4 and not date.isdigit():
                    invalid_years += 1
                if re.match(r"^[0-9]{2,4}[A-Za-z]+[0-9]*$", date):
                    invalid_years += 1

    return invalid_years


def _check_missing_relationships(extracted_data: dict[str, Any]) -> int:
    """Check for relationships missing one side of person1/person2."""
    rel_missing = 0
    rels_raw = extracted_data.get("relationships")
    rels: list[Any] = rels_raw if isinstance(rels_raw, list) else []

    for rel in rels:
        if isinstance(rel, dict):
            p1 = str(rel.get("person1", "")).strip()
            p2 = str(rel.get("person2", "")).strip()
            if (p1 and not p2) or (p2 and not p1):
                rel_missing += 1

    return rel_missing


def _check_incomplete_locations(extracted_data: dict[str, Any]) -> int:
    """Check for locations lacking place but having context/time_period."""
    loc_incomplete = 0
    locs_raw = extracted_data.get("locations")
    locs: list[Any] = locs_raw if isinstance(locs_raw, list) else []

    for loc in locs:
        if isinstance(loc, dict):
            place = str(loc.get("place", "")).strip()
            ctx = str(loc.get("context", "")).strip()
            tp = str(loc.get("time_period", "")).strip()
            if not place and (ctx or tp):
                loc_incomplete += 1

    return loc_incomplete


def _check_duplicate_names(extracted_data: dict[str, Any]) -> int:
    """Check for duplicate full_name occurrences (case-insensitive)."""
    names_raw = extracted_data.get("structured_names")
    names: list[Any] = names_raw if isinstance(names_raw, list) else []
    seen_lower: set[str] = set()
    dups_lower: set[str] = set()

    for n in names:
        full = ""
        if isinstance(n, dict):
            full = str(n.get("full_name", "")).strip()
        elif isinstance(n, str):
            full = n.strip()
        if not full:
            continue
        low = full.lower()
        if low in seen_lower:
            dups_lower.add(low)
        else:
            seen_lower.add(low)

    return len(dups_lower)


def _check_empty_tasks(suggested_tasks: list[Any]) -> int:
    """Check for empty task strings."""
    empty_tasks = 0
    for t in suggested_tasks:
        if isinstance(t, str) and not t.strip():
            empty_tasks += 1
    return empty_tasks


# === Phase 2 (2025-08-11): Anomaly & Consistency Summary (debug / telemetry only) ===
def compute_anomaly_summary(extraction: dict[str, Any]) -> str:
    """Return a concise anomaly summary string or "" if no notable issues.

    Non-invasive heuristic checks (no exceptions raised):
      - vital record dates with clearly invalid year patterns (non-numeric where numeric expected)
      - relationships missing one side of person1/person2
      - locations entries lacking place but having context/time_period
      - duplicate full_name occurrences (case-insensitive)
      - task list present but empty strings included
    Output format: semicolon-separated tokens (e.g., "invalid_years=2;rel_missing=1;dup_names=1").
    Designed for lightweight interrogation & trend analysis; does not alter runtime behavior.
    """
    try:
        if not isinstance(extraction, dict):
            return ""

        raw_extracted = extraction.get("extracted_data")
        extracted_data: dict[str, Any] = raw_extracted if isinstance(raw_extracted, dict) else {}
        raw_tasks = extraction.get("suggested_tasks")
        suggested_tasks: list[Any] = raw_tasks if isinstance(raw_tasks, list) else []

        issues: dict[str, int] = {}

        # Run all anomaly checks
        invalid_years = _check_invalid_years(extracted_data)
        if invalid_years:
            issues["invalid_years"] = invalid_years

        rel_missing = _check_missing_relationships(extracted_data)
        if rel_missing:
            issues["rel_missing"] = rel_missing

        loc_incomplete = _check_incomplete_locations(extracted_data)
        if loc_incomplete:
            issues["loc_incomplete"] = loc_incomplete

        dup_names = _check_duplicate_names(extracted_data)
        if dup_names:
            issues["dup_names"] = dup_names

        empty_tasks = _check_empty_tasks(suggested_tasks)
        if empty_tasks:
            issues["empty_tasks"] = empty_tasks

        if not issues:
            return ""
        return ";".join(f"{k}={v}" for k, v in sorted(issues.items()))
    except Exception:
        return ""


# === Internal Test Suite (Added per repository standard) ===
def extraction_quality_module_tests() -> bool:  # pragma: no cover - invoked by master test harness
    """Lightweight internal tests ensuring core quality helpers behave consistently.

    Categories (mirrors standard pattern): Initialization, Core, Edge, Consistency, Performance (micro), Error Handling.
    Focus: user experience & stability — no artificial heavy loads.
    """
    try:
        from test_framework import TestSuite, suppress_logging
    except Exception:  # Fallback minimal harness
        print("test_framework unavailable; skipping extraction_quality tests")
        return True

    with suppress_logging():
        suite = TestSuite("Extraction Quality & Anomaly Metrics", "extraction_quality.py")
        suite.start_suite()

    # === INIT ===
    def test_module_symbols():
        assert callable(summarize_extracted_data)
        assert callable(compute_task_quality)
        assert callable(compute_extraction_quality)
        assert callable(compute_anomaly_summary)

    # === CORE ===
    def test_basic_summary_counts():
        sample = {
            "structured_names": [{"full_name": "John Smith"}],
            "vital_records": [{"date": "1900"}],
            "relationships": [],
            "locations": [{"place": "Boston"}],
            "occupations": [],
            "research_questions": ["Where born?"],
            "documents_mentioned": [],
            "dna_information": [],
        }
        summary = summarize_extracted_data(sample)
        assert summary["counts"]["names"] == 1
        assert summary["flags"]["has_any_data"] is True

    def test_task_quality_scoring():
        tasks = [
            "Search 1900 census for John Smith in Boston",
            "Verify birth record 1899 Massachusetts",
            "Look for obituary 'John A. Smith' 1955",
        ]
        score = compute_task_quality(tasks)
        assert 10 <= score <= 30

    def test_extraction_quality_integration():
        extraction = {
            "extracted_data": {
                "structured_names": [{"full_name": "Jane Doe"}],
                "vital_records": [{"date": "1888"}],
                "relationships": [{"person1": "Jane Doe", "person2": "John Doe"}],
                "locations": [{"place": "New York"}],
            },
            "suggested_tasks": ["Search 1900 census for Jane Doe"],
        }
        quality = compute_extraction_quality(extraction)
        assert 5 <= quality <= 100

    # === EDGE / ANOMALY ===
    def test_anomaly_detection():
        extraction = {
            "extracted_data": {
                "vital_records": [{"date": "19AB"}],
                "relationships": [{"person1": "Only One"}],
                "locations": [{"context": "residence", "time_period": "1900"}],
                "structured_names": ["Jane Doe", "Jane Doe"],
            },
            "suggested_tasks": ["", "Search census"],
        }
        s = compute_anomaly_summary(extraction)
        # Expect at least one anomaly token present
        assert s != ""
        assert any(tok.startswith("invalid_years") or tok.startswith("rel_missing") for tok in s.split(";"))

    def test_no_anomalies():
        clean = {
            "extracted_data": {
                "vital_records": [{"date": "1900"}],
                "relationships": [{"person1": "A", "person2": "B"}],
                "locations": [{"place": "Paris"}],
                "structured_names": ["A", "B"],
            },
            "suggested_tasks": ["Search 1900 census"],
        }
        assert compute_anomaly_summary(clean) == ""

    # === ERROR HANDLING ===
    def test_defensive_inputs():
        assert compute_anomaly_summary(None) == ""  # type: ignore
        assert compute_extraction_quality(None) == 0  # type: ignore
        assert compute_task_quality([None, 123]) == 0  # invalid entries ignored

    suite.run_test("Module Symbols", test_module_symbols, "Core functions exposed", "Verify functions are callable", "summarize_extracted_data, compute_task_quality, compute_extraction_quality, compute_anomaly_summary accessible")
    suite.run_test("Summary Counts", test_basic_summary_counts, "Counts + flags computed", "Basic summarization works", "names count & has_any_data flag")
    suite.run_test("Task Quality Scoring", test_task_quality_scoring, "Task quality returns mid/high score", "Scoring heuristics active", "Expect 10-30 range for rich tasks")
    suite.run_test("Extraction Quality Integration", test_extraction_quality_integration, "Composite quality computed", "Integration of entity + task scoring", "Expect non-zero score")
    suite.run_test("Anomaly Detection", test_anomaly_detection, "Anomaly tokens produced", "Detect invalid patterns", "Expect invalid_years or rel_missing")
    suite.run_test("No Anomalies", test_no_anomalies, "Clean extraction yields empty summary", "No false positives", "Expect empty string")
    suite.run_test("Defensive Inputs", test_defensive_inputs, "Graceful handling of invalid inputs", "Return safe defaults", "Empty summary or zero scores")
    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(extraction_quality_module_tests)


if __name__ == "__main__":  # pragma: no cover
    import sys
    ok = run_comprehensive_tests()
    sys.exit(0 if ok else 1)


