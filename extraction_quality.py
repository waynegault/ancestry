#!/usr/bin/env python3

"""
Extraction Quality Summary and Consistency Checks

Lightweight helpers to compute a compact completeness/consistency summary from
extracted_data. Designed for debug-level logging without changing runtime behavior.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, List
import re


def summarize_extracted_data(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
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

def compute_task_quality(tasks: Optional[List[Any]]) -> int:
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
    return int(round(scaled))


def compute_extraction_quality(extraction: Dict[str, Any]) -> int:
    """Compute a heuristic overall quality score (0-100) for an extraction result.

    Incorporates entity richness plus task specificity (via compute_task_quality).
    Backwards compatible: still returns a single integer. Components allocation:
      - Entities & relationships portion (base_score) up to 70
      - Task quality portion up to 30
    """
    if not isinstance(extraction, dict):
        return 0
    raw_extracted = extraction.get("extracted_data")
    extracted_data: Dict[str, Any] = raw_extracted if isinstance(raw_extracted, dict) else {}
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

    # Base entity richness score (max 70)
    base_score = 0.0
    base_score += min(names * 5, 20)
    base_score += min(vitals * 4, 20)
    base_score += min(rels * 5, 15)
    base_score += min(locs * 2, 10)
    base_score += min(occs * 1, 5)
    base_score += min(questions * 2, 10)
    base_score += min(docs * 1, 5)
    base_score += min(dna * 1, 5)

    # Penalties on base portion
    if names == 0:
        base_score -= 10

    # Task quality component (0-30)
    task_quality_component = compute_task_quality(suggested_tasks)

    # Small bonus if healthy number of tasks and decent specificity
    if 3 <= tasks <= 8 and task_quality_component >= 15:
        task_quality_component = min(30, task_quality_component + 3)

    # Penalty if no tasks at all
    if tasks == 0:
        task_quality_component = max(0, task_quality_component - 8)

    total = max(0.0, min(100.0, base_score + task_quality_component))
    return int(round(total))


# === Phase 2 (2025-08-11): Anomaly & Consistency Summary (debug / telemetry only) ===
def compute_anomaly_summary(extraction: Dict[str, Any]) -> str:
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
        extracted_data: Dict[str, Any] = raw_extracted if isinstance(raw_extracted, dict) else {}
        raw_tasks = extraction.get("suggested_tasks")
        suggested_tasks: List[Any] = raw_tasks if isinstance(raw_tasks, list) else []
        issues: Dict[str, int] = {}

        # Vital record date anomalies
        invalid_years = 0
        vitals_raw = extracted_data.get("vital_records")
        vitals: List[Any] = vitals_raw if isinstance(vitals_raw, list) else []
        for rec in vitals:
            if isinstance(rec, dict):
                date = str(rec.get("date", "")).strip()
                if date:
                    if len(date) == 4 and not date.isdigit():
                        invalid_years += 1
                    if re.match(r"^[0-9]{2,4}[A-Za-z]+[0-9]*$", date):
                        invalid_years += 1
        if invalid_years:
            issues["invalid_years"] = invalid_years

        # Relationships missing one side
        rel_missing = 0
        rels_raw = extracted_data.get("relationships")
        rels: List[Any] = rels_raw if isinstance(rels_raw, list) else []
        for rel in rels:
            if isinstance(rel, dict):
                p1 = str(rel.get("person1", "")).strip()
                p2 = str(rel.get("person2", "")).strip()
                if (p1 and not p2) or (p2 and not p1):
                    rel_missing += 1
        if rel_missing:
            issues["rel_missing"] = rel_missing

        # Incomplete locations
        loc_incomplete = 0
        locs_raw = extracted_data.get("locations")
        locs: List[Any] = locs_raw if isinstance(locs_raw, list) else []
        for loc in locs:
            if isinstance(loc, dict):
                place = str(loc.get("place", "")).strip()
                ctx = str(loc.get("context", "")).strip()
                tp = str(loc.get("time_period", "")).strip()
                if not place and (ctx or tp):
                    loc_incomplete += 1
        if loc_incomplete:
            issues["loc_incomplete"] = loc_incomplete

        # Duplicate names
        names_raw = extracted_data.get("structured_names")
        names: List[Any] = names_raw if isinstance(names_raw, list) else []
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
        if dups_lower:
            issues["dup_names"] = len(dups_lower)

        # Empty task strings
        empty_tasks = 0
        for t in suggested_tasks:
            if isinstance(t, str) and not t.strip():
                empty_tasks += 1
        if empty_tasks:
            issues["empty_tasks"] = empty_tasks

        if not issues:
            return ""
        return ";".join(f"{k}={v}" for k, v in sorted(issues.items()))
    except Exception:
        return ""


