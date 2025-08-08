#!/usr/bin/env python3

"""
Extraction Quality Summary and Consistency Checks

Lightweight helpers to compute a compact completeness/consistency summary from
extracted_data. Designed for debug-level logging without changing runtime behavior.
"""
from __future__ import annotations

from typing import Any, Dict


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

