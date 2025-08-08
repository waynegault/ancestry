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

