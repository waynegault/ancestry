"""
API Search Core Intelligence & Advanced Genealogical Discovery Engine

Lightweight API search core used by Action 10 and Action 9. Performs TreesUI list search,
scores results using universal GEDCOM scorer, provides table-row formatting compatible with
Action 10, and presents post-selection details (family + relationship path).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from api_search_utils import get_api_family_details
from api_utils import call_discovery_relationship_api, call_treesui_list_api
from config import config_schema
from gedcom_utils import _clean_display_date, _parse_date, calculate_match_score
from genealogy_presenter import present_post_selection
from logging_config import logger
from relationship_utils import convert_discovery_api_path_to_unified_format
from universal_scoring import calculate_display_bonuses

# -----------------------------
# Scoring helpers (minimal port)
# -----------------------------

def _extract_candidate_data(raw: dict, idx: int, clean: Callable[[Any], Optional[str]]) -> dict[str, Any]:
    # Extract name - TreesUI parser returns FullName, GivenName, Surname
    full_name = raw.get("FullName") or (f"{raw.get('GivenName','')} {raw.get('Surname','')}").strip() or "Unknown"
    pid = raw.get("PersonId", f"Unknown_{idx}")

    def _p(s: Optional[str]) -> Optional[str]:
        return clean(s) if isinstance(s, str) else None

    # Parse dates
    bdate_s, ddate_s = raw.get("BirthDate"), raw.get("DeathDate")
    bdate_o = _parse_date(bdate_s) if callable(_parse_date) and bdate_s else None
    ddate_o = _parse_date(ddate_s) if callable(_parse_date) and ddate_s else None

    # Extract first name and surname from parsed data
    first_name = _p(raw.get("GivenName")) or (_p(full_name.split()[0]) if full_name and full_name != "Unknown" else None)
    surname = _p(raw.get("Surname")) or (_p(full_name.split()[-1]) if full_name and full_name != "Unknown" and len(full_name.split()) > 1 else None)

    return {
        "norm_id": pid,
        "display_id": pid,
        "first_name": first_name,
        "surname": surname,
        "full_name_disp": full_name,
        "gender": raw.get("Gender"),
        "birth_year": raw.get("BirthYear"),
        "birth_date_obj": bdate_o,
        "birth_place_disp": _p(raw.get("BirthPlace")),
        "death_year": raw.get("DeathYear"),
        "death_date_obj": ddate_o,
        "death_place_disp": _p(raw.get("DeathPlace")),
        "is_living": raw.get("IsLiving"),
    }


def _calculate_candidate_score(cand: dict[str, Any], criteria: dict[str, Any]) -> tuple[float, dict[str, Any], list[str]]:
    try:
        score, field_scores, reasons = calculate_match_score(criteria, cand, None, None)
        return float(score or 0), dict(field_scores or {}), list(reasons or [])
    except Exception as e:
        logger.error(f"Scoring error for {cand.get('norm_id')}: {e}", exc_info=True)
        return 0.0, {}, []


def _build_processed_candidate(raw: dict, cand: dict[str, Any], score: float, field_scores: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    bstr, dstr = raw.get("BirthDate"), raw.get("DeathDate")
    return {
        "id": cand.get("norm_id", "Unknown"),
        "name": cand.get("full_name_disp", "Unknown"),
        "gender": cand.get("gender"),
        "birth_date": _clean_display_date(bstr) if callable(_clean_display_date) else (bstr or "N/A"),
        "birth_place": raw.get("BirthPlace", "N/A"),
        "birth_year": cand.get("birth_year"),  # Add birth_year for header display
        "death_date": _clean_display_date(dstr) if callable(_clean_display_date) else (dstr or "N/A"),
        "death_place": raw.get("DeathPlace", "N/A"),
        "death_year": cand.get("death_year"),  # Add death_year for header display
        "score": score,
        "field_scores": field_scores,
        "reasons": reasons,
        "raw_data": raw,
    }


def _process_and_score_suggestions(suggestions: list[dict], criteria: dict[str, Any]) -> list[dict]:
    def clean_param(p: Any) -> Optional[str]:
        return (p.strip().lower() if p and isinstance(p, str) else None)
    processed: list[dict] = []
    for idx, raw in enumerate(suggestions or []):
        if not isinstance(raw, dict):
            continue
        cand = _extract_candidate_data(raw, idx, clean_param)
        score, field_scores, reasons = _calculate_candidate_score(cand, criteria)
        processed.append(_build_processed_candidate(raw, cand, score, field_scores, reasons))
        # Debug logging to see what scores each result is getting
        logger.debug(f"Scored {idx}: {cand.get('full_name_disp')} (b. {cand.get('birth_year')} in {cand.get('birth_place_disp')}) = {score} points")
    processed.sort(key=lambda x: x.get("score", 0), reverse=True)
    logger.debug(f"Top 3 scored results: {[(p.get('name'), p.get('birth_date'), p.get('score')) for p in processed[:3]]}")
    return processed

# -----------------------------
# Display helpers (Action 10 compatible)
# -----------------------------

def _extract_field_scores_for_display(candidate: dict) -> dict[str, int]:
    fs = candidate.get("field_scores", {}) or {}
    return {
        "givn_s": int(fs.get("givn", 0)),
        "surn_s": int(fs.get("surn", 0)),
        "name_bonus_orig": int(fs.get("bonus", 0)),
        "gender_s": int(fs.get("gender", 0)),
        "byear_s": int(fs.get("byear", 0)),
        "bdate_s": int(fs.get("bdate", 0)),
        "bplace_s": int(fs.get("bplace", 0)),
        "dyear_s": int(fs.get("dyear", 0)),
        "ddate_s": int(fs.get("ddate", 0)),
        "dplace_s": int(fs.get("dplace", 0)),
    }


def _calc_display_bonuses_wrap(scores: dict[str, int]) -> dict[str, int]:
    b = calculate_display_bonuses(scores, key_prefix="_s")
    return {
        "birth_date_score_component": b["birth_date_component"],
        "death_date_score_component": b["death_date_component"],
        "birth_bonus_s_disp": b["birth_bonus"],
        "death_bonus_s_disp": b["death_bonus"],
    }


def _create_table_row_for_candidate(candidate: dict) -> list[str]:
    s = _extract_field_scores_for_display(candidate)
    b = _calc_display_bonuses_wrap(s)
    name = candidate.get("name", "N/A")
    name_short = name[:30] + ("..." if len(name) > 30 else "")
    name_score = f"[{s['givn_s'] + s['surn_s']}]" + (f"[+{s['name_bonus_orig']}]" if s["name_bonus_orig"] > 0 else "")
    bdate = f"{candidate.get('birth_date','N/A')} [{b['birth_date_score_component']}]"
    bp = str(candidate.get("birth_place", "N/A"))
    bplace = f"{(bp[:20] + ('...' if len(bp)>20 else ''))} [{s['bplace_s']}]" + (f" [+{b['birth_bonus_s_disp']}]" if b["birth_bonus_s_disp"]>0 else "")
    ddate = f"{candidate.get('death_date','N/A')} [{b['death_date_score_component']}]"
    dp = str(candidate.get("death_place", "N/A"))
    dplace = f"{(dp[:20] + ('...' if len(dp)>20 else ''))} [{s['dplace_s']}]" + (f" [+{b['death_bonus_s_disp']}]" if b["death_bonus_s_disp"]>0 else "")
    total = int(candidate.get("score", 0))
    alive_pen = int(candidate.get("field_scores", {}).get("alive_penalty", 0))
    total_cell = f"{total}{f' [{alive_pen}]' if alive_pen < 0 else ''}"
    return [str(candidate.get("id","N/A")), f"{name_short} {name_score}", bdate, bplace, ddate, dplace, total_cell]

# -----------------------------
# Search and post-selection
# -----------------------------

def _resolve_base_and_tree(session_manager: Any) -> tuple[str, Optional[str]]:
    """
    Resolve base_url and tree_id for API calls.

    In browserless mode, session_manager.my_tree_id may be None because get_my_tree_id()
    requires a browser driver. So we fall back to test config for testing.
    """
    base_url = getattr(config_schema.api, "base_url", "") or ""
    # Try session_manager first (works when browser is active)
    tree_id = getattr(session_manager, "my_tree_id", None)
    # Fall back to test config for testing
    if not tree_id:
        tree_id = getattr(config_schema.test, "test_tree_id", None)
    return base_url, str(tree_id) if tree_id else None


def search_ancestry_api_for_person(session_manager: Any, search_criteria: dict[str, Any], max_results: int = 20) -> list[dict]:
    base_url, tree_id = _resolve_base_and_tree(session_manager)
    if not (base_url and tree_id):
        logger.error("Missing base_url or tree_id for API search")
        return []
    suggestions = call_treesui_list_api(session_manager, tree_id, base_url, search_criteria) or []
    processed = _process_and_score_suggestions(suggestions, search_criteria)
    return processed[: max(1, max_results)]


def _extract_year_from_candidate(selected_candidate_processed: dict, field_key: str, fallback_key: str) -> Optional[int]:
    """Extract and convert year value from candidate data."""
    val = selected_candidate_processed.get("field_scores", {}).get(field_key) or selected_candidate_processed.get(fallback_key)
    try:
        return int(val) if val and str(val).isdigit() else None
    except Exception:
        return None


def _get_relationship_paths(
    session_manager_local: Any,
    person_id: str,
    owner_tree_id: Optional[str],
    base_url: str,
    owner_name: str,
    target_name: str,
) -> tuple[Optional[str], Optional[list]]:
    """Retrieve relationship paths using relation ladder with labels API."""
    from api_utils import call_relation_ladder_with_labels_api

    formatted_path: Optional[str] = None
    unified_path = None

    # Try relation ladder with labels API first (best option - returns clean JSON with names and dates)
    if owner_tree_id:
        owner_profile_id = getattr(session_manager_local, "my_profile_id", None) or getattr(config_schema, "reference_person_id", None)
        if owner_profile_id:
            ladder_data = call_relation_ladder_with_labels_api(
                session_manager_local, owner_profile_id, owner_tree_id, person_id, base_url, timeout=20
            )
            if ladder_data and "kinshipPersons" in ladder_data:
                # Convert to formatted path
                formatted_path = _format_kinship_persons_path(ladder_data["kinshipPersons"], owner_name)

    # Fall back to discovery API if needed
    if not formatted_path:
        owner_profile_id = getattr(session_manager_local, "my_profile_id", None) or getattr(config_schema, "reference_person_id", None)
        if owner_profile_id:
            disc = call_discovery_relationship_api(session_manager_local, person_id, str(owner_profile_id), base_url, timeout=20)
            if isinstance(disc, dict):
                unified_path = convert_discovery_api_path_to_unified_format(disc, target_name)

    return formatted_path, unified_path


def _format_kinship_persons_path(kinship_persons: list[dict], owner_name: str) -> str:
    """Format kinshipPersons array from relation ladder API into readable path."""
    if not kinship_persons or len(kinship_persons) < 2:
        return "(No relationship path available)"

    # Build the relationship path
    path_lines = []

    # Track names we've seen to avoid repeating years
    seen_names = set()

    # Add first person as standalone line with years
    first_person = kinship_persons[0]
    first_name = first_person.get("name", "Unknown")
    first_lifespan = first_person.get("lifeSpan", "")
    first_years = f" ({first_lifespan})" if first_lifespan else ""
    path_lines.append(f"   - {first_name}{first_years}")
    seen_names.add(first_name.lower())

    # Process remaining path steps without repeating the person's name at the start
    for i in range(len(kinship_persons) - 1):
        current_person = kinship_persons[i]
        next_person = kinship_persons[i + 1]

        relationship = next_person.get("relationship", "relative")
        next_name = next_person.get("name", "Unknown")
        next_lifespan = next_person.get("lifeSpan", "")
        current_name = current_person.get("name", "Unknown")

        # Format lifespan - only if we haven't seen this name before
        next_years = ""
        if next_name.lower() not in seen_names:
            next_years = f" ({next_lifespan})" if next_lifespan else ""
            seen_names.add(next_name.lower())

        # Handle "You are..." relationships specially (convert to inverse relationship)
        # e.g., "You are the son of Derrick" -> "Derrick is the father of Wayne"
        if relationship.startswith("You are the "):
            # Extract the relationship type (e.g., "son", "daughter")
            rel_type = relationship.replace("You are the ", "").replace(f" of {current_name}", "").strip()
            # Convert to inverse relationship
            inverse_rel = {
                "son": "father",
                "daughter": "mother",
                "grandson": "grandfather",
                "granddaughter": "grandmother",
            }.get(rel_type, "parent")
            line = f"   - {current_name} is the {inverse_rel} of {next_name}{next_years}"
        else:
            # Normal relationship format
            line = f"   - {relationship} is {next_name}{next_years}"

        path_lines.append(line)

    # Header
    header = f"Relationship to {owner_name}:"

    return f"{header}\n" + "\n".join(path_lines)


def _handle_supplementary_info_phase(selected_candidate_processed: dict, session_manager_local: Any) -> None:
    try:
        person_id = str(selected_candidate_processed.get("id"))
        base_url, owner_tree_id = _resolve_base_and_tree(session_manager_local)
        owner_name = getattr(session_manager_local, "tree_owner_name", None) or getattr(config_schema, "reference_person_name", "Reference Person")
        target_name = selected_candidate_processed.get("name", "Target Person")
        # Family details
        family_data = get_api_family_details(session_manager_local, person_id, owner_tree_id)
        # Relationship paths
        formatted_path, unified_path = _get_relationship_paths(
            session_manager_local, person_id, owner_tree_id, base_url, owner_name, target_name
        )
        # Extract years for header
        birth_year = _extract_year_from_candidate(selected_candidate_processed, "b_year", "birth_year")
        death_year = _extract_year_from_candidate(selected_candidate_processed, "d_year", "death_year")
        # Present results
        present_post_selection(
            display_name=selected_candidate_processed.get("name", "Unknown"),
            birth_year=birth_year,
            death_year=death_year,
            family_data=family_data or {"parents": [], "siblings": [], "spouses": [], "children": []},
            owner_name=owner_name,
            relation_labels=None,
            unified_path=unified_path,
            formatted_path=formatted_path,
        )
    except Exception as e:
        logger.error(f"Post-selection presentation failed: {e}", exc_info=True)

__all__ = [
    "_create_table_row_for_candidate",
    "_extract_year_from_candidate",
    "_get_relationship_paths",
    "_handle_supplementary_info_phase",
    "search_ancestry_api_for_person",
]

# --- Internal Tests ---
from test_framework import TestSuite  # type: ignore[import-not-found]


def _api_search_core_module_tests() -> bool:
    """Basic internal tests for api_search_core key functions."""
    suite = TestSuite("api_search_core", __name__)
    suite.start_suite()

    def _test_table_row_formatting() -> None:
        candidate = {
            "id": 1,
            "name": "Johnathan Doe the Elder",
            "birth_date": "1900",
            "death_date": "1970",
            "birth_place": "Exampletown, Scotland",
            "death_place": "Sampleville, England",
            "field_scores": {"alive_penalty": 0, "bplace_s": 0, "dplace_s": 0, "givn_s": 0, "surn_s": 0},
            "score": 42,
        }
        row = _create_table_row_for_candidate(candidate)
        assert isinstance(row, list) and len(row) == 7

    suite.run_test(
        test_name="Table row formatting",
        test_func=_test_table_row_formatting,
        functions_tested="_create_table_row_for_candidate",
        test_summary="Ensure candidate table row produces 7 columns with strings",
        expected_outcome="List of 7 strings returned without exceptions",
    )

    def _test_resolve_base_and_tree() -> None:
        class Dummy:  # simple object without attributes
            pass
        dummy = Dummy()
        base_url, tree_id = _resolve_base_and_tree(dummy)
        assert isinstance(base_url, str)
        # tree_id may be None if not configured; only assert type when present
        assert (tree_id is None) or isinstance(tree_id, str)

    suite.run_test(
        test_name="Resolve base and tree",
        test_func=_test_resolve_base_and_tree,
        functions_tested="_resolve_base_and_tree",
        test_summary="Verify base_url is a string and tree_id is optional string",
        expected_outcome="Returns tuple[str, Optional[str]]",
    )

    def _test_search_empty_suggestions() -> None:
        # Use unittest.mock.patch instead of global statement
        from unittest.mock import patch

        def fake_list_api(sm: Any, tree_id: str, base: str, criteria: dict[str, Any]) -> list[dict]:
            # Unused parameters are intentional for API compatibility
            _ = (sm, tree_id, base, criteria)
            return []

        class SM:  # minimal session manager
            pass

        with patch('api_search_core.call_treesui_list_api', fake_list_api):
            results = search_ancestry_api_for_person(SM(), {"GivenName": "John"})
            assert isinstance(results, list) and len(results) == 0

    suite.run_test(
        test_name="Search with empty suggestions",
        test_func=_test_search_empty_suggestions,
        functions_tested="search_ancestry_api_for_person",
        test_summary="Ensure empty suggestions yields empty processed list",
        expected_outcome="Returns [] without error",
    )

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner  # type: ignore[import-not-found]

run_comprehensive_tests = create_standard_test_runner(_api_search_core_module_tests)


if __name__ == "__main__":
    run_comprehensive_tests()
