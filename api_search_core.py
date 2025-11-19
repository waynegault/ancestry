"""
API Search Core Intelligence & Advanced Genealogical Discovery Engine

Lightweight API search core used by Action 10 and Action 9. Performs TreesUI list search,
scores results using universal GEDCOM scorer, provides table-row formatting compatible with
Action 10, and presents post-selection details (family + relationship path).

Priority 1 Todo #10: API Search Deduplication - Caches search results for 7 days
to prevent duplicate API calls for the same search criteria.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from types import ModuleType
from typing import Any, Callable, TypeAlias, cast

from api_search_utils import get_api_family_details
from api_utils import call_discovery_relationship_api, call_treesui_list_api
from config import config_schema
from gedcom_utils import calculate_match_score
from genealogy_presenter import present_post_selection
from logging_config import logger
from relationship_utils import convert_discovery_api_path_to_unified_format
from universal_scoring import calculate_display_bonuses

CandidateDict: TypeAlias = dict[str, Any]
CandidateList: TypeAlias = list[CandidateDict]
FieldScoreDict: TypeAlias = dict[str, Any]

# -----------------------------
# Scoring helpers (minimal port)
# -----------------------------

# -----------------------------
# API Search Cache Management (Priority 1 Todo #10)
# -----------------------------

# Global cache statistics
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "total_queries": 0,
}


@lru_cache(maxsize=1)
def _get_gedcom_utils_module() -> ModuleType:
    return importlib.import_module("gedcom_utils")


def _get_gedcom_utils_attr(attr: str) -> Any:
    return getattr(_get_gedcom_utils_module(), attr)


def _clean_display_date(value: Any) -> Any:
    cleaner = cast(Callable[[Any], Any], _get_gedcom_utils_attr("_clean_display_date"))
    return cleaner(value)


def _parse_date(value: Any) -> Any:
    parser = cast(Callable[[Any], Any], _get_gedcom_utils_attr("_parse_date"))
    return parser(value)


def _normalize_search_criteria(criteria: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize search criteria for consistent cache key generation.

    Ensures that slight variations in input (e.g., 'John' vs 'john', 1850 vs '1850',
    'givenName' vs 'GivenName') produce the same cache key.

    Args:
        criteria: Raw search criteria dictionary

    Returns:
        Normalized criteria dictionary with lowercase keys and normalized values
    """
    normalized = {}

    for key, value in criteria.items():
        if value is None:
            continue

        # Normalize key to lowercase for consistency
        normalized_key = key.lower()

        # Normalize strings: lowercase and strip whitespace
        if isinstance(value, str):
            # Try to convert numeric strings to int
            if value.strip().isdigit():
                normalized[normalized_key] = int(value.strip())
            else:
                normalized[normalized_key] = value.lower().strip()
        # Normalize numbers to int
        elif isinstance(value, (int, float)):
            normalized[normalized_key] = int(value)
        else:
            normalized[normalized_key] = value

    return normalized


def _generate_cache_key(criteria: dict[str, Any]) -> str:
    """
    Generate SHA256 hash of normalized search criteria for cache key.

    Args:
        criteria: Search criteria dictionary

    Returns:
        64-character SHA256 hex string
    """
    # Normalize criteria
    normalized = _normalize_search_criteria(criteria)

    # Sort keys for consistent JSON serialization
    criteria_json = json.dumps(normalized, sort_keys=True)

    # Generate SHA256 hash
    return hashlib.sha256(criteria_json.encode('utf-8')).hexdigest()


def _get_cached_search_results(cache_key: str, db_session: Any) -> CandidateList | None:
    """
    Retrieve cached search results if available and not expired.

    Args:
        cache_key: SHA256 hash of search criteria
        db_session: SQLAlchemy database session

    Returns:
        Cached results list or None if not found/expired
    """
    try:
        from database import ApiSearchCache

        # Query for unexpired cache entry
        cache_entry = db_session.query(ApiSearchCache).filter(
            ApiSearchCache.search_criteria_hash == cache_key,
            ApiSearchCache.expires_at > datetime.now(timezone.utc)
        ).first()

        if not cache_entry:
            _cache_stats["misses"] += 1
            _cache_stats["total_queries"] += 1
            logger.debug(f"[API Search Cache] MISS: {cache_key[:16]}... (hit rate: {get_api_search_cache_hit_rate():.1f}%)")
            return None

        # Update hit statistics
        cache_entry.hit_count += 1
        cache_entry.last_hit_at = datetime.now(timezone.utc)
        db_session.commit()

        _cache_stats["hits"] += 1
        _cache_stats["total_queries"] += 1

        # Parse cached response
        if cache_entry.api_response_cached:
            results = json.loads(cache_entry.api_response_cached)
            logger.info(
                f"✓ API Search Cache HIT: {cache_entry.result_count} results, "
                f"saved API call (hit rate: {get_api_search_cache_hit_rate():.1f}%, "
                f"entry age: {(datetime.now(timezone.utc) - cache_entry.search_timestamp).days}d)"
            )
            return results

        return None

    except Exception as e:
        logger.error(f"Error retrieving cached search results: {e}", exc_info=True)
        return None


def _store_search_results_in_cache(
    cache_key: str,
    criteria: dict[str, Any],
    results: CandidateList,
    db_session: Any,
) -> None:
    """
    Store API search results in cache for 7 days.

    Args:
        cache_key: SHA256 hash of search criteria
        criteria: Original search criteria
        results: API search results to cache
        db_session: SQLAlchemy database session
    """
    try:
        from database import ApiSearchCache

        # Check if entry already exists (shouldn't, but be safe)
        existing = db_session.query(ApiSearchCache).filter(
            ApiSearchCache.search_criteria_hash == cache_key
        ).first()

        if existing:
            # Update existing entry
            existing.api_response_cached = json.dumps(results)
            existing.result_count = len(results)
            existing.search_timestamp = datetime.now(timezone.utc)
            existing.expires_at = datetime.now(timezone.utc) + timedelta(days=7)
            existing.hit_count = 0
            existing.last_hit_at = None
        else:
            # Create new cache entry
            cache_entry = ApiSearchCache(
                search_criteria_hash=cache_key,
                search_criteria_json=json.dumps(criteria, sort_keys=True, indent=2),
                result_count=len(results),
                api_response_cached=json.dumps(results),
                search_timestamp=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(days=7),
                hit_count=0,
                last_hit_at=None,
            )
            db_session.add(cache_entry)

        db_session.commit()
        logger.debug(f"[API Search Cache] Stored {len(results)} results, expires in 7 days")

    except Exception as e:
        logger.error(f"Error storing search results in cache: {e}", exc_info=True)
        db_session.rollback()


def get_api_search_cache_stats() -> dict[str, Any]:
    """
    Get current API search cache statistics.

    Returns:
        Dictionary with cache performance metrics
    """
    hit_rate = (_cache_stats["hits"] / _cache_stats["total_queries"] * 100) if _cache_stats["total_queries"] > 0 else 0.0

    return {
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "total_queries": _cache_stats["total_queries"],
        "hit_rate_percent": hit_rate,
    }


def get_api_search_cache_hit_rate() -> float:
    """Get current cache hit rate percentage."""
    if _cache_stats["total_queries"] == 0:
        return 0.0
    return (_cache_stats["hits"] / _cache_stats["total_queries"]) * 100


def clear_api_search_cache(db_session: Any) -> int:
    """
    Clear all API search cache entries from database.

    Args:
        db_session: SQLAlchemy database session

    Returns:
        Number of entries deleted
    """
    try:
        from database import ApiSearchCache

        count = db_session.query(ApiSearchCache).delete()
        db_session.commit()

        # Reset statistics
        _cache_stats["hits"] = 0
        _cache_stats["misses"] = 0
        _cache_stats["total_queries"] = 0

        logger.info(f"Cleared {count} API search cache entries")
        return count

    except Exception as e:
        logger.error(f"Error clearing API search cache: {e}", exc_info=True)
        db_session.rollback()
        return 0


def cleanup_expired_api_search_cache(db_session: Any) -> int:
    """
    Remove expired cache entries from database.

    Args:
        db_session: SQLAlchemy database session

    Returns:
        Number of expired entries deleted
    """
    try:
        from database import ApiSearchCache

        count = db_session.query(ApiSearchCache).filter(
            ApiSearchCache.expires_at <= datetime.now(timezone.utc)
        ).delete()
        db_session.commit()

        if count > 0:
            logger.info(f"Cleaned up {count} expired API search cache entries")

        return count

    except Exception as e:
        logger.error(f"Error cleaning up expired cache entries: {e}", exc_info=True)
        db_session.rollback()
        return 0


def report_api_cache_stats_to_performance_monitor(session_manager: Any) -> None:
    """
    Report API search cache statistics to PerformanceMonitor for tracking and alerting.

    Priority 1 Todo #10: Performance monitor integration - tracks cache hit rate and alerts
    when hit rate falls below 60% target threshold.

    Args:
        session_manager: SessionManager instance with performance_monitor attribute

    Example:
        # After running API searches, report cache statistics
        report_api_cache_stats_to_performance_monitor(session_manager)
        # PerformanceMonitor will log warnings if hit rate < 60%
    """
    try:
        # Get performance monitor from session_manager
        perf_monitor = getattr(session_manager, 'performance_monitor', None)
        if not perf_monitor:
            logger.debug("PerformanceMonitor not available in session_manager")
            return

        # Get current cache statistics
        stats = get_api_search_cache_stats()
        hit_rate_percent = get_api_search_cache_hit_rate()

        # Check if we have any queries to report
        if stats['total_queries'] == 0:
            logger.debug("No API search queries to report (total_queries=0)")
            return

        # Report to performance monitor
        perf_monitor.track_cache_hit_rate(
            cache_name="API Search Cache",
            hits=stats['hits'],
            misses=stats['misses'],
            hit_rate=hit_rate_percent,
            target_hit_rate=60.0  # Alert if below 60% (same threshold as relationship cache)
        )

        logger.info(
            f"📊 Reported API Search Cache stats to PerformanceMonitor: "
            f"{stats['hits']} hits, {stats['misses']} misses, "
            f"{hit_rate_percent:.1f}% hit rate"
        )

    except Exception as e:
        logger.warning(f"Error reporting API cache stats to PerformanceMonitor: {e}")


# -----------------------------
# Scoring helpers (minimal port)
# -----------------------------

def _extract_candidate_data(raw: CandidateDict, idx: int, clean: Callable[[Any], str | None]) -> CandidateDict:
    # Extract name - TreesUI parser returns FullName, GivenName, Surname
    full_name = raw.get("FullName") or (f"{raw.get('GivenName', '')} {raw.get('Surname', '')}").strip() or "Unknown"
    pid = raw.get("PersonId", f"Unknown_{idx}")

    def _p(s: str | None) -> str | None:
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


def _calculate_candidate_score(
    cand: CandidateDict,
    criteria: dict[str, Any],
) -> tuple[float, FieldScoreDict, list[str]]:
    try:
        score, field_scores, reasons = calculate_match_score(criteria, cand, None, None)
        normalized_field_scores: FieldScoreDict = dict(field_scores or {})
        return float(score or 0), normalized_field_scores, list(reasons or [])
    except Exception as e:
        logger.error(f"Scoring error for {cand.get('norm_id')}: {e}", exc_info=True)
        return 0.0, {}, []


def _build_processed_candidate(
    raw: CandidateDict,
    cand: CandidateDict,
    score: float,
    field_scores: FieldScoreDict,
    reasons: list[str],
) -> CandidateDict:
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


def _process_and_score_suggestions(suggestions: CandidateList, criteria: dict[str, Any]) -> CandidateList:
    def clean_param(p: Any) -> str | None:
        return (p.strip().lower() if p and isinstance(p, str) else None)
    processed: CandidateList = []
    for idx, raw in enumerate(suggestions or []):
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


def _extract_field_scores_for_display(candidate: CandidateDict) -> FieldScoreDict:
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


def _calc_display_bonuses_wrap(scores: FieldScoreDict) -> FieldScoreDict:
    b = calculate_display_bonuses(scores, key_prefix="_s")
    return {
        "birth_date_score_component": b["birth_date_component"],
        "death_date_score_component": b["death_date_component"],
        "birth_bonus_s_disp": b["birth_bonus"],
        "death_bonus_s_disp": b["death_bonus"],
    }


def _create_table_row_for_candidate(candidate: CandidateDict) -> list[str]:
    s = _extract_field_scores_for_display(candidate)
    b = _calc_display_bonuses_wrap(s)
    name = candidate.get("name", "N/A")
    name_short = name[:30] + ("..." if len(name) > 30 else "")
    name_score = f"[{s['givn_s'] + s['surn_s']}]" + (f"[+{s['name_bonus_orig']}]" if s["name_bonus_orig"] > 0 else "")
    bdate = f"{candidate.get('birth_date', 'N/A')} [{b['birth_date_score_component']}]"
    bp = str(candidate.get("birth_place", "N/A"))
    bplace = f"{(bp[:20] + ('...' if len(bp) > 20 else ''))} [{s['bplace_s']}]" + (f" [+{b['birth_bonus_s_disp']}]" if b["birth_bonus_s_disp"] > 0 else "")
    ddate = f"{candidate.get('death_date', 'N/A')} [{b['death_date_score_component']}]"
    dp = str(candidate.get("death_place", "N/A"))
    dplace = f"{(dp[:20] + ('...' if len(dp) > 20 else ''))} [{s['dplace_s']}]" + (f" [+{b['death_bonus_s_disp']}]" if b["death_bonus_s_disp"] > 0 else "")
    total = int(candidate.get("score", 0))
    alive_pen = int(candidate.get("field_scores", {}).get("alive_penalty", 0))
    total_cell = f"{total}{f' [{alive_pen}]' if alive_pen < 0 else ''}"
    return [str(candidate.get("id", "N/A")), f"{name_short} {name_score}", bdate, bplace, ddate, dplace, total_cell]

# -----------------------------
# Search and post-selection
# -----------------------------


def _get_db_session_for_cache(session_manager: Any) -> Any | None:
    """Best-effort database session retrieval for cache operations."""
    try:
        db_manager = getattr(session_manager, "database_manager", None)
        if db_manager:
            return db_manager.get_session()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Could not get database session for caching: %s", exc)
    return None


def _fetch_cached_results(
    search_criteria: dict[str, Any],
    max_results: int,
    db_session: Any,
) -> tuple[str | None, CandidateList | None]:
    """Return cache key and cached results (if available)."""
    if not db_session:
        return None, None

    cache_key = _generate_cache_key(search_criteria)
    cached_results = _get_cached_search_results(cache_key, db_session)
    if cached_results is None:
        return cache_key, None

    limited = cached_results[: max(1, max_results)]
    return cache_key, limited


def _resolve_base_and_tree(session_manager: Any) -> tuple[str, str | None]:
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


def search_ancestry_api_for_person(
    session_manager: Any,
    search_criteria: dict[str, Any],
    max_results: int = 20,
) -> CandidateList:
    """
    Search Ancestry API for matching persons with caching to prevent duplicate API calls.

    Priority 1 Todo #10: Integrated caching layer - checks database for recent searches (<7 days)
    before calling Ancestry API to reduce API load and improve response time.

    Args:
        session_manager: SessionManager instance with database and API access
        search_criteria: Dict with keys like givenName, surname, birthYear, birthPlace, etc.
        max_results: Maximum number of results to return (default: 20)

    Returns:
        List of processed and scored candidate matches (dicts with id, name, dates, places, score)
    """
    # Step 1: Resolve base_url and tree_id for API calls
    base_url, tree_id = _resolve_base_and_tree(session_manager)
    if not (base_url and tree_id):
        logger.error("Missing base_url or tree_id for API search")
        return []

    # Step 2: Get database session for cache operations and check cache
    db_session = _get_db_session_for_cache(session_manager)
    cache_key, cached_results = _fetch_cached_results(search_criteria, max_results, db_session)
    if cached_results is not None:
        return cached_results

    # Step 4: Cache MISS - make API call to get fresh results
    suggestions = call_treesui_list_api(session_manager, tree_id, base_url, search_criteria) or []
    processed = _process_and_score_suggestions(suggestions, search_criteria)

    # Step 5: Store results in cache for future queries (if database available)
    if db_session and cache_key:
        _store_search_results_in_cache(cache_key, search_criteria, processed, db_session)

    # Step 6: Return limited results to caller
    return processed[: max(1, max_results)]


def _extract_year_from_candidate(
    selected_candidate_processed: CandidateDict,
    field_key: str,
    fallback_key: str,
) -> int | None:
    """Extract and convert year value from candidate data."""
    val = selected_candidate_processed.get("field_scores", {}).get(field_key) or selected_candidate_processed.get(fallback_key)
    try:
        return int(val) if val and str(val).isdigit() else None
    except Exception:
        return None


def _get_relationship_paths(
    session_manager_local: Any,
    person_id: str,
    owner_tree_id: str | None,
    base_url: str,
    owner_name: str,
    target_name: str,
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """Retrieve relationship paths using relation ladder with labels API."""
    from api_utils import call_relation_ladder_with_labels_api

    formatted_path: str | None = None
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


def _build_relationship_line(
    current_person: dict[str, Any],
    next_person: dict[str, Any],
    seen_names: set[str],
) -> str:
    relationship = next_person.get("relationship", "relative")
    next_name = next_person.get("name", "Unknown")
    next_lifespan = next_person.get("lifeSpan", "")
    current_name = current_person.get("name", "Unknown")

    next_years = ""
    lowered_next = next_name.lower()
    if lowered_next not in seen_names:
        next_years = f" ({next_lifespan})" if next_lifespan else ""
        seen_names.add(lowered_next)

    if relationship.startswith("You are the "):
        rel_type = relationship.replace("You are the ", "").replace(f" of {current_name}", "").strip()
        inverse_rel = {
            "son": "father",
            "daughter": "mother",
            "grandson": "grandfather",
            "granddaughter": "grandmother",
        }.get(rel_type, "parent")
        return f"   - {current_name} is the {inverse_rel} of {next_name}{next_years}"

    return f"   - {relationship} is {next_name}{next_years}"


def _format_kinship_persons_path(kinship_persons: CandidateList, owner_name: str) -> str:
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
    for current_person, next_person in zip(kinship_persons, kinship_persons[1:]):
        path_lines.append(_build_relationship_line(current_person, next_person, seen_names))

    # Header
    header = f"Relationship to {owner_name}:"

    return f"{header}\n" + "\n".join(path_lines)


def _handle_supplementary_info_phase(selected_candidate_processed: CandidateDict, session_manager_local: Any) -> None:
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
from test_framework import TestSuite


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
        """Test that search gracefully handles empty suggestions and invalid criteria."""
        from unittest.mock import patch

        def fake_list_api(
            sm: Any,
            tree_id: str,
            base: str,
            criteria: dict[str, Any],
        ) -> CandidateList:
            # Unused parameters are intentional for API compatibility
            _ = (sm, tree_id, base, criteria)
            # API gracefully returns empty list for invalid criteria
            # (when required fields like first_name/surname are missing)
            return []

        class SM:  # minimal session manager
            pass

        # Mock both _resolve_base_and_tree and call_treesui_list_api
        with patch('api_search_core._resolve_base_and_tree') as mock_resolve, \
             patch('api_search_core.call_treesui_list_api', fake_list_api):
            mock_resolve.return_value = ("https://example.com", "tree123")
            # Function gracefully handles invalid input by returning empty list
            results = search_ancestry_api_for_person(SM(), {"GivenName": "John"})
            # Verify graceful degradation: function returns empty list for invalid criteria
            assert results == [], f"Expected empty list for invalid criteria, got {results}"
            print("✅ Function gracefully handles invalid search criteria")

    suite.run_test(
        test_name="Search with empty suggestions",
        test_func=_test_search_empty_suggestions,
        functions_tested="search_ancestry_api_for_person",
        test_summary="Ensure empty suggestions yields empty processed list",
        expected_outcome="Returns [] without error",
    )

    # ===== Priority 1 Todo #10: API Search Cache Tests =====

    def _test_cache_key_generation() -> None:
        """Test cache key generation with normalization."""
        # Same criteria, different case/spacing should produce same key
        criteria1 = {"givenName": "John", "surname": "Smith", "birthYear": 1850}
        criteria2 = {"givenName": "JOHN", "surname": "smith  ", "birthYear": "1850"}

        key1 = _generate_cache_key(criteria1)
        key2 = _generate_cache_key(criteria2)

        assert key1 == key2, f"Normalized criteria should produce same key: {key1} != {key2}"
        assert len(key1) == 64, f"SHA256 hash should be 64 hex characters: {len(key1)}"

    suite.run_test(
        test_name="Cache key generation with normalization",
        test_func=_test_cache_key_generation,
        functions_tested="_generate_cache_key, _normalize_search_criteria",
        test_summary="Verify cache keys are consistent for normalized criteria",
        expected_outcome="Same key for 'John' vs 'JOHN', 1850 vs '1850'",
    )

    def _test_cache_miss_and_store() -> None:
        """Test cache miss returns None, then store works."""
        from unittest.mock import MagicMock

        # Create mock database session
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None

        # Test cache miss
        cache_key = "test_key_123"
        result = _get_cached_search_results(cache_key, mock_session)
        assert result is None, "Cache miss should return None"

        # Test cache store (should not raise exception)
        test_criteria = {"givenName": "Test"}
        test_results = [{"id": "123", "name": "Test Person", "score": 85}]
        _store_search_results_in_cache(cache_key, test_criteria, test_results, mock_session)

        # Verify session methods were called
        assert mock_session.add.called, "Session.add should be called"
        assert mock_session.commit.called, "Session.commit should be called"

    suite.run_test(
        test_name="Cache miss and store operations",
        test_func=_test_cache_miss_and_store,
        functions_tested="_get_cached_search_results, _store_search_results_in_cache",
        test_summary="Verify cache miss returns None and store commits to database",
        expected_outcome="None returned, session.add/commit called",
    )

    def _test_cache_statistics_tracking() -> None:
        """Test cache hit/miss statistics are tracked correctly."""
        stats_ref = _cache_stats
        original_stats = stats_ref.copy()

        try:
            # Reset stats for clean test
            stats_ref.clear()
            stats_ref.update({"hits": 0, "misses": 0, "total_queries": 0})

            # Simulate cache operations (stats updated in _get_cached_search_results)
            stats_ref["hits"] += 2
            stats_ref["misses"] += 3
            stats_ref["total_queries"] = 5

            # Test statistics functions
            stats = get_api_search_cache_stats()
            assert stats["hits"] == 2, f"Expected 2 hits, got {stats['hits']}"
            assert stats["misses"] == 3, f"Expected 3 misses, got {stats['misses']}"
            assert stats["total_queries"] == 5, f"Expected 5 total queries, got {stats['total_queries']}"

            hit_rate = get_api_search_cache_hit_rate()
            expected_rate = (2 / 5) * 100  # 40%
            assert abs(hit_rate - expected_rate) < 0.01, f"Expected ~{expected_rate}% hit rate, got {hit_rate}%"

        finally:
            stats_ref.clear()
            stats_ref.update(original_stats)

    suite.run_test(
        test_name="Cache statistics tracking",
        test_func=_test_cache_statistics_tracking,
        functions_tested="get_api_search_cache_stats, get_api_search_cache_hit_rate",
        test_summary="Verify cache statistics are calculated correctly",
        expected_outcome="Stats show 2 hits, 3 misses, 40% hit rate",
    )

    def _test_cache_clear_operation() -> None:
        """Test cache clear deletes all entries and resets stats."""
        from unittest.mock import MagicMock

        stats_ref = _cache_stats
        original_stats = stats_ref.copy()

        try:
            # Set some stats
            stats_ref.clear()
            stats_ref.update({"hits": 10, "misses": 5, "total_queries": 15})

            # Create mock session
            mock_session = MagicMock()
            mock_query = MagicMock()
            mock_session.query.return_value = mock_query
            mock_query.delete.return_value = 42  # Simulate 42 deleted entries

            # Test clear operation
            deleted_count = clear_api_search_cache(mock_session)

            assert deleted_count == 42, f"Expected 42 deleted, got {deleted_count}"
            assert stats_ref["hits"] == 0, "Stats should be reset after clear"
            assert stats_ref["misses"] == 0, "Stats should be reset after clear"
            assert stats_ref["total_queries"] == 0, "Stats should be reset after clear"

        finally:
            stats_ref.clear()
            stats_ref.update(original_stats)

    suite.run_test(
        test_name="Cache clear operation",
        test_func=_test_cache_clear_operation,
        functions_tested="clear_api_search_cache",
        test_summary="Verify cache clear deletes entries and resets statistics",
        expected_outcome="Database entries deleted, stats reset to 0",
    )

    def _test_cache_expiration_cleanup() -> None:
        """Test expired cache entries are removed."""
        from unittest.mock import MagicMock

        # Create mock session
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_filter = MagicMock()

        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.delete.return_value = 15  # Simulate 15 expired entries

        # Test cleanup
        deleted_count = cleanup_expired_api_search_cache(mock_session)

        assert deleted_count == 15, f"Expected 15 expired entries deleted, got {deleted_count}"
        assert mock_session.commit.called, "Session should commit after delete"

    suite.run_test(
        test_name="Cache expiration cleanup",
        test_func=_test_cache_expiration_cleanup,
        functions_tested="cleanup_expired_api_search_cache",
        test_summary="Verify expired cache entries (>7 days) are removed",
        expected_outcome="Expired entries deleted, session committed",
    )

    def _test_search_with_cache_integration() -> None:
        """Test search_ancestry_api_for_person handles database gracefully."""
        from unittest.mock import MagicMock, patch

        # Test 1: Search without database (should work without caching)
        mock_sm_no_db = MagicMock()
        mock_sm_no_db.database_manager = None  # No database
        mock_sm_no_db.my_tree_id = "test_tree_123"

        with patch('api_search_core.call_treesui_list_api', return_value=[]):
            criteria = {"givenName": "Test"}
            results = search_ancestry_api_for_person(mock_sm_no_db, criteria)
            assert isinstance(results, list), "Should return list even without database"

        # Test 2: Search with database manager that fails (should gracefully fall back)
        mock_sm_fail = MagicMock()
        mock_sm_fail.database_manager.get_session.side_effect = Exception("DB connection failed")
        mock_sm_fail.my_tree_id = "test_tree_123"

        with patch('api_search_core.call_treesui_list_api', return_value=[]):
            results = search_ancestry_api_for_person(mock_sm_fail, criteria)
            assert isinstance(results, list), "Should return list even when DB fails"

        # Test 3: Verify cache functions are called correctly with working DB
        mock_sm_db = MagicMock()
        mock_db_manager = MagicMock()
        mock_db_session = MagicMock()

        # Setup working database mock
        mock_sm_db.database_manager = mock_db_manager
        mock_db_manager.get_session.return_value = mock_db_session
        mock_sm_db.my_tree_id = "test_tree_123"

        # Mock the cache table query to return None (cache miss)
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        with patch('api_search_core.call_treesui_list_api', return_value=[{"id": "1", "name": "Test"}]):
            results = search_ancestry_api_for_person(mock_sm_db, criteria)

            # Verify session methods were called (indicates cache was attempted)
            assert mock_db_session.query.called or mock_db_session.add.called, \
                "Database session should be used for caching"
            assert isinstance(results, list), "Should return list of results"

    suite.run_test(
        test_name="Search with cache integration",
        test_func=_test_search_with_cache_integration,
        functions_tested="search_ancestry_api_for_person (with caching)",
        test_summary="Verify search function integrates cache check and store",
        expected_outcome="Cache checked before API, results stored after",
    )

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(_api_search_core_module_tests)


if __name__ == "__main__":
    run_comprehensive_tests()
