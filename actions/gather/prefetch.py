from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from core.error_handling import MaxApiFailuresExceededError
from standard_imports import setup_module
from test_framework import TestSuite, create_standard_test_runner

if TYPE_CHECKING:
    from core.session_manager import SessionManager

logger = setup_module(globals(), __name__)


@dataclass(frozen=True)
class PrefetchConfig:
    """Lightweight configuration for Action 6 prefetch orchestration."""

    relationship_prob_max_per_page: int
    enable_ethnicity_enrichment: bool
    ethnicity_min_cm: int
    dna_match_priority_threshold_cm: int
    critical_failure_threshold: int = 10

    def __post_init__(self) -> None:  # pragma: no cover - dataclass wiring
        object.__setattr__(self, "relationship_prob_max_per_page", max(0, self.relationship_prob_max_per_page))
        object.__setattr__(self, "ethnicity_min_cm", max(0, self.ethnicity_min_cm))
        object.__setattr__(self, "dna_match_priority_threshold_cm", max(0, self.dna_match_priority_threshold_cm))
        object.__setattr__(self, "critical_failure_threshold", max(1, self.critical_failure_threshold))


@dataclass(frozen=True)
class PrefetchHooks:
    """Callables required by the prefetch pipeline during the migration."""

    fetch_combined_details: Callable[[SessionManager, str], Optional[dict[str, Any]]]
    fetch_badge_details: Callable[[SessionManager, str], Optional[dict[str, Any]]]
    fetch_ladder_details: Callable[[SessionManager, str, str, Optional[str]], Optional[dict[str, Any]]]
    fetch_ethnicity_batch: Callable[[SessionManager, str], Optional[dict[str, Optional[int]]]]


@dataclass(frozen=True)
class PrefetchResult:
    """Structured return type for sequential API prefetching."""

    prefetched_data: dict[str, dict[str, Any]]
    endpoint_durations: dict[str, float]
    endpoint_counts: dict[str, int]


@dataclass
class _PrefetchStats:
    """Tracks counters for sequential API prefetch operations."""

    critical_failures: int = 0
    ethnicity_fetch_count: int = 0
    ethnicity_skipped: int = 0
    next_progress_threshold_index: int = 0


@dataclass
class _PrefetchPlan:
    """Immutable plan describing how the prefetch run should behave."""

    stats: _PrefetchStats
    badge_candidates: set[str]
    priority_uuids: set[str]
    high_priority_uuids: set[str]
    ethnicity_candidates: set[str]
    num_candidates: int
    my_tree_id: Optional[str]


@dataclass
class _EthnicityScreeningStats:
    """Track screening outcomes for ethnicity enrichment."""

    already_up_to_date: int = 0
    threshold_filtered: int = 0


def _safe_cm_value(match_data: dict[str, Any]) -> int:
    """Return cM value from match data with defensive conversion."""

    try:
        return int(match_data.get("cm_dna", 0) or 0)
    except (TypeError, ValueError):
        logger.debug("Unable to parse cm_dna for priority classification; defaulting to 0")
        return 0


def _determine_match_priority(match_data: dict[str, Any], threshold_cm: int) -> tuple[str, int, bool, bool]:
    """Map match attributes to priority level."""

    cm_value = _safe_cm_value(match_data)
    has_tree = bool(match_data.get("in_my_tree"))
    is_starred = bool(match_data.get("starred"))

    if is_starred or cm_value > 50:
        return "high", cm_value, has_tree, is_starred
    if cm_value > threshold_cm or (cm_value > 5 and has_tree):
        return "medium", cm_value, has_tree, is_starred
    return "low", cm_value, has_tree, is_starred


def _log_priority_decision(
    priority: str,
    uuid_val: str,
    cm_value: int,
    has_tree: bool,
    is_starred: bool,
    log_state: dict[str, int],
    threshold_cm: int,
) -> None:
    """Emit limited debug output for priority classification."""

    emitted = log_state.get(priority, 0)
    suppressed_key = f"suppressed_{priority}"

    if emitted < 5:
        if priority == "high":
            logger.debug(f"High priority match {uuid_val[:8]}: {cm_value} cM, starred={is_starred}")
        elif priority == "medium":
            logger.debug(f"Medium priority match {uuid_val[:8]}: {cm_value} cM, has_tree={has_tree}")
        else:
            logger.debug(
                "Skipping relationship probability fetch for low-priority match %s (%s cM < %s cM threshold, no tree)",
                uuid_val[:8],
                cm_value,
                threshold_cm,
            )
    else:
        log_state[suppressed_key] = log_state.get(suppressed_key, 0) + 1

    log_state[priority] = emitted + 1


def _classify_match_priorities(
    matches_to_process_later: list[dict[str, Any]],
    fetch_candidates_uuid: set[str],
    threshold_cm: int,
) -> tuple[set[str], set[str], set[str]]:
    """Classify matches into priority tiers for API call optimization."""

    high_priority_uuids: set[str] = set()
    medium_priority_uuids: set[str] = set()

    log_state: dict[str, int] = {
        "high": 0,
        "medium": 0,
        "low": 0,
        "suppressed_high": 0,
        "suppressed_medium": 0,
        "suppressed_low": 0,
    }

    for match_data in matches_to_process_later:
        uuid_val = match_data.get("uuid")
        if not uuid_val or uuid_val not in fetch_candidates_uuid:
            continue

        priority, cm_value, has_tree, is_starred = _determine_match_priority(match_data, threshold_cm)

        if priority == "high":
            high_priority_uuids.add(uuid_val)
        elif priority == "medium":
            medium_priority_uuids.add(uuid_val)

        _log_priority_decision(priority, uuid_val, cm_value, has_tree, is_starred, log_state, threshold_cm)

    priority_uuids = high_priority_uuids.union(medium_priority_uuids)
    logger.debug(
        "API Call Filtering: %d high priority, %d medium priority, %d low priority (skipped)",
        len(high_priority_uuids),
        len(medium_priority_uuids),
        len(fetch_candidates_uuid) - len(priority_uuids),
    )

    if log_state["suppressed_high"]:
        logger.debug("Suppressed debug output for %d additional high priority matches", log_state["suppressed_high"])
    if log_state["suppressed_medium"]:
        logger.debug(
            "Suppressed debug output for %d additional medium priority matches", log_state["suppressed_medium"]
        )
    if log_state["suppressed_low"]:
        logger.debug("Suppressed debug output for %d additional low priority matches", log_state["suppressed_low"])

    return high_priority_uuids, medium_priority_uuids, priority_uuids


def _relationship_priority_sort_key(match_data: dict[str, Any]) -> tuple[int, int, str]:
    """Sort priority matches by descending cM, then tree presence."""

    cm_value = _safe_cm_value(match_data)
    has_tree = 0 if match_data.get("in_my_tree") else 1
    uuid_val = match_data.get("uuid") or ""
    return (-cm_value, has_tree, uuid_val)


def _limit_relationship_probability_requests(
    matches_to_process_later: list[dict[str, Any]],
    high_priority_uuids: set[str],
    medium_priority_uuids: set[str],
    max_per_page: int,
) -> tuple[set[str], set[str], int]:
    """Restrict relationship-probability fetches to the highest-value matches."""

    if max_per_page <= 0:
        return high_priority_uuids.union(medium_priority_uuids), set(medium_priority_uuids), 0

    allowed_high = set(high_priority_uuids)
    remaining_slots = max(max_per_page - len(allowed_high), 0)

    if remaining_slots >= len(medium_priority_uuids):
        return allowed_high.union(medium_priority_uuids), set(medium_priority_uuids), 0

    matches_by_uuid = {
        match_data.get("uuid"): match_data
        for match_data in matches_to_process_later
        if match_data.get("uuid") in medium_priority_uuids
    }

    ranked_medium = sorted(matches_by_uuid.values(), key=_relationship_priority_sort_key)
    selected_medium: set[str] = set()
    for match in ranked_medium[:remaining_slots]:
        if not match:
            continue
        uuid_val = match.get("uuid")
        if isinstance(uuid_val, str) and uuid_val:
            selected_medium.add(uuid_val)

    trimmed_count = len(medium_priority_uuids) - len(selected_medium)
    combined = allowed_high.union(selected_medium)
    return combined, selected_medium, trimmed_count


def _match_requires_ethnicity_refresh(match_info: dict[str, Any], stats: _EthnicityScreeningStats) -> bool:
    """Return True if the match should request refreshed ethnicity data."""

    flag = match_info.get("_needs_ethnicity_refresh")
    if flag is None or bool(flag):
        return True

    stats.already_up_to_date += 1
    return False


def _is_below_ethnicity_threshold(
    match_info: dict[str, Any],
    threshold_cm: int,
    stats: _EthnicityScreeningStats,
) -> bool:
    """Return True when the match falls below the enrichment cM threshold."""

    if threshold_cm <= 0:
        return False

    if _safe_cm_value(match_info) < threshold_cm:
        stats.threshold_filtered += 1
        return True

    return False


def _log_ethnicity_skip_counts(stats: _EthnicityScreeningStats, threshold_cm: int) -> None:
    """Emit debug logs summarizing ethnicity enrichment skips."""

    if stats.already_up_to_date:
        logger.debug("🧬 Ethnicity refresh skipped for %d matches already up to date", stats.already_up_to_date)

    if threshold_cm > 0 and stats.threshold_filtered:
        logger.debug(
            "🧬 Ethnicity enrichment threshold %s cM filtered %s priority matches",
            threshold_cm,
            stats.threshold_filtered,
        )


def _determine_ethnicity_candidates(
    matches_to_process_later: list[dict[str, Any]],
    priority_uuids: set[str],
    config: PrefetchConfig,
) -> set[str]:
    """Identify which matches qualify for ethnicity enrichment."""

    if not config.enable_ethnicity_enrichment:
        logger.debug("Ethnicity enrichment disabled via configuration; skipping ethnicity API calls.")
        return set()

    filtered_candidates: set[str] = set()
    screening_stats = _EthnicityScreeningStats()
    threshold_cm = config.ethnicity_min_cm

    for match_data in matches_to_process_later:
        uuid_candidate = match_data.get("uuid")
        if not uuid_candidate or uuid_candidate not in priority_uuids:
            continue

        if not _match_requires_ethnicity_refresh(match_data, screening_stats):
            continue

        if _is_below_ethnicity_threshold(match_data, threshold_cm, screening_stats):
            continue

        filtered_candidates.add(uuid_candidate)

    _log_ethnicity_skip_counts(screening_stats, threshold_cm)

    return filtered_candidates


def _identify_badge_candidates(
    matches_to_process_later: list[dict[str, Any]],
    fetch_candidates_uuid: set[str],
) -> set[str]:
    """Collect UUIDs eligible for badge and ladder enrichment."""

    return {
        match_data["uuid"]
        for match_data in matches_to_process_later
        if match_data.get("in_my_tree") and match_data.get("uuid") in fetch_candidates_uuid
    }


def _prepare_prefetch_plan(
    session_manager: SessionManager,
    fetch_candidates_uuid: set[str],
    matches_to_process_later: list[dict[str, Any]],
    config: PrefetchConfig,
) -> _PrefetchPlan:
    """Derive the prefetch execution plan from current configuration."""

    stats = _PrefetchStats()
    num_candidates = len(fetch_candidates_uuid)
    badge_candidates = _identify_badge_candidates(matches_to_process_later, fetch_candidates_uuid)
    logger.debug("Identified %d candidates for Badge/Ladder fetch.", len(badge_candidates))

    high_priority_uuids, medium_priority_uuids, priority_uuids = _classify_match_priorities(
        matches_to_process_later,
        fetch_candidates_uuid,
        config.dna_match_priority_threshold_cm,
    )

    (
        priority_uuids,
        medium_priority_uuids,
        trimmed_medium_count,
    ) = _limit_relationship_probability_requests(
        matches_to_process_later,
        high_priority_uuids,
        medium_priority_uuids,
        config.relationship_prob_max_per_page,
    )

    if trimmed_medium_count > 0:
        logger.debug(
            "Relationship probability fetch limit active (%s/page). Trimmed %s medium-priority matches.",
            config.relationship_prob_max_per_page,
            trimmed_medium_count,
        )

    ethnicity_candidates = _determine_ethnicity_candidates(matches_to_process_later, priority_uuids, config)

    return _PrefetchPlan(
        stats=stats,
        badge_candidates=badge_candidates,
        priority_uuids=priority_uuids,
        high_priority_uuids=high_priority_uuids,
        ethnicity_candidates=ethnicity_candidates,
        num_candidates=num_candidates,
        my_tree_id=session_manager.my_tree_id,
    )


def _prefetch_combined_details(
    session_manager: SessionManager,
    uuid_val: str,
    hooks: PrefetchHooks,
    stats: _PrefetchStats,
    endpoint_durations: dict[str, float],
    endpoint_counts: dict[str, int],
    config: PrefetchConfig,
) -> Optional[dict[str, Any]]:
    """Fetch combined details and record timing metadata."""

    combined_start = time.time()
    result = _handle_combined_details_fetch(session_manager, uuid_val, hooks, stats, config)
    endpoint_durations["combined_details"] += time.time() - combined_start
    endpoint_counts["combined_details"] += 1
    return result


def _prefetch_relationship_probability(
    uuid_val: str,
    plan: _PrefetchPlan,
    batch_relationship_prob_data: dict[str, Optional[str]],
    endpoint_durations: dict[str, float],
    endpoint_counts: dict[str, int],
    batch_combined_details: dict[str, Optional[dict[str, Any]]],
) -> None:
    """Fetch relationship probability for priority matches."""

    if uuid_val not in plan.priority_uuids:
        return

    rel_start = time.time()
    _fetch_optional_relationship_data(
        uuid_val,
        plan.priority_uuids,
        batch_relationship_prob_data,
        batch_combined_details,
    )
    endpoint_durations["relationship_prob"] += time.time() - rel_start
    endpoint_counts["relationship_prob"] += 1


def _prefetch_badge_metadata(
    session_manager: SessionManager,
    uuid_val: str,
    plan: _PrefetchPlan,
    hooks: PrefetchHooks,
    temp_badge_results: dict[str, Optional[dict[str, Any]]],
    endpoint_durations: dict[str, float],
    endpoint_counts: dict[str, int],
) -> None:
    """Fetch badge metadata for matches tied to the user tree."""

    if uuid_val not in plan.badge_candidates:
        return

    badge_start = time.time()
    _fetch_optional_badge_data(
        session_manager,
        uuid_val,
        plan.badge_candidates,
        temp_badge_results,
        hooks,
    )
    endpoint_durations["badge_details"] += time.time() - badge_start
    endpoint_counts["badge_details"] += 1


def _prefetch_ethnicity_data(
    session_manager: SessionManager,
    uuid_val: str,
    plan: _PrefetchPlan,
    hooks: PrefetchHooks,
    config: PrefetchConfig,
    batch_ethnicity_data: dict[str, Optional[dict[str, Optional[int]]]],
    endpoint_durations: dict[str, float],
    endpoint_counts: dict[str, int],
) -> None:
    """Fetch ethnicity enrichment data when allowed."""

    if not config.enable_ethnicity_enrichment:
        plan.stats.ethnicity_skipped += 1
        batch_ethnicity_data[uuid_val] = None
        return

    if uuid_val not in plan.ethnicity_candidates:
        plan.stats.ethnicity_skipped += 1
        batch_ethnicity_data[uuid_val] = None
        return

    ethnicity_start = time.time()
    _process_ethnicity_candidate(
        session_manager,
        uuid_val,
        plan.ethnicity_candidates,
        batch_ethnicity_data,
        plan.stats,
        hooks,
    )
    endpoint_durations["ethnicity"] += time.time() - ethnicity_start
    endpoint_counts["ethnicity"] += 1


def _log_prefetch_progress(
    processed_count: int,
    num_candidates: int,
    start_time: float,
) -> None:
    """Emit progress updates for lengthy prefetches with ETA."""

    should_log = False
    if num_candidates <= 5 or processed_count % 5 == 0 or processed_count == num_candidates:
        should_log = True

    if not should_log:
        return

    elapsed = time.time() - start_time
    avg_time = elapsed / max(processed_count, 1)
    remaining = num_candidates - processed_count
    eta_seconds = remaining * avg_time
    eta_str = f"{eta_seconds:.0f}s" if eta_seconds < 60 else f"{eta_seconds / 60:.1f}m"
    percent = int((processed_count / num_candidates) * 100) if num_candidates else 100

    logger.info(
        "📊 Prefetch %d/%d (%d%%) | Elapsed: %.1fs | Avg: %.1fs/match | ETA: %s",
        processed_count,
        num_candidates,
        percent,
        elapsed,
        avg_time,
        eta_str,
    )


def _enforce_session_health_for_prefetch(
    session_manager: SessionManager,
    processed_count: int,
    num_candidates: int,
    config: PrefetchConfig,
) -> None:
    """Abort processing if the session health check fails."""

    if processed_count % 10 != 0:
        return

    if session_manager.check_session_health():
        return

    logger.critical("🚨 Session death detected at item %d/%d. Aborting.", processed_count, num_candidates)
    raise MaxApiFailuresExceededError(
        f"Session death detected during sequential processing at item {processed_count}",
        failure_count=processed_count,
        max_failures=config.critical_failure_threshold,
    )


def _raise_prefetch_threshold_if_needed(
    stats: _PrefetchStats,
    config: PrefetchConfig,
    exc: Exception | None = None,
) -> None:
    """Raise when the critical API failure threshold is reached."""

    if stats.critical_failures < config.critical_failure_threshold:
        return

    logger.critical(
        "Exceeded critical API failure threshold (%d/%d). Halting batch.",
        stats.critical_failures,
        config.critical_failure_threshold,
    )
    message = (
        f"Critical API failure threshold reached ({stats.critical_failures} failures of "
        f"{config.critical_failure_threshold})."
    )
    error_kwargs = {
        "failure_count": stats.critical_failures,
        "max_failures": config.critical_failure_threshold,
    }
    if exc is None:
        raise MaxApiFailuresExceededError(message, **error_kwargs)
    raise MaxApiFailuresExceededError(message, **error_kwargs) from exc


def _handle_combined_details_fetch(
    session_manager: SessionManager,
    uuid_val: str,
    hooks: PrefetchHooks,
    stats: _PrefetchStats,
    config: PrefetchConfig,
) -> Optional[dict[str, Any]]:
    """Fetch mandatory combined details for a match and update counters."""

    try:
        combined_result = hooks.fetch_combined_details(session_manager, uuid_val)
        if combined_result is None:
            logger.warning("Combined details for %s returned None.", uuid_val[:8])
            stats.critical_failures += 1
            _raise_prefetch_threshold_if_needed(stats, config)
        return combined_result
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Exception fetching combined details for %s: %s", uuid_val[:8], exc, exc_info=True)
        stats.critical_failures += 1
        _raise_prefetch_threshold_if_needed(stats, config, exc)
        return None


def _fetch_optional_relationship_data(
    uuid_val: str,
    priority_uuids: set[str],
    batch_relationship_prob_data: dict[str, Optional[str]],
    batch_combined_details: dict[str, Optional[dict[str, Any]]],
) -> None:
    """Fetch relationship probability when priority thresholds demand it."""

    if uuid_val not in priority_uuids:
        return

    combined_data = batch_combined_details.get(uuid_val)
    if combined_data and combined_data.get("relationship_str"):
        batch_relationship_prob_data[uuid_val] = combined_data.get("relationship_str")
        return

    logger.debug("Relationship string not found in combined details for %s", uuid_val[:8])
    batch_relationship_prob_data[uuid_val] = None


def _fetch_optional_badge_data(
    session_manager: SessionManager,
    uuid_val: str,
    badge_candidates: set[str],
    temp_badge_results: dict[str, Optional[dict[str, Any]]],
    hooks: PrefetchHooks,
) -> None:
    """Fetch badge metadata for tree members."""

    if uuid_val not in badge_candidates:
        return

    try:
        temp_badge_results[uuid_val] = hooks.fetch_badge_details(session_manager, uuid_val)
    except Exception as exc:  # pragma: no cover - logging only
        logger.error("Exception fetching badge details for %s: %s", uuid_val[:8], exc, exc_info=False)
        temp_badge_results[uuid_val] = None


def _process_ethnicity_candidate(
    session_manager: SessionManager,
    uuid_val: str,
    ethnicity_candidates: set[str],
    batch_ethnicity_data: dict[str, Optional[dict[str, Optional[int]]]],
    stats: _PrefetchStats,
    hooks: PrefetchHooks,
) -> None:
    """Fetch ethnicity data when the match qualifies."""

    if uuid_val not in ethnicity_candidates:
        stats.ethnicity_skipped += 1
        batch_ethnicity_data[uuid_val] = None
        return

    try:
        ethnicity_result = hooks.fetch_ethnicity_batch(session_manager, uuid_val)
        batch_ethnicity_data[uuid_val] = ethnicity_result
        stats.ethnicity_fetch_count += 1
    except Exception as exc:  # pragma: no cover - logging only
        logger.error("Exception fetching ethnicity for %s: %s", uuid_val[:8], exc, exc_info=False)
        batch_ethnicity_data[uuid_val] = None


def _build_cfpid_mapping(temp_badge_results: dict[str, Optional[dict[str, Any]]]) -> tuple[list[str], dict[str, str]]:
    """Translate badge data into CFPID lookup structures."""

    cfpid_to_uuid_map: dict[str, str] = {}
    cfpid_list: list[str] = []

    for uuid_val, badge_result in temp_badge_results.items():
        if not badge_result:
            continue
        cfpid = badge_result.get("their_cfpid")
        if cfpid:
            cfpid_list.append(cfpid)
            cfpid_to_uuid_map[cfpid] = uuid_val

    return cfpid_list, cfpid_to_uuid_map


def _merge_badge_and_ladder_data(
    session_manager: SessionManager,
    hooks: PrefetchHooks,
    cfpid: str,
    uuid_val: str,
    my_tree_id: str,
    badge_data: dict[str, Any],
    enriched_tree_data: dict[str, Optional[dict[str, Any]]],
) -> None:
    """Fetch ladder details and merge them with existing badge data."""

    badge_payload = badge_data or {}
    match_display_name = (
        badge_payload.get("their_firstname") or badge_payload.get("display_name") or badge_payload.get("name")
    )

    try:
        ladder_result = hooks.fetch_ladder_details(
            session_manager,
            cfpid,
            my_tree_id,
            match_display_name,
        )
        if not ladder_result:
            return

        combined_tree_info = badge_payload.copy() if badge_payload else {}
        combined_tree_info.update(ladder_result)
        enriched_tree_data[uuid_val] = combined_tree_info or ladder_result
    except Exception as exc:  # pragma: no cover - logging only
        logger.error(
            "Exception fetching ladder for CFPID %s (UUID %s): %s",
            cfpid,
            (uuid_val or "UNKNOWN")[:8],
            exc,
            exc_info=False,
        )


def _fetch_ladder_details_for_badges(
    session_manager: SessionManager,
    my_tree_id: Optional[str],
    temp_badge_results: dict[str, Optional[dict[str, Any]]],
    hooks: PrefetchHooks,
) -> tuple[dict[str, Optional[dict[str, Any]]], int]:
    """Combine badge data with ladder enrichment where available."""

    if not my_tree_id or not temp_badge_results:
        return dict(temp_badge_results), 0

    cfpid_list, cfpid_to_uuid_map = _build_cfpid_mapping(temp_badge_results)
    enriched_tree_data = dict(temp_badge_results)
    ladder_call_count = 0

    for cfpid in cfpid_list:
        uuid_val = cfpid_to_uuid_map.get(cfpid)
        if not uuid_val:
            continue
        ladder_call_count += 1
        badge_data = temp_badge_results.get(uuid_val) or {}
        _merge_badge_and_ladder_data(
            session_manager,
            hooks,
            cfpid,
            uuid_val,
            my_tree_id,
            badge_data,
            enriched_tree_data,
        )

    return enriched_tree_data, ladder_call_count


def _log_prefetch_summary(
    fetch_duration: float,
    stats: _PrefetchStats,
    endpoint_durations: dict[str, float],
    endpoint_counts: dict[str, int],
    config: PrefetchConfig,
) -> None:
    """Summarize prefetch work after completion with detailed metrics."""

    logger.info("--- Finished SEQUENTIAL API Pre-fetch. Duration: %.2fs ---", fetch_duration)
    logger.info("🔬 API Performance Breakdown:")
    for endpoint, duration in endpoint_durations.items():
        count = endpoint_counts.get(endpoint, 0)
        if count > 0:
            avg = duration / count
            logger.info("   - %-20s: %3d calls | %6.2fs total | %5.2fs avg", endpoint, count, duration, avg)

    if not config.enable_ethnicity_enrichment:
        logger.debug("🧬 Ethnicity enrichment disabled; skipping summary metrics.")
        return

    if stats.ethnicity_fetch_count or stats.ethnicity_skipped:
        logger.debug(
            "🧬 Ethnicity fetches: %s prioritized, %s skipped (low priority)",
            stats.ethnicity_fetch_count,
            stats.ethnicity_skipped,
        )


def perform_api_prefetches(
    session_manager: SessionManager,
    fetch_candidates_uuid: set[str],
    matches_to_process_later: list[dict[str, Any]],
    config: PrefetchConfig,
    hooks: PrefetchHooks,
) -> PrefetchResult:
    """Sequentially fetch supporting API data for Action 6 matches."""

    batch_combined_details: dict[str, Optional[dict[str, Any]]] = {}
    batch_tree_data: dict[str, Optional[dict[str, Any]]] = {}
    batch_relationship_prob_data: dict[str, Optional[str]] = {}
    batch_ethnicity_data: dict[str, Optional[dict[str, Optional[int]]]] = {}

    endpoint_durations: dict[str, float] = {
        "combined_details": 0.0,
        "relationship_prob": 0.0,
        "badge_details": 0.0,
        "ladder_details": 0.0,
        "ethnicity": 0.0,
    }
    endpoint_counts: dict[str, int] = {
        "combined_details": 0,
        "relationship_prob": 0,
        "badge_details": 0,
        "ladder_details": 0,
        "ethnicity": 0,
    }

    if not fetch_candidates_uuid:
        logger.debug("No fetch candidates provided for API pre-fetch.")
        return PrefetchResult(
            prefetched_data={
                "combined": {},
                "tree": {},
                "rel_prob": {},
                "ethnicity": {},
            },
            endpoint_durations=endpoint_durations,
            endpoint_counts=endpoint_counts,
        )

    plan = _prepare_prefetch_plan(
        session_manager,
        fetch_candidates_uuid,
        matches_to_process_later,
        config,
    )

    fetch_start_time = time.time()
    logger.debug("--- Starting SEQUENTIAL API Pre-fetch (%d candidates) ---", plan.num_candidates)

    temp_badge_results: dict[str, Optional[dict[str, Any]]] = {}
    for processed_count, uuid_val in enumerate(fetch_candidates_uuid, start=1):
        _enforce_session_health_for_prefetch(
            session_manager,
            processed_count,
            plan.num_candidates,
            config,
        )

        batch_combined_details[uuid_val] = _prefetch_combined_details(
            session_manager,
            uuid_val,
            hooks,
            plan.stats,
            endpoint_durations,
            endpoint_counts,
            config,
        )

        _prefetch_relationship_probability(
            uuid_val,
            plan,
            batch_relationship_prob_data,
            endpoint_durations,
            endpoint_counts,
            batch_combined_details,
        )

        _prefetch_badge_metadata(
            session_manager,
            uuid_val,
            plan,
            hooks,
            temp_badge_results,
            endpoint_durations,
            endpoint_counts,
        )

        _prefetch_ethnicity_data(
            session_manager,
            uuid_val,
            plan,
            hooks,
            config,
            batch_ethnicity_data,
            endpoint_durations,
            endpoint_counts,
        )

        _log_prefetch_progress(processed_count, plan.num_candidates, fetch_start_time)

    ladder_start = time.time()
    batch_tree_data, ladder_calls = _fetch_ladder_details_for_badges(
        session_manager,
        plan.my_tree_id,
        temp_badge_results,
        hooks,
    )
    endpoint_durations["ladder_details"] += time.time() - ladder_start
    endpoint_counts["ladder_details"] += ladder_calls

    fetch_duration = time.time() - fetch_start_time
    _log_prefetch_summary(fetch_duration, plan.stats, endpoint_durations, endpoint_counts, config)

    prefetched_payload = {
        "combined": batch_combined_details,
        "tree": batch_tree_data,
        "rel_prob": batch_relationship_prob_data,
        "ethnicity": batch_ethnicity_data,
    }

    return PrefetchResult(
        prefetched_data=prefetched_payload,
        endpoint_durations=endpoint_durations,
        endpoint_counts=endpoint_counts,
    )


def get_prefetched_data_for_match(
    uuid_val: str,
    prefetched_data: dict[str, dict[str, Any]],
) -> tuple[
    Optional[dict[str, Any]],
    Optional[dict[str, Any]],
    Optional[str],
    Optional[dict[str, Optional[int]]],
]:
    """Return prefetched payloads for an individual match.

    This helper mirrors the legacy ``_get_prefetched_data_for_match`` function.
    """

    prefetched_combined = prefetched_data.get("combined", {}).get(uuid_val)
    prefetched_tree = prefetched_data.get("tree", {}).get(uuid_val)
    prefetched_rel_prob = prefetched_data.get("rel_prob", {}).get(uuid_val)
    prefetched_ethnicity = prefetched_data.get("ethnicity", {}).get(uuid_val)
    return prefetched_combined, prefetched_tree, prefetched_rel_prob, prefetched_ethnicity


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


def _test_prefetch_config_clamps_values() -> bool:
    cfg = PrefetchConfig(
        relationship_prob_max_per_page=-5,
        enable_ethnicity_enrichment=True,
        ethnicity_min_cm=-10,
        dna_match_priority_threshold_cm=-7,
        critical_failure_threshold=0,
    )
    assert cfg.relationship_prob_max_per_page == 0
    assert cfg.ethnicity_min_cm == 0
    assert cfg.dna_match_priority_threshold_cm == 0
    assert cfg.critical_failure_threshold == 1
    return True


def _test_prefetch_hooks_contract() -> bool:
    def _placeholder(*_args: Any, **_kwargs: Any) -> None:
        return None

    hooks = PrefetchHooks(
        fetch_combined_details=_placeholder,
        fetch_badge_details=_placeholder,
        fetch_ladder_details=_placeholder,
        fetch_ethnicity_batch=_placeholder,
    )
    assert hooks.fetch_combined_details is _placeholder
    return True


def _make_test_hooks(ladder_callable: Callable[..., Optional[dict[str, Any]]]) -> PrefetchHooks:
    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    return PrefetchHooks(
        fetch_combined_details=_noop,
        fetch_badge_details=_noop,
        fetch_ladder_details=ladder_callable,
        fetch_ethnicity_batch=_noop,
    )


def _test_merge_badge_and_ladder_success() -> bool:
    from unittest.mock import MagicMock

    session_manager = MagicMock()
    ladder_payload = {"relationship": "3rd cousin", "confidence": 0.92}
    captured_args: dict[str, Any] = {}

    def _fake_fetch(sess: SessionManager, cfpid: str, tree_id: str, display_name: Optional[str]) -> dict[str, Any]:
        captured_args["sess"] = sess
        captured_args["cfpid"] = cfpid
        captured_args["tree_id"] = tree_id
        captured_args["display_name"] = display_name
        return ladder_payload

    hooks = _make_test_hooks(_fake_fetch)
    enriched: dict[str, Optional[dict[str, Any]]] = {}
    badge_data = {"their_firstname": "Ada", "existing": "value"}

    _merge_badge_and_ladder_data(
        session_manager,
        hooks,
        "cfpid-123",
        "uuid-abc",
        "tree-1",
        badge_data,
        enriched,
    )

    assert captured_args["sess"] is session_manager
    assert captured_args["cfpid"] == "cfpid-123"
    assert captured_args["tree_id"] == "tree-1"
    assert captured_args["display_name"] == "Ada"
    merged_entry = enriched.get("uuid-abc")
    assert merged_entry is not None
    assert merged_entry["existing"] == "value"
    assert merged_entry["relationship"] == "3rd cousin"
    return True


def _test_merge_badge_and_ladder_handles_errors() -> bool:
    from unittest.mock import MagicMock

    session_manager = MagicMock()
    call_count = {"value": 0}

    def _failing_fetch(*_args: Any, **_kwargs: Any) -> None:
        call_count["value"] += 1
        raise RuntimeError("boom")

    hooks = _make_test_hooks(_failing_fetch)
    enriched: dict[str, Optional[dict[str, Any]]] = {}

    _merge_badge_and_ladder_data(
        session_manager,
        hooks,
        "cfpid-err",
        "uuid-err",
        "tree-err",
        {},
        enriched,
    )

    assert call_count["value"] == 1
    assert "uuid-err" not in enriched
    return True


def module_tests() -> bool:
    suite = TestSuite("actions.gather.prefetch", "actions/gather/prefetch.py")
    suite.run_test(
        "Config clamps values",
        _test_prefetch_config_clamps_values,
        "Ensures PrefetchConfig enforces positive integer bounds.",
    )
    suite.run_test(
        "Hooks are storable",
        _test_prefetch_hooks_contract,
        "Validates PrefetchHooks accepts placeholder callables during scaffolding.",
    )
    suite.run_test(
        "Ladder merge success",
        _test_merge_badge_and_ladder_success,
        "Ensures badge and ladder data merge with proper hook invocation.",
    )
    suite.run_test(
        "Ladder merge handles errors",
        _test_merge_badge_and_ladder_handles_errors,
        "Confirms helper swallows ladder errors without mutating results.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = module_tests()
    raise SystemExit(0 if success else 1)
