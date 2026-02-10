
import contextlib
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timezone
from typing import Any, cast
from urllib.parse import urljoin

import requests
from requests.exceptions import ConnectionError

from actions.gather.performance_logging import log_api_performance
from api.api_constants import (
    API_PATH_MATCH_BADGE_DETAILS,
    API_PATH_MATCH_DETAILS,
    API_PATH_PROFILE_DETAILS,
)
from caching.cache import cache as disk_cache
from config import config_schema
from core.api_manager import RequestConfig, RetryPolicy
from core.session_manager import SessionManager
from core.unified_cache_manager import get_unified_cache
from core.utils import format_name
from genealogy.dna.dna_ethnicity_utils import (
    extract_match_ethnicity_percentages,
    fetch_ethnicity_comparison,
    load_ethnicity_metadata,
)
from research.relationship_utils import (
    convert_api_path_to_unified_format,
    format_relationship_path_unified,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response Parsing Helpers (moved from action6_gather to eliminate circular import)
# ---------------------------------------------------------------------------


def _extract_relationship_string(predictions: list[Any]) -> str | None:
    """Extract formatted relationship string from predictions."""
    if not predictions:
        return None

    valid_preds: list[dict[str, Any]] = []
    for candidate in predictions:
        if isinstance(candidate, dict) and "distributionProbability" in candidate and "pathsToMatch" in candidate:
            valid_preds.append(candidate)

    if not valid_preds:
        return None

    best_pred = max(valid_preds, key=lambda x: x.get("distributionProbability", 0.0))
    top_prob_raw = best_pred.get("distributionProbability", 0.0)
    top_prob = float(top_prob_raw) if isinstance(top_prob_raw, (int, float)) else 0.0

    paths_raw = best_pred.get("pathsToMatch", [])
    path_entries = paths_raw if isinstance(paths_raw, Sequence) else []

    labels: list[str] = []
    for path in path_entries:
        if isinstance(path, dict):
            path_dict = cast(dict[str, Any], path)
            label_value = path_dict.get("label")
            if isinstance(label_value, str):
                labels.append(label_value)

    max_labels_param = 2
    final_labels = labels[:max_labels_param]
    relationship_str_val = " or ".join(map(str, final_labels))
    return f"{relationship_str_val} [{top_prob:.1f}%]"


def _parse_details_response(details_response: Any, match_uuid: str) -> dict[str, Any] | None:
    """Parse match details API response."""
    if not (details_response and isinstance(details_response, dict)):
        if isinstance(details_response, requests.Response):
            logger.error(
                f"Match Details API failed for UUID {match_uuid}. Status: {details_response.status_code} {details_response.reason}"
            )
        else:
            logger.error(f"Match Details API did not return dict for UUID {match_uuid}. Type: {type(details_response)}")
        return None

    details_dict = cast(dict[str, Any], details_response)
    relationship_part = cast(dict[str, Any], details_dict.get("relationship", {}))

    # Extract relationship predictions
    predictions = details_dict.get("predictions", [])
    relationship_str = _extract_relationship_string(predictions)

    return {
        "admin_profile_id": details_dict.get("adminUcdmId"),
        "admin_username": details_dict.get("adminDisplayName"),
        "tester_profile_id": details_dict.get("userId"),
        "tester_username": details_dict.get("displayName"),
        "tester_initials": details_dict.get("displayInitials"),
        "gender": details_dict.get("subjectGender"),
        "shared_segments": relationship_part.get("sharedSegments"),
        "longest_shared_segment": relationship_part.get("longestSharedSegment"),
        "meiosis": relationship_part.get("meiosis"),
        "from_my_fathers_side": bool(details_dict.get("fathersSide", False)),
        "from_my_mothers_side": bool(details_dict.get("mothersSide", False)),
        "relationship_str": relationship_str,
    }


def _call_api_request(
    url: str,
    session_manager: SessionManager | None = None,
    method: str = "GET",
    data: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    use_csrf_token: bool = False,
    headers: dict[str, str] | None = None,
    referer_url: str | None = None,
    api_description: str = "API Request",
    _allow_redirects: bool = True,
    force_text_response: bool = False,
) -> Any:
    """
    Make an API request using SessionManager's APIManager.
    Replaces legacy _call_api_request / _api_req.
    """
    if session_manager is None:
        raise ValueError("session_manager is required for API requests")

    # Handle 'json' vs 'json_data'
    final_json_data = json_data if json_data is not None else json

    # Construct headers
    final_headers = headers.copy() if headers else {}
    if referer_url:
        final_headers["Referer"] = referer_url

    config = RequestConfig(
        url=url,
        method=method,
        data=data,
        json_data=final_json_data,
        headers=final_headers,
        use_csrf_token=use_csrf_token,
        api_description=api_description,
        allow_redirects=_allow_redirects,
        retry_policy=RetryPolicy.API,
        force_text_response=force_text_response,
    )

    result = session_manager.api_manager.request(
        config,
        browser_manager=session_manager.browser_manager,
        rate_limiter=session_manager.rate_limiter,
        session_manager=session_manager,
    )

    if result.success:
        # Fix for Action 6 regression:
        # If JSON parsing failed (returning string data) but we didn't request text,
        # return the response object so the caller can handle 303s/errors.
        if isinstance(result.data, str) and not force_text_response:
            return result.response
        return result.data

    # If failed, return response object if available (legacy behavior expected by some callers)
    if result.response:
        return result.response

    return None


def _get_api_headers() -> dict[str, str]:
    """Get standard API headers for match details requests."""
    return {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "priority": "u=0, i",
        "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    }


def _sync_session_cookies(session_manager: SessionManager) -> None:
    """Sync session cookies if available."""
    try:
        session_manager.sync_cookies_to_requests()
    except Exception as cookie_sync_error:
        logger.warning("Session-level cookie sync hint failed (ignored): %s", cookie_sync_error)


def _ensure_action6_session_ready(
    session_manager: SessionManager,
    *,
    context: str,
    require_browser: bool = True,
) -> bool:
    """Ensure the SessionManager is ready for Action 6 operations."""
    try:
        if session_manager.is_sess_valid():
            if session_manager._is_session_death_cascade():
                logger.info("Action6 session recovery: clearing death cascade flag after successful validation")
                session_manager._reset_session_health_monitoring()
            return True

        logger.warning(f"Session invalid during {context} - attempting recovery")
        # Action 6 prefetch often runs while the UI is mid-navigation; avoid gating readiness
        # on UI-only cookies like 'trees'/'OptanonConsent' by using a known Action-6 action tag.
        if session_manager.ensure_session_ready(action_name=f"coord:{context}"):
            return True

        if not require_browser:
            logger.warning(f"Session recovery failed for {context}, but browser not strictly required - proceeding")
            return False

        return False

    except Exception as e:
        logger.error(f"Session check failed for {context}: {e}")
        return False


def _get_cached_profile(profile_id: str) -> dict[str, Any] | None:
    """Get profile from persistent cache if available."""
    cache_key = f"profile_details_{profile_id}"
    try:
        cache = get_unified_cache()
        cached_data = cache.get("ancestry", "profile_details", cache_key)
        if cached_data is not None and isinstance(cached_data, dict):
            return cached_data
    except Exception as e:
        logger.warning(f"Error reading profile cache for {profile_id}: {e}")
    return None


def _cache_profile(profile_id: str, profile_data: dict[str, Any]) -> None:
    """Cache profile data using UnifiedCacheManager."""
    cache_key = f"profile_details_{profile_id}"
    try:
        cache = get_unified_cache()
        # Cache profile data with a reasonable TTL (24 hours - profiles don't change often)
        cache.set(
            "ancestry",
            "profile_details",
            cache_key,
            profile_data,
            ttl=86400,  # 24 hours in seconds
        )
    except Exception as e:
        logger.warning(f"Error caching profile data for {profile_id}: {e}")


def _get_ethnicity_config() -> tuple[list[str], dict[str, str]]:
    """Load ethnicity metadata and return (region_keys, key->column mapping)."""
    metadata_raw: Any = load_ethnicity_metadata()
    if not isinstance(metadata_raw, dict):
        logger.debug("Ethnicity metadata unavailable or invalid; skipping ethnicity enrichment")
        return [], {}
    metadata: dict[str, Any] = metadata_raw

    regions = cast(list[dict[str, Any]], metadata.get("tree_owner_regions", []))
    if not regions:
        logger.debug("No ethnicity regions configured; skipping ethnicity enrichment")
        return [], {}

    region_keys: list[str] = []
    column_map: dict[str, str] = {}
    for region in regions:
        key = region.get("key")
        column_name = region.get("column_name")
        if key and column_name:
            region_keys.append(str(key))
            column_map[str(key)] = str(column_name)
    return region_keys, column_map


def fetch_ethnicity_for_batch(session_manager: SessionManager, match_uuid: str) -> dict[str, int | None] | None:
    """Fetch and parse ethnicity comparison data for a single match."""
    my_uuid = session_manager.my_uuid
    if not my_uuid or not match_uuid:
        return None

    region_keys, column_map = _get_ethnicity_config()
    if not region_keys or not column_map:
        return None

    match_guid = str(match_uuid).upper()
    comparison_data = fetch_ethnicity_comparison(session_manager, my_uuid, match_guid)
    if not comparison_data:
        return None

    percentages: dict[str, int | None] = extract_match_ethnicity_percentages(comparison_data, region_keys)
    payload: dict[str, int | None] = {}
    for region_key, percentage in percentages.items():
        column_name = column_map.get(str(region_key))
        if column_name:
            with contextlib.suppress(TypeError, ValueError):
                payload[column_name] = int(percentage) if percentage is not None else None

    return payload if payload else None


def _fetch_match_details_api(
    session_manager: SessionManager, my_uuid: str, match_uuid: str
) -> dict[str, Any] | None:
    """Fetch match details from API."""
    details_url = urljoin(
        config_schema.api.base_url,
        API_PATH_MATCH_DETAILS.format(my_uuid=my_uuid, match_uuid=match_uuid),
    )
    logger.debug(f"Fetching /details API for UUID {match_uuid}...")

    _sync_session_cookies(session_manager)

    try:
        details_response = _call_api_request(
            url=details_url,
            session_manager=session_manager,
            method="GET",
            headers=_get_api_headers(),
            use_csrf_token=False,
            api_description="Match Details API (Batch)",
        )
        return _parse_details_response(details_response, match_uuid)

    except ConnectionError as conn_err:
        logger.error(
            f"ConnectionError fetching /details for UUID {match_uuid}: {conn_err}",
            exc_info=False,
        )
        raise
    except Exception as e:
        logger.error(
            f"Error processing /details response for UUID {match_uuid}: {e}",
            exc_info=True,
        )
        if isinstance(e, requests.exceptions.RequestException):
            raise
        return None


def _check_combined_details_cache(match_uuid: str, api_start_time: float) -> dict[str, Any] | None:
    """Check cache for combined details."""
    cache_key = f"combined_details_{match_uuid}"

    # Try disk cache first
    try:
        if disk_cache:
            # cast to Any to avoid partial unknown warnings from diskcache library
            cached_data = cast(Any, disk_cache).get(cache_key)
            if cached_data and isinstance(cached_data, dict):
                log_api_performance("combined_details_cached", api_start_time, "disk_cache_hit")
                return cast(dict[str, Any], cached_data)
    except Exception as disk_exc:
        logger.debug(f"Disk cache check failed for {match_uuid}: {disk_exc}")

    # Fallback to unified cache
    try:
        cache = get_unified_cache()
        cached_data = cache.get("ancestry", "combined_details", cache_key)
        if cached_data is not None and isinstance(cached_data, dict):
            log_api_performance("combined_details_cached", api_start_time, "cache_hit")
            return cached_data
    except Exception as cache_exc:
        logger.debug(f"Memory cache check failed for {match_uuid}: {cache_exc}")

    return None


def _parse_last_login_date(last_login_str: str, tester_profile_id: str) -> datetime | None:
    """Parse last login date string."""
    try:
        if last_login_str.endswith("Z"):
            return datetime.fromisoformat(last_login_str.replace("Z", "+00:00"))
        dt_naive_or_aware = datetime.fromisoformat(last_login_str)
        return (
            dt_naive_or_aware.replace(tzinfo=UTC)
            if dt_naive_or_aware.tzinfo is None
            else dt_naive_or_aware.astimezone(UTC)
        )
    except (ValueError, TypeError) as date_parse_err:
        logger.warning(f"Could not parse LastLoginDate '{last_login_str}' for {tester_profile_id}: {date_parse_err}")
        return None


def _fetch_profile_details_api(
    session_manager: SessionManager, tester_profile_id: str, match_uuid: str
) -> dict[str, Any] | None:
    """Fetch profile details from API."""
    profile_url = urljoin(
        config_schema.api.base_url,
        f"{API_PATH_PROFILE_DETAILS}?userId={tester_profile_id.upper()}",
    )
    logger.debug(f"Fetching /profiles/details for Profile ID {tester_profile_id} (Match UUID {match_uuid})...")

    _sync_session_cookies(session_manager)

    try:
        profile_response = _call_api_request(
            url=profile_url,
            session_manager=session_manager,
            method="GET",
            headers=_get_api_headers(),
            use_csrf_token=False,
            api_description="Profile Details API (Batch)",
        )
        if profile_response and isinstance(profile_response, dict):
            logger.debug(f"Successfully fetched /profiles/details for {tester_profile_id}.")
            profile_dict = cast(dict[str, Any], profile_response)

            last_login_dt = None
            last_login_str = profile_dict.get("LastLoginDate")
            if last_login_str:
                last_login_dt = _parse_last_login_date(last_login_str, tester_profile_id)

            contactable_val = profile_dict.get("IsContactable")
            is_contactable = bool(contactable_val) if contactable_val is not None else False

            profile_data = {"last_logged_in_dt": last_login_dt, "contactable": is_contactable}
            _cache_profile(tester_profile_id, profile_data)
            return profile_data

        if isinstance(profile_response, requests.Response):
            logger.warning(
                f"Failed /profiles/details fetch for UUID {match_uuid}. Status: {profile_response.status_code}."
            )
        else:
            logger.warning(
                f"Failed /profiles/details fetch for UUID {match_uuid} (Invalid response: {type(profile_response)})."
            )
        return None

    except ConnectionError as conn_err:
        logger.error(
            f"ConnectionError fetching /profiles/details for {tester_profile_id}: {conn_err}",
            exc_info=False,
        )
        raise
    except Exception as e:
        logger.error(
            f"Error processing /profiles/details for {tester_profile_id}: {e}",
            exc_info=True,
        )
        if isinstance(e, requests.exceptions.RequestException):
            raise
        return None


def _add_profile_details_to_combined_data(
    combined_data: dict[str, Any], session_manager: SessionManager, match_uuid: str
) -> None:
    """Add profile details to combined data."""
    combined_data["last_logged_in_dt"] = None
    combined_data["contactable"] = False

    tester_profile_id = combined_data.get("tester_profile_id")
    if not tester_profile_id:
        logger.debug(f"Skipping /profiles/details fetch for {match_uuid}: Tester profile ID not found in /details.")
        return

    if not _ensure_action6_session_ready(
        session_manager,
        context=f"profile details fetch ({tester_profile_id})",
    ):
        logger.error(
            f"_fetch_combined_details: Skipping /profiles/details fetch for {tester_profile_id} due to unrecoverable session."
        )
        return

    cached_profile = _get_cached_profile(tester_profile_id)
    if cached_profile is not None:
        combined_data["last_logged_in_dt"] = cached_profile.get("last_logged_in_dt")
        combined_data["contactable"] = cached_profile.get("contactable", False)
    else:
        profile_data = _fetch_profile_details_api(session_manager, tester_profile_id, match_uuid)
        if profile_data:
            combined_data["last_logged_in_dt"] = profile_data.get("last_logged_in_dt")
            combined_data["contactable"] = profile_data.get("contactable", False)


def _cache_combined_details(combined_data: dict[str, Any], match_uuid: str) -> None:
    """Cache combined details."""
    if combined_data:
        cache_key = f"combined_details_{match_uuid}"

        # Cache to disk (persistent) - Best effort
        try:
            if disk_cache:
                # cast to Any to avoid partial unknown warnings from diskcache library
                cast(Any, disk_cache).set(cache_key, combined_data, expire=3600 * 24)  # 24 hours persistence
                logger.debug(f"Cached combined details to disk for {match_uuid}")
        except Exception as disk_exc:
            logger.debug(f"Failed to cache combined details to disk for {match_uuid}: {disk_exc}")

        # Cache to memory (fast access) - Critical for session performance
        try:
            cache = get_unified_cache()
            cache.set("ancestry", "combined_details", cache_key, combined_data, ttl=3600)
            logger.debug(f"Cached combined details for {match_uuid}")
        except Exception as cache_exc:
            logger.debug(f"Failed to cache combined details to memory for {match_uuid}: {cache_exc}")


def _validate_session_for_combined_details(session_manager: SessionManager, match_uuid: str) -> None:
    """Validate session for combined details fetch (with automatic recovery)."""
    if _ensure_action6_session_ready(session_manager, context=f"combined details fetch ({match_uuid})"):
        return

    if session_manager.should_halt_operations():
        logger.warning(f"_fetch_combined_details: Halting due to session death cascade for UUID {match_uuid}")
        raise ConnectionError(f"Session death cascade detected - halting combined details fetch (UUID: {match_uuid})")

    raise ConnectionError(f"Unable to recover WebDriver session for combined details fetch (UUID: {match_uuid})")


def fetch_combined_details(session_manager: SessionManager, match_uuid: str) -> dict[str, Any] | None:
    """
    Fetches combined match details (DNA stats, Admin/Tester IDs) and profile details
    (login date, contactable status) for a single match using two API calls.
    """
    api_start_time = time.time()

    cached_data = _check_combined_details_cache(match_uuid, api_start_time)
    if cached_data is not None:
        return cached_data

    my_uuid = session_manager.my_uuid
    if not my_uuid or not match_uuid:
        logger.warning(f"_fetch_combined_details: Missing my_uuid ({my_uuid}) or match_uuid ({match_uuid}).")
        log_api_performance("combined_details", api_start_time, "error_missing_uuid")
        return None

    _validate_session_for_combined_details(session_manager, match_uuid)

    combined_data = _fetch_match_details_api(session_manager, my_uuid, match_uuid)
    if combined_data is None:
        return None

    _add_profile_details_to_combined_data(combined_data, session_manager, match_uuid)

    _cache_combined_details(combined_data, match_uuid)
    log_api_performance("combined_details", api_start_time, "success" if combined_data else "failed", session_manager)

    return combined_data if combined_data else None


def _get_cached_badge_details(match_uuid: str) -> dict[str, Any] | None:
    """Try to get badge details from cache."""
    cache_key = f"badge_details_{match_uuid}"
    try:
        cache = get_unified_cache()
        cached_data = cache.get("ancestry", "badge_details", cache_key)
        if cached_data is not None and isinstance(cached_data, dict):
            return cached_data
    except Exception as cache_exc:
        logger.debug(f"Cache check failed for badge details {match_uuid}: {cache_exc}")
    return None


def _validate_badge_session(session_manager: SessionManager, match_uuid: str) -> None:
    """Validate session for badge details fetch."""
    if _ensure_action6_session_ready(session_manager, context=f"badge details fetch ({match_uuid})"):
        return

    if session_manager.should_halt_operations():
        logger.warning(f"_fetch_batch_badge_details: Halting due to session death cascade for UUID {match_uuid}")
        raise ConnectionError(f"Session death cascade detected - halting badge details fetch (UUID: {match_uuid})")

    raise ConnectionError(f"Unable to recover WebDriver session for badge details fetch (UUID: {match_uuid})")


def _cache_badge_details(match_uuid: str, result_data: dict[str, Any]) -> None:
    """Cache badge details for future use."""
    cache_key = f"badge_details_{match_uuid}"
    try:
        cache = get_unified_cache()
        cache.set("ancestry", "badge_details", cache_key, result_data, ttl=3600)
        logger.debug(f"Cached badge details for {match_uuid}")
    except Exception as cache_exc:
        logger.debug(f"Failed to cache badge details for {match_uuid}: {cache_exc}")


def _process_badge_response(badge_response: Any, match_uuid: str) -> dict[str, Any] | None:
    """Process badge details API response."""
    if not badge_response or not isinstance(badge_response, dict):
        if isinstance(badge_response, requests.Response):
            logger.warning(f"Failed /badgedetails fetch for UUID {match_uuid}. Status: {badge_response.status_code}.")
        else:
            logger.warning(f"Invalid badge details response for UUID {match_uuid}. Type: {type(badge_response)}")
        return None

    badge_dict = cast(dict[str, Any], badge_response)
    person_badged = cast(dict[str, Any], badge_dict.get("personBadged", {}))
    if not person_badged:
        logger.warning(f"Badge details response for UUID {match_uuid} missing 'personBadged' key.")
        return None

    their_cfpid = person_badged.get("personId")
    raw_firstname = person_badged.get("firstName")
    formatted_name_val = format_name(raw_firstname)
    their_firstname_formatted = (
        formatted_name_val.split()[0] if formatted_name_val and formatted_name_val != "Valued Relative" else "Unknown"
    )

    return {
        "their_cfpid": their_cfpid,
        "their_firstname": their_firstname_formatted,
        "their_lastname": person_badged.get("lastName", "Unknown"),
        "their_birth_year": person_badged.get("birthYear"),
    }


def fetch_batch_badge_details(session_manager: SessionManager, match_uuid: str) -> dict[str, Any] | None:
    """
    Fetches badge details for a specific match UUID. Used primarily to get the
    match's CFPID (Person ID within the user's tree) and basic tree profile info.
    """
    # Try cache first
    cached_result = _get_cached_badge_details(match_uuid)
    if cached_result:
        return cached_result

    my_uuid = session_manager.my_uuid
    if not my_uuid or not match_uuid:
        logger.warning("_fetch_batch_badge_details: Missing my_uuid or match_uuid.")
        return None

    # Validate session
    _validate_badge_session(session_manager, match_uuid)

    badge_url = urljoin(
        config_schema.api.base_url,
        API_PATH_MATCH_BADGE_DETAILS.format(my_uuid=my_uuid, match_uuid=match_uuid),
    )
    badge_referer = urljoin(config_schema.api.base_url, "/discoveryui-matches/list/")
    logger.debug(f"Fetching /badgedetails API for UUID {match_uuid}...")

    try:
        badge_response = _call_api_request(
            url=badge_url,
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False,
            api_description="Badge Details API (Batch)",
            referer_url=badge_referer,
        )

        result = _process_badge_response(badge_response, match_uuid)
        if result:
            _cache_badge_details(match_uuid, result)
        return result

    except ConnectionError as conn_err:
        logger.error(
            f"ConnectionError fetching badge details for UUID {match_uuid}: {conn_err}",
            exc_info=False,
        )
        raise
    except Exception as e:
        logger.error(f"Error processing badge details for UUID {match_uuid}: {e}", exc_info=True)
        if isinstance(e, requests.exceptions.RequestException):
            raise
        return None


def _format_kinship_path_for_action6(kinship_persons: list[dict[str, Any]]) -> str:
    """Format kinshipPersons array from relation ladder API into readable path."""
    if not kinship_persons or len(kinship_persons) < 2:
        return "(No relationship path available)"

    # Build the relationship path
    path_lines: list[str] = []
    seen_names: set[str] = set()

    # Add first person as standalone line with years
    first_person = kinship_persons[0]
    first_name = first_person.get("name", "Unknown")
    first_lifespan = first_person.get("lifeSpan", "")
    first_years = f" ({first_lifespan})" if first_lifespan else ""
    path_lines.append(f"{first_name}{first_years}")
    seen_names.add(first_name.lower())

    # Process remaining path steps
    for i in range(len(kinship_persons) - 1):
        next_person = kinship_persons[i + 1]
        relationship = next_person.get("relationship", "relative")
        next_name = next_person.get("name", "Unknown")
        next_lifespan = next_person.get("lifeSpan", "")

        # Format lifespan only if we haven't seen this name before
        next_years = ""
        if next_name.lower() not in seen_names:
            next_years = f" ({next_lifespan})" if next_lifespan else ""
            seen_names.add(next_name.lower())

        # Format the relationship step
        path_lines.append(f" → {relationship} → {next_name}{next_years}")

    return " ".join(path_lines)


def _has_kinship_entries(result: dict[str, Any] | None) -> bool:
    """Return True when the ladder response contains kinship entries."""

    return bool(result and result.get("kinship_persons") and isinstance(result.get("kinship_persons"), list))


def _fetch_ladder_via_relation_api(
    session_manager: SessionManager,
    cfpid: str,
    tree_id: str,
) -> dict[str, Any] | None:
    """Fallback to the relationladderwithlabels endpoint when the shared helper returns no data."""

    user_id = session_manager.my_profile_id or session_manager.my_uuid
    if not user_id or not tree_id:
        logger.debug(
            "Cannot invoke ladder fallback for %s - missing user_id (%s) or tree_id (%s)",
            cfpid,
            user_id,
            tree_id,
        )
        return None

    try:
        from api.api_utils import call_relation_ladder_with_labels_api
    except ImportError:
        logger.debug("Relation ladder fallback import failed; skipping fallback for %s", cfpid)
        return None

    selenium_cfg = getattr(config_schema, "selenium", None)
    timeout_value = getattr(selenium_cfg, "api_timeout", 30) if selenium_cfg else 30

    base_url = config_schema.api.base_url

    fallback_raw = call_relation_ladder_with_labels_api(
        session_manager=session_manager,
        user_id=user_id,
        tree_id=tree_id,
        person_id=cfpid,
        base_url=base_url,
        timeout=int(timeout_value),
    )

    if not fallback_raw:
        return None

    kinship = fallback_raw.get("kinshipPersons") or fallback_raw.get("kinship_persons")
    if not kinship or not isinstance(kinship, list):
        return None

    logger.info(
        "Recovered %d kinship entries for cfpid %s via relation ladder fallback",
        len(kinship),
        cfpid,
    )

    return {
        "person_id": cfpid,
        "reference_person_id": fallback_raw.get("mePid"),
        "kinship_persons": kinship,
        "raw_data": fallback_raw,
    }


def _load_relationship_ladder_data(
    session_manager: SessionManager,
    cfpid: str,
    tree_id: str,
) -> dict[str, Any] | None:
    """Load ladder information with automatic fallback for sparse responses."""

    from api.api_utils import get_relationship_path_data

    primary = get_relationship_path_data(
        session_manager=session_manager,
        person_id=cfpid,
    )

    if _has_kinship_entries(primary):
        return primary

    fallback = _fetch_ladder_via_relation_api(session_manager, cfpid, tree_id)
    if fallback:
        return fallback

    if not primary:
        logger.warning("Relationship ladder API returned no data for cfpid %s", cfpid)
        return None

    logger.warning("Relationship ladder API returned empty kinship data for cfpid %s", cfpid)
    return primary


def _normalize_relationship_phrase(raw_value: str | None) -> str:
    """Clean verbose relationship phrases returned by the API."""

    if not raw_value:
        return ""

    cleaned = raw_value.strip()
    lower_cleaned = cleaned.lower()

    prefix_variants = (
        "you are the ",
        "you are ",
        "they are your ",
        "they are the ",
        "this person is your ",
    )
    for prefix in prefix_variants:
        if lower_cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            lower_cleaned = cleaned.lower()
            break

    suffix_variants = (
        " of you",
        " of the user",
        " of the tree owner",
        " of your tree",
    )
    for suffix in suffix_variants:
        if lower_cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break

    return cleaned.strip().rstrip(".")


def _extract_relationship_from_narrative(narrative: str | None) -> str | None:
    """Parse the narrative header to derive a concise relationship label."""

    if not narrative:
        return None

    lines = [line.strip() for line in narrative.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    relationship_line = lines[1]
    if " is " not in relationship_line:
        return None

    _, remainder = relationship_line.split(" is ", 1)
    for marker in ("'s ", "' "):
        if marker in remainder:
            remainder = remainder.split(marker, 1)[1]
            break

    relationship_text = remainder.rstrip(":").strip()
    return relationship_text or None


def _resolve_tree_owner_name(session_manager: SessionManager) -> str:
    """Resolve the best available display name for the tree owner/reference person."""

    owner_name = getattr(session_manager, "tree_owner_name", None)
    if owner_name:
        return owner_name

    reference_name = getattr(config_schema, "reference_person_name", None)
    if reference_name:
        return reference_name
    return "Tree Owner"


def _normalize_kinship_entries(kinship_persons: list[dict[str, Any]]) -> list[dict[str, str | None]]:
    """Normalize raw kinship data into the structure used by formatters."""

    normalized_entries: list[dict[str, str | None]] = []
    for person in kinship_persons:
        normalized_entries.append(
            {
                "name": person.get("name", "Unknown"),
                "relationship": person.get("relationship", ""),
                "lifespan": person.get("lifeSpan") or person.get("lifespan") or "",
                "gender": person.get("gender"),
            }
        )
    return normalized_entries


def _build_unified_relationship_path(
    normalized_entries: list[dict[str, str | None]],
    target_name: str,
    owner_name: str,
) -> tuple[str | None, list[dict[str, str | None]] | None]:
    """Attempt to build a unified relationship narrative from normalized entries."""

    try:
        unified_path = convert_api_path_to_unified_format(normalized_entries, target_name)
    except Exception as conv_exc:  # pragma: no cover - diagnostic logging only
        logger.debug("Failed to normalize kinship path for unified format: %s", conv_exc, exc_info=False)
        return None, None

    if len(unified_path) < 2:
        logger.debug("Unified relationship path too short (%d entries); falling back", len(unified_path))
        return None, None

    try:
        narrative = format_relationship_path_unified(unified_path, target_name, owner_name, None)
    except Exception as fmt_exc:  # pragma: no cover - diagnostic logging only
        logger.debug("Unified relationship formatting failed: %s", fmt_exc, exc_info=False)
        return None, None

    if narrative.startswith("(No relationship path data available"):
        logger.debug("Unified formatter returned placeholder narrative; falling back")
        return None, None

    return narrative, unified_path


def _derive_actual_relationship_label(
    kinship_persons: list[dict[str, Any]],
    cfpid: str,
    narrative: str | None,
) -> str | None:
    """Determine the most useful relationship label from API data or the narrative."""

    for person in kinship_persons:
        if str(person.get("personId")) == str(cfpid):
            parsed = _normalize_relationship_phrase(person.get("relationship"))
            if parsed:
                return parsed

    fallback = _extract_relationship_from_narrative(narrative)
    if fallback:
        return fallback

    return None


def _format_relationship_path_from_kinship(
    kinship_persons: list[dict[str, Any]],
    session_manager: SessionManager,
    match_display_name: str | None,
) -> tuple[str, list[dict[str, str | None]] | None]:
    """Convert kinshipPersons data into a narrative relationship path."""

    if not kinship_persons:
        return "(No relationship path available)", None

    owner_name = _resolve_tree_owner_name(session_manager)
    normalized_entries = _normalize_kinship_entries(kinship_persons)
    target_name = match_display_name or normalized_entries[0].get("name") or "Relative"

    narrative, unified_path = _build_unified_relationship_path(normalized_entries, target_name, owner_name)
    if narrative:
        return narrative, unified_path

    fallback_narrative = _format_kinship_path_for_action6(kinship_persons)
    return fallback_narrative, None


def fetch_batch_ladder(
    session_manager: SessionManager,
    cfpid: str,
    tree_id: str,
    match_display_name: str | None = None,
) -> dict[str, Any] | None:
    """
    Fetches the relationship ladder details (relationship path, actual relationship)
    between the user and a specific person (CFPID) within the user's tree.
    """
    logger.debug(f"Fetching ladder for cfpid {cfpid} in tree {tree_id}")

    enhanced_result = _load_relationship_ladder_data(session_manager, cfpid, tree_id)
    if enhanced_result is None:
        logger.error(f"Enhanced API failed to return ladder data for {cfpid}")
        return None

    kinship_persons = enhanced_result.get("kinship_persons", [])
    if not kinship_persons:
        logger.debug(f"No kinship entries available for {cfpid} after fallback attempts")
        return None

    narrative, unified_path = _format_relationship_path_from_kinship(
        kinship_persons,
        session_manager,
        match_display_name,
    )

    relationship_label = _derive_actual_relationship_label(
        kinship_persons,
        cfpid,
        narrative,
    )

    if not relationship_label:
        logger.debug(f"Unable to derive relationship label for cfpid {cfpid}")

    ladder_payload: dict[str, Any] = {
        "actual_relationship": relationship_label,
        "relationship_path": narrative,
    }

    if unified_path:
        ladder_payload["relationship_path_unified"] = unified_path

    return ladder_payload


# =============================================================================
# TESTS
# =============================================================================


def _test_normalize_relationship_phrase() -> bool:
    """Test relationship phrase normalization."""
    # Test prefix removal
    assert _normalize_relationship_phrase("You are the Great Grandfather") == "Great Grandfather"
    assert _normalize_relationship_phrase("This person is your 2nd Cousin") == "2nd Cousin"

    # Test suffix removal
    assert _normalize_relationship_phrase("Son of the tree owner") == "Son"
    assert _normalize_relationship_phrase("Daughter of your tree") == "Daughter"

    # Test combined
    assert _normalize_relationship_phrase("You are the Uncle of the user") == "Uncle"

    # Test basic cleanup
    assert _normalize_relationship_phrase("  Father.  ") == "Father"
    assert not _normalize_relationship_phrase(None)

    return True


def _test_extract_relationship_from_narrative() -> bool:
    """Test relationship extraction from narrative text."""
    # Standard format
    narrative1 = "Relationship\nJohn Doe is your Father"
    # Note: Current implementation does not strip "your " if no possession marker is present
    assert _extract_relationship_from_narrative(narrative1) == "your Father"

    # With possession
    narrative2 = "Relationship\nJane Doe is Mary's Mother"
    assert _extract_relationship_from_narrative(narrative2) == "Mother"

    # Null/Empty cases
    assert _extract_relationship_from_narrative(None) is None
    assert _extract_relationship_from_narrative("Too short") is None

    return True


def _test_normalize_kinship_entries() -> bool:
    """Test normalization of kinship API entries."""
    raw = [
        {"name": "John", "relationship": "Father", "lifeSpan": "1900-1980"},
        {"name": "Jane", "relationship": "Grandmother", "lifespan": "1880-1950", "gender": "Female"},
    ]

    normalized = _normalize_kinship_entries(raw)

    assert len(normalized) == 2
    assert normalized[0]["name"] == "John"
    assert normalized[0]["relationship"] == "Father"
    assert normalized[0]["lifespan"] == "1900-1980"

    assert normalized[1]["name"] == "Jane"
    assert normalized[1]["lifespan"] == "1880-1950"
    assert normalized[1]["gender"] == "Female"

    return True


def module_tests() -> bool:
    """Run module tests for api_implementations."""
    from testing.test_framework import TestSuite

    suite = TestSuite("api_implementations", "actions/gather/api_implementations.py")

    suite.run_test(
        "Normalize Relationship Phrase",
        _test_normalize_relationship_phrase,
        "Validates removal of verbose prefixes/suffixes from relationship strings",
    )

    suite.run_test(
        "Extract Relationship Narrative",
        _test_extract_relationship_from_narrative,
        "Validates parsing of relationship labels from narrative text blocks",
    )

    suite.run_test(
        "Normalize Kinship Entries",
        _test_normalize_kinship_entries,
        "Validates structural normalization of API response list",
    )

    return suite.finish_suite()


if __name__ == "__main__":
    import sys

    from testing.test_framework import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
