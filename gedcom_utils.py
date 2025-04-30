# gedcom_utils.py
"""
Utility functions and class for loading, parsing, caching, and querying
GEDCOM data using ged4py. Includes relationship mapping, path calculation,
and fuzzy matching/scoring.
Consolidates helper functions and core logic from temp.py v7.36.
V16.0: Consolidated utils from temp.py, added standalone scoring function.
V16.1: Added standalone self-check functionality.
V16.2: Fixed IndentationError in fast_bidirectional_bfs.
V16.3: Fixed additional IndentationError in fast_bidirectional_bfs.
V17.0: Refactored core logic into GedcomData class, added tag constants,
       refined exception handling, improved standalone functional test.
"""

# --- Standard library imports ---
import logging
import sys
import re
import time
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set, Deque, Union, Any
from collections import deque
from datetime import datetime, timezone
import difflib
import traceback

# --- Third-party imports ---
from ged4py.parser import GedcomReader
from ged4py.model import Individual, Record, Name

# --- Local application imports ---
from utils import format_name, ordinal_case
from config import config_instance

# --- Constants ---
TAG_BIRTH = "BIRT"
TAG_DEATH = "DEAT"
TAG_HUSBAND = "HUSB"
TAG_WIFE = "WIFE"
TAG_CHILD = "CHIL"
TAG_FAMILY_CHILD = "FAMC"  # Family where individual is child
TAG_FAMILY_SPOUSE = "FAMS"  # Family where individual is spouse
TAG_DATE = "DATE"
TAG_PLACE = "PLAC"
TAG_SEX = "SEX"
TAG_NAME = "NAME"

# --- Logging Setup ---
# Use centralized logging config setup in main scripts
# Define a basic config here in case it's run standalone or utils fails
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("gedcom_utils")


# ==============================================
# Utility Functions (Independent of GedcomData)
# ==============================================


def _is_individual(obj) -> bool:
    """Checks if object is an Individual safely handling None values"""
    return obj is not None and isinstance(obj, Individual)


# End of _is_individual


def _is_record(obj) -> bool:
    """Checks if object is a Record safely handling None values"""
    return obj is not None and isinstance(obj, Record)


# End of _is_record


def _is_name(obj) -> bool:
    """Checks if object is a Name safely handling None values"""
    return obj is not None and isinstance(obj, Name)


# End of _is_name


def _normalize_id(xref_id: Optional[str]) -> Optional[str]:
    """Normalizes INDI/FAM etc IDs (e.g., '@I123@' -> 'I123')."""
    if xref_id and isinstance(xref_id, str):
        match = re.match(r"^@?([IFSTNMCXO][0-9A-Z\-]+)@?$", xref_id.strip().upper())
        if match:
            return match.group(1)
    return None


# End of _normalize_id


def extract_and_fix_id(raw_id):
    """
    Cleans and validates a raw ID string (e.g., '@I123@', 'F45').
    Returns the normalized ID (e.g., 'I123', 'F45') or None if invalid.
    """
    if not raw_id or not isinstance(raw_id, str):
        return None
    id_clean = raw_id.strip().strip("@").upper()
    m = re.match(r"^([IFSTNMCXO][0-9A-Z\-]+)$", id_clean)
    if m:
        return m.group(1)
    m2 = re.search(r"([IFSTNMCXO][0-9]+)", id_clean)
    if m2:
        logger.debug(
            f"extract_and_fix_id: Used fallback regex for '{raw_id}' -> '{m2.group(1)}'"
        )
        return m2.group(1)
    logger.warning(f"extract_and_fix_id: Could not extract valid ID from '{raw_id}'")
    return None


# End of extract_and_fix_id


def _get_full_name(indi) -> str:
    """Safely gets formatted name using Name.format() or falls back. Handles None/errors."""
    if not _is_individual(indi):
        if hasattr(indi, "value") and _is_individual(indi.value):
            indi = indi.value
        else:
            logger.warning(
                f"_get_full_name called with non-Individual type: {type(indi)}"
            )
            return "Unknown (Invalid Type)"
    try:
        name_rec = indi.name
        if _is_name(name_rec):
            formatted_name = name_rec.format()
            cleaned_name = format_name(
                formatted_name
            )  # Use imported/fallback format_name
            return cleaned_name if cleaned_name else "Unknown (Empty Name)"
        elif name_rec is None:
            return "Unknown (No Name Tag)"
        else:
            indi_id_log = _normalize_id(getattr(indi, "xref_id", None)) or "Unknown ID"
            logger.warning(
                f"Indi @{indi_id_log}@ unexpected .name type: {type(name_rec)}"
            )
            return f"Unknown (Type {type(name_rec).__name__})"
    except AttributeError:
        return "Unknown (Attr Error)"
    except Exception as e:
        indi_id_log = _normalize_id(getattr(indi, "xref_id", None)) or "Unknown ID"
        logger.error(
            f"Error formatting name for @{indi_id_log}@: {e}",
            exc_info=False,
        )
        return "Unknown (Error)"


# End of _get_full_name


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Attempts to parse various GEDCOM date string formats into datetime objects."""
    if not date_str or not isinstance(date_str, str):
        return None
    original_date_str = date_str
    date_str = date_str.strip().upper()
    # Basic cleaning - more complex parsing could be added here
    clean_date_str = re.sub(r"^(ABT|EST|BEF|AFT|BET|FROM|TO)\s+", "", date_str).strip()
    clean_date_str = re.sub(
        r"(\s+(AND|&)\s+\d{4}.*|\s+TO\s+\d{4}.*)", "", clean_date_str
    ).strip()
    clean_date_str = re.sub(r"/\d*$", "", clean_date_str).strip()
    clean_date_str = re.sub(r"\s*\(.*\)\s*", "", clean_date_str).strip()
    formats = ["%d %b %Y", "%d %B %Y", "%b %Y", "%B %Y", "%Y"]
    for fmt in formats:
        try:
            if fmt == "%Y" and not clean_date_str.isdigit():
                continue
            dt = datetime.strptime(clean_date_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            continue
        except Exception as e:
            logger.debug(
                f"Date parsing err for '{original_date_str}' (clean:'{clean_date_str}', fmt:'{fmt}'): {e}"
            )
            continue
    return None


# End of _parse_date


def _clean_display_date(raw_date_str: Optional[str]) -> str:
    """Removes surrounding brackets if date exists, handles empty brackets."""
    if not raw_date_str or raw_date_str == "N/A":
        return "N/A"
    cleaned = raw_date_str.strip()
    cleaned = re.sub(r"^\((.+)\)$", r"\1", cleaned).strip()
    cleaned = (
        cleaned.replace("ABT ", "~")
        .replace("EST ", "~")
        .replace("BEF ", "<")
        .replace("AFT ", ">")
    )
    cleaned = (
        cleaned.replace("BET ", "")
        .replace(" FROM ", "")
        .replace(" TO ", "-")
        .replace(" AND ", "-")
    )
    return cleaned if cleaned else "N/A"


# End of _clean_display_date


def _get_event_info(individual, event_tag: str) -> Tuple[Optional[datetime], str, str]:
    """Gets date/place for an event using tag.value. Handles non-string dates."""
    date_obj: Optional[datetime] = None
    date_str: str = "N/A"
    place_str: str = "N/A"
    indi_id_log = "Invalid/Unknown"

    if _is_individual(individual):
        indi_id_log = (
            _normalize_id(getattr(individual, "xref_id", None)) or "Unknown ID"
        )
    elif hasattr(individual, "value") and _is_individual(individual.value):
        individual = individual.value
        indi_id_log = (
            _normalize_id(getattr(individual, "xref_id", None)) or "Unknown ID"
        )
    else:
        logger.warning(f"_get_event_info invalid input type: {type(individual)}")
        return date_obj, date_str, place_str

    try:
        event_record = individual.sub_tag(event_tag.upper())
        if event_record:
            date_tag = event_record.sub_tag(TAG_DATE)
            if date_tag and hasattr(date_tag, "value"):
                raw_date_val = date_tag.value
                if isinstance(raw_date_val, str):
                    processed_date_str = raw_date_val.strip()
                    date_str = processed_date_str if processed_date_str else "N/A"
                    date_obj = _parse_date(date_str)
                elif raw_date_val is not None:
                    date_str = str(raw_date_val)
                    date_obj = _parse_date(date_str)

            place_tag = event_record.sub_tag(TAG_PLACE)
            if place_tag and hasattr(place_tag, "value"):
                raw_place_val = place_tag.value
                if isinstance(raw_place_val, str):
                    processed_place_str = raw_place_val.strip()
                    place_str = processed_place_str if processed_place_str else "N/A"
                elif raw_place_val is not None:
                    place_str = str(raw_place_val)
    except AttributeError as ae:
        logger.debug(f"Attr error getting event '{event_tag}' for {indi_id_log}: {ae}")
    except Exception as e:
        logger.error(
            f"Error accessing event {event_tag} for @{indi_id_log}@: {e}", exc_info=True
        )
    return date_obj, date_str, place_str


# End of _get_event_info


def format_life_dates(indi) -> str:
    """Returns a formatted string with birth and death dates."""
    if not _is_individual(indi):
        logger.warning(
            f"format_life_dates called with non-Individual type: {type(indi)}"
        )
        return ""

    b_date_obj, b_date_str, b_place = _get_event_info(indi, TAG_BIRTH)
    d_date_obj, d_date_str, d_place = _get_event_info(indi, TAG_DEATH)

    b_date_str_cleaned = _clean_display_date(b_date_str)
    d_date_str_cleaned = _clean_display_date(d_date_str)

    birth_info = f"b. {b_date_str_cleaned}" if b_date_str_cleaned != "N/A" else ""
    death_info = f"d. {d_date_str_cleaned}" if d_date_str_cleaned != "N/A" else ""
    life_parts = [info for info in [birth_info, death_info] if info]
    return f" ({', '.join(life_parts)})" if life_parts else ""


# End of format_life_dates


def format_full_life_details(indi) -> Tuple[str, str]:
    """Returns formatted birth and death details (date and place) for display."""
    if not _is_individual(indi):
        logger.warning(
            f"format_full_life_details called with non-Individual type: {type(indi)}"
        )
        return "(Error: Invalid data)", ""

    b_date_obj, b_date_str, b_place = _get_event_info(indi, TAG_BIRTH)
    b_date_str_cleaned = _clean_display_date(b_date_str)

    birth_info = (
        f"Born: {b_date_str_cleaned if b_date_str_cleaned != 'N/A' else '(Date unknown)'} "
        f"in {b_place if b_place != 'N/A' else '(Place unknown)'}"
    )

    d_date_obj, d_date_str, d_place = _get_event_info(indi, TAG_DEATH)
    d_date_str_cleaned = _clean_display_date(d_date_str)

    death_info = ""
    if d_date_str_cleaned != "N/A" or d_place != "N/A":
        death_info = (
            f"   Died: {d_date_str_cleaned if d_date_str_cleaned != 'N/A' else '(Date unknown)'} "
            f"in {d_place if d_place != 'N/A' else '(Place unknown)'}"
        )
    return birth_info, death_info


# End of format_full_life_details


def format_relative_info(relative) -> str:
    """Formats information about a relative (name and life dates) for display."""
    indi_obj = None
    if _is_individual(relative):
        indi_obj = relative
    elif hasattr(relative, "value") and _is_individual(relative.value):
        indi_obj = relative.value
    else:
        raw_id = getattr(relative, "xref_id", "N/A")
        norm_id = _normalize_id(raw_id)
        return f"  - (Invalid Relative Data: ID={norm_id or 'N/A'}, Type={type(relative).__name__})"

    rel_name = _get_full_name(indi_obj)
    life_info = format_life_dates(indi_obj)
    return f"  - {rel_name}{life_info}"


# End of format_relative_info


def _reconstruct_path(
    start_id: str,
    end_id: str,
    meeting_id: str,
    visited_fwd: Dict[str, Optional[str]],
    visited_bwd: Dict[str, Optional[str]],
) -> List[str]:
    """
    Reconstructs the path from start to end via the meeting point using predecessor maps.
    Returns a list of normalized IDs.
    """
    path_fwd: List[str] = []
    curr = meeting_id
    while curr is not None:
        path_fwd.append(curr)
        curr = visited_fwd.get(curr)
    path_fwd.reverse()

    path_bwd: List[str] = []
    curr = visited_bwd.get(meeting_id)
    while curr is not None:
        path_bwd.append(curr)
        curr = visited_bwd.get(curr)

    # Combine, removing duplicate meeting ID if necessary
    path = path_fwd + path_bwd

    # Sanity checks
    if not path:
        logger.error("_reconstruct_path: Failed to reconstruct any path.")
        return []
    if path[0] != start_id:
        logger.warning(
            f"_reconstruct_path: Path doesn't start with start_id ({path[0]} != {start_id}). Prepending."
        )
        path.insert(0, start_id)
    if path[-1] != end_id:
        logger.warning(
            f"_reconstruct_path: Path doesn't end with end_id ({path[-1]} != {end_id}). Appending."
        )
        path.append(end_id)

    logger.debug(f"_reconstruct_path: Final reconstructed path IDs: {path}")
    return path


# End of _reconstruct_path


def fast_bidirectional_bfs(
    start_id: str,
    end_id: str,
    id_to_parents: Dict[str, Set[str]],
    id_to_children: Dict[str, Set[str]],
    max_depth=25,
    node_limit=150000,
    timeout_sec=45,
    log_progress=False,
) -> List[str]:
    """
    Performs bidirectional BFS using pre-built maps and predecessors.
    Returns path as list of normalized IDs.
    """
    start_time = time.time()
    if start_id == end_id:
        return [start_id]
    if id_to_parents is None or id_to_children is None:  # Check if maps are None
        logger.error("[FastBiBFS] Relationship maps are None.")
        return []

    queue_fwd: Deque[Tuple[str, int]] = deque([(start_id, 0)])
    visited_fwd: Dict[str, Optional[str]] = {start_id: None}
    queue_bwd: Deque[Tuple[str, int]] = deque([(end_id, 0)])
    visited_bwd: Dict[str, Optional[str]] = {end_id: None}
    processed = 0
    meeting_id: Optional[str] = None

    logger.debug(f"[FastBiBFS] Starting BFS: {start_id} <-> {end_id}")

    while queue_fwd and queue_bwd and meeting_id is None:
        if time.time() - start_time > timeout_sec:
            logger.warning(f"  [FastBiBFS] Timeout after {timeout_sec} seconds.")
            return []
        if processed > node_limit:
            logger.warning(
                f"  [FastBiBFS] Node limit {node_limit} reached. Processed: ~{processed}."
            )
            return []
        if log_progress and processed > 0 and processed % 10000 == 0:
            logger.info(
                f"[FastBiBFS] Progress: ~{processed} nodes, QF:{len(queue_fwd)}, QB:{len(queue_bwd)}"
            )

        # Expand Forward
        if queue_fwd:
            current_id_fwd, depth_fwd = queue_fwd.popleft()
            if current_id_fwd is None:
                continue
            processed += 1
            if depth_fwd >= max_depth:
                continue

            if current_id_fwd in visited_bwd:
                meeting_id = current_id_fwd
                logger.debug(
                    f"  [FastBiBFS] Path found (FWD meets BWD) at {meeting_id} (Depth FWD: {depth_fwd})."
                )
                break

            neighbors_fwd = id_to_parents.get(
                current_id_fwd, set()
            ) | id_to_children.get(current_id_fwd, set())
            for neighbor_id in neighbors_fwd:
                if neighbor_id is None:
                    continue
                if neighbor_id not in visited_fwd:
                    visited_fwd[neighbor_id] = current_id_fwd
                    queue_fwd.append((neighbor_id, depth_fwd + 1))
                    if neighbor_id in visited_bwd:
                        meeting_id = neighbor_id
                        logger.debug(
                            f"  [FastBiBFS] Path found (FWD adds node visited by BWD) at {meeting_id} (Depth FWD: {depth_fwd+1})."
                        )
                        break
            if meeting_id:
                break

        # Expand Backward
        if queue_bwd and meeting_id is None:
            current_id_bwd, depth_bwd = queue_bwd.popleft()
            if current_id_bwd is None:
                continue
            processed += 1
            if depth_bwd >= max_depth:
                continue

            if current_id_bwd in visited_fwd:
                meeting_id = current_id_bwd
                logger.debug(
                    f"  [FastBiBFS] Path found (BWD meets FWD) at {meeting_id} (Depth BWD: {depth_bwd})."
                )
                break

            neighbors_bwd = id_to_parents.get(
                current_id_bwd, set()
            ) | id_to_children.get(current_id_bwd, set())
            for neighbor_id in neighbors_bwd:
                if neighbor_id is None:
                    continue
                if neighbor_id not in visited_bwd:
                    visited_bwd[neighbor_id] = current_id_bwd
                    queue_bwd.append((neighbor_id, depth_bwd + 1))
                    if neighbor_id in visited_fwd:
                        meeting_id = neighbor_id
                        logger.debug(
                            f"  [FastBiBFS] Path found (BWD adds node visited by FWD) at {meeting_id} (Depth BWD: {depth_bwd+1})."
                        )
                        break
            # No need for break here, main loop check handles it

    if meeting_id:
        logger.debug(
            f"[FastBiBFS] Intersection found at {meeting_id}. Reconstructing path..."
        )
        path_ids = _reconstruct_path(
            start_id, end_id, meeting_id, visited_fwd, visited_bwd
        )
        logger.debug(
            f"[FastBiBFS] Path reconstruction complete. Length: {len(path_ids)}"
        )
        return path_ids
    else:
        reason = "Queues Emptied"
        if time.time() - start_time > timeout_sec:
            reason = "Timeout"
        elif processed > node_limit:
            reason = "Node Limit Reached"
        # Add more specific checks if needed (using GedcomData instance if passed or global index)
        logger.warning(
            f"[FastBiBFS] No path found between {start_id} and {end_id}. Reason: {reason}. Processed ~{processed} nodes."
        )
        return []


# End of fast_bidirectional_bfs


def explain_relationship_path(
    path_ids: List[str],
    reader,  # Needed to find individuals for names/sex
    id_to_parents: Dict[str, Set[str]],
    id_to_children: Dict[str, Set[str]],
    indi_index: Dict[str, Any],  # Pass index for lookup
) -> str:
    """
    Return a human-readable explanation of the relationship path with relationship labels.
    Uses helper functions and passed-in maps/index.
    """
    if not path_ids or len(path_ids) < 2:
        return "(No relationship path explanation available)"
    if id_to_parents is None or id_to_children is None or indi_index is None:
        return "(Error: Data maps or index unavailable for explanation)"

    steps = []
    ordinal_formatter = ordinal_case  # Assumes ordinal_case is available

    for i in range(len(path_ids) - 1):
        id_a, id_b = path_ids[i], path_ids[i + 1]
        # Use index for potentially faster lookup than reader.individual_record()
        indi_a = indi_index.get(id_a)
        indi_b = indi_index.get(id_b)

        name_a = _get_full_name(indi_a) if indi_a else f"Unknown ({id_a})"
        name_b = _get_full_name(indi_b) if indi_b else f"Unknown ({id_b})"

        label = "related"
        if id_b in id_to_parents.get(id_a, set()):
            sex_a = (
                getattr(indi_a, TAG_SEX.lower(), None) if indi_a else None
            )  # Use constant
            sex_a_char = (
                str(sex_a).upper()[0]
                if sex_a and isinstance(sex_a, str) and str(sex_a).upper() in ("M", "F")
                else None
            )
            label = (
                "daughter"
                if sex_a_char == "F"
                else "son" if sex_a_char == "M" else "child"
            )
        elif id_b in id_to_children.get(id_a, set()):
            sex_b = (
                getattr(indi_b, TAG_SEX.lower(), None) if indi_b else None
            )  # Use constant
            sex_b_char = (
                str(sex_b).upper()[0]
                if sex_b and isinstance(sex_b, str) and str(sex_b).upper() in ("M", "F")
                else None
            )
            label = (
                "father"
                if sex_b_char == "M"
                else "mother" if sex_b_char == "F" else "parent"
            )
        else:
            parents_a = id_to_parents.get(id_a, set())
            parents_b = id_to_parents.get(id_b, set())
            if parents_a and parents_b and (parents_a & parents_b) and id_a != id_b:
                sex_a = (
                    getattr(indi_a, TAG_SEX.lower(), None) if indi_a else None
                )  # Use constant
                sex_a_char = (
                    str(sex_a).upper()[0]
                    if sex_a
                    and isinstance(sex_a, str)
                    and str(sex_a).upper() in ("M", "F")
                    else None
                )
                label = (
                    "sister"
                    if sex_a_char == "F"
                    else "brother" if sex_a_char == "M" else "sibling"
                )
            else:
                logger.warning(
                    f"Could not determine direct relation between {id_a} ({name_a}) and {id_b} ({name_b}) for path explanation."
                )
                label = "connected to"

        steps.append(f"{name_a} is the {ordinal_formatter(label)} of {name_b}")

    start_person_indi = indi_index.get(path_ids[0])
    start_person_name = (
        _get_full_name(start_person_indi)
        if start_person_indi
        else f"Unknown ({path_ids[0]})"
    )
    explanation_str = "\n -> ".join(steps)
    return f"{start_person_name}\n -> {explanation_str}"


# End of explain_relationship_path


def calculate_match_score(
    search_criteria: Dict,
    candidate_data: Dict,
    scoring_weights: Dict,
    name_flexibility: Dict,
    date_flexibility: Dict,
) -> Tuple[float, List[str]]:
    """
    Calculates a match score between search criteria and candidate data.
    Uses weights and flexibility settings from config.
    Returns score and list of reasons.
    """
    score = 0.0
    match_reasons: List[str] = []

    weights = scoring_weights if scoring_weights is not None else {}
    date_flex = (
        date_flexibility if date_flexibility is not None else {"year_match_range": 1}
    )
    name_flex = (
        name_flexibility if name_flexibility is not None else {"fuzzy_threshold": 0.8}
    )

    year_score_range = date_flex.get("year_match_range", 1)
    fuzzy_threshold = name_flex.get("fuzzy_threshold", 0.8)

    # --- Safely extract and normalize search criteria ---
    target_first_name_raw = search_criteria.get("first_name")
    target_surname_raw = search_criteria.get("surname")
    target_pob_raw = search_criteria.get("birth_place")
    target_pod_raw = search_criteria.get("death_place")

    target_first_name_lower = (
        target_first_name_raw.lower() if isinstance(target_first_name_raw, str) else ""
    )
    target_surname_lower = (
        target_surname_raw.lower() if isinstance(target_surname_raw, str) else ""
    )
    target_pob_lower = target_pob_raw.lower() if isinstance(target_pob_raw, str) else ""
    target_pod_lower = target_pod_raw.lower() if isinstance(target_pod_raw, str) else ""

    target_birth_year = search_criteria.get("birth_year")
    target_birth_date_obj = search_criteria.get("birth_date_obj")
    target_death_year = search_criteria.get("death_year")
    target_death_date_obj = search_criteria.get("death_date_obj")
    target_gender_clean = search_criteria.get("gender")  # m or f or None
    # --- End Search Criteria Extraction ---

    # --- Safely extract and normalize candidate data ---
    c_first_name_raw = candidate_data.get("first_name")
    c_surname_raw = candidate_data.get("surname")
    c_birth_place_raw = candidate_data.get("birth_place")
    c_death_place_raw = candidate_data.get("death_place")

    c_first_name_lower = (
        c_first_name_raw.lower() if isinstance(c_first_name_raw, str) else ""
    )
    c_surname_lower = c_surname_raw.lower() if isinstance(c_surname_raw, str) else ""
    c_birth_place_lower = (
        c_birth_place_raw.lower() if isinstance(c_birth_place_raw, str) else ""
    )
    c_death_place_lower = (
        c_death_place_raw.lower() if isinstance(c_death_place_raw, str) else ""
    )

    c_birth_year = candidate_data.get("birth_year")
    c_birth_date_obj = candidate_data.get("birth_date_obj")
    c_death_year = candidate_data.get("death_year")
    c_death_date_obj = candidate_data.get("death_date_obj")
    c_gender_clean = candidate_data.get("gender")  # m or f or None
    # --- End Candidate Data Extraction ---

    # --- Scoring Logic ---
    first_name_match = bool(
        target_first_name_lower
        and c_first_name_lower
        and target_first_name_lower == c_first_name_lower
    )
    surname_match = bool(
        target_surname_lower
        and c_surname_lower
        and target_surname_lower == c_surname_lower
    )

    if first_name_match and surname_match:
        # --- Exact Full Name Path ---
        score += weights.get("exact_first_name", 20)
        match_reasons.append("Exact First")
        score += weights.get("exact_surname", 20)
        match_reasons.append("Exact Surname")
        score += weights.get("boost_exact_full_name", 20)
        match_reasons.append("Boost Exact Name")

        birth_year_bonus_match = False
        if (
            target_birth_date_obj
            and c_birth_date_obj
            and target_birth_date_obj.date() == c_birth_date_obj.date()
        ):
            score += weights.get("exact_birth_date", 20)
            match_reasons.append("Exact Birth Date")
        elif (
            target_birth_year
            and c_birth_year
            and abs(c_birth_year - target_birth_year) <= year_score_range
        ):
            score += weights.get("year_birth", 15)
            birth_year_bonus_match = True
            match_reasons.append(f"Birth Year ~{target_birth_year} ({c_birth_year})")

        if (
            target_death_date_obj
            and c_death_date_obj
            and target_death_date_obj.date() == c_death_date_obj.date()
        ):
            score += weights.get("exact_death_date", 20)
            match_reasons.append("Exact Death Date")
        elif (
            target_death_year
            and c_death_year
            and abs(c_death_year - target_death_year) <= year_score_range
        ):
            score += weights.get("year_death", 15)
            match_reasons.append(f"Death Year ~{target_death_year} ({c_death_year})")
        elif (
            target_death_date_obj is None
            and c_death_date_obj is None
            and target_death_year is None
            and c_death_year is None
        ):
            score += weights.get("death_dates_both_absent", 5)
            match_reasons.append("Death Dates Both Absent")

        # Check places only if they are non-empty strings after lowercasing
        if (
            target_pob_lower
            and c_birth_place_lower
            and target_pob_lower in c_birth_place_lower
        ):
            score += weights.get("contains_pob", 15)
            match_reasons.append(f"POB contains '{target_pob_lower}'")
        if (
            target_pod_lower
            and c_death_place_lower
            and target_pod_lower in c_death_place_lower
        ):
            score += weights.get("contains_pod", 15)
            match_reasons.append(f"POD contains '{target_pod_lower}'")

        gender_bonus_match = False
        if (
            target_gender_clean
            and c_gender_clean
            and target_gender_clean == c_gender_clean
        ):
            score += weights.get("gender_match", 20)
            gender_bonus_match = True
            match_reasons.append(f"Gender ({target_gender_clean.upper()})")
        elif (
            target_gender_clean
            and c_gender_clean
            and target_gender_clean != c_gender_clean
        ):
            score += weights.get("gender_mismatch_penalty", -20)
            match_reasons.append(
                f"Gender Mismatch ({c_gender_clean.upper()} vs {target_gender_clean.upper()})"
            )

        if birth_year_bonus_match:
            score += weights.get("boost_exact_name_year", 2)
            match_reasons.append("Boost Exact Name + Year")
    else:
        # --- Fuzzy/Other Match Path ---
        name_score = 0.0
        fuzzy_first = (
            difflib.SequenceMatcher(
                None, target_first_name_lower, c_first_name_lower
            ).ratio()
            if target_first_name_lower and c_first_name_lower
            else 0.0
        )
        fuzzy_surname = (
            difflib.SequenceMatcher(None, target_surname_lower, c_surname_lower).ratio()
            if target_surname_lower and c_surname_lower
            else 0.0
        )

        if first_name_match:
            name_score += weights.get("exact_first_name", 20)
            match_reasons.append("Exact First")
        elif fuzzy_first >= fuzzy_threshold:
            name_score += weights.get("fuzzy_first_name", 15) * fuzzy_first
            match_reasons.append(f"Fuzzy First ({fuzzy_first:.2f})")

        if surname_match:
            name_score += weights.get("exact_surname", 20)
            match_reasons.append("Exact Surname")
        elif fuzzy_surname >= fuzzy_threshold:
            name_score += weights.get("fuzzy_surname", 15) * fuzzy_surname
            match_reasons.append(f"Fuzzy Surname ({fuzzy_surname:.2f})")

        if name_score > 0:
            score += name_score
            if (
                target_birth_date_obj
                and c_birth_date_obj
                and target_birth_date_obj.date() == c_birth_date_obj.date()
            ):
                score += weights.get("exact_birth_date", 20)
                match_reasons.append("Exact Birth Date")
            elif (
                target_birth_year
                and c_birth_year
                and abs(c_birth_year - target_birth_year) <= year_score_range
            ):
                score += weights.get("year_birth_fuzzy", 5)
                match_reasons.append(
                    f"Birth Year ~{target_birth_year} ({c_birth_year})"
                )

            if (
                target_death_date_obj
                and c_death_date_obj
                and target_death_date_obj.date() == c_death_date_obj.date()
            ):
                score += weights.get("exact_death_date", 20)
                match_reasons.append("Exact Death Date")
            elif (
                target_death_year
                and c_death_year
                and abs(c_death_year - target_death_year) <= year_score_range
            ):
                score += weights.get("year_death_fuzzy", 5)
                match_reasons.append(
                    f"Death Year ~{target_death_year} ({c_death_year})"
                )
            elif (
                target_death_date_obj is None
                and c_death_date_obj is None
                and target_death_year is None
                and c_death_year is None
            ):
                score += weights.get("death_dates_both_absent", 5)
                match_reasons.append("Death Dates Both Absent")

            # Check places only if they are non-empty strings after lowercasing
            if (
                target_pob_lower
                and c_birth_place_lower
                and target_pob_lower in c_birth_place_lower
            ):
                score += weights.get("contains_pob_fuzzy", 1)
                match_reasons.append(f"POB contains '{target_pob_lower}'")
            if (
                target_pod_lower
                and c_death_place_lower
                and target_pod_lower in c_death_place_lower
            ):
                score += weights.get("contains_pod_fuzzy", 1)
                match_reasons.append(f"POD contains '{target_pod_lower}'")

            if (
                target_gender_clean
                and c_gender_clean
                and target_gender_clean == c_gender_clean
            ):
                score += weights.get("gender_match_fuzzy", 3)
                match_reasons.append(f"Gender ({target_gender_clean.upper()})")
            elif (
                target_gender_clean
                and c_gender_clean
                and target_gender_clean != c_gender_clean
            ):
                score += weights.get("gender_mismatch_penalty_fuzzy", -3)
                match_reasons.append(
                    f"Gender Mismatch ({c_gender_clean.upper()} vs {target_gender_clean.upper()})"
                )

    return round(score), sorted(list(set(match_reasons)))


# End of calculate_match_score


# ==============================================
# GedcomData Class
# ==============================================


class GedcomData:
    """
    Handles loading, caching, and querying of GEDCOM data.
    Encapsulates the GedcomReader, indexes, and relationship maps.
    """

    def __init__(self, gedcom_path: Union[str, Path]):
        """Initializes the GedcomData object, loads the file, and builds caches."""
        self.path = Path(gedcom_path)
        self.reader: Optional[GedcomReader] = None
        self.indi_index: Dict[str, Any] = {}
        self.id_to_parents: Dict[str, Set[str]] = {}
        self.id_to_children: Dict[str, Set[str]] = {}
        self.indi_index_build_time: float = 0
        self.family_maps_build_time: float = 0


        if not self.path.exists() or not self.path.is_file():
            logger.critical(f"GEDCOM file not found or is not a file: {self.path}")
            raise FileNotFoundError(f"GEDCOM file not found: {self.path}")

        try:
            logger.info(f"Loading GEDCOM file: {self.path}")
            load_start = time.time()
            self.reader = GedcomReader(str(self.path))  # ged4py needs string path
            load_time = time.time() - load_start
            logger.info(f"GEDCOM file loaded in {load_time:.2f}s.")
        except Exception as e:
            logger.critical(
                f"Failed to load or parse GEDCOM file {self.path}: {e}", exc_info=True
            )
            raise  # Re-raise after logging

        # Build caches upon initialization
        self.build_caches()

    # End of __init__

    def build_caches(self):
        """Builds or rebuilds the individual index and family maps."""
        self._build_indi_index()
        self._build_family_maps()

    # End of build_caches

    def _build_indi_index(self):
        """Builds a dictionary mapping normalized ID to Individual object."""
        if not self.reader:
            return  # Should not happen if init succeeded
        start_time = time.time()
        logger.info("[Cache] Building INDI index...")
        self.indi_index = {}  # Reset index
        count = 0
        try:
            for indi in self.reader.records0("INDI"):
                if _is_individual(indi) and hasattr(indi, "xref_id") and indi.xref_id:
                    norm_id = _normalize_id(indi.xref_id)
                    if norm_id:
                        self.indi_index[norm_id] = indi
                        count += 1
                    else:
                        logger.debug(
                            f"Skipping INDI with unnormalizable xref_id: {indi.xref_id}"
                        )
                elif hasattr(indi, "xref_id"):
                    logger.debug(
                        f"Skipping non-Individual record: Type={type(indi).__name__}, Xref={indi.xref_id}"
                    )
                else:
                    logger.debug(
                        f"Skipping record with no xref_id: Type={type(indi).__name__}"
                    )
        except Exception as e:
            logger.error(f"Error during INDI index build: {e}", exc_info=True)
            self.indi_index = {}
        elapsed = time.time() - start_time
        self.indi_index_build_time = elapsed
        logger.info(
            f"[Cache] INDI index built with {count} individuals in {elapsed:.2f}s."
        )

    # End of _build_indi_index

    def _build_family_maps(self):
        """Builds id_to_parents and id_to_children maps."""
        if not self.reader:
            return  # Should not happen if init succeeded
        start_time = time.time()
        logger.info("[Cache] Building family maps (direct tag access)...")
        self.id_to_parents = {}  # Reset map
        self.id_to_children = {}  # Reset map
        fam_count = 0
        processed_links = 0
        try:
            for fam in self.reader.records0("FAM"):
                fam_count += 1
                if not _is_record(fam):
                    continue
                parents = set()
                husband_tag = fam.sub_tag(TAG_HUSBAND)
                wife_tag = fam.sub_tag(TAG_WIFE)
                if (
                    husband_tag
                    and hasattr(husband_tag, "xref_id")
                    and husband_tag.xref_id
                ):
                    parent_id_h = _normalize_id(husband_tag.xref_id)
                    if parent_id_h:
                        parents.add(parent_id_h)
                if wife_tag and hasattr(wife_tag, "xref_id") and wife_tag.xref_id:
                    parent_id_w = _normalize_id(wife_tag.xref_id)
                    if parent_id_w:
                        parents.add(parent_id_w)

                children_tags = fam.sub_tags(TAG_CHILD)
                for child_tag in children_tags:
                    # Check reference is valid before normalizing
                    if (
                        child_tag
                        and hasattr(child_tag, "xref_id")
                        and child_tag.xref_id
                    ):
                        child_id = _normalize_id(child_tag.xref_id)
                        if child_id:
                            processed_links += 1
                            self.id_to_parents.setdefault(child_id, set()).update(
                                parents
                            )
                            for parent_id in parents:
                                if parent_id:
                                    self.id_to_children.setdefault(
                                        parent_id, set()
                                    ).add(child_id)
                    elif child_tag is not None:
                        logger.debug(
                            f"Skipping CHIL record in FAM {getattr(fam, 'xref_id', 'N/A')} with invalid/missing xref_id: Type={type(child_tag).__name__}"
                        )

        except AttributeError as ae:
            logger.error(
                f"AttributeError during family map build: {ae}.", exc_info=True
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during family map build: {e}", exc_info=True
            )

        elapsed = time.time() - start_time
        self.family_maps_build_time = elapsed
        logger.info(
            f"[Cache] Family maps built: {fam_count} FAMs processed. Found {processed_links} parent/child links. "
            f"Map sizes: {len(self.id_to_parents)} child->parents, {len(self.id_to_children)} parent->children in {elapsed:.2f}s."
        )

    # End of _build_family_maps

    def find_individual_by_id(self, norm_id: Optional[str]):
        """Finds an individual by normalized ID using the internal index."""
        if not norm_id or not isinstance(norm_id, str):
            logger.warning("find_individual_by_id called with invalid norm_id")
            return None
        if not self.indi_index:
            logger.error("INDI_INDEX not built. Cannot lookup individual by ID.")
            return None
        found_indi = self.indi_index.get(norm_id)
        if not found_indi:
            logger.debug(
                f"Individual with normalized ID {norm_id} not found in INDI_INDEX."
            )
        return found_indi

    # End of find_individual_by_id

    def _find_family_records_where_individual_is_child(self, target_id):
        """Helper to find family records where an individual is a child."""
        parent_families = []
        if not self.reader or not target_id:
            return parent_families
        try:
            for family_record in self.reader.records0("FAM"):
                if not _is_record(family_record):
                    continue
                children_in_fam = family_record.sub_tags(TAG_CHILD)
                if children_in_fam:
                    for child_tag in children_in_fam:
                        if (
                            child_tag
                            and hasattr(child_tag, "xref_id")
                            and _normalize_id(child_tag.xref_id) == target_id
                        ):
                            parent_families.append(family_record)
                            break
        except Exception as e:
            logger.error(f"Error finding FAMC for ID {target_id}: {e}", exc_info=True)
        return parent_families

    # End of _find_family_records_where_individual_is_child

    def _find_family_records_where_individual_is_parent(self, target_id):
        """Helper to find family records where an individual is a parent."""
        parent_families = []
        if not self.reader or not target_id:
            return parent_families
        try:
            for family_record in self.reader.records0("FAM"):
                if not _is_record(family_record):
                    continue
                husband_tag = family_record.sub_tag(TAG_HUSBAND)
                wife_tag = family_record.sub_tag(TAG_WIFE)
                is_target_husband = False
                if (
                    husband_tag
                    and hasattr(husband_tag, "xref_id")
                    and _normalize_id(husband_tag.xref_id) == target_id
                ):
                    is_target_husband = True
                is_target_wife = False
                if (
                    wife_tag
                    and hasattr(wife_tag, "xref_id")
                    and _normalize_id(wife_tag.xref_id) == target_id
                ):
                    is_target_wife = True
                if is_target_husband or is_target_wife:
                    parent_families.append(
                        (family_record, is_target_husband, is_target_wife)
                    )
        except Exception as e:
            logger.error(f"Error finding FAMS for ID {target_id}: {e}", exc_info=True)
        return parent_families

    # End of _find_family_records_where_individual_is_parent

    def get_related_individuals(self, individual, relationship_type: str) -> List[Any]:
        """
        Gets parents, spouses, children, or siblings using family record lookups.
        Returns a list of Individual objects.
        """
        related_individuals: List[Any] = []
        unique_related_ids: Set[str] = set()

        if not self.reader:
            return related_individuals
        if (
            not _is_individual(individual)
            or not hasattr(individual, "xref_id")
            or not individual.xref_id
        ):
            logger.warning(
                f"get_related_individuals: Invalid input individual object: {type(individual)}"
            )
            return related_individuals
        target_id = _normalize_id(individual.xref_id)
        if not target_id:
            return related_individuals

        try:
            if relationship_type == "parents":
                parent_families = self._find_family_records_where_individual_is_child(
                    target_id
                )
                potential_parents = []
                for family_record in parent_families:
                    husband = family_record.sub_tag(TAG_HUSBAND)
                    wife = family_record.sub_tag(TAG_WIFE)
                    # Prefer linked value if available
                    if (
                        husband
                        and hasattr(husband, "value")
                        and _is_individual(husband.value)
                    ):
                        potential_parents.append(husband.value)
                    elif _is_individual(husband):
                        potential_parents.append(husband)
                    if wife and hasattr(wife, "value") and _is_individual(wife.value):
                        potential_parents.append(wife.value)
                    elif _is_individual(wife):
                        potential_parents.append(wife)

                for parent in potential_parents:
                    if parent and hasattr(parent, "xref_id") and parent.xref_id:
                        parent_id = _normalize_id(parent.xref_id)
                        if parent_id and parent_id not in unique_related_ids:
                            related_individuals.append(parent)
                            unique_related_ids.add(parent_id)

            elif relationship_type == "siblings":
                parent_families = self._find_family_records_where_individual_is_child(
                    target_id
                )
                potential_siblings = []
                for fam in parent_families:
                    fam_children = fam.sub_tags(TAG_CHILD)
                    if fam_children:
                        potential_siblings.extend(
                            (
                                c.value
                                if hasattr(c, "value") and _is_individual(c.value)
                                else c
                            )
                            for c in fam_children
                            if hasattr(c, "xref_id")
                        )

                for sibling in potential_siblings:
                    if (
                        sibling
                        and _is_individual(sibling)
                        and hasattr(sibling, "xref_id")
                        and sibling.xref_id
                    ):
                        sibling_id = _normalize_id(sibling.xref_id)
                        if (
                            sibling_id
                            and sibling_id not in unique_related_ids
                            and sibling_id != target_id
                        ):
                            related_individuals.append(sibling)
                            unique_related_ids.add(sibling_id)

            elif relationship_type in ["spouses", "children"]:
                parent_families = self._find_family_records_where_individual_is_parent(
                    target_id
                )
                if relationship_type == "spouses":
                    for (
                        family_record,
                        is_target_husband,
                        is_target_wife,
                    ) in parent_families:
                        other_spouse_tag = (
                            family_record.sub_tag(TAG_WIFE)
                            if is_target_husband
                            else family_record.sub_tag(TAG_HUSBAND)
                        )
                        other_spouse_indi = None
                        if (
                            other_spouse_tag
                            and hasattr(other_spouse_tag, "value")
                            and _is_individual(other_spouse_tag.value)
                        ):
                            other_spouse_indi = other_spouse_tag.value
                        elif _is_individual(other_spouse_tag):
                            other_spouse_indi = other_spouse_tag

                        if (
                            other_spouse_indi
                            and hasattr(other_spouse_indi, "xref_id")
                            and other_spouse_indi.xref_id
                        ):
                            spouse_id = _normalize_id(other_spouse_indi.xref_id)
                            if spouse_id and spouse_id not in unique_related_ids:
                                related_individuals.append(other_spouse_indi)
                                unique_related_ids.add(spouse_id)
                else:  # children
                    for family_record, _, _ in parent_families:
                        children_list = family_record.sub_tags(TAG_CHILD)
                        if children_list:
                            for child_tag in children_list:
                                child_indi = None
                                if (
                                    child_tag
                                    and hasattr(child_tag, "value")
                                    and _is_individual(child_tag.value)
                                ):
                                    child_indi = child_tag.value
                                elif _is_individual(child_tag):
                                    child_indi = child_tag

                                if (
                                    child_indi
                                    and hasattr(child_indi, "xref_id")
                                    and child_indi.xref_id
                                ):
                                    child_id = _normalize_id(child_indi.xref_id)
                                    if child_id and child_id not in unique_related_ids:
                                        related_individuals.append(child_indi)
                                        unique_related_ids.add(child_id)
            else:
                logger.warning(
                    f"Unknown relationship type requested: '{relationship_type}'"
                )
        except AttributeError as ae:
            logger.error(
                f"AttributeError finding {relationship_type} for {target_id}: {ae}",
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                f"Unexpected error finding {relationship_type} for {target_id}: {e}",
                exc_info=True,
            )

        # Sort results by ID for consistent display ordering
        related_individuals.sort(
            key=lambda x: (_normalize_id(getattr(x, "xref_id", None)) or "")
        )
        return related_individuals

    # End of get_related_individuals

    def get_relationship_path(self, id1: str, id2: str) -> str:
        """
        Calculates and formats relationship path using fast bidirectional BFS.
        """
        id1_norm = _normalize_id(id1)
        id2_norm = _normalize_id(id2)
        if not self.reader:
            return "Error: GEDCOM Reader unavailable."
        if not id1_norm or not id2_norm:
            return "Invalid input IDs."
        if id1_norm == id2_norm:
            return "Individuals are the same."

        # Ensure maps are available (should be built in __init__)
        if not self.id_to_parents and not self.id_to_children:
            logger.warning("Relationship maps are empty, attempting rebuild.")
            self._build_family_maps()
            if not self.id_to_parents and not self.id_to_children:
                return (
                    "Error: Family relationship maps could not be built or retrieved."
                )
        # Ensure index is available
        if not self.indi_index:
            logger.warning("Individual index is empty, attempting rebuild.")
            self._build_indi_index()
            if not self.indi_index:
                return "Error: Individual index could not be built or retrieved."

        max_depth = 25
        node_limit = 150000
        timeout_sec = 45
        logger.debug(
            f"Calculating relationship path (FastBiBFS): {id1_norm} <-> {id2_norm}"
        )
        search_start = time.time()

        path_ids = fast_bidirectional_bfs(  # Call standalone BFS function
            id1_norm,
            id2_norm,
            self.id_to_parents,
            self.id_to_children,
            max_depth,
            node_limit,
            timeout_sec,
            log_progress=False,
        )
        search_time = time.time() - search_start
        logger.debug(f"[PROFILE] BFS search completed in {search_time:.2f}s.")

        if not path_ids:
            profile_info = (
                f"[PROFILE] Search: {search_time:.2f}s, "
                f"MapsBuild: {self.family_maps_build_time:.2f}s, "
                f"IndexBuild: {self.indi_index_build_time:.2f}s"
            )
            return f"No relationship path found (FastBiBFS could not connect).\n{profile_info}"

        explanation_start = time.time()
        explanation_str = explain_relationship_path(  # Call standalone explain function
            path_ids,
            self.reader,
            self.id_to_parents,
            self.id_to_children,
            self.indi_index,
        )
        explanation_time = time.time() - explanation_start
        logger.debug(f"[PROFILE] Path explanation built in {explanation_time:.2f}s.")

        total_process_time = (
            search_time + explanation_time
        )  # Build time already accounted for
        profile_info = (
            f"[PROFILE] Total Time: {total_process_time:.2f}s "
            f"(BFS: {search_time:.2f}s, Explain: {explanation_time:.2f}s) "
            f"[Build Times: Maps={self.family_maps_build_time:.2f}s, Index={self.indi_index_build_time:.2f}s]"
        )
        logger.debug(profile_info)
        return f"{explanation_str}\n{profile_info}"

    # End of get_relationship_path

    def find_potential_matches(
        self,
        first_name: Optional[str],
        surname: Optional[str],
        dob_str: Optional[str],  # Birth date string
        pob: Optional[str],  # Birth place
        dod_str: Optional[str],  # Death date string
        pod: Optional[str],  # Death place
        gender: Optional[str] = None,
        max_results: int = 10,  # Fetch more initially for internal sorting/filtering
        scoring_weights: Optional[Dict] = None,  # Should be passed from config
        name_flexibility: Optional[Dict] = None,  # Should be passed from config
        date_flexibility: Optional[Dict] = None,  # Should be passed from config
    ) -> List[Dict]:
        """
        Finds potential matches in GEDCOM based on various criteria using fuzzy matching.
        Uses the common calculate_match_score function for scoring.
        Returns a list of match dictionaries.
        """
        if not self.reader:
            logger.error("find_potential_matches: No reader.")
            return []
        if not self.indi_index:
            logger.error("find_potential_matches: INDI_INDEX not built.")
            return []
        if (
            calculate_match_score is None
            or scoring_weights is None
            or name_flexibility is None
            or date_flexibility is None
        ):
            logger.error(
                "Scoring function or configurations not available. Cannot perform fuzzy match scoring."
            )
            return []

        weights = scoring_weights
        date_flex = date_flexibility
        name_flex = name_flexibility
        year_filter_range = 30
        clean_param = lambda p: p.strip().lower() if p and isinstance(p, str) else None

        target_first_name_lower = clean_param(first_name)
        target_surname_lower = clean_param(surname)
        target_pob_lower = clean_param(pob)
        target_pod_lower = clean_param(pod)
        target_gender_clean = (
            gender.strip().lower()[0]
            if gender
            and isinstance(gender, str)
            and gender.strip().lower() in ("m", "f")
            else None
        )

        target_birth_year: Optional[int] = None
        target_birth_date_obj: Optional[datetime] = None
        if dob_str:
            target_birth_date_obj = _parse_date(dob_str)
            target_birth_year = (
                target_birth_date_obj.year if target_birth_date_obj else None
            )
        target_death_year: Optional[int] = None
        target_death_date_obj: Optional[datetime] = None
        if dod_str:
            target_death_date_obj = _parse_date(dod_str)
            target_death_year = (
                target_death_date_obj.year if target_death_date_obj else None
            )

        search_criteria_dict = {
            "first_name": target_first_name_lower,
            "surname": target_surname_lower,
            "birth_year": target_birth_year,
            "birth_date_obj": target_birth_date_obj,
            "birth_place": target_pob_lower,
            "death_year": target_death_year,
            "death_date_obj": target_death_date_obj,
            "death_place": target_pod_lower,
            "gender": target_gender_clean,
        }

        if not any(
            [
                target_first_name_lower,
                target_surname_lower,
                target_birth_year,
                target_pob_lower,
                target_death_year,
                target_pod_lower,
            ]
        ):
            logger.warning(
                "Fuzzy search called with insufficient criteria (name, date, or place needed)."
            )
            return []

        candidate_count = 0
        scored_results = []
        for indi_id_norm, indi in self.indi_index.items():
            candidate_count += 1
            indi_id_raw = getattr(indi, "xref_id", None)
            try:
                indi_full_name = _get_full_name(indi)
                if indi_full_name.startswith("Unknown"):
                    continue

                birth_date_obj, birth_date_str_ged, birth_place_str_ged_raw = (
                    _get_event_info(indi, TAG_BIRTH)
                )
                death_date_obj, death_date_str_ged, death_place_str_ged_raw = (
                    _get_event_info(indi, TAG_DEATH)
                )
                birth_year_ged = birth_date_obj.year if birth_date_obj else None
                death_year_ged = death_date_obj.year if death_date_obj else None

                # Pre-filtering
                if (
                    birth_year_ged
                    and target_birth_year
                    and abs(birth_year_ged - target_birth_year) > year_filter_range
                ):
                    continue
                if (
                    death_year_ged
                    and target_death_year
                    and abs(death_year_ged - target_death_year) > year_filter_range
                ):
                    continue

                indi_name_parts = indi_full_name.lower().split()
                c_first_name_lower = indi_name_parts[0] if indi_name_parts else None
                c_surname_lower = (
                    indi_name_parts[-1] if len(indi_name_parts) > 1 else None
                )
                indi_gender_raw = getattr(indi, TAG_SEX.lower(), None)
                c_gender_clean = (
                    str(indi_gender_raw).strip().lower()[0]
                    if indi_gender_raw
                    and isinstance(indi_gender_raw, str)
                    and str(indi_gender_raw).strip().lower() in ("m", "f")
                    else None
                )

                candidate_data_dict = {
                    "first_name": c_first_name_lower,
                    "surname": c_surname_lower,
                    "birth_year": birth_year_ged,
                    "birth_date_obj": birth_date_obj,
                    "birth_place": (
                        birth_place_str_ged_raw.lower()
                        if birth_place_str_ged_raw != "N/A"
                        else None
                    ),
                    "death_year": death_year_ged,
                    "death_date_obj": death_date_obj,
                    "death_place": (
                        death_place_str_ged_raw.lower()
                        if death_place_str_ged_raw != "N/A"
                        else None
                    ),
                    "gender": c_gender_clean,
                }

                score, reasons = calculate_match_score(
                    search_criteria_dict,
                    candidate_data_dict,
                    weights,
                    name_flex,
                    date_flex,
                )

                if score > 0:
                    display_birth_date = _clean_display_date(birth_date_str_ged)
                    display_death_date = _clean_display_date(death_date_str_ged)
                    scored_results.append(
                        {
                            "id": indi_id_raw,
                            "norm_id": indi_id_norm,
                            "name": indi_full_name,
                            "birth_date": display_birth_date,
                            "birth_place": (
                                birth_place_str_ged_raw
                                if birth_place_str_ged_raw != "N/A"
                                else None
                            ),
                            "death_date": display_death_date,
                            "death_place": (
                                death_place_str_ged_raw
                                if death_place_str_ged_raw != "N/A"
                                else None
                            ),
                            "score": round(score),
                            "reasons": ", ".join(reasons) if reasons else "Score > 0",
                        }
                    )
            except Exception as loop_err:
                logger.error(
                    f"!!! ERROR processing individual {indi_id_raw} in find_potential_matches: {loop_err}",
                    exc_info=True,
                )
                continue

        logger.debug(
            f"Finished processing {candidate_count} individuals. Found {len(scored_results)} matches with score > 0."
        )
        scored_results.sort(
            key=lambda x: (
                x["score"],
                _parse_date(x.get("birth_date"))
                or datetime.max.replace(tzinfo=timezone.utc),
            ),
            reverse=True,
        )
        limited_results = scored_results[:max_results]
        logger.info(
            f"find_potential_matches returning top {len(limited_results)} of {len(scored_results)} total scored matches."
        )
        return limited_results

    # End of find_potential_matches


# End of GedcomData class


# --- Standalone Test Block ---
def self_check(verbose: bool = True) -> bool:
    """Performs internal self-checks for gedcom_utils.py."""
    status = True
    messages = []

    # Check if core dependencies loaded correctly
    messages.append(
        f"Logger: {'OK' if 'logger' in globals() and isinstance(logger, logging.Logger) else 'FAILED'}"
    )
    if "logger" not in globals() or not isinstance(logger, logging.Logger):
        status = False

    # Check if key utility functions are defined
    # Note: Some functions might be defined but rely on ged4py classes/objects
    key_functions = [
        ("_is_individual", _is_individual),
        ("_is_record", _is_record),
        ("_is_name", _is_name),
        ("_normalize_id", _normalize_id),
        ("extract_and_fix_id", extract_and_fix_id),
        ("_get_full_name", _get_full_name),
        ("_parse_date", _parse_date),
        ("_clean_display_date", _clean_display_date),
        ("_get_event_info", _get_event_info),
        ("format_life_dates", format_life_dates),
        ("format_full_life_details", format_full_life_details),
        ("format_relative_info", format_relative_info),
        ("_reconstruct_path", _reconstruct_path),
        ("explain_relationship_path", explain_relationship_path),
        ("fast_bidirectional_bfs", fast_bidirectional_bfs),
        ("calculate_match_score", calculate_match_score),
    ]
    # Check class and its core methods
    class_defined = "GedcomData" in globals() and GedcomData is not None
    messages.append(f"\nGedcomData Class: {'DEFINED' if class_defined else 'MISSING'}")
    if not class_defined:
        status = False
    else:
        # Check methods within the class
        class_methods = [
            ("__init__", GedcomData.__init__),
            ("build_caches", GedcomData.build_caches),
            ("_build_indi_index", GedcomData._build_indi_index),
            ("_build_family_maps", GedcomData._build_family_maps),
            ("find_individual_by_id", GedcomData.find_individual_by_id),
            (
                "_find_family_records_where_individual_is_child",
                GedcomData._find_family_records_where_individual_is_child,
            ),
            (
                "_find_family_records_where_individual_is_parent",
                GedcomData._find_family_records_where_individual_is_parent,
            ),
            ("get_related_individuals", GedcomData.get_related_individuals),
            ("get_relationship_path", GedcomData.get_relationship_path),
            ("find_potential_matches", GedcomData.find_potential_matches),
        ]
        key_functions.extend(class_methods)  # Add class methods to the check list

    messages.append("\n--- Function/Method Definitions ---")
    for func_name, func_obj in key_functions:
        is_method = (
            func_name in [m[0] for m in class_methods] if class_defined else False
        )
        prefix = "    (Method) " if is_method else "  "
        messages.append(
            f"{prefix}{func_name}: {'DEFINED' if func_obj is not None and callable(func_obj) else 'MISSING'}"
        )
        if func_obj is None or not callable(func_obj):
            status = False

    messages.append("\n--- Note ---")
    messages.append(
        "  Functional tests (requiring a GEDCOM file and config) are run separately below if __name__ == '__main__'."
    )
    messages.append(
        "  This check primarily verifies module imports, ged4py availability, and function/class definitions."
    )

    if verbose:
        print("\n[gedcom_utils.py self-check results]")
        for msg in messages:
            print("-", msg)
        print(f"Self-check status: {'PASS' if status else 'FAIL'}\n")

    # Log the final status
    if status:
        logger.info("gedcom_utils self-check status: PASS")
    else:
        logger.error("gedcom_utils self-check status: FAIL")

    return status


# End of self_check


if __name__ == "__main__":
    # --- Imports needed specifically for this block ---
    from config import config_instance  # Import config_instance
    import traceback  # Import traceback

    # --- End imports ---

    print("Running gedcom_utils.py self-check...")
    self_check_passed = self_check(verbose=True)
    print("\nThis is the gedcom_utils module. Import it into other scripts.")

    # --- Improved functional self-test using .env and GEDCOM file ---
    print("\n--- Functional Self-Test (using .env and GEDCOM) ---")
    gedcom_data: Optional[GedcomData] = None
    try:
        # Use config_instance loaded via import above
        if not config_instance or not hasattr(config_instance, "GEDCOM_FILE_PATH"):
            print(
                "[Functional Test] Config instance or GEDCOM_FILE_PATH not available."
            )
            raise SystemExit(1)  # Cannot proceed

        gedcom_path = config_instance.GEDCOM_FILE_PATH

        # --- Define Test IDs (MUST EXIST IN YOUR GEDCOM) ---
        # <<< Replace 'I1' and 'I50' with actual, valid @I...@ IDs (without @) from your GEDCOM file >>>
        # <<< Ensure they have a known relationship (e.g., parent/child, siblings) >>>
        TEST_INDI_ID_1 = "I102281560836"
        TEST_INDI_ID_2 = "I102281560744"
        # <<< Replace 'Wayne' and 'Gault' with a name known to be in your GEDCOM >>>
        TEST_SEARCH_NAME = "Wayne"
        TEST_SEARCH_SURNAME = "Gault"

        print(f"GEDCOM file path from config: {gedcom_path}")

        if not gedcom_path or not gedcom_path.exists() or not gedcom_path.is_file():
            print(
                f"[Functional Test] GEDCOM file not found or invalid: {gedcom_path}. Skipping functional tests."
            )
        # Use direct check for class existence
        elif "GedcomReader" not in globals() or GedcomReader is None:
            print(
                "[Functional Test] ged4py library/GedcomReader class not available. Skipping functional tests."
            )
        else:
            print("[Functional Test] Loading GedcomData...")
            gedcom_data = GedcomData(gedcom_path)  # Load data and build caches

            print(
                f"\n[Functional Test] Testing find_individual_by_id({TEST_INDI_ID_1})..."
            )
            indi1 = gedcom_data.find_individual_by_id(TEST_INDI_ID_1)
            if indi1:
                print(f"  Found: {_get_full_name(indi1)} {format_life_dates(indi1)}")
                # --- Check if indi1 was found before getting relatives ---
                print(
                    f"\n[Functional Test] Testing get_related_individuals({TEST_INDI_ID_1}, 'parents')..."
                )
                parents = gedcom_data.get_related_individuals(
                    indi1, "parents"
                )  # Now safe to call
                if parents:
                    print("  Parents found:")
                    for p in parents:
                        print(format_relative_info(p))
                else:
                    print("  No parents found.")
                # --- End Check ---
            else:
                print(f"  FAILED to find {TEST_INDI_ID_1}")
                # Skip tests that require indi1 if not found
                print(
                    f"\n[Functional Test] Skipping parent lookup for {TEST_INDI_ID_1} as individual was not found."
                )

            print(
                f"\n[Functional Test] Testing get_relationship_path({TEST_INDI_ID_1}, {TEST_INDI_ID_2})..."
            )
            # This test might still fail if ID1 or ID2 is invalid, but BFS handles missing nodes gracefully
            path_result = gedcom_data.get_relationship_path(
                TEST_INDI_ID_1, TEST_INDI_ID_2
            )
            print(f"  Path Result:\n{path_result}\n")

            print(
                f"\n[Functional Test] Testing find_potential_matches('{TEST_SEARCH_NAME}', '{TEST_SEARCH_SURNAME}')..."
            )
            # Load scoring config from config (assuming they are loaded there)
            scoring_weights = getattr(config_instance, "COMMON_SCORING_WEIGHTS", {})
            name_flex = getattr(config_instance, "NAME_FLEXIBILITY", {})
            date_flex = getattr(config_instance, "DATE_FLEXIBILITY", {})

            if not scoring_weights:
                logger.warning("COMMON_SCORING_WEIGHTS not found in config.")
            if not name_flex:
                logger.warning("NAME_FLEXIBILITY not found in config.")
            if not date_flex:
                logger.warning("DATE_FLEXIBILITY not found in config.")

            matches = gedcom_data.find_potential_matches(
                first_name=TEST_SEARCH_NAME,
                surname=TEST_SEARCH_SURNAME,
                dob_str=None,
                pob=None,
                dod_str=None,
                pod=None,  # Add more criteria if needed
                scoring_weights=scoring_weights,
                name_flexibility=name_flex,
                date_flexibility=date_flex,
                max_results=5,
            )
            if matches:
                print(f"  Found {len(matches)} potential match(es):")
                for match in matches:
                    print(
                        f"    - ID: {match.get('id', 'N/A')}, Name: {match.get('name', 'N/A')}, Score: {match.get('score', 0)}, Reasons: {match.get('reasons', '')}"
                    )
            else:
                print(
                    f"  No potential matches found for '{TEST_SEARCH_NAME} {TEST_SEARCH_SURNAME}'."
                )

    except FileNotFoundError as fnf_err:
        print(f"[Functional Test] ERROR: {fnf_err}")
    except ImportError as imp_err:
        print(f"[Functional Test] ERROR: Required library missing - {imp_err}")
    except AttributeError as attr_err:
        print(
            f"[Functional Test] ERROR: Missing attribute, check ged4py object structure or function calls: {attr_err}"
        )
        traceback.print_exc()
    except Exception as e:
        print(
            f"[Functional Test] Exception during functional test: {type(e).__name__} - {e}"
        )
        traceback.print_exc()  # Print full traceback for unexpected errors

    print("--- End of Functional Self-Test ---\n")

    # Exit based on the initial self-check only
    sys.exit(0 if self_check_passed else 1)
# End of gedcom_utils.py __main__ block
