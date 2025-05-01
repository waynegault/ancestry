# gedcom_utils.py
"""
Utility functions and class for loading, parsing, caching, and querying
GEDCOM data using ged4py. Includes relationship mapping, path calculation,
and fuzzy matching/scoring.
Consolidates helper functions and core logic from temp.py v7.36.
V17.13: Added debug prints to test runner.
"""

# --- Standard library imports ---
import logging
import sys
import re
import time
import os
from pathlib import Path
from typing import (
    List,
    Optional,
    Dict,
    Tuple,
    Set,
    Deque,
    Union,
    Any,
    Callable,
    TypeAlias,
)
from collections import deque
from datetime import datetime, timezone
import difflib
import traceback

# --- Third-party imports ---
from ged4py.parser import GedcomReader
from ged4py.model import Individual, Record, Name, NameRec
from utils import format_name, ordinal_case
from config import config_instance

# --- Constants ---
GedcomIndividualType: TypeAlias = Individual
GedcomRecordType: TypeAlias = Record
GedcomNameType: TypeAlias = Name
GedcomNameRecType: TypeAlias = NameRec
GedcomReaderType: TypeAlias = GedcomReader
TAG_INDI = "INDI"
TAG_BIRTH = "BIRT"
TAG_DEATH = "DEAT"
TAG_HUSBAND = "HUSB"
TAG_WIFE = "WIFE"
TAG_CHILD = "CHIL"
TAG_FAMILY_CHILD = "FAMC"
TAG_FAMILY_SPOUSE = "FAMS"
TAG_DATE = "DATE"
TAG_PLACE = "PLAC"
TAG_SEX = "SEX"
TAG_NAME = "NAME"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gedcom_utils")


# ==============================================
# Utility Functions (Independent of GedcomData)
# ==============================================

def _is_individual(obj: Any) -> bool:
    return (
         obj is not None and isinstance(obj, GedcomIndividualType)
    )


def _is_record(obj: Any) -> bool:
    return  obj is not None and isinstance(obj, GedcomRecordType)


def _is_name_rec(obj: Any) -> bool:
    return  obj is not None and isinstance(obj, GedcomNameRecType)


def _normalize_id(xref_id: Optional[str]) -> Optional[str]:
    if not xref_id or not isinstance(xref_id, str):
        return None
    # Updated regex to restrict to digits/hyphens after prefix
    match = re.match(r"^@?([IFSNMCXO][0-9\-]+)@?$", xref_id.strip().upper())
    if match:
        return match.group(1)
    else:
        # Updated fallback regex to match same pattern
        search_match = re.search(r"([IFSNMCXO][0-9\-]+)", xref_id.strip().upper())
        if search_match:
            logger.debug(f"...")
            return search_match.group(1)
        else:
            logger.warning(f"...")
            return None


def extract_and_fix_id(raw_id: Any) -> Optional[str]:
    if not raw_id:
        return None
    id_to_normalize: Optional[str] = None
    if isinstance(raw_id, str):
        id_to_normalize = raw_id
    elif _is_record(raw_id) and hasattr(raw_id, "xref_id"):
        id_to_normalize = getattr(raw_id, "xref_id", None)
    else:
        logger.debug(
            f"extract_and_fix_id: Invalid input type '{type(raw_id).__name__}'."
        )
        return None
    return _normalize_id(id_to_normalize)


def _get_full_name(indi: GedcomIndividualType) -> str:
    """Safely gets formatted name using Name.format(). Handles None/errors."""
    if not _is_individual(indi):
        if hasattr(indi, "value") and _is_individual(getattr(indi, "value", None)):
            indi = indi.value
        else:
            logger.warning(
                f"_get_full_name called with non-Individual type: {type(indi)}"
            )
            return "Unknown (Invalid Type)"
    try:
        # First try to access the name attribute directly (as in temp.py)
        if hasattr(indi, "name"):
            name_rec = indi.name
            if _is_name_rec(name_rec):
                # Use the ged4py Name object's format method
                formatted_name = name_rec.format()
                # Clean up and format the name
                cleaned_name = format_name(formatted_name)
                return (
                    cleaned_name
                    if cleaned_name and cleaned_name != "Unknown"
                    else "Unknown (Empty Name)"
                )

        # Fallback to using sub_tag method if name attribute doesn't exist or isn't a Name object
        name_rec = indi.sub_tag(TAG_NAME)
        if _is_name_rec(name_rec):
            ged4py_formatted_name = name_rec.format()
            cleaned_name = format_name(ged4py_formatted_name)
            if cleaned_name and cleaned_name != "Unknown":
                return cleaned_name

        # Try sub_tag_value as a last resort
        name_val = indi.sub_tag_value(TAG_NAME)
        if isinstance(name_val, str) and name_val.strip():
            logger.debug(f"Using sub_tag_value fallback for name: {name_val}")
            cleaned_name = format_name(name_val)
            if cleaned_name and cleaned_name != "Unknown":
                return cleaned_name

        return "Unknown (No Name Found)"
    except AttributeError as ae:
        indi_id_log = extract_and_fix_id(indi) or "Unknown ID"
        logger.debug(f"AttributeError getting name for @{indi_id_log}@: {ae}")
        return "Unknown (Attribute Error)"
    except Exception as e:
        indi_id_log = extract_and_fix_id(indi) or "Unknown ID"
        logger.error(f"Error formatting name for @{indi_id_log}@: {e}", exc_info=False)
        return "Unknown (Error)"


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str or not isinstance(date_str, str):
        return None
    original_date_str = date_str
    date_str = date_str.strip().upper()
    clean_date_str = re.sub(r"^(ABT|EST|CAL|INT|BEF|AFT)\s+", "", date_str).strip()
    clean_date_str = re.sub(r"^(BET|FROM)\s+", "", clean_date_str).strip()
    clean_date_str = re.sub(r"\s+(AND|TO)\s+.*", "", clean_date_str).strip()
    clean_date_str = re.sub(r"\s*@#D[A-Z]+@\s*$", "", clean_date_str).strip()
    clean_date_str = re.sub(r"\s*\(.*\)\s*", "", clean_date_str).strip()
    clean_date_str = re.sub(r"\s+(BC|AD)$", "", clean_date_str).strip()
    formats = ["%d %b %Y", "%d %B %Y", "%b %Y", "%B %Y", "%Y"]
    for fmt in formats:
        try:
            if fmt == "%Y" and not re.fullmatch(r"\d{1,4}", clean_date_str):
                continue
            dt_naive = datetime.strptime(clean_date_str, fmt)
            return dt_naive.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
        except Exception as e:
            logger.debug(
                f"Date parsing err for '{original_date_str}' (clean:'{clean_date_str}', fmt:'{fmt}'): {e}"
            )
            continue
    logger.debug(
        f"Failed to parse date string: '{original_date_str}' (Cleaned: '{clean_date_str}')"
    )
    return None


def _clean_display_date(raw_date_str: Optional[str]) -> str:
    if not raw_date_str or not isinstance(raw_date_str, str) or raw_date_str == "N/A":
        return "N/A"
    cleaned = raw_date_str.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        content = cleaned[1:-1].strip()
        cleaned = content if content else "N/A"
    cleaned = re.sub(r"^(ABT|ABOUT)\s+", "~", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(EST|ESTIMATED)\s+", "~", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^CAL\s+", "~", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^INT\s+", "~", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^BEF\s+", "<", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^AFT\s+", ">", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(BET|BETWEEN)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^(FROM)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+(AND|TO)\s+", "-", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*@#D[A-Z]+@\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned if cleaned else "N/A"


def _get_event_info(
    individual: GedcomIndividualType, event_tag: str
) -> Tuple[Optional[datetime], str, str]:
    date_obj: Optional[datetime] = None
    date_str: str = "N/A"
    place_str: str = "N/A"
    if not _is_individual(individual):
        if hasattr(individual, "value") and _is_individual(
            getattr(individual, "value", None)
        ):
            individual = individual.value
        else:
            logger.warning(f"_get_event_info invalid input type: {type(individual)}")
            return date_obj, date_str, place_str
    indi_id_log = extract_and_fix_id(individual) or "Unknown ID"
    try:
        event_record = individual.sub_tag(event_tag.upper())
        if not event_record:
            return date_obj, date_str, place_str
        date_tag = event_record.sub_tag(TAG_DATE)
        raw_date_val = getattr(date_tag, "value", None) if date_tag else None
        if isinstance(raw_date_val, str) and raw_date_val.strip():
            date_str = raw_date_val.strip()
            date_obj = _parse_date(date_str)
        elif raw_date_val is not None:
            date_str = str(raw_date_val)
            date_obj = _parse_date(date_str)
        place_tag = event_record.sub_tag(TAG_PLACE)
        raw_place_val = getattr(place_tag, "value", None) if place_tag else None
        if isinstance(raw_place_val, str) and raw_place_val.strip():
            place_str = raw_place_val.strip()
        elif raw_place_val is not None:
            place_str = str(raw_place_val)
    except AttributeError as ae:
        logger.debug(
            f"Attribute error getting event '{event_tag}' for {indi_id_log}: {ae}"
        )
    except Exception as e:
        logger.error(
            f"Error accessing event {event_tag} for @{indi_id_log}@: {e}", exc_info=True
        )
    return date_obj, date_str, place_str


def format_life_dates(indi: GedcomIndividualType) -> str:
    if not _is_individual(indi):
        logger.warning(
            f"format_life_dates called with non-Individual type: {type(indi)}"
        )
        return ""
    _, b_date_str, _ = _get_event_info(indi, TAG_BIRTH)
    _, d_date_str, _ = _get_event_info(indi, TAG_DEATH)
    b_date_str_cleaned = _clean_display_date(b_date_str)
    d_date_str_cleaned = _clean_display_date(d_date_str)
    birth_info = f"b. {b_date_str_cleaned}" if b_date_str_cleaned != "N/A" else ""
    death_info = f"d. {d_date_str_cleaned}" if d_date_str_cleaned != "N/A" else ""
    life_parts = [info for info in [birth_info, death_info] if info]
    return f" ({', '.join(life_parts)})" if life_parts else ""


def format_full_life_details(indi: GedcomIndividualType) -> Tuple[str, str]:
    if not _is_individual(indi):
        logger.warning(
            f"format_full_life_details called with non-Individual type: {type(indi)}"
        )
        return "(Error: Invalid data)", ""
    _, b_date_str, b_place = _get_event_info(indi, TAG_BIRTH)
    b_date_str_cleaned = _clean_display_date(b_date_str)
    b_place_cleaned = b_place if b_place != "N/A" else "(Place unknown)"
    birth_info = f"Born: {b_date_str_cleaned if b_date_str_cleaned != 'N/A' else '(Date unknown)'} in {b_place_cleaned}"
    _, d_date_str, d_place = _get_event_info(indi, TAG_DEATH)
    d_date_str_cleaned = _clean_display_date(d_date_str)
    d_place_cleaned = d_place if d_place != "N/A" else "(Place unknown)"
    death_info = ""
    if d_date_str_cleaned != "N/A" or d_place != "N/A":
        death_info = f"   Died: {d_date_str_cleaned if d_date_str_cleaned != 'N/A' else '(Date unknown)'} in {d_place_cleaned}"
    return birth_info, death_info


def format_relative_info(relative: Any) -> str:
    indi_obj: Optional[GedcomIndividualType] = None
    if _is_individual(relative):
        indi_obj = relative
    elif hasattr(relative, "value") and _is_individual(
        getattr(relative, "value", None)
    ):
        indi_obj = relative.value
    elif hasattr(relative, "xref_id") and isinstance(
        getattr(relative, "xref_id", None), str
    ):
        norm_id = extract_and_fix_id(relative)
        return f"  - (Relative Data: ID={norm_id or 'N/A'}, Type={type(relative).__name__})"
    else:
        return f"  - (Invalid Relative Data: Type={type(relative).__name__})"
    rel_name = _get_full_name(indi_obj)
    life_info = format_life_dates(indi_obj)
    return f"  - {rel_name}{life_info}"


def _reconstruct_path(
    start_id: str,
    end_id: str,
    meeting_id: str,
    visited_fwd: Dict[str, Optional[str]],
    visited_bwd: Dict[str, Optional[str]],
) -> List[str]:
    path_fwd: List[str] = []
    curr = meeting_id
    loop_guard = 0
    max_loops = len(visited_fwd) + 5
    while curr is not None and loop_guard < max_loops:
        path_fwd.append(curr)
        curr = visited_fwd.get(curr)
        loop_guard += 1
    if loop_guard >= max_loops:
        logger.error("_reconstruct_path: Loop guard hit reconstructing forward path!")
    path_fwd.reverse()
    path_bwd: List[str] = []
    curr = visited_bwd.get(meeting_id)
    loop_guard = 0
    max_loops = len(visited_bwd) + 5
    while curr is not None and loop_guard < max_loops:
        path_bwd.append(curr)
        curr = visited_bwd.get(curr)
        loop_guard += 1
    if loop_guard >= max_loops:
        logger.error("_reconstruct_path: Loop guard hit reconstructing backward path!")
    path = path_fwd + path_bwd
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
    start_time = time.time()
    if start_id == end_id:
        return [start_id]
    if id_to_parents is None or id_to_children is None:
        logger.error("[FastBiBFS] Relationship maps are None.")
        return []
    if not start_id or not end_id:
        logger.error("[FastBiBFS] Start or end ID is missing.")
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
            logger.warning(f"[FastBiBFS] Timeout after {timeout_sec:.1f} seconds.")
            return []
        if processed > node_limit:
            logger.warning(
                f"[FastBiBFS] Node limit ({node_limit}) reached. Processed: ~{processed}."
            )
            return []
        if log_progress and processed > 0 and processed % 10000 == 0:
            logger.info(
                f"[FastBiBFS] Progress: ~{processed} nodes, QF:{len(queue_fwd)}, QB:{len(queue_bwd)}"
            )
        if not queue_fwd:
            break
        current_id_fwd, depth_fwd = queue_fwd.popleft()
        processed += 1
        if current_id_fwd in visited_bwd:
            meeting_id = current_id_fwd
            logger.debug(
                f"[FastBiBFS] Path found (FWD meets BWD) at {meeting_id} (Depth FWD: {depth_fwd})."
            )
            break
        if depth_fwd >= max_depth:
            continue
        neighbors_fwd = id_to_parents.get(current_id_fwd, set()) | id_to_children.get(
            current_id_fwd, set()
        )
        for neighbor_id in neighbors_fwd:
            if neighbor_id not in visited_fwd:
                visited_fwd[neighbor_id] = current_id_fwd
                queue_fwd.append((neighbor_id, depth_fwd + 1))
            if neighbor_id in visited_bwd:
                meeting_id = neighbor_id
                logger.debug(
                    f"[FastBiBFS] Path found (FWD adds node visited by BWD) at {meeting_id} (Depth FWD: {depth_fwd+1})."
                )
                break
        if meeting_id:
            break
        if not queue_bwd:
            break
        current_id_bwd, depth_bwd = queue_bwd.popleft()
        processed += 1
        if current_id_bwd in visited_fwd:
            meeting_id = current_id_bwd
            logger.debug(
                f"[FastBiBFS] Path found (BWD meets FWD) at {meeting_id} (Depth BWD: {depth_bwd})."
            )
            break
        if depth_bwd >= max_depth:
            continue
        neighbors_bwd = id_to_parents.get(current_id_bwd, set()) | id_to_children.get(
            current_id_bwd, set()
        )
        for neighbor_id in neighbors_bwd:
            if neighbor_id not in visited_bwd:
                visited_bwd[neighbor_id] = current_id_bwd
                queue_bwd.append((neighbor_id, depth_bwd + 1))
            if neighbor_id in visited_fwd:
                meeting_id = neighbor_id
                logger.debug(
                    f"[FastBiBFS] Path found (BWD adds node visited by FWD) at {meeting_id} (Depth BWD: {depth_bwd+1})."
                )
                break
        if meeting_id:
            break
    if meeting_id:
        logger.debug(
            f"[FastBiBFS] Intersection found at {meeting_id}. Reconstructing path..."
        )
        try:
            path_ids = _reconstruct_path(
                start_id, end_id, meeting_id, visited_fwd, visited_bwd
            )
            logger.debug(
                f"[FastBiBFS] Path reconstruction complete. Length: {len(path_ids)}"
            )
            return path_ids
        except Exception as recon_err:
            logger.error(
                f"[FastBiBFS] Error during path reconstruction: {recon_err}",
                exc_info=True,
            )
            return []
    else:
        reason = "Queues Emptied"
    if time.time() - start_time > timeout_sec:
        reason = "Timeout"
    elif processed > node_limit:
        reason = "Node Limit Reached"
    logger.warning(
        f"[FastBiBFS] No path found between {start_id} and {end_id}. Reason: {reason}. Processed ~{processed} nodes."
    )
    return []


def explain_relationship_path(
    path_ids: List[str],
    reader: GedcomReaderType,
    id_to_parents: Dict[str, Set[str]],
    id_to_children: Dict[str, Set[str]],
    indi_index: Dict[str, GedcomIndividualType],
) -> str:
    if not path_ids or len(path_ids) < 2:
        return "(No relationship path explanation available)"
    if reader is None:
        return "(Error: GedcomReader not available)"
    if id_to_parents is None or id_to_children is None or indi_index is None:
        return "(Error: Data maps or index unavailable)"
    steps: List[str] = []
    ordinal_formatter = ordinal_case
    for i in range(len(path_ids) - 1):
        id_a, id_b = path_ids[i], path_ids[i + 1]
        indi_a = indi_index.get(id_a)
        indi_b = indi_index.get(id_b)
        name_a = _get_full_name(indi_a) if indi_a else f"Unknown ({id_a})"
        name_b = _get_full_name(indi_b) if indi_b else f"Unknown ({id_b})"
        label = "related"
        if id_b in id_to_parents.get(id_a, set()):
            sex_b = getattr(indi_b, TAG_SEX.lower(), None) if indi_b else None
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
        elif id_a in id_to_parents.get(id_b, set()):
            sex_a = getattr(indi_a, TAG_SEX.lower(), None) if indi_a else None
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
        else:
            parents_a = id_to_parents.get(id_a, set())
            parents_b = id_to_parents.get(id_b, set())
            if parents_a and parents_b and parents_a == parents_b:
                sex_b = getattr(indi_b, TAG_SEX.lower(), None) if indi_b else None
                sex_b_char = (
                    str(sex_b).upper()[0]
                    if sex_b
                    and isinstance(sex_b, str)
                    and str(sex_b).upper() in ("M", "F")
                    else None
                )
                label = (
                    "sister"
                    if sex_b_char == "F"
                    else "brother" if sex_b_char == "M" else "sibling"
                )
            else:
                logger.warning(
                    f"Could not determine direct relation between {id_a} ({name_a}) and {id_b} ({name_b}) for path explanation."
                )
                label = "connected to"
        steps.append(f"{name_a} is the {label} of {name_b}")
    start_person_indi = indi_index.get(path_ids[0])
    start_person_name = (
        _get_full_name(start_person_indi)
        if start_person_indi
        else f"Unknown ({path_ids[0]})"
    )
    explanation_str = f"{start_person_name}\n -> " + "\n -> ".join(steps)
    return explanation_str


def calculate_match_score(
    search_criteria: Dict,
    candidate_data: Dict,
    scoring_weights: Optional[Dict] = None,
    name_flexibility: Optional[Dict] = None,
    date_flexibility: Optional[Dict] = None,
) -> Tuple[float, List[str]]:
    score = 0.0
    match_reasons: List[str] = []

    # --- Get Scoring Parameters ---
    weights = (
        scoring_weights
        if scoring_weights is not None
        else (
            getattr(config_instance, "COMMON_SCORING_WEIGHTS", {})
            if config_instance
            else {}
        )
    )
    date_flex = (
        date_flexibility
        if date_flexibility is not None
        else getattr(config_instance, "DATE_FLEXIBILITY", {}) if config_instance else {}
    )
    name_flex = (
        name_flexibility
        if name_flexibility is not None
        else getattr(config_instance, "NAME_FLEXIBILITY", {}) if config_instance else {}
    )
    year_score_range = date_flex.get("year_match_range", 1)
    fuzzy_threshold = name_flex.get("fuzzy_threshold", 0.8)
    check_starts_with = name_flex.get("check_starts_with", True)

    # --- Prepare Target Data ---
    t_fname_raw = search_criteria.get("first_name")
    t_sname_raw = search_criteria.get("surname")
    t_pob_raw = search_criteria.get("birth_place")
    t_pod_raw = search_criteria.get("death_place")
    t_b_year = search_criteria.get("birth_year")
    t_b_date = search_criteria.get("birth_date_obj")
    t_d_year = search_criteria.get("death_year")
    t_d_date = search_criteria.get("death_date_obj")
    t_gender = search_criteria.get("gender")
    t_fname = t_fname_raw.lower() if isinstance(t_fname_raw, str) else ""
    t_sname = t_sname_raw.lower() if isinstance(t_sname_raw, str) else ""
    t_pob = t_pob_raw.lower() if isinstance(t_pob_raw, str) else ""
    t_pod = t_pod_raw.lower() if isinstance(t_pod_raw, str) else ""

    # --- Prepare Candidate Data ---
    c_fname_raw = candidate_data.get("first_name")
    c_sname_raw = candidate_data.get("surname")
    c_bplace_raw = candidate_data.get("birth_place")
    c_dplace_raw = candidate_data.get("death_place")
    c_b_year = candidate_data.get("birth_year")
    c_b_date = candidate_data.get("birth_date_obj")
    c_d_year = candidate_data.get("death_year")
    c_d_date = candidate_data.get("death_date_obj")
    c_gender = candidate_data.get("gender")
    c_fname = c_fname_raw.lower() if isinstance(c_fname_raw, str) else ""
    c_sname = c_sname_raw.lower() if isinstance(c_sname_raw, str) else ""
    c_bplace = c_bplace_raw if isinstance(c_bplace_raw, str) else ""
    c_dplace = c_dplace_raw if isinstance(c_dplace_raw, str) else ""

    # --- Name Scoring ---
    exact_first = bool(t_fname and c_fname and t_fname == c_fname)
    exact_surname = bool(t_sname and c_sname and t_sname == c_sname)
    fuzzy_first_ratio = (
        difflib.SequenceMatcher(None, t_fname, c_fname).ratio()
        if t_fname and c_fname
        else 0.0
    )
    fuzzy_surname_ratio = (
        difflib.SequenceMatcher(None, t_sname, c_sname).ratio()
        if t_sname and c_sname
        else 0.0
    )
    starts_first = bool(
        check_starts_with and t_fname and c_fname and c_fname.startswith(t_fname)
    )
    starts_surname = bool(
        check_starts_with and t_sname and c_sname and c_sname.startswith(t_sname)
    )
    name_score = 0.0
    temp_name_reasons = []
    if exact_first:
        name_score += weights.get("exact_first_name", 20)
        temp_name_reasons.append("Exact First")
    elif fuzzy_first_ratio >= fuzzy_threshold:
        name_score += weights.get("fuzzy_first_name", 15) * fuzzy_first_ratio
        temp_name_reasons.append(f"Fuzzy First ({fuzzy_first_ratio:.2f})")
    elif starts_first:
        name_score += weights.get("starts_first_name", 3)
        temp_name_reasons.append("Starts First")
    if exact_surname:
        name_score += weights.get("exact_surname", 20)
        temp_name_reasons.append("Exact Surname")
    elif fuzzy_surname_ratio >= fuzzy_threshold:
        name_score += weights.get("fuzzy_surname", 15) * fuzzy_surname_ratio
        temp_name_reasons.append(f"Fuzzy Surname ({fuzzy_surname_ratio:.2f})")
    elif starts_surname:
        name_score += weights.get("starts_surname", 5)
        temp_name_reasons.append("Starts Surname")
    if exact_first and exact_surname:
        name_score += weights.get("boost_exact_full_name", 20)
        temp_name_reasons.append("Boost Exact Name")
    score += name_score
    match_reasons.extend(temp_name_reasons)

    # --- Date Scoring ---
    exact_birth_date_match = bool(
        t_b_date and c_b_date and t_b_date.date() == c_b_date.date()
    )
    exact_death_date_match = bool(
        t_d_date and c_d_date and t_d_date.date() == c_d_date.date()
    )
    birth_year_match = bool(
        t_b_year is not None
        and c_b_year is not None
        and abs(c_b_year - t_b_year) <= year_score_range
    )
    death_year_match = bool(
        t_d_year is not None
        and c_d_year is not None
        and abs(c_d_year - t_d_year) <= year_score_range
    )
    death_dates_absent = bool(
        t_d_date is None and c_d_date is None and t_d_year is None and c_d_year is None
    )
    temp_date_reasons = []
    birth_year_matched_for_boost = False
    if exact_birth_date_match:
        score += weights.get("exact_birth_date", 20)
        temp_date_reasons.append("Exact Birth Date")
        birth_year_matched_for_boost = True
    elif birth_year_match:
        year_weight = (
            weights.get("year_birth", 15)
            if (exact_first and exact_surname)
            else weights.get("year_birth_fuzzy", 5)
        )
        score += year_weight
        temp_date_reasons.append(f"Birth Year ~{t_b_year} ({c_b_year})")
        birth_year_matched_for_boost = True
    if (
        not (exact_first and exact_surname)
        and birth_year_match
        and not exact_birth_date_match
    ):
        temp_date_reasons[
            -1
        ] += " (Fuzzy Wt)"  # Check added for not exact_birth_date_match
    if exact_death_date_match:
        score += weights.get("exact_death_date", 20)
        temp_date_reasons.append("Exact Death Date")
    elif death_year_match:
        year_weight = (
            weights.get("year_death", 15)
            if (exact_first and exact_surname)
            else weights.get("year_death_fuzzy", 5)
        )
        score += year_weight
        temp_date_reasons.append(f"Death Year ~{t_d_year} ({c_d_year})")
    if (
        not (exact_first and exact_surname)
        and death_year_match
        and not exact_death_date_match
    ):
        temp_date_reasons[
            -1
        ] += " (Fuzzy Wt)"  # Check added for not exact_death_date_match
    elif death_dates_absent:
        score += weights.get("death_dates_both_absent", 5)
        temp_date_reasons.append("Death Dates Both Absent")
    match_reasons.extend(temp_date_reasons)
    if exact_first and exact_surname and birth_year_matched_for_boost:
        score += weights.get("boost_exact_name_year", 2)
        match_reasons.append("Boost Exact Name + Year")

    # --- Place Scoring ---
    temp_place_reasons = []
    if t_pob and c_bplace:  # Check if both exist
        # Convert both to lowercase for case-insensitive comparison
        t_pob_lower = t_pob.lower()
        c_bplace_lower = (c_bplace or "").lower()
        # Determine weight first
        place_weight = (
            weights.get("contains_pob", 15)
            if (exact_first and exact_surname)
            else weights.get("contains_pob_fuzzy", 1)
        )
        if t_pob_lower in c_bplace_lower:
            score += place_weight
            temp_place_reasons.append(f"POB contains '{t_pob}'")

    if t_pod and c_dplace:  # Check if both exist
        # Convert both to lowercase
        t_pod_lower = t_pod.lower()
        c_dplace_lower = (c_dplace or "").lower()
        place_weight = (
            weights.get("contains_pod", 15)
            if (exact_first and exact_surname)
            else weights.get("contains_pod_fuzzy", 1)
        )
        if t_pod_lower in c_dplace_lower:
            score += place_weight
            temp_place_reasons.append(f"POD contains '{t_pod}'")

    match_reasons.extend(temp_place_reasons)

    # --- Gender Scoring ---
    temp_gender_reasons = []
    if t_gender and c_gender:
        if t_gender == c_gender:
            gender_weight = (
                weights.get("gender_match", 20)
                if (exact_first and exact_surname)
                else weights.get("gender_match_fuzzy", 3)
            )
            score += gender_weight
            temp_gender_reasons.append(f"Gender ({t_gender.upper()})")
        else:
            penalty = (
                weights.get("gender_mismatch_penalty", -20)
                if (exact_first and exact_surname)
                else weights.get("gender_mismatch_penalty_fuzzy", -3)
            )
            score += penalty
            temp_gender_reasons.append(
                f"Gender Mismatch ({c_gender.upper()} vs {t_gender.upper()})"
            )
    match_reasons.extend(temp_gender_reasons)

    # --- Final Score ---
    final_score = max(0.0, round(score))
    unique_reasons = sorted(list(set(match_reasons)))
    return final_score, unique_reasons
# End of function calculate_match_score

# ==============================================
# GedcomData Class (definition remains unchanged)
# ==============================================
class GedcomData:
    # --- Methods remain identical to previous version ---
    def __init__(self, gedcom_path: Union[str, Path]):
        self.path = Path(gedcom_path).resolve()
        self.reader: Optional[GedcomReaderType] = None
        self.indi_index: Dict[str, GedcomIndividualType] = {}
        self.id_to_parents: Dict[str, Set[str]] = {}
        self.id_to_children: Dict[str, Set[str]] = {}
        self.indi_index_build_time: float = 0
        self.family_maps_build_time: float = 0
        if not self.path.is_file():
            logger.critical(f"GEDCOM file not found or is not a file: {self.path}")
            raise FileNotFoundError(f"GEDCOM file not found: {self.path}")
        try:
            logger.info(f"Loading GEDCOM file: {self.path}")
            load_start = time.time()
            self.reader = GedcomReader(str(self.path))
            load_time = time.time() - load_start
            logger.info(f"GEDCOM file loaded in {load_time:.2f}s.")
        except Exception as e:
            logger.critical(
                f"Failed to load or parse GEDCOM file {self.path}: {e}", exc_info=True
            )
            raise
        self.build_caches()

    def build_caches(self):
        self._build_indi_index()
        self._build_family_maps()

    def _build_indi_index(self):
        if not self.reader:
            return
        start_time = time.time()
        logger.info("[Cache] Building INDI index...")
        self.indi_index = {}
        count = 0
        skipped = 0
        try:
            for indi_record in self.reader.records0(TAG_INDI):  # Corrected tag
                if (
                    _is_individual(indi_record)
                    and hasattr(indi_record, "xref_id")
                    and indi_record.xref_id
                ):
                    norm_id = _normalize_id(indi_record.xref_id)
                    if norm_id:
                        self.indi_index[norm_id] = indi_record
                        count += 1
                    else:
                        skipped += 1
                        logger.debug(
                            f"Skipping INDI with unnormalizable xref_id: {indi_record.xref_id}"
                        )
                elif hasattr(indi_record, "xref_id"):
                    skipped += 1
                    logger.debug(
                        f"Skipping non-Individual record: Type={type(indi_record).__name__}, Xref={indi_record.xref_id}"
                    )
                else:
                    skipped += 1
                    logger.debug(
                        f"Skipping record with no xref_id: Type={type(indi_record).__name__}"
                    )
        except Exception as e:
            logger.error(f"Error during INDI index build: {e}", exc_info=True)
            self.indi_index = {}
        elapsed = time.time() - start_time
        self.indi_index_build_time = elapsed
        logger.info(
            f"[Cache] INDI index built with {count} individuals ({skipped} skipped) in {elapsed:.2f}s."
        )

    def _build_family_maps(self):
        if not self.reader:
            return
        start_time = time.time()
        logger.info("[Cache] Building family maps (direct tag access)...")
        self.id_to_parents = {}
        self.id_to_children = {}
        fam_count = 0
        processed_links = 0
        skipped_links = 0
        try:
            for fam in self.reader.records0("FAM"):
                fam_count += 1
                if not _is_record(fam):
                    continue
                parents: Set[str] = set()
                for parent_tag in [TAG_HUSBAND, TAG_WIFE]:
                    parent_ref = fam.sub_tag(parent_tag)
                    if parent_ref and hasattr(parent_ref, "xref_id"):
                        parent_id = _normalize_id(parent_ref.xref_id)
                    if parent_id:
                        parents.add(parent_id)
                    else:
                        logger.debug(
                            f"Skipping parent with invalid ID {getattr(parent_ref, 'xref_id', '?')} in FAM {getattr(fam, 'xref_id', 'N/A')}"
                        )  # Safely access xref_id
                children_tags = fam.sub_tags(TAG_CHILD)
                for child_tag in children_tags:
                    if child_tag and hasattr(child_tag, "xref_id"):
                        child_id = _normalize_id(child_tag.xref_id)
                        if child_id:
                            if parents:
                                self.id_to_parents.setdefault(child_id, set()).update(
                                    parents
                                )
                                processed_links += len(parents)
                            for parent_id in parents:
                                self.id_to_children.setdefault(parent_id, set()).add(
                                    child_id
                                )
                        else:
                            skipped_links += 1
                            logger.debug(
                                f"Skipping child with invalid ID {getattr(child_tag, 'xref_id', '?')} in FAM {getattr(fam, 'xref_id', 'N/A')}"
                            )
                    elif child_tag is not None:
                        skipped_links += 1
                        logger.debug(
                            f"Skipping CHIL record in FAM {getattr(fam, 'xref_id', 'N/A')} with invalid format: Type={type(child_tag).__name__}"
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
            f"[Cache] Family maps built: {fam_count} FAMs processed. Found {processed_links} parent/child links ({skipped_links} skipped). Map sizes: {len(self.id_to_parents)} child->parents, {len(self.id_to_children)} parent->children in {elapsed:.2f}s."
        )

    def find_individual_by_id(
        self, norm_id: Optional[str]
    ) -> Optional[GedcomIndividualType]:
        if not norm_id or not isinstance(norm_id, str):
            logger.warning("find_individual_by_id called with invalid norm_id")
            return None
        if not self.indi_index:
            logger.warning("INDI_INDEX not built. Attempting build now.")
            self._build_indi_index()
        if not self.indi_index:
            logger.error("INDI_INDEX build failed. Cannot lookup individual by ID.")
            return None
        found_indi = self.indi_index.get(norm_id)
        if not found_indi:
            logger.debug(
                f"Individual with normalized ID {norm_id} not found in INDI_INDEX."
            )
        return found_indi

    def _find_family_records(
        self, target_id: str, role_tag: str
    ) -> List[GedcomRecordType]:
        matching_families: List[GedcomRecordType] = []
        if not self.reader or not target_id or not role_tag:
            return matching_families
        try:
            indi = self.find_individual_by_id(target_id)
            link_tag = (
                TAG_FAMILY_SPOUSE
                if role_tag in [TAG_HUSBAND, TAG_WIFE]
                else TAG_FAMILY_CHILD if role_tag == TAG_CHILD else None
            )
            if indi and link_tag:
                fam_links = indi.sub_tags(link_tag)
                for fam_link in fam_links:
                    fam_record: Optional[GedcomRecordType] = None
                    if (
                        fam_link
                        and hasattr(fam_link, "value")
                        and _is_record(fam_link.value)
                    ):
                        fam_record = fam_link.value
                    elif fam_link and hasattr(fam_link, "xref_id"):
                        fam_id = _normalize_id(fam_link.xref_id)
                    if fam_id:
                        fam_record = self.reader.record("FAM", fam_id)  # type: ignore
                    if _is_record(fam_record):
                        role_check_tag = fam_record.sub_tag(role_tag)
                        if (
                            role_check_tag
                            and hasattr(role_check_tag, "xref_id")
                            and _normalize_id(role_check_tag.xref_id) == target_id
                        ):
                            matching_families.append(fam_record)
                        else:
                            logger.debug(
                                f"FAM link {getattr(fam_record, 'xref_id', '?')} found via {link_tag} for {target_id}, but role tag '{role_tag}' didn't match."
                            )
            else:  # Fallback scan
                logger.debug(
                    f"Falling back to full FAM scan for {target_id} in role {role_tag}."
                )
                for family_record in self.reader.records0("FAM"):
                    if not _is_record(family_record):
                        continue
                    role_tag_in_fam = family_record.sub_tag(role_tag)
                    if (
                        role_tag_in_fam
                        and hasattr(role_tag_in_fam, "xref_id")
                        and _normalize_id(role_tag_in_fam.xref_id) == target_id
                    ):
                        matching_families.append(family_record)
        except AttributeError as ae:
            logger.error(
                f"AttributeError finding FAMs for ID {target_id}, role {role_tag}: {ae}",
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                f"Error finding FAMs for ID {target_id}, role {role_tag}: {e}",
                exc_info=True,
            )
        return matching_families

    def _find_family_records_where_individual_is_child(
        self, target_id: str
    ) -> List[GedcomRecordType]:
        return self._find_family_records(target_id, TAG_CHILD)

    def _find_family_records_where_individual_is_parent(
        self, target_id: str
    ) -> List[Tuple[GedcomRecordType, bool, bool]]:
        matching_families_with_role: List[Tuple[GedcomRecordType, bool, bool]] = []
        husband_families = self._find_family_records(target_id, TAG_HUSBAND)
        wife_families = self._find_family_records(target_id, TAG_WIFE)
        for fam in husband_families:
            matching_families_with_role.append((fam, True, False))
        for fam in wife_families:
            if not any(
                existing_fam == fam
                for existing_fam, _, _ in matching_families_with_role
            ):
                matching_families_with_role.append((fam, False, True))
        return matching_families_with_role

    def get_related_individuals(
        self, individual: GedcomIndividualType, relationship_type: str
    ) -> List[GedcomIndividualType]:
        related_individuals: List[GedcomIndividualType] = []
        unique_related_ids: Set[str] = set()
        if not _is_individual(individual) or not hasattr(individual, "xref_id"):
            logger.warning(
                f"get_related_individuals: Invalid input individual object: {type(individual)}"
            )
            return related_individuals
        target_id = _normalize_id(individual.xref_id)
        if not target_id:
            return related_individuals
        if not self.id_to_parents and not self.id_to_children:
            logger.warning(
                "get_related_individuals: Relationship maps empty. Attempting build."
            )
            self._build_family_maps()
        if not self.id_to_parents and not self.id_to_children:
            logger.error(
                "get_related_individuals: Maps still empty after build attempt."
            )
            return related_individuals
        try:
            related_ids: Set[str] = set()
            if relationship_type == "parents":
                related_ids = self.id_to_parents.get(target_id, set())
            elif relationship_type == "children":
                related_ids = self.id_to_children.get(target_id, set())
            elif relationship_type == "siblings":
                parents = self.id_to_parents.get(target_id, set())
                if parents:
                    potential_siblings = set()
                for parent_id in parents:
                    potential_siblings.update(self.id_to_children.get(parent_id, set()))
                    related_ids = potential_siblings - {target_id}
                else:
                    related_ids = set()
            elif relationship_type == "spouses":
                parent_families = self._find_family_records_where_individual_is_parent(
                    target_id
                )
                for fam_record, is_husband, is_wife in parent_families:
                    other_spouse_tag = TAG_WIFE if is_husband else TAG_HUSBAND
                    spouse_ref = fam_record.sub_tag(other_spouse_tag)
                    if spouse_ref and hasattr(spouse_ref, "xref_id"):
                        spouse_id = _normalize_id(spouse_ref.xref_id)
                    if spouse_id:
                        related_ids.add(spouse_id)
            else:
                logger.warning(
                    f"Unknown relationship type requested: '{relationship_type}'"
                )
                return []
            for rel_id in related_ids:
                if rel_id != target_id:
                    indi = self.find_individual_by_id(rel_id)
                if indi:
                    related_individuals.append(indi)
                else:
                    logger.debug(
                        f"Could not find Individual object for related ID: {rel_id}"
                    )
        except Exception as e:
            logger.error(
                f"Error getting {relationship_type} for {target_id}: {e}", exc_info=True
            )
            return []
        related_individuals.sort(
            key=lambda x: (_normalize_id(getattr(x, "xref_id", None)) or "")
        )
        return related_individuals

    def get_relationship_path(self, id1: str, id2: str) -> str:
        id1_norm = _normalize_id(id1)
        id2_norm = _normalize_id(id2)
        if not self.reader:
            return "Error: GEDCOM Reader unavailable."
        if not id1_norm or not id2_norm:
            return "Invalid input IDs."
        if id1_norm == id2_norm:
            return "Individuals are the same."
        if not self.id_to_parents and not self.id_to_children:
            logger.warning("Relationship maps are empty, attempting rebuild.")
            self._build_family_maps()
        if not self.id_to_parents and not self.id_to_children:
            return "Error: Family relationship maps could not be built."
        if not self.indi_index:
            logger.warning("Individual index is empty, attempting rebuild.")
            self._build_indi_index()
        if not self.indi_index:
            return "Error: Individual index could not be built."
        max_depth = 25
        node_limit = 150000
        timeout_sec = 45
        logger.debug(
            f"Calculating relationship path (FastBiBFS): {id1_norm} <-> {id2_norm}"
        )
        search_start = time.time()
        path_ids = fast_bidirectional_bfs(
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
            profile_info = f"[PROFILE] Search: {search_time:.2f}s, MapsBuild: {self.family_maps_build_time:.2f}s, IndexBuild: {self.indi_index_build_time:.2f}s"
            return f"No relationship path found (FastBiBFS could not connect).\n{profile_info}"
        explanation_start = time.time()
        try:
            explanation_str = explain_relationship_path(
                path_ids,
                self.reader,
                self.id_to_parents,
                self.id_to_children,
                self.indi_index,
            )
        except Exception as explain_err:
            logger.error(
                f"Error generating path explanation: {explain_err}", exc_info=True
            )
            explanation_str = "(Error generating explanation)"
        explanation_time = time.time() - explanation_start
        logger.debug(f"[PROFILE] Path explanation built in {explanation_time:.2f}s.")
        total_process_time = search_time + explanation_time
        profile_info = f"[PROFILE] Total Time: {total_process_time:.2f}s (BFS: {search_time:.2f}s, Explain: {explanation_time:.2f}s) [Build Times: Maps={self.family_maps_build_time:.2f}s, Index={self.indi_index_build_time:.2f}s]"
        logger.debug(profile_info)
        return f"{explanation_str}\n{profile_info}"

    def find_potential_matches(
        self,
        first_name: Optional[str],
        surname: Optional[str],
        dob_str: Optional[str] = None,
        pob: Optional[str] = None,
        dod_str: Optional[str] = None,
        pod: Optional[str] = None,
        gender: Optional[str] = None,
        max_results: int = 10,
        scoring_weights: Optional[Dict] = None,
        name_flexibility: Optional[Dict] = None,
        date_flexibility: Optional[Dict] = None,
    ) -> List[Dict]:
        if not self.reader:
            logger.error("find_potential_matches: No reader.")
            return []
        if not self.indi_index:
            logger.error("find_potential_matches: INDI_INDEX not built.")
            return []
        if calculate_match_score is None:
            logger.error("Scoring function not available.")
            return []
        weights = (
            scoring_weights
            if scoring_weights is not None
            else (
                getattr(config_instance, "COMMON_SCORING_WEIGHTS", {})
                if config_instance
                else {}
            )
        )
        name_flex = (
            name_flexibility
            if name_flexibility is not None
            else (
                getattr(config_instance, "NAME_FLEXIBILITY", {})
                if config_instance
                else {}
            )
        )
        date_flex = (
            date_flexibility
            if date_flexibility is not None
            else (
                getattr(config_instance, "DATE_FLEXIBILITY", {})
                if config_instance
                else {}
            )
        )
        if not weights or not name_flex or not date_flex:
            logger.warning(
                "Scoring parameters missing. Using empty defaults, scoring may be ineffective."
            )
            weights = weights or {}
            name_flex = name_flex or {}
            date_flex = date_flex or {}
        year_filter_range = 30
        clean_param = lambda p: p.strip().lower() if p and isinstance(p, str) else ""
        t_fname_lower = clean_param(first_name)
        t_sname_lower = clean_param(surname)
        t_pob_lower = clean_param(pob)
        t_pod_lower = clean_param(pod)
        t_gender_clean = (
            gender.strip().lower()[0]
            if gender
            and isinstance(gender, str)
            and gender.strip().lower() in ("m", "f")
            else None
        )
        t_b_date_obj = _parse_date(dob_str) if dob_str else None
        t_b_year = t_b_date_obj.year if t_b_date_obj else None
        t_d_date_obj = _parse_date(dod_str) if dod_str else None
        t_d_year = t_d_date_obj.year if t_d_date_obj else None
        search_criteria_dict = {
            "first_name": t_fname_lower,
            "surname": t_sname_lower,
            "birth_year": t_b_year,
            "birth_date_obj": t_b_date_obj,
            "birth_place": t_pob_lower,
            "death_year": t_d_year,
            "death_date_obj": t_d_date_obj,
            "death_place": t_pod_lower,
            "gender": t_gender_clean,
        }
        if not any(
            v
            for k, v in search_criteria_dict.items()
            if k not in ["birth_date_obj", "death_date_obj"]
        ):
            logger.warning("Fuzzy search called with no valid text/year criteria.")
            return []
        candidate_count = 0
        scored_results = []
        search_start_time = time.time()
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
                if t_b_year is not None and birth_year_ged is not None:
                    if abs(birth_year_ged - t_b_year) > year_filter_range:
                        continue
                        
                if t_d_year is not None and death_year_ged is not None:
                    if abs(death_year_ged - t_d_year) > year_filter_range:
                        continue
                name_parts = indi_full_name.split()
                c_first_name = name_parts[0] if name_parts else ""
                c_surname = name_parts[-1] if len(name_parts) > 1 else ""
                indi_gender_raw = getattr(indi, TAG_SEX.lower(), None)
                c_gender_clean = (
                    str(indi_gender_raw).strip().lower()[0]
                    if indi_gender_raw
                    and isinstance(indi_gender_raw, str)
                    and str(indi_gender_raw).strip().lower() in ("m", "f")
                    else None
                )
                candidate_data_dict = {
                    "first_name": c_first_name,
                    "surname": c_surname,
                    "birth_year": birth_year_ged,
                    "birth_date_obj": birth_date_obj,
                    "birth_place": (
                        birth_place_str_ged_raw
                        if birth_place_str_ged_raw != "N/A"
                        else None
                    ),
                    "death_year": death_year_ged,
                    "death_date_obj": death_date_obj,
                    "death_place": (
                        death_place_str_ged_raw
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
                            "score": score,
                            "reasons": ", ".join(reasons) if reasons else "Score > 0",
                            "birth_date_obj": birth_date_obj,
                        }
                    )
            except Exception as loop_err:
                logger.error(
                    f"!!! ERROR processing individual {indi_id_raw or indi_id_norm} in find_potential_matches: {loop_err}",
                    exc_info=True,
                )
                continue
        search_duration = time.time() - search_start_time
        logger.debug(
            f"Finished processing {candidate_count} individuals in {search_duration:.2f}s. Found {len(scored_results)} matches with score > 0."
        )
        scored_results.sort(
            key=lambda x: (
                x["score"],
                x.get("birth_date_obj") or datetime.max.replace(tzinfo=timezone.utc),
            ),
            reverse=True,
        )
        limited_results = scored_results[:max_results]
        for res in limited_results:
            res["score"] = round(res["score"])
            res.pop("birth_date_obj", None)
        logger.info(
            f"find_potential_matches returning top {len(limited_results)} of {len(scored_results)} total scored matches."
        )
        return limited_results

# End of GedcomData class


# --- Standalone Test Block ---
if __name__ == "__main__":
    from config import config_instance

    # --- Test Runner Setup ---
    test_results_main: List[Tuple[str, str, str]] = []

    def _run_test_main(
        test_name: str, test_func: Callable, *args, **kwargs
    ) -> Tuple[str, str, str]:
        """Corrected test runner V9: Explicit PASS conditions check."""
        loggr = logger if "logger" in globals() and logger else logging.getLogger()
        loggr.info(f"[ RUNNING ] {test_name}")
        status = "FAIL"
        message = ""
        expect_none = kwargs.pop("expected_none", False)
        try:
            result = test_func(*args, **kwargs)  # Pass cleaned kwargs

            # --- Corrected PASS/FAIL Logic V9 ---
            passed = False  # Assume failure
            if expect_none:
                if result is None:
                    passed = True  # PASS: Expected None and got None
                else:
                    message = f"Expected None, got {type(result).__name__}"  # FAIL: Did not get None
            # If not expecting None, PASS *only* if the result is explicitly True
            elif result is True:
                passed = True  # PASS: Lambda assertion returned True
            # Otherwise (if not expecting None and result is not True), it's a FAIL
            else:
                message = f"Assertion failed or invalid return (returned {result} of type {type(result).__name__})"  # FAIL

            status = "PASS" if passed else "FAIL"

        except Exception as e:
            status = "FAIL"
            message = f"Exception: {type(e).__name__}: {str(e)}"
            loggr.error(f"Exception details for {test_name}: {message}")

        # --- Logging/Return ---
        log_level = logging.INFO if status == "PASS" else logging.ERROR
        log_message = f"[ {status:<6} ] {test_name}{f': {message}' if message and status == 'FAIL' else ''}"
        loggr.log(log_level, log_message)
        test_results_main.append((test_name, status, message))
        return (test_name, status, message)

    # --- End Test Runner Setup ---

    print("\n--- gedcom_utils.py Standalone Test Suite ---")
    overall_status_main = "PASS"

    # === Section 1: Standalone Utility Function Tests ===
    print("\n--- Section 1: Standalone Utility Tests ---")
    _run_test_main(
        "_normalize_id (valid)",
        lambda: _normalize_id("@I123@") == "I123" and _normalize_id("F45") == "F45",
    )
    # This lambda should return True
    _run_test_main(
        "_normalize_id (invalid str)", lambda: _normalize_id("Invalid") is None
    )
    _run_test_main("_normalize_id (empty str)", lambda: _normalize_id("") is None)
    _run_test_main("_normalize_id (None input)", lambda: _normalize_id(None) is None)
    _run_test_main(
        "extract_and_fix_id (valid str)",
        lambda: extract_and_fix_id("@I123@") == "I123"
        and extract_and_fix_id("F45") == "F45",
    )
    # This lambda should return True
    _run_test_main(
        "extract_and_fix_id (invalid str)",
        lambda: extract_and_fix_id("Invalid") is None,
    )
    _run_test_main(
        "extract_and_fix_id (invalid type)", lambda: extract_and_fix_id(123) is None
    )
    _run_test_main(
        "extract_and_fix_id (None input)", lambda: extract_and_fix_id(None) is None
    )
    mock_record_simple = type("MockRecord", (object,), {"xref_id": "@I999@"})()
    # This lambda just calls the function, runner checks expect_none
    _run_test_main(
        "extract_and_fix_id (mock obj)",
        lambda: extract_and_fix_id(mock_record_simple),
        expected_none=True,
    )
    _run_test_main("_parse_date (YYYY)", lambda: _parse_date("1980").year == 1980)
    _run_test_main(
        "_parse_date (Mon YYYY)",
        lambda: _parse_date("Jan 1995").month == 1
        and _parse_date("January 1995").month == 1,
    )
    _run_test_main(
        "_parse_date (DD Mon YYYY)",
        lambda: _parse_date("15 Feb 2001").day == 15
        and _parse_date("15 February 2001").day == 15,
    )
    _run_test_main(
        "_parse_date (with prefix)",
        lambda: _parse_date("ABT 1950").year == 1950
        and _parse_date("BEF 20 MAR 1960").day == 20,
    )
    _run_test_main(
        "_parse_date (range)", lambda: _parse_date("BET 1910 AND 1912").year == 1910
    )
    # This lambda just calls the function, runner checks expect_none
    _run_test_main(
        "_parse_date (invalid)", lambda: _parse_date("Invalid Date"), expected_none=True
    )
    _run_test_main(
        "_clean_display_date (basic)",
        lambda: _clean_display_date("1 JAN 1900") == "1 JAN 1900",
    )
    _run_test_main(
        "_clean_display_date (prefix)",
        lambda: _clean_display_date("ABT 1950") == "~1950",
    )
    _run_test_main(
        "_clean_display_date (brackets)",
        lambda: _clean_display_date("(1920)") == "1920",
    )
    _run_test_main(
        "_clean_display_date (empty brackets)",
        lambda: _clean_display_date("()") == "N/A",
    )
    _run_test_main(
        "_clean_display_date (range)",
        lambda: _clean_display_date("BET 1910 AND 1912") == "1910-1912",
    )
    _run_test_main(
        "calculate_match_score (basic)",
        lambda: isinstance(
            calculate_match_score({"first_name": "a"}, {"first_name": "b"}), tuple
        ),
    )

    # === Section 2: GedcomData Functional Tests ===
    print(
        "\n--- Section 2: GedcomData Functional Tests (requires config & GEDCOM file) ---"
    )
    gedcom_data: Optional[GedcomData] = None
    gedcom_load_status = "SKIPPED"
    gedcom_load_message = "Prerequisites not met (config or ged4py)"  # Default message
    can_load_gedcom = False  # Start as False
    gedcom_path: Optional[Path] = None  # Initialize path

    if (
        config_instance
        and hasattr(config_instance, "GEDCOM_FILE_PATH")
        and config_instance.GEDCOM_FILE_PATH
    ):
        # Config path exists, now check if it's a valid file
        potential_path = Path(config_instance.GEDCOM_FILE_PATH)
        if potential_path.is_file():
            gedcom_path = potential_path  # Store the valid Path object
            can_load_gedcom = True  # <<< --- THE FIX IS HERE --- >>>
            gedcom_load_message = f"Configured path found: {gedcom_path.name}"
        else:
            # Path was configured but doesn't point to a file
            gedcom_load_message = f"GEDCOM_FILE_PATH '{config_instance.GEDCOM_FILE_PATH}' is not a valid file."
            can_load_gedcom = False  # Ensure it stays False
    else:
        # Config path was missing or empty
        gedcom_load_message = "GEDCOM_FILE_PATH not configured or empty in config."
        can_load_gedcom = False  # Ensure it stays False

    # Now, the instantiation test attempt:
    if can_load_gedcom and gedcom_path:  # Check both flag and valid path
        test_name = "GedcomData Instantiation"
        logger.info(f"[ RUNNING ] {test_name}")
        try:
            # Make sure to use the validated gedcom_path
            gedcom_data = GedcomData(gedcom_path)
            gedcom_load_status = "PASS"
            gedcom_load_message = f"Loaded {gedcom_path.name}"
            logger.info(f"[ PASS    ] {test_name}: {gedcom_load_message}")
        except Exception as e:
            gedcom_load_status = "FAIL"
            # Use traceback for better error reporting during tests
            gedcom_load_message = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"[ FAIL    ] {test_name}: {gedcom_load_message}")
        test_results_main.append((test_name, gedcom_load_status, gedcom_load_message))
    else:
        # This block now correctly executes only if can_load_gedcom is False
        test_results_main.append(
            ("GedcomData Instantiation", "SKIPPED", gedcom_load_message)
        )

    # --- The rest of the Section 2 tests follow ---
    # (No changes needed below this point for this specific fix)

    print(
        "\n>>> IMPORTANT: Functional tests below require modifying placeholder IDs/Names <<<"
    )
    # ... (rest of the test script remains the same) ...
    TEST_INDI_ID_1 = "I102281560836"
    TEST_INDI_ID_2 = "I102281560744"
    TEST_SEARCH_NAME = "Wayne"
    TEST_SEARCH_SURNAME = "Gault"
    print(f">>> Using Test IDs: {TEST_INDI_ID_1}, {TEST_INDI_ID_2}")
    print(f">>> Using Test Name: {TEST_SEARCH_NAME} {TEST_SEARCH_SURNAME}\n")

    functional_skip_reason = (
        "GedcomData failed to load" if gedcom_load_status != "PASS" else ""
    )

    test_name_find = f"find_individual_by_id({TEST_INDI_ID_1})"
    if gedcom_data:
        _run_test_main(
            test_name_find,
            lambda: gedcom_data.find_individual_by_id(TEST_INDI_ID_1) is not None,
        )
    else:
        test_results_main.append((test_name_find, "SKIPPED", functional_skip_reason))

    test_name_rel = f"get_related_individuals({TEST_INDI_ID_1}, 'parents')"
    if gedcom_data:
        indi1_obj = gedcom_data.find_individual_by_id(TEST_INDI_ID_1)
        if indi1_obj:
            _run_test_main(
                test_name_rel,
                lambda: isinstance(
                    gedcom_data.get_related_individuals(indi1_obj, "parents"), list
                ),
            )
        else:
            test_results_main.append(
                (
                    test_name_rel,
                    "SKIPPED",
                    f"Prerequisite failed: Could not find {TEST_INDI_ID_1}",
                )
            )
    else:
        test_results_main.append((test_name_rel, "SKIPPED", functional_skip_reason))

    test_name_path = f"get_relationship_path({TEST_INDI_ID_1}, {TEST_INDI_ID_2})"
    if gedcom_data:
        _run_test_main(
            test_name_path,
            lambda: "Error:"
            not in gedcom_data.get_relationship_path(TEST_INDI_ID_1, TEST_INDI_ID_2),
        )
    else:
        test_results_main.append((test_name_path, "SKIPPED", functional_skip_reason))

    test_name_match = (
        f"find_potential_matches('{TEST_SEARCH_NAME}', '{TEST_SEARCH_SURNAME}')"
    )
    if gedcom_data:
        weights = getattr(config_instance, "COMMON_SCORING_WEIGHTS", None)
        name_flex = getattr(config_instance, "NAME_FLEXIBILITY", None)
        date_flex = getattr(config_instance, "DATE_FLEXIBILITY", None)
        if weights and name_flex and date_flex:
            _run_test_main(
                test_name_match,
                lambda: isinstance(
                    gedcom_data.find_potential_matches(
                        first_name=TEST_SEARCH_NAME,
                        surname=TEST_SEARCH_SURNAME,
                        scoring_weights=weights,
                        name_flexibility=name_flex,
                        date_flexibility=date_flex,
                    ),
                    list,
                ),
            )
        else:
            test_results_main.append(
                (test_name_match, "SKIPPED", "Scoring config missing")
            )
    else:
        test_results_main.append((test_name_match, "SKIPPED", functional_skip_reason))

    # --- Final Summary ---
    print("\n--- Test Summary ---")
    name_width = (
        max(len(name) for name, _, _ in test_results_main) if test_results_main else 45
    )
    name_width = max(name_width, 50)
    status_width = 8
    header = f"{'Test Name':<{name_width}} | {'Status':<{status_width}} | {'Message'}"
    print(header)
    print("-" * (name_width + status_width + 12))
    final_fail_count = 0
    final_skip_count = 0
    for name, status, message in test_results_main:
        if status == "FAIL":
            final_fail_count += 1
            overall_status_main = "FAIL"
        elif status == "SKIPPED":
            final_skip_count += 1
        print(
            f"{name:<{name_width}} | {status:<{status_width}} | {message if status != 'PASS' else ''}"
        )
    print("-" * (len(header)))
    total_tests = len(test_results_main)
    passed_tests = total_tests - final_fail_count - final_skip_count
    summary_line = f"Result: {overall_status_main} ({passed_tests} passed, {final_fail_count} failed, {final_skip_count} skipped out of {total_tests} tests)"
    if overall_status_main == "PASS":
        print(summary_line)
        print("--- gedcom_utils.py standalone test run PASSED ---")
    else:
        print(summary_line)
        print("--- gedcom_utils.py standalone test run FAILED ---")
    sys.exit(0 if overall_status_main == "PASS" else 1)
# End of gedcom_utils.py __main__ block
