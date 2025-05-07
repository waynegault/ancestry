# --- START OF FILE gedcom_utils.py ---

# gedcom_utils.py
"""
Utility functions and class for loading, parsing, caching, and querying
GEDCOM data using ged4py. Includes relationship mapping, path calculation,
and fuzzy matching/scoring. Pre-processes individual data for faster access.
V.20240502.FinalCode.SyntaxFix
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
try:
    from ged4py.parser import GedcomReader
    from ged4py.model import Individual, Record, Name, NameRec
except ImportError:
    GedcomReader = type(None)
    Individual = type(None)
    Record = type(None)
    Name = type(None)
    NameRec = type(None)  # type: ignore
    print(
        "ERROR: ged4py library not found. This script requires ged4py (`pip install ged4py`)"
    )

try:
    import dateparser

    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False
    print(
        "WARNING: dateparser library not found. Date parsing will be limited. Run 'pip install dateparser'"
    )


# --- Local application imports ---
try:
    from utils import format_name, ordinal_case  # Assumed available
    from config import config_instance  # Assumed available
except ImportError:
    print(
        "WARNING: Could not import from utils or config. Using dummy values for standalone execution/testing."
    )

    class DummyConfig:  # Define dummy config with necessary attributes
        COMMON_SCORING_WEIGHTS = {
            "year_birth": 20,
            "contains_first_name": 25,
            "contains_surname": 25,
            "bonus_both_names_contain": 25,
            "exact_birth_date": 25,
            "approx_year_birth": 10,
            "exact_death_date": 25,
            "year_death": 20,
            "approx_year_death": 10,
            "death_dates_both_absent": 15,
            "gender_match": 25,
            "contains_pob": 15,
            "contains_pod": 15,
        }
        DATE_FLEXIBILITY = {"year_match_range": 10}
        NAME_FLEXIBILITY = {}
        GEDCOM_FILE_PATH = None  # Add this attribute to avoid the error

    config_instance = DummyConfig()

    def format_name(name):
        return str(name).title() if name else "Unknown"

    def ordinal_case(text):
        return str(text)


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
TAG_GIVN = "GIVN"
TAG_SURN = "SURN"


# --- Logging Setup ---
logger = logging.getLogger("gedcom_utils")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


# ==============================================
# Utility Functions (Moved to Top)
# ==============================================
def _is_individual(obj: Any) -> bool:
    return obj is not None and (
        isinstance(obj, GedcomIndividualType)
        if GedcomIndividualType is not type(None)
        else False
    )


def _is_record(obj: Any) -> bool:
    return obj is not None and (
        isinstance(obj, GedcomRecordType)
        if GedcomRecordType is not type(None)
        else False
    )


def _is_name_rec(obj: Any) -> bool:
    return obj is not None and (
        isinstance(obj, GedcomNameRecType)
        if GedcomNameRecType is not type(None)
        else False
    )


def _normalize_id(xref_id: Optional[str]) -> Optional[str]:
    if not xref_id or not isinstance(xref_id, str):
        return None
    match = re.match(r"^@?([IFSNMCXO][0-9\-]+)@?$", xref_id.strip().upper())
    if match:
        return match.group(1)
    search_match = re.search(r"([IFSNMCXO][0-9\-]+)", xref_id.strip().upper())
    if search_match:
        logger.debug(
            f"Normalized ID '{search_match.group(1)}' using fallback regex from '{xref_id}'."
        )
        return search_match.group(1)
    logger.warning(f"Could not normalize potential ID: '{xref_id}'")
    return None


def extract_and_fix_id(raw_id: Any) -> Optional[str]:
    if not raw_id:
        return None
    id_to_normalize: Optional[str] = None
    if isinstance(raw_id, str):
        id_to_normalize = raw_id
    elif hasattr(raw_id, "xref_id") and (_is_record(raw_id) or _is_individual(raw_id)):
        id_to_normalize = getattr(raw_id, "xref_id", None)
    else:
        logger.debug(
            f"extract_and_fix_id: Invalid input type '{type(raw_id).__name__}'."
        )
        return None
    return _normalize_id(id_to_normalize)


def _get_full_name(indi: GedcomIndividualType) -> str:
    """Safely gets formatted name, checking for .format method. V3"""
    if not _is_individual(indi):
        if hasattr(indi, "value") and _is_individual(getattr(indi, "value", None)):
            indi = indi.value
        else:
            logger.warning(
                f"_get_full_name called with non-Individual type: {type(indi)}"
            )
            return "Unknown (Invalid Type)"

    indi_id_log = extract_and_fix_id(indi) or "Unknown ID"
    formatted_name = None
    name_source = "Unknown"

    try:
        # --- Attempt 1: Use indi.name if it has .format ---
        if hasattr(indi, "name"):
            name_rec = indi.name
            if name_rec and hasattr(name_rec, "format") and callable(name_rec.format):
                try:
                    formatted_name = name_rec.format()
                    name_source = "indi.name.format()"
                    logger.debug(
                        f"Name for {indi_id_log} from {name_source}: '{formatted_name}'"
                    )
                except Exception as fmt_err:
                    logger.warning(
                        f"Error calling indi.name.format() for {indi_id_log}: {fmt_err}"
                    )
                    formatted_name = None  # Reset on error

        # --- Attempt 2: Use indi.sub_tag(TAG_NAME) if Attempt 1 failed and it has .format ---
        if formatted_name is None:
            name_tag = indi.sub_tag(TAG_NAME)
            if name_tag and hasattr(name_tag, "format") and callable(name_tag.format):
                try:
                    formatted_name = name_tag.format()
                    name_source = "indi.sub_tag(TAG_NAME).format()"
                    logger.debug(
                        f"Name for {indi_id_log} from {name_source}: '{formatted_name}'"
                    )
                except Exception as fmt_err:
                    logger.warning(
                        f"Error calling indi.sub_tag(TAG_NAME).format() for {indi_id_log}: {fmt_err}"
                    )
                    formatted_name = None

        # --- Attempt 3: Manually combine GIVN and SURN if formatting failed ---
        if formatted_name is None:
            name_tag = indi.sub_tag(
                TAG_NAME
            )  # Get tag again or reuse from above if needed
            if name_tag:  # Check if NAME tag exists
                givn = name_tag.sub_tag_value(TAG_GIVN)
                surn = name_tag.sub_tag_value(TAG_SURN)
                # Combine, prioritizing surname placement
                if givn and surn:
                    formatted_name = f"{givn} {surn}"
                elif givn:
                    formatted_name = givn
                elif surn:
                    formatted_name = surn  # Or potentially format as /SURN/?
                name_source = "manual GIVN/SURN combination"
                logger.debug(
                    f"Name for {indi_id_log} from {name_source}: '{formatted_name}'"
                )

        # --- Attempt 4: Use indi.sub_tag_value(TAG_NAME) as last resort ---
        if formatted_name is None:
            name_val = indi.sub_tag_value(TAG_NAME)
            if isinstance(name_val, str) and name_val.strip() and name_val != "/":
                formatted_name = name_val
                name_source = "indi.sub_tag_value(TAG_NAME)"
                logger.debug(
                    f"Name for {indi_id_log} from {name_source}: '{formatted_name}'"
                )

        # --- Final Cleaning and Return ---
        if formatted_name:
            # Apply utils.format_name for styling
            cleaned_name = format_name(formatted_name)
            return (
                cleaned_name
                if cleaned_name and cleaned_name != "Unknown"
                else f"Unknown ({name_source} Error)"
            )
        else:
            return "Unknown (No Name Found)"

    except Exception as e:
        logger.error(
            f"Unexpected error in _get_full_name for @{indi_id_log}@: {e}",
            exc_info=True,
        )
        return "Unknown (Error)"


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parses various GEDCOM date formats into timezone-aware datetime objects (UTC),
    prioritizing full date parsing but falling back to extracting the first year.
    V13 - Corrected range splitting regex.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    original_date_str = date_str
    logger.debug(f"Attempting to parse date: '{original_date_str}'")
    if date_str.startswith("(") and date_str.endswith(")"):
        date_str = date_str[1:-1].strip()
    if not date_str:
        logger.debug("Date string empty after removing parentheses.")
        return None
    date_str = date_str.strip().upper()
    if re.match(r"^(UNKNOWN|\?UNKNOWN|\?|DECEASED|IN INFANCY|0)$", date_str):
        logger.debug(
            f"Identified non-parseable string: '{original_date_str}' -> '{date_str}'"
        )
        return None
    if re.fullmatch(r"^\d{1,2}\s+[A-Z]{3,}$", date_str) or re.fullmatch(
        r"^[A-Z]{3,}$", date_str
    ):
        logger.debug(
            f"Ignoring date string without year: '{original_date_str}' -> '{date_str}'"
        )
        return None
    keywords_to_remove = r"\b(?:MAYBE|PRIOR|CALCULATED|AROUND|BAPTISED|WFT|BTWN|BFR|SP|QTR\.?\d?|CIRCA|ABOUT:|AFTER|BEFORE)\b\.?\s*|\b(?:AGE:?\s*\d+)\b|\b(?:WIFE\s+OF.*)\b|\b(?:HUSBAND\s+OF.*)\b"
    cleaned_str = date_str
    previous_len = -1
    while len(cleaned_str) != previous_len:
        previous_len = len(cleaned_str)
        cleaned_str = re.sub(
            keywords_to_remove, "", cleaned_str, flags=re.IGNORECASE
        ).strip()
    cleaned_str = re.sub(r"\s+SP$", "", cleaned_str).strip()
    cleaned_str = re.split(r"\s+(?:AND|OR|TO)\s+", cleaned_str, maxsplit=1)[
        0
    ].strip()  # CORRECTED SPLIT
    year_range_match = re.match(r"^(\d{4})\s*[-–]\s*\d{4}$", cleaned_str)
    if year_range_match:
        cleaned_str = year_range_match.group(1)
        logger.debug(f"Treated as year range, using first year: '{cleaned_str}'")
    prefixes = r"^(?:ABT|EST|CAL|INT|BEF|AFT|BET|FROM)\.?\s+"
    cleaned_str = re.sub(prefixes, "", cleaned_str, count=1).strip()
    cleaned_str = re.sub(r"(\d+)(?:ST|ND|RD|TH)", r"\1", cleaned_str).strip()
    cleaned_str = re.sub(r"\s+(?:BC|AD)$", "", cleaned_str).strip()
    if re.match(r"^0{3,4}(?:[-/\s]\d{1,2}[-/\s]\d{1,2})?$", cleaned_str):
        logger.debug(
            f"Treating year 0000 pattern as invalid: '{original_date_str}' -> '{cleaned_str}'"
        )
        return None
    cleaned_str = re.sub(r"[,;:]", " ", cleaned_str)
    cleaned_str = re.sub(r"([A-Z]{3})\.", r"\1", cleaned_str)
    cleaned_str = re.sub(r"([A-Z])(\d)", r"\1 \2", cleaned_str)
    cleaned_str = re.sub(r"(\d)([A-Z])", r"\1 \2", cleaned_str)
    cleaned_str = re.sub(r"\s+", " ", cleaned_str).strip()
    if not cleaned_str:
        logger.debug(f"Date string empty after cleaning: '{original_date_str}'")
        return None
    logger.debug(f"Cleaned date string for parsing: '{cleaned_str}'")
    parsed_dt = None
    if DATEPARSER_AVAILABLE:
        try:
            settings = {"PREFER_DAY_OF_MONTH": "first", "REQUIRE_PARTS": ["year"]}
            parsed_dt = dateparser.parse(cleaned_str, settings=settings)
            if parsed_dt:
                logger.debug(f"dateparser succeeded for '{cleaned_str}'")
            else:
                logger.debug(
                    f"dateparser returned None for '{cleaned_str}', trying strptime..."
                )
        except Exception as e:
            logger.error(
                f"Error using dateparser for '{original_date_str}' (cleaned: '{cleaned_str}'): {e}",
                exc_info=False,
            )
    if not parsed_dt:
        formats = [
            "%d %b %Y",
            "%d %B %Y",
            "%b %Y",
            "%B %Y",
            "%Y",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y/%m/%d",
            "%d-%b-%Y",
            "%d-%m-%Y",
            "%Y-%m-%d",
            "%B %d %Y",
        ]
        for fmt in formats:
            try:
                if fmt == "%Y" and not re.fullmatch(r"\d{3,4}", cleaned_str):
                    continue
                dt_naive = datetime.strptime(cleaned_str, fmt)
                logger.debug(f"Parsed '{cleaned_str}' using strptime format '{fmt}'")
                parsed_dt = dt_naive
                break
            except ValueError:
                continue
            except Exception as e:
                logger.debug(f"Strptime error for format '{fmt}': {e}")
                continue
    if not parsed_dt:
        logger.debug(
            f"Full parsing failed for '{cleaned_str}', attempting year extraction."
        )
        year_match = re.search(r"\b(\d{3,4})\b", cleaned_str)
        if year_match:
            year_str = year_match.group(1)
            try:  # <<< START OF TRY BLOCK FOR YEAR EXTRACTION >>>
                year = int(year_str)
                if 500 <= year <= datetime.now().year + 5:
                    logger.debug(f"Extracted year {year} as fallback.")
                    parsed_dt = datetime(year, 1, 1)
                else:
                    logger.debug(f"Extracted year {year} out of plausible range.")
            except ValueError:  # <<< CORRECTED except BLOCK >>>
                logger.debug(f"Could not convert extracted year '{year_str}' to int.")
            # <<< END except BLOCK >>>
    if isinstance(parsed_dt, datetime):
        if parsed_dt.year == 0:
            logger.warning(
                f"Parsed date resulted in year 0, treating as invalid: '{original_date_str}' -> {parsed_dt}"
            )
            return None
        if parsed_dt.tzinfo is None:
            return parsed_dt.replace(tzinfo=timezone.utc)
        else:
            return parsed_dt.astimezone(timezone.utc)
    else:
        logger.warning(
            f"All parsing attempts failed for: '{original_date_str}' -> cleaned: '{cleaned_str}'"
        )
        return None


def _clean_display_date(raw_date_str: Optional[str]) -> str:  # ... implementation ...
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
) -> Tuple[Optional[datetime], str, str]:  # ... implementation ...
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


def format_life_dates(indi: GedcomIndividualType) -> str:  # ... implementation ...
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


def format_full_life_details(
    indi: GedcomIndividualType,
) -> Tuple[str, str]:  # ... implementation ...
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


def format_relative_info(relative: Any) -> str:  # ... implementation ...
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
    visited_fwd: Dict[str, Optional[str]],  # {node: predecessor_from_start}
    visited_bwd: Dict[str, Optional[str]],  # {node: predecessor_from_end}
) -> List[str]:
    """Helper function for BFS to reconstruct the path from visited dictionaries. V2"""
    path: List[str] = []
    # Trace back from meeting point to start_id
    curr = meeting_id
    while curr is not None:
        path.append(curr)
        curr = visited_fwd.get(curr)
    path.reverse()  # Now path is start_id -> ... -> meeting_id

    # Trace back from meeting point to end_id (but skip meeting_id itself)
    curr = visited_bwd.get(
        meeting_id
    )  # Start from predecessor of meeting_id in backward search
    path_end: List[str] = []
    while curr is not None:
        path_end.append(curr)
        curr = visited_bwd.get(curr)
    # path_end is now [predecessor_of_meeting, ..., end_id] - needs reversing
    path_end.reverse()

    # Combine the two parts
    full_path = path + path_end

    # Basic validation
    if not full_path or full_path[0] != start_id or full_path[-1] != end_id:
        logger.error(
            f"Path reconstruction failed or invalid! Start:{start_id}, End:{end_id}, Meet:{meeting_id}, Result:{full_path}"
        )
        # Attempt manual reconstruction if possible (simple cases)
        if meeting_id == end_id and path and path[0] == start_id:
            return path  # FWD search found END directly
        if meeting_id == start_id and path_end and path_end[-1] == end_id:
            return [start_id] + path_end  # BWD search found START directly
        return []  # Return empty if reconstruction is clearly wrong

    logger.debug(f"_reconstruct_path: Final reconstructed path IDs: {full_path}")
    return full_path


def fast_bidirectional_bfs(
    start_id: str,
    end_id: str,
    id_to_parents: Dict[str, Set[str]],
    id_to_children: Dict[str, Set[str]],
    max_depth=25,
    node_limit=150000,
    timeout_sec=45,
    log_progress=False,
) -> List[str]:  # ... implementation ...
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
    """Generates a human-readable explanation of the relationship path. V4"""
    if not path_ids or len(path_ids) < 2:
        return "(No relationship path explanation available)"
    if id_to_parents is None or id_to_children is None or indi_index is None:
        return "(Error: Data maps or index unavailable)"

    steps: List[str] = []
    start_person_indi = indi_index.get(path_ids[0])
    start_person_name = (
        _get_full_name(start_person_indi)
        if start_person_indi
        else f"Unknown ({path_ids[0]})"
    )
    # Start with the first person's name - no arrow needed yet.
    # steps.append(start_person_name) # Removed - first name added before loop

    current_person_name = start_person_name

    for i in range(len(path_ids) - 1):
        id_a, id_b = path_ids[i], path_ids[i + 1]
        indi_a = indi_index.get(id_a)  # Person A object (previous step)
        indi_b = indi_index.get(id_b)  # Person B object (current step in explanation)
        name_b = _get_full_name(indi_b) if indi_b else f"Unknown ({id_b})"

        relationship_phrase = None  # How B relates to A

        # Determine gender of person B for labels like son/daughter etc.
        sex_b = getattr(indi_b, TAG_SEX.lower(), None) if indi_b else None
        sex_b_char = (
            str(sex_b).upper()[0]
            if sex_b and isinstance(sex_b, str) and str(sex_b).upper() in ("M", "F")
            else None
        )

        # Check 1: Is B a PARENT of A?
        if id_b in id_to_parents.get(id_a, set()):
            parent_label = (
                "father"
                if sex_b_char == "M"
                else "mother" if sex_b_char == "F" else "parent"
            )
            relationship_phrase = f"whose {parent_label} is {name_b}"

        # Check 2: Is B a CHILD of A?
        elif id_b in id_to_children.get(id_a, set()):
            child_label = (
                "son"
                if sex_b_char == "M"
                else "daughter" if sex_b_char == "F" else "child"
            )
            relationship_phrase = f"whose {child_label} is {name_b}"

        # Check 3: Is B a SIBLING of A? (Share at least one parent)
        else:
            parents_a = id_to_parents.get(id_a, set())
            parents_b = id_to_parents.get(id_b, set())
            if parents_a and parents_b and not parents_a.isdisjoint(parents_b):
                sibling_label = (
                    "brother"
                    if sex_b_char == "M"
                    else "sister" if sex_b_char == "F" else "sibling"
                )
                relationship_phrase = f"whose {sibling_label} is {name_b}"

        # Check 4: Spouses? (Requires specific FAM lookup or pre-built map - add basic check)
        if (
            relationship_phrase is None and indi_a
        ):  # Only check spouses if not parent/child/sibling
            spouse_found = False
            # Check families where A is a parent
            parent_families = gedcom_data._find_family_records_where_individual_is_parent(
                id_a
            )  # Use internal method if GedcomData object is accessible, otherwise need reader
            for fam_rec, is_husband, is_wife in parent_families:
                other_spouse_tag = TAG_WIFE if is_husband else TAG_HUSBAND
                spouse_ref = fam_rec.sub_tag(other_spouse_tag)
                if spouse_ref and hasattr(spouse_ref, "xref_id"):
                    spouse_id = _normalize_id(spouse_ref.xref_id)
                    if spouse_id == id_b:  # Found B as a spouse of A
                        spouse_label = (
                            "husband"
                            if sex_b_char == "M"
                            else "wife" if sex_b_char == "F" else "spouse"
                        )
                        relationship_phrase = f"whose {spouse_label} is {name_b}"
                        spouse_found = True
                        break
            if not spouse_found:  # Fallback if spouse check needed but failed
                logger.warning(
                    f"Could not determine direct relation (incl. spouse) between {id_a} and {id_b}."
                )
                relationship_phrase = f"connected to {name_b}"
        elif (
            relationship_phrase is None
        ):  # Fallback if indi_a was None or other checks failed
            logger.warning(
                f"Could not determine direct relation between {id_a} and {id_b}."
            )
            relationship_phrase = f"related to {name_b}"

        steps.append(f"  -> {relationship_phrase}")
        # Update current person for the next iteration's phrasing (optional but good practice)
        # current_person_name = name_b

    # Join the start name and all the steps
    explanation_str = start_person_name + "\n" + "\n".join(steps)
    return explanation_str


# ==============================================
# Scoring Function (V18 - Corrected Syntax)
# ==============================================
def calculate_match_score(
    search_criteria: Dict,
    candidate_processed_data: Dict[str, Any],  # Expects pre-processed data
    scoring_weights: Optional[Dict[str, int]] = None,
    name_flexibility: Optional[Dict] = None,
    date_flexibility: Optional[Dict] = None,
) -> Tuple[float, Dict[str, int], List[str]]:
    """
    Calculates match score using pre-processed candidate data.
    Handles OR logic for death place matching (contains OR both absent).
    Prioritizes exact date > exact year > approx year for date scoring.
    V18.PreProcess compatible - Syntax Fixed.
    """
    match_reasons: List[str] = []
    field_scores = {
        "givn": 0,
        "surn": 0,
        "gender": 0,
        "byear": 0,
        "bdate": 0,
        "bplace": 0,
        "bbonus": 0,  # Birth bonus
        "dyear": 0,
        "ddate": 0,
        "dplace": 0,
        "dbonus": 0,  # Death bonus
        "bonus": 0,  # Name bonus
    }
    weights = (
        scoring_weights
        if scoring_weights is not None
        else getattr(config_instance, "COMMON_SCORING_WEIGHTS", {})
    )
    date_flex = (
        date_flexibility
        if date_flexibility is not None
        else getattr(config_instance, "DATE_FLEXIBILITY", {})
    )
    year_score_range = date_flex.get("year_match_range", 10)

    # Prepare Target Data
    t_fname_raw = search_criteria.get("first_name")
    t_fname = t_fname_raw.lower() if isinstance(t_fname_raw, str) else ""
    t_sname_raw = search_criteria.get("surname")
    t_sname = t_sname_raw.lower() if isinstance(t_sname_raw, str) else ""
    t_pob_raw = search_criteria.get("birth_place")
    t_pob = t_pob_raw.lower() if isinstance(t_pob_raw, str) else ""
    t_pod_raw = search_criteria.get("death_place")
    t_pod = t_pod_raw.lower() if isinstance(t_pod_raw, str) else ""
    t_b_year = search_criteria.get("birth_year")
    t_b_date = search_criteria.get("birth_date_obj")
    t_d_year = search_criteria.get("death_year")
    t_d_date = search_criteria.get("death_date_obj")
    t_gender = search_criteria.get("gender")

    # Get Candidate Data from Pre-processed dict
    c_id_debug = candidate_processed_data.get("norm_id", "N/A_in_proc_cache")
    c_fname_raw = candidate_processed_data.get("first_name", "")
    c_fname = c_fname_raw.lower() if isinstance(c_fname_raw, str) else ""
    c_sname_raw = candidate_processed_data.get("surname", "")
    c_sname = c_sname_raw.lower() if isinstance(c_sname_raw, str) else ""
    c_bplace_raw = candidate_processed_data.get("birth_place_disp")
    c_bplace = c_bplace_raw.lower() if isinstance(c_bplace_raw, str) else ""
    c_dplace_raw = candidate_processed_data.get("death_place_disp")
    c_dplace = c_dplace_raw.lower() if isinstance(c_dplace_raw, str) else ""
    c_b_year = candidate_processed_data.get("birth_year")
    c_b_date = candidate_processed_data.get("birth_date_obj")
    c_d_year = candidate_processed_data.get("death_year")
    c_d_date = candidate_processed_data.get("death_date_obj")
    c_gender = candidate_processed_data.get("gender_norm")

    # Name Scoring
    first_name_matched = False
    surname_matched = False
    if t_fname and c_fname and t_fname in c_fname:
        points_givn = weights.get("contains_first_name", 0)
        if points_givn != 0:
            field_scores["givn"] = points_givn
            match_reasons.append(f"Contains First Name ({points_givn}pts)")
            first_name_matched = True
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Matched contains_first_name. Set field_scores['givn'] = {points_givn}"
            )
    if t_sname and c_sname and t_sname in c_sname:
        points_surn = weights.get("contains_surname", 0)
        if points_surn != 0:
            field_scores["surn"] = points_surn
            match_reasons.append(f"Contains Surname ({points_surn}pts)")
            surname_matched = True
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Matched contains_surname. Set field_scores['surn'] = {points_surn}"
            )
    if t_fname and t_sname and first_name_matched and surname_matched:
        bonus_points = weights.get("bonus_both_names_contain", 0)
        if bonus_points != 0:
            field_scores["bonus"] = bonus_points
            match_reasons.append(f"Bonus Both Names ({bonus_points}pts)")
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Applied bonus_both_names_contain. Set field_scores['bonus'] = {bonus_points}"
            )

    # Date Flag Calculation (Restored except clauses)
    exact_birth_date_match = bool(
        t_b_date
        and c_b_date
        and isinstance(t_b_date, datetime)
        and isinstance(c_b_date, datetime)
        and t_b_date.date() == c_b_date.date()
    )
    exact_death_date_match = bool(
        t_d_date
        and c_d_date
        and isinstance(t_d_date, datetime)
        and isinstance(c_d_date, datetime)
        and t_d_date.date() == c_d_date.date()
    )
    birth_year_match = False
    birth_year_approx_match = False
    if t_b_year is not None and c_b_year is not None:
        try:
            t_b_year_int = int(t_b_year)
            c_b_year_int = int(c_b_year)
            birth_year_match = t_b_year_int == c_b_year_int
            if not birth_year_match:
                birth_year_approx_match = (
                    abs(c_b_year_int - t_b_year_int) <= year_score_range
                )
        except:  # <<< Restored missing except
            pass  # Keep flags False if error occurs

    death_year_match = False
    death_year_approx_match = False
    if t_d_year is not None and c_d_year is not None:
        try:
            t_d_year_int = int(t_d_year)
            c_d_year_int = int(c_d_year)
            death_year_match = t_d_year_int == c_d_year_int
            if not death_year_match:
                death_year_approx_match = (
                    abs(c_d_year_int - t_d_year_int) <= year_score_range
                )
        except:  # <<< Restored missing except
            pass  # Keep flags False

    death_dates_absent = bool(
        t_d_date is None and c_d_date is None and t_d_year is None and c_d_year is None
    )

    # Date Scoring
    birth_score_added = False
    if exact_birth_date_match:
        points_bdate = weights.get("exact_birth_date", 0)
        if points_bdate != 0:
            field_scores["bdate"] = points_bdate
            match_reasons.append(f"Exact Birth Date ({points_bdate}pts)")
            birth_score_added = True
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Matched exact_birth_date. Set field_scores['bdate'] = {points_bdate}"
            )
    if not birth_score_added and birth_year_match:
        points_byear = weights.get("year_birth", 0)
        if points_byear != 0:
            field_scores["byear"] = points_byear
            match_reasons.append(f"Exact Birth Year ({c_b_year}) ({points_byear}pts)")
            birth_score_added = True
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Matched year_birth. Set field_scores['byear'] = {points_byear}"
            )
    if not birth_score_added and birth_year_approx_match:
        points_byear_approx = weights.get("approx_year_birth", 0)
        if points_byear_approx != 0:
            field_scores["byear"] = points_byear_approx
            match_reasons.append(
                f"Approx Birth Year ({c_b_year} vs {t_b_year}) ({points_byear_approx}pts)"
            )
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Matched approx_year_birth. Set field_scores['byear'] = {points_byear_approx}"
            )

    death_score_added = False
    if exact_death_date_match:
        points_ddate = weights.get("exact_death_date", 0)
        if points_ddate != 0:
            field_scores["ddate"] = points_ddate
            match_reasons.append(f"Exact Death Date ({points_ddate}pts)")
            death_score_added = True
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Matched exact_death_date. Set field_scores['ddate'] = {points_ddate}"
            )
    elif death_year_match:
        points_dyear = weights.get("year_death", 0)
        if points_dyear != 0:
            field_scores["dyear"] = points_dyear
            match_reasons.append(f"Exact Death Year ({c_d_year}) ({points_dyear}pts)")
            death_score_added = True
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Matched year_death. Set field_scores['dyear'] = {points_dyear}"
            )
    elif death_year_approx_match:
        points_dyear_approx = weights.get("approx_year_death", 0)
        if points_dyear_approx != 0:
            field_scores["dyear"] = points_dyear_approx
            match_reasons.append(
                f"Approx Death Year ({c_d_year} vs {t_d_year}) ({points_dyear_approx}pts)"
            )
            death_score_added = True
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Matched approx_year_death. Set field_scores['dyear'] = {points_dyear_approx}"
            )
    elif death_dates_absent and not death_score_added:
        points_ddate_abs = weights.get("death_dates_both_absent", 0)
        if points_ddate_abs != 0:
            field_scores["ddate"] = points_ddate_abs
            match_reasons.append(f"Death Dates Absent ({points_ddate_abs}pts)")
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Matched death_dates_both_absent. Set field_scores['ddate'] = {points_ddate_abs}"
            )

    # Place Scoring
    if t_pob and c_bplace and t_pob in c_bplace:
        points_pob = weights.get("contains_pob", 0)
        if points_pob != 0:
            field_scores["bplace"] = points_pob
            match_reasons.append(f"Birth Place Contains ({points_pob}pts)")
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Matched contains_pob. Set field_scores['bplace'] = {points_pob}"
            )
    condition1_pod_match = bool(t_pod and c_dplace and t_pod in c_dplace)
    is_target_pod_absent = not bool(t_pod)
    is_candidate_dplace_absent = not bool(c_dplace)
    condition2_pod_absent = bool(is_target_pod_absent and is_candidate_dplace_absent)
    if condition1_pod_match or condition2_pod_absent:
        points_pod = weights.get("contains_pod", 0)
        if points_pod != 0:
            field_scores["dplace"] = points_pod
            reason = (
                f"Death Place Contains ({points_pod}pts)"
                if condition1_pod_match
                else f"Death Places Both Absent ({points_pod}pts)"
            )
            match_reasons.append(reason)
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Matched Death Place condition ({'Contains' if condition1_pod_match else 'Both Absent'}). Set field_scores['dplace'] = {points_pod}"
            )

    # Gender Scoring
    if t_gender and c_gender and t_gender == c_gender:
        points_gender = weights.get("gender_match", 0)
        if points_gender != 0:
            field_scores["gender"] = points_gender
            match_reasons.append(
                f"Gender Match ({t_gender.upper()}) ({points_gender}pts)"
            )
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Matched gender_match. Set field_scores['gender'] = {points_gender}"
            )

    # Birth Bonus Scoring (if both birth year and birth place matched)
    if field_scores["byear"] > 0 and field_scores["bplace"] > 0:
        birth_bonus_points = weights.get("bonus_birth_info", 0)
        if birth_bonus_points != 0:
            field_scores["bbonus"] = birth_bonus_points
            match_reasons.append(f"Bonus Birth Info ({birth_bonus_points}pts)")
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Applied bonus_birth_info. Set field_scores['bbonus'] = {birth_bonus_points}"
            )

    # Death Bonus Scoring (if both death year/date and death place matched)
    if (field_scores["dyear"] > 0 or field_scores["ddate"] > 0) and field_scores[
        "dplace"
    ] > 0:
        death_bonus_points = weights.get("bonus_death_info", 0)
        if death_bonus_points != 0:
            field_scores["dbonus"] = death_bonus_points
            match_reasons.append(f"Bonus Death Info ({death_bonus_points}pts)")
            logger.debug(
                f"SCORE DEBUG ({c_id_debug}): Applied bonus_death_info. Set field_scores['dbonus'] = {death_bonus_points}"
            )

    # Calculate Final Total Score
    final_total_score = sum(field_scores.values())
    final_total_score = max(0.0, round(final_total_score))
    unique_reasons = sorted(list(set(match_reasons)))

    # Final Debug Logs
    logger.debug(
        f"SCORE DEBUG ({c_id_debug}): Final Field Scores Dict before return: {field_scores}"
    )
    logger.debug(
        f"SCORE DEBUG ({c_id_debug}): Calculated Total Score from dict: {final_total_score}"
    )

    # Return a COPY of the dictionary
    return final_total_score, field_scores.copy(), unique_reasons


# End of calculate_match_score


# ==============================================
# GedcomData Class
# ==============================================
class GedcomData:
    def __init__(self, gedcom_path: Union[str, Path]):
        self.path = Path(gedcom_path).resolve()
        self.reader: Optional[GedcomReaderType] = None
        self.indi_index: Dict[str, GedcomIndividualType] = {}  # Index of INDI records
        self.processed_data_cache: Dict[str, Dict[str, Any]] = (
            {}
        )  # NEW: Cache for processed data
        self.id_to_parents: Dict[str, Set[str]] = {}
        self.id_to_children: Dict[str, Set[str]] = {}
        self.indi_index_build_time: float = 0
        self.family_maps_build_time: float = 0
        self.data_processing_time: float = 0  # NEW: Time for pre-processing

        if not self.path.is_file():
            logger.critical(f"GEDCOM file not found: {self.path}")
            raise FileNotFoundError(f"GEDCOM file not found: {self.path}")
        try:
            logger.info(f"Loading GEDCOM file: {self.path}")
            load_start = time.time()
            self.reader = GedcomReader(str(self.path))
            load_time = time.time() - load_start
            logger.info(f"GEDCOM file loaded in {load_time:.2f}s.")
        except Exception as e:
            logger.critical(
                f"Failed to load/parse GEDCOM file {self.path}: {e}", exc_info=True
            )
            raise
        self.build_caches()  # Build caches upon initialization

    def build_caches(self):
        """Builds the individual index, family maps, and pre-processes data."""
        if not self.reader:
            logger.error("[Cache Build] Cannot build caches: GedcomReader is None.")
            return
        self._build_indi_index()
        # Only build maps and process data if index was successful
        if self.indi_index:
            self._build_family_maps()
            self._pre_process_individual_data()  # NEW: Call pre-processing
        else:
            logger.error(
                "[Cache Build] Skipping map build and data pre-processing due to empty INDI index."
            )

    def _build_indi_index(self):
        """Builds a dictionary mapping normalized IDs to Individual records."""
        if not self.reader:
            logger.error("[Cache Build] Cannot build INDI index: GedcomReader is None.")
            return
        start_time = time.time()
        logger.info("[Cache] Building INDI index...")
        self.indi_index = {}
        count = 0
        skipped = 0
        try:
            for indi_record in self.reader.records0(TAG_INDI):
                if (
                    _is_individual(indi_record)
                    and hasattr(indi_record, "xref_id")
                    and indi_record.xref_id
                ):
                    norm_id = _normalize_id(indi_record.xref_id)
                    if norm_id:
                        if norm_id in self.indi_index:
                            logger.warning(
                                f"Duplicate normalized INDI ID found: {norm_id}. Overwriting."
                            )
                        self.indi_index[norm_id] = indi_record
                        count += 1
                    elif logger.isEnabledFor(logging.DEBUG):
                        skipped += 1
                        logger.debug(
                            f"Skipping INDI with unnormalizable xref_id: {indi_record.xref_id}"
                        )
                elif logger.isEnabledFor(logging.DEBUG):
                    skipped += 1
                    if hasattr(indi_record, "xref_id"):
                        logger.debug(
                            f"Skipping non-Individual record: Type={type(indi_record).__name__}, Xref={indi_record.xref_id}"
                        )
                    else:
                        logger.debug(
                            f"Skipping record with no xref_id: Type={type(indi_record).__name__}"
                        )
        except StopIteration:
            logger.info("[Cache] Finished iterating INDI records for index.")
        except Exception as e:
            logger.error(
                f"[Cache Build] Error during INDI index build: {e}. Index may be incomplete.",
                exc_info=True,
            )
        elapsed = time.time() - start_time
        self.indi_index_build_time = elapsed
        if count > 0:
            logger.info(
                f"[Cache] INDI index built with {count} individuals ({skipped} skipped) in {elapsed:.2f}s."
            )
        else:
            logger.error(
                f"[Cache Build] INDI index is EMPTY after build attempt ({skipped} skipped) in {elapsed:.2f}s."
            )

    def _build_family_maps(self):
        """Builds dictionaries mapping child IDs to parent IDs and parent IDs to child IDs."""
        if not self.reader:
            logger.error(
                "[Cache Build] Cannot build family maps: GedcomReader is None."
            )
            return
        start_time = time.time()
        logger.info("[Cache] Building family maps...")
        self.id_to_parents = {}
        self.id_to_children = {}
        fam_count = 0
        processed_links = 0
        skipped_links = 0
        try:
            for fam in self.reader.records0("FAM"):
                fam_count += 1
                if not _is_record(fam):
                    logger.debug(f"Skipping non-record FAM entry: {type(fam)}")
                    continue
                fam_id_log = getattr(fam, "xref_id", "N/A_FAM")
                parents: Set[str] = set()
                for parent_tag in [TAG_HUSBAND, TAG_WIFE]:
                    parent_ref = fam.sub_tag(parent_tag)
                    if parent_ref and hasattr(parent_ref, "xref_id"):
                        parent_id = _normalize_id(parent_ref.xref_id)
                        if parent_id:
                            parents.add(parent_id)
                        elif logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"Skipping parent with invalid/unnormalizable ID {getattr(parent_ref, 'xref_id', '?')} in FAM {fam_id_log}"
                            )
                children_tags = fam.sub_tags(TAG_CHILD)
                for child_tag in children_tags:
                    if child_tag and hasattr(child_tag, "xref_id"):
                        child_id = _normalize_id(child_tag.xref_id)
                        if child_id:
                            for parent_id in parents:
                                self.id_to_children.setdefault(parent_id, set()).add(
                                    child_id
                                )
                            if parents:
                                self.id_to_parents.setdefault(child_id, set()).update(
                                    parents
                                )
                                processed_links += 1
                            elif logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    f"Child {child_id} found in FAM {fam_id_log} but no valid parents identified in this specific record."
                                )
                        elif logger.isEnabledFor(logging.DEBUG):
                            skipped_links += 1
                            logger.debug(
                                f"Skipping child with invalid/unnormalizable ID {getattr(child_tag, 'xref_id', '?')} in FAM {fam_id_log}"
                            )
                    elif child_tag is not None and logger.isEnabledFor(logging.DEBUG):
                        skipped_links += 1
                        logger.debug(
                            f"Skipping CHIL record in FAM {fam_id_log} with invalid format: Type={type(child_tag).__name__}"
                        )
        except StopIteration:
            logger.info("[Cache] Finished iterating FAM records for maps.")
        except Exception as e:
            logger.error(
                f"[Cache Build] Unexpected error during family map build: {e}. Maps may be incomplete.",
                exc_info=True,
            )
        elapsed = time.time() - start_time
        self.family_maps_build_time = elapsed
        parent_map_count = len(self.id_to_parents)
        child_map_count = len(self.id_to_children)
        logger.info(
            f"[Cache] Family maps built: {fam_count} FAMs processed. Added {processed_links} child-parent relationships ({skipped_links} skipped invalid links/IDs). Map sizes: {parent_map_count} child->parents entries, {child_map_count} parent->children entries in {elapsed:.2f}s."
        )
        if parent_map_count == 0 and child_map_count == 0 and fam_count > 0:
            logger.warning(
                "[Cache Build] Family maps are EMPTY despite processing FAM records. Check GEDCOM structure or parsing logic."
            )

    def _pre_process_individual_data(self):
        """NEW: Extracts and caches key data points for each individual."""
        if not self.indi_index:
            logger.error("Cannot pre-process data: INDI index is not built.")
            return
        start_time = time.time()
        logger.info("[Pre-Process] Extracting key data for individuals...")
        self.processed_data_cache = {}
        processed_count = 0
        errors = 0
        for norm_id, indi in self.indi_index.items():
            try:
                # Get full name first using the utility
                full_name_disp = _get_full_name(indi)  # Use the robust getter

                # Derive scoring name parts from the full name (simple split)
                name_parts = (
                    full_name_disp.split() if full_name_disp != "Unknown" else []
                )
                first_name_score = name_parts[0] if name_parts else ""
                surname_score = name_parts[-1] if len(name_parts) > 1 else ""
                # Extract raw names from tags if needed for specific logic elsewhere
                name_rec = indi.sub_tag(TAG_NAME)
                givn_raw = name_rec.sub_tag_value(TAG_GIVN) if name_rec else None
                surn_raw = name_rec.sub_tag_value(TAG_SURN) if name_rec else None

                # Extract gender
                sex_raw = indi.sub_tag_value(TAG_SEX)
                sex_lower = str(sex_raw).lower() if sex_raw else None
                gender_norm = sex_lower if sex_lower in ["m", "f"] else None

                # Extract birth info
                birth_date_obj, birth_date_str, birth_place_raw = _get_event_info(
                    indi, TAG_BIRTH
                )
                birth_year = birth_date_obj.year if birth_date_obj else None
                birth_date_disp = _clean_display_date(birth_date_str)
                birth_place_disp = birth_place_raw if birth_place_raw != "N/A" else None

                # Extract death info
                death_date_obj, death_date_str, death_place_raw = _get_event_info(
                    indi, TAG_DEATH
                )
                death_year = death_date_obj.year if death_date_obj else None
                death_date_disp = _clean_display_date(death_date_str)
                death_place_disp = death_place_raw if death_place_raw != "N/A" else None

                self.processed_data_cache[norm_id] = {
                    "norm_id": norm_id,
                    "display_id": getattr(indi, "xref_id", norm_id),
                    "givn_raw": givn_raw,  # Keep raw tag value if needed
                    "surn_raw": surn_raw,  # Keep raw tag value if needed
                    "first_name": first_name_score,  # For scoring
                    "surname": surname_score,  # For scoring
                    "full_name_disp": full_name_disp,  # For display
                    "gender_raw": sex_raw,
                    "gender_norm": gender_norm,
                    "birth_date_obj": birth_date_obj,
                    "birth_date_str": birth_date_str,
                    "birth_date_disp": birth_date_disp,
                    "birth_year": birth_year,
                    "birth_place_raw": birth_place_raw,
                    "birth_place_disp": birth_place_disp,
                    "death_date_obj": death_date_obj,
                    "death_date_str": death_date_str,
                    "death_date_disp": death_date_disp,
                    "death_year": death_year,
                    "death_place_raw": death_place_raw,
                    "death_place_disp": death_place_disp,
                }
                processed_count += 1
            except Exception as e:
                logger.error(
                    f"Error pre-processing individual {norm_id}: {e}", exc_info=False
                )
                errors += 1
        elapsed = time.time() - start_time
        self.data_processing_time = elapsed
        logger.info(
            f"[Pre-Process] Processed data for {processed_count} individuals ({errors} errors) in {elapsed:.2f}s."
        )
        if not self.processed_data_cache:
            logger.error(
                "[Pre-Process] Processed data cache is EMPTY after build attempt."
            )

    def get_processed_indi_data(self, norm_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves pre-processed data for an individual from the cache."""
        if not self.processed_data_cache:
            logger.warning(
                "Attempting to get processed data, but cache is empty. Triggering pre-processing."
            )
            self._pre_process_individual_data()
        return self.processed_data_cache.get(norm_id)

    def find_individual_by_id(
        self, norm_id: Optional[str]
    ) -> Optional[GedcomIndividualType]:
        """Finds an individual by their normalized ID using the index."""
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
        if not found_indi and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Individual with normalized ID {norm_id} not found in INDI_INDEX."
            )
        return found_indi

    def get_related_individuals(
        self, individual: GedcomIndividualType, relationship_type: str
    ) -> List[GedcomIndividualType]:
        """Gets parents, children, siblings, or spouses using cached maps."""
        related_individuals: List[GedcomIndividualType] = []
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
                    potential_siblings = set().union(
                        *(self.id_to_children.get(p_id, set()) for p_id in parents)
                    )
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
                elif logger.isEnabledFor(logging.DEBUG):
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

    def _find_family_records(
        self, target_id: str, role_tag: str
    ) -> List[GedcomRecordType]:
        """Helper to find FAM records where target_id plays the specified role (HUSB, WIFE, CHIL). Less efficient scan."""
        matching_families: List[GedcomRecordType] = []
        if not self.reader or not target_id or not role_tag:
            return matching_families
        try:
            logger.debug(
                f"Scanning FAM records for {target_id} in role {role_tag} (less efficient)."
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
        except Exception as e:
            logger.error(
                f"Error finding FAMs via scan for ID {target_id}, role {role_tag}: {e}",
                exc_info=True,
            )
        return matching_families

    def _find_family_records_where_individual_is_parent(
        self, target_id: str
    ) -> List[Tuple[GedcomRecordType, bool, bool]]:
        """Finds FAM records where target_id is HUSB or WIFE using scan (less efficient than maps)."""
        matching_families_with_role: List[Tuple[GedcomRecordType, bool, bool]] = []
        husband_families = self._find_family_records(target_id, TAG_HUSBAND)
        wife_families = self._find_family_records(target_id, TAG_WIFE)
        processed_fam_ids = set()
        for fam in husband_families:
            fam_id = getattr(fam, "xref_id", None)
            if fam_id and fam_id not in processed_fam_ids:
                matching_families_with_role.append((fam, True, False))
                processed_fam_ids.add(fam_id)
        for fam in wife_families:
            fam_id = getattr(fam, "xref_id", None)
            if fam_id and fam_id not in processed_fam_ids:
                matching_families_with_role.append((fam, False, True))
                processed_fam_ids.add(fam_id)
        return matching_families_with_role

    def get_relationship_path(self, id1: str, id2: str) -> str:
        """Finds and explains the relationship path between two individuals using BFS."""
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
        )
        search_time = time.time() - search_start
        logger.debug(f"[PROFILE] BFS search completed in {search_time:.2f}s.")
        if not path_ids:
            profile_info = f"[PROFILE] Search: {search_time:.2f}s, MapsBuild: {self.family_maps_build_time:.2f}s, IndexBuild: {self.indi_index_build_time:.2f}s, PreProcess: {self.data_processing_time:.2f}s"
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
        profile_info = f"[PROFILE] Total Time: {total_process_time:.2f}s (BFS: {search_time:.2f}s, Explain: {explanation_time:.2f}s) [Build Times: Maps={self.family_maps_build_time:.2f}s, Index={self.indi_index_build_time:.2f}s, PreProcess={self.data_processing_time:.2f}s]"
        logger.debug(profile_info)
        return f"{explanation_str}\n{profile_info}"


# ==============================================
# Standalone Test Block (Updated)
# ==============================================
if __name__ == "__main__":
    # --- Test Runner Setup ---
    test_results_main: List[Tuple[str, str, str]] = []

    def _run_test_main(
        test_name: str, test_func: Callable, *args, **kwargs
    ) -> Tuple[str, str, str]:
        loggr = logger
        loggr.info(f"[ RUNNING ] {test_name}")
        status = "FAIL"
        message = ""
        expect_none = kwargs.pop("expected_none", False)
        try:
            result = test_func(*args, **kwargs)
            passed = False
            if expect_none:
                if result is None:
                    passed = True
                else:
                    message = f"Expected None, got {type(result).__name__}"
            elif result is True:
                passed = True
            else:
                message = f"Assertion failed or invalid return (returned {result} of type {type(result).__name__})"
            status = "PASS" if passed else "FAIL"
        except Exception as e:
            status = "FAIL"
            message = f"Exception: {type(e).__name__}: {str(e)}"
            loggr.error(f"Exception details for {test_name}: {message}", exc_info=True)
        log_level = logging.INFO if status == "PASS" else logging.ERROR
        log_message = f"[ {status:<6} ] {test_name}{f': {message}' if message and status == 'FAIL' else ''}"
        loggr.log(log_level, log_message)
        test_results_main.append((test_name, status, message))
        return (test_name, status, message)

    print("\n--- gedcom_utils.py Standalone Test Suite ---")
    overall_status_main = "PASS"

    # === Section 1: Standalone Utility Function Tests ===
    print("\n--- Section 1: Standalone Utility Tests ---")
    _run_test_main(
        "_normalize_id (valid)",
        lambda: _normalize_id("@I123@") == "I123" and _normalize_id("F45") == "F45",
    )
    _run_test_main(
        "_normalize_id (invalid str)",
        lambda: _normalize_id("Invalid"),
        expected_none=True,
    )
    _run_test_main(
        "_normalize_id (empty str)", lambda: _normalize_id(""), expected_none=True
    )
    _run_test_main(
        "_normalize_id (None input)", lambda: _normalize_id(None), expected_none=True
    )
    _run_test_main(
        "extract_and_fix_id (valid str)",
        lambda: extract_and_fix_id("@I123@") == "I123"
        and extract_and_fix_id("F45") == "F45",
    )
    _run_test_main(
        "extract_and_fix_id (invalid str)",
        lambda: extract_and_fix_id("Invalid"),
        expected_none=True,
    )
    _run_test_main(
        "extract_and_fix_id (invalid type)",
        lambda: extract_and_fix_id(123),
        expected_none=True,
    )
    _run_test_main(
        "extract_and_fix_id (None input)",
        lambda: extract_and_fix_id(None),
        expected_none=True,
    )

    _run_test_main(
        "_parse_date (YYYY)",
        lambda: (dt := _parse_date("1980")) is not None and dt.year == 1980,
    )
    _run_test_main(
        "_parse_date (Mon YYYY)",
        lambda: (dt := _parse_date("Jan 1995")) is not None
        and dt.month == 1
        and dt.year == 1995,
    )
    _run_test_main(
        "_parse_date (DD Mon YYYY)",
        lambda: (dt := _parse_date("15 Feb 2001")) is not None
        and dt.day == 15
        and dt.month == 2
        and dt.year == 2001,
    )
    _run_test_main(
        "_parse_date (with prefix)",
        lambda: (dt1 := _parse_date("ABT 1950")) is not None
        and dt1.year == 1950
        and (dt2 := _parse_date("BEF 20 MAR 1960")) is not None
        and dt2.day == 20
        and dt2.month == 3,
    )
    _run_test_main(
        "_parse_date (range)",
        lambda: (dt := _parse_date("BET 1910 AND 1912")) is not None
        and dt.year == 1910,
    )
    _run_test_main(
        "_parse_date (invalid)", lambda: _parse_date("Invalid Date"), expected_none=True
    )
    _run_test_main(
        "_parse_date (slash)",
        lambda: (dt := _parse_date("10/4/1993")) is not None and dt.year == 1993,
    )
    _run_test_main(
        "_parse_date (year only prefix)",
        lambda: (dt := _parse_date("ABT. 1850")) is not None and dt.year == 1850,
    )
    _run_test_main(
        "_parse_date (year fallback)",
        lambda: (dt := _parse_date("Junk 1776 text")) is not None
        and dt.year == 1776
        and dt.month == 1
        and dt.day == 1,
    )
    _run_test_main(
        "_parse_date (ordinals)",
        lambda: (dt := _parse_date("15TH JUNE 1923")) is not None
        and dt.day == 15
        and dt.month == 6,
    )
    _run_test_main(
        "_parse_date (month day year comma)",
        lambda: (dt := _parse_date("July 13, 1952")) is not None
        and dt.month == 7
        and dt.day == 13,
    )
    _run_test_main(
        "_parse_date (abt dot month year)",
        lambda: (dt := _parse_date("Abt. Nov 1787")) is not None
        and dt.month == 11
        and dt.year == 1787,
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

    # --- calculate_match_score test (using pre-processed structure) ---
    test_crit = {"first_name": "a"}
    test_cand_proc = {
        "id": "T1",
        "norm_id": "T1",
        "first_name": "a",
        "surname": "",
        "birth_year": None,
        "birth_date_obj": None,
        "birth_place_disp": None,
        "death_year": None,
        "death_date_obj": None,
        "death_place_disp": None,
        "gender_norm": None,
    }
    _run_test_main(
        "calculate_match_score (basic)",
        lambda: isinstance(
            calculate_match_score(
                test_crit, test_cand_proc, scoring_weights={"contains_first_name": 10}
            ),
            tuple,
        ),
    )

    # === Section 2: GedcomData Functional Tests ===
    print(
        "\n--- Section 2: GedcomData Functional Tests (requires config & GEDCOM file) ---"
    )
    gedcom_data: Optional[GedcomData] = None
    gedcom_load_status = "SKIPPED"
    gedcom_load_message = "Prerequisites not met (config or ged4py)"
    can_load_gedcom = False
    gedcom_path: Optional[Path] = None

    # Attempt to load GedcomData using config
    if (
        config_instance
        and hasattr(config_instance, "GEDCOM_FILE_PATH")
        and config_instance.GEDCOM_FILE_PATH
    ):
        potential_path = Path(config_instance.GEDCOM_FILE_PATH)
        if potential_path.is_file():
            gedcom_path = potential_path
            can_load_gedcom = True
            gedcom_load_message = f"Configured path found: {gedcom_path.name}"
        else:
            gedcom_load_message = f"GEDCOM_FILE_PATH '{config_instance.GEDCOM_FILE_PATH}' is not a valid file."
            can_load_gedcom = False
    else:
        gedcom_load_message = "GEDCOM_FILE_PATH not configured or empty in config."
        can_load_gedcom = False

    # Run instantiation test
    if can_load_gedcom and gedcom_path:
        test_name = "GedcomData Instantiation"
        logger.info(f"[ RUNNING ] {test_name}")
        try:
            gedcom_data = GedcomData(gedcom_path)
            gedcom_load_status = "PASS"
            gedcom_load_message = f"Loaded {gedcom_path.name}"
            logger.info(f"[ PASS    ] {test_name}: {gedcom_load_message}")
        except Exception as e:
            gedcom_load_status = "FAIL"
            gedcom_load_message = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"[ FAIL    ] {test_name}: {gedcom_load_message}")
        test_results_main.append((test_name, gedcom_load_status, gedcom_load_message))
    else:
        test_results_main.append(
            ("GedcomData Instantiation", "SKIPPED", gedcom_load_message)
        )

    # --- Run functional tests ONLY if GedcomData loaded successfully ---
    if gedcom_data and gedcom_load_status == "PASS":
        print(
            "\n>>> IMPORTANT: Functional tests below require modifying placeholder IDs/Names <<<"
        )
        # Use IDs relevant to your specific GEDCOM for meaningful tests
        TEST_INDI_ID_1 = "I102281560836"  # Example: Wayne Gordon Gault
        TEST_INDI_ID_2 = "I102281560744"  # Example: Fraser Gault (Uncle)
        print(f">>> Using Test IDs: {TEST_INDI_ID_1}, {TEST_INDI_ID_2}")

        # Test find_individual_by_id
        _run_test_main(
            f"find_individual_by_id({TEST_INDI_ID_1})",
            lambda: gedcom_data.find_individual_by_id(TEST_INDI_ID_1) is not None,
        )

        # Test get_related_individuals
        indi1_obj = gedcom_data.find_individual_by_id(TEST_INDI_ID_1)
        test_name_rel = f"get_related_individuals({TEST_INDI_ID_1}, 'parents')"
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

        # Test get_relationship_path
        test_name_path = f"get_relationship_path({TEST_INDI_ID_1}, {TEST_INDI_ID_2})"
        _run_test_main(
            test_name_path,
            lambda: "Error:"
            not in gedcom_data.get_relationship_path(TEST_INDI_ID_1, TEST_INDI_ID_2),
        )

    else:  # Skip functional tests if load failed
        functional_skip_reason = "GedcomData failed to load"
        test_results_main.append(
            (f"find_individual_by_id(placeholder)", "SKIPPED", functional_skip_reason)
        )
        test_results_main.append(
            (
                f"get_related_individuals(placeholder, 'parents')",
                "SKIPPED",
                functional_skip_reason,
            )
        )
        test_results_main.append(
            (
                f"get_relationship_path(placeholder1, placeholder2)",
                "SKIPPED",
                functional_skip_reason,
            )
        )

    # --- Final Summary ---
    print("\n--- Test Summary ---")
    name_width = max((len(name) for name, _, _ in test_results_main), default=50)
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

# --- End of gedcom_utils.py ---
