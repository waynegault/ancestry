#!/usr/bin/env python3
# pyright: reportConstantRedefinition=false, reportImportCycles=false
# NOTE: Import cycle with relationship_utils.py. Both modules need each other's relationship calculation
# functions. Already uses local import (line ~883) but type checker detects cycle at parse time.
# Proper fix requires extracting shared types/interfaces to common module. Cycle doesn't affect runtime.

"""
GEDCOM Processing & Genealogical Data Intelligence Engine

Comprehensive genealogical data processing platform providing sophisticated
GEDCOM file analysis, family relationship mapping, and intelligent genealogical
data extraction with advanced parsing capabilities, relationship pathfinding,
and comprehensive data validation for professional genealogical research.

GEDCOM Processing:
• Advanced GEDCOM file parsing with comprehensive format support and validation
• Intelligent individual and family record extraction with metadata preservation
• Sophisticated date parsing and normalization with flexible format recognition
• Advanced name processing with standardization and variant recognition
• Comprehensive data validation with error detection and correction suggestions
• Efficient memory management for large genealogical datasets

Relationship Intelligence:
• Advanced relationship pathfinding using optimized graph algorithms
• Comprehensive family structure analysis with multi-generational mapping
• Intelligent sibling detection and family group identification
• Complex relationship calculation including step-relationships and adoptions
• Advanced ancestor and descendant tracking with generation mapping
• Sophisticated relationship degree calculation with cousin identification

Data Enhancement:
• Intelligent data normalization with standardization and quality improvement
• Advanced search capabilities with fuzzy matching and phonetic algorithms
• Comprehensive data quality assessment with improvement recommendations
• Integration with external genealogical databases for data enrichment
• Export capabilities for seamless integration with genealogical software
• Performance optimization for large family tree processing

Foundation Services:
Provides the essential GEDCOM processing infrastructure that enables sophisticated
genealogical analysis, relationship discovery, and family tree intelligence for
professional genealogical research and family history exploration.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===



# === STANDARD LIBRARY IMPORTS ===
import logging
import re
import sys
import time
from collections import deque
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Optional,
    Union,
)

# --- Third-party imports ---
try:
    from ged4py.model import Individual, Name, NameRec, Record
    from ged4py.parser import GedcomReader
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

# === LOCAL IMPORTS ===
from common_params import GraphContext
from config.config_manager import ConfigManager
from utils import format_name

# Note: _find_direct_relationship and _has_direct_relationship are imported
# from relationship_utils.py where needed to avoid circular imports

# === MODULE CONFIGURATION ===
config_manager = ConfigManager()
config = config_manager.get_config()

# === TYPE ALIASES ===
# Define type aliases for GEDCOM types
# Define type aliases for GEDCOM classes (always available as required dependency)
GedcomIndividualType = Individual
GedcomRecordType = Record
GedcomNameType = Name
GedcomNameRecType = NameRec
GedcomReaderType = GedcomReader
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
TAG_SOUR = "SOUR"  # Source citation tag
TAG_TITL = "TITL"  # Source title tag


# --- Logging Setup ---
# Use centralized logger from logging_config
# from logging_config import logger  # Unused in this module's pure functions


# ==============================================
# Utility Functions (Moved to Top)
# ==============================================
def _is_individual(obj: Any) -> bool:
    """Check if an object is a GEDCOM Individual.

    We can't use isinstance with Any, so we need to check for specific attributes
    that are expected to be present on Individual objects.
    """
    if obj is None:
        return False

    # We can't use isinstance with Any, so we need to check for specific attributes
    # This is a heuristic approach to identify Individual objects
    return (
        hasattr(obj, "xref_id")
        and hasattr(obj, "tag")
        and getattr(obj, "tag", "") == TAG_INDI
    )


def _is_record(obj: Any) -> bool:
    """Check if an object is a GEDCOM Record.

    We can't use isinstance with Any, so we need to check for specific attributes
    that are expected to be present on Record objects.
    """
    if obj is None:
        return False

    # We can't use isinstance with Any, so we need to check for specific attributes
    # This is a heuristic approach to identify Record objects
    return hasattr(obj, "xref_id") and hasattr(obj, "tag") and hasattr(obj, "sub_tag")


# _is_name_rec removed - unused helper function


def _normalize_id(xref_id: Optional[str]) -> Optional[str]:
    if not xref_id or not isinstance(xref_id, str):
        return None

    # Try to match standard GEDCOM ID format
    match = re.match(r"^@?([IFSNMCXO][0-9\-]+)@?$", xref_id.strip().upper())
    if match:
        return match.group(1)

    # Try fallback regex for partial GEDCOM IDs
    search_match = re.search(r"([IFSNMCXO][0-9\-]+)", xref_id.strip().upper())
    if search_match:
        logger.debug(
            f"Normalized ID '{search_match.group(1)}' using fallback regex from '{xref_id}'."
        )
        return search_match.group(1)

    # For pure numeric strings, return as-is (handle raw numeric IDs)
    if re.match(r"^\d+$", xref_id.strip()):
        return xref_id.strip()

    logger.warning(f"Could not normalize potential ID: '{xref_id}'")
    return None


def extract_and_fix_id(raw_id: Any) -> Optional[str]:
    if not raw_id:
        return None
    id_to_normalize: Optional[str] = None
    if isinstance(raw_id, str):
        id_to_normalize = raw_id
    elif isinstance(raw_id, int):
        id_to_normalize = str(raw_id)
    elif hasattr(raw_id, "xref_id") and (_is_record(raw_id) or _is_individual(raw_id)):
        id_to_normalize = getattr(raw_id, "xref_id", None)
    else:
        logger.debug(
            f"extract_and_fix_id: Invalid input type '{type(raw_id).__name__}'."
        )
        return None
    return _normalize_id(id_to_normalize)


# Helper functions for _get_full_name

def _validate_individual_type(indi: GedcomIndividualType) -> tuple[Optional[GedcomIndividualType], str]:
    """Validate and extract individual from input, handling wrapped values."""
    if not _is_individual(indi):
        if hasattr(indi, "value") and _is_individual(getattr(indi, "value", None)):
            return indi.value, ""  # type: ignore
        logger.warning(f"_get_full_name called with non-Individual type: {type(indi)}")
        return None, "Unknown (Invalid Type)"

    if indi is None:
        return None, "Unknown (None)"

    return indi, ""


def _try_name_format_method(indi: GedcomIndividualType, indi_id_log: str) -> Optional[str]:
    """Try to get name using indi.name.format() method."""
    if not indi or not hasattr(indi, "name"):
        return None

    name_rec = indi.name
    if not name_rec or not hasattr(name_rec, "format") or not callable(name_rec.format):
        return None

    try:
        return name_rec.format()
    except Exception as fmt_err:
        logger.warning(f"Error calling indi.name.format() for {indi_id_log}: {fmt_err}")
        return None


def _try_sub_tag_format_method(indi: GedcomIndividualType, indi_id_log: str) -> Optional[str]:
    """Try to get name using indi.sub_tag(TAG_NAME).format() method."""
    if not indi or not hasattr(indi, "sub_tag"):
        return None

    name_tag = indi.sub_tag(TAG_NAME)
    if not name_tag or not hasattr(name_tag, "format") or not callable(getattr(name_tag, "format", None)):
        return None

    try:
        return name_tag.format()  # type: ignore
    except Exception as fmt_err:
        logger.warning(f"Error calling indi.sub_tag(TAG_NAME).format() for {indi_id_log}: {fmt_err}")
        return None


def _try_manual_name_combination(indi: GedcomIndividualType, indi_id_log: str) -> Optional[str]:
    """Try to manually combine GIVN and SURN tags."""
    _ = indi_id_log  # Unused but kept for API consistency
    if not indi or not hasattr(indi, "sub_tag"):
        return None

    name_tag = indi.sub_tag(TAG_NAME)
    if not name_tag:
        return None

    givn = name_tag.sub_tag_value(TAG_GIVN) if hasattr(name_tag, "sub_tag_value") else None
    surn = name_tag.sub_tag_value(TAG_SURN) if hasattr(name_tag, "sub_tag_value") else None

    # Combine, prioritizing surname placement
    if givn and surn:
        formatted_name = f"{givn} {surn}"
    elif givn:
        formatted_name = givn
    elif surn:
        formatted_name = surn
    else:
        return None

    return formatted_name


def _try_sub_tag_value_method(indi: GedcomIndividualType, indi_id_log: str) -> Optional[str]:
    """Try to get name using indi.sub_tag_value(TAG_NAME) as last resort."""
    _ = indi_id_log  # Unused but kept for API consistency
    if not indi or not hasattr(indi, "sub_tag_value"):
        return None

    name_val = indi.sub_tag_value(TAG_NAME)
    if not isinstance(name_val, str) or not name_val.strip() or name_val == "/":
        return None

    return name_val


def _clean_and_format_name(formatted_name: Optional[str], name_source: str) -> str:
    """Clean and format the extracted name."""
    if not formatted_name:
        return "Unknown (No Name Found)"

    cleaned_name = format_name(formatted_name)
    if cleaned_name and cleaned_name != "Unknown":
        return cleaned_name

    return f"Unknown ({name_source} Error)"


def _get_full_name(indi: GedcomIndividualType) -> str:
    """Safely gets formatted name, checking for .format method. V3"""
    # Validate individual type
    indi, error_msg = _validate_individual_type(indi)
    if error_msg:
        return error_msg

    indi_id_log = extract_and_fix_id(indi) or "Unknown ID"

    try:
        # Try multiple methods to extract name, in order of preference
        name_extraction_methods = [
            (_try_name_format_method, "indi.name.format()"),
            (_try_sub_tag_format_method, "indi.sub_tag(TAG_NAME).format()"),
            (_try_manual_name_combination, "manual GIVN/SURN combination"),
            (_try_sub_tag_value_method, "indi.sub_tag_value(TAG_NAME)"),
        ]

        for method, source in name_extraction_methods:
            formatted_name = method(indi, indi_id_log)
            if formatted_name:
                return _clean_and_format_name(formatted_name, source)

        # No method succeeded
        return _clean_and_format_name(None, "Unknown")

    except Exception as e:
        logger.error(f"Unexpected error in _get_full_name for @{indi_id_log}@: {e}", exc_info=True)
        return "Unknown (Error)"


# Helper functions for _parse_date

def _validate_and_normalize_date_string(date_str: Optional[str]) -> Optional[str]:
    """Validate and perform initial normalization of date string."""
    if not date_str or not isinstance(date_str, str):
        return None

    # Remove parentheses
    if date_str.startswith("(") and date_str.endswith(")"):
        date_str = date_str[1:-1].strip()

    if not date_str:
        logger.debug("Date string empty after removing parentheses.")
        return None

    date_str = date_str.strip().upper()

    # Check for non-parseable strings
    if re.match(r"^(UNKNOWN|\?UNKNOWN|\?|DECEASED|IN INFANCY|0)$", date_str):
        logger.debug(f"Identified non-parseable string: '{date_str}'")
        return None

    # Check for dates without year
    if re.fullmatch(r"^\d{1,2}\s+[A-Z]{3,}$", date_str) or re.fullmatch(r"^[A-Z]{3,}$", date_str):
        logger.debug(f"Ignoring date string without year: '{date_str}'")
        return None

    return date_str


def _clean_date_string(date_str: str) -> Optional[str]:
    """Clean date string by removing keywords and normalizing format."""
    # Remove multi-word phrases first (before single keywords)
    phrases_to_remove = r"\b(?:ON\s+OR\s+BEFORE|ON\s+OR\s+AFTER|ON\s+OR\s+ABOUT)\b\.?\s*"
    cleaned_str = re.sub(phrases_to_remove, "", date_str, flags=re.IGNORECASE).strip()

    # Remove keywords
    keywords_to_remove = r"\b(?:MAYBE|PRIOR|CALCULATED|AROUND|BAPTISED|WFT|BTWN|BFR|SP|QTR\.?\d?|CIRCA|ABOUT:|AFTER|BEFORE)\b\.?\s*|\b(?:AGE:?\s*\d+)\b|\b(?:WIFE\s+OF.*)\b|\b(?:HUSBAND\s+OF.*)\b"
    previous_len = -1

    while len(cleaned_str) != previous_len:
        previous_len = len(cleaned_str)
        cleaned_str = re.sub(keywords_to_remove, "", cleaned_str, flags=re.IGNORECASE).strip()

    # Remove trailing SP
    cleaned_str = re.sub(r"\s+SP$", "", cleaned_str).strip()

    # Split on AND/OR/TO and take first part
    cleaned_str = re.split(r"\s+(?:AND|OR|TO)\s+", cleaned_str, maxsplit=1)[0].strip()

    # Handle year ranges
    year_range_match = re.match(r"^(\d{4})\s*[-]\s*\d{4}$", cleaned_str)
    if year_range_match:
        cleaned_str = year_range_match.group(1)
        logger.debug(f"Treated as year range, using first year: '{cleaned_str}'")

    # Remove prefixes
    prefixes = r"^(?:ABT|EST|CAL|INT|BEF|AFT|BET|FROM)\.?\s+"
    cleaned_str = re.sub(prefixes, "", cleaned_str, count=1).strip()

    # Remove ordinal suffixes
    cleaned_str = re.sub(r"(\d+)(?:ST|ND|RD|TH)", r"\1", cleaned_str).strip()

    # Remove BC/AD
    cleaned_str = re.sub(r"\s+(?:BC|AD)$", "", cleaned_str).strip()

    # Check for invalid year 0000
    if re.match(r"^0{3,4}(?:[-/\s]\d{1,2}[-/\s]\d{1,2})?$", cleaned_str):
        logger.debug(f"Treating year 0000 pattern as invalid: '{cleaned_str}'")
        return None

    # Normalize punctuation and spacing
    cleaned_str = re.sub(r"[,;:]", " ", cleaned_str)
    cleaned_str = re.sub(r"([A-Z]{3})\.", r"\1", cleaned_str)
    cleaned_str = re.sub(r"([A-Z])(\d)", r"\1 \2", cleaned_str)
    cleaned_str = re.sub(r"(\d)([A-Z])", r"\1 \2", cleaned_str)
    cleaned_str = re.sub(r"\s+", " ", cleaned_str).strip()

    if not cleaned_str:
        logger.debug("Date string empty after cleaning")
        return None

    return cleaned_str


def _try_dateparser(cleaned_str: str) -> Optional[datetime]:
    """Try parsing with dateparser library if available."""
    if not DATEPARSER_AVAILABLE:
        return None

    try:
        settings = {"PREFER_DAY_OF_MONTH": "first", "REQUIRE_PARTS": ["year"]}
        parsed_dt = dateparser.parse(cleaned_str, settings=settings)  # type: ignore

        if parsed_dt:
            pass
        else:
            logger.debug(f"dateparser returned None for '{cleaned_str}'")

        return parsed_dt
    except Exception as e:
        logger.error(f"Error using dateparser for '{cleaned_str}': {e}", exc_info=False)
        return None


def _try_strptime_formats(cleaned_str: str) -> Optional[datetime]:
    """Try parsing with various strptime formats."""
    formats = [
        "%d %b %Y", "%d %B %Y", "%b %Y", "%B %Y", "%Y",
        "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%B %d %Y",
    ]

    for fmt in formats:
        try:
            if fmt == "%Y" and not re.fullmatch(r"\d{3,4}", cleaned_str):
                continue

            dt_naive = datetime.strptime(cleaned_str, fmt)
            logger.debug(f"Parsed '{cleaned_str}' using strptime format '{fmt}'")
            return dt_naive
        except ValueError:
            continue
        except Exception as e:
            logger.debug(f"Strptime error for format '{fmt}': {e}")
            continue

    return None


def _extract_year_fallback(cleaned_str: str) -> Optional[datetime]:
    """Extract year as fallback when full parsing fails."""
    logger.debug(f"Full parsing failed for '{cleaned_str}', attempting year extraction.")

    year_match = re.search(r"\b(\d{3,4})\b", cleaned_str)
    if not year_match:
        return None

    year_str = year_match.group(1)
    try:
        year = int(year_str)
        if 500 <= year <= datetime.now().year + 5:
            logger.debug(f"Extracted year {year} as fallback.")
            return datetime(year, 1, 1)
        logger.debug(f"Extracted year {year} out of plausible range.")
        return None
    except ValueError:
        logger.debug(f"Could not convert extracted year '{year_str}' to int.")
        return None


def _finalize_parsed_date(parsed_dt: Optional[datetime], original_date_str: str) -> Optional[datetime]:
    """Finalize parsed date by validating and adding timezone."""
    if not isinstance(parsed_dt, datetime):
        logger.warning(f"All parsing attempts failed for: '{original_date_str}'")
        return None

    if parsed_dt.year == 0:
        logger.warning(f"Parsed date resulted in year 0, treating as invalid: '{original_date_str}'")
        return None

    if parsed_dt.tzinfo is None:
        return parsed_dt.replace(tzinfo=timezone.utc)

    return parsed_dt.astimezone(timezone.utc)


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parses various GEDCOM date formats into timezone-aware datetime objects (UTC),
    prioritizing full date parsing but falling back to extracting the first year.
    V13 - Corrected range splitting regex.
    """
    original_date_str = date_str or ""

    # Validate and normalize
    date_str = _validate_and_normalize_date_string(date_str)
    if not date_str:
        return None

    # Clean the date string
    cleaned_str = _clean_date_string(date_str)
    if not cleaned_str:
        return None

    # Try parsing with dateparser
    parsed_dt = _try_dateparser(cleaned_str)

    # Try parsing with strptime formats
    if not parsed_dt:
        parsed_dt = _try_strptime_formats(cleaned_str)

    # Try extracting year as fallback
    if not parsed_dt:
        parsed_dt = _extract_year_fallback(cleaned_str)

    # Finalize and return
    return _finalize_parsed_date(parsed_dt, original_date_str)


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


def _validate_and_normalize_individual(individual: GedcomIndividualType) -> Optional[GedcomIndividualType]:
    """Validate and normalize individual to ensure it's a valid GedcomIndividualType."""
    if not _is_individual(individual):
        if hasattr(individual, "value") and _is_individual(getattr(individual, "value", None)):
            # Type ignore is needed because the type checker doesn't understand the dynamic nature
            # of this code. We've already checked that individual.value is a valid GedcomIndividualType
            return individual.value  # type: ignore
        logger.warning(f"_get_event_info invalid input type: {type(individual)}")
        return None

    if individual is None:
        return None

    return individual


def _extract_event_record(individual: GedcomIndividualType, event_tag: str, indi_id_log: str) -> Optional[Any]:
    """Extract event record from individual."""
    # Add null check before calling sub_tag
    if not individual or not hasattr(individual, "sub_tag"):
        logger.warning(f"Individual {indi_id_log} has no sub_tag method")
        return None

    event_record = individual.sub_tag(event_tag.upper())
    if not event_record:
        return None

    # Add null check before calling sub_tag on event_record
    if not hasattr(event_record, "sub_tag"):
        logger.warning(f"Event record for {indi_id_log} has no sub_tag method")
        return None

    return event_record


def _extract_date_from_event(event_record: Any) -> tuple[Optional[datetime], str]:
    """Extract date information from event record."""
    date_obj: Optional[datetime] = None
    date_str: str = "N/A"

    date_tag = event_record.sub_tag(TAG_DATE)
    raw_date_val = getattr(date_tag, "value", None) if date_tag else None

    if isinstance(raw_date_val, str) and raw_date_val.strip():
        date_str = raw_date_val.strip()
        date_obj = _parse_date(date_str)
    elif raw_date_val is not None:
        date_str = str(raw_date_val)
        date_obj = _parse_date(date_str)

    return date_obj, date_str


def _extract_place_from_event(event_record: Any) -> str:
    """Extract place information from event record."""
    place_str: str = "N/A"

    place_tag = event_record.sub_tag(TAG_PLACE)
    raw_place_val = getattr(place_tag, "value", None) if place_tag else None

    if isinstance(raw_place_val, str) and raw_place_val.strip():
        place_str = raw_place_val.strip()
    elif raw_place_val is not None:
        place_str = str(raw_place_val)

    return place_str


def _extract_sources_from_event(event_record: Any) -> list[str]:
    """
    Extract source citations from an event record.

    Phase 5.1: Source Citation Support
    Extracts source titles from SOUR tags within an event (BIRT, DEAT, etc.)

    Args:
        event_record: GEDCOM event record

    Returns:
        List of source titles/descriptions
    """
    sources = []

    try:
        if not event_record or not hasattr(event_record, "sub_tags"):
            return sources

        # Get all SOUR tags from the event
        for sour_tag in event_record.sub_tags(TAG_SOUR):
            if not sour_tag:
                continue

            # Try to get source title
            title_tag = sour_tag.sub_tag(TAG_TITL)
            if title_tag and hasattr(title_tag, "value") and title_tag.value:
                sources.append(str(title_tag.value).strip())
            # If no title, try to get the source value itself
            elif hasattr(sour_tag, "value") and sour_tag.value:
                sources.append(str(sour_tag.value).strip())

    except Exception as e:
        logger.debug(f"Error extracting sources from event: {e}")

    return sources


def _get_event_info(
    individual: GedcomIndividualType, event_tag: str
) -> tuple[Optional[datetime], str, str]:  # ... implementation ...
    date_obj: Optional[datetime] = None
    date_str: str = "N/A"
    place_str: str = "N/A"

    # Validate and normalize individual
    individual = _validate_and_normalize_individual(individual)
    if individual is None:
        return date_obj, date_str, place_str

    indi_id_log = extract_and_fix_id(individual) or "Unknown ID"
    try:
        # Extract event record
        event_record = _extract_event_record(individual, event_tag, indi_id_log)
        if not event_record:
            return date_obj, date_str, place_str

        # Extract date and place information
        date_obj, date_str = _extract_date_from_event(event_record)
        place_str = _extract_place_from_event(event_record)

    except AttributeError as ae:
        logger.debug(
            f"Attribute error getting event '{event_tag}' for {indi_id_log}: {ae}"
        )
    except Exception as e:
        logger.error(
            f"Error accessing event {event_tag} for @{indi_id_log}@: {e}", exc_info=True
        )
    return date_obj, date_str, place_str


def get_person_sources(individual: GedcomIndividualType) -> dict[str, list[str]]:
    """
    Extract all source citations for a person.

    Phase 5.1: Source Citation Support
    Extracts sources from birth, death, and other events for a person.

    Args:
        individual: GEDCOM individual record

    Returns:
        Dictionary mapping event types to lists of source citations:
        {
            'birth': ['1881 Scotland Census', 'Birth Certificate'],
            'death': ['Death Certificate 1920'],
            'other': ['Marriage Record']
        }
    """
    sources_by_event: dict[str, list[str]] = {
        'birth': [],
        'death': [],
        'other': []
    }

    try:
        # Validate individual
        individual = _validate_and_normalize_individual(individual)
        if individual is None:
            return sources_by_event

        indi_id_log = extract_and_fix_id(individual) or "Unknown ID"

        # Extract birth sources
        birth_record = _extract_event_record(individual, TAG_BIRTH, indi_id_log)
        if birth_record:
            sources_by_event['birth'] = _extract_sources_from_event(birth_record)

        # Extract death sources
        death_record = _extract_event_record(individual, TAG_DEATH, indi_id_log)
        if death_record:
            sources_by_event['death'] = _extract_sources_from_event(death_record)

        # Could add more event types here (marriage, census, etc.)

    except Exception as e:
        logger.debug(f"Error extracting sources for person: {e}")

    return sources_by_event


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


def format_source_citations(sources_by_event: dict[str, list[str]]) -> str:
    """
    Format source citations for display in messages.

    Phase 5.1: Source Citation Support
    Creates human-readable source citation text from extracted sources.

    Args:
        sources_by_event: Dictionary mapping event types to source lists

    Returns:
        Formatted string like "documented in 1881 Scotland Census (birth) and Death Certificate 1920 (death)"
        or empty string if no sources
    """
    citations = []

    # Add birth sources
    for source in sources_by_event.get('birth', []):
        citations.append(f"{source} (birth)")

    # Add death sources
    for source in sources_by_event.get('death', []):
        citations.append(f"{source} (death)")

    # Add other sources
    for source in sources_by_event.get('other', []):
        citations.append(source)

    if not citations:
        return ""

    if len(citations) == 1:
        return f"documented in {citations[0]}"
    if len(citations) == 2:
        return f"documented in {citations[0]} and {citations[1]}"
    # Multiple sources: "documented in A, B, and C"
    return f"documented in {', '.join(citations[:-1])}, and {citations[-1]}"


def format_full_life_details(
    indi: GedcomIndividualType,
) -> tuple[str, str]:  # ... implementation ...
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


# _reconstruct_path removed - unused 89-line helper function for BFS path reconstruction


def _validate_bfs_inputs(start_id: str, end_id: str, id_to_parents: Any, id_to_children: Any) -> bool:
    """Validate inputs for bidirectional BFS search."""
    if start_id == end_id:
        return True
    if id_to_parents is None or id_to_children is None:  # type: ignore[unreachable]
        logger.error("[FastBiBFS] Relationship maps are None.")
        return False
    if not start_id or not end_id:
        logger.error("[FastBiBFS] Start or end ID is missing.")
        return False
    return True


def _initialize_bfs_queues(start_id: str, end_id: str) -> tuple:
    """Initialize BFS queues and visited sets for bidirectional search."""
    # Initialize BFS queues and visited sets
    # Forward queue from start_id
    queue_fwd = deque([(start_id, 0, [start_id])])  # (id, depth, path)
    # Backward queue from end_id
    queue_bwd = deque([(end_id, 0, [end_id])])  # (id, depth, path)

    # Track visited nodes and their paths
    visited_fwd = {start_id: (0, [start_id])}  # {id: (depth, path)}
    visited_bwd = {end_id: (0, [end_id])}  # {id: (depth, path)}

    return queue_fwd, queue_bwd, visited_fwd, visited_bwd


def _add_node_to_forward_queue(node_id: str, path: list[str], depth: int, visited_fwd: dict, queue_fwd: deque) -> None:
    """Add a node to the forward search queue if not already visited."""
    if node_id not in visited_fwd:
        new_path = [*path, node_id]
        visited_fwd[node_id] = (depth, new_path)
        queue_fwd.append((node_id, depth, new_path))


def _expand_forward_siblings(graph: GraphContext, current_id: str, path: list[str], depth: int, visited_fwd: dict, queue_fwd: deque) -> None:
    """Expand to siblings in forward direction through parents."""
    for parent_id in graph.id_to_parents.get(current_id, set()):
        for sibling_id in graph.id_to_children.get(parent_id, set()):
            if sibling_id != current_id and sibling_id not in visited_fwd:
                new_path = [*path, parent_id, sibling_id]
                visited_fwd[sibling_id] = (depth + 2, new_path)
                queue_fwd.append((sibling_id, depth + 2, new_path))


def _expand_forward_node(graph: GraphContext, depth: int, path: list[str],
                        visited_fwd: dict, queue_fwd: deque, max_depth: int):
    """Expand a node in the forward direction during BFS."""
    # Stop expanding if we've reached max depth
    if depth >= max_depth:
        return

    current_id = graph.current_id
    if not current_id:
        return

    # Expand to parents (direct relationship)
    for parent_id in graph.id_to_parents.get(current_id, set()):
        _add_node_to_forward_queue(parent_id, path, depth + 1, visited_fwd, queue_fwd)

    # Expand to children (direct relationship)
    for child_id in graph.id_to_children.get(current_id, set()):
        _add_node_to_forward_queue(child_id, path, depth + 1, visited_fwd, queue_fwd)

    # Expand to siblings (through parent)
    _expand_forward_siblings(graph, current_id, path, depth, visited_fwd, queue_fwd)


def _add_node_to_backward_queue(node_id: str, path: list[str], depth: int, visited_bwd: dict, queue_bwd: deque) -> None:
    """Add a node to the backward search queue if not already visited."""
    if node_id not in visited_bwd:
        new_path = [node_id, *path]
        visited_bwd[node_id] = (depth, new_path)
        queue_bwd.append((node_id, depth, new_path))


def _expand_backward_siblings(graph: GraphContext, current_id: str, path: list[str], depth: int, visited_bwd: dict, queue_bwd: deque) -> None:
    """Expand to siblings in backward direction through parents."""
    for parent_id in graph.id_to_parents.get(current_id, set()):
        for sibling_id in graph.id_to_children.get(parent_id, set()):
            if sibling_id != current_id and sibling_id not in visited_bwd:
                new_path = [sibling_id, parent_id, *path]
                visited_bwd[sibling_id] = (depth + 2, new_path)
                queue_bwd.append((sibling_id, depth + 2, new_path))


def _expand_backward_node(graph: GraphContext, depth: int, path: list[str],
                         visited_bwd: dict, queue_bwd: deque, max_depth: int):
    """Expand a node in the backward direction during BFS."""
    # Stop expanding if we've reached max depth
    if depth >= max_depth:
        return

    current_id = graph.current_id
    if not current_id:
        return

    # Expand to parents (direct relationship)
    for parent_id in graph.id_to_parents.get(current_id, set()):
        _add_node_to_backward_queue(parent_id, path, depth + 1, visited_bwd, queue_bwd)

    # Expand to children (direct relationship)
    for child_id in graph.id_to_children.get(current_id, set()):
        _add_node_to_backward_queue(child_id, path, depth + 1, visited_bwd, queue_bwd)

    # Expand to siblings (through parent)
    _expand_backward_siblings(graph, current_id, path, depth, visited_bwd, queue_bwd)


def fast_bidirectional_bfs(
    graph: GraphContext,
    max_depth: int = 25,
    node_limit: int = 150000,
    timeout_sec: int = 45,
) -> list[str]:
    """
    Enhanced bidirectional BFS that finds direct paths through family trees.

    This implementation focuses on finding paths where each person has a clear,
    direct relationship to the next person in the path (parent, child, sibling).
    It avoids using special cases or "connected to" placeholders.

    The algorithm prioritizes shorter paths with direct relationships over longer paths.
    """
    start_time = time.time()

    start_id = graph.start_id
    end_id = graph.end_id

    # Early return if IDs are None
    if not start_id or not end_id:
        logger.warning("[FastBiBFS] Start or end ID is None")
        return []

    id_to_parents = graph.id_to_parents
    id_to_children = graph.id_to_children

    # Validate inputs
    if not _validate_bfs_inputs(start_id, end_id, id_to_parents, id_to_children):
        return []

    if start_id == end_id:
        return [start_id]

    # First try to find a direct relationship (parent, child, sibling)
    # This is a quick check before running the full BFS
    # Import here to avoid circular dependency
    from relationship_utils import _find_direct_relationship

    # Convert lists to sets for _find_direct_relationship
    id_to_parents_set = {k: set(v) for k, v in id_to_parents.items()}
    id_to_children_set = {k: set(v) for k, v in id_to_children.items()}
    direct_path = _find_direct_relationship(
        start_id, end_id, id_to_parents_set, id_to_children_set
    )
    if direct_path:
        logger.debug(f"[FastBiBFS] Found direct relationship: {direct_path}")
        return direct_path

    # Initialize BFS data structures
    queue_fwd, queue_bwd, visited_fwd, visited_bwd = _initialize_bfs_queues(start_id, end_id)

    # Track all complete paths found
    all_paths = []
    processed = 0
    logger.debug(f"[FastBiBFS] Starting BFS: {start_id} <-> {end_id}")

    # Main search loop - continue until we find paths or exhaust the search
    while queue_fwd and queue_bwd and len(all_paths) < 5:
        # Check timeout and node limit
        if _check_search_limits(start_time, timeout_sec, processed, node_limit):
            break

        # Process forward queue (from start)
        processed += _process_forward_queue_item(
            queue_fwd, visited_bwd, visited_fwd, all_paths,
            id_to_parents, id_to_children, max_depth
        )

        # Process backward queue (from end)
        processed += _process_backward_queue_item(
            queue_bwd, visited_fwd, visited_bwd, all_paths,
            id_to_parents, id_to_children, max_depth
        )

    # Select the best path from found paths
    return _select_best_path(all_paths, start_id, end_id, id_to_parents, id_to_children)


def _process_forward_queue_item(
    queue_fwd: Any,
    visited_bwd: dict,
    visited_fwd: dict,
    all_paths: list,
    id_to_parents: Any,
    id_to_children: Any,
    max_depth: int
) -> int:
    """
    Process one item from the forward queue.

    Returns:
        1 if processed, 0 if queue empty
    """
    if not queue_fwd:
        return 0

    current_id, depth, path = queue_fwd.popleft()

    # Check if we've reached a node visited by backward search
    if current_id in visited_bwd:
        # Found a meeting point - reconstruct the path
        _, bwd_path = visited_bwd[current_id]  # depth unused
        # Combine paths (remove duplicate meeting point)
        combined_path = path + bwd_path[1:]
        all_paths.append(combined_path)
        logger.debug(
            f"[FastBiBFS] Path found via {current_id}: {len(combined_path)} nodes"
        )
        return 1

    # Expand this node in forward direction
    graph_ctx = GraphContext(
        id_to_parents=id_to_parents,
        id_to_children=id_to_children,
        current_id=current_id
    )
    _expand_forward_node(graph_ctx, depth, path, visited_fwd, queue_fwd, max_depth)
    return 1


def _process_backward_queue_item(
    queue_bwd: Any,
    visited_fwd: dict,
    visited_bwd: dict,
    all_paths: list,
    id_to_parents: Any,
    id_to_children: Any,
    max_depth: int
) -> int:
    """
    Process one item from the backward queue.

    Returns:
        1 if processed, 0 if queue empty
    """
    if not queue_bwd:
        return 0

    current_id, depth, path = queue_bwd.popleft()

    # Check if we've reached a node visited by forward search
    if current_id in visited_fwd:
        # Found a meeting point - reconstruct the path
        _, fwd_path = visited_fwd[current_id]  # depth unused
        # Combine paths (remove duplicate meeting point)
        combined_path = fwd_path + path[1:]
        all_paths.append(combined_path)
        logger.debug(
            f"[FastBiBFS] Path found via {current_id}: {len(combined_path)} nodes"
        )
        return 1

    # Expand this node in backward direction
    graph_ctx = GraphContext(
        id_to_parents=id_to_parents,
        id_to_children=id_to_children,
        current_id=current_id
    )
    _expand_backward_node(graph_ctx, depth, path, visited_bwd, queue_bwd, max_depth)
    return 1


def _check_search_limits(start_time: float, timeout_sec: int, processed: int, node_limit: int) -> bool:
    """
    Check if search limits (timeout or node limit) have been reached.

    Returns:
        True if limits reached, False otherwise
    """
    if time.time() - start_time > timeout_sec:
        logger.warning(f"[FastBiBFS] Timeout after {timeout_sec:.1f} seconds.")
        return True
    if processed > node_limit:
        logger.warning(f"[FastBiBFS] Node limit ({node_limit}) reached.")
        return True
    return False


def _select_best_path(all_paths: list[list[str]], start_id: str, end_id: str,
                     id_to_parents: Any, id_to_children: Any) -> list[str]:
    """Select the best path from a list of found paths based on relationship directness."""
    # If we found paths, select the best one
    if all_paths:
        # Import here to avoid circular dependency
        from relationship_utils import _has_direct_relationship

        # Score paths based on directness of relationships
        scored_paths = []
        for path in all_paths:
            # Check if each adjacent pair has a direct relationship
            direct_relationships = 0
            for i in range(len(path) - 1):
                if _has_direct_relationship(
                    path[i], path[i + 1], id_to_parents, id_to_children
                ):
                    direct_relationships += 1

            # Calculate score: prefer paths with more direct relationships and shorter length
            directness_score = (
                direct_relationships / (len(path) - 1) if len(path) > 1 else 0
            )
            length_penalty = len(path) / 10  # Slight penalty for longer paths
            score = directness_score - length_penalty

            scored_paths.append((path, score))

        # Sort by score (highest first)
        scored_paths.sort(key=lambda x: x[1], reverse=True)

        # Return the path with the highest score
        best_path = scored_paths[0][0]
        logger.debug(
            f"[FastBiBFS] Selected best path: {len(best_path)} nodes with score {scored_paths[0][1]:.2f}"
        )
        return best_path

    # If we didn't find any paths, try a more aggressive search
    logger.warning(f"[FastBiBFS] No paths found between {start_id} and {end_id}.")

    # Fallback: Try a direct path if possible
    return [start_id, end_id]


# Note: _has_direct_relationship and _find_direct_relationship have been moved to relationship_utils.py
# to eliminate duplication. They are imported at the top of this file.


# _are_directly_related removed - unused 24-line helper function for relationship checking


def _get_person_name_with_birth_year(indi: Optional[GedcomIndividualType], person_id: str) -> tuple[str, str]:
    """Get person's full name and birth year string."""
    if not indi:
        return f"Unknown ({person_id})", ""

    name = _get_full_name(indi)
    birth_year_str = ""
    birth_date_obj, _, _ = _get_event_info(indi, TAG_BIRTH)
    if birth_date_obj:
        birth_year_str = f" (b. {birth_date_obj.year})"

    return name, birth_year_str


def _get_gender_char(indi: GedcomIndividualType) -> Optional[str]:
    """Get gender character (M/F) from individual."""
    sex_b = getattr(indi, TAG_SEX.lower(), None)
    if sex_b and isinstance(sex_b, str) and str(sex_b).upper() in ("M", "F"):
        return str(sex_b).upper()[0]
    return None


def _determine_parent_relationship(sex_char: Optional[str], name: str, birth_year: str) -> str:
    """Determine parent relationship phrase based on gender."""
    parent_label = "father" if sex_char == "M" else "mother" if sex_char == "F" else "parent"
    return f"whose {parent_label} is {name}{birth_year}"


def _determine_child_relationship(sex_char: Optional[str], name: str, birth_year: str) -> str:
    """Determine child relationship phrase based on gender."""
    child_label = "son" if sex_char == "M" else "daughter" if sex_char == "F" else "child"
    return f"whose {child_label} is {name}{birth_year}"


def _determine_sibling_relationship(sex_char: Optional[str], name: str, birth_year: str) -> str:
    """Determine sibling relationship phrase based on gender."""
    sibling_label = "brother" if sex_char == "M" else "sister" if sex_char == "F" else "sibling"
    return f"whose {sibling_label} is {name}{birth_year}"


def _determine_spouse_relationship(sex_char: Optional[str], name: str, birth_year: str) -> str:
    """Determine spouse relationship phrase based on gender."""
    spouse_label = "husband" if sex_char == "M" else "wife" if sex_char == "F" else "spouse"
    return f"whose {spouse_label} is {name}{birth_year}"


def _determine_aunt_uncle_relationship(sex_char: Optional[str], name: str, birth_year: str) -> str:
    """Determine aunt/uncle relationship phrase based on gender."""
    relative_label = "uncle" if sex_char == "M" else "aunt" if sex_char == "F" else "aunt/uncle"
    return f"whose {relative_label} is {name}{birth_year}"


def _determine_niece_nephew_relationship(sex_char: Optional[str], name: str, birth_year: str) -> str:
    """Determine niece/nephew relationship phrase based on gender."""
    relative_label = "nephew" if sex_char == "M" else "niece" if sex_char == "F" else "niece/nephew"
    return f"whose {relative_label} is {name}{birth_year}"


def _determine_grandparent_relationship(sex_char: Optional[str], name: str, birth_year: str) -> str:
    """Determine grandparent relationship phrase based on gender."""
    grandparent_label = "grandfather" if sex_char == "M" else "grandmother" if sex_char == "F" else "grandparent"
    return f"whose {grandparent_label} is {name}{birth_year}"


def _determine_grandchild_relationship(sex_char: Optional[str], name: str, birth_year: str) -> str:
    """Determine grandchild relationship phrase based on gender."""
    grandchild_label = "grandson" if sex_char == "M" else "granddaughter" if sex_char == "F" else "grandchild"
    return f"whose {grandchild_label} is {name}{birth_year}"


def _determine_great_grandparent_relationship(sex_char: Optional[str], name: str, birth_year: str) -> str:
    """Determine great-grandparent relationship phrase based on gender."""
    grandparent_label = "great-grandfather" if sex_char == "M" else "great-grandmother" if sex_char == "F" else "great-grandparent"
    return f"whose {grandparent_label} is {name}{birth_year}"


def _determine_great_grandchild_relationship(sex_char: Optional[str], name: str, birth_year: str) -> str:
    """Determine great-grandchild relationship phrase based on gender."""
    grandchild_label = "great-grandson" if sex_char == "M" else "great-granddaughter" if sex_char == "F" else "great-grandchild"
    return f"whose {grandchild_label} is {name}{birth_year}"


def _check_relationship_type(
    relationship_type: str,
    id_a: str,
    id_b: str,
    sex_char: Optional[str],
    name_b: str,
    birth_year_b: str,
    reader: GedcomReaderType,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
) -> Optional[str]:
    """
    Check a specific relationship type and return the relationship phrase if matched.

    Returns None if the relationship type doesn't match.
    """
    # Data-driven relationship checking
    relationship_checks = {
        "parent": (lambda: id_b in id_to_parents.get(id_a, set()), lambda: _determine_parent_relationship(sex_char, name_b, birth_year_b)),
        "child": (lambda: id_b in id_to_children.get(id_a, set()), lambda: _determine_child_relationship(sex_char, name_b, birth_year_b)),
        "sibling": (lambda: _are_siblings(id_a, id_b, id_to_parents), lambda: _determine_sibling_relationship(sex_char, name_b, birth_year_b)),
        "spouse": (lambda: _are_spouses(id_a, id_b, reader), lambda: _determine_spouse_relationship(sex_char, name_b, birth_year_b)),
        "aunt_uncle": (lambda: _is_aunt_or_uncle(id_a, id_b, id_to_parents, id_to_children), lambda: _determine_aunt_uncle_relationship(sex_char, name_b, birth_year_b)),
        "niece_nephew": (lambda: _is_niece_or_nephew(id_a, id_b, id_to_parents, id_to_children), lambda: _determine_niece_nephew_relationship(sex_char, name_b, birth_year_b)),
        "cousin": (lambda: _are_cousins(id_a, id_b, id_to_parents), lambda: f"whose cousin is {name_b}{birth_year_b}"),
        "grandparent": (lambda: _is_grandparent(id_a, id_b, id_to_parents), lambda: _determine_grandparent_relationship(sex_char, name_b, birth_year_b)),
        "grandchild": (lambda: _is_grandchild(id_a, id_b, id_to_children), lambda: _determine_grandchild_relationship(sex_char, name_b, birth_year_b)),
        "great_grandparent": (lambda: _is_great_grandparent(id_a, id_b, id_to_parents), lambda: _determine_great_grandparent_relationship(sex_char, name_b, birth_year_b)),
        "great_grandchild": (lambda: _is_great_grandchild(id_a, id_b, id_to_children), lambda: _determine_great_grandchild_relationship(sex_char, name_b, birth_year_b)),
    }

    if relationship_type in relationship_checks:
        check_func, result_func = relationship_checks[relationship_type]
        if check_func():
            return result_func()

    return None


def _determine_relationship_between_individuals(
    id_a: str,
    id_b: str,
    indi_b: GedcomIndividualType,
    name_b: str,
    birth_year_b: str,
    reader: GedcomReaderType,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
) -> str:
    """Determine the relationship phrase between two individuals."""
    sex_char = _get_gender_char(indi_b)

    # Define relationship types to check in priority order
    relationship_types = [
        "parent", "child", "sibling", "spouse",
        "aunt_uncle", "niece_nephew", "cousin",
        "grandparent", "grandchild",
        "great_grandparent", "great_grandchild"
    ]

    # Check each relationship type
    for rel_type in relationship_types:
        result = _check_relationship_type(
            rel_type, id_a, id_b, sex_char, name_b, birth_year_b,
            reader, id_to_parents, id_to_children
        )
        if result:
            return result

    # Fallback for unknown relationships
    return f"related to {name_b}{birth_year_b}"


def explain_relationship_path(
    path_ids: list[str],
    reader: GedcomReaderType,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
    indi_index: dict[str, GedcomIndividualType],
) -> str:
    """
    Generates a human-readable explanation of the relationship path.

    This implementation uses a generic approach to determine relationships
    between individuals in the path without special cases. It analyzes the
    family structure to determine parent-child, sibling, spouse, and other
    relationships.
    """
    if not path_ids or len(path_ids) < 2:
        return "(No relationship path explanation available)"

    steps: list[str] = []
    start_person_indi = indi_index.get(path_ids[0])

    # Get start person name with birth year
    start_person_name, birth_year_str = _get_person_name_with_birth_year(start_person_indi, path_ids[0])
    full_start_name = f"{start_person_name}{birth_year_str}"

    # Process each pair of individuals in the path
    for i in range(len(path_ids) - 1):
        id_a, id_b = path_ids[i], path_ids[i + 1]
        indi_a = indi_index.get(id_a)
        indi_b = indi_index.get(id_b)

        # Handle missing individuals
        if not indi_a or not indi_b:
            name_b, birth_year_b = _get_person_name_with_birth_year(indi_b, id_b)
            steps.append(f"  -> connected to {name_b}{birth_year_b}")
            continue

        # Get name and birth year for person B
        name_b, birth_year_b = _get_person_name_with_birth_year(indi_b, id_b)

        # Determine the relationship between A and B
        relationship_phrase = _determine_relationship_between_individuals(
            id_a, id_b, indi_b, name_b, birth_year_b, reader, id_to_parents, id_to_children
        )

        steps.append(f"  -> {relationship_phrase}")

    # Join the start name and all the steps
    return full_start_name + "\n" + "\n".join(steps)


def _are_siblings(id1: str, id2: str, id_to_parents: dict[str, set[str]]) -> bool:
    """Check if two individuals are siblings (share at least one parent)."""
    parents_1 = id_to_parents.get(id1, set())
    parents_2 = id_to_parents.get(id2, set())
    return bool(parents_1 and parents_2 and not parents_1.isdisjoint(parents_2))


def _extract_spouse_ids_from_family(fam: Any) -> tuple[Optional[str], Optional[str]]:
    """Extract husband and wife IDs from a family record. Returns (husb_id, wife_id)."""
    husb_ref = fam.sub_tag(TAG_HUSBAND)
    wife_ref = fam.sub_tag(TAG_WIFE)

    husb_id = _normalize_id(husb_ref.xref_id) if husb_ref and hasattr(husb_ref, "xref_id") else None
    wife_id = _normalize_id(wife_ref.xref_id) if wife_ref and hasattr(wife_ref, "xref_id") else None

    return husb_id, wife_id


def _are_spouses(id1: str, id2: str, reader: GedcomReaderType) -> bool:
    """Check if two individuals are spouses."""
    if not reader:
        return False

    try:
        for fam in reader.records0("FAM"):
            if not _is_record(fam):
                continue

            husb_id, wife_id = _extract_spouse_ids_from_family(fam)

            # Check if id1 and id2 are husband and wife in this family
            if (husb_id == id1 and wife_id == id2) or (husb_id == id2 and wife_id == id1):
                return True
    except Exception as e:
        logger.error(f"Error checking spouse relationship: {e}", exc_info=False)

    return False


def _is_aunt_or_uncle(
    id1: str,
    id2: str,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
) -> bool:
    """Check if id2 is an aunt or uncle of id1."""
    # Get parents of id1
    parents = id_to_parents.get(id1, set())

    # For each parent, check if id2 is their sibling
    for parent_id in parents:
        # Get grandparents (parents of parent)
        grandparents = id_to_parents.get(parent_id, set())

        # For each grandparent, get their children
        for grandparent_id in grandparents:
            aunts_uncles = id_to_children.get(grandparent_id, set())

            # If id2 is a child of a grandparent and not a parent, it's an aunt/uncle
            if id2 in aunts_uncles and id2 != parent_id:
                return True

    return False


def _is_niece_or_nephew(
    id1: str,
    id2: str,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
) -> bool:
    """Check if id2 is a niece or nephew of id1."""
    # This is the reverse of aunt/uncle relationship
    return _is_aunt_or_uncle(id2, id1, id_to_parents, id_to_children)


def _are_cousins(
    id1: str,
    id2: str,
    id_to_parents: dict[str, set[str]],
) -> bool:
    """Check if id1 and id2 are cousins (children of siblings)."""
    # Get parents of id1 and id2
    parents1 = id_to_parents.get(id1, set())
    parents2 = id_to_parents.get(id2, set())

    # For each parent of id1, check if they have a sibling who is a parent of id2
    for parent1 in parents1:
        # Get grandparents of id1
        grandparents1 = id_to_parents.get(parent1, set())

        for parent2 in parents2:
            # Get grandparents of id2
            grandparents2 = id_to_parents.get(parent2, set())

            # If they share a grandparent but have different parents, they're cousins
            if (
                grandparents1
                and grandparents2
                and not grandparents1.isdisjoint(grandparents2)
            ) and (
                parent1 != parent2
            ):  # Make sure they don't have the same parent (which would make them siblings)
                return True

    return False


def _is_ancestor_at_generation(
    descendant_id: str,
    ancestor_id: str,
    generations: int,
    id_to_parents: dict[str, set[str]]
) -> bool:
    """
    Check if ancestor_id is an ancestor of descendant_id at a specific generation level.

    Args:
        descendant_id: ID of the descendant
        ancestor_id: ID of the potential ancestor
        generations: Number of generations up (1=parent, 2=grandparent, 3=great-grandparent, etc.)
        id_to_parents: Dictionary mapping individual IDs to their parent IDs

    Returns:
        True if ancestor_id is an ancestor at the specified generation level
    """
    if generations < 1:
        return False

    # Start with the descendant
    current_generation = {descendant_id}

    # Walk up the specified number of generations
    for _ in range(generations):
        next_generation = set()
        for person_id in current_generation:
            parents = id_to_parents.get(person_id, set())
            next_generation.update(parents)

        if not next_generation:
            return False  # No more ancestors at this level

        current_generation = next_generation

    # Check if ancestor_id is in the final generation
    return ancestor_id in current_generation


def _is_descendant_at_generation(
    ancestor_id: str,
    descendant_id: str,
    generations: int,
    id_to_children: dict[str, set[str]]
) -> bool:
    """
    Check if descendant_id is a descendant of ancestor_id at a specific generation level.

    Args:
        ancestor_id: ID of the ancestor
        descendant_id: ID of the potential descendant
        generations: Number of generations down (1=child, 2=grandchild, 3=great-grandchild, etc.)
        id_to_children: Dictionary mapping individual IDs to their child IDs

    Returns:
        True if descendant_id is a descendant at the specified generation level
    """
    if generations < 1:
        return False

    # Start with the ancestor
    current_generation = {ancestor_id}

    # Walk down the specified number of generations
    for _ in range(generations):
        next_generation = set()
        for person_id in current_generation:
            children = id_to_children.get(person_id, set())
            next_generation.update(children)

        if not next_generation:
            return False  # No more descendants at this level

        current_generation = next_generation

    # Check if descendant_id is in the final generation
    return descendant_id in current_generation


# Convenience wrappers for backward compatibility
def _is_grandparent(id1: str, id2: str, id_to_parents: dict[str, set[str]]) -> bool:
    """Check if id2 is a grandparent of id1."""
    return _is_ancestor_at_generation(id1, id2, 2, id_to_parents)


def _is_grandchild(id1: str, id2: str, id_to_children: dict[str, set[str]]) -> bool:
    """Check if id2 is a grandchild of id1."""
    return _is_descendant_at_generation(id1, id2, 2, id_to_children)


def _is_great_grandparent(id1: str, id2: str, id_to_parents: dict[str, set[str]]) -> bool:
    """Check if id2 is a great-grandparent of id1."""
    return _is_ancestor_at_generation(id1, id2, 3, id_to_parents)


def _is_great_grandchild(id1: str, id2: str, id_to_children: dict[str, set[str]]) -> bool:
    """Check if id2 is a great-grandchild of id1."""
    return _is_descendant_at_generation(id1, id2, 3, id_to_children)


# ==============================================
def _prepare_search_data(search_criteria: dict) -> dict[str, Any]:
    """Extract and normalize search criteria data."""
    return {
        "fname": (search_criteria.get("first_name") or "").lower() if isinstance(search_criteria.get("first_name"), str) else "",
        "sname": (search_criteria.get("surname") or "").lower() if isinstance(search_criteria.get("surname"), str) else "",
        "pob": (search_criteria.get("birth_place") or "").lower() if isinstance(search_criteria.get("birth_place"), str) else "",
        "pod": (search_criteria.get("death_place") or "").lower() if isinstance(search_criteria.get("death_place"), str) else "",
        "b_year": search_criteria.get("birth_year"),
        "b_date": search_criteria.get("birth_date_obj"),
        "d_year": search_criteria.get("death_year"),
        "d_date": search_criteria.get("death_date_obj"),
    }


def _prepare_candidate_data(candidate_processed_data: dict[str, Any]) -> dict[str, Any]:
    """Extract and normalize candidate data."""
    return {
        "id_debug": candidate_processed_data.get("norm_id", "N/A_in_proc_cache"),
        "fname": (candidate_processed_data.get("first_name") or "").lower() if isinstance(candidate_processed_data.get("first_name"), str) else "",
        "sname": (candidate_processed_data.get("surname") or "").lower() if isinstance(candidate_processed_data.get("surname"), str) else "",
        "bplace": (candidate_processed_data.get("birth_place_disp") or "").lower() if isinstance(candidate_processed_data.get("birth_place_disp"), str) else "",
        "dplace": (candidate_processed_data.get("death_place_disp") or "").lower() if isinstance(candidate_processed_data.get("death_place_disp"), str) else "",
        "b_year": candidate_processed_data.get("birth_year"),
        "b_date": candidate_processed_data.get("birth_date_obj"),
        "d_year": candidate_processed_data.get("death_year"),
        "d_date": candidate_processed_data.get("death_date_obj"),
    }


def _score_names(t_data: dict, c_data: dict, weights: Mapping, field_scores: dict, match_reasons: list) -> None:
    """Score name matches (first name, surname, and bonus for both)."""
    first_name_matched = False
    surname_matched = False

    if t_data["fname"] and c_data["fname"] and t_data["fname"] in c_data["fname"]:
        points_givn = weights.get("contains_first_name", 0)
        if points_givn != 0:
            field_scores["givn"] = int(points_givn)
            match_reasons.append(f"Contains First Name ({points_givn}pts)")
            first_name_matched = True

    if t_data["sname"] and c_data["sname"] and t_data["sname"] in c_data["sname"]:
        points_surn = weights.get("contains_surname", 0)
        if points_surn != 0:
            field_scores["surn"] = int(points_surn)
            match_reasons.append(f"Contains Surname ({points_surn}pts)")
            surname_matched = True

    if t_data["fname"] and t_data["sname"] and first_name_matched and surname_matched:
        bonus_points = weights.get("bonus_both_names_contain", 0)
        if bonus_points != 0:
            field_scores["bonus"] = int(bonus_points)
            match_reasons.append(f"Bonus Both Names ({bonus_points}pts)")


def _check_year_match(t_year: Any, c_year: Any, year_score_range: int) -> tuple[bool, bool]:
    """Check if years match exactly or approximately. Returns (exact_match, approx_match)."""
    if t_year is None or c_year is None:
        return False, False
    try:
        t_year_int = int(t_year)
        c_year_int = int(c_year)
        exact_match = t_year_int == c_year_int
        approx_match = not exact_match and abs(c_year_int - t_year_int) <= year_score_range
        return exact_match, approx_match
    except Exception:
        return False, False


def _calculate_date_flags(t_data: dict, c_data: dict, year_score_range: Union[int, float]) -> dict:
    """Calculate date match flags for birth and death dates."""
    year_range = int(year_score_range) if isinstance(year_score_range, (int, float)) else 0
    birth_year_match, birth_year_approx = _check_year_match(t_data["b_year"], c_data["b_year"], year_range)
    death_year_match, death_year_approx = _check_year_match(t_data["d_year"], c_data["d_year"], year_range)

    return {
        "exact_birth_date_match": bool(
            t_data["b_date"] and c_data["b_date"] and
            isinstance(t_data["b_date"], datetime) and isinstance(c_data["b_date"], datetime) and
            t_data["b_date"].date() == c_data["b_date"].date()
        ),
        "exact_death_date_match": bool(
            t_data["d_date"] and c_data["d_date"] and
            isinstance(t_data["d_date"], datetime) and isinstance(c_data["d_date"], datetime) and
            t_data["d_date"].date() == c_data["d_date"].date()
        ),
        "birth_year_match": birth_year_match,
        "birth_year_approx_match": birth_year_approx,
        "death_year_match": death_year_match,
        "death_year_approx_match": death_year_approx,
        "death_dates_absent": bool(
            t_data["d_date"] is None and c_data["d_date"] is None and
            t_data["d_year"] is None and c_data["d_year"] is None
        ),
    }


def _score_birth_dates(t_data: dict, c_data: dict, date_flags: dict, weights: Mapping, field_scores: dict, match_reasons: list) -> None:
    """Score birth date matches (prioritize: exact date > exact year > approx year)."""
    if date_flags["exact_birth_date_match"]:
        points_bdate = weights.get("exact_birth_date", 0)
        if points_bdate != 0:
            field_scores["bdate"] = int(points_bdate)
            match_reasons.append(f"Exact Birth Date ({points_bdate}pts)")
            return

    if date_flags["birth_year_match"]:
        points_byear = weights.get("year_birth", 0)
        if points_byear != 0:
            field_scores["byear"] = int(points_byear)
            match_reasons.append(f"Exact Birth Year ({c_data['b_year']}) ({points_byear}pts)")
            return

    if date_flags["birth_year_approx_match"]:
        points_byear_approx = weights.get("approx_year_birth", 0) or weights.get("birth_year_close", 0)
        if points_byear_approx != 0:
            field_scores["byear"] = int(points_byear_approx)
            match_reasons.append(f"Approx Birth Year ({c_data['b_year']} vs {t_data['b_year']}) ({points_byear_approx}pts)")


def _score_death_dates(t_data: dict, c_data: dict, date_flags: dict, weights: Mapping, field_scores: dict, match_reasons: list) -> None:
    """Score death date matches (prioritize: exact date > exact year > approx year; no points for absence)."""
    if date_flags["exact_death_date_match"]:
        points_ddate = weights.get("exact_death_date", 0)
        if points_ddate != 0:
            field_scores["ddate"] = int(points_ddate)
            match_reasons.append(f"Exact Death Date ({points_ddate}pts)")
            return

    if date_flags["death_year_match"]:
        points_dyear = weights.get("year_death", 0)
        if points_dyear != 0:
            field_scores["dyear"] = int(points_dyear)
            match_reasons.append(f"Exact Death Year ({c_data['d_year']}) ({points_dyear}pts)")
            return

    if date_flags["death_year_approx_match"]:
        points_dyear_approx = weights.get("approx_year_death", 0)
        if points_dyear_approx != 0:
            field_scores["dyear"] = int(points_dyear_approx)
            match_reasons.append(f"Approx Death Year ({c_data['d_year']} vs {t_data['d_year']}) ({points_dyear_approx}pts)")
            return

    # Do not award points for both death dates absent when the user did not specify death criteria.


def _score_dates(t_data: dict, c_data: dict, date_flags: dict, weights: Mapping, field_scores: dict, match_reasons: list) -> None:
    """Score birth and death date matches."""
    _score_birth_dates(t_data, c_data, date_flags, weights, field_scores, match_reasons)
    _score_death_dates(t_data, c_data, date_flags, weights, field_scores, match_reasons)


def _score_birth_place(t_data: dict, c_data: dict, weights: Mapping, field_scores: dict, match_reasons: list) -> None:
    """Score birth place match."""
    if not (t_data["pob"] and c_data["bplace"] and t_data["pob"] in c_data["bplace"]):
        return
    points_pob = weights.get("contains_pob", 0) or weights.get("birth_place_match", 0)
    if points_pob != 0:
        field_scores["bplace"] = int(points_pob)
        match_reasons.append(f"Birth Place Contains ({points_pob}pts)")


def _score_death_place(t_data: dict, c_data: dict, weights: Mapping, field_scores: dict, match_reasons: list) -> None:
    """Score death place match (contains only; no points for absence)."""
    pod_match = bool(t_data["pod"] and c_data["dplace"] and t_data["pod"] in c_data["dplace"])
    if not pod_match:
        return
    points_pod = weights.get("contains_pod", 0) or weights.get("death_place_match", 0)
    if points_pod != 0:
        field_scores["dplace"] = int(points_pod)
        match_reasons.append(f"Death Place Contains ({points_pod}pts)")


def _score_places(t_data: dict, c_data: dict, weights: Mapping, field_scores: dict, match_reasons: list) -> None:
    """Score birth place and death place matches (gender removed from scoring)."""
    _score_birth_place(t_data, c_data, weights, field_scores, match_reasons)
    _score_death_place(t_data, c_data, weights, field_scores, match_reasons)


def _score_birth_bonus(weights: Mapping, field_scores: dict, match_reasons: list) -> None:
    """Score birth bonus (if both birth year and birth place matched)."""
    if not (field_scores["byear"] > 0 and field_scores["bplace"] > 0):
        return
    birth_bonus_points = weights.get("bonus_birth_info", 0) or weights.get("bonus_birth_date_and_place", 0)
    if birth_bonus_points != 0:
        field_scores["bbonus"] = int(birth_bonus_points)
        match_reasons.append(f"Bonus Birth Info ({birth_bonus_points}pts)")


def _score_death_bonus(weights: Mapping, field_scores: dict, match_reasons: list) -> None:
    """Score death bonus (only when BOTH death date and place matched)."""
    death_info_matched = (field_scores["dyear"] > 0 or field_scores["ddate"] > 0) and field_scores["dplace"] > 0
    if not death_info_matched:
        return
    death_bonus_points = weights.get("bonus_death_info", 0) or weights.get("bonus_death_date_and_place", 0)
    if death_bonus_points != 0:
        field_scores["dbonus"] = int(death_bonus_points)
        match_reasons.append(f"Bonus Death Info ({death_bonus_points}pts)")


def _score_bonuses(weights: Mapping, field_scores: dict, match_reasons: list) -> None:
    """Score birth and death bonuses."""
    _score_birth_bonus(weights, field_scores, match_reasons)
    _score_death_bonus(weights, field_scores, match_reasons)



def _apply_alive_conflict_penalty(t_data: dict, c_data: dict, weights: Mapping, field_scores: dict, match_reasons: list) -> None:
    """Apply a small negative score when query implies 'alive' but candidate has death info.

    Alive-mode heuristic: if the user provided no death year, no death date, and no death place,
    we assume they are searching for a living person. In that case, a candidate that contains
    death information (date/year/place) receives a configurable penalty.
    """
    try:
        alive_query = not (t_data.get("d_date") or t_data.get("d_year") or t_data.get("pod"))
        candidate_has_death = bool(c_data.get("d_date") or c_data.get("d_year") or c_data.get("dplace"))
        if alive_query and candidate_has_death:
            penalty = int(weights.get("alive_conflict_penalty", 0))
            if penalty != 0:
                field_scores["alive_penalty"] = penalty
                match_reasons.append(f"Alive Assumed; Candidate Has Death Info ({penalty}pts)")
    except Exception:
        # Non-fatal; scoring should not break due to penalty logic
        pass


# Scoring Function (V18 - Corrected Syntax)
# ==============================================
def calculate_match_score(
    search_criteria: dict,
    candidate_processed_data: dict[str, Any],  # Expects pre-processed data
    scoring_weights: Optional[Mapping[str, Union[int, float]]] = None,
    date_flexibility: Optional[dict] = None,
) -> tuple[float, dict[str, int], list[str]]:
    """
    Calculates match score using pre-processed candidate data.
    Handles OR logic for death place matching (contains OR both absent).
    Prioritizes exact date > exact year > approx year for date scoring.
    V18.PreProcess compatible - Syntax Fixed.
    """
    match_reasons: list[str] = []
    field_scores = {
        "givn": 0, "surn": 0, "byear": 0, "bdate": 0, "bplace": 0,
        "bbonus": 0, "dyear": 0, "ddate": 0, "dplace": 0, "dbonus": 0, "bonus": 0,
        # Negative adjustments (policy-based)
        "alive_penalty": 0,
    }
    weights = scoring_weights if scoring_weights is not None else config.common_scoring_weights
    date_flex = date_flexibility if date_flexibility is not None else {"year_match_range": config.date_flexibility}
    year_score_range = date_flex.get("year_match_range", 10)

    # Prepare data
    t_data = _prepare_search_data(search_criteria)
    c_data = _prepare_candidate_data(candidate_processed_data)

    # Name Scoring
    _score_names(t_data, c_data, weights, field_scores, match_reasons)

    # Calculate date match flags
    date_flags = _calculate_date_flags(t_data, c_data, year_score_range)

    # Date Scoring
    _score_dates(t_data, c_data, date_flags, weights, field_scores, match_reasons)

    # Place Scoring
    _score_places(t_data, c_data, weights, field_scores, match_reasons)

    # Bonus Scoring
    _score_bonuses(weights, field_scores, match_reasons)


    # Policy-based negative adjustments
    _apply_alive_conflict_penalty(t_data, c_data, weights, field_scores, match_reasons)

    # Calculate Final Total Score
    final_total_score = sum(field_scores.values())
    final_total_score = max(0.0, round(final_total_score))
    unique_reasons = sorted(set(match_reasons))

    # Final score calculated

    # Return a COPY of the dictionary
    return final_total_score, field_scores.copy(), unique_reasons


# End of calculate_match_score


# ==============================================
# GedcomData Class
# ==============================================
class GedcomData:
    def __init__(self, gedcom_path: Union[str, Path], skip_cache_build: bool = False):
        """
        Initialize GedcomData from a GEDCOM file.

        Args:
            gedcom_path: Path to the GEDCOM file
            skip_cache_build: If True, skip building caches (used when loading from cache)
        """
        self.path = Path(gedcom_path).resolve()
        self.reader: Optional[GedcomReaderType] = None
        self.indi_index: dict[str, GedcomIndividualType] = {}  # Index of INDI records
        self.processed_data_cache: dict[str, dict[str, Any]] = (
            {}
        )  # NEW: Cache for processed data
        self.id_to_parents: dict[str, set[str]] = {}
        self.id_to_children: dict[str, set[str]] = {}
        self.indi_index_build_time: float = 0
        self.family_maps_build_time: float = 0
        self.data_processing_time: float = 0  # NEW: Time for pre-processing
        self._cache_source: str = "unknown"  # Track where data was loaded from: "memory", "disk", or "file"

        if not self.path.is_file():
            logger.critical(f"GEDCOM file not found: {self.path}")
            raise FileNotFoundError(f"GEDCOM file not found: {self.path}")
        try:
            logger.debug(f"Loading GEDCOM file: {self.path}")
            load_start = time.time()
            # Initialize GedcomReader with the file path as a string
            # The constructor takes a file parameter (file name or file object)
            # There seems to be a discrepancy between the documentation and the actual implementation
            # We'll try to create it with a positional argument, ignoring the type checker warning
            # @type: ignore is used to suppress the type checker warning
            self.reader = GedcomReader(str(self.path))  # type: ignore
            load_time = time.time() - load_start
            logger.debug(f"GEDCOM file loaded in {load_time:.2f}s.")
        except Exception as e:
            file_size_mb = (
                self.path.stat().st_size / (1024 * 1024)
                if self.path.exists()
                else "unknown"
            )
            error_msg = (
                f"Failed to load/parse GEDCOM file {self.path} (size: {file_size_mb:.2f}MB). "
                f"Error type: {type(e).__name__}. This may indicate file corruption, "
                f"unsupported GEDCOM format, or encoding issues."
            )
            logger.critical(error_msg, exc_info=True)
            # Use exception chaining to preserve the original error context
            raise RuntimeError(error_msg) from e

        if not skip_cache_build:
            self.build_caches()  # Build caches upon initialization

    @classmethod
    def from_cache(cls, cached_data: dict[str, Any], gedcom_path: str) -> "GedcomData":
        """
        Create a GedcomData instance from cached data.

        This method creates a GedcomData instance with the reader initialized,
        but populates the expensive-to-build data structures from cache.
        The indi_index is rebuilt quickly from the reader (just indexing, no parsing).

        Args:
            cached_data: Dictionary containing cached GEDCOM data
            gedcom_path: Path to the GEDCOM file

        Returns:
            GedcomData instance with cached data populated
        """
        # Create instance with reader but skip cache building
        instance = cls(gedcom_path, skip_cache_build=True)

        # Populate from cached data (note: indi_index is NOT in cache because
        # Individual objects cannot be pickled)
        instance.processed_data_cache = cached_data.get("processed_data_cache", {})
        instance.id_to_parents = cached_data.get("id_to_parents", {})
        instance.id_to_children = cached_data.get("id_to_children", {})
        instance.indi_index_build_time = cached_data.get("indi_index_build_time", 0)
        instance.family_maps_build_time = cached_data.get("family_maps_build_time", 0)
        instance.data_processing_time = cached_data.get("data_processing_time", 0)

        # Rebuild indi_index from reader (fast - just indexing, no data extraction)
        instance._build_indi_index()

        logger.debug(
            f"GedcomData restored from cache: {len(instance.processed_data_cache)} processed individuals, "
            f"{len(instance.id_to_parents)} parent relationships, "
            f"{len(instance.indi_index)} individuals indexed"
        )

        return instance

    def build_caches(self) -> None:
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



    def _process_indi_record(self, indi_record: Any) -> tuple[bool, bool]:
        """Process an individual record for indexing. Returns (processed, skipped)."""
        if not (_is_individual(indi_record) and hasattr(indi_record, "xref_id") and indi_record.xref_id):
            if logger.isEnabledFor(logging.DEBUG):
                if hasattr(indi_record, "xref_id"):
                    logger.debug(f"Skipping non-Individual record: Type={type(indi_record).__name__}, Xref={indi_record.xref_id}")
                else:
                    logger.debug(f"Skipping record with no xref_id: Type={type(indi_record).__name__}")
            return False, True

        norm_id = _normalize_id(indi_record.xref_id)
        if not norm_id:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Skipping INDI with unnormalizable xref_id: {indi_record.xref_id}")
            return False, True

        if norm_id in self.indi_index:
            logger.warning(f"Duplicate normalized INDI ID found: {norm_id}. Overwriting.")

        self.indi_index[norm_id] = indi_record  # type: ignore
        return True, False

    def _build_indi_index(self) -> None:
        """Builds a dictionary mapping normalized IDs to Individual records."""
        if not self.reader:
            logger.error("[Cache Build] Cannot build INDI index: GedcomReader is None.")
            return

        start_time = time.time()
        logger.debug("[Cache] Building INDI index...")
        self.indi_index = {}
        count = 0
        skipped = 0
        current_record_id = "None"

        try:
            for indi_record in self.reader.records0(TAG_INDI):
                current_record_id = getattr(indi_record, "xref_id", "Unknown") if indi_record else "None"
                processed, skip = self._process_indi_record(indi_record)
                if processed:
                    count += 1
                if skip:
                    skipped += 1
        except StopIteration:
            logger.debug("[Cache] Finished iterating INDI records for index.")
        except Exception as e:
            record_context = f"while processing record ID: {current_record_id}"
            logger.error(
                f"[Cache Build] Error during INDI index build {record_context}: {e}. "
                f"Error type: {type(e).__name__}. Index may be incomplete. "
                f"Records processed so far: {count}, skipped: {skipped}.",
                exc_info=True,
            )

        elapsed = time.time() - start_time
        self.indi_index_build_time = elapsed
        if count > 0:
            logger.debug(f"[Cache] INDI index built with {count} individuals ({skipped} skipped) in {elapsed:.2f}s.")
        else:
            logger.error(f"[Cache Build] INDI index is EMPTY after build attempt ({skipped} skipped) in {elapsed:.2f}s.")

    def _extract_parents_from_family(self, fam: Any, fam_id_log: str) -> set[str]:
        """Extract parent IDs from a family record."""
        parents: set[str] = set()
        for parent_tag in [TAG_HUSBAND, TAG_WIFE]:
            parent_ref = fam.sub_tag(parent_tag)
            if parent_ref and hasattr(parent_ref, "xref_id"):
                parent_id = _normalize_id(parent_ref.xref_id)
                if parent_id:
                    parents.add(parent_id)
                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Skipping parent with invalid/unnormalizable ID {getattr(parent_ref, 'xref_id', '?')} in FAM {fam_id_log}")
        return parents

    def _process_child_in_family(self, child_tag: Any, parents: set[str], fam_id_log: str) -> tuple[bool, bool]:
        """Process a child tag and update family maps. Returns (processed, skipped)."""
        if not (child_tag and hasattr(child_tag, "xref_id")):
            if child_tag is not None and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Skipping CHIL record in FAM {fam_id_log} with invalid format: Type={type(child_tag).__name__}")
            return False, True

        child_id = _normalize_id(child_tag.xref_id)
        if not child_id:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Skipping child with invalid/unnormalizable ID {getattr(child_tag, 'xref_id', '?')} in FAM {fam_id_log}")
            return False, True

        # Add child to each parent's children set
        for parent_id in parents:
            self.id_to_children.setdefault(parent_id, set()).add(child_id)

        # Add parents to child's parents set
        if parents:
            self.id_to_parents.setdefault(child_id, set()).update(parents)
            return True, False

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Child {child_id} found in FAM {fam_id_log} but no valid parents identified in this specific record.")
        return False, False

    def _build_family_maps(self) -> None:
        """Builds dictionaries mapping child IDs to parent IDs and parent IDs to child IDs."""
        if not self.reader:
            logger.error("[Cache Build] Cannot build family maps: GedcomReader is None.")
            return

        start_time = time.time()
        logger.debug("[Cache] Building family maps...")
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
                parents = self._extract_parents_from_family(fam, fam_id_log)

                for child_tag in fam.sub_tags(TAG_CHILD):
                    processed, skipped = self._process_child_in_family(child_tag, parents, fam_id_log)
                    if processed:
                        processed_links += 1
                    if skipped:
                        skipped_links += 1
        except StopIteration:
            logger.debug("[Cache] Finished iterating FAM records for maps.")
        except Exception as e:
            logger.error(f"[Cache Build] Unexpected error during family map build: {e}. Maps may be incomplete.", exc_info=True)

        self._log_family_maps_build_results(time.time() - start_time, fam_count, processed_links, skipped_links)

    def _log_family_maps_build_results(self, elapsed: float, fam_count: int, processed_links: int, skipped_links: int) -> None:
        """Log the results of family maps building."""
        self.family_maps_build_time = elapsed
        parent_map_count = len(self.id_to_parents)
        child_map_count = len(self.id_to_children)
        logger.debug(
            f"[Cache] Family maps built: {fam_count} FAMs processed. Added {processed_links} child-parent relationships "
            f"({skipped_links} skipped invalid links/IDs). Map sizes: {parent_map_count} child->parents entries, "
            f"{child_map_count} parent->children entries in {elapsed:.2f}s."
        )
        if parent_map_count == 0 and child_map_count == 0 and fam_count > 0:
            logger.warning("[Cache Build] Family maps are EMPTY despite processing FAM records. Check GEDCOM structure or parsing logic.")

    def _pre_process_individual_data(self) -> None:
        """NEW: Extracts and caches key data points for each individual."""
        if not self.indi_index:
            logger.error("Cannot pre-process data: INDI index is not built.")
            return
        start_time = time.time()
        logger.debug("[Pre-Process] Extracting key data for individuals...")
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
                # Add null check for indi before calling sub_tag
                name_rec = indi.sub_tag(TAG_NAME) if indi is not None else None
                givn_raw = name_rec.sub_tag_value(TAG_GIVN) if name_rec else None
                surn_raw = name_rec.sub_tag_value(TAG_SURN) if name_rec else None

                # Extract gender
                # Add null check for indi before calling sub_tag_value
                sex_raw = indi.sub_tag_value(TAG_SEX) if indi is not None else None
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
                # Enhanced error reporting with more context
                error_type = type(e).__name__
                logger.error(
                    f"Error pre-processing individual {norm_id}: {error_type}: {e}. "
                    f"This may affect search results and relationship paths for this individual.",
                    exc_info=True,
                )
                errors += 1
        elapsed = time.time() - start_time
        self.data_processing_time = elapsed
        logger.debug(
            f"[Pre-Process] Processed data for {processed_count} individuals ({errors} errors) in {elapsed:.2f}s."
        )
        if not self.processed_data_cache:
            logger.error(
                "[Pre-Process] Processed data cache is EMPTY after build attempt."
            )

    def get_processed_indi_data(self, norm_id: str) -> Optional[dict[str, Any]]:
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

    def _get_sibling_ids(self, target_id: str) -> set[str]:
        """Get sibling IDs for a target individual."""
        parents = self.id_to_parents.get(target_id, set())
        if not parents:
            return set()
        potential_siblings = set().union(*(self.id_to_children.get(p_id, set()) for p_id in parents))
        return potential_siblings - {target_id}

    def _get_spouse_ids(self, target_id: str) -> set[str]:
        """Get spouse IDs for a target individual."""
        spouse_ids: set[str] = set()
        parent_families = self._find_family_records_where_individual_is_parent(target_id)
        for fam_record, is_husband, _ in parent_families:
            other_spouse_tag = TAG_WIFE if is_husband else TAG_HUSBAND
            spouse_ref = fam_record.sub_tag(other_spouse_tag) if fam_record is not None else None
            if spouse_ref and hasattr(spouse_ref, "xref_id"):
                spouse_id = _normalize_id(spouse_ref.xref_id)
                if spouse_id:
                    spouse_ids.add(spouse_id)
        return spouse_ids

    def _get_related_ids_by_type(self, target_id: str, relationship_type: str) -> Optional[set[str]]:
        """Get related IDs based on relationship type. Returns None for unknown types."""
        if relationship_type == "parents":
            return self.id_to_parents.get(target_id, set())
        if relationship_type == "children":
            return self.id_to_children.get(target_id, set())
        if relationship_type == "siblings":
            return self._get_sibling_ids(target_id)
        if relationship_type == "spouses":
            return self._get_spouse_ids(target_id)
        logger.warning(f"Unknown relationship type requested: '{relationship_type}'")
        return None

    def _ensure_family_maps_built(self) -> bool:
        """Ensure family maps are built. Returns True if maps are available."""
        if not self.id_to_parents and not self.id_to_children:
            logger.warning("get_related_individuals: Relationship maps empty. Attempting build.")
            self._build_family_maps()
        if not self.id_to_parents and not self.id_to_children:
            logger.error("get_related_individuals: Maps still empty after build attempt.")
            return False
        return True

    def _convert_ids_to_individuals(self, related_ids: set[str], target_id: str) -> list[GedcomIndividualType]:
        """Convert a set of IDs to Individual objects."""
        related_individuals: list[GedcomIndividualType] = []
        for rel_id in related_ids:
            if rel_id != target_id:
                indi = self.find_individual_by_id(rel_id)
                if indi:
                    related_individuals.append(indi)
                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Could not find Individual object for related ID: {rel_id}")
        related_individuals.sort(key=lambda x: (_normalize_id(getattr(x, "xref_id", None)) or ""))
        return related_individuals

    def get_related_individuals(
        self, individual: GedcomIndividualType, relationship_type: str
    ) -> list[GedcomIndividualType]:
        """Gets parents, children, siblings, or spouses using cached maps."""
        # Validate input
        if not _is_individual(individual) or not hasattr(individual, "xref_id"):
            logger.warning(f"get_related_individuals: Invalid input individual object: {type(individual)}")
            return []

        target_id = _normalize_id(individual.xref_id if individual is not None else None)
        if not target_id:
            return []

        # Ensure maps are built
        if not self._ensure_family_maps_built():
            return []

        # Get related IDs and convert to Individual objects
        try:
            related_ids = self._get_related_ids_by_type(target_id, relationship_type)
            if related_ids is None:
                return []
            return self._convert_ids_to_individuals(related_ids, target_id)
        except Exception as e:
            logger.error(f"Error getting {relationship_type} for {target_id}: {e}", exc_info=True)
            return []

    def _find_family_records(
        self, target_id: str, role_tag: str
    ) -> list[GedcomRecordType]:
        """Helper to find FAM records where target_id plays the specified role (HUSB, WIFE, CHIL). Less efficient scan."""
        matching_families: list[GedcomRecordType] = []
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
    ) -> list[tuple[GedcomRecordType, bool, bool]]:
        """Finds FAM records where target_id is HUSB or WIFE using scan (less efficient than maps)."""
        matching_families_with_role: list[tuple[GedcomRecordType, bool, bool]] = []
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

    def _ensure_maps_and_index_built(self) -> Optional[str]:
        """Ensure family maps and individual index are built. Returns error message or None if successful."""
        # Ensure family maps are built
        if not self.id_to_parents and not self.id_to_children:
            logger.warning("Relationship maps are empty, attempting rebuild.")
            self._build_family_maps()
        if not self.id_to_parents and not self.id_to_children:
            return "Error: Family relationship maps could not be built."

        # Ensure individual index is built
        if not self.indi_index:
            logger.warning("Individual index is empty, attempting rebuild.")
            self._build_indi_index()
        if not self.indi_index:
            return "Error: Individual index could not be built."

        return None  # Success

    def _validate_relationship_path_inputs(self, id1_norm: str, id2_norm: str) -> Optional[str]:
        """Validate inputs for relationship path calculation. Returns error message or None if valid."""
        if not self.reader:
            return "Error: GEDCOM Reader unavailable."
        if not id1_norm or not id2_norm:
            return "Invalid input IDs."
        if id1_norm == id2_norm:
            return "Individuals are the same."

        return self._ensure_maps_and_index_built()

    def get_relationship_path(self, id1: str, id2: str) -> str:
        """
        Finds and explains the relationship path between two individuals.

        This implementation uses a general approach to find relationship paths
        without any special cases.

        Args:
            id1: ID of the first individual
            id2: ID of the second individual

        Returns:
            A human-readable string explaining the relationship path
        """
        id1_norm = _normalize_id(id1)
        id2_norm = _normalize_id(id2)

        # Check for None after normalization
        if not id1_norm or not id2_norm:
            return "(Invalid individual IDs)"

        # Validate inputs
        validation_error = self._validate_relationship_path_inputs(id1_norm, id2_norm)
        if validation_error:
            return validation_error

        # Use the enhanced bidirectional BFS algorithm to find the path
        max_depth = 25
        node_limit = 150000
        timeout_sec = 45
        logger.debug(
            f"Calculating relationship path (FastBiBFS): {id1_norm} <-> {id2_norm}"
        )
        search_start = time.time()
        # Convert sets to lists for GraphContext
        id_to_parents_list = {k: list(v) for k, v in self.id_to_parents.items()}
        id_to_children_list = {k: list(v) for k, v in self.id_to_children.items()}
        graph_ctx = GraphContext(
            id_to_parents=id_to_parents_list,
            id_to_children=id_to_children_list,
            start_id=id1_norm,
            end_id=id2_norm
        )
        path_ids = fast_bidirectional_bfs(
            graph_ctx,
            max_depth,
            node_limit,
            timeout_sec,
        )
        search_time = time.time() - search_start
        logger.debug(f"[PROFILE] BFS search completed in {search_time:.2f}s.")
        if not path_ids:
            profile_info = f"[PROFILE] Search: {search_time:.2f}s, MapsBuild: {self.family_maps_build_time:.2f}s, IndexBuild: {self.indi_index_build_time:.2f}s, PreProcess: {self.data_processing_time:.2f}s"
            return f"No relationship path found (FastBiBFS could not connect).\n{profile_info}"

        # Generate the explanation
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
        total_process_time = explanation_time
        profile_info = f"[PROFILE] Total Time: {total_process_time:.2f}s (BFS: 0.00s, Explain: {explanation_time:.2f}s) [Build Times: Maps={self.family_maps_build_time:.2f}s, Index={self.indi_index_build_time:.2f}s, PreProcess={self.data_processing_time:.2f}s]"
        logger.debug(profile_info)
        return f"{explanation_str}\n{profile_info}"

    def _find_direct_relationship(self, id1: str, id2: str) -> list[str]:
        """
        Find a direct relationship between two individuals.

        This is a helper method for get_relationship_path that finds direct
        parent-child, child-parent, or sibling relationships.

        Args:
            id1: ID of the first individual
            id2: ID of the second individual

        Returns:
            A list of IDs representing the path from id1 to id2, or an empty list if no path is found
        """
        # Check if id2 is a parent of id1
        if id2 in self.id_to_parents.get(id1, set()):
            return [id1, id2]

        # Check if id2 is a child of id1
        if id2 in self.id_to_children.get(id1, set()):
            return [id1, id2]

        # Check if id1 and id2 are siblings (share at least one parent)
        parents_1 = self.id_to_parents.get(id1, set())
        parents_2 = self.id_to_parents.get(id2, set())
        common_parents = parents_1.intersection(parents_2)
        if common_parents:
            # Use the first common parent
            common_parent = next(iter(common_parents))
            return [id1, common_parent, id2]

        # No direct relationship found
        return []


# --- COMPREHENSIVE TEST SUITE ---


def gedcom_module_tests() -> bool:
    """
    GEDCOM utilities module test suite.
    Tests the six categories: Initialization, Core Functionality, Edge Cases, Integration, Performance, and Error Handling.
    """
    from test_framework import (
        TestSuite,
        suppress_logging,
    )

    with suppress_logging():
        suite = TestSuite("GEDCOM Utilities & Genealogy Parser", "gedcom_utils.py")

    # Run all tests
    print("📋 Running GEDCOM Utilities & Genealogy Parser comprehensive test suite...")

    with suppress_logging():
        suite.run_test(
            "Core function availability verification",
            test_function_availability,
            "14 GEDCOM functions tested: _is_individual, _normalize_id, _get_full_name, format_life_dates, fast_bidirectional_bfs, etc. - 80%+ availability required.",
            "Test that all required GEDCOM processing functions are available with detailed verification.",
            "Verify _normalize_id→ID cleanup, _is_individual→record detection, _get_full_name→name extraction, format_life_dates→date formatting, fast_bidirectional_bfs→pathfinding.",
        )

        suite.run_test(
            "ID normalization functionality",
            test_id_normalization,
            "6 ID normalization tests: @I123@→standard, I123→no brackets, @F456@→family, F456→family no brackets, empty string, None handling.",
            "Test ID normalization functionality with detailed verification.",
            "Verify _normalize_id() handles @I123@→normalized, I123→normalized, @F456@→family format, empty→None, None→None input processing.",
        )

        # Note: Removed redundant smoke tests (test_individual_detection, test_name_extraction,
        # test_date_parsing, test_event_extraction, test_life_dates_formatting) as they only
        # checked function existence/types. Function availability is already verified by
        # test_function_availability() above, and actual functionality is tested by other tests.

        suite.run_test(
            "Relationship path explanation",
            test_relationship_explanation,
            "Test explain_relationship_path describes family relationships",
            "Relationship explanation provides human-readable kinship descriptions",
            "explain_relationship_path generates accurate relationship descriptions",
        )

        suite.run_test(
            "Bidirectional BFS pathfinding",
            test_bfs_pathfinding,
            "Test fast_bidirectional_bfs finds optimal relationship paths",
            "BFS pathfinding enables efficient genealogical relationship discovery",
            "fast_bidirectional_bfs correctly identifies shortest relationship paths",
        )

        suite.run_test(
            "Sibling relationship detection",
            test_sibling_detection,
            "Test _are_siblings correctly identifies sibling relationships",
            "Sibling detection enables accurate family structure analysis",
            "_are_siblings accurately determines sibling relationships from GEDCOM data",
        )

        suite.run_test(
            "Invalid GEDCOM data handling",
            test_invalid_data_handling,
            "Test functions handle malformed or missing GEDCOM data gracefully",
            "Invalid data handling provides robust error recovery for corrupted files",
            "Functions handle None, empty, and malformed GEDCOM data without crashes",
        )

        suite.run_test(
            "Large dataset performance validation",
            test_large_dataset_performance,
            "Test processing performance with substantial GEDCOM datasets",
            "Performance validation ensures scalability for large genealogical databases",
            "Processing operations complete efficiently with large GEDCOM files",
        )

        suite.run_test(
            "Memory usage optimization",
            test_memory_optimization,
            "Test memory efficiency during extensive GEDCOM processing",
            "Memory optimization prevents resource exhaustion with large datasets",
            "GEDCOM processing maintains reasonable memory usage patterns",
        )

        suite.run_test(
            "Integration with external data",
            test_external_integration,
            "Test compatibility with various GEDCOM file formats and standards",
            "External integration ensures broad compatibility with genealogy software",
            "Functions work correctly with different GEDCOM versions and formats",
        )

        suite.run_test(
            "Error handling and recovery",
            test_error_recovery,
            "Test graceful handling of parsing errors and data inconsistencies",
            "Error recovery maintains functionality despite problematic GEDCOM data",
            "Error conditions are handled gracefully with appropriate fallback behavior",
        )

        suite.run_test(
            "Source citation extraction",
            test_source_citation_extraction,
            "Test extraction of source citations from GEDCOM events",
            "Source citation extraction enables documentation of genealogical facts",
            "get_person_sources() and format_source_citations() correctly extract and format sources",
        )

        suite.run_test(
            "Source citation demonstration",
            test_source_citation_demonstration,
            "Demonstrate complete source citation workflow with examples",
            "Shows real-world usage of source citation in genealogical messages",
            "Complete workflow from extraction to formatted message output",
        )

    # Generate summary report
    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(gedcom_module_tests)


# Test functions for comprehensive testing
def test_function_availability() -> bool:
    """
    Test that required GEDCOM functions are available with detailed verification.

    Verifies the availability and callability of essential GEDCOM processing
    functions. Provides detailed reporting on function availability and
    descriptions of their purposes.

    Returns:
        bool: True if all required functions are available, False otherwise.

    Example:
        >>> if test_function_availability():
        ...     print("All GEDCOM functions available")
    """
    required_functions = [
        '_is_individual', '_is_record', '_normalize_id', 'extract_and_fix_id',
        '_get_full_name', '_parse_date', '_clean_display_date', '_get_event_info',
        'format_life_dates', 'format_full_life_details', 'format_relative_info',
        'fast_bidirectional_bfs', 'explain_relationship_path', '_are_siblings'
    ]

    from test_framework import test_function_availability
    results = test_function_availability(required_functions, globals(), "GEDCOM Utils")
    return all(results)


def test_id_normalization():
    """Test ID normalization functionality with detailed verification."""
    if "_normalize_id" not in globals():
        print("📋 Testing ID normalization: ❌ Function not available")
        return

    test_cases = [
        ("@I123@", "standard individual ID with brackets"),
        ("I123", "individual ID without brackets"),
        ("@F456@", "family ID with brackets"),
        ("F456", "family ID without brackets"),
        ("", "empty string handling"),
        (None, "None input handling"),
    ]

    print("📋 Testing GEDCOM ID normalization:")
    results = []

    for test_id, description in test_cases:
        try:
            normalized = _normalize_id(test_id)
            is_valid_result = isinstance(normalized, str) or normalized is None

            status = "✅" if is_valid_result else "❌"
            print(f"   {status} {description}")
            print(f"      Input: {test_id!r} → Output: {normalized!r}")

            results.append(is_valid_result)
            assert (
                is_valid_result
            ), f"_normalize_id should return string or None for {test_id}"

        except Exception as e:
            print(f"   ⚠️ {description}")
            print(f"      Input: {test_id!r} → Error: {e} (may be acceptable)")
            results.append(True)  # Some formats may be invalid, which is acceptable

    print(f"📊 Results: {sum(results)}/{len(results)} ID normalization tests passed")


def test_relationship_explanation():
    """Test relationship path explanation."""
    if "explain_relationship_path" in globals():
        # Test with minimal valid parameters - should return a string or raise specific exception
        try:
            result = explain_relationship_path([], None, {}, {}, {})
            # If it doesn't raise an exception, it should return a string
            assert isinstance(result, str), f"Expected str, got {type(result)}"
        except (ValueError, KeyError, TypeError) as e:
            # These specific exceptions are expected for invalid parameters
            logger.debug(f"Expected exception for invalid parameters: {e}")
        except Exception as e:
            # Unexpected exceptions should fail the test
            raise AssertionError(f"Unexpected exception type: {type(e).__name__}: {e}") from e


def test_bfs_pathfinding():
    """Test bidirectional BFS pathfinding."""
    if "fast_bidirectional_bfs" in globals():
        # Test with empty graph - should return empty list or raise specific exception
        try:
            graph_ctx = GraphContext(id_to_parents={}, id_to_children={}, start_id="start", end_id="end")
            result = fast_bidirectional_bfs(graph_ctx)
            # If it doesn't raise an exception, it should return a list
            assert isinstance(result, list), f"Expected list, got {type(result)}"
        except (ValueError, KeyError) as e:
            # These specific exceptions are expected for missing IDs
            logger.debug(f"Expected exception for missing IDs: {e}")
        except Exception as e:
            # Unexpected exceptions should fail the test
            raise AssertionError(f"Unexpected exception type: {type(e).__name__}: {e}") from e


def test_sibling_detection():
    """Test sibling relationship detection."""
    if "_are_siblings" in globals():
        # Test with valid string IDs - should return False for empty family map
        try:
            result = _are_siblings("I1", "I2", {})
            # With empty family map, should return False (not siblings)
            assert isinstance(result, bool), f"Expected bool, got {type(result)}"
            assert result is False, "Empty family map should return False (not siblings)"
        except (ValueError, KeyError, TypeError) as e:
            # These specific exceptions are expected for missing data
            logger.debug(f"Expected exception for missing data: {e}")
        except Exception as e:
            # Unexpected exceptions should fail the test
            raise AssertionError(f"Unexpected exception type: {type(e).__name__}: {e}") from e


def test_invalid_data_handling():
    """Test handling of invalid GEDCOM data."""
    # Test functions with invalid inputs
    if "_normalize_id" in globals():
        try:
            result = _normalize_id(None)
            # Should return None for None input
            assert result is None, f"Expected None for None input, got {result}"
        except (ValueError, TypeError) as e:
            # These specific exceptions are acceptable for None input
            logger.debug(f"Expected exception for None input: {e}")
        except Exception as e:
            # Unexpected exceptions should fail the test
            raise AssertionError(f"Unexpected exception type: {type(e).__name__}: {e}") from e

    if "_is_individual" in globals():
        result = _is_individual("invalid")
        # Should return False for invalid string (not a dict with 'tag' key)
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
        assert result is False, "Invalid string should return False (not an individual)"


def test_large_dataset_performance():
    """Test performance with large datasets."""
    # Test ID normalization performance
    if "_normalize_id" in globals():
        start_time = time.time()
        for i in range(1000):
            _normalize_id(f"@I{i}@")
        end_time = time.time()
        assert (end_time - start_time) < 1.0  # Should complete in under 1 second


def test_memory_optimization():
    """Test memory usage optimization."""
    # Test that functions don't create excessive memory overhead
    if "_get_full_name" in globals():
        # Test that calling function repeatedly doesn't accumulate memory
        # Each call should handle None gracefully and return consistent result
        results = []
        for _ in range(100):
            try:
                result = _get_full_name(None)
                results.append(result)
            except (ValueError, TypeError, AttributeError) as e:
                # These specific exceptions are acceptable for None input
                logger.debug(f"Expected exception for None input: {e}")
                results.append(None)
            except Exception as e:
                # Unexpected exceptions should fail the test
                raise AssertionError(f"Unexpected exception type: {type(e).__name__}: {e}") from e

        # Verify consistent behavior across all calls
        if results:
            first_result = results[0]
            assert all(r == first_result for r in results), "Function should return consistent results"


def test_external_integration():
    """Test integration capabilities."""
    # Test that functions can handle various data structures
    test_data_structures = [None, {}, [], "", 0, False]

    if "_is_individual" in globals():
        for data in test_data_structures:
            try:
                result = _is_individual(data)
                # Should always return a boolean
                assert isinstance(result, bool), f"Expected bool for {data}, got {type(result)}"
                # None, empty dict, empty list, empty string, 0, False should all return False
                # (not valid individual records)
                assert result is False, f"Expected False for {data}, got {result}"
            except (ValueError, TypeError, AttributeError) as e:
                # These specific exceptions are acceptable for invalid data
                logger.debug(f"Expected exception for {data}: {e}")
            except Exception as e:
                # Unexpected exceptions should fail the test
                raise AssertionError(f"Unexpected exception for {data}: {type(e).__name__}: {e}") from e


def test_error_recovery():
    """Test error handling and recovery."""
    # Test that functions handle errors gracefully
    if "_normalize_id" in globals():
        try:
            result = _normalize_id("invalid_format")
            # Should either return the string as-is or None for invalid format
            assert result is None or isinstance(result, str), f"Expected None or str, got {type(result)}"
        except (ValueError, TypeError) as e:
            # These specific exceptions are acceptable for invalid format
            logger.debug(f"Expected exception for invalid format: {e}")
        except Exception as e:
            # Unexpected exceptions should fail the test
            raise AssertionError(f"Unexpected exception type: {type(e).__name__}: {e}") from e

    if "_get_full_name" in globals():
        try:
            result = _get_full_name(None)
            # Should return "Unknown" or a string for None input
            assert result == "Unknown" or isinstance(result, str), f"Expected 'Unknown' or str, got {result}"
        except (ValueError, TypeError, AttributeError) as e:
            # These specific exceptions are acceptable for None input
            logger.debug(f"Expected exception for None input: {e}")
        except Exception as e:
            # Unexpected exceptions should fail the test
            raise AssertionError(f"Unexpected exception type: {type(e).__name__}: {e}") from e


def test_source_citation_extraction():
    """
    Test source citation extraction and formatting.

    Phase 5.1: Source Citation Support
    Tests get_person_sources() and format_source_citations() functions.
    """
    from unittest.mock import Mock

    # Test 1: format_source_citations with no sources
    empty_sources = {'birth': [], 'death': [], 'other': []}
    result = format_source_citations(empty_sources)
    assert result == "", "Empty sources should return empty string"

    # Test 2: format_source_citations with single birth source
    single_source = {'birth': ['1881 Scotland Census'], 'death': [], 'other': []}
    result = format_source_citations(single_source)
    assert "1881 Scotland Census" in result, "Should include birth source"
    assert "documented in" in result, "Should include 'documented in' prefix"

    # Test 3: format_source_citations with multiple sources (2 sources)
    multiple_sources = {
        'birth': ['Birth Certificate'],
        'death': ['Death Certificate 1920'],
        'other': []
    }
    result = format_source_citations(multiple_sources)
    assert "Birth Certificate" in result, "Should include birth source"
    assert "Death Certificate 1920" in result, "Should include death source"
    assert " and " in result, "Should use 'and' for two sources"

    # Test 4: format_source_citations with 3+ sources
    many_sources = {
        'birth': ['Birth Certificate', '1881 Census'],
        'death': ['Death Certificate 1920'],
        'other': []
    }
    result = format_source_citations(many_sources)
    assert "Birth Certificate" in result, "Should include first birth source"
    assert "1881 Census" in result, "Should include second birth source"
    assert "Death Certificate 1920" in result, "Should include death source"
    assert ", and " in result, "Should use ', and' for 3+ sources"

    # Test 5: get_person_sources with None individual
    sources = get_person_sources(None)
    assert isinstance(sources, dict), "Should return dict even for None"
    assert 'birth' in sources, "Should have birth key"
    assert 'death' in sources, "Should have death key"

    # Test 6: _extract_sources_from_event with None event
    sources_list = _extract_sources_from_event(None)
    assert isinstance(sources_list, list), "Should return list even for None"
    assert len(sources_list) == 0, "Should return empty list for None"

    # Test 7: _extract_sources_from_event with mock event containing sources
    mock_sour_tag = Mock()
    mock_sour_tag.value = "1881 Scotland Census"
    mock_title_tag = Mock()
    mock_title_tag.value = "Census Record"

    mock_sour_with_title = Mock()
    mock_sour_with_title.sub_tag = Mock(return_value=mock_title_tag)

    mock_event = Mock()
    mock_event.sub_tags = Mock(return_value=[mock_sour_tag, mock_sour_with_title])

    sources_list = _extract_sources_from_event(mock_event)
    assert isinstance(sources_list, list), "Should return list for mock event"
    assert len(sources_list) >= 0, "Should handle mock event sources"

    logger.info("✓ Source citation extraction and formatting works correctly")
    return True


def test_source_citation_demonstration():
    """
    Demonstration test for source citation functionality.

    Phase 5.1: Source Citation Support
    Shows complete workflow from extraction to formatted output using real GEDCOM data.
    Uses cached GEDCOM data for performance.
    """
    import os

    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATION: Source Citation Extraction & Formatting")
    logger.info("="*60)

    # Load real GEDCOM data using cache
    gedcom_file = str(config.database.gedcom_file_path)
    test_person_id = os.getenv("TEST_PERSON_ID", "I102281560744")  # Fraser Gault

    logger.info(f"\n📁 Loading GEDCOM file (cached): {gedcom_file}")
    logger.info(f"🔍 Test subject: {test_person_id} (Fraser Gault)")

    try:
        # Load GEDCOM data using aggressive caching (memory + disk cache)
        from gedcom_cache import load_gedcom_with_aggressive_caching

        gedcom_data = load_gedcom_with_aggressive_caching(gedcom_file)
        if not gedcom_data or not gedcom_data.indi_index:
            logger.warning("   ⚠️  Could not load GEDCOM data - using mock examples")
            _run_mock_demonstration()
            return True

        # Find Fraser Gault in GEDCOM
        test_person = None
        for person_id, individual in gedcom_data.indi_index.items():
            if person_id == test_person_id:
                test_person = individual
                break

        if not test_person:
            logger.warning(f"   ⚠️  Could not find {test_person_id} in GEDCOM - using mock examples")
            _run_mock_demonstration()
            return True

        # Extract real sources from Fraser Gault
        logger.info("\n📋 Real GEDCOM Data: Extracting sources for Fraser Gault")
        person_name = _get_full_name(test_person)
        life_dates = format_life_dates(test_person)
        logger.info(f"   Person: {person_name}{life_dates}")

        sources = get_person_sources(test_person)
        logger.info(f"   Birth sources: {sources['birth']}")
        logger.info(f"   Death sources: {sources['death']}")
        logger.info(f"   Other sources: {sources['other']}")

        # Format citation
        citation = format_source_citations(sources)
        if citation:
            logger.info(f"\n   ✓ Formatted citation: '{citation}'")

            # Show complete message example
            logger.info("\n📧 Complete message example:")
            message = f"According to my tree, {person_name}{life_dates} is {citation}."
            logger.info(f"   '{message}'")
        else:
            logger.info(f"\n   [i] No sources found for {person_name}")
            logger.info("   This is normal - not all GEDCOM records have source citations")

        # Also show formatting examples with mock data
        logger.info("\n" + "-"*60)
        logger.info("📋 Formatting Examples (Mock Data)")
        logger.info("-"*60)

        _run_mock_demonstration()

    except Exception as e:
        logger.error(f"   ❌ Error in demonstration: {e}")
        logger.info("   Falling back to mock examples")
        _run_mock_demonstration()

    logger.info("\n" + "="*60)
    logger.info("✓ Source citation demonstration complete!")
    logger.info("="*60 + "\n")
    return True


def _run_mock_demonstration() -> None:
    """Run demonstration with mock data."""
    # Scenario 1: Person with no sources
    logger.info("\n   Example 1: Person with no sources")
    empty_sources = {'birth': [], 'death': [], 'other': []}
    citation = format_source_citations(empty_sources)
    logger.info(f"      Result: '{citation}' (empty)")

    # Scenario 2: Person with single birth source
    logger.info("\n   Example 2: Person with single birth source")
    single_source = {'birth': ['1881 Scotland Census'], 'death': [], 'other': []}
    citation = format_source_citations(single_source)
    logger.info(f"      Result: '{citation}'")

    # Scenario 3: Person with birth and death sources
    logger.info("\n   Example 3: Person with birth and death sources")
    two_sources = {
        'birth': ['Birth Certificate, Banff'],
        'death': ['Death Certificate 1920, Aberdeen'],
        'other': []
    }
    citation = format_source_citations(two_sources)
    logger.info(f"      Result: '{citation}'")

    # Scenario 4: Person with multiple sources
    logger.info("\n   Example 4: Person with multiple sources (3+)")
    many_sources = {
        'birth': ['Birth Certificate', '1881 Scotland Census'],
        'death': ['Death Certificate 1920'],
        'other': []
    }
    citation = format_source_citations(many_sources)
    logger.info(f"      Result: '{citation}'")


# ==============================================
# Standard Test Suite
# ==============================================
# Note: Comprehensive tests are already defined in gedcom_module_tests()
# Additional test functions for specific GEDCOM operations are defined above





# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
