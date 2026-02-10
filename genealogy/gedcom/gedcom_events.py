#!/usr/bin/env python3

"""GEDCOM event extraction, date parsing pipeline, and life detail formatting.

Provides functions to:
- Parse and normalise GEDCOM date strings into timezone-aware datetime objects.
- Extract event records (birth, death, etc.) with date, place, and source data.
- Format life dates, source citations, and full life details for display.
"""

# === PATH SETUP FOR PACKAGE IMPORTS ===
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# === CORE INFRASTRUCTURE ===
import logging
import os
import re
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

from genealogy.gedcom.gedcom_utils import (
    TAG_BIRTH,
    TAG_DATE,
    TAG_DEATH,
    TAG_INDI,
    TAG_NAME,
    TAG_PLACE,
    TAG_SOUR,
    TAG_TITL,
    GedcomIndividualType,
    _is_individual,
    extract_and_fix_id,
)

# dateparser is optional
dateparser: Any | None = None
try:
    import dateparser as _dateparser

    dateparser = _dateparser
    _dateparser_available = True
except ImportError:
    dateparser = None
    _dateparser_available = False

DATEPARSER_AVAILABLE = _dateparser_available

from genealogy.gedcom.gedcom_parser import get_full_name
from testing.test_framework import TestSuite, create_standard_test_runner

logger = logging.getLogger(__name__)


# ==============================================
# Date Parsing Pipeline
# ==============================================


def _validate_and_normalize_date_string(date_str: str | None) -> str | None:
    """Validate and perform initial normalization of date string."""
    if not date_str:
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


def _clean_date_string(date_str: str) -> str | None:
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


def _try_dateparser(cleaned_str: str) -> datetime | None:
    """Try parsing with dateparser library if available."""
    if not DATEPARSER_AVAILABLE or dateparser is None:
        return None

    try:
        settings = {"PREFER_DAY_OF_MONTH": "first", "REQUIRE_PARTS": ["year"]}
        parsed_dt = dateparser.parse(cleaned_str, settings=settings)

        if parsed_dt:
            pass
        else:
            logger.debug(f"dateparser returned None for '{cleaned_str}'")

        return parsed_dt
    except Exception as e:
        logger.error(f"Error using dateparser for '{cleaned_str}': {e}", exc_info=False)
        return None


def _try_strptime_formats(cleaned_str: str) -> datetime | None:
    """Try parsing with various strptime formats."""
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
            return dt_naive
        except ValueError:
            continue
        except Exception as e:
            logger.debug(f"Strptime error for format '{fmt}': {e}")
            continue

    return None


def _extract_year_fallback(cleaned_str: str) -> datetime | None:
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


def _finalize_parsed_date(parsed_dt: datetime | None, original_date_str: str) -> datetime | None:
    """Finalize parsed date by validating and adding timezone."""
    if not isinstance(parsed_dt, datetime):
        logger.warning(f"All parsing attempts failed for: '{original_date_str}'")
        return None

    if parsed_dt.year == 0:
        logger.warning(f"Parsed date resulted in year 0, treating as invalid: '{original_date_str}'")
        return None

    if parsed_dt.tzinfo is None:
        return parsed_dt.replace(tzinfo=UTC)

    return parsed_dt.astimezone(UTC)


def _parse_date(date_str: str | None) -> datetime | None:
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


def _clean_display_date(raw_date_str: str | None) -> str:  # ... implementation ...
    if not raw_date_str or raw_date_str == "N/A":
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


# ==============================================
# Event Extraction
# ==============================================


def _validate_and_normalize_individual(individual: GedcomIndividualType) -> GedcomIndividualType | None:
    """Validate and normalize individual to ensure it's a valid GedcomIndividualType."""
    if not _is_individual(individual):
        wrapped_value = getattr(individual, "value", None)
        if _is_individual(wrapped_value):
            return cast(GedcomIndividualType, wrapped_value)
        logger.warning(f"get_event_info invalid input type: {type(individual)}")
        return None

    if individual is None:
        return None

    return individual


def _extract_event_record(individual: GedcomIndividualType, event_tag: str, indi_id_log: str) -> Any | None:
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


def _extract_date_from_event(event_record: Any) -> tuple[datetime | None, str]:
    """Extract date information from event record."""
    date_obj: datetime | None = None
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
    sources: list[str] = []
    try:
        # Get all source citations for this event
        sour_tags = event_record.sub_tags(TAG_SOUR)
        for sour_tag in sour_tags:
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


def get_event_info(
    individual: GedcomIndividualType, event_tag: str
) -> tuple[datetime | None, str, str]:  # ... implementation ...
    date_obj: datetime | None = None
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
        logger.debug(f"Attribute error getting event '{event_tag}' for {indi_id_log}: {ae}")
    except Exception as e:
        logger.error(f"Error accessing event {event_tag} for @{indi_id_log}@: {e}", exc_info=True)
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
    sources_by_event: dict[str, list[str]] = {'birth': [], 'death': [], 'other': []}

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


# ==============================================
# Formatting Helpers
# ==============================================


def format_life_dates(indi: GedcomIndividualType) -> str:  # ... implementation ...
    if not _is_individual(indi):
        logger.warning(f"format_life_dates called with non-Individual type: {type(indi)}")
        return ""
    _, b_date_str, _ = get_event_info(indi, TAG_BIRTH)
    _, d_date_str, _ = get_event_info(indi, TAG_DEATH)
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
    citations: list[str] = []

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
        logger.warning(f"format_full_life_details called with non-Individual type: {type(indi)}")
        return "(Error: Invalid data)", ""
    _, b_date_str, b_place = get_event_info(indi, TAG_BIRTH)
    b_date_str_cleaned = _clean_display_date(b_date_str)
    b_place_cleaned = b_place if b_place != "N/A" else "(Place unknown)"
    birth_info = f"Born: {b_date_str_cleaned if b_date_str_cleaned != 'N/A' else '(Date unknown)'} in {b_place_cleaned}"
    _, d_date_str, d_place = get_event_info(indi, TAG_DEATH)
    d_date_str_cleaned = _clean_display_date(d_date_str)
    d_place_cleaned = d_place if d_place != "N/A" else "(Place unknown)"
    death_info = ""
    if d_date_str_cleaned != "N/A" or d_place != "N/A":
        death_info = (
            f"   Died: {d_date_str_cleaned if d_date_str_cleaned != 'N/A' else '(Date unknown)'} in {d_place_cleaned}"
        )
    return birth_info, death_info


def format_relative_info(relative: Any) -> str:  # ... implementation ...
    indi_obj: GedcomIndividualType | None = None
    if _is_individual(relative):
        indi_obj = relative
    elif hasattr(relative, "value") and _is_individual(getattr(relative, "value", None)):
        indi_obj = relative.value
    elif hasattr(relative, "xref_id") and isinstance(getattr(relative, "xref_id", None), str):
        norm_id = extract_and_fix_id(relative)
        return f"  - (Relative Data: ID={norm_id or 'N/A'}, Type={type(relative).__name__})"
    else:
        return f"  - (Invalid Relative Data: Type={type(relative).__name__})"
    rel_name = get_full_name(indi_obj)
    life_info = format_life_dates(indi_obj)
    return f"  - {rel_name}{life_info}"


# ==============================================
# Test Helpers
# ==============================================

_PATCH_GEDCOM_SENTINEL = object()


@contextmanager
def _temporary_globals(overrides: dict[str, Any]):
    previous: dict[str, Any] = {}
    try:
        for key, value in overrides.items():
            previous[key] = globals().get(key, _PATCH_GEDCOM_SENTINEL)
            globals()[key] = value
        yield
    finally:
        for key, original in previous.items():
            if original is _PATCH_GEDCOM_SENTINEL:
                globals().pop(key, None)
            else:
                globals()[key] = original


class _FakeSourceTag:
    def __init__(self, title: str) -> None:
        self._title = title

    def sub_tag(self, tag: str) -> Any:
        if tag == TAG_TITL:
            return SimpleNamespace(value=self._title)
        return None


class _FakeEventRecord:
    def __init__(
        self,
        *,
        date_value: str,
        place_value: str,
        source_titles: list[str] | None = None,
    ) -> None:
        self._date_tag = SimpleNamespace(value=date_value)
        self._place_tag = SimpleNamespace(value=place_value)
        self._source_tags = [_FakeSourceTag(title) for title in (source_titles or [])]

    def sub_tag(self, tag: str) -> Any:
        if tag == TAG_DATE:
            return self._date_tag
        if tag == TAG_PLACE:
            return self._place_tag
        return None

    def sub_tags(self, tag: str) -> list[Any]:
        if tag == TAG_SOUR:
            return list(self._source_tags)
        return []


class _FakeIndividualRecord:
    def __init__(
        self,
        *,
        xref_id: str = "@I1@",
        name_obj: Any = None,
        name_tag: Any = None,
        sub_tag_values: dict[str, Any] | None = None,
        event_map: dict[str, Any] | None = None,
    ) -> None:
        self.tag = TAG_INDI
        self.xref_id = xref_id
        self.name = name_obj
        self._name_tag = name_tag
        self._sub_tag_values = sub_tag_values or {}
        self._event_map = event_map or {}

    def sub_tag(self, tag: str) -> Any:
        if tag == TAG_NAME:
            return self._name_tag
        return self._event_map.get(tag)

    def sub_tag_value(self, tag: str) -> Any:
        return self._sub_tag_values.get(tag)


# ==============================================
# Tests
# ==============================================


def _test_parse_date_uses_dateparser_when_available() -> bool:
    class FakeDateParser:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any] | None]] = []

        def parse(self, cleaned_str: str, settings: dict[str, Any] | None = None) -> datetime:
            self.calls.append((cleaned_str, settings))
            return datetime(1875, 4, 5)

    fake_parser = FakeDateParser()
    with _temporary_globals({"DATEPARSER_AVAILABLE": True, "dateparser": fake_parser}):
        parsed = _parse_date("ABT 5 APR 1875")
    assert parsed is not None
    assert parsed.year == 1875 and parsed.tzinfo == UTC
    assert fake_parser.calls
    return True


def _test_parse_date_falls_back_to_year_extraction() -> bool:
    with _temporary_globals({"DATEPARSER_AVAILABLE": False, "dateparser": None}):
        parsed = _parse_date("BET 1900 AND 1905")
    assert parsed is not None
    assert parsed.year == 1900 and parsed.tzinfo == UTC
    return True


def _test_clean_display_date_transforms_modifiers() -> bool:
    assert _clean_display_date("(ABT 1850)") == "~1850"
    assert _clean_display_date("BET 1900 AND 1905") == "1900-1905"
    assert _clean_display_date(None) == "N/A"
    return True


def _test_extract_event_helpers_capture_date_place_and_sources() -> bool:
    event = _FakeEventRecord(
        date_value="12 JAN 1900",
        place_value="Boston, MA",
        source_titles=["Birth Register", "Family Bible"],
    )
    indi = cast(GedcomIndividualType, _FakeIndividualRecord(event_map={TAG_BIRTH: event}))
    record = _extract_event_record(indi, TAG_BIRTH, "I1")
    assert record is event

    date_obj, date_str = _extract_date_from_event(event)
    assert date_str == "12 JAN 1900"
    assert date_obj and date_obj.year == 1900

    place = _extract_place_from_event(event)
    assert place == "Boston, MA"

    sources = _extract_sources_from_event(event)
    assert sources == ["Birth Register", "Family Bible"]
    return True


def module_tests() -> bool:
    suite = TestSuite("gedcom_events", "gedcom_events.py")
    suite.run_test(
        "Date parsing via dateparser",
        _test_parse_date_uses_dateparser_when_available,
        "Confirms dateparser output is accepted and normalized to UTC.",
    )
    suite.run_test(
        "Date parsing fallback",
        _test_parse_date_falls_back_to_year_extraction,
        "Ensures year extraction fallback triggers when dateparser/strptime fail.",
    )
    suite.run_test(
        "Display date cleaning",
        _test_clean_display_date_transforms_modifiers,
        "Verifies display cleaning handles parentheses and modifiers.",
    )
    suite.run_test(
        "Event helper extraction",
        _test_extract_event_helpers_capture_date_place_and_sources,
        "Ensures event records return date/place/source information for tables.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    if os.environ.get("RUN_MODULE_TESTS") == "1":
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    print("gedcom_events provides event/date extraction helpers; no standalone CLI entry point.")
    print("Set RUN_MODULE_TESTS=1 before execution to run the embedded regression tests.")
    sys.exit(0)
