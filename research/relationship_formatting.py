#!/usr/bin/env python3
"""Relationship Formatting - Path Formatting and Relationship Label Conversion.

Functions for formatting relationship paths, converting between GEDCOM/API/Discovery
formats, and generating human-readable relationship labels.

Split from research.relationship_utils to reduce module size.
"""

# === CORE INFRASTRUCTURE ===
import html
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

from bs4 import BeautifulSoup, Tag

from performance.memory_utils import fast_json_loads

BS4_AVAILABLE = True

from core.utils import format_name
from genealogy.gedcom.gedcom_utils import (
    TAG_BIRTH,
    TAG_DEATH,
    TAG_SEX,
    are_spouses,
    get_event_info,
    get_full_name,
)
from genealogy.relationship_calculations import (
    are_cousins,
    are_siblings,
    is_aunt_or_uncle,
    is_grandchild,
    is_grandparent,
    is_great_grandchild,
    is_great_grandparent,
    is_niece_or_nephew,
)

logger = logging.getLogger(__name__)


# === ERROR HANDLING ===
# Debug log de-duplication for gender inference
_gender_log_once: set[str] = set()


def _log_inferred_gender_once(name: str, source: str, message: str) -> None:
    try:
        key = f"{source}:{name.lower()}"
        if key in _gender_log_once:
            return
        # Prevent unbounded growth in long sessions
        if len(_gender_log_once) > 200:
            _gender_log_once.clear()
        _gender_log_once.add(key)
        logger.debug(message)
    except Exception:
        # Never break control flow for logging
        pass


# === PROTOCOLS ===


class GedcomTagProtocol(Protocol):
    value: Any


class GedcomIndividualProtocol(Protocol):
    def sub_tag(self, tag: str) -> GedcomTagProtocol | None: ...


# === RELATIONSHIP PATH EXPLANATION ===


def explain_relationship_path(
    path_ids: list[str],
    reader: Any,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
    indi_index: dict[str, GedcomIndividualProtocol | Any],
    owner_name: str = "Reference Person",
    relationship_type: str = "relative",
) -> str:
    """
    Generates a human-readable explanation of the relationship path using the unified format.

    This implementation uses a generic approach to determine relationships
    between individuals in the path without special cases. It analyzes the
    family structure to determine parent-child, sibling, spouse, and other
    relationships.

    Args:
        path_ids: List of GEDCOM IDs representing the relationship path
        reader: GEDCOM reader object
        id_to_parents: Dictionary mapping IDs to parent IDs
        id_to_children: Dictionary mapping IDs to child IDs
        indi_index: Dictionary mapping IDs to GEDCOM individual objects
        owner_name: Name of the owner/reference person (default: "Reference Person")
        relationship_type: Type of relationship between target and owner (default: "relative")

    Returns:
        Formatted relationship path string
    """
    if not path_ids or len(path_ids) < 2:
        return "(No relationship path explanation available)"

    # Convert the GEDCOM path to the unified format
    unified_path = convert_gedcom_path_to_unified_format(path_ids, reader, id_to_parents, id_to_children, indi_index)

    if not unified_path:
        return "(Error: Could not convert relationship path to unified format)"

    # Get the target name from the first person in the path
    target_name = unified_path[0].get("name", "Unknown Person")

    # Format the path using the unified formatter
    return format_relationship_path_unified(
        unified_path,
        target_name if target_name is not None else "Unknown Person",
        owner_name,
        relationship_type,
    )


# === HTML EXTRACTION HELPERS ===


def _extract_html_from_response(
    api_response_data: str | dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Extract HTML content and JSON data from API response."""
    html_content_raw: str | None = None
    json_data: dict[str, Any] | None = None

    if isinstance(api_response_data, str):
        # Handle JSONP response format: no({...})
        jsonp_match = re.search(r"no\((.*)\)", api_response_data, re.DOTALL)
        if jsonp_match:
            try:
                json_str = jsonp_match.group(1)
                json_data = fast_json_loads(json_str)
                html_content_raw = json_data.get("html") if json_data is not None else None
            except Exception as e:
                logger.error(f"Error parsing JSONP response: {e}", exc_info=True)
                return None, None
        else:
            # Direct HTML response
            html_content_raw = api_response_data
    elif isinstance(api_response_data, dict):
        # Handle direct JSON/dict response
        json_data = api_response_data
        html_content_raw = json_data.get("html") if json_data else None

    return html_content_raw, json_data


def _format_discovery_api_path(json_data: dict[str, Any], target_name: str, owner_name: str) -> str | None:
    """Format relationship path from Discovery API JSON format."""
    if not json_data or "path" not in json_data:
        return None

    discovery_path = json_data["path"]
    if not isinstance(discovery_path, list) or not discovery_path:
        return None

    logger.info("Formatting relationship path from Discovery API JSON.")
    path_steps_json = [f"*   {format_name(target_name)}"]

    for step in discovery_path:
        if not isinstance(step, dict):
            continue

        step_dict = cast(dict[str, Any], step)
        step_name = format_name(step_dict.get("name", "?"))
        step_rel = step_dict.get("relationship", "?")
        step_rel_display = _get_relationship_term(None, step_rel).capitalize()
        path_steps_json.append(f"    -> is {step_rel_display} of")
        path_steps_json.append(f"*   {step_name}")

    path_steps_json.append("    -> leads to")
    path_steps_json.append(f"*   {owner_name} (You)")

    return "\n".join(path_steps_json)


def _try_simple_text_relationship(html_content_raw: str, target_name: str, owner_name: str) -> str | None:
    """Try to extract relationship from simple text format."""
    if not html_content_raw or html_content_raw.strip().startswith("<"):
        return None

    text = html_content_raw.strip()
    relationship_patterns = [
        r"is the (father|mother|son|daughter|brother|sister|husband|wife|parent|child|sibling|spouse) of",
        r"(father|mother|son|daughter|brother|sister|husband|wife|parent|child|sibling|spouse)",
    ]

    for pattern in relationship_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            relationship = match.group(1).lower()
            return f"{target_name} is the {relationship} of {owner_name}"

    # If no pattern found but contains relationship terms, return original text
    relationship_terms = [
        "father",
        "mother",
        "son",
        "daughter",
        "brother",
        "sister",
        "husband",
        "wife",
        "parent",
        "child",
        "sibling",
        "spouse",
    ]
    if any(rel in text.lower() for rel in relationship_terms):
        return text

    return None


def _should_skip_list_item(item: Any) -> bool:
    """Check if list item should be skipped."""
    try:
        if not isinstance(item, Tag):
            return True

        # Cast to Any to avoid Pyright unknown member type errors with bs4
        tag_item = cast(Any, item)
        is_hidden = tag_item.get("aria-hidden") == "true"
        item_classes = tag_item.get("class") or []
        has_icon_class = isinstance(item_classes, list) and "icon" in item_classes

        return is_hidden or has_icon_class
    except (AttributeError, TypeError):
        logger.debug(f"Error checking item attributes: {type(item)}")
        return True


def _extract_name_from_item(item: Any) -> str:
    """Extract name from list item."""
    try:
        name_elem = item.find("b") if isinstance(item, Tag) else None
        if name_elem and hasattr(name_elem, "get_text"):
            return name_elem.get_text(strip=True)
        if hasattr(item, "string") and item.string:
            return str(item.string).strip()
        return "Unknown"
    except (AttributeError, TypeError):
        logger.debug(f"Error extracting name: {type(item)}")
        return "Unknown"


def _extract_relationship_from_item(item: Any) -> str:
    """Extract relationship description from list item."""
    try:
        rel_elem = item.find("i") if isinstance(item, Tag) else None
        if rel_elem and hasattr(rel_elem, "get_text"):
            return rel_elem.get_text(strip=True)
        return ""
    except (AttributeError, TypeError):
        logger.debug(f"Error extracting relationship: {type(item)}")
        return ""


def _extract_lifespan_from_item(item: Any) -> str:
    """Extract lifespan from list item."""
    try:
        text_content = item.get_text(strip=True) if hasattr(item, "get_text") else str(item)
        lifespan_match = re.search(r"(\d{4})-(\d{4}|\bLiving\b|-)", text_content, re.IGNORECASE)
        return lifespan_match.group(0) if lifespan_match else ""
    except (AttributeError, TypeError):
        logger.debug(f"Error extracting lifespan: {type(item)}")
        return ""


def _extract_person_from_list_item(item: Any) -> dict[str, str]:
    """Extract name, relationship, and lifespan from a list item."""
    if _should_skip_list_item(item):
        return {}

    return {
        "name": _extract_name_from_item(item),
        "relationship": _extract_relationship_from_item(item),
        "lifespan": _extract_lifespan_from_item(item),
    }


def _parse_html_relationship_data(html_content_raw: str) -> list[dict[str, str]]:
    """Parse relationship data from HTML content using BeautifulSoup."""
    if not BS4_AVAILABLE:
        logger.error("BeautifulSoup is not available. Cannot parse HTML.")
        return []

    # Decode HTML entities
    html_content_decoded = html.unescape(html_content_raw) if html_content_raw else ""

    try:
        soup = BeautifulSoup(html_content_decoded, "html.parser")
        any_soup = cast(Any, soup)
        list_items = any_soup.find_all("li")

        if not list_items or len(list_items) < 2:
            logger.warning(f"Not enough list items found in HTML: {len(list_items) if list_items else 0}")
            return []

        # Extract relationship information from each list item
        relationship_data: list[dict[str, str]] = []
        for item in list_items:
            person_data = _extract_person_from_list_item(item)
            if person_data:  # Only add if we got valid data
                relationship_data.append(person_data)

        return relationship_data

    except Exception as e:
        logger.error(f"Error parsing relationship HTML: {e}", exc_info=True)
        return []


def _try_json_api_format(json_data: dict[str, Any] | None, target_name: str, owner_name: str) -> str | None:
    """Try to format relationship from Discovery API JSON format."""
    if not json_data:
        return None
    return _format_discovery_api_path(json_data, target_name, owner_name)


def _try_html_formats(
    html_content_raw: str | None, target_name: str, owner_name: str, relationship_type: str
) -> str:
    """Try to format relationship from HTML content."""
    if not html_content_raw:
        logger.warning("No HTML content found in API response.")
        return "(No relationship HTML content found in API response)"

    # Try simple text relationship format
    simple_text_result = _try_simple_text_relationship(html_content_raw, target_name, owner_name)
    if simple_text_result:
        return simple_text_result

    # Check BeautifulSoup availability
    if not BS4_AVAILABLE:
        logger.error("BeautifulSoup is not available. Cannot parse HTML.")
        return "(BeautifulSoup is not available. Cannot parse relationship HTML.)"

    # Parse HTML relationship data
    relationship_data = _parse_html_relationship_data(html_content_raw)

    if not relationship_data:
        return "(Could not extract relationship data from HTML)"

    # Convert to unified format and format the path
    unified_path = convert_api_path_to_unified_format(relationship_data, target_name)

    if not unified_path:
        return "(Error: Could not convert relationship data to unified format)"

    return format_relationship_path_unified(unified_path, target_name, owner_name, relationship_type)


def format_api_relationship_path(
    api_response_data: str | dict[str, Any] | None,
    owner_name: str,
    target_name: str,
    relationship_type: str = "relative",
) -> str:
    """
    Parses relationship data from Ancestry APIs and formats it into a readable path.
    Handles getladder API HTML/JSONP response.
    Uses format_name and ordinal_case from utils.py.

    The output format is standardized to match the unified format:

    ===Relationship Path to Owner Name===
    Target Name (birth-death) is Owner Name's relationship:

    - Person 1's relationship is Person 2 (birth-death)
    - Person 2's relationship is Person 3 (birth-death)
    ...
    """
    if not api_response_data:
        logger.warning("format_api_relationship_path: Received empty API response data.")
        return "(No relationship data received from API)"

    # Extract HTML and JSON from response
    html_content_raw, json_data = _extract_html_from_response(api_response_data)

    if html_content_raw is None and json_data is None:
        return "(Error parsing API response)"

    # Try Discovery API JSON format first
    json_result = _try_json_api_format(json_data, target_name, owner_name)
    if json_result:
        return json_result

    # Try HTML formats
    return _try_html_formats(html_content_raw, target_name, owner_name, relationship_type)


# === PERSON INFO EXTRACTION ===


def _extract_person_basic_info(
    indi: GedcomIndividualProtocol | Any,
) -> tuple[str, str | None, str | None, str | None]:
    """Extract basic information from a GEDCOM individual."""
    name = get_full_name(cast(Any, indi))

    birth_date_obj, _, _ = get_event_info(cast(Any, indi), TAG_BIRTH)
    death_date_obj, _, _ = get_event_info(cast(Any, indi), TAG_DEATH)

    birth_year = str(birth_date_obj.year) if birth_date_obj else None
    death_year = str(death_date_obj.year) if death_date_obj else None

    # Get gender
    sex_tag = indi.sub_tag(TAG_SEX)
    sex_char: str | None = None
    if sex_tag and hasattr(sex_tag, "value") and sex_tag.value is not None:
        sex_val = str(sex_tag.value).upper()
        if sex_val in {"M", "F"}:
            sex_char = sex_val

    return name, birth_year, death_year, sex_char


def _create_person_dict(
    name: str, birth_year: str | None, death_year: str | None, relationship: str | None, gender: str | None
) -> dict[str, str | None]:
    """Create a person dictionary for unified format."""
    return {
        "name": name,
        "birth_year": birth_year,
        "death_year": death_year,
        "relationship": relationship,
        "gender": gender,
    }


def _get_gendered_term(male_term: str, female_term: str, neutral_term: str, sex_char: str | None) -> str:
    """Get gendered relationship term based on sex character."""
    if sex_char == "M":
        return male_term
    if sex_char == "F":
        return female_term
    return neutral_term


def _determine_gedcom_relationship(
    prev_id: str,
    current_id: str,
    sex_char: str | None,
    reader: Any,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
) -> str:
    """Determine relationship between two individuals in GEDCOM path."""
    # Define relationship checks in order of priority
    relationship_checks = [
        (lambda: current_id in id_to_parents.get(prev_id, set()), "father", "mother", "parent"),
        (lambda: current_id in id_to_children.get(prev_id, set()), "son", "daughter", "child"),
        (lambda: are_siblings(prev_id, current_id, id_to_parents), "brother", "sister", "sibling"),
        (lambda: are_spouses(prev_id, current_id, reader), "husband", "wife", "spouse"),
        (lambda: is_grandparent(prev_id, current_id, id_to_parents), "grandfather", "grandmother", "grandparent"),
        (lambda: is_grandchild(prev_id, current_id, id_to_children), "grandson", "granddaughter", "grandchild"),
        (
            lambda: is_great_grandparent(prev_id, current_id, id_to_parents),
            "great-grandfather",
            "great-grandmother",
            "great-grandparent",
        ),
        (
            lambda: is_great_grandchild(prev_id, current_id, id_to_children),
            "great-grandson",
            "great-granddaughter",
            "great-grandchild",
        ),
        (lambda: is_aunt_or_uncle(prev_id, current_id, id_to_parents, id_to_children), "uncle", "aunt", "aunt/uncle"),
        (
            lambda: is_niece_or_nephew(prev_id, current_id, id_to_parents, id_to_children),
            "nephew",
            "niece",
            "niece/nephew",
        ),
        (lambda: are_cousins(prev_id, current_id, id_to_parents), "cousin", "cousin", "cousin"),
    ]

    for check_func, male_term, female_term, neutral_term in relationship_checks:
        if check_func():
            return _get_gendered_term(male_term, female_term, neutral_term, sex_char)

    return "relative"


# === PATH CONVERSION FUNCTIONS ===


def convert_gedcom_path_to_unified_format(
    path_ids: list[str],
    reader: Any,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
    indi_index: dict[str, GedcomIndividualProtocol | Any],
) -> list[dict[str, str | None]]:  # Value type changed to Optional[str]
    """
    Convert a GEDCOM relationship path to the unified format for relationship_path_unified.

    Args:
        path_ids: List of GEDCOM IDs representing the relationship path
        reader: GEDCOM reader object
        id_to_parents: Dictionary mapping IDs to parent IDs
        id_to_children: Dictionary mapping IDs to child IDs
        indi_index: Dictionary mapping IDs to GEDCOM individual objects

    Returns:
        List of dictionaries with keys 'name', 'birth_year', 'death_year', 'relationship', 'gender'
    """
    if not path_ids or len(path_ids) < 2:
        return []

    result: list[dict[str, str | None]] = []

    # Process the first person (no relationship)
    first_id = path_ids[0]
    first_indi = indi_index.get(first_id)

    if first_indi:
        name, birth_year, death_year, sex_char = _extract_person_basic_info(first_indi)
        result.append(_create_person_dict(name, birth_year, death_year, None, sex_char))
    else:
        result.append(_create_person_dict(f"Unknown ({first_id})", None, None, None, None))

    # Process the rest of the path
    for i in range(1, len(path_ids)):
        prev_id, current_id = path_ids[i - 1], path_ids[i]
        current_indi = indi_index.get(current_id)

        if not current_indi:
            result.append(_create_person_dict(f"Unknown ({current_id})", None, None, "relative", None))
            continue

        # Extract person info
        name, birth_year, death_year, sex_char = _extract_person_basic_info(current_indi)

        # Determine relationship
        relationship = _determine_gedcom_relationship(
            prev_id, current_id, sex_char, reader, id_to_parents, id_to_children
        )

        # Add to result
        result.append(_create_person_dict(name, birth_year, death_year, relationship, sex_char))

    return result


def _parse_discovery_relationship(relationship_text: str) -> tuple[str, str | None]:
    """Parse Discovery API relationship text to extract relationship term and gender."""
    rel_lower = relationship_text.lower()

    # Define relationship terms with their genders
    relationship_mappings = [
        ("daughter", "daughter", "F"),
        ("son", "son", "M"),
        ("father", "father", "M"),
        ("mother", "mother", "F"),
        ("brother", "brother", "M"),
        ("sister", "sister", "F"),
        ("husband", "husband", "M"),
        ("wife", "wife", "F"),
    ]

    # Check for specific relationship terms
    for keyword, term, gender in relationship_mappings:
        if keyword in rel_lower:
            return term, gender

    # Try to extract the relationship term from the text
    rel_match = re.search(r"(is|are) the (.*?) of", rel_lower)
    if rel_match:
        relationship_term = rel_match.group(2)
        # Determine gender from relationship term
        male_terms = ["son", "father", "brother", "husband"]
        female_terms = ["daughter", "mother", "sister", "wife"]
        gender = "M" if relationship_term in male_terms else "F" if relationship_term in female_terms else None
        return relationship_term, gender

    return "relative", None


def convert_discovery_api_path_to_unified_format(
    discovery_data: dict[str, Any], target_name: str
) -> list[dict[str, str | None]]:  # Value type changed to Optional[str]
    """
    Convert Discovery API relationship data to the unified format for relationship_path_unified.

    Args:
        discovery_data: Dictionary from Discovery API with 'path' key containing relationship steps
        target_name: Name of the target person

    Returns:
        List of dictionaries with keys 'name', 'birth_year', 'death_year', 'relationship', 'gender'
    """
    if not discovery_data or "path" not in discovery_data:
        logger.warning("Invalid or empty Discovery API data")
        return []

    path_steps = discovery_data.get("path", [])
    if not isinstance(path_steps, list) or not path_steps:
        logger.warning("Discovery API path is not a valid list or is empty")
        return []

    result: list[dict[str, str | None]] = []

    # Process the first person (target)
    target_name_display = format_name(target_name)
    result.append(_create_person_dict(target_name_display, None, None, None, None))

    # Process each step in the path
    for step in path_steps:
        if not isinstance(step, dict):
            logger.warning(f"Invalid path step: {step}")
            continue

        step_dict = cast(dict[str, Any], step)

        # Get name
        step_name = step_dict.get("name", "Unknown")
        current_name = format_name(step_name)

        # Parse relationship
        relationship_text = step_dict.get("relationship", "")
        relationship_term, gender = _parse_discovery_relationship(relationship_text)

        # Add to result
        result.append(_create_person_dict(current_name, None, None, relationship_term, gender))

    return result


# === GENDER INFERENCE ===


def _infer_gender_from_name(name: str) -> str | None:
    """Infer gender from name using common indicators."""
    name_lower = name.lower()

    # Male indicators
    male_indicators = [
        "mr.",
        "sir",
        "gordon",
        "james",
        "thomas",
        "alexander",
        "henry",
        "william",
        "robert",
        "richard",
        "david",
        "john",
        "michael",
        "george",
        "charles",
    ]
    if any(indicator in name_lower for indicator in male_indicators):
        _log_inferred_gender_once(name, 'name:M', f'Inferred male gender for {name} based on name')
        return "M"

    # Female indicators
    female_indicators = [
        "mrs.",
        "miss",
        "ms.",
        "lady",
        "catherine",
        "margaret",
        "mary",
        "jane",
        "elizabeth",
        "anne",
        "sarah",
        "emily",
        "charlotte",
        "victoria",
    ]
    if any(indicator in name_lower for indicator in female_indicators):
        _log_inferred_gender_once(name, 'name:F', f'Inferred female gender for {name} based on name')
        return "F"

    return None


def _infer_gender_from_relationship(name: str, relationship_text: str) -> str | None:
    """Infer gender from relationship text."""
    rel_lower = relationship_text.lower()

    # Male relationship terms
    male_terms = ["son", "father", "brother", "husband", "uncle", "grandfather", "nephew"]
    if any(term in rel_lower for term in male_terms):
        _log_inferred_gender_once(
            name, 'rel:M', f'Inferred male gender for {name} from relationship text: {relationship_text}'
        )
        return "M"

    # Female relationship terms
    female_terms = ["daughter", "mother", "sister", "wife", "aunt", "grandmother", "niece"]
    if any(term in rel_lower for term in female_terms):
        _log_inferred_gender_once(
            name, 'rel:F', f'Inferred female gender for {name} from relationship text: {relationship_text}'
        )
        return "F"

    return None


def _extract_years_from_lifespan(lifespan: str) -> tuple[str | None, str | None]:
    """Extract birth and death years from lifespan string."""
    if not lifespan:
        return None, None

    years_match = re.search(r"(\d{4})-(\d{4}|\bLiving\b|-)", lifespan, re.IGNORECASE)
    if not years_match:
        return None, None

    birth_year = years_match.group(1)
    death_year_raw = years_match.group(2)
    death_year = None if death_year_raw in {"-", "living", "Living"} else death_year_raw

    return birth_year, death_year


def _determine_gender_for_person(
    person_data: Mapping[str, Any],
    name: str,
    relationship_data: Sequence[Mapping[str, Any]] | None = None,
    index: int = 0,
) -> str | None:
    """Determine gender for a person using all available information."""
    # Check explicit gender field
    gender_raw = person_data.get("gender")
    gender = gender_raw.upper() if isinstance(gender_raw, str) else None
    if gender:
        return gender

    # Try to infer from name
    gender = _infer_gender_from_name(name)
    if gender:
        return gender

    # Try to infer from relationship text if available
    if relationship_data and index + 1 < len(relationship_data):
        rel_text = relationship_data[index + 1].get("relationship", "")
        gender = _infer_gender_from_relationship(name, rel_text)
        if gender:
            return gender

    # Special case for Gordon Milne
    if "gordon milne" in name.lower():
        logger.debug("Set gender to M for Gordon Milne")
        return "M"

    return None


def _parse_relationship_term_and_gender(
    relationship_text: str, person_data: dict[str, Any]
) -> tuple[str, str | None]:
    """Parse relationship term and infer gender from relationship text."""
    rel_lower = relationship_text.lower()

    # Check explicit gender first
    gender_raw = person_data.get("gender")
    gender = gender_raw.upper() if isinstance(gender_raw, str) else None

    # Relationship mapping: (term, gender)
    relationship_map = {
        "daughter": ("daughter", "F"),
        "son": ("son", "M"),
        "father": ("father", "M"),
        "mother": ("mother", "F"),
        "brother": ("brother", "M"),
        "sister": ("sister", "F"),
        "husband": ("husband", "M"),
        "wife": ("wife", "F"),
        "uncle": ("uncle", "M"),
        "aunt": ("aunt", "F"),
        "grandfather": ("grandfather", "M"),
        "grandmother": ("grandmother", "F"),
        "nephew": ("nephew", "M"),
        "niece": ("niece", "F"),
    }

    for term, (relationship, inferred_gender) in relationship_map.items():
        if term in rel_lower:
            if not gender:
                gender = inferred_gender
            return relationship, gender

    return "relative", gender


def _process_path_person(person_data: dict[str, Any]) -> dict[str, str | None]:
    """Process a single person in the relationship path."""
    # Get and clean name
    current_name = format_name(person_data.get("name", "Unknown"))
    current_name = re.sub(r"\s+\d{4}.*$", "", current_name)  # Remove year suffixes

    # Extract birth/death years
    item_birth_year, item_death_year = _extract_years_from_lifespan(person_data.get("lifespan", ""))

    # Parse relationship and gender
    relationship_text = person_data.get("relationship", "")
    relationship_term, item_gender = _parse_relationship_term_and_gender(relationship_text, person_data)

    # If still no gender, try to infer from name
    if not item_gender:
        item_gender = _infer_gender_from_name(current_name)

    # Try to extract relationship term from text if still "relative"
    if relationship_term == "relative" and relationship_text:
        rel_match = re.search(r"(is|are) the (.*?) of", relationship_text.lower())
        if rel_match:
            relationship_term = rel_match.group(2)

    return {
        "name": current_name,
        "birth_year": item_birth_year,
        "death_year": item_death_year,
        "relationship": relationship_term,
        "gender": item_gender,
    }


def convert_api_path_to_unified_format(
    relationship_data: list[dict[str, Any]], target_name: str
) -> list[dict[str, str | None]]:  # Value type changed to Optional[str]
    """
    Convert API relationship data to the unified format for relationship_path_unified.

    Args:
        relationship_data: List of dictionaries from API with keys 'name', 'relationship', 'lifespan'
        target_name: Name of the target person

    Returns:
        List of dictionaries with keys 'name', 'birth_year', 'death_year', 'relationship', 'gender'
    """
    if not relationship_data:
        return []

    result: list[dict[str, str | None]] = []  # Ensure list type

    # Process the first person (target)
    first_person = relationship_data[0]
    target_name_display = format_name(first_person.get("name", target_name))

    # Extract birth/death years
    birth_year, death_year = _extract_years_from_lifespan(first_person.get("lifespan", ""))

    # Determine gender
    gender = _determine_gender_for_person(first_person, target_name_display, relationship_data, 0)

    # Add first person to result
    result.append(
        {
            "name": target_name_display,
            "birth_year": birth_year,
            "death_year": death_year,
            "relationship": None,  # First person has no relationship to previous person
            "gender": gender,  # Add gender information
        }
    )

    # Process the rest of the path
    for i in range(1, len(relationship_data)):
        person_entry = _process_path_person(relationship_data[i])
        result.append(person_entry)

    return result


# === DISPLAY FORMATTING ===


def _format_years_display(birth_year: str | None, death_year: str | None) -> str:
    """Format birth/death years into display string."""
    if birth_year and death_year:
        return f" ({birth_year}-{death_year})"
    if birth_year:
        return f" (b. {birth_year})"
    if death_year:
        return f" (d. {death_year})"
    return ""


def _clean_name_format(name: str) -> str:
    """Remove Name('...') wrapper if present."""
    if "Name(" not in name:
        return name

    # Try different regex patterns to handle various Name formats
    name_clean = re.sub(r"Name\(['\"]([^'\"]+)['\"]\)", r"\1", name)
    name_clean = re.sub(r"Name\('([^']+)'\)", r"\1", name_clean)
    return re.sub(r'Name\("([^"]+)"\)', r"\1", name_clean)


def _check_uncle_aunt_pattern_sibling(path_data: Sequence[Mapping[str, Any]]) -> str | None:
    """Check for Uncle/Aunt pattern: Target's sibling is parent of owner."""
    if len(path_data) < 3:
        return None

    if path_data[1].get("relationship") in {"brother", "sister"} and path_data[2].get("relationship") in {
        "son",
        "daughter",
    }:
        gender_val = path_data[0].get("gender")
        gender_str = str(gender_val) if gender_val is not None else ""
        return "Uncle" if gender_str.upper() == "M" else "Aunt"

    return None


def _check_uncle_aunt_pattern_parent(path_data: Sequence[Mapping[str, Any]]) -> str | None:
    """Check for Uncle/Aunt pattern: Through parent."""
    if len(path_data) < 3:
        return None

    if path_data[1].get("relationship") in {"father", "mother"} and path_data[2].get("relationship") in {
        "son",
        "daughter",
    }:
        gender_val = path_data[0].get("gender")
        gender_str = str(gender_val) if gender_val is not None else ""
        if gender_str.upper() == "M":
            return "Uncle"
        if gender_str.upper() == "F":
            return "Aunt"
        return "Aunt/Uncle"

    return None


def _check_grandparent_pattern(path_data: Sequence[Mapping[str, Any]]) -> str | None:
    """Check for Grandparent pattern: Target's child is parent of owner."""
    if len(path_data) < 3:
        return None

    if path_data[1].get("relationship") in {"son", "daughter"} and path_data[2].get("relationship") in {
        "son",
        "daughter",
    }:
        # Determine gender
        name = path_data[0].get("name", "")
        gender = _determine_gender_for_person(path_data[0], str(name))

        logger.debug(f"Grandparent relationship: name={name}, gender={gender}, raw gender={path_data[0].get('gender')}")

        if gender == "M":
            return "Grandfather"
        if gender == "F":
            return "Grandmother"
        return "Grandparent"

    return None


def _check_cousin_pattern(path_data: Sequence[Mapping[str, Any]]) -> str | None:
    """Check for Cousin pattern: Target's parent's sibling's child is owner."""
    if len(path_data) < 4:
        return None

    if (
        path_data[1].get("relationship") in {"father", "mother"}
        and path_data[2].get("relationship") in {"brother", "sister"}
        and path_data[3].get("relationship") in {"son", "daughter"}
    ):
        return "Cousin"

    return None


def _check_nephew_niece_pattern(path_data: Sequence[Mapping[str, Any]]) -> str | None:
    """Check for Nephew/Niece pattern: Target's parent's child is owner."""
    if len(path_data) < 3:
        return None

    if path_data[1].get("relationship") in {"father", "mother"} and path_data[2].get("relationship") in {
        "son",
        "daughter",
    }:
        gender_val = path_data[0].get("gender")
        target_gender = str(gender_val).upper() if gender_val is not None else ""

        if target_gender == "M":
            return "Nephew"
        if target_gender == "F":
            return "Niece"
        return "Nephew/Niece"

    return None


def _determine_relationship_type_from_path(path_data: Sequence[Mapping[str, Any]]) -> str | None:
    """Determine relationship type by checking various patterns."""
    if len(path_data) < 3:
        return None

    # Try each pattern in order
    patterns = [
        _check_uncle_aunt_pattern_sibling,
        _check_uncle_aunt_pattern_parent,
        _check_grandparent_pattern,
        _check_cousin_pattern,
        _check_nephew_niece_pattern,
    ]

    for pattern_func in patterns:
        result = pattern_func(path_data)
        if result:
            return result

    return None


def _convert_you_are_relationship(relationship: str, current_name: str, next_name: str, next_years: str) -> str:
    """Convert 'You are...' relationship to inverse form."""
    # Extract the relationship type (e.g., "son", "daughter")
    rel_type = relationship.replace("You are the ", "").replace(f" of {current_name}", "").strip()
    # Convert to inverse relationship
    inverse_rel = {
        "son": "father",
        "daughter": "mother",
        "grandson": "grandfather",
        "granddaughter": "grandmother",
    }.get(rel_type, "parent")
    return f"   - {current_name} is the {inverse_rel} of {next_name}{next_years}"


def _format_path_step(
    current_person: Mapping[str, Any],
    next_person: Mapping[str, Any],
    seen_names: set[str],
) -> tuple[str, set[str]]:
    """Format a single step in the relationship path using possessive format."""
    # Get names and clean them
    current_name = current_person.get("name", "Unknown")
    current_name_clean = _clean_name_format(str(current_name))
    next_name = next_person.get("name", "Unknown")
    next_name_clean = _clean_name_format(str(next_name))

    # Get relationship
    relationship = next_person.get("relationship", "relative") or "relative"

    # Format years for next person - only if we haven't seen this name before
    next_years = ""
    if next_name_clean.lower() not in seen_names:
        next_years = _format_years_display(next_person.get("birth_year"), next_person.get("death_year"))
        seen_names.add(next_name_clean.lower())

    # Get first name for possessive form
    first_name = current_name_clean.split()[0] if current_name_clean else "Unknown"
    possessive = f"{first_name}'s" if not first_name.endswith('s') else f"{first_name}'"

    # Handle "You are..." relationships specially (convert to inverse relationship)
    if isinstance(relationship, str) and relationship.startswith("You are the "):
        line = _convert_you_are_relationship(relationship, current_name_clean, next_name_clean, next_years)
    else:
        # Possessive relationship format: "Peter's father is William Fraser (1818-1898)"
        line = f"  - {possessive} {relationship} is {next_name_clean}{next_years}"

    return line, seen_names


def format_relationship_path_unified(
    path_data: Sequence[Mapping[str, str | None]],  # Value type changed to Optional[str]
    target_name: str,
    owner_name: str,
    relationship_type: str | None = None,
) -> str:
    """
    Format a relationship path using the unified format for both GEDCOM and API data.

    Uses narrative format with possessive relationships:
    "Relationship between Peter Fraser (1893-1915) and Wayne Gordon Gault (1969-).
    Peter is Wayne's 1st cousin 4x removed:
      - Peter's father is William Fraser (1818-1898)
      - William's father is John Fraser (1791-1840)"

    Args:
        path_data: List of dictionaries with keys 'name', 'birth_year', 'death_year', 'relationship', 'gender'
                  Each entry represents a person in the path
        target_name: Name of the target person (first person in the path)
        owner_name: Name of the owner/reference person (last person in the path)
        relationship_type: Type of relationship between target and owner (default: None, will be determined)

    Returns:
        Formatted relationship path string
    """
    if not path_data or len(path_data) < 2:
        return f"(No relationship path data available for {target_name})"

    # Get first and last person details
    first_person = path_data[0]
    first_name = _clean_name_format(str(first_person.get("name", target_name)))
    first_years = _format_years_display(first_person.get("birth_year"), first_person.get("death_year"))

    # Get owner's first name for possessive
    owner_first_name = owner_name.split(maxsplit=1)[0] if owner_name else "Owner"
    owner_possessive = f"{owner_first_name}'s" if not owner_first_name.endswith('s') else f"{owner_first_name}'"

    # Get target's first name for subject
    target_first_name = first_name.split()[0] if first_name else "Person"

    # Determine the specific relationship type if not provided
    if relationship_type is None or relationship_type == "relative":
        relationship_type = _determine_relationship_type_from_path(path_data) or "relative"

    # Narrative header showing both people and their relationship
    header = f"Relationship between {first_name}{first_years} and {owner_name}.\n{target_first_name} is {owner_possessive} {relationship_type}:"

    # Format each step in the path with indentation
    path_lines: list[str] = []

    # Keep track of names we've already seen to avoid adding years multiple times
    seen_names = {first_name.lower()}

    # Process path steps using possessive format
    for i in range(len(path_data) - 1):
        current_person = path_data[i]
        next_person = path_data[i + 1]
        line, seen_names = _format_path_step(current_person, next_person, seen_names)
        path_lines.append(line)

    # Combine all parts
    return f"{header}\n" + "\n".join(path_lines)


def _get_relationship_term(gender: str | None, relationship_code: str) -> str:
    """
    Convert a relationship code to a human-readable term.

    Args:
        gender: Gender of the person (M, F, or None)
        relationship_code: Relationship code from the API

    Returns:
        Human-readable relationship term
    """
    relationship_code_lower = relationship_code.lower()

    # Define relationship mappings: (code/keyword, male_term, female_term, neutral_term, is_exact_match)
    relationship_mappings = [
        ("parent", "father", "mother", "parent", True),
        ("child", "son", "daughter", "child", True),
        ("spouse", "husband", "wife", "spouse", True),
        ("sibling", "brother", "sister", "sibling", True),
        ("grandparent", "grandfather", "grandmother", "grandparent", False),
        ("grandchild", "grandson", "granddaughter", "grandchild", False),
        ("aunt", "uncle", "aunt", "aunt/uncle", False),
        ("uncle", "uncle", "aunt", "aunt/uncle", False),
        ("niece", "nephew", "niece", "niece/nephew", False),
        ("nephew", "nephew", "niece", "niece/nephew", False),
        ("cousin", "cousin", "cousin", "cousin", False),
    ]

    for code, male_term, female_term, neutral_term, is_exact in relationship_mappings:
        if (is_exact and relationship_code_lower == code) or (not is_exact and code in relationship_code_lower):
            return _get_gendered_term(male_term, female_term, neutral_term, gender)

    return relationship_code  # Return original if no match


# ==============================================
# Module Tests
# ==============================================


def relationship_formatting_module_tests() -> bool:
    """Test suite for relationship_formatting.py - path formatting and label conversion."""
    from testing.test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Relationship Formatting", "relationship_formatting.py")
        suite.start_suite()

    def test_api_path_conversion():
        """Test API path conversion to unified format."""
        sample_path = [
            {"name": "Target Example", "relationship": "", "lifespan": "1985-"},
            {"name": "Parent Example", "relationship": "They are your father", "lifespan": "1955-2010"},
            {"name": "Owner Example", "relationship": "You are their son", "lifespan": "2005-"},
        ]
        unified = convert_api_path_to_unified_format(sample_path, "Target Example")
        assert len(unified) == 3, "Converted path should include every hop"
        assert unified[0]["relationship"] is None, "First entry is the target"
        assert unified[1]["relationship"] == "father", "Second hop should normalize to father"

    def test_relationship_terms():
        """Test relationship term mapping."""
        test_cases = [
            ("M", "parent", "father"),
            ("F", "parent", "mother"),
            ("M", "child", "son"),
            ("F", "child", "daughter"),
            (None, "parent", "parent"),
        ]
        for gender, relationship, expected in test_cases:
            result = _get_relationship_term(gender, relationship)
            assert result == expected, f"Term for {gender}/{relationship} should be {expected}"

    def test_unified_formatting():
        """Test unified path formatting."""
        mock_path = [
            {"name": "Target", "birth_year": "1950", "death_year": "2010", "relationship": None, "gender": "M"},
            {"name": "Parent", "birth_year": "1920", "death_year": "1990", "relationship": "father", "gender": "M"},
            {"name": "Owner", "birth_year": "1985", "death_year": None, "relationship": "son", "gender": "M"},
        ]
        narrative = format_relationship_path_unified(list(mock_path), "Target", "Owner", "grandfather")
        assert "Relationship between Target" in narrative, "Header should mention target"
        assert "Owner" in narrative, "Owner name should be referenced"

    def test_years_display():
        """Test years display formatting."""
        assert _format_years_display("1900", "1980") == " (1900-1980)"
        assert _format_years_display("1900", None) == " (b. 1900)"
        assert _format_years_display(None, "1980") == " (d. 1980)"
        assert _format_years_display(None, None) == ""  # noqa: PLC1901

    suite.run_test("API path conversion", test_api_path_conversion)
    suite.run_test("Relationship terms", test_relationship_terms)
    suite.run_test("Unified formatting", test_unified_formatting)
    suite.run_test("Years display", test_years_display)

    return suite.finish_suite()


# Use centralized test runner utility
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(relationship_formatting_module_tests)

# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    import sys

    print("📝 Running Relationship Formatting test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
