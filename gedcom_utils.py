# gedcom_utils.py
"""
GEDCOM-related helper functions and utilities.
"""
import re
import logging
from typing import Optional, Dict
from config import config_instance
from logging_config import logger

# --- GEDCOM library availability and imports ---
try:
    from ged4py.parser import GedcomReader
    from ged4py.model import Individual, Record, Name

    GEDCOM_LIB_AVAILABLE = True
except ImportError:
    GedcomReader = None
    Individual = None
    Record = None
    Name = None
    GEDCOM_LIB_AVAILABLE = False


def _is_individual(obj):
    """Checks if object is an Individual safely handling None values"""
    return obj is not None and type(obj).__name__ == "Individual"


def _normalize_id(xref_id: Optional[str]) -> Optional[str]:
    """Normalizes INDI/FAM etc IDs (e.g., '@I123@' -> 'I123')."""
    if xref_id and isinstance(xref_id, str):
        match = re.match(r"^@?([IFSTNMCXO][0-9A-Z\-]+)@?$", xref_id.strip().upper())
        if match:
            return match.group(1)
    return None


def _get_full_name(indi) -> str:
    """Safely gets formatted name using Name.format(). Handles None/errors."""
    if not _is_individual(indi):
        return "Unknown (Not Individual)"
    try:
        name_rec = indi.name  # Access the name record
        if hasattr(name_rec, "format"):
            formatted_name = name_rec.format()
            cleaned_name = " ".join(formatted_name.split()).title()
            cleaned_name = re.sub(r"\s*/([^/]+)/\s*$", r" \1", cleaned_name).strip()
            return cleaned_name if cleaned_name else "Unknown (Empty Name)"
        elif name_rec is None:
            return "Unknown (No Name Tag)"
        else:
            indi_id_log = (
                _normalize_id(indi.xref_id)
                if hasattr(indi, "xref_id") and indi.xref_id
                else "Unknown ID"
            )
            logger.warning(
                f"Indi @{indi_id_log}@ unexpected .name type: {type(name_rec)}"
            )
            return f"Unknown (Type {type(name_rec).__name__})"
    except AttributeError:
        return "Unknown (Attr Error)"
    except Exception as e:
        indi_id_log = (
            _normalize_id(indi.xref_id)
            if hasattr(indi, "xref_id") and indi.xref_id
            else "Unknown ID"
        )
        logger.error(
            f"Error formatting name for @{indi_id_log}@: {e}",
            exc_info=False,
        )
        return "Unknown (Error)"


def format_life_dates(indi) -> str:
    """Returns a formatted string with birth and death dates."""
    b_date_obj, b_date_str, b_place = _get_event_info(indi, "BIRT")
    d_date_obj, d_date_str, d_place = _get_event_info(indi, "DEAT")
    b_date_str_cleaned = _clean_display_date(b_date_str)
    d_date_str_cleaned = _clean_display_date(d_date_str)
    birth_info = f"b. {b_date_str_cleaned}" if b_date_str_cleaned != "N/A" else ""
    death_info = f"d. {d_date_str_cleaned}" if d_date_str_cleaned != "N/A" else ""
    life_parts = [info for info in [birth_info, death_info] if info]
    return f" ({', '.join(life_parts)})" if life_parts else ""


def format_relative_info(relative) -> str:
    """Formats information about a relative (name and life dates) for display."""
    if not _is_individual(relative):
        return "  - (Invalid Relative Data)"
    rel_name = _get_full_name(relative)
    life_info = format_life_dates(relative)
    return f"  - {rel_name}{life_info}"


def _find_family_records_where_individual_is_child(reader, target_id):
    """Helper function to find family records where an individual is listed as a child."""
    parent_families = []
    for family_record in reader.records0("FAM"):
        if not _is_record(family_record):
            continue
        children_in_fam = family_record.sub_tags("CHIL")
        if children_in_fam:
            for child in children_in_fam:
                if _is_individual(child) and _normalize_id(child.xref_id) == target_id:
                    parent_families.append(family_record)
                    break
    return parent_families


def _find_family_records_where_individual_is_parent(reader, target_id):
    """Helper function to find family records where an individual is listed as a parent (HUSB or WIFE)."""
    parent_families = []
    for family_record in reader.records0("FAM"):
        if not _is_record(family_record) or not family_record.xref_id:
            continue
        husband = family_record.sub_tag("HUSB")
        wife = family_record.sub_tag("WIFE")
        is_target_husband = (
            _is_individual(husband) and _normalize_id(husband.xref_id) == target_id
        )
        is_target_wife = (
            _is_individual(wife) and _normalize_id(wife.xref_id) == target_id
        )
        if is_target_husband or is_target_wife:
            parent_families.append((family_record, is_target_husband, is_target_wife))
    return parent_families


def get_related_individuals(reader, individual, relationship_type: str) -> list:
    """Gets parents, spouses, children, or siblings using family record lookups."""
    related_individuals = []
    unique_related_ids = set()
    if not reader:
        logging.error("get_related_individuals: No reader.")
        return related_individuals
    if not _is_individual(individual) or not individual.xref_id:
        logging.warning(f"get_related_individuals: Invalid input individual.")
        return related_individuals
    target_id = _normalize_id(individual.xref_id)
    if not target_id:
        logging.warning(
            f"get_related_individuals: Cannot normalize target ID {individual.xref_id}"
        )
        return related_individuals
    try:
        if relationship_type == "parents":
            parent_families = _find_family_records_where_individual_is_child(
                reader, target_id
            )
            potential_parents = []
            for family_record in parent_families:
                husband = family_record.sub_tag("HUSB")
                wife = family_record.sub_tag("WIFE")
                if _is_individual(husband):
                    potential_parents.append(husband)
                if _is_individual(wife):
                    potential_parents.append(wife)
            for parent in potential_parents:
                if parent is not None and hasattr(parent, "xref_id") and parent.xref_id:
                    parent_id = _normalize_id(parent.xref_id)
                    if parent_id and parent_id not in unique_related_ids:
                        related_individuals.append(parent)
                        unique_related_ids.add(parent_id)
        elif relationship_type == "siblings":
            parent_families = _find_family_records_where_individual_is_child(
                reader, target_id
            )
            potential_siblings = []
            for fam in parent_families:
                fam_children = fam.sub_tags("CHIL")
                if fam_children:
                    potential_siblings.extend(
                        c for c in fam_children if _is_individual(c)
                    )
            for sibling in potential_siblings:
                if (
                    sibling is not None
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
            parent_families = _find_family_records_where_individual_is_parent(
                reader, target_id
            )
            if relationship_type == "spouses":
                for family_record, is_target_husband, is_target_wife in parent_families:
                    other_spouse = None
                    if is_target_husband:
                        other_spouse = family_record.sub_tag("WIFE")
                    elif is_target_wife:
                        other_spouse = family_record.sub_tag("HUSB")
                    if (
                        other_spouse is not None
                        and _is_individual(other_spouse)
                        and hasattr(other_spouse, "xref_id")
                        and other_spouse.xref_id
                    ):
                        spouse_id = _normalize_id(other_spouse.xref_id)
                        if spouse_id and spouse_id not in unique_related_ids:
                            related_individuals.append(other_spouse)
                            unique_related_ids.add(spouse_id)
            else:  # relationship_type == "children"
                for family_record, _, _ in parent_families:
                    children_list = family_record.sub_tags("CHIL")
                    if children_list:
                        for child in children_list:
                            if (
                                child is not None
                                and _is_individual(child)
                                and hasattr(child, "xref_id")
                                and child.xref_id
                            ):
                                child_id = _normalize_id(child.xref_id)
                                if child_id and child_id not in unique_related_ids:
                                    related_individuals.append(child)
                                    unique_related_ids.add(child_id)
        else:
            logging.warning(
                f"Unknown relationship type requested: '{relationship_type}'"
            )
    except AttributeError as ae:
        logging.error(
            f"AttributeError finding {relationship_type} for {target_id}: {ae}",
            exc_info=True,
        )
    except Exception as e:
        logging.error(
            f"Unexpected error finding {relationship_type} for {target_id}: {e}",
            exc_info=True,
        )
    related_individuals.sort(key=lambda x: (_normalize_id(x.xref_id) or ""))
    return related_individuals


def find_potential_matches(
    reader,
    first_name: Optional[str],
    surname: Optional[str],
    dob_str: Optional[str],
    pob: Optional[str],
    dod_str: Optional[str],
    pod: Optional[str],
    gender: Optional[str] = None,
):
    """
    Finds potential matches in GEDCOM based on various criteria including death info.
    Prioritizes name matches.
    """
    matches = []
    for indi in reader.records0("INDI"):
        indi_full_name = _get_full_name(indi)
        birth_date_str_ged = None
        birth_place_str_ged = None
        death_date_str_ged = None
        death_place_str_ged = None
        for event in getattr(indi, "events", []):
            etype = getattr(event, "tag", "").lower()
            if etype == "birth":
                birth_date_str_ged = getattr(event, "date", None)
                birth_place_str_ged = getattr(event, "place", None)
            elif etype == "death":
                death_date_str_ged = getattr(event, "date", None)
                death_place_str_ged = getattr(event, "place", None)
        score = 0
        reasons = []
        if first_name and first_name.lower() in indi_full_name.lower():
            score += 10
            reasons.append("First name match")
        if surname and surname.lower() in indi_full_name.lower():
            score += 10
            reasons.append("Surname match")
        if dob_str and birth_date_str_ged and dob_str in str(birth_date_str_ged):
            score += 8
            reasons.append(f"Birth date match ({birth_date_str_ged})")
        if (
            pob
            and birth_place_str_ged
            and pob.lower() in str(birth_place_str_ged).lower()
        ):
            score += 4
            reasons.append(f"Place of birth match ({birth_place_str_ged})")
        if dod_str and death_date_str_ged and dod_str in str(death_date_str_ged):
            score += 4
            reasons.append(f"Death date match ({death_date_str_ged})")
        if (
            pod
            and death_place_str_ged
            and pod.lower() in str(death_place_str_ged).lower()
        ):
            score += 2
            reasons.append(f"Place of death match ({death_place_str_ged})")
        matches.append(
            {
                "id": getattr(indi, "xref_id", None),
                "name": indi_full_name,
                "birth_date": birth_date_str_ged,
                "birth_place": birth_place_str_ged,
                "death_date": death_date_str_ged,
                "death_place": death_place_str_ged,
                "score": score,
                "reasons": ", ".join(reasons),
            }
        )
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches


def display_gedcom_family_details(reader, individual):
    """Helper function to display formatted family details from GEDCOM data."""
    if not individual:
        print("No individual selected.")
        return
    print(f"\nFamily for {getattr(individual, 'name', '(Unknown)')}:\n")
    # Parents
    parents = []
    for fam in getattr(individual, "families_as_child", []):
        father = getattr(fam, "husband", None)
        mother = getattr(fam, "wife", None)
        if father:
            parents.append(father)
        if mother:
            parents.append(mother)
    print("Parents:")
    if parents:
        for p in parents:
            print(f"  - {getattr(p, 'name', '(Unknown)')}")
    else:
        print("  (None found)")
    # Spouses
    spouses = []
    for fam in getattr(individual, "families_as_parent", []):
        spouse = (
            getattr(fam, "wife", None)
            if getattr(individual, "sex", None) == "M"
            else getattr(fam, "husband", None)
        )
        if spouse:
            spouses.append(spouse)
    print("Spouses:")
    if spouses:
        for s in spouses:
            print(f"  - {getattr(s, 'name', '(Unknown)')}")
    else:
        print("  (None found)")
    # Children
    children = []
    for fam in getattr(individual, "families_as_parent", []):
        children.extend(getattr(fam, "children", []))
    print("Children:")
    if children:
        for c in children:
            print(f"  - {getattr(c, 'name', '(Unknown)')}")
    else:
        print("  (None found)")


# Helper for family record type check (used above)
def _is_record(obj):
    return obj is not None and type(obj).__name__ == "Record"


def _get_event_info(indi, event_tag):
    """Extracts (date_obj, date_str, place) for a given event tag (e.g., 'BIRT', 'DEAT')."""
    for event in getattr(indi, "events", []):
        if getattr(event, "tag", None) == event_tag:
            return (
                getattr(event, "date_obj", None),
                getattr(event, "date", None),
                getattr(event, "place", None),
            )
    return None, None, None


def _clean_display_date(date_str):
    """Cleans and formats a date string for display."""
    if not date_str:
        return "N/A"
    return str(date_str).replace("ABT ", "~").replace("BEF ", "<").replace("AFT ", ">")


if __name__ == "__main__":
    print("gedcom_utils.py loaded successfully.")
    print("Available functions:")
    print(" - _is_individual(obj)")
    print(" - _normalize_id(xref_id)")
    print(" - _get_full_name(indi)")
    print(" - format_life_dates(indi)")
    print(" - format_relative_info(relative)")
    print(" - get_related_individuals(reader, individual, relationship_type)")
    print(" - find_potential_matches(...)")
    print(" - display_gedcom_family_details(reader, individual)")
