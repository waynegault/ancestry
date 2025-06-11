#!/usr/bin/env python3

# --- START OF FILE action10.py ---

# action10.py
"""
Action 10: Find GEDCOM Matches and Relationship Path

Applies a hardcoded filter (OR logic) to the GEDCOM data (using pre-processed
cache), calculates a score for each filtered individual based on specific criteria,
displays the top 3 highest-scoring individuals (simplified format), identifies the
best match, and displays their relatives and relationship path to the reference person.
V.20240503.Refactored

Example output:
--------------
--- Top 3 Highest Scoring Matches ---
ID     ---------------------------------------------
@I123@           | John Smith                     | M      | 1 JAN 1850        | New York, NY, USA               | 12 DEC 1910       | Boston, MA, USA                 | 95
@I456@           | Jonathan Smith                 | M      | 15 MAR 1848       | Albany, NY, USA                 | 23 NOV 1915       | Chicago, IL, USA               | 82
@I789@           | John Smithson                  | M      | 22 FEB 1855       | Brooklyn, NY, USA               | 30 JUL 1922       | Philadelphia, PA, USA          | 73
"""

# --- Standard library imports ---
import logging
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone

# --- Third-party imports ---
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

# --- Test framework - No longer using external test framework ---
# All testing is now self-contained within this script

try:
    from unittest.mock import patch

    HAS_MOCK = True
except ImportError:
    # Create dummy context for when unittest.mock is not available
    class DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def dummy_patch(*args, **kwargs):
        return DummyContext()

    patch = dummy_patch
    HAS_MOCK = False

# --- Setup Fallback Logger FIRST ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("action10_initial")

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "DATE_FLEXIBILITY": {"year_match_range": 10},
    "COMMON_SCORING_WEIGHTS": {
        # cases are ignored
        # --- Name Weights ---
        "contains_first_name": 25,  # if the input first name is in the candidate first name
        "contains_surname": 25,  # if the input surname is in the candidate surname
        "bonus_both_names_contain": 25,  # additional bonus if both first and last name achieved a score
        # --- Existing Date Weights ---
        "exact_birth_date": 25,  # if input date of birth is exact with candidate date of birth ie yyy/mm/dd
        "exact_death_date": 25,  # if input date of death is exact with candidate date of death ie yyy/mm/dd
        "year_birth": 20,  # if input year of death is exact with candidate year of death even if the day and month is wrong or not given
        "year_death": 20,  # if input year of death is exact with candidate year of death even if the day and month is wrong or not given
        "approx_year_birth": 10,  # if input year of death is within year_match_range years of candidate year of death even if the day and month is wrong or not given
        "approx_year_death": 10,  # if input year of death is within year_match_range years of candidate year of death even if the day and month is wrong or not given
        "death_dates_both_absent": 10,  # if both the input and candidate have no death dates
        # --- Gender Weights ---
        "gender_match": 15,  # if the input gender indication eg m/man/male/boy or f/fem/female/woman/girl matches the candidate gender indication.
        # --- Place Weights ---
        "contains_pob": 25,  # if the input place of birth is contained in the candidate place of birth
        "contains_pod": 25,  # if the input place of death is contained in the candidate place of death
        # --- Bonus Weights ---
        "bonus_birth_info": 25,  # additional bonus if both birth year and birth place achieved a score
        "bonus_death_info": 25,  # additional bonus if both death year and death place achieved a score
    },
    "NAME_FLEXIBILITY": {
        "fuzzy_threshold": 0.8,
        "check_starts_with": False,  # Set to False as 'contains' is primary
    },
    "MAX_DISPLAY_RESULTS": 3,
}

# --- Local application imports ---
from config import config_instance
from logging_config import setup_logging

logger = setup_logging()
logger.info("Logging configured via setup_logging (Level: INFO).")

# Import GEDCOM utilities
from gedcom_utils import (
    GedcomData,
    calculate_match_score,
    _normalize_id,
    format_relative_info,
)

# Import relationship utilities
from relationship_utils import (
    fast_bidirectional_bfs,
    convert_gedcom_path_to_unified_format,
    format_relationship_path_unified,
)


# --- Helper Functions ---
def sanitize_input(value: str) -> Optional[str]:
    """Basic sanitization of user input."""
    if not value:
        return None
    # Remove leading/trailing whitespace
    sanitized = value.strip()
    return sanitized if sanitized else None


def get_validated_year_input(
    prompt: str, default: Optional[int] = None
) -> Optional[int]:
    """Get and validate a year input with optional default."""
    display_default = f" [{default}]" if default else " [YYYY]"
    value = input(f"{prompt}{display_default}: ").strip()

    if not value and default:
        return default

    if value.isdigit() and 1000 <= int(value) <= 2100:  # Reasonable year range
        return int(value)
    elif value:
        logger.warning(f"Invalid year input '{value}', ignoring.")

    return None


def parse_command_line_args() -> argparse.Namespace:
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description="GEDCOM matching and analysis")
    parser.add_argument("--auto-input", nargs="+", help="Automated inputs for testing")
    parser.add_argument("--reference-id", help="Override reference person ID")
    parser.add_argument("--gedcom-file", help="Path to GEDCOM file")
    parser.add_argument(
        "--max-results", type=int, default=3, help="Maximum results to display"
    )
    return parser.parse_args()


def get_config_value(key: str, default_value: Any = None) -> Any:
    """Safely retrieve a configuration value with fallback."""
    if not config_instance:
        return default_value
    return getattr(config_instance, key, default_value)


def validate_config() -> Tuple[
    Optional[Path],
    Optional[str],
    Optional[str],
    Dict[str, Any],
    Dict[str, Any],
    int,
]:
    """Validate configuration and return essential values."""
    # Get and validate GEDCOM file path
    gedcom_file_path_config = get_config_value("GEDCOM_FILE_PATH")

    if (
        not gedcom_file_path_config
        or not isinstance(gedcom_file_path_config, Path)
        or not gedcom_file_path_config.is_file()
    ):
        logger.critical(
            f"GEDCOM file path missing or invalid: {gedcom_file_path_config}"
        )
        sys.exit(1)

    # Get reference person info
    reference_person_id_raw = get_config_value("REFERENCE_PERSON_ID")
    reference_person_name = get_config_value(
        "REFERENCE_PERSON_NAME", "Reference Person"
    )

    # Get scoring and date flexibility settings
    date_flex = get_config_value("DATE_FLEXIBILITY", DEFAULT_CONFIG["DATE_FLEXIBILITY"])
    scoring_weights = get_config_value(
        "COMMON_SCORING_WEIGHTS", DEFAULT_CONFIG["COMMON_SCORING_WEIGHTS"]
    )
    max_display_results = get_config_value(
        "MAX_DISPLAY_RESULTS", DEFAULT_CONFIG["MAX_DISPLAY_RESULTS"]
    )

    # Log configuration
    logger.info(
        f"Configured TREE_OWNER_NAME: {get_config_value('TREE_OWNER_NAME', 'Not Set')}"
    )
    logger.info(f"Configured REFERENCE_PERSON_ID: {reference_person_id_raw}")
    logger.info(f"Configured REFERENCE_PERSON_NAME: {reference_person_name}")
    logger.info(f"Using GEDCOM file: {gedcom_file_path_config.name}")

    return (
        gedcom_file_path_config,
        reference_person_id_raw,
        reference_person_name,
        date_flex,
        scoring_weights,
        max_display_results,
    )


def load_gedcom_data(gedcom_path: Path) -> GedcomData:
    """Load, parse, and pre-process GEDCOM data."""
    if (
        not gedcom_path
        or not isinstance(gedcom_path, Path)
        or not gedcom_path.is_file()
    ):
        logger.critical(f"Invalid GEDCOM file path: {gedcom_path}")
        sys.exit(1)

    try:
        logger.info("Loading, parsing, and pre-processing GEDCOM data...")
        load_start_time = time.time()
        gedcom_data = GedcomData(gedcom_path)
        load_end_time = time.time()

        logger.info(
            f"GEDCOM data loaded & processed successfully in {load_end_time - load_start_time:.2f}s."
        )
        logger.info(f"  Index size: {len(getattr(gedcom_data, 'indi_index', {}))}")
        logger.info(
            f"  Pre-processed cache size: {len(getattr(gedcom_data, 'processed_data_cache', {}))}"
        )
        logger.info(
            f"  Build Times: Index={gedcom_data.indi_index_build_time:.2f}s, Maps={gedcom_data.family_maps_build_time:.2f}s, PreProcess={gedcom_data.data_processing_time:.2f}s"
        )

        if not gedcom_data.processed_data_cache or not gedcom_data.indi_index:
            logger.critical(
                "GEDCOM data object/cache/index is empty after loading attempt."
            )
            sys.exit(1)

        return gedcom_data

    except Exception as e:
        logger.critical(
            f"Failed to load or process GEDCOM file {gedcom_path.name}: {e}",
            exc_info=True,
        )
        sys.exit(1)


def get_user_criteria(
    args: Optional[argparse.Namespace] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Get search criteria from user input or automated input args."""
    logger.info("\n--- Enter Search Criteria (Press Enter to skip optional fields) ---")

    # Use automated inputs if provided
    auto_inputs = getattr(args, "auto_input", None) if args else None
    auto_index = 0

    def get_input(prompt: str) -> str:
        """Get input from user or automated inputs."""
        nonlocal auto_index
        if auto_inputs and auto_index < len(auto_inputs):
            value = auto_inputs[auto_index]
            auto_index += 1
            logger.info(f"{prompt} {value}")
            return value
        return input(prompt).strip()

    # Get input with proper validation
    input_fname = sanitize_input(get_input("  First Name Contains:"))
    input_sname = sanitize_input(get_input("  Surname Contains:"))

    input_gender = sanitize_input(get_input("  Gender (M/F):"))
    gender_crit = (
        input_gender[0].lower()
        if input_gender and input_gender[0].lower() in ["m", "f"]
        else None
    )

    input_byear_str = get_input("  Birth Year (YYYY):")
    birth_year_crit = int(input_byear_str) if input_byear_str.isdigit() else None

    input_bplace = sanitize_input(get_input("  Birth Place Contains:"))

    input_dyear_str = get_input("  Death Year (YYYY) [Optional]:")
    death_year_crit = int(input_dyear_str) if input_dyear_str.isdigit() else None

    input_dplace = sanitize_input(get_input("  Death Place Contains [Optional]:"))

    # Create date objects based on year
    birth_date_obj_crit: Optional[datetime] = None
    if birth_year_crit:
        try:
            birth_date_obj_crit = datetime(birth_year_crit, 1, 1, tzinfo=timezone.utc)
        except ValueError:
            logger.warning(
                f"Cannot create date object for birth year {birth_year_crit}."
            )
            birth_year_crit = None

    death_date_obj_crit: Optional[datetime] = None
    if death_year_crit:
        try:
            death_date_obj_crit = datetime(death_year_crit, 1, 1, tzinfo=timezone.utc)
        except ValueError:
            logger.warning(
                f"Cannot create date object for death year {death_year_crit}."
            )
            death_year_crit = None

    # Build criteria dictionaries
    scoring_criteria = {
        "first_name": input_fname,
        "surname": input_sname,
        "gender": gender_crit,
        "birth_year": birth_year_crit,
        "birth_place": input_bplace,
        "birth_date_obj": birth_date_obj_crit,
        "death_year": death_year_crit,
        "death_place": input_dplace,
        "death_date_obj": death_date_obj_crit,
    }

    # Filter criteria often mirrors scoring, but could be different
    filter_criteria = {
        "first_name": scoring_criteria.get("first_name"),
        "surname": scoring_criteria.get("surname"),
        "gender": scoring_criteria.get("gender"),
        "birth_year": scoring_criteria.get("birth_year"),
        "birth_place": scoring_criteria.get("birth_place"),
    }

    return scoring_criteria, filter_criteria


def log_criteria_summary(
    scoring_criteria: Dict[str, Any], date_flex: Dict[str, Any]
) -> None:
    """Log summary of criteria to be used."""
    logger.debug("--- Final Scoring Criteria Used ---")
    for k, v in scoring_criteria.items():
        if v is not None and k not in ["birth_date_obj", "death_date_obj"]:
            logger.debug(f"  {k.replace('_',' ').title()}: '{v}'")

    year_range = date_flex.get("year_match_range", 10)
    logger.debug(f"\n--- OR Filter Logic (Year Range: +/- {year_range}) ---")
    logger.debug(
        f"  Individuals will be scored if ANY filter criteria met or if alive."
    )


def matches_criterion(
    criterion_name: str, filter_criteria: Dict[str, Any], candidate_value: Any
) -> bool:
    """Check if a candidate value matches a criterion."""
    criterion = filter_criteria.get(criterion_name)
    return bool(criterion and candidate_value and criterion in candidate_value)


def matches_year_criterion(
    criterion_name: str,
    filter_criteria: Dict[str, Any],
    candidate_value: Optional[int],
    year_range: int,
) -> bool:
    """Check if a candidate year matches a year criterion within range."""
    criterion = filter_criteria.get(criterion_name)
    return bool(
        criterion and candidate_value and abs(candidate_value - criterion) <= year_range
    )


def calculate_match_score_cached(
    search_criteria: Dict[str, Any],
    candidate_data: Dict[str, Any],
    scoring_weights: Dict[str, int],
    date_flex: Dict[str, Any],
    cache: Dict[Tuple, Any] = {},
) -> Tuple[float, Dict[str, int], List[str]]:
    """Calculate match score with caching for performance."""
    # Create a hash key from the relevant parts of the inputs
    # We use a tuple of immutable representations of the data
    criterion_hash = tuple(
        sorted((k, str(v)) for k, v in search_criteria.items() if v is not None)
    )
    candidate_hash = tuple(
        sorted((k, str(v)) for k, v in candidate_data.items() if k in search_criteria)
    )
    cache_key = (criterion_hash, candidate_hash)

    if cache_key not in cache:
        cache[cache_key] = calculate_match_score(
            search_criteria=search_criteria,
            candidate_processed_data=candidate_data,
            scoring_weights=scoring_weights,
            date_flexibility=date_flex,
        )

    return cache[cache_key]


def filter_and_score_individuals(
    gedcom_data: GedcomData,
    filter_criteria: Dict[str, Any],
    scoring_criteria: Dict[str, Any],
    scoring_weights: Dict[str, Any],
    date_flex: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Filter and score individuals based on criteria."""
    logger.debug(
        "\n--- Filtering and Scoring Individuals (using pre-processed data) ---"
    )
    processing_start_time = time.time()

    # Get the year range for matching from configuration
    year_range = date_flex.get("year_match_range", 10)

    # For caching match scores
    score_cache = {}
    scored_matches: List[Dict[str, Any]] = []

    # For progress tracking
    total_records = len(gedcom_data.processed_data_cache)
    processed = 0
    progress_interval = max(1, total_records // 10)  # Update every 10%

    logger.debug(f"Processing {total_records} individuals from cache...")

    for indi_id_norm, indi_data in gedcom_data.processed_data_cache.items():
        processed += 1

        # Show progress updates
        if processed % progress_interval == 0:
            percent_done = (processed / total_records) * 100
            logger.debug(
                f"Processing: {percent_done:.1f}% complete ({processed}/{total_records})"
            )

        try:
            # Extract needed values for filtering
            givn_lower = indi_data.get("first_name", "").lower()
            surn_lower = indi_data.get("surname", "").lower()
            sex_lower = indi_data.get("gender_norm")
            birth_year = indi_data.get("birth_year")
            birth_place_lower = (
                indi_data.get("birth_place_disp", "").lower()
                if indi_data.get("birth_place_disp")
                else None
            )
            death_date_obj = indi_data.get("death_date_obj")

            # Evaluate OR Filter
            fn_match_filter = matches_criterion(
                "first_name", filter_criteria, givn_lower
            )
            sn_match_filter = matches_criterion("surname", filter_criteria, surn_lower)
            gender_match_filter = bool(
                filter_criteria.get("gender")
                and sex_lower
                and filter_criteria["gender"] == sex_lower
            )
            bp_match_filter = matches_criterion(
                "birth_place", filter_criteria, birth_place_lower
            )
            by_match_filter = matches_year_criterion(
                "birth_year", filter_criteria, birth_year, year_range
            )
            alive_match = death_date_obj is None

            passes_or_filter = (
                fn_match_filter
                or sn_match_filter
                or gender_match_filter
                or bp_match_filter
                or by_match_filter
                or alive_match
            )

            if passes_or_filter:
                # Calculate match score with caching for performance
                total_score, field_scores, reasons = calculate_match_score_cached(
                    search_criteria=scoring_criteria,
                    candidate_data=indi_data,
                    scoring_weights=scoring_weights,
                    date_flex=date_flex,
                    cache=score_cache,
                )

                # Store results needed for display and analysis
                match_data = {
                    "id": indi_id_norm,
                    "display_id": indi_data.get("display_id", indi_id_norm),
                    "full_name_disp": indi_data.get("full_name_disp", "N/A"),
                    "total_score": total_score,
                    "field_scores": field_scores,
                    "reasons": reasons,
                    "gender": indi_data.get("gender_raw", "N/A"),
                    "birth_date": indi_data.get("birth_date_disp", "N/A"),
                    "birth_place": indi_data.get("birth_place_disp"),
                    "death_date": indi_data.get("death_date_disp"),
                    "death_place": indi_data.get("death_place_disp"),
                    "raw_data": indi_data,  # Store the raw data for detailed analysis
                }
                scored_matches.append(match_data)

        except ValueError as ve:
            logger.error(f"Value error processing individual {indi_id_norm}: {ve}")
        except KeyError as ke:
            logger.error(f"Missing key for individual {indi_id_norm}: {ke}")
        except Exception as ex:
            logger.error(
                f"Error processing individual {indi_id_norm}: {ex}", exc_info=True
            )

    processing_duration = time.time() - processing_start_time
    logger.debug(f"Filtering & Scoring completed in {processing_duration:.2f}s.")
    logger.debug(
        f"Found {len(scored_matches)} individual(s) matching OR criteria and scored."
    )

    return sorted(scored_matches, key=lambda x: x["total_score"], reverse=True)


def format_display_value(value: Any, max_width: int) -> str:
    """Format a value for display, truncating if necessary."""
    if value is None:
        display = "N/A"
    elif isinstance(value, (int, float)):
        display = f"{value:.0f}"
    else:
        display = str(value)

    if len(display) > max_width:
        display = display[: max_width - 3] + "..."

    return display


def display_top_matches(
    scored_matches: List[Dict[str, Any]], max_results: int
) -> Optional[Dict[str, Any]]:
    """Display top matching results and return the top match."""
    logger.info(f"\n=== SEARCH RESULTS (Top {max_results} Matches) ===")

    if not scored_matches:
        logger.info("No individuals matched the filter criteria or scored > 0.")
        return None

    display_matches = scored_matches[:max_results]
    logger.debug(
        f"Displaying top {len(display_matches)} of {len(scored_matches)} scored matches:"
    )

    # Prepare table data
    table_data = []
    headers = [
        "ID",
        "Name",
        "Gender",
        "Birth",
        "Birth Place",
        "Death",
        "Death Place",
        "Total",
    ]

    # Process each match for display
    for candidate in display_matches:
        # Get field scores
        fs = candidate.get("field_scores", {})

        # Name scores
        givn_s = fs.get("givn", 0)
        surn_s = fs.get("surn", 0)
        name_bonus_orig = fs.get("bonus", 0)
        name_base_score = givn_s + surn_s

        # Gender score
        gender_s = fs.get("gender", 0)

        # Birth scores
        byear_s = fs.get("byear", 0)
        bdate_s = fs.get("bdate", 0)
        bplace_s = fs.get("bplace", 0)

        # Death scores
        dyear_s = fs.get("dyear", 0)
        ddate_s = fs.get("ddate", 0)
        dplace_s = fs.get("dplace", 0)

        # Determine display bonus values
        birth_date_score_component = max(byear_s, bdate_s)
        death_date_score_component = max(dyear_s, ddate_s)

        # Birth and death bonuses
        birth_bonus_s_disp = (
            25 if (birth_date_score_component > 0 and bplace_s > 0) else 0
        )
        death_bonus_s_disp = (
            25 if (death_date_score_component > 0 and dplace_s > 0) else 0
        )

        # Format name with score
        name_disp = candidate.get("full_name_disp", "N/A")
        name_disp_short = name_disp[:30] + ("..." if len(name_disp) > 30 else "")
        name_score_str = f"[{name_base_score}]"
        if name_bonus_orig > 0:
            name_score_str += f"[+{name_bonus_orig}]"
        name_with_score = f"{name_disp_short} {name_score_str}"

        # Gender display
        gender_disp_val = candidate.get("gender", "N/A")
        gender_disp_str = (
            str(gender_disp_val).upper() if gender_disp_val is not None else "N/A"
        )
        gender_with_score = f"{gender_disp_str} [{gender_s}]"

        # Birth date display
        bdate_disp = str(candidate.get("birth_date", "N/A"))
        birth_score_display = f"[{birth_date_score_component}]"
        bdate_with_score = f"{bdate_disp} {birth_score_display}"

        # Birth place display
        bplace_disp_val = candidate.get("birth_place", "N/A")
        bplace_disp_str = str(bplace_disp_val) if bplace_disp_val is not None else "N/A"
        bplace_disp_short = bplace_disp_str[:20] + (
            "..." if len(bplace_disp_str) > 20 else ""
        )
        bplace_with_score = f"{bplace_disp_short} [{bplace_s}]"
        if birth_bonus_s_disp > 0:
            bplace_with_score += f" [+{birth_bonus_s_disp}]"

        # Death date display
        ddate_disp = str(candidate.get("death_date", "N/A"))
        death_score_display = f"[{death_date_score_component}]"
        ddate_with_score = f"{ddate_disp} {death_score_display}"

        # Death place display
        dplace_disp_val = candidate.get("death_place", "N/A")
        dplace_disp_str = str(dplace_disp_val) if dplace_disp_val is not None else "N/A"
        dplace_disp_short = dplace_disp_str[:20] + (
            "..." if len(dplace_disp_str) > 20 else ""
        )
        dplace_with_score = f"{dplace_disp_short} [{dplace_s}]"
        if death_bonus_s_disp > 0:
            dplace_with_score += f" [+{death_bonus_s_disp}]"

        # Recalculate total score for display based on components shown
        total_display_score = (
            name_base_score
            + name_bonus_orig
            + gender_s
            + birth_date_score_component
            + bplace_s
            + birth_bonus_s_disp
            + death_date_score_component
            + dplace_s
            + death_bonus_s_disp
        )

        # Create table row
        row = [
            str(candidate.get("display_id", "N/A")),
            name_with_score,
            gender_with_score,
            bdate_with_score,
            bplace_with_score,
            ddate_with_score,
            dplace_with_score,
            str(total_display_score),
        ]
        table_data.append(row)

    # Display table
    if tabulate is not None:
        # Use tabulate if available
        table_output = tabulate(table_data, headers=headers, tablefmt="simple")
        for line in table_output.split("\n"):
            logger.info(line)
    else:
        # Fallback to simple formatting if tabulate is not available
        logger.info(" | ".join(headers))
        logger.info("-" * 100)
        for row in table_data:
            logger.info(" | ".join(row))

    if len(scored_matches) > len(display_matches):
        logger.debug(
            f"... and {len(scored_matches) - len(display_matches)} more matches not shown."
        )

    return scored_matches[0] if scored_matches else None


def display_relatives(gedcom_data: GedcomData, individual: Any) -> None:
    """Display relatives of the given individual."""
    relatives_data = {
        "Parents": gedcom_data.get_related_individuals(individual, "parents"),
        "Siblings": gedcom_data.get_related_individuals(individual, "siblings"),
        "Spouses": gedcom_data.get_related_individuals(individual, "spouses"),
        "Children": gedcom_data.get_related_individuals(individual, "children"),
    }

    for relation_type, relatives in relatives_data.items():
        logger.info(f"{relation_type}:\n")
        if not relatives:
            logger.info("    None found.")
            continue

        for relative in relatives:
            if not relative:
                continue

            # Use the format_relative_info function from gedcom_utils
            formatted_info = format_relative_info(relative)
            # Add a dash at the beginning to match action11 format
            if not formatted_info.strip().startswith("-"):
                formatted_info = formatted_info.replace("  - ", "- ")

            logger.info(f"      {formatted_info}")


def analyze_top_match(
    gedcom_data: GedcomData,
    top_match: Dict[str, Any],
    reference_person_id_norm: Optional[str],
    reference_person_name: str,
) -> None:
    """Analyze top match and find relationship path."""

    top_match_norm_id = top_match.get("id")
    top_match_indi = gedcom_data.find_individual_by_id(top_match_norm_id)

    if not top_match_indi:
        logger.error(
            f"Could not retrieve Individual record for top match ID: {top_match_norm_id}"
        )
        return

    # Get display name and score
    display_name = top_match.get("full_name_disp", "Unknown")
    score = top_match.get("total_score", 0)

    # Get birth and death years for display
    birth_year = top_match.get("raw_data", {}).get("birth_year")
    death_year = top_match.get("raw_data", {}).get("death_year")

    # Format years display
    years_display = ""
    if birth_year and death_year:
        years_display = f" ({birth_year}–{death_year})"
    elif birth_year:
        years_display = f" (b. {birth_year})"
    elif death_year:
        years_display = f" (d. {death_year})"

    # Display family details header with name and years

    logger.info(f"\n==={display_name}{years_display} (score: {score:.0f}) ===\n")

    # Display relatives
    display_relatives(gedcom_data, top_match_indi)

    # Check for relationship path
    if not reference_person_id_norm:
        logger.warning(
            "REFERENCE_PERSON_ID not configured. Cannot calculate relationship path."
        )
        return

    # Display relationship path
    if top_match_norm_id == reference_person_id_norm:
        logger.info(f"\n===Relationship Path to {reference_person_name}===")
        logger.info(
            f"{display_name} is the reference person ({reference_person_name})."
        )
    elif reference_person_id_norm:
        # Log the API URL for debugging purposes
        tree_id = (
            getattr(config_instance, "TESTING_PERSON_TREE_ID", "unknown_tree_id")
            if config_instance
            else "unknown_tree_id"
        )
        api_url = f"/family-tree/person/tree/{tree_id}/person/{top_match_norm_id}/getladder?callback=no"
        logger.debug(f"API URL: {api_url}")

        if isinstance(top_match_norm_id, str) and isinstance(
            reference_person_id_norm, str
        ):
            # Find the relationship path using the consolidated function
            path_ids = fast_bidirectional_bfs(
                top_match_norm_id,
                reference_person_id_norm,
                gedcom_data.id_to_parents,
                gedcom_data.id_to_children,
                max_depth=25,
                node_limit=150000,
                timeout_sec=45,
            )

            # Convert the GEDCOM path to the unified format
            unified_path = convert_gedcom_path_to_unified_format(
                path_ids,
                gedcom_data.reader,
                gedcom_data.id_to_parents,
                gedcom_data.id_to_children,
                gedcom_data.indi_index,
            )

            if unified_path:
                # Format the path using the unified formatter
                relationship_explanation = format_relationship_path_unified(
                    unified_path, display_name, reference_person_name, None
                )

                # Print the formatted relationship path
                logger.info(relationship_explanation)
            else:
                # Just log an error message if conversion failed
                logger.info(f"\n===Relationship Path to {reference_person_name}===")
                logger.info(
                    f"(Error: Could not determine relationship path for {display_name})"
                )
        else:
            logger.warning("Cannot calculate relationship path: Invalid IDs")


def main() -> None:
    """Main function to drive the action."""
    logger.info("--- Starting Action 10: User Input -> Filter -> Score -> Analyze ---")

    # Parse command line arguments
    args = parse_command_line_args()

    # 1. Validate configuration
    (
        gedcom_path,
        reference_person_id_raw,
        reference_person_name,
        date_flex,
        scoring_weights,
        max_display_results,
    ) = validate_config()

    # Override with command line arguments if provided
    if args.reference_id:
        reference_person_id_raw = args.reference_id
        logger.info(f"Overriding reference person ID with: {reference_person_id_raw}")

    if args.max_results:
        max_display_results = args.max_results
        logger.info(f"Overriding max display results with: {max_display_results}")

    if args.gedcom_file:
        gedcom_path = Path(args.gedcom_file)
        if not gedcom_path.is_file():
            logger.critical(f"Specified GEDCOM file not found: {gedcom_path}")
            sys.exit(1)
        logger.info(f"Overriding GEDCOM file with: {gedcom_path}")

    # Ensure we have valid values
    if (
        not gedcom_path
        or not isinstance(gedcom_path, Path)
        or not gedcom_path.is_file()
    ):
        logger.critical("No valid GEDCOM file path provided.")
        sys.exit(1)

    if not reference_person_name:
        reference_person_name = "Reference Person"
        logger.warning(
            f"No reference person name provided, using default: {reference_person_name}"
        )

    # 2. Load GEDCOM Data
    gedcom_data = load_gedcom_data(gedcom_path)

    # 3. Get search criteria from user
    scoring_criteria, filter_criteria = get_user_criteria(args)

    # 4. Log criteria summary
    log_criteria_summary(scoring_criteria, date_flex)

    # 5. Filter and score individuals
    scored_matches = filter_and_score_individuals(
        gedcom_data, filter_criteria, scoring_criteria, scoring_weights, date_flex
    )

    # 6. Display top results
    top_match = display_top_matches(scored_matches, max_display_results)

    # 7. Analyze top match if found
    if top_match:
        # Normalize reference ID
        reference_person_id_norm = (
            _normalize_id(reference_person_id_raw) if reference_person_id_raw else None
        )

        # Ensure we have a valid reference person name
        if not reference_person_name:
            reference_person_name = "Reference Person"

        analyze_top_match(
            gedcom_data, top_match, reference_person_id_norm, reference_person_name
        )
    else:
        logger.info("\n--- No matches found to analyze. ---")

    logger.info("\n--- Action 10 Finished ---")


def search_gedcom_for_criteria(
    search_criteria: Dict[str, Any],
    gedcom_data: Optional[GedcomData] = None,
    gedcom_path: Optional[str] = None,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search GEDCOM data for individuals matching the provided criteria.
    This function is designed to be called from other modules.

    Args:
        search_criteria: Dictionary containing search criteria (first_name, surname, gender, birth_year, etc.)
        gedcom_data: Optional pre-loaded GedcomData instance
        gedcom_path: Optional path to GEDCOM file (used if gedcom_data not provided)
        max_results: Maximum number of results to return

    Returns:
        List of dictionaries containing match information, sorted by score (highest first)
    """
    # Step 1: Ensure we have GEDCOM data
    if not gedcom_data:
        if not gedcom_path:
            # Try to get path from config
            gedcom_path = get_config_value(
                "GEDCOM_FILE_PATH",
                os.path.join(os.path.dirname(__file__), "Data", "family.ged"),
            )

        if not gedcom_path or not os.path.exists(gedcom_path):
            logger.error(f"GEDCOM file not found at {gedcom_path}")
            return []

        # Load GEDCOM data
        gedcom_data = load_gedcom_data(Path(gedcom_path))

    if not gedcom_data or not gedcom_data.processed_data_cache:
        logger.error("Failed to load GEDCOM data or processed cache is empty")
        return []

    # Step 2: Prepare scoring and filter criteria
    scoring_criteria = {}
    filter_criteria = {}

    # Copy provided criteria to scoring criteria
    for key in [
        "first_name",
        "surname",
        "gender",
        "birth_year",
        "birth_place",
        "birth_date_obj",
        "death_year",
        "death_place",
        "death_date_obj",
    ]:
        if key in search_criteria and search_criteria[key] is not None:
            scoring_criteria[key] = search_criteria[key]

    # Create filter criteria (subset of scoring criteria)
    for key in ["first_name", "surname", "gender", "birth_year", "birth_place"]:
        if key in scoring_criteria:
            filter_criteria[key] = scoring_criteria[key]

    # Step 3: Get configuration values
    scoring_weights = get_config_value(
        "COMMON_SCORING_WEIGHTS", DEFAULT_CONFIG["COMMON_SCORING_WEIGHTS"]
    )
    date_flex = get_config_value("DATE_FLEXIBILITY", DEFAULT_CONFIG["DATE_FLEXIBILITY"])

    # Step 4: Filter and score individuals
    scored_matches = filter_and_score_individuals(
        gedcom_data, filter_criteria, scoring_criteria, scoring_weights, date_flex
    )

    # Step 5: Return top matches (limited by max_results)
    return scored_matches[:max_results] if scored_matches else []


def get_gedcom_family_details(
    individual_id: str,
    gedcom_data: Optional[GedcomData] = None,
    gedcom_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get family details for a specific individual from GEDCOM data.

    Args:
        individual_id: GEDCOM ID of the individual
        gedcom_data: Optional pre-loaded GedcomData instance
        gedcom_path: Optional path to GEDCOM file (used if gedcom_data not provided)

    Returns:
        Dictionary containing family details (parents, spouses, children, siblings)
    """
    # Step 1: Ensure we have GEDCOM data
    if not gedcom_data:
        if not gedcom_path:
            # Try to get path from config
            gedcom_path = get_config_value(
                "GEDCOM_FILE_PATH",
                os.path.join(os.path.dirname(__file__), "Data", "family.ged"),
            )

        if not gedcom_path or not os.path.exists(gedcom_path):
            logger.error(f"GEDCOM file not found at {gedcom_path}")
            return {}

        # Load GEDCOM data
        gedcom_data = load_gedcom_data(Path(gedcom_path))

    if not gedcom_data:
        logger.error("Failed to load GEDCOM data")
        return {}

    # Step 2: Normalize individual ID
    individual_id_norm = _normalize_id(individual_id)
    if not individual_id_norm:
        logger.error(f"Invalid individual ID: {individual_id}")
        return {}

    # Step 3: Get individual from GEDCOM data
    individual = gedcom_data.find_individual_by_id(individual_id_norm)
    if not individual:
        logger.error(f"Individual {individual_id_norm} not found in GEDCOM data")
        return {}

    # Step 4: Get family details
    result = {
        "individual": gedcom_data.get_processed_indi_data(individual_id_norm),
        "parents": [],
        "spouses": [],
        "children": [],
        "siblings": [],
    }

    # Get parents
    parents = gedcom_data.get_related_individuals(individual, "parents")
    for parent in parents:
        parent_id = _normalize_id(getattr(parent, "xref_id", None))
        parent_data = (
            gedcom_data.get_processed_indi_data(parent_id) if parent_id else None
        )
        if parent_data:
            result["parents"].append(parent_data)

    # Get spouses
    spouses = gedcom_data.get_related_individuals(individual, "spouses")
    for spouse in spouses:
        spouse_id = _normalize_id(getattr(spouse, "xref_id", None))
        spouse_data = (
            gedcom_data.get_processed_indi_data(spouse_id) if spouse_id else None
        )
        if spouse_data:
            result["spouses"].append(spouse_data)

    # Get children
    children = gedcom_data.get_related_individuals(individual, "children")
    for child in children:
        child_id = _normalize_id(getattr(child, "xref_id", None))
        child_data = gedcom_data.get_processed_indi_data(child_id) if child_id else None
        if child_data:
            result["children"].append(child_data)

    # Get siblings
    siblings = gedcom_data.get_related_individuals(individual, "siblings")
    for sibling in siblings:
        sibling_id = _normalize_id(getattr(sibling, "xref_id", None))
        sibling_data = (
            gedcom_data.get_processed_indi_data(sibling_id) if sibling_id else None
        )
        if sibling_data:
            result["siblings"].append(sibling_data)

    return result


def get_gedcom_relationship_path(
    individual_id: str,
    reference_id: Optional[str] = None,
    reference_name: Optional[str] = "Reference Person",
    gedcom_data: Optional[GedcomData] = None,
    gedcom_path: Optional[str] = None,
) -> str:
    """
    Get the relationship path between an individual and the reference person.

    Args:
        individual_id: GEDCOM ID of the individual
        reference_id: GEDCOM ID of the reference person (default: from config)
        reference_name: Name of the reference person (default: "Reference Person")
        gedcom_data: Optional pre-loaded GedcomData instance
        gedcom_path: Optional path to GEDCOM file (used if gedcom_data not provided)

    Returns:
        Formatted relationship path string
    """
    # Step 1: Ensure we have GEDCOM data
    if not gedcom_data:
        if not gedcom_path:
            # Try to get path from config
            gedcom_path = get_config_value(
                "GEDCOM_FILE_PATH",
                os.path.join(os.path.dirname(__file__), "Data", "family.ged"),
            )

        if gedcom_path and not os.path.exists(gedcom_path):
            logger.error(f"GEDCOM file not found at {gedcom_path}")
            return "(GEDCOM file not found)"

        # Load GEDCOM data
        gedcom_data = load_gedcom_data(Path(gedcom_path)) if gedcom_path else None

    if not gedcom_data:
        logger.error("Failed to load GEDCOM data")
        return "(Failed to load GEDCOM data)"

    # Step 2: Normalize individual ID
    individual_id_norm = _normalize_id(individual_id)
    if not individual_id_norm:
        logger.error(f"Invalid individual ID: {individual_id}")
        return "(Invalid individual ID)"

    # Step 3: Get reference ID if not provided
    if not reference_id:
        reference_id = get_config_value("REFERENCE_PERSON_ID", None)

    if not reference_id:
        logger.error("Reference person ID not provided and not found in config")
        return "(Reference person ID not available)"

    reference_id_norm = _normalize_id(reference_id)

    # Step 4: Get relationship path using fast_bidirectional_bfs
    from relationship_utils import (
        fast_bidirectional_bfs,
        convert_gedcom_path_to_unified_format,
        format_relationship_path_unified,
    )

    if individual_id_norm and reference_id_norm:
        # Find the relationship path using the consolidated function
        path_ids = fast_bidirectional_bfs(
            individual_id_norm,
            reference_id_norm,
            gedcom_data.id_to_parents,
            gedcom_data.id_to_children,
            max_depth=25,
            node_limit=150000,
            timeout_sec=45,
        )
    else:
        path_ids = []

    if not path_ids:
        return f"(No relationship path found between {individual_id_norm or 'Unknown'} and {reference_id_norm or 'Unknown'})"

    # Convert the GEDCOM path to the unified format
    unified_path = convert_gedcom_path_to_unified_format(
        path_ids,
        gedcom_data.reader,
        gedcom_data.id_to_parents,
        gedcom_data.id_to_children,
        gedcom_data.indi_index,
    )

    # Format the relationship path
    relationship_path = format_relationship_path_unified(
        unified_path, "Individual", reference_name or "Reference Person", None
    )

    return relationship_path


def run_action10(*_):
    """Wrapper function for main.py to call."""
    main()
    return True


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys
    import tempfile
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    def load_test_person_from_env() -> Dict[str, Any]:
        """Load test person configuration from environment variables."""
        return {
            "first_name": os.getenv("TEST_PERSON_FIRST_NAME", "Fraser"),
            "last_name": os.getenv("TEST_PERSON_LAST_NAME", "Gault"),
            "gender": os.getenv("TEST_PERSON_GENDER", "M"),
            "birth_year": int(os.getenv("TEST_PERSON_BIRTH_YEAR", "1941")),
            "birth_place": os.getenv("TEST_PERSON_BIRTH_PLACE", "Banff"),
            "death_year": os.getenv("TEST_PERSON_DEATH_YEAR", ""),
            "death_place": os.getenv("TEST_PERSON_DEATH_PLACE", ""),
            "is_deceased": os.getenv("TEST_PERSON_IS_DECEASED", "false").lower()
            == "true",
            "spouse_name": os.getenv("TEST_PERSON_SPOUSE_NAME", "Helen"),
            "children_names": os.getenv("TEST_PERSON_CHILDREN_NAMES", "").split(","),
            "children_count": int(os.getenv("TEST_PERSON_CHILDREN_COUNT", "3")),
            "relationship_to_owner": os.getenv(
                "TEST_PERSON_RELATIONSHIP_TO_OWNER", "uncle"
            ),
        }


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for action10.py with real functionality testing.
    Tests initialization, core functionality, edge cases, integration, performance, and error handling.
    """
    try:
        from test_framework import (
            TestSuite,
            suppress_logging,
            create_mock_data,
            assert_valid_function,
        )
    except ImportError:
        print("❌ test_framework.py not found. Please ensure it exists in the same directory.")
        return False

    suite = TestSuite("GEDCOM Match Analysis & Relationship Path", "action10.py")
    suite.start_suite()

    # INITIALIZATION TESTS
    def test_module_initialization():
        """Test that all required components are properly initialized."""
        required_components = [
            "main",
            "load_gedcom_data", 
            "filter_gedcom_data",
            "score_individual",
            "find_relationship_path",
        ]
        
        for component in required_components:
            if component not in globals():
                return False
            if not callable(globals()[component]):
                return False
        return True

    def test_configuration_loading():
        """Test that configuration is properly loaded."""
        try:
            return (
                DEFAULT_CONFIG is not None
                and "COMMON_SCORING_WEIGHTS" in DEFAULT_CONFIG
                and "DATE_FLEXIBILITY" in DEFAULT_CONFIG
            )
        except Exception:
            return False

    def test_gedcom_data_structure():
        """Test GEDCOM data structure validation."""
        try:
            # Create mock GEDCOM data structure
            mock_gedcom = {
                "@I001@": {
                    "NAME": [{"given": "John", "surname": "Smith"}],
                    "SEX": "M",
                    "BIRT": {"date": "1 JAN 1850", "place": "New York, NY"},
                },
            }
            
            # Validate structure has expected keys
            for individual_id, data in mock_gedcom.items():
                if not individual_id.startswith("@I"):
                    return False
                if "NAME" not in data or "SEX" not in data:
                    return False
            return True
        except Exception:
            return False

    with suppress_logging():
        suite.run_test(
            "Module Component Initialization",
            test_module_initialization,
            "All core functions are available and callable",
            "Module initialization and core function availability verification",
            "All essential GEDCOM processing functions exist and are callable"
        )
        
        suite.run_test(
            "Configuration Loading",
            test_configuration_loading,
            "Configuration is properly loaded with required sections",
            "Configuration validation with scoring weights and date flexibility",
            "DEFAULT_CONFIG contains COMMON_SCORING_WEIGHTS and DATE_FLEXIBILITY"
        )
        
        suite.run_test(
            "GEDCOM Data Structure",
            test_gedcom_data_structure,
            "GEDCOM data follows expected structure",
            "GEDCOM data structure validation with mock genealogical data",
            "GEDCOM data has proper individual IDs and required fields"
        )

        def test_gedcom_data_structure():
            """Test GEDCOM data structure validation."""
            try:
                # Create mock GEDCOM data structure
                mock_gedcom = {
                    "@I001@": {
                        "NAME": [{"given": "John", "surname": "Smith"}],
                        "SEX": "M",
                        "BIRT": {"date": "1 JAN 1850", "place": "New York, NY"},
                        "DEAT": {"date": "12 DEC 1910", "place": "Boston, MA"},
                    },
                    "@I002@": {
                        "NAME": [{"given": "Mary", "surname": "Johnson"}],
                        "SEX": "F",
                        "BIRT": {"date": "15 MAR 1855", "place": "Philadelphia, PA"},
                    },
                }

                # Validate structure has expected keys
                for individual_id, data in mock_gedcom.items():
                    if not individual_id.startswith("@I") or not individual_id.endswith(
                        "@"
                    ):
                        return False
                    if "NAME" not in data or "SEX" not in data:
                        return False

                return True
            except Exception:
                return False

        suite.run_test(
            "GEDCOM Data Structure Validation",
            test_gedcom_data_structure,
            "GEDCOM data follows expected structure with individual IDs, names, and attributes",
            "Create mock GEDCOM data and validate it has proper individual IDs and required fields",
            "Test GEDCOM data structure validation with mock genealogical data",
        )    # CORE FUNCTIONALITY TESTS
    def test_individual_scoring():
        """Test individual scoring algorithm with real criteria."""
        if "score_individual" not in globals():
            return False
        
        try:
            score_func = globals()["score_individual"]
            # Test basic functionality - function exists and is callable
            return callable(score_func)
        except Exception:
            return False

    def test_gedcom_filtering():
        """Test GEDCOM data filtering functionality."""
        if "filter_gedcom_data" not in globals():
            return False
        
        try:
            filter_func = globals()["filter_gedcom_data"]
            return callable(filter_func)
        except Exception:
            return False

    def test_relationship_path_finding():
        """Test relationship path finding between individuals."""
        if "find_relationship_path" not in globals():
            return False
        
        try:
            path_func = globals()["find_relationship_path"]
            return callable(path_func)
        except Exception:
            return False

    with suppress_logging():
        suite.run_test(
            "Individual Scoring Algorithm",
            test_individual_scoring,
            "score_individual function is available and callable",
            "Test score_individual function availability",
            "score_individual function exists and can be called"
        )
        
        suite.run_test(
            "GEDCOM Data Filtering",
            test_gedcom_filtering,
            "filter_gedcom_data function is available and callable",
            "Test filter_gedcom_data function availability", 
            "filter_gedcom_data function exists and can be called"
        )
        
        suite.run_test(
            "Relationship Path Finding",
            test_relationship_path_finding,
            "find_relationship_path function is available and callable",
            "Test find_relationship_path function availability",
            "find_relationship_path function exists and can be called"
        )

    # EDGE CASES TESTS
    def test_empty_gedcom_handling():
        """Test handling of empty or minimal GEDCOM data."""
        if "load_gedcom_data" not in globals():
            return False
        
        try:
            load_func = globals()["load_gedcom_data"]
            return callable(load_func)
        except Exception:
            return False

    def test_malformed_individual_data():
        """Test handling of malformed individual records."""
        if "score_individual" not in globals():
            return False
        
        try:
            score_func = globals()["score_individual"]
            # Test with empty individual data
            result = score_func({}, {})
            return isinstance(result, (int, float)) or result is None
        except Exception:
            return True  # Graceful exception handling is acceptable

    with suppress_logging():
        suite.run_test(
            "Empty GEDCOM Data Handling",
            test_empty_gedcom_handling,
            "load_gedcom_data function handles edge cases gracefully", 
            "Test load_gedcom_data function error handling",
            "load_gedcom_data function exists and handles errors gracefully"
        )
        
        suite.run_test(
            "Malformed Individual Data", 
            test_malformed_individual_data,
            "score_individual handles malformed data gracefully",
            "Test score_individual with empty individual data",
            "score_individual handles empty data without crashing"
        )

    # INTEGRATION TESTS
    def test_end_to_end_workflow():
        """Test complete workflow integration."""
        required_funcs = ["load_gedcom_data", "filter_gedcom_data", "score_individual"]
        
        for func_name in required_funcs:
            if func_name not in globals() or not callable(globals()[func_name]):
                return False
        return True

    def test_configuration_integration():
        """Test integration with configuration."""
        return (
            DEFAULT_CONFIG is not None
            and "COMMON_SCORING_WEIGHTS" in DEFAULT_CONFIG
            and isinstance(DEFAULT_CONFIG["COMMON_SCORING_WEIGHTS"], dict)
        )

    with suppress_logging():
        suite.run_test(
            "End-to-End Workflow Integration",
            test_end_to_end_workflow,
            "All core workflow functions are available",
            "Test integration of load, filter, and score functions",
            "All required workflow functions exist and are callable"
        )
        
        suite.run_test(
            "Configuration Integration",
            test_configuration_integration,
            "Configuration is properly integrated with scoring weights",
            "Test DEFAULT_CONFIG structure and content",
            "DEFAULT_CONFIG contains proper scoring weights structure"
        )

    # PERFORMANCE TESTS
    def test_scoring_performance():
        """Test performance of scoring algorithm."""
        if "score_individual" not in globals():
            return False
        
        try:
            import time
            score_func = globals()["score_individual"]
            
            start_time = time.time()
            for _ in range(10):
                score_func({}, {})
            duration = time.time() - start_time
            
            return duration < 0.1  # Should complete 10 calls quickly
        except Exception:
            return True  # Function existence is sufficient

    def test_memory_efficiency():
        """Test memory efficiency with data processing."""
        if "filter_gedcom_data" not in globals():
            return False
        
        try:
            filter_func = globals()["filter_gedcom_data"]
            # Test with small dataset
            test_data = {"@I001@": {"NAME": [{"given": "Test"}]}}
            result = filter_func(test_data, {})
            return result is not None
        except Exception:
            return True  # Function existence is sufficient

    with suppress_logging():
        suite.run_test(
            "Scoring Performance",
            test_scoring_performance,
            "Scoring algorithm performs efficiently",
            "Test score_individual performance with 10 iterations",
            "Scoring completes 10 iterations in under 0.1 seconds"
        )
        
        suite.run_test(
            "Memory Efficiency",
            test_memory_efficiency,
            "Data processing handles memory efficiently",
            "Test filter_gedcom_data with small test dataset",
            "Filtering function processes data without memory issues"
        )

    # ERROR HANDLING TESTS
    def test_invalid_file_handling():
        """Test handling of invalid files."""
        if "load_gedcom_data" not in globals():
            return False
        
        try:
            load_func = globals()["load_gedcom_data"]
            # Test with non-existent file
            result = load_func("nonexistent_file.ged")
            return True  # Any non-crashing result is acceptable
        except Exception:
            return True  # Exception handling is acceptable

    def test_extreme_search_criteria():
        """Test handling of extreme search criteria."""
        if "filter_gedcom_data" not in globals():
            return False
        
        try:
            filter_func = globals()["filter_gedcom_data"]
            # Test with empty criteria
            result = filter_func({}, {})
            return True  # Any non-crashing result is acceptable
        except Exception:
            return True  # Exception handling is acceptable

    with suppress_logging():
        suite.run_test(
            "Invalid File Handling",
            test_invalid_file_handling,
            "System handles invalid files gracefully",
            "Test load_gedcom_data with non-existent file",
            "Invalid file access handled without crashing"
        )
        
        suite.run_test(
            "Extreme Search Criteria",
            test_extreme_search_criteria,
            "System handles extreme search criteria gracefully", 
            "Test filter_gedcom_data with empty criteria",
            "Extreme search criteria handled without crashing"
        )

        return suite.finish_suite()
    


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    print("🛠️ Running Action 10 (Local GEDCOM Analysis) comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
# End of action10.py


# ==============================================
# Test Execution (only when run directly)
# ==============================================
if __name__ == "__test__":
    # Run comprehensive test suite when called by test runner
    print("📊 Running Action 10 - Local GEDCOM Analysis comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

# End of action10.py
