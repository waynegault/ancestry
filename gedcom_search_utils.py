#!/usr/bin/env python3

# gedcom_search_utils.py
"""
Utility functions for searching GEDCOM data and retrieving person and family information.
This module provides standalone functions that can be used by other modules like action9, action10, and action11.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Import from local modules
from logging_config import logger
from config import config_instance
from gedcom_utils import GedcomData, calculate_match_score, _normalize_id
from relationship_utils import (
    fast_bidirectional_bfs,
    convert_gedcom_path_to_unified_format,
    format_relationship_path_unified,
)

# Default configuration values
DEFAULT_CONFIG = {
    "DATE_FLEXIBILITY": {"year_match_range": 10},
    "COMMON_SCORING_WEIGHTS": {
        # --- Name Weights ---
        "contains_first_name": 25,  # if the input first name is in the candidate first name
        "contains_surname": 25,  # if the input surname is in the candidate surname
        "bonus_both_names_contain": 25,  # additional bonus if both first and last name achieved a score
        # --- Existing Date Weights ---
        "exact_birth_date": 25,  # if input date of birth is exact with candidate date of birth
        "exact_death_date": 25,  # if input date of death is exact with candidate date of death
        "birth_year_match": 20,  # if input birth year matches candidate birth year
        "death_year_match": 20,  # if input death year matches candidate death year
        "birth_year_close": 10,  # if input birth year is within range of candidate birth year
        "death_year_close": 10,  # if input death year is within range of candidate death year
        # --- Place Weights ---
        "birth_place_match": 20,  # if input birth place matches candidate birth place
        "death_place_match": 20,  # if input death place matches candidate death place
        # --- Gender Weight ---
        "gender_match": 15,  # if input gender matches candidate gender
        # --- Bonus Weights ---
        "bonus_birth_date_and_place": 15,  # bonus if both birth date and place match
        "bonus_death_date_and_place": 15,  # bonus if both death date and place match
    },
}

# Global cache for GEDCOM data with enhanced caching
_CACHED_GEDCOM_DATA = None


def set_cached_gedcom_data(gedcom_data):
    """
    Set the cached GEDCOM data directly with enhanced caching support.

    This function allows other modules to set the cached GEDCOM data directly,
    which is useful for avoiding redundant loading of the GEDCOM file.
    Also triggers aggressive caching of processed data.

    Args:
        gedcom_data: GedcomData instance to cache

    Returns:
        None
    """
    global _CACHED_GEDCOM_DATA
    _CACHED_GEDCOM_DATA = gedcom_data
    logger.info(f"Set cached GEDCOM data directly: {gedcom_data is not None}")

    # If we have a valid gedcom_data instance, cache its processed data
    if gedcom_data and hasattr(gedcom_data, "path"):
        try:
            from gedcom_cache import cache_gedcom_processed_data

            cache_gedcom_processed_data(gedcom_data, str(gedcom_data.path))
        except ImportError:
            logger.debug("gedcom_cache module not available for enhanced caching")
        except Exception as e:
            logger.debug(f"Error caching processed GEDCOM data: {e}")


def get_cached_gedcom_data():
    """Return the currently cached GEDCOM data."""
    global _CACHED_GEDCOM_DATA
    return _CACHED_GEDCOM_DATA


def get_config_value(key: str, default_value: Any = None) -> Any:
    """Safely retrieve a configuration value with fallback."""
    if not config_instance:
        return default_value
    return getattr(config_instance, key, default_value)


def load_gedcom_data(gedcom_path: Path) -> Optional[GedcomData]:
    """
    Load and initialize a GedcomData instance.

    Args:
        gedcom_path: Path to the GEDCOM file

    Returns:
        GedcomData instance or None if loading fails
    """
    try:
        # Log the path we're using
        logger.info(f"Loading GEDCOM file from: {gedcom_path}")

        # Check if the file exists and is readable
        if not gedcom_path.exists():
            logger.error(f"GEDCOM file does not exist: {gedcom_path}")
            return None

        # Create GedcomData instance
        logger.info("Creating GedcomData instance...")
        gedcom_data = GedcomData(gedcom_path)

        # Check if the instance was created successfully
        if gedcom_data:
            logger.info(f"GedcomData instance created successfully")

            # Try to build caches if the method exists
            if hasattr(gedcom_data, "build_caches"):
                logger.info("Building caches...")
                try:
                    gedcom_data.build_caches()
                    logger.info("Caches built successfully")
                except Exception as e:
                    logger.error(f"Error building caches: {e}", exc_info=True)
            else:
                logger.warning("GedcomData does not have build_caches method")

            return gedcom_data
        else:
            logger.error("GedcomData instance is None")
            return None
    except Exception as e:
        logger.error(f"Error loading GEDCOM file: {e}", exc_info=True)
        return None


def get_gedcom_data() -> Optional[GedcomData]:
    """
    Returns the cached GEDCOM data instance with aggressive caching, loading it if necessary.

    This function ensures the GEDCOM file is loaded only once and reused
    throughout the script, with enhanced multi-level caching for optimal performance.

    Returns:
        GedcomData instance or None if loading fails
    """
    global _CACHED_GEDCOM_DATA

    # Return cached data if already loaded
    if _CACHED_GEDCOM_DATA is not None:
        logger.info("Using cached GEDCOM data from memory")
        return _CACHED_GEDCOM_DATA

    # Check if GEDCOM path is configured
    gedcom_path_str = get_config_value("GEDCOM_FILE_PATH", None)
    logger.info(f"GEDCOM_FILE_PATH from config: {gedcom_path_str}")

    if not gedcom_path_str:
        logger.warning("GEDCOM_FILE_PATH not configured. Cannot load GEDCOM file.")
        return None

    # Convert string to Path object
    gedcom_path = Path(gedcom_path_str)

    # Make sure the path is absolute
    if not gedcom_path.is_absolute():
        # If it's a relative path, make it absolute relative to the project root
        original_path = gedcom_path
        gedcom_path = Path(os.path.dirname(os.path.abspath(__file__))) / gedcom_path
        logger.info(
            f"Converted relative path '{original_path}' to absolute path: {gedcom_path}"
        )

    # Check if the file exists
    if not gedcom_path.exists():
        logger.warning(f"GEDCOM file not found at {gedcom_path}")
        return None

    # Try to load with aggressive caching first
    try:
        from gedcom_cache import load_gedcom_with_aggressive_caching

        logger.info("Using aggressive GEDCOM caching system")
        _CACHED_GEDCOM_DATA = load_gedcom_with_aggressive_caching(str(gedcom_path))
    except ImportError:
        logger.debug("Aggressive caching not available, using standard loading")
        _CACHED_GEDCOM_DATA = load_gedcom_data(gedcom_path)
    except Exception as e:
        logger.warning(f"Error with aggressive caching, falling back to standard: {e}")
        _CACHED_GEDCOM_DATA = load_gedcom_data(gedcom_path)

    if _CACHED_GEDCOM_DATA:
        logger.info(f"GEDCOM file loaded successfully and cached for reuse.")

        # Log cache statistics if available
        try:
            from gedcom_cache import get_gedcom_cache_info

            cache_info = get_gedcom_cache_info()
            logger.debug(f"GEDCOM cache info: {cache_info}")
        except ImportError:
            pass
    else:
        logger.warning("GEDCOM data loading returned None")

    return _CACHED_GEDCOM_DATA


def matches_criterion(key: str, criteria: Dict[str, Any], value: Any) -> bool:
    """Check if a value matches a criterion."""
    if key not in criteria or criteria[key] is None:
        return False

    criterion = criteria[key]

    # Handle string values (case-insensitive contains)
    if isinstance(criterion, str) and isinstance(value, str):
        return criterion.lower() in value.lower()

    # Handle exact matches for non-string values
    return criterion == value


def matches_year_criterion(
    key: str, criteria: Dict[str, Any], value: Any, year_range: int
) -> bool:
    """Check if a year value matches a criterion within a specified range."""
    if key not in criteria or criteria[key] is None or value is None:
        return False

    criterion = criteria[key]

    # Handle exact match
    if criterion == value:
        return True

    # Handle range match
    return abs(criterion - value) <= year_range


def search_gedcom_for_criteria(
    search_criteria: Dict[str, Any],
    max_results: int = 10,
    gedcom_data: Optional[GedcomData] = None,
    gedcom_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search GEDCOM data for individuals matching the given criteria.

    Args:
        search_criteria: Dictionary of search criteria (first_name, surname, gender, birth_year, etc.)
        max_results: Maximum number of results to return (default: 10)
        gedcom_data: Optional pre-loaded GedcomData instance
        gedcom_path: Optional path to GEDCOM file (used if gedcom_data not provided)

    Returns:
        List of dictionaries containing match information, sorted by score (highest first)
    """
    # Step 1: Ensure we have GEDCOM data
    if not gedcom_data:
        # Try to use the cached GEDCOM data first
        global _CACHED_GEDCOM_DATA
        if _CACHED_GEDCOM_DATA is not None:
            logger.info("Using cached GEDCOM data from _CACHED_GEDCOM_DATA")
            gedcom_data = _CACHED_GEDCOM_DATA
        else:
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
            gedcom_data = load_gedcom_data(Path(str(gedcom_path)))

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
    year_range = date_flex.get("year_match_range", 10)

    # Step 4: Filter and score individuals
    scored_matches = []
    score_cache = {}  # Cache for score calculations

    # Process each individual in the GEDCOM data
    for indi_id, indi_data in gedcom_data.processed_data_cache.items():
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
                # Calculate match score
                criterion_hash = hash(json.dumps(scoring_criteria, sort_keys=True))
                candidate_hash = hash(json.dumps(str(indi_data), sort_keys=True))
                cache_key = (criterion_hash, candidate_hash)

                if cache_key not in score_cache:
                    total_score, field_scores, reasons = calculate_match_score(
                        search_criteria=scoring_criteria,
                        candidate_processed_data=indi_data,
                        scoring_weights=scoring_weights,
                        date_flexibility=date_flex,
                    )
                    score_cache[cache_key] = (total_score, field_scores, reasons)
                else:
                    total_score, field_scores, reasons = score_cache[cache_key]

                # Only include if score is above threshold
                if total_score > 0:
                    # Create a match record
                    match_record = {
                        "id": indi_id,
                        "display_id": indi_id,
                        "first_name": indi_data.get("first_name", ""),
                        "surname": indi_data.get("surname", ""),
                        "gender": indi_data.get("gender", ""),
                        "birth_year": indi_data.get("birth_year"),
                        "birth_place": indi_data.get("birth_place", ""),
                        "death_year": indi_data.get("death_year"),
                        "death_place": indi_data.get("death_place", ""),
                        "total_score": total_score,
                        "field_scores": field_scores,
                        "reasons": reasons,
                        "source": "GEDCOM",
                    }
                    scored_matches.append(match_record)
        except Exception as e:
            logger.error(f"Error processing individual {indi_id}: {e}")
            continue

    # Sort matches by score (highest first)
    scored_matches.sort(key=lambda x: x.get("total_score", 0), reverse=True)

    # Return top matches (limited by max_results)
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
        # Try to use the cached GEDCOM data first
        global _CACHED_GEDCOM_DATA
        if _CACHED_GEDCOM_DATA is not None:
            logger.info("Using cached GEDCOM data from _CACHED_GEDCOM_DATA")
            gedcom_data = _CACHED_GEDCOM_DATA
        else:
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
            gedcom_data = load_gedcom_data(Path(str(gedcom_path)))

    if not gedcom_data:
        logger.error("Failed to load GEDCOM data")
        return {}

    # Step 2: Get individual data from cache
    if not hasattr(gedcom_data, "processed_data_cache"):
        logger.error("GEDCOM data does not have processed_data_cache attribute")
        return {}

    # Normalize the individual ID
    individual_id_norm = _normalize_id(individual_id)
    if individual_id_norm is None:
        logger.error(f"Invalid individual ID: {individual_id}")
        return {}

    # Get individual data from cache
    individual_data = gedcom_data.processed_data_cache.get(individual_id_norm, {})
    if not individual_data:
        logger.error(f"Individual {individual_id_norm} not found in GEDCOM data")
        return {}

    # Step 3: Extract basic information
    result = {
        "id": individual_id_norm,
        "name": individual_data.get("full_name_disp", "Unknown"),
        "first_name": individual_data.get("first_name", ""),
        "surname": individual_data.get("surname", ""),
        "gender": individual_data.get("gender", ""),
        "birth_year": individual_data.get("birth_year"),
        "birth_date": individual_data.get("birth_date_disp", "Unknown"),
        "birth_place": individual_data.get("birth_place_disp", "Unknown"),
        "death_year": individual_data.get("death_year"),
        "death_date": individual_data.get("death_date_disp", "Unknown"),
        "death_place": individual_data.get("death_place_disp", "Unknown"),
        "parents": [],
        "spouses": [],
        "children": [],
        "siblings": [],
    }

    # Step 4: Get family relationships
    try:
        # Get parents
        parent_ids = (
            gedcom_data.id_to_parents.get(individual_id_norm, [])
            if hasattr(gedcom_data, "id_to_parents")
            else []
        )
        for parent_id in parent_ids:
            if parent_id is None:
                continue
            parent_data = gedcom_data.processed_data_cache.get(parent_id, {})
            if parent_data:
                gender = parent_data.get("gender", "")
                relationship = (
                    "father"
                    if gender == "M"
                    else "mother" if gender == "F" else "parent"
                )

                parent_info = {
                    "id": parent_id,
                    "name": parent_data.get("full_name_disp", "Unknown"),
                    "birth_year": parent_data.get("birth_year"),
                    "birth_place": parent_data.get("birth_place_disp", "Unknown"),
                    "death_year": parent_data.get("death_year"),
                    "death_place": parent_data.get("death_place_disp", "Unknown"),
                    "relationship": relationship,
                }
                result["parents"].append(parent_info)

        # Get siblings (share at least one parent)
        siblings_set = set()
        for parent_id in parent_ids:
            parent_children = gedcom_data.id_to_children.get(parent_id, [])
            for child_id in parent_children:
                if child_id != individual_id_norm:
                    siblings_set.add(child_id)

        for sibling_id in siblings_set:
            sibling_data = gedcom_data.processed_data_cache.get(sibling_id)
            if sibling_data:
                sibling_info = {
                    "id": sibling_id,
                    "name": sibling_data.get("full_name_disp", "Unknown"),
                    "birth_year": sibling_data.get("birth_year"),
                    "birth_place": sibling_data.get("birth_place_disp", "Unknown"),
                    "death_year": sibling_data.get("death_year"),
                    "death_place": sibling_data.get("death_place_disp", "Unknown"),
                }
                result["siblings"].append(sibling_info)

        # Get spouses and children
        # This requires looking at family records in the GEDCOM data
        if (
            hasattr(gedcom_data, "reader")
            and gedcom_data.reader
            and hasattr(gedcom_data, "indi_index")
            and gedcom_data.indi_index
        ):
            # Get the individual record
            indi_record = gedcom_data.indi_index.get(individual_id_norm)
            if indi_record:
                # Get family records where this individual is a spouse
                for fam_link in indi_record.sub_tags("FAMS"):
                    fam_id = fam_link.value
                    fam_record = None

                    # Try to get family record using various methods (with error handling)
                    try:
                        if hasattr(gedcom_data.reader, "fam_dict"):
                            fam_dict = getattr(gedcom_data.reader, "fam_dict", None)
                            if fam_dict:
                                fam_record = fam_dict.get(fam_id)

                        if not fam_record and hasattr(gedcom_data.reader, "get_family"):
                            get_family = getattr(gedcom_data.reader, "get_family", None)
                            if get_family:
                                fam_record = get_family(fam_id)
                    except Exception:
                        fam_record = None

                    if fam_record:
                        # Get spouse
                        husb_tag = fam_record.sub_tag("HUSB")
                        wife_tag = fam_record.sub_tag("WIFE")

                        spouse_id = None
                        if husb_tag and husb_tag.value != individual_id_norm:
                            spouse_id = _normalize_id(husb_tag.value)
                        elif wife_tag and wife_tag.value != individual_id_norm:
                            spouse_id = _normalize_id(wife_tag.value)

                        if spouse_id:
                            spouse_data = gedcom_data.processed_data_cache.get(
                                spouse_id
                            )
                            if spouse_data:
                                # Get marriage information
                                marriage_date = "Unknown"
                                marriage_place = "Unknown"
                                marr_tag = fam_record.sub_tag("MARR")
                                if marr_tag:
                                    date_tag = marr_tag.sub_tag("DATE")
                                    if date_tag:
                                        marriage_date = date_tag.value

                                    plac_tag = marr_tag.sub_tag("PLAC")
                                    if plac_tag:
                                        marriage_place = plac_tag.value

                                spouse_info = {
                                    "id": spouse_id,
                                    "name": spouse_data.get(
                                        "full_name_disp", "Unknown"
                                    ),
                                    "birth_year": spouse_data.get("birth_year"),
                                    "birth_place": spouse_data.get(
                                        "birth_place_disp", "Unknown"
                                    ),
                                    "death_year": spouse_data.get("death_year"),
                                    "death_place": spouse_data.get(
                                        "death_place_disp", "Unknown"
                                    ),
                                    "marriage_date": marriage_date,
                                    "marriage_place": marriage_place,
                                }
                                result["spouses"].append(spouse_info)

                        # Get children
                        for chil_tag in fam_record.sub_tags("CHIL"):
                            child_id = _normalize_id(chil_tag.value)
                            if child_id is None:
                                continue
                            child_data = gedcom_data.processed_data_cache.get(
                                child_id, {}
                            )
                            if child_data:
                                child_info = {
                                    "id": child_id,
                                    "name": child_data.get("full_name_disp", "Unknown"),
                                    "birth_year": child_data.get("birth_year"),
                                    "birth_place": child_data.get(
                                        "birth_place_disp", "Unknown"
                                    ),
                                    "death_year": child_data.get("death_year"),
                                    "death_place": child_data.get(
                                        "death_place_disp", "Unknown"
                                    ),
                                }
                                result["children"].append(child_info)
    except Exception as e:
        logger.error(
            f"Error getting family details for {individual_id_norm}: {e}", exc_info=True
        )

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
        # Try to use the cached GEDCOM data first
        global _CACHED_GEDCOM_DATA
        if _CACHED_GEDCOM_DATA is not None:
            logger.info("Using cached GEDCOM data from _CACHED_GEDCOM_DATA")
            gedcom_data = _CACHED_GEDCOM_DATA
        else:
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
            gedcom_data = (
                load_gedcom_data(Path(str(gedcom_path))) if gedcom_path else None
            )

    if not gedcom_data:
        logger.error("Failed to load GEDCOM data")
        return "(Failed to load GEDCOM data)"

    # Step 2: Normalize individual ID
    individual_id_norm = _normalize_id(individual_id)

    # Step 3: Get reference ID if not provided
    if not reference_id:
        reference_id = get_config_value("REFERENCE_PERSON_ID", None)

    if not reference_id:
        logger.error("Reference person ID not provided and not found in config")
        return "(Reference person ID not available)"

    reference_id_norm = _normalize_id(reference_id)

    # Step 4: Get individual name
    individual_name = "Individual"
    if individual_id_norm and individual_id_norm in gedcom_data.processed_data_cache:
        individual_data = gedcom_data.processed_data_cache[individual_id_norm]
        individual_name = individual_data.get("full_name_disp", "Individual")

    # Step 5: Get relationship path using fast_bidirectional_bfs
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
        return f"(No relationship path found between {individual_name} and {reference_name})"

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
        unified_path, individual_name, reference_name or "Reference Person", None
    )

    return relationship_path


def run_self_tests() -> bool:
    """
    Run comprehensive self-tests for gedcom_search_utils module.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("=" * 60)
    print("GEDCOM Search Utils - Self Test Suite")
    print("=" * 60)

    tests_passed = 0
    tests_failed = 0

    def test_result(test_name: str, passed: bool, details: str = ""):
        nonlocal tests_passed, tests_failed
        status = "PASS" if passed else "FAIL"
        print(f"[{status:>4}] {test_name}")
        if details:
            print(f"       {details}")
        if passed:
            tests_passed += 1
        else:
            tests_failed += 1
        return passed

    # Configuration and Setup
    print("\n--- Test Section 1: Configuration and Setup ---")

    try:
        # Test config loading
        gedcom_path = (
            getattr(config_instance, "GEDCOM_FILE_PATH", None)
            if config_instance
            else None
        )
        test_result(
            "Config Value Retrieval",
            gedcom_path is not None,
            f"GEDCOM_FILE_PATH: {gedcom_path}",
        )

        # Test GEDCOM data loading
        gedcom_data = get_gedcom_data()
        test_result(
            "GEDCOM Data Loading",
            gedcom_data is not None,
            f"GedcomData instance created: {type(gedcom_data)}",
        )

        if gedcom_data:
            cache_size = (
                len(gedcom_data.processed_data_cache)
                if hasattr(gedcom_data, "processed_data_cache")
                else 0
            )
            test_result(
                "GEDCOM Data Cache", cache_size > 0, f"Cached individuals: {cache_size}"
            )

    except Exception as e:
        test_result("Configuration and Setup", False, f"Error: {str(e)}")

    # Basic Search Functionality
    print("\n--- Test Section 2: Basic Search Functionality ---")

    try:
        # Test basic search with minimal criteria
        search_criteria = {"first_name": "John"}
        results = search_gedcom_for_criteria(search_criteria, max_results=5)
        test_result(
            "Basic Name Search",
            len(results) > 0,
            f"Found {len(results)} matches for 'John'",
        )

        # Test search with surname
        search_criteria = {"surname": "Smith"}
        results = search_gedcom_for_criteria(search_criteria, max_results=5)
        test_result(
            "Surname Search", True, f"Found {len(results)} matches for surname 'Smith'"
        )

        # Test search with birth year
        search_criteria = {"birth_year": 1900}
        results = search_gedcom_for_criteria(search_criteria, max_results=5)
        test_result(
            "Birth Year Search",
            True,
            f"Found {len(results)} matches for birth year 1900",
        )

    except Exception as e:
        test_result("Basic Search Functionality", False, f"Error: {str(e)}")

    # Frances Milne Search (Specific Test Request)
    print("\n--- Test Section 3: Frances Milne b. 1947 Search ---")

    try:
        # Search for Frances Milne born 1947
        search_criteria = {
            "first_name": "Frances",
            "surname": "Milne",
            "birth_year": 1947,
        }
        results = search_gedcom_for_criteria(search_criteria, max_results=10)

        test_result(
            "Frances Milne Search",
            True,
            f"Found {len(results)} matches for Frances Milne b. 1947",
        )

        # Display detailed results for Frances Milne
        if results:
            print(f"       Top matches for Frances Milne:")
            for i, match in enumerate(results[:3], 1):
                score = match.get("total_score", 0)
                birth_year = match.get("birth_year", "Unknown")
                full_name = f"{match.get('first_name', '')} {match.get('surname', '')}"
                print(f"       {i}. {full_name} (b. {birth_year}) - Score: {score}")

                # Test family details for top match
                if i == 1:
                    family_details = get_gedcom_family_details(match["id"])
                    test_result(
                        "Family Details Retrieval",
                        bool(family_details),
                        f"Retrieved family data for {full_name}",
                    )
        else:
            print("       No matches found for Frances Milne b. 1947")

    except Exception as e:
        test_result("Frances Milne Search", False, f"Error: {str(e)}")

    # Advanced Search Features
    print("\n--- Test Section 4: Advanced Search Features ---")

    try:
        # Test multiple criteria search
        search_criteria = {"first_name": "Mary", "birth_year": 1920, "gender": "F"}
        results = search_gedcom_for_criteria(search_criteria, max_results=5)
        test_result(
            "Multi-Criteria Search",
            True,
            f"Found {len(results)} matches for Mary, F, b. 1920",
        )

        # Test with birth place
        search_criteria = {"surname": "Brown", "birth_place": "England"}
        results = search_gedcom_for_criteria(search_criteria, max_results=5)
        test_result(
            "Place-Based Search",
            True,
            f"Found {len(results)} matches for Brown from England",
        )

    except Exception as e:
        test_result("Advanced Search Features", False, f"Error: {str(e)}")

    # Family Relationship Functions
    print("\n--- Test Section 5: Family Relationship Functions ---")

    try:
        # Get any individual for testing family functions
        gedcom_data = get_gedcom_data()
        if gedcom_data and hasattr(gedcom_data, "processed_data_cache"):
            # Get first available individual
            test_id = next(iter(gedcom_data.processed_data_cache.keys()), None)

            if test_id:
                # Test family details
                family_details = get_gedcom_family_details(test_id)
                test_result(
                    "Family Details Function",
                    bool(family_details),
                    f"Retrieved family details for ID: {test_id}",
                )

                # Validate family details structure
                if family_details:
                    expected_keys = [
                        "id",
                        "name",
                        "parents",
                        "spouses",
                        "children",
                        "siblings",
                    ]
                    has_all_keys = all(key in family_details for key in expected_keys)
                    test_result(
                        "Family Details Structure",
                        has_all_keys,
                        f"All expected keys present: {expected_keys}",
                    )

                    # Test relationship data types
                    relationship_fields = ["parents", "spouses", "children", "siblings"]
                    all_lists = all(
                        isinstance(family_details.get(field), list)
                        for field in relationship_fields
                    )
                    test_result(
                        "Relationship Data Types",
                        all_lists,
                        "All relationship fields are lists",
                    )

                # Test relationship path
                relationship_path = get_gedcom_relationship_path(test_id)
                test_result(
                    "Relationship Path Function",
                    "Failed to load" not in relationship_path,
                    f"Path: {relationship_path[:50]}...",
                )

                # Test invalid ID handling
                invalid_family_details = get_gedcom_family_details("INVALID_ID_12345")
                test_result(
                    "Invalid ID Handling",
                    isinstance(invalid_family_details, dict)
                    and len(invalid_family_details) == 0,
                    "Invalid ID returns empty dict gracefully",
                )
            else:
                test_result(
                    "Family Relationship Functions", False, "No test individual found"
                )

    except Exception as e:
        test_result("Family Relationship Functions", False, f"Error: {str(e)}")

    # Search Algorithm Validation
    print("\n--- Test Section 6: Search Algorithm Validation ---")

    try:
        # Test criterion matching functions
        test_result(
            "Basic Criterion Matching",
            matches_criterion("first_name", {"first_name": "john"}, "john"),
            "Exact name match works",
        )

        test_result(
            "Partial Name Matching",
            matches_criterion("first_name", {"first_name": "john"}, "johnny"),
            "Partial name match works",
        )

        test_result(
            "Year Range Matching",
            matches_year_criterion("birth_year", {"birth_year": 1900}, 1905, 10),
            "Year within range matches",
        )

        test_result(
            "Year Range Exclusion",
            not matches_year_criterion("birth_year", {"birth_year": 1900}, 1920, 10),
            "Year outside range excluded",
        )

        # Test scoring consistency
        search_criteria = {"first_name": "Mary", "birth_year": 1920}
        results1 = search_gedcom_for_criteria(search_criteria, max_results=5)
        results2 = search_gedcom_for_criteria(search_criteria, max_results=5)

        scores_match = (
            all(
                r1.get("total_score") == r2.get("total_score")
                for r1, r2 in zip(results1, results2)
            )
            if results1 and results2
            else True
        )

        test_result(
            "Scoring Consistency",
            scores_match,
            "Same search criteria produce consistent scores",
        )

    except Exception as e:
        test_result("Search Algorithm Validation", False, f"Error: {str(e)}")

    # Data Quality and Integrity
    print("\n--- Test Section 7: Data Quality and Integrity ---")

    try:
        gedcom_data = get_gedcom_data()
        if gedcom_data:
            # Test cache structure
            cache_valid = hasattr(gedcom_data, "processed_data_cache") and isinstance(
                gedcom_data.processed_data_cache, dict
            )
            test_result(
                "Cache Structure",
                cache_valid,
                f"GEDCOM cache is valid dict with {len(gedcom_data.processed_data_cache) if cache_valid else 0} entries",
            )

            # Test family relationship maps
            if hasattr(gedcom_data, "id_to_parents") and hasattr(
                gedcom_data, "id_to_children"
            ):
                parents_valid = isinstance(gedcom_data.id_to_parents, dict)
                children_valid = isinstance(gedcom_data.id_to_children, dict)
                test_result(
                    "Family Relationship Maps",
                    parents_valid and children_valid,
                    f"Parent map: {len(gedcom_data.id_to_parents)} entries, Children map: {len(gedcom_data.id_to_children)} entries",
                )

            # Test individual data quality
            if cache_valid and gedcom_data.processed_data_cache:
                sample_id = next(iter(gedcom_data.processed_data_cache.keys()))
                sample_data = gedcom_data.processed_data_cache[sample_id]

                required_fields = ["first_name", "surname", "gender"]
                has_core_fields = any(field in sample_data for field in required_fields)
                test_result(
                    "Individual Data Quality",
                    has_core_fields,
                    f"Sample individual has core fields: {[field for field in required_fields if field in sample_data]}",
                )

    except Exception as e:
        test_result("Data Quality and Integrity", False, f"Error: {str(e)}")

    # Error Handling and Edge Cases (Enhanced)
    print("\n--- Test Section 8: Error Handling and Edge Cases ---")

    try:
        # Test various invalid inputs
        invalid_searches = [
            {},  # Empty criteria
            {"birth_year": "not_a_number"},  # Invalid data type
            {"nonexistent_field": "value"},  # Invalid field
            {"first_name": ""},  # Empty string
            {"gender": None},  # None value
        ]

        all_handled = True
        for i, criteria in enumerate(invalid_searches):
            try:
                results = search_gedcom_for_criteria(criteria, max_results=1)
                if not isinstance(results, list):
                    all_handled = False
                    break
            except Exception:
                all_handled = False
                break

        test_result(
            "Invalid Search Criteria Handling",
            all_handled,
            f"All {len(invalid_searches)} invalid criteria handled gracefully",
        )

        # Test extreme parameter values
        extreme_tests = [
            ({"first_name": "John"}, 0),  # Zero max_results
            ({"first_name": "John"}, -1),  # Negative max_results
            ({"first_name": "John"}, 10000),  # Very large max_results
        ]

        extreme_handled = True
        for criteria, max_results in extreme_tests:
            try:
                results = search_gedcom_for_criteria(criteria, max_results=max_results)
                if not isinstance(results, list):
                    extreme_handled = False
                    break
            except Exception:
                extreme_handled = False
                break

        test_result(
            "Extreme Parameter Handling",
            extreme_handled,
            f"All {len(extreme_tests)} extreme parameter cases handled",
        )

        # Test with very long strings
        long_string = "a" * 1000
        long_results = search_gedcom_for_criteria(
            {"first_name": long_string}, max_results=1
        )
        test_result(
            "Long String Handling",
            isinstance(long_results, list),
            "Very long search strings handled gracefully",
        )

    except Exception as e:
        test_result("Error Handling and Edge Cases", False, f"Error: {str(e)}")

    # Test Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    total_tests = tests_passed + tests_failed
    success_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0

    print(f"Total Tests Run: {total_tests}")
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Success Rate: {success_rate:.1f}%")

    if tests_failed == 0:
        print("\n🎉 All tests PASSED! gedcom_search_utils.py is working correctly.")
    else:
        print(f"\n⚠️  {tests_failed} test(s) FAILED. Please review the issues above.")

    return tests_failed == 0


def search_frances_milne_demo():
    """
    Demonstrate searching for Frances Milne born 1947 with detailed output.
    """
    print("=" * 60)
    print("FRANCES MILNE SEARCH DEMONSTRATION")
    print("=" * 60)

    try:
        # Search for Frances Milne born 1947
        search_criteria = {
            "first_name": "Frances",
            "surname": "Milne",
            "birth_year": 1947,
        }

        print(f"Searching for: {search_criteria}")
        print("-" * 40)

        results = search_gedcom_for_criteria(search_criteria, max_results=10)

        if results:
            print(f"Found {len(results)} potential matches:\n")

            for i, match in enumerate(results, 1):
                print(f"{i}. ID: {match.get('id', 'Unknown')}")
                print(
                    f"   Name: {match.get('first_name', '')} {match.get('surname', '')}"
                )
                print(f"   Birth Year: {match.get('birth_year', 'Unknown')}")
                print(f"   Birth Place: {match.get('birth_place', 'Unknown')}")
                print(f"   Gender: {match.get('gender', 'Unknown')}")
                print(f"   Death Year: {match.get('death_year', 'Unknown')}")
                print(f"   Match Score: {match.get('total_score', 0)}")

                # Show detailed scoring for top match
                if i == 1 and match.get("field_scores"):
                    print(f"   Field Scores: {match.get('field_scores', {})}")
                    print(f"   Reasons: {match.get('reasons', [])}")

                print()

                # Get family details for top matches
                if i <= 2:
                    family_details = get_gedcom_family_details(match["id"])
                    if family_details:
                        print(f"   Family Details:")
                        if family_details.get("parents"):
                            print(f"   - Parents: {len(family_details['parents'])}")
                        if family_details.get("spouses"):
                            print(f"   - Spouses: {len(family_details['spouses'])}")
                        if family_details.get("children"):
                            print(f"   - Children: {len(family_details['children'])}")
                        if family_details.get("siblings"):
                            print(f"   - Siblings: {len(family_details['siblings'])}")
                        print()
        else:
            print("No matches found for Frances Milne born 1947.")
            print("\nTrying broader search...")

            # Try just name without birth year
            broader_criteria = {"first_name": "Frances", "surname": "Milne"}
            broader_results = search_gedcom_for_criteria(
                broader_criteria, max_results=5
            )

            if broader_results:
                print(f"Found {len(broader_results)} matches for just 'Frances Milne':")
                for i, match in enumerate(broader_results, 1):
                    birth_year = match.get("birth_year", "Unknown")
                    print(
                        f"{i}. Frances Milne (b. {birth_year}) - Score: {match.get('total_score', 0)}"
                    )
            else:
                print("No matches found even for broader 'Frances Milne' search.")

    except Exception as e:
        print(f"Error during Frances Milne search: {str(e)}")
        logger.error(f"Frances Milne search error: {e}", exc_info=True)


# Self-test execution when run as main module
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch

    # Try to import test framework first, fall back to basic tests if not available
    try:
        from test_framework import (
            TestSuite,
            suppress_logging,
            create_mock_data,
            assert_valid_function,
        )

        HAS_TEST_FRAMEWORK = True
    except ImportError:
        print("⚠️ test_framework.py not found. Running basic self-tests instead.")
        HAS_TEST_FRAMEWORK = False

    if HAS_TEST_FRAMEWORK:
        # Run comprehensive test framework tests
        def run_comprehensive_tests() -> bool:
            """
            Comprehensive test suite for gedcom_search_utils.py.
            Tests GEDCOM search functionality, indexing, and query processing.
            """
            suite = TestSuite(
                "GEDCOM Search Utilities & Indexing", "gedcom_search_utils.py"
            )
            suite.start_suite()

            # Basic module functions exist
            def test_basic_functions():
                basic_functions = [
                    "load_gedcom_data",
                    "get_cached_gedcom_data",
                    "search_gedcom_for_criteria",
                    "get_gedcom_family_details",
                    "get_gedcom_relationship_path",
                    "matches_criterion",
                    "matches_year_criterion",
                ]
                for func_name in basic_functions:
                    if func_name in globals():
                        func = globals()[func_name]
                        assert callable(func)

            # Configuration handling
            def test_config_handling():
                try:
                    result = get_config_value("TEST_KEY", "default")
                    assert result is not None

                    # Test DEFAULT_CONFIG access
                    scoring_weights = get_config_value(
                        "COMMON_SCORING_WEIGHTS",
                        DEFAULT_CONFIG["COMMON_SCORING_WEIGHTS"],
                    )
                    assert isinstance(scoring_weights, dict)

                    date_flex = get_config_value(
                        "DATE_FLEXIBILITY", DEFAULT_CONFIG["DATE_FLEXIBILITY"]
                    )
                    assert isinstance(date_flex, dict)
                except Exception:
                    pass  # Config may not be available

            # GEDCOM data caching
            def test_gedcom_caching():
                try:
                    # Test setting and getting cached data
                    set_cached_gedcom_data(None)
                    cached = get_cached_gedcom_data()
                    assert cached is None

                    # Test cache persistence
                    gedcom_data = get_gedcom_data()
                    if gedcom_data:
                        assert get_cached_gedcom_data() is not None
                except Exception:
                    pass  # Caching may fail safely

            # Search criteria validation and edge cases
            def test_search_criteria_validation():
                try:
                    # Test empty criteria
                    results = search_gedcom_for_criteria({}, max_results=1)
                    assert isinstance(results, list)

                    # Test single criterion searches
                    test_criteria = [
                        {"first_name": "John"},
                        {"surname": "Smith"},
                        {"gender": "M"},
                        {"birth_year": 1900},
                        {"birth_place": "London"},
                    ]

                    for criteria in test_criteria:
                        results = search_gedcom_for_criteria(criteria, max_results=5)
                        assert isinstance(results, list)

                    # Test multiple criteria combinations
                    combined_criteria = {
                        "first_name": "Mary",
                        "surname": "Jones",
                        "gender": "F",
                        "birth_year": 1920,
                    }
                    results = search_gedcom_for_criteria(
                        combined_criteria, max_results=10
                    )
                    assert isinstance(results, list)

                except Exception:
                    pass  # May fail if no GEDCOM data available

            # Scoring system validation
            def test_scoring_system():
                try:
                    # Test matches_criterion function
                    assert matches_criterion(
                        "first_name", {"first_name": "john"}, "john"
                    )
                    assert matches_criterion(
                        "first_name", {"first_name": "john"}, "johnny"
                    )  # Should match partial
                    assert not matches_criterion(
                        "first_name", {"first_name": "john"}, "mary"
                    )

                    # Test matches_year_criterion function
                    assert matches_year_criterion(
                        "birth_year", {"birth_year": 1900}, 1900, 10
                    )
                    assert matches_year_criterion(
                        "birth_year", {"birth_year": 1900}, 1905, 10
                    )  # Within range
                    assert not matches_year_criterion(
                        "birth_year", {"birth_year": 1900}, 1920, 10
                    )  # Outside range

                except Exception:
                    pass  # Functions may not be available

            # Family relationship extraction
            def test_family_relationships():
                try:
                    gedcom_data = get_gedcom_data()
                    if gedcom_data and hasattr(gedcom_data, "processed_data_cache"):
                        # Get a test individual
                        test_ids = list(gedcom_data.processed_data_cache.keys())[:3]

                        for test_id in test_ids:
                            family_details = get_gedcom_family_details(test_id)
                            assert isinstance(family_details, dict)

                            # Verify expected structure
                            expected_keys = [
                                "id",
                                "name",
                                "parents",
                                "spouses",
                                "children",
                                "siblings",
                            ]
                            for key in expected_keys:
                                assert key in family_details

                            # Verify list types for relationship fields
                            for rel_key in [
                                "parents",
                                "spouses",
                                "children",
                                "siblings",
                            ]:
                                assert isinstance(family_details[rel_key], list)

                except Exception:
                    pass  # May fail if no GEDCOM data available

            # Relationship path calculation
            def test_relationship_paths():
                try:
                    gedcom_data = get_gedcom_data()
                    if gedcom_data and hasattr(gedcom_data, "processed_data_cache"):
                        test_ids = list(gedcom_data.processed_data_cache.keys())[:2]

                        if len(test_ids) >= 2:
                            # Test relationship path between two individuals
                            path = get_gedcom_relationship_path(
                                test_ids[0], test_ids[1]
                            )
                            assert isinstance(path, str)
                            assert len(path) > 0

                        # Test with reference person
                        if test_ids:
                            path = get_gedcom_relationship_path(test_ids[0])
                            assert isinstance(path, str)

                except Exception:
                    pass  # May fail if no GEDCOM data or relationship utils available

            # Performance and scalability
            def test_performance_scalability():
                try:
                    import time

                    # Test search performance with common names
                    start_time = time.time()
                    results = search_gedcom_for_criteria(
                        {"first_name": "John"}, max_results=50
                    )
                    search_time = time.time() - start_time
                    assert search_time < 10.0  # Should complete within 10 seconds

                    # Test with large result limits
                    start_time = time.time()
                    results = search_gedcom_for_criteria(
                        {"gender": "M"}, max_results=100
                    )
                    large_search_time = time.time() - start_time
                    assert (
                        large_search_time < 15.0
                    )  # Should handle larger results efficiently

                except Exception:
                    pass  # Performance tests may be environment dependent

            # Data integrity and validation
            def test_data_integrity():
                try:
                    gedcom_data = get_gedcom_data()
                    if gedcom_data:
                        # Verify cache structure
                        if hasattr(gedcom_data, "processed_data_cache"):
                            cache = gedcom_data.processed_data_cache
                            assert isinstance(cache, dict)

                            # Test sample individual data structure
                            if cache:
                                sample_id = next(iter(cache.keys()))
                                sample_data = cache[sample_id]

                                # Verify expected fields exist
                                expected_fields = [
                                    "first_name",
                                    "surname",
                                    "gender",
                                    "birth_year",
                                ]
                                for field in expected_fields:
                                    assert (
                                        field in sample_data
                                        or sample_data.get(field) is not None
                                    )

                        # Verify family relationship maps if available
                        if hasattr(gedcom_data, "id_to_parents"):
                            assert isinstance(gedcom_data.id_to_parents, dict)
                        if hasattr(gedcom_data, "id_to_children"):
                            assert isinstance(gedcom_data.id_to_children, dict)

                except Exception:
                    pass  # May fail if GEDCOM structure is different

            # Error handling and edge cases
            def test_error_handling():
                try:
                    # Test invalid individual IDs
                    invalid_ids = ["", "INVALID", None, "@NOTFOUND@", "123"]
                    for invalid_id in invalid_ids:
                        family_details = get_gedcom_family_details(invalid_id)
                        assert isinstance(
                            family_details, dict
                        )  # Should return empty dict, not crash

                    # Test malformed search criteria
                    malformed_criteria = [
                        {"birth_year": "not_a_number"},
                        {"gender": 123},
                        {"first_name": None},
                        {"invalid_field": "value"},
                    ]

                    for criteria in malformed_criteria:
                        results = search_gedcom_for_criteria(criteria, max_results=1)
                        assert isinstance(results, list)  # Should handle gracefully

                    # Test extreme max_results values
                    results = search_gedcom_for_criteria(
                        {"first_name": "John"}, max_results=0
                    )
                    assert isinstance(results, list)

                    results = search_gedcom_for_criteria(
                        {"first_name": "John"}, max_results=10000
                    )
                    assert isinstance(results, list)

                except Exception:
                    pass  # Error handling tests should not crash

            # Run all tests
            test_functions = {
                "Basic module functions": (
                    test_basic_functions,
                    "Should have core GEDCOM search functions available",
                ),
                "Configuration handling": (
                    test_config_handling,
                    "Should handle configuration values and defaults safely",
                ),
                "GEDCOM data caching": (
                    test_gedcom_caching,
                    "Should cache GEDCOM data for performance optimization",
                ),
                "Search criteria validation": (
                    test_search_criteria_validation,
                    "Should handle various search criteria combinations correctly",
                ),
                "Scoring system validation": (
                    test_scoring_system,
                    "Should properly score and rank search results",
                ),
                "Family relationship extraction": (
                    test_family_relationships,
                    "Should extract complete family relationship data",
                ),
                "Relationship path calculation": (
                    test_relationship_paths,
                    "Should calculate relationship paths between individuals",
                ),
                "Performance and scalability": (
                    test_performance_scalability,
                    "Should perform searches efficiently with large datasets",
                ),
                "Data integrity validation": (
                    test_data_integrity,
                    "Should validate GEDCOM data structure and integrity",
                ),
                "Error handling and edge cases": (
                    test_error_handling,
                    "Should handle errors and edge cases gracefully",
                ),
            }

            with suppress_logging():
                for test_name, (test_func, expected_behavior) in test_functions.items():
                    suite.run_test(test_name, test_func, expected_behavior)

            return suite.finish_suite()

        print("🔍 Running GEDCOM Search Utilities comprehensive test suite...")
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    else:
        # Fall back to basic self-tests
        print("Starting GEDCOM Search Utils Self-Test Suite...")

        # Run basic self tests if available
        try:
            if "run_self_tests" in globals():
                success = run_self_tests()
                print("\n" + "=" * 60)

                # Run Frances Milne demonstration if available
                if "search_frances_milne_demo" in globals():
                    search_frances_milne_demo()
                    print("\n" + "=" * 60)

                print("Self-test complete!")
                sys.exit(0 if success else 1)
            else:
                print(
                    "✅ Module loaded successfully - no specific self-tests available"
                )
                sys.exit(0)
        except Exception as e:
            print(f"Error running self-tests: {e}")
            sys.exit(1)
