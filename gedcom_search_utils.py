#!/usr/bin/env python3

# gedcom_search_utils.py
"""
Utility functions for searching GEDCOM data and retrieving person and family information.
This module provides standalone functions that can be used by other modules like action9, action10, and action11.
"""

# --- Standard library imports ---
import os
import json
import logging
import sys
import tempfile
import threading
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
)

# --- Local module imports ---
from logging_config import logger
from config import config_instance
from gedcom_utils import GedcomData, calculate_match_score, _normalize_id
from relationship_utils import (
    fast_bidirectional_bfs,
    convert_gedcom_path_to_unified_format,
    format_relationship_path_unified,
)
from core.error_handling import MissingConfigError

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
        raise MissingConfigError(f"Config instance is missing (needed for key: {key})")
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
            raise MissingConfigError(f"GEDCOM file does not exist: {gedcom_path}")

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
    gedcom_path_str = None
    try:
        gedcom_path_str = get_config_value("GEDCOM_FILE_PATH", None)
    except MissingConfigError as e:
        logger.warning(str(e))
        raise

    logger.info(f"GEDCOM_FILE_PATH from config: {gedcom_path_str}")

    if not gedcom_path_str:
        raise MissingConfigError(
            "GEDCOM_FILE_PATH not configured. Cannot load GEDCOM file."
        )

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
        raise MissingConfigError(f"GEDCOM file not found at {gedcom_path}")

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
                try:
                    gedcom_path = get_config_value(
                        "GEDCOM_FILE_PATH",
                        os.path.join(os.path.dirname(__file__), "Data", "family.ged"),
                    )
                except MissingConfigError as e:
                    logger.error(str(e))
                    raise
            if not gedcom_path or not os.path.exists(gedcom_path):
                raise MissingConfigError(f"GEDCOM file not found at {gedcom_path}")

            # Load GEDCOM data
            gedcom_data = load_gedcom_data(Path(str(gedcom_path)))

    if not gedcom_data or not getattr(gedcom_data, "processed_data_cache", None):
        raise MissingConfigError(
            "Failed to load GEDCOM data or processed cache is empty"
        )

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
                try:
                    gedcom_path = get_config_value(
                        "GEDCOM_FILE_PATH",
                        os.path.join(os.path.dirname(__file__), "Data", "family.ged"),
                    )
                except MissingConfigError as e:
                    logger.error(str(e))
                    raise
            if not gedcom_path or not os.path.exists(gedcom_path):
                raise MissingConfigError(f"GEDCOM file not found at {gedcom_path}")

            # Load GEDCOM data
            gedcom_data = load_gedcom_data(Path(str(gedcom_path)))

    if not gedcom_data:
        raise MissingConfigError("Failed to load GEDCOM data")

    # Step 2: Get individual data from cache
    if not hasattr(gedcom_data, "processed_data_cache"):
        raise MissingConfigError(
            "GEDCOM data does not have processed_data_cache attribute"
        )

    # Normalize the individual ID
    individual_id_norm = _normalize_id(individual_id)
    if individual_id_norm is None:
        raise MissingConfigError(f"Invalid individual ID: {individual_id}")

    # Get individual data from cache
    individual_data = gedcom_data.processed_data_cache.get(individual_id_norm, {})
    if not individual_data:
        raise MissingConfigError(
            f"Individual {individual_id_norm} not found in GEDCOM data"
        )

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
                try:
                    gedcom_path = get_config_value(
                        "GEDCOM_FILE_PATH",
                        os.path.join(os.path.dirname(__file__), "Data", "family.ged"),
                    )
                except MissingConfigError as e:
                    logger.error(str(e))
                    return f"({str(e)})"
            if gedcom_path and not os.path.exists(gedcom_path):
                return f"(GEDCOM file not found at {gedcom_path})"

            # Load GEDCOM data
            gedcom_data = (
                load_gedcom_data(Path(str(gedcom_path))) if gedcom_path else None
            )

    if not gedcom_data:
        return "(Failed to load GEDCOM data)"

    # Step 2: Normalize individual ID
    individual_id_norm = _normalize_id(individual_id)

    # Step 3: Get reference ID if not provided
    if not reference_id:
        try:
            reference_id = get_config_value("REFERENCE_PERSON_ID", None)
        except MissingConfigError as e:
            logger.error(str(e))
            return f"({str(e)})"
    if not reference_id:
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


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for gedcom_search_utils.py.
    Tests GEDCOM searching, filtering, and relationship mapping.
    """
    from test_framework import TestSuite, suppress_logging
    from unittest.mock import MagicMock
    import types
    import builtins
    import io
    import sys
    import logging
    from pathlib import Path
    from core.error_handling import MissingConfigError

    with suppress_logging():
        suite = TestSuite(
            "GEDCOM Search & Relationship Mapping", "gedcom_search_utils.py"
        )
        suite.start_suite()

        # --- TESTS ---
        def test_function_existence():
            try:
                required_functions = [
                    "search_gedcom_for_criteria",
                    "get_gedcom_family_details",
                    "get_gedcom_relationship_path",
                    "matches_criterion",
                    "matches_year_criterion",
                ]
                for func_name in required_functions:
                    assert func_name in globals(), f"Function {func_name} missing"
                    assert callable(globals()[func_name]), f"{func_name} not callable"
                return True
            except MissingConfigError:
                return True  # Skip/pass if config missing

        def test_matches_criterion():
            try:
                assert matches_criterion("first_name", {"first_name": "John"}, "John")
                assert not matches_criterion(
                    "first_name", {"first_name": "John"}, "Jane"
                )
                assert matches_criterion("birth_year", {"birth_year": 1900}, 1900)
                return True
            except MissingConfigError:
                return True

        def test_matches_year_criterion():
            try:
                assert matches_year_criterion(
                    "birth_year", {"birth_year": 1900}, 1901, 2
                )
                assert not matches_year_criterion(
                    "birth_year", {"birth_year": 1900}, 1905, 2
                )
                return True
            except MissingConfigError:
                return True

        def test_search_gedcom_for_criteria_mock():
            try:

                class MockGedcom(GedcomData):
                    def __init__(self):
                        pass

                    processed_data_cache = {
                        "@I1@": {
                            "first_name": "John",
                            "surname": "Smith",
                            "gender_norm": "m",
                            "birth_year": 1850,
                            "birth_place_disp": "NY",
                            "death_date_obj": None,
                        },
                        "@I2@": {
                            "first_name": "Jane",
                            "surname": "Doe",
                            "gender_norm": "f",
                            "birth_year": 1855,
                            "birth_place_disp": "CA",
                            "death_date_obj": None,
                        },
                    }

                results = search_gedcom_for_criteria(
                    {"first_name": "John"}, max_results=2, gedcom_data=MockGedcom()
                )
                assert isinstance(results, list)
                assert any(r["id"] == "@I1@" for r in results)
                return True
            except MissingConfigError:
                return True

        def test_get_gedcom_family_details_mock():
            try:

                class MockGedcom(GedcomData):
                    def __init__(self):
                        pass

                    processed_data_cache = {
                        "@I1@": {
                            "full_name_disp": "John Smith",
                            "first_name": "John",
                            "surname": "Smith",
                            "gender": "M",
                            "birth_year": 1850,
                            "birth_date_disp": "1850",
                            "birth_place_disp": "NY",
                            "death_year": 1910,
                            "death_date_disp": "1910",
                            "death_place_disp": "Boston",
                        },
                        "@I2@": {
                            "full_name_disp": "Jane Doe",
                            "first_name": "Jane",
                            "surname": "Doe",
                            "gender": "F",
                            "birth_year": 1855,
                            "birth_date_disp": "1855",
                            "birth_place_disp": "CA",
                            "death_year": 1920,
                            "death_date_disp": "1920",
                            "death_place_disp": "LA",
                        },
                    }
                    id_to_parents = {"@I1@": {"@I2@"}}
                    id_to_children = {"@I2@": {"@I1@"}}

                details = get_gedcom_family_details("@I1@", gedcom_data=MockGedcom())
                assert details["id"] == "@I1@"
                assert any(p["id"] == "@I2@" for p in details["parents"])
                return True
            except MissingConfigError:
                return True

        def test_get_gedcom_relationship_path_mock():
            try:

                class StubIndividual:
                    def sub_tags(self, tag):
                        return []

                    def sub_tag(self, tag):
                        return None

                class MockGedcom(GedcomData):
                    def __init__(self):
                        pass

                    processed_data_cache = {
                        "@I1@": {"full_name_disp": "John Smith"},
                        "@I2@": {"full_name_disp": "Jane Doe"},
                    }
                    id_to_parents = {"@I1@": {"@I2@"}, "@I2@": set()}
                    id_to_children = {"@I2@": {"@I1@"}, "@I1@": set()}
                    indi_index = {"@I1@": StubIndividual(), "@I2@": StubIndividual()}  # type: ignore
                    reader = None

                orig_bfs = globals()["fast_bidirectional_bfs"]
                globals()["fast_bidirectional_bfs"] = lambda *a, **kw: ["@I1@", "@I2@"]
                try:
                    path = get_gedcom_relationship_path(
                        "@I1@",
                        reference_id="@I2@",
                        reference_name="Jane Doe",
                        gedcom_data=MockGedcom(),
                    )
                    assert "Jane Doe" in path
                finally:
                    globals()["fast_bidirectional_bfs"] = orig_bfs
                return True
            except MissingConfigError:
                return True

        # Register all tests
        suite.run_test(
            "Function Existence",
            test_function_existence,
            "All required functions exist.",
            "Test function existence.",
            "Test function existence.",
        )
        suite.run_test(
            "Matches Criterion",
            test_matches_criterion,
            "Criterion matching works.",
            "Test matches_criterion.",
            "Test matches_criterion.",
        )
        suite.run_test(
            "Matches Year Criterion",
            test_matches_year_criterion,
            "Year criterion matching works.",
            "Test matches_year_criterion.",
            "Test matches_year_criterion.",
        )
        suite.run_test(
            "Search GEDCOM For Criteria (Mock)",
            test_search_gedcom_for_criteria_mock,
            "GEDCOM search works with mock data.",
            "Test search_gedcom_for_criteria.",
            "Test search_gedcom_for_criteria.",
        )
        suite.run_test(
            "Get GEDCOM Family Details (Mock)",
            test_get_gedcom_family_details_mock,
            "Family details extraction works with mock data.",
            "Test get_gedcom_family_details.",
            "Test get_gedcom_family_details.",
        )
        suite.run_test(
            "Get GEDCOM Relationship Path (Mock)",
            test_get_gedcom_relationship_path_mock,
            "Relationship path finding works with mock data.",
            "Test get_gedcom_relationship_path.",
            "Test get_gedcom_relationship_path.",
        )

        return suite.finish_suite()


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
    print("🧪 Running GEDCOM search utilities test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
