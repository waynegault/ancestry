# action10.py
"""
Standalone script to perform Action 10: GEDCOM Report.
Loads a local GEDCOM file, prompts user for search criteria, finds matches,
allows selection, displays details, family, and relationship path
to the explicitly defined reference person ("Wayne Gordon Gault").
V16.0: Refactored from temp.py v7.36, using functions from gedcom_utils.py.
"""

# --- Standard library imports ---
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Any, Dict, List  # Import List
import time

# import inspect # Keep for config path debug if needed

# --- Local application imports ---
# Use centralized logging config setup
try:
    from logging_config import setup_logging, logger
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("action10")
    logger.warning("Using fallback logger for Action 10.")

# --- Load Config and SCORING CONSTANTS (Mandatory) ---
config_instance = None
COMMON_SCORING_WEIGHTS = None
NAME_FLEXIBILITY = None
DATE_FLEXIBILITY = None

try:
    # Import instance and scoring dicts directly from config
    from config import (
        config_instance,
        COMMON_SCORING_WEIGHTS,
        NAME_FLEXIBILITY,
        DATE_FLEXIBILITY,
    )

    logger.info("Successfully imported config_instance and scoring dictionaries.")
    # Basic validation that scoring weights are dictionaries
    if (
        not isinstance(COMMON_SCORING_WEIGHTS, dict)
        or not isinstance(NAME_FLEXIBILITY, dict)
        or not isinstance(DATE_FLEXIBILITY, dict)
    ):
        raise TypeError(
            "One or more scoring configurations imported from config.py is not a dictionary."
        )
    # Optional warning if weights dict is empty (might lead to 0 scores)
    if not COMMON_SCORING_WEIGHTS:
        logger.warning(
            "COMMON_SCORING_WEIGHTS dictionary imported from config.py is empty. Scoring may not function as expected."
        )
except ImportError as e:
    logger.critical(
        f"Failed to import config_instance or scoring dictionaries from config.py: {e}. Cannot proceed.",
        exc_info=True,
    )
    print(f"\nFATAL ERROR: Failed to import required components from config.py: {e}")
    sys.exit(1)
except TypeError as config_err:
    logger.critical(f"Configuration Error: {config_err}")
    print(f"\nFATAL ERROR: {config_err}")
    sys.exit(1)
except Exception as e:
    logger.critical(f"Unexpected error loading configuration: {e}", exc_info=True)
    print(f"\nFATAL ERROR: Unexpected error loading configuration: {e}")
    sys.exit(1)

# Ensure critical config components are loaded before proceeding
if (
    config_instance is None
    or COMMON_SCORING_WEIGHTS is None
    or NAME_FLEXIBILITY is None
    or DATE_FLEXIBILITY is None
):
    logger.critical("One or more critical configuration components failed to load.")
    print("\nFATAL ERROR: Configuration load failed.")
    sys.exit(1)


# --- Import GEDCOM Utilities ---
# Import specific functions needed from gedcom_utils
try:
    from gedcom_utils import (
        GedcomReader,
        Individual,  # Keep Individual type hint
        GEDCOM_LIB_AVAILABLE,  # Keep check flag
        build_indi_index,  # Import cache builder
        build_family_maps,  # Import cache builder
        find_individual_by_id,  # Import ID lookup
        find_potential_matches,  # Import fuzzy search
        get_related_individuals,  # Import family retrieval
        format_full_life_details,  # Import detail formatter
        format_relative_info,  # Import relative formatter
        get_relationship_path,  # Import path calculation
        _get_full_name,  # Import name getter for WGG search
        extract_and_fix_id,  # Import ID normalizer/fixer
    )

    logger.info("Successfully imported required functions from gedcom_utils.")

    # Check if GEDCOM_LIB_AVAILABLE is set True by gedcom_utils import
    if not GEDCOM_LIB_AVAILABLE:
        logger.critical("ged4py library is not available based on gedcom_utils check.")

except ImportError as e:
    logger.critical(
        f"Failed to import gedcom_utils module or its core components: {e}.",
        exc_info=True,
    )
    # Set needed variables to None or False if import fails critically
    GedcomReader = None
    Individual = None  # type: ignore
    GEDCOM_LIB_AVAILABLE = False
    build_indi_index = None  # type: ignore
    build_family_maps = None  # type: ignore
    find_individual_by_id = None  # type: ignore
    find_potential_matches = None  # type: ignore
    get_related_individuals = None  # type: ignore
    format_full_life_details = None  # type: ignore
    format_relative_info = None  # type: ignore
    get_relationship_path = None  # type: ignore
    _get_full_name = None  # type: ignore
    extract_and_fix_id = None  # type: ignore


# --- Import General Utilities (for format_name, ordinal_case fallback check) ---
# The format_name and ordinal_case functions are primarily used by gedcom_utils,
# but we keep this import block as a reminder that they are external dependencies
# if they were ever needed directly in action10.
try:
    # Import format_name, ordinal_case if needed directly, otherwise gedcom_utils handles it
    from utils import format_name  # Example import

    logger.debug("Successfully imported format_name from utils.")
except ImportError:
    logger.debug(
        "Failed to import format_name from utils. Relying on gedcom_utils fallbacks if needed."
    )


# --- Main GEDCOM Report Handler ---
def handle_gedcom_report(
    reader,
    wgg_indi: Optional[Any],  # Pass WGG Individual object
    wgg_id: Optional[str],  # Pass WGG normalized ID
    scoring_weights: Dict,
    name_flexibility: Dict,
    date_flexibility: Dict,
    max_results_display: int = 3,  # Limit displayed results from search
):
    """
    Handler for GEDCOM Report functionality.
    Prompts user for criteria, finds matches, displays details, family,
    and relationship path to WGG.
    Uses functions from gedcom_utils.
    """
    # Check if core gedcom_utils functions are available
    if not all(
        [
            reader,
            find_potential_matches,
            find_individual_by_id,
            get_related_individuals,
            format_full_life_details,
            format_relative_info,
            get_relationship_path,
            _get_full_name,
            extract_and_fix_id,
        ]
    ):
        logger.error(
            "handle_gedcom_report: Required gedcom_utils functions not loaded."
        )
        print("\nERROR: Core GEDCOM utils missing.")
        return False  # Indicate failure

    # Reference specific person name for logging/display
    reference_person_name = "Wayne Gordon Gault"
    logger.info(
        f"\n--- Person Details & Relationship to {reference_person_name} (GEDCOM Report) ---"
    )

    # Check if WGG was found in GEDCOM during startup
    if not wgg_indi or not wgg_id:
        logger.warning(
            f"Reference person '{reference_person_name}' not found or invalid in GEDCOM. Path calculation unavailable."
        )
        # Note: We proceed with search/details display even if WGG is not found,
        # but relationship path will be skipped.

    logger.info("\nEnter search criteria for the person of interest:")
    # Use try/except for input to handle EOFError (Ctrl+D) or other input issues
    try:
        first_name = input(" First Name (optional): ").strip() or None
        surname = input(" Surname (optional): ").strip() or None
        dob_str = input(" Birth Date/Year (optional): ").strip() or None
        pob = input(" Birth Place (optional): ").strip() or None
        dod_str = (
            input(" Death Date/Year (optional): ").strip() or None
        )  # Added Death Date input
        pod = (
            input(" Death Place (optional): ").strip() or None
        )  # Added Death Place input
        gender = input(" Gender (M/F, optional): ").strip() or None
        if gender:
            gender = (
                gender[0].lower() if gender[0].lower() in ["m", "f"] else None
            )  # Normalize gender input
    except EOFError:
        print("\nInput cancelled.")
        return False  # Indicate operation cancelled
    except Exception as input_err:
        logger.error(f"Error reading user input: {input_err}", exc_info=True)
        print("\nError reading input.")
        return False  # Indicate failure

    # Check if any criteria were entered
    if not any([first_name, surname, dob_str, pob, dod_str, pod, gender]):
        logger.info("\nNo search criteria entered. Report cancelled.")
        return True  # Indicate success (cancelled by user intent)

    logger.info("Searching GEDCOM...")
    try:
        # Call the fuzzy search function from gedcom_utils
        matches = find_potential_matches(
            reader=reader,
            first_name=first_name,
            surname=surname,
            dob_str=dob_str,
            pob=pob,
            dod_str=dod_str,  # Pass death criteria
            pod=pod,  # Pass death criteria
            gender=gender,
            max_results=10,  # Fetch slightly more than display limit for better internal sorting
            scoring_weights=scoring_weights,  # Pass config dictionaries
            name_flexibility=name_flexibility,  # Pass config dictionaries
            date_flexibility=date_flexibility,  # Pass config dictionaries
        )
    except AttributeError as ae:
        logger.error(
            f"AttributeError during GEDCOM search - function missing?: {ae}",
            exc_info=True,
        )
        print("\nSearch error (function missing?). Check logs.")
        return False  # Indicate failure
    except Exception as search_err:
        logger.error(
            f"Error during find_potential_matches: {search_err}", exc_info=True
        )
        print("\nAn error occurred during the search. Check logs.")
        return False  # Indicate failure

    # --- Display Matches and Handle Selection ---
    if not matches:
        print("\nNo potential matches found.")
        return True  # Indicate success (no matches found is a valid outcome)

    # Always auto-select the top match, as per temp.py v7.36 logic
    selected_match = matches[0]
    print(f"\n--- Top Match & Close Results ({len(matches)} total matches found) ---")

    # Display the top match and potentially a few more close ones
    display_matches_count = min(len(matches), max_results_display)
    for i, match in enumerate(matches[:display_matches_count]):
        # Format match details for display
        b_info = (
            f"b. {match['birth_date']}"
            if match.get("birth_date") and match["birth_date"] != "N/A"
            else ""
        )
        d_info = (
            f"d. {match['death_date']}"
            if match.get("death_date") and match["death_date"] != "N/A"
            else ""
        )
        date_info = (
            f" ({', '.join(filter(None, [b_info, d_info]))}" if b_info or d_info else ""
        )
        birth_place_info = (
            f"in {match['birth_place']}"
            if match.get("birth_place") and match["birth_place"] != "N/A"
            else ""
        )
        death_place_info = (
            f"in {match['death_place']}"
            if match.get("death_place") and match["death_place"] != "N/A"
            else ""
        )

        print(f"  {i+1}. {match['name']}")
        print(f"     Born : {match.get('birth_date', '?')} {birth_place_info}")
        # Only print death line if there is death info
        if match.get("death_date") != "N/A" or match.get("death_place") != "N/A":
            print(f"     Died : {match.get('death_date', '?')} {death_place_info}")
        print(
            f"     Score: {match.get('score', '?')} (Reasons: {match.get('reasons', 'N/A')})"
        )
        # Optional: print ID for debug
        # print(f"     ID   : {match.get('id', 'N/A')}")

    print(f"\n---> Auto-selecting top match: {selected_match['name']}")

    # --- Process Selected Match ---
    # Use the normalized ID stored in the match dict
    selected_id_norm = selected_match.get("norm_id")

    if not selected_id_norm:
        logger.error(
            f"Invalid normalized ID in selected match: '{selected_match.get('id', 'N/A')}'"
        )
        print("\nERROR: Could not process ID for the selected match.")
        return False  # Indicate failure

    # Retrieve the Individual object from the reader using the normalized ID
    selected_indi = find_individual_by_id(reader, selected_id_norm)
    if not selected_indi:
        logger.error(
            f"Could not retrieve INDI record for normalized ID: {selected_id_norm}"
        )
        print("\nERROR: Failed to retrieve details for the selected person.")
        return False  # Indicate failure

    # --- Display Details and Family using gedcom_utils functions ---
    print("\n=== PERSON DETAILS (GEDCOM) ===")
    indi_name_disp = _get_full_name(selected_indi)
    # Use format_full_life_details from gedcom_utils
    birth_info_disp, death_info_disp = format_full_life_details(selected_indi)
    gender_raw = getattr(selected_indi, "sex", None)
    gender_disp = (
        str(gender_raw).upper()
        if gender_raw
        and isinstance(gender_raw, str)
        and str(gender_raw).upper() in ("M", "F")
        else "N/A"
    )

    print(f"Name: {indi_name_disp}")
    print(f"Gender: {gender_disp}")
    print(birth_info_disp)
    if death_info_disp:  # Only print death details if they exist
        print(death_info_disp)

    # Generate Ancestry link if tree_id and person_id are available
    tree_id = getattr(config_instance, "MY_TREE_ID", None)
    # Use the normalized ID for the Ancestry link if it looks like an Ancestry ID (starts with I)
    # Ancestry IDs typically start with I followed by digits/hyphens, like I123-456 or I123456
    ancestry_id_for_link = (
        selected_id_norm
        if selected_id_norm
        and (selected_id_norm.startswith("I") or selected_id_norm.startswith("U"))
        else None
    )

    if tree_id and ancestry_id_for_link:
        # Basic check to ensure it looks like a valid Ancestry person ID format
        if re.match(r"^[IU][0-9A-Z-]+$", ancestry_id_for_link):
            print(
                f"Link in Tree: https://www.ancestry.co.uk/family-tree/person/tree/{tree_id}/person/{ancestry_id_for_link}/facts"
            )
        elif selected_id_norm:
            # If normalized ID exists but doesn't look like Ancestry format, just print the GEDCOM ID
            print(f"Link in Tree: (GEDCOM ID '{selected_id_norm}')")
        else:
            print("Link in Tree: (unavailable)")
    elif selected_id_norm:
        # If no tree ID is configured but normalized ID exists, print the GEDCOM ID
        print(f"Link in Tree: (GEDCOM ID '{selected_id_norm}')")
    else:
        print("Link in Tree: (unavailable)")

    print("\n--- Family Details (GEDCOM) ---")
    try:
        # Display Parents using get_related_individuals and format_relative_info
        print("\nParents:")
        parents = get_related_individuals(reader, selected_indi, "parents")
        if parents:
            for p in parents:
                print(
                    format_relative_info(p)
                )  # format_relative_info handles prefix and details
        else:
            print("  (None found)")

        # Display Siblings using get_related_individuals and format_relative_info
        print("\nSiblings:")
        siblings = get_related_individuals(reader, selected_indi, "siblings")
        if siblings:
            for s in siblings:
                print(
                    format_relative_info(s)
                )  # format_relative_info handles prefix and details
        else:
            print("  (None found)")

        # Display Spouses using get_related_individuals and format_relative_info
        print("\nSpouse(s):")
        spouses = get_related_individuals(reader, selected_indi, "spouses")
        if spouses:
            for s in spouses:
                print(
                    format_relative_info(s)
                )  # format_relative_info handles prefix and details
        else:
            print("  (None found)")

        # Display Children using get_related_individuals and format_relative_info
        print("\nChildren:")
        children = get_related_individuals(reader, selected_indi, "children")
        if children:
            for c in children:
                print(
                    format_relative_info(c)
                )  # format_relative_info handles prefix and details
        else:
            print("  (None found)")

    except AttributeError as ae:
        logger.error(
            f"AttributeError displaying family - function missing?: {ae}", exc_info=True
        )
        print("\nError displaying family (function missing?). Check logs.")
        # Do not return False here, as the main details were displayed

    except Exception as display_err:
        logger.error(f"Error displaying family details: {display_err}", exc_info=True)
        print("\nError displaying family details. Check logs.")
        # Do not return False here

    # --- Display Relationship Path to WGG ---
    # Only calculate and display if WGG was found in GEDCOM during startup
    if wgg_id and selected_id_norm:
        print(
            f"\nCalculating relationship path to {reference_person_name}..."
        )  # Use specific name

        try:
            # Call get_relationship_path from gedcom_utils
            relationship_path_str = get_relationship_path(
                reader, selected_id_norm, wgg_id
            )
            print(
                f"\n--- Relationship Path to {reference_person_name} (GEDCOM) ---"
            )  # Use specific name in heading
            print(
                relationship_path_str.strip()
            )  # Print the result (includes profile info line from utils)

        except AttributeError as ae:
            logger.error(
                f"AttributeError calculating path - function missing?: {ae}",
                exc_info=True,
            )
            print("\nPath calculation error (function missing?). Check logs.")
        except Exception as e:
            logger.error(f"Error calculating relationship path: {e}", exc_info=True)
            print("\nCould not determine relationship path. Check logs.")

    # Inform user if WGG was not found, skipping path calculation
    elif not wgg_id:
        print(
            f"\n(Skipping relationship calculation as '{reference_person_name}' was not found in GEDCOM)"
        )  # Use specific name

    return True  # Indicate successful report completion


# End of handle_gedcom_report


# --- Main Execution ---
def main():
    """Main execution flow for Action 10 (GEDCOM Report)."""
    logger.info("--- Action 10: GEDCOM Report Starting ---")

    # Check if required libraries/utils are available
    if (
        not GEDCOM_LIB_AVAILABLE
        or GedcomReader is None
        or find_potential_matches is None
    ):
        logger.critical(
            "Required GEDCOM library (ged4py) or core gedcom_utils functions are unavailable."
        )
        print("\nERROR: Required libraries/utils not loaded. Cannot run Action 10.")
        sys.exit(1)

    # Check if scoring configuration is available
    if (
        COMMON_SCORING_WEIGHTS is None
        or NAME_FLEXIBILITY is None
        or DATE_FLEXIBILITY is None
    ):
        logger.critical(
            "Scoring configuration variables are None after attempting load."
        )
        print("\nERROR: Scoring configuration load failed. Cannot run Action 10.")
        sys.exit(1)

    reader = None
    wgg_indi = None  # Store WGG Individual object found in GEDCOM
    wgg_id = None  # Store WGG normalized ID found in GEDCOM

    try:
        # --- Phase 1: Load GEDCOM File ---
        gedcom_path_str = getattr(config_instance, "GEDCOM_FILE_PATH", None)
        if not gedcom_path_str:
            raise ValueError(
                "GEDCOM_FILE_PATH not set in config. Please update your .env file."
            )
        gedcom_path = Path(gedcom_path_str)
        if not gedcom_path.is_file():
            raise FileNotFoundError(
                f"GEDCOM file not found: {gedcom_path}. Please check GEDCOM_FILE_PATH in your .env file."
            )

        logger.info(f"Loading GEDCOM: {gedcom_path}...")
        print(f"Loading GEDCOM: {gedcom_path.name}...")  # User feedback
        start_time = time.time()
        reader = GedcomReader(str(gedcom_path))
        load_time = time.time() - start_time
        logger.info(f"GEDCOM loaded successfully in {load_time:.2f} seconds.")
        print("GEDCOM loaded.")

        # --- Phase 2: Build Caches and Find Reference Person (WGG) ---
        # Build index and family maps using gedcom_utils functions
        logger.info("Building individual index...")
        print("Building index...")  # User feedback
        build_indi_index(reader)
        logger.info("Building family relationship maps...")
        print("Building relationship maps...")  # User feedback
        build_family_maps(reader)  # This call also caches internally

        # Search for the reference person "Wayne Gordon Gault" by name in the GEDCOM index
        logger.info("Searching for reference person 'Wayne Gordon Gault' in GEDCOM...")
        wgg_search_name_lower = "wayne gordon gault"

        # Check if INDI_INDEX was successfully built before searching
        if build_indi_index is not None and getattr(gedcom_utils, "INDI_INDEX", None):
            # Iterate through the built index
            for indi_id, indi_obj in gedcom_utils.INDI_INDEX.items():
                # Use the imported _get_full_name to get the formatted name
                if _get_full_name:
                    name_str = _get_full_name(indi_obj)
                    if name_str and name_str.lower() == wgg_search_name_lower:
                        wgg_indi = indi_obj  # Store the Individual object
                        wgg_id = indi_id  # Store the normalized ID (from index key)
                        logger.info(
                            f"Found WGG reference person: {name_str} [@{wgg_id}@] "
                        )
                        print(f"Found Reference Person: {name_str}")  # User feedback
                        break  # Stop after finding the first match

            if not wgg_id:
                logger.warning(
                    "Reference person 'Wayne Gordon Gault' not found in GEDCOM index."
                )
                print(
                    "Warning: Reference person 'Wayne Gordon Gault' not found in GEDCOM."
                )  # User feedback
        else:
            logger.error("INDI_INDEX is not available. Cannot search for WGG.")
            print(
                "Error: Individual index not built to find reference person."
            )  # User feedback

        # --- Phase 3: Run the Report Handler ---
        # Pass the reader, found WGG details, and config dictionaries to the handler
        handle_gedcom_report(
            reader,
            wgg_indi,
            wgg_id,
            COMMON_SCORING_WEIGHTS,
            NAME_FLEXIBILITY,
            DATE_FLEXIBILITY,
            max_results_display=5,  # Display top 5 matches by score (can be config option)
        )

    # --- Error Handling for Setup Phase ---
    except (ValueError, FileNotFoundError) as setup_err:
        logger.critical(
            f"Fatal Setup Error: {setup_err}", exc_info=False
        )  # No traceback for known file/value errors
        print(f"\nERROR: Setup failed - {setup_err}")
        sys.exit(1)  # Exit on fatal setup error
    except ImportError as import_err:
        # This block should ideally not be hit if the initial check passes, but kept as fallback
        logger.critical(f"Fatal Import Error: {import_err}", exc_info=True)
        print(f"\nERROR: Import failed - {import_err}. Check library installations.")
        sys.exit(1)  # Exit on fatal import error
    except AttributeError as attr_err:
        logger.critical(
            f"Fatal AttributeError - potentially missing function/variable after import attempts: {attr_err}",
            exc_info=True,
        )
        print(
            f"\nERROR: Missing required component: {attr_err}. Check logs and library installations."
        )
        sys.exit(1)  # Exit on fatal attribute error
    except Exception as e:
        logger.critical(f"Unexpected error in main execution: {e}", exc_info=True)
        print(f"\nUnexpected error: {e}. Check logs for details.")
        sys.exit(1)  # Exit on any other unexpected critical error

    finally:
        # Cleanup or final messages
        logger.info("--- Action 10: GEDCOM Report Finished ---")
        print("\nAction 10 finished.")


# End of main

# Script entry point check
if __name__ == "__main__":
    # Initial check for ged4py library availability based on the flag set by gedcom_utils import
    if GEDCOM_LIB_AVAILABLE:
        # If library is available, proceed to main which handles other checks (config, specific utils)
        main()
    else:
        # If GEDCOM_LIB_AVAILABLE is False (due to ged4py import error), exit immediately
        print(
            "\nCRITICAL ERROR: Required GEDCOM library (ged4py) is not installed or failed to load."
        )
        print("Please install it using: pip install ged4py")
        logging.getLogger().critical("Exiting: Required GEDCOM library not loaded.")
        sys.exit(1)



# End of action10.py