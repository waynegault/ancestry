# action10.py
"""
Standalone script to perform Action 1: GEDCOM Report.
Loads a local GEDCOM file, prompts the user for search criteria,
finds potential matches using fuzzy search, allows selection,
and displays the selected person's details, family, and relationship
to a predefined reference person (Wayne Gordon Gault) found in the GEDCOM.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional
import time

# Add parent directory to sys.path to import utils, config, etc.
# Adjust depth as needed based on project structure.
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Import Local Modules ---
try:
    from logging_config import setup_logging
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    def setup_logging(log_file="action10.log", log_level="INFO"):
        logger = logging.getLogger("action10_fallback")
        return logger


try:
    from config import config_instance
except ImportError:
    logging.error("Failed to import config_instance. Cannot load GEDCOM file path.")

    class DummyConfig:
        GEDCOM_FILE_PATH = None

    config_instance = DummyConfig()

# Setup logging for this action script
logger = setup_logging(
    log_file="gedcom_processor.log", log_level="INFO"
)  # Use shared log file

# --- Import GEDCOM Utilities ---
try:
    from gedcom_utils import (
        GedcomReader,
        Individual,
        Record,
        Name,  # ged4py objects
        GEDCOM_LIB_AVAILABLE,
        build_indi_index,
        find_individual_by_id,
        _normalize_id,
        _get_full_name,
        find_potential_matches,
        extract_and_fix_id,
        get_relationship_path,
        format_full_life_details,
        format_relative_info,
        get_related_individuals,
        logger as gedcom_logger,  # Use the logger from utils if needed, or stick to current logger
    )

    # If using gedcom_utils logger, could alias: logger = gedcom_logger
except ImportError as e:
    logger.critical(
        f"Failed to import from gedcom_utils: {e}. Cannot run GEDCOM report.",
        exc_info=True,
    )
    # Set flag to prevent execution
    GEDCOM_LIB_AVAILABLE = False  # Ensure this reflects reality


# --- GEDCOM Display Function ---


def display_gedcom_family_details(reader, individual):
    """Helper function to display formatted family details from GEDCOM data."""
    if not reader or not _is_individual(individual):  # Use internal helper if needed
        logger.error("  Error: Cannot display GEDCOM details for invalid input.")
        return

    indi_name = _get_full_name(individual)
    logger.info(f"\n--- Individual Details (GEDCOM) ---")
    logger.info(f"Name: {indi_name}")
    birth_info, death_info = format_full_life_details(individual)
    logger.info(birth_info)
    if death_info:
        logger.info(death_info)

    # Display Parents
    logger.info("\n Parents:")
    parents = get_related_individuals(reader, individual, "parents")
    if parents:
        [logger.info(format_relative_info(p)) for p in parents]
    else:
        logger.info("  (None found)")

    # Display Siblings
    logger.info("\n Siblings:")
    siblings = get_related_individuals(reader, individual, "siblings")
    if siblings:
        [logger.info(format_relative_info(s)) for s in siblings]
    else:
        logger.info("  (None found)")

    # Display Spouses
    logger.info("\n Spouse(s):")
    spouses = get_related_individuals(reader, individual, "spouses")
    if spouses:
        [logger.info(format_relative_info(s)) for s in spouses]
    else:
        logger.info("  (None found)")

    # Display Children
    logger.info("\n Children:")
    children = get_related_individuals(reader, individual, "children")
    if children:
        [logger.info(format_relative_info(c)) for c in children]
    else:
        logger.info("  (None found)")


# Helper for _is_individual (needed by display_gedcom_family_details)
def _is_individual(obj) -> bool:
    """Checks if object is an Individual safely handling None values"""
    return obj is not None and type(obj).__name__ == "Individual"


# --- Main GEDCOM Report Handler ---


def handle_gedcom_report(
    reader,
    wayne_gault_indi: Optional[Individual],  # Pass the object
    wayne_gault_id_gedcom: Optional[str],  # Pass the ID string
    max_results: int = 3,
):
    """Handler for GEDCOM Report functionality."""
    logger.info("\n--- GEDCOM Report ---")

    # Check if reference person was found during initialization
    if not wayne_gault_indi or not wayne_gault_id_gedcom:
        logger.error("Reference person (Wayne Gordon Gault) not found in local GEDCOM.")
        logger.warning("Relationship path calculation will not be available.")
        # Allow proceeding, but relationship part will fail or be skipped

    # Prompt for search criteria
    logger.info("\nEnter search criteria for the person of interest:")
    first_name = input(" First Name (optional): ").strip() or None
    surname = input(" Surname (optional): ").strip() or None
    dob_str = input(" Birth Date/Year (optional): ").strip() or None
    pob = input(" Birth Place (optional): ").strip() or None
    dod_str = input(" Death Date/Year (optional): ").strip() or None
    pod = input(" Death Place (optional): ").strip() or None
    gender = input(" Gender (M/F, optional): ").strip() or None
    if gender:
        gender = gender[0].lower() if gender[0].lower() in ["m", "f"] else None

    if not any([first_name, surname, dob_str, pob, dod_str, pod, gender]):
        logger.info("\nNo search criteria entered. Report cancelled.")
        return

    # Find potential matches using fuzzy search from gedcom_utils
    logger.info("Searching GEDCOM...")
    matches = find_potential_matches(
        reader, first_name, surname, dob_str, pob, dod_str, pod, gender, max_results
    )

    if not matches:
        logger.info("\nNo potential matches found in GEDCOM based on criteria.")
        return

    # Select Match
    selected_match = None
    if len(matches) == 1:
        selected_match = matches[0]
        logger.info(f"Auto-selected only match: {selected_match['name']}")
    else:
        logger.info("\nPotential Matches Found:")
        for i, match in enumerate(matches):  # Display all found up to max_results
            b_info = f"b. {match['birth_date']}" if match["birth_date"] != "N/A" else ""
            d_info = f"d. {match['death_date']}" if match["death_date"] != "N/A" else ""
            date_info = (
                f" ({', '.join(filter(None, [b_info, d_info]))})"
                if b_info or d_info
                else ""
            )
            logger.info(
                f"  {i+1}. {match['name']}{date_info} (Score: {match['score']}, {match['reasons']})"
            )

        try:
            choice = int(
                input(f"\nSelect person by number (1-{len(matches)}), or 0 to cancel: ")
            )
            if 0 < choice <= len(matches):
                selected_match = matches[choice - 1]
            else:
                logger.info("Selection cancelled or invalid.")
                return
        except ValueError:
            logger.error("Invalid selection. Please enter a number.")
            return
        except Exception as e:
            logger.error(f"Error during selection: {e}", exc_info=True)
            return

    # Process Selected Match
    if selected_match:
        selected_id_raw = selected_match["id"]
        selected_id_norm = extract_and_fix_id(selected_id_raw)

        if not selected_id_norm:
            logger.error(
                f"Invalid ID extracted from selected match: '{selected_id_raw}'"
            )
            return

        selected_indi = find_individual_by_id(reader, selected_id_norm)

        if not selected_indi:
            logger.error(
                f"Could not retrieve individual record for ID {selected_id_norm} from GEDCOM."
            )
            return

        # Display Details and Family
        display_gedcom_family_details(reader, selected_indi)

        # Display Relationship Path (if reference person is available)
        if wayne_gault_id_gedcom and selected_id_norm:
            logger.info("\nCalculating relationship to Wayne Gordon Gault...")
            try:
                relationship_path = get_relationship_path(
                    reader, selected_id_norm, wayne_gault_id_gedcom
                )
                logger.info(f"\n--- Relationship Path (GEDCOM) ---")
                logger.info(relationship_path)
            except Exception as e:
                logger.error(f"Error calculating relationship path: {e}", exc_info=True)
                logger.info("\nCould not determine relationship path.")
        elif not wayne_gault_id_gedcom:
            logger.info(
                "\n(Skipping relationship calculation as reference person was not found in GEDCOM)"
            )
        else:
            logger.info(
                "\n(Skipping relationship calculation due to missing selected ID)"
            )


# --- Main Execution ---


def main():
    """Main execution flow for Action 10 (GEDCOM Report)."""
    logger.info("--- Action 10: GEDCOM Report Starting ---")

    if not GEDCOM_LIB_AVAILABLE or GedcomReader is None:
        logger.critical(
            "ged4py library is unavailable or failed to import. Cannot proceed."
        )
        sys.exit(1)

    reader = None
    wayne_gault_indi = None
    wayne_gault_id_gedcom = None

    try:
        # --- Load GEDCOM ---
        gedcom_path_str = getattr(config_instance, "GEDCOM_FILE_PATH", None)
        if not gedcom_path_str:
            raise ValueError("GEDCOM_FILE_PATH not set in config.")
        gedcom_path = Path(gedcom_path_str)
        if not gedcom_path.is_file():
            raise FileNotFoundError(f"GEDCOM file not found: {gedcom_path}")

        logger.info(f"Loading GEDCOM: {gedcom_path}...")
        start_time = time.time()
        reader = GedcomReader(str(gedcom_path))
        load_time = time.time() - start_time
        logger.info(f"GEDCOM loaded successfully in {load_time:.2f} seconds.")

        # --- Pre-build Cache and Find Reference Person ---
        logger.info("Building individual index...")
        build_indi_index(reader)  # Builds the INDI_INDEX in gedcom_utils

        logger.info(
            "Pre-searching for reference person 'Wayne Gordon Gault' in GEDCOM..."
        )
        wgg_search_name_lower = "wayne gordon gault"
        # Search using the pre-built index (INDI_INDEX is in gedcom_utils)
        # We need access to the index here, or a function to search it.
        # Let's modify find_individual_by_id slightly or add a name search utility.
        # For now, iterate through the index values directly:
        if "INDI_INDEX" in sys.modules["gedcom_utils"].__dict__:
            utils_indi_index = sys.modules["gedcom_utils"].INDI_INDEX
            for indi_id, indi_obj in utils_indi_index.items():
                name_str = _get_full_name(
                    indi_obj
                )  # Use helper from this file or import
                if name_str.lower() == wgg_search_name_lower:
                    wayne_gault_indi = indi_obj
                    wayne_gault_id_gedcom = _normalize_id(
                        indi_obj.xref_id
                    )  # Use helper
                    logger.info(
                        f"Found WGG in GEDCOM: {_get_full_name(wayne_gault_indi)} [@{wayne_gault_id_gedcom}@] "
                    )
                    break
            if not wayne_gault_id_gedcom:
                logger.warning(
                    "Reference person 'Wayne Gordon Gault' not found in GEDCOM index."
                )
        else:
            logger.error("Could not access INDI_INDEX from gedcom_utils.")

        # --- Execute the Report Handler ---
        handle_gedcom_report(reader, wayne_gault_indi, wayne_gault_id_gedcom)

    except (ValueError, FileNotFoundError, ImportError) as setup_err:
        logger.critical(f"Fatal Setup Error: {setup_err}", exc_info=True)
    except Exception as e:
        logger.critical(f"An unexpected error occurred in main: {e}", exc_info=True)
    finally:
        # No specific cleanup needed for GEDCOM reader here
        logger.info("--- Action 10: GEDCOM Report Finished ---")


if __name__ == "__main__":
    # Ensure necessary libraries are available before running
    if GEDCOM_LIB_AVAILABLE:
        main()
    else:
        logger.critical(
            "Exiting: Required GEDCOM library (ged4py) or gedcom_utils not available."
        )
        sys.exit(1)
