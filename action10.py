# --- START OF FILE action10.py ---

# action10.py
"""
Action 10: Find GEDCOM Matches and Relationship Path

Applies a hardcoded filter (OR logic) to the GEDCOM data (using pre-processed
cache), calculates a score for each filtered individual based on specific criteria,
displays the top 3 highest-scoring individuals (simplified format), identifies the
best match, and displays their relatives and relationship path to the reference person.
V.20240502.FinalRefined.SyntaxFix4
"""

# --- Standard library imports ---
import logging
import sys
import time
import difflib
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

# --- Setup Fallback Logger FIRST ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("action10_initial")

# --- Local application imports ---
try:
    from config import config_instance
    from logging_config import setup_logging

    log_filename = "action10.log"
    if hasattr(config_instance, "LOG_DIR") and config_instance.LOG_DIR:
        log_filename = config_instance.LOG_DIR / "action10.log"
    else:
        log_filename = Path(__file__).parent / "action10.log"

    logger = setup_logging(log_file=str(log_filename), log_level="INFO")
    logger.info("Logging configured via setup_logging (Level: INFO).")

except ImportError as e:
    logger.critical(f"Failed to import configuration or logging setup: {e}")
    logger.warning("Continuing with basic logging configuration.")
    config_instance = None
except Exception as e:
    logger.critical(f"Unexpected error during logging setup: {e}", exc_info=True)
    logger.warning("Continuing with basic logging configuration.")
    config_instance = None

# Import GEDCOM utilities
try:
    from gedcom_utils import (
        GedcomData,
        _get_full_name,
        format_relative_info,
        calculate_match_score,
        _normalize_id,
    )
except ImportError as e:
    logger.critical(f"Failed to import gedcom_utils: {e}. Script cannot run.")
    sys.exit(1)
except Exception as e:
    logger.critical(f"Unexpected error importing gedcom_utils: {e}", exc_info=True)
    sys.exit(1)


# --- Constants ---
MAX_DISPLAY_RESULTS = 3  # Show only Top 3


# --- Core Functions ---
def main() -> None:
    """Main function to drive the action."""
    logger.info("--- Starting Action 10: User Input -> Filter -> Score -> Analyze ---")

    # 1. Configuration Validation
    if not config_instance:
        logger.critical("Configuration (config_instance) not loaded. Cannot proceed.")
        sys.exit(1)

    logger.info(f"Configured TREE_OWNER_NAME: {getattr(config_instance, 'TREE_OWNER_NAME', 'Not Set')}")
    logger.info(f"Configured REFERENCE_PERSON_ID: {getattr(config_instance, 'REFERENCE_PERSON_ID', 'Not Set')}")
    logger.info(f"Configured REFERENCE_PERSON_NAME: {getattr(config_instance, 'REFERENCE_PERSON_NAME', 'Not Set')}")

    gedcom_file_path_config: Optional[Path] = getattr(config_instance, "GEDCOM_FILE_PATH", None)
    reference_person_id_raw: Optional[str] = getattr(config_instance, "REFERENCE_PERSON_ID", None)
    reference_person_name: Optional[str] = getattr(config_instance, "REFERENCE_PERSON_NAME", "Reference Person")

    if not gedcom_file_path_config or not gedcom_file_path_config.is_file():
        logger.critical(f"GEDCOM file path missing or invalid: {gedcom_file_path_config}")
        sys.exit(1)

    gedcom_path: Path = gedcom_file_path_config
    logger.info(f"Using GEDCOM file: {gedcom_path.name}")


    # 2. Load GEDCOM Data
    gedcom_data: Optional[GedcomData] = None
    try:
        logger.info("Loading, parsing, and pre-processing GEDCOM data...")
        load_start_time = time.time()
        gedcom_data = GedcomData(gedcom_path)
        load_end_time = time.time()
        logger.info(f"GEDCOM data loaded & processed successfully in {load_end_time - load_start_time:.2f}s.")
        logger.info(f"  Index size: {len(getattr(gedcom_data, 'indi_index', {}))}")
        logger.info(f"  Pre-processed cache size: {len(getattr(gedcom_data, 'processed_data_cache', {}))}")
        logger.info(f"  Build Times: Index={gedcom_data.indi_index_build_time:.2f}s, Maps={gedcom_data.family_maps_build_time:.2f}s, PreProcess={gedcom_data.data_processing_time:.2f}s")
    except Exception as e:
        logger.critical(f"Failed to load or process GEDCOM file {gedcom_path.name}: {e}", exc_info=True)
        sys.exit(1)

    if gedcom_data is None or not gedcom_data.processed_data_cache or not gedcom_data.indi_index:
        logger.critical("GEDCOM data object/cache/index is None or empty after loading attempt.")
        sys.exit(1)

    # --- Step 3: Get User Input for Criteria ---
    logger.info("\n--- Enter Search Criteria (Press Enter to skip optional fields) ---")

    input_fname = input("  First Name Contains: ").strip()
    input_sname = input("  Surname Contains: ").strip()
    input_gender = input("  Gender (M/F): ").strip().lower()
    input_byear_str = input("  Birth Year (YYYY): ").strip()
    input_bplace = input("  Birth Place Contains: ").strip()
    input_dyear_str = input("  Death Year (YYYY) [Optional]: ").strip()
    input_dplace = input("  Death Place Contains [Optional]: ").strip()

    # Process inputs
    first_name_crit = input_fname if input_fname else None
    surname_crit = input_sname if input_sname else None
    gender_crit = input_gender[0] if input_gender and input_gender[0] in ['m', 'f'] else None
    birth_place_crit = input_bplace if input_bplace else None
    death_place_crit = input_dplace if input_dplace else None

    birth_year_crit: Optional[int] = None
    if input_byear_str.isdigit():
        try:
            birth_year_crit = int(input_byear_str)
        except ValueError:
            logger.warning(f"Invalid birth year input '{input_byear_str}', ignoring.")
    elif input_byear_str:
         logger.warning(f"Non-numeric birth year input '{input_byear_str}', ignoring.")

    death_year_crit: Optional[int] = None
    if input_dyear_str.isdigit():
        try:
            death_year_crit = int(input_dyear_str)
        except ValueError:
            logger.warning(f"Invalid death year input '{input_dyear_str}', ignoring.")
    elif input_dyear_str:
        logger.warning(f"Non-numeric death year input '{input_dyear_str}', ignoring.")

    # Create date objects based *only* on year for now (simplifies input)
    # We need the objects for the scoring function signature, even if based only on year
    birth_date_obj_crit: Optional[datetime] = None
    if birth_year_crit:
        try:
             # Use Jan 1st for the object if only year is known
             birth_date_obj_crit = datetime(birth_year_crit, 1, 1, tzinfo=timezone.utc)
        except ValueError: # Handle invalid year like 0
             logger.warning(f"Cannot create date object for birth year {birth_year_crit}.")
             birth_year_crit = None # Nullify year if object creation fails

    death_date_obj_crit: Optional[datetime] = None
    if death_year_crit:
        try:
             death_date_obj_crit = datetime(death_year_crit, 1, 1, tzinfo=timezone.utc)
        except ValueError:
             logger.warning(f"Cannot create date object for death year {death_year_crit}.")
             death_year_crit = None

    # Build the criteria dictionaries
    scoring_criteria = {
        "first_name": first_name_crit,
        "surname": surname_crit,
        "gender": gender_crit,
        "birth_year": birth_year_crit,
        "birth_place": birth_place_crit,
        "birth_date_obj": birth_date_obj_crit,
        "death_year": death_year_crit,
        "death_place": death_place_crit,
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
    date_flex = getattr(config_instance, "DATE_FLEXIBILITY", {}); YEAR_RANGE_FILTER = date_flex.get("year_match_range", 10)

    logger.info(f"\n--- Final Scoring Criteria Used ---"); [logger.info(f"  {k.replace('_',' ').title()}: '{v}'") for k, v in scoring_criteria.items() if v is not None and k not in ["birth_date_obj", "death_date_obj"]]
    logger.info(f"\n--- OR Filter Logic (Year Range: +/- {YEAR_RANGE_FILTER}) ---"); logger.info(f"  Individuals will be scored if ANY filter criteria met or if alive.")


    # --- Step 4: Filter and Score Individuals ---
    logger.info("\n--- Filtering and Scoring Individuals (using pre-processed data) ---")
    processing_start_time = time.time(); scored_matches: List[Dict[str, Any]] = []; scoring_weights = getattr(config_instance, "COMMON_SCORING_WEIGHTS", {})
    if not scoring_weights: logger.error("Scoring weights missing."); sys.exit(1)
    logger.info(f"Processing {len(gedcom_data.processed_data_cache)} individuals from cache...")

    for indi_id_norm, indi_data in gedcom_data.processed_data_cache.items():
        try:
            # Extract needed values for filtering
            givn_lower = indi_data.get("first_name", "").lower(); surn_lower = indi_data.get("surname", "").lower()
            sex_lower = indi_data.get("gender_norm"); birth_year = indi_data.get("birth_year")
            birth_place_lower = indi_data.get("birth_place_disp", "").lower() if indi_data.get("birth_place_disp") else None
            death_date_obj = indi_data.get("death_date_obj")

            # Evaluate OR Filter
            fc = filter_criteria
            fn_match_filter = bool(fc.get('first_name') and givn_lower and fc['first_name'] in givn_lower)
            sn_match_filter = bool(fc.get('surname') and surn_lower and fc['surname'] in surn_lower)
            gender_match_filter = bool(fc.get('gender') and sex_lower and fc['gender'] == sex_lower)
            bp_match_filter = bool(fc.get('birth_place') and birth_place_lower and fc['birth_place'] in birth_place_lower)
            by_match_filter = bool(fc.get('birth_year') and birth_year and abs(birth_year - fc['birth_year']) <= YEAR_RANGE_FILTER)
            alive_match = (death_date_obj is None)
            passes_or_filter = ( fn_match_filter or sn_match_filter or gender_match_filter or bp_match_filter or by_match_filter or alive_match )

            if passes_or_filter:
                total_score, _, _ = calculate_match_score(
                    search_criteria=scoring_criteria, candidate_processed_data=indi_data,
                    scoring_weights=scoring_weights, date_flexibility=date_flex,
                )
                # Store results needed for display and analysis
                match_data = {
                    "id": indi_id_norm, "display_id": indi_data.get("display_id", indi_id_norm),
                    "full_name_disp": indi_data.get("full_name_disp", "N/A"), "total_score": total_score,
                    "gender": indi_data.get("gender_raw", "N/A"), "birth_date": indi_data.get("birth_date_disp", "N/A"),
                    "birth_place": indi_data.get("birth_place_disp"), "death_date": indi_data.get("death_date_disp"),
                    "death_place": indi_data.get("death_place_disp"),
                }
                scored_matches.append(match_data)
        except Exception as loop_err: logger.error(f"Error processing cached data for individual {indi_id_norm}: {loop_err}", exc_info=True)

    processing_duration = time.time() - processing_start_time; logger.info(f"Filtering & Scoring completed in {processing_duration:.2f}s.")
    logger.info(f"Found {len(scored_matches)} individual(s) matching OR criteria and scored.")

    # --- Step 5: Sort and Display Top Results (Simplified) ---
    scored_matches.sort(key=lambda x: x["total_score"], reverse=True)
    logger.info(f"\n--- Top {MAX_DISPLAY_RESULTS} Highest Scoring Matches ---")

    if not scored_matches:
        logger.info("No individuals matched the filter criteria or scored > 0.")
        logger.info("\n--- No matches found to analyze. ---")
    else:
        display_matches_final = scored_matches[:MAX_DISPLAY_RESULTS]
        logger.info(f"Displaying top {len(display_matches_final)} of {len(scored_matches)} scored matches:")
        # Column Width Calculation
        id_w = max((len(m.get("display_id", "")) for m in display_matches_final), default=15); id_w = max(id_w, 15)
        name_w = max((len(m.get("full_name_disp", "")) for m in display_matches_final), default=30); name_w = max(name_w, 30)
        gender_w = 6; bdate_w = 18; ddate_w = 18; total_score_w = 11
        bplace_w = max((len(m.get("birth_place", "") or "") for m in display_matches_final), default=30); bplace_w = max(bplace_w, 30)
        dplace_w = max((len(m.get("death_place", "") or "") for m in display_matches_final), default=30); dplace_w = max(dplace_w, 30)
        # Simplified Header
        header = (f"{'ID':<{id_w}} | {'Name':<{name_w}} | {'Sex':<{gender_w}} | {'Birth Date':<{bdate_w}} | {'Birth Place':<{bplace_w}} | {'Death Date':<{ddate_w}} | {'Death Place':<{dplace_w}} | {'Total Score':<{total_score_w}}"); logger.info(header); logger.info("-" * len(header))
        # Simplified Row Printing
        for match in display_matches_final:
            name_disp = match.get("full_name_disp", "N/A");
            if len(name_disp) > name_w: name_disp = name_disp[:name_w-3] + "..."
            bplace_disp = match.get("birth_place") or "N/A";
            if len(bplace_disp) > bplace_w: bplace_disp = bplace_disp[:bplace_w-3] + "..."
            dplace_disp = match.get("death_place") or "N/A";
            if len(dplace_disp) > dplace_w: dplace_disp = dplace_disp[:dplace_w-3] + "..."
            total_score_val = match.get('total_score', 'N/A')
            total_score_disp = f"{total_score_val:<{total_score_w}.0f}" if isinstance(total_score_val, (int, float)) else f"{'N/A':<{total_score_w}}"
            row = (f"{match.get('display_id', 'N/A'):<{id_w}} | {name_disp:<{name_w}} | {match.get('gender', 'N/A'):<{gender_w}} | {match.get('birth_date', 'N/A'):<{bdate_w}} | {bplace_disp:<{bplace_w}} | {match.get('death_date', 'N/A'):<{ddate_w}} | {dplace_disp:<{dplace_w}} | {total_score_disp}"); logger.info(row)
        if len(scored_matches) > len(display_matches_final): logger.info(f"... and {len(scored_matches) - len(display_matches_final)} more matches not shown.")

        # --- Step 6: Analyze Top Match ---
        logger.info("\n--- Analysis of Top Match ---")
        top_match = scored_matches[0]
        top_match_norm_id = top_match.get("id"); top_match_display_id = top_match.get("display_id", top_match_norm_id)
        top_match_indi = gedcom_data.find_individual_by_id(top_match_norm_id)

        if top_match_indi:
            logger.info(f"Best Match: {top_match.get('full_name_disp', 'N/A')} (ID: {top_match_display_id}, Score: {top_match['total_score']:.0f})")
            logger.info("\n  Relatives:")
            parents = gedcom_data.get_related_individuals(top_match_indi, "parents"); siblings = gedcom_data.get_related_individuals(top_match_indi, "siblings"); spouses = gedcom_data.get_related_individuals(top_match_indi, "spouses"); children = gedcom_data.get_related_individuals(top_match_indi, "children")
            if parents: logger.info("    Parents:") ; [logger.info(format_relative_info(p)) for p in parents]
            else: logger.info("    Parents: None found.")
            if siblings: logger.info("    Siblings:") ; [logger.info(format_relative_info(s)) for s in siblings]
            else: logger.info("    Siblings: None found.")
            if spouses: logger.info("    Spouses:") ; [logger.info(format_relative_info(s)) for s in spouses]
            else: logger.info("    Spouses: None found.")
            if children: logger.info("    Children:") ; [logger.info(format_relative_info(c)) for c in children]
            else: logger.info("    Children: None found.")

            # Re-check normalized reference ID before path calculation
            reference_person_id_norm = _normalize_id(reference_person_id_raw) if reference_person_id_raw else None # Normalize ID from config
            if not reference_person_id_norm: logger.warning("REFERENCE_PERSON_ID not configured or could not be normalized. Cannot calculate relationship path.")
            elif top_match_norm_id == reference_person_id_norm: logger.info(f"\n  Relationship Path: Top match is the reference person ({reference_person_name}).")
            else:
                logger.info(f"\n  Relationship Path to {reference_person_name} ({reference_person_id_norm}):")
                relationship_explanation = gedcom_data.get_relationship_path(top_match_norm_id, reference_person_id_norm)
                for line in relationship_explanation.splitlines(): logger.info(f"    {line}")
        else: logger.error(f"Could not retrieve Individual record for top match ID: {top_match_norm_id}")

    logger.info("\n--- Action 10 Finished ---")

if __name__ == "__main__":
    main()
# End of action10.py
