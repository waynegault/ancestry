# action11.py
"""
Standalone script to perform Action 2: API Report.
Initializes a session with Ancestry (potentially requiring login),
prompts the user for search criteria (name), uses Ancestry API
endpoints (/suggest, /person-card) via the SessionManager's _api_req method
to find and display the person's details and basic family information fetched from the API.
"""

import logging
import sys
import os
import urllib.parse
import random  # Used for User-Agent fallback potentially
import time
import json  # For potential JSON parsing if needed directly
import requests

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

    def setup_logging(log_file="action11.log", log_level="INFO"):
        logger = logging.getLogger("action11_fallback")
        return logger


try:
    from config import config_instance
except ImportError:
    logging.error("Failed to import config_instance. API functionality may be limited.")

    class DummyConfig:
        GEDCOM_FILE_PATH = None
        BASE_URL = "https://www.ancestry.com"  # Default fallback
        USER_AGENTS = ["Mozilla/5.0"]

    config_instance = DummyConfig()

# Setup logging for this action script
logger = setup_logging(
    log_file="gedcom_processor.log", log_level="INFO"
)  # Use shared log file


# --- Import API Utilities ---
# We need the session_manager instance and initialize function from api_utils
# We also need API_UTILS_AVAILABLE flag
try:
    from api_utils import (
        initialize_session,
        session_manager,  # Import the singleton instance
        API_UTILS_AVAILABLE,
        logger as api_logger,  # Use the logger from api_utils if needed
    )

    # NOTE: _api_req is NOT imported here; it's a method of session_manager
    # If using api_utils logger: logger = api_logger

except ImportError as e:
    logger.critical(
        f"Failed to import from api_utils: {e}. Cannot run API report.", exc_info=True
    )
    API_UTILS_AVAILABLE = False  # Ensure flag reflects reality

    # Define dummy session_manager if import failed completely
    class DummySessionManager:
        driver_live = False
        session_ready = False
        my_tree_id = None
        my_profile_id = None
        my_uuid = None
        driver = None

        # Add dummy _api_req to the dummy class
        def _api_req(self, *args, **kwargs):
            logger.error("API request attempted but api_utils failed to load.")
            return {"error": "'api_utils' module unavailable"}

        def quit_driver(self):
            pass

    session_manager = DummySessionManager()


# --- Main API Report Handler ---


def handle_api_report():
    """Handler for Action 11 - API Report using suggest and person-card."""
    logger.info("\n--- API Report ---")
    if not API_UTILS_AVAILABLE:
        logger.error(
            "API functionality is disabled because 'utils' module dependencies are missing or failed to load."
        )
        return

    # Initialize API Session (handles login, CSRF, etc.)
    if not initialize_session():  # This function now lives in api_utils
        logger.error(
            "Failed to initialize session. Cannot proceed with API operations."
        )
        return

    # We need tree_id, profile_id, uuid from the initialized session_manager
    my_tree_id = getattr(session_manager, "my_tree_id", None)
    my_profile_id = getattr(session_manager, "my_profile_id", None)
    my_uuid = getattr(session_manager, "my_uuid", None)

    if not my_tree_id:
        logger.error(
            "Tree ID not found in session. Cannot use person-picker/suggest API."
        )
        # Depending on API, might still be possible to proceed with other searches?
        # For this specific flow, tree_id is needed for /suggest.
        return
    if not my_uuid:
        logger.error(
            "User UUID not found in session. Cannot construct match list URL for context."
        )
        # This is needed for Referer/Origin context
        return

    # Prompt for search criteria (simple name search for /suggest)
    logger.info("\nEnter search criteria for the person of interest (API Search):")
    first_name = input(" First Name: ").strip() or ""
    surname = input(" Surname: ").strip() or ""
    if not (first_name or surname):
        logger.error(
            "API search requires at least First Name or Surname. Report cancelled."
        )
        return

    # --- Prepare Context for API calls ---
    # Context like Referer, Origin, CSRF, User-Agent is handled internally by
    # the session_manager._api_req method based on the current driver state.
    # We just need to provide the core request parameters (URL, method, data).
    base_url = getattr(config_instance, "BASE_URL", "https://www.ancestry.com").rstrip(
        "/"
    )
    referer = f"{base_url}/discoveryui-matches/list/{my_uuid}"  # Suggest API often needs context

    # --- Call /person-picker/suggest API ---
    suggest_url = (
        f"{base_url}/api/person-picker/suggest/{my_tree_id}?"
        f"partialFirstName={urllib.parse.quote(first_name)}&partialLastName={urllib.parse.quote(surname)}"
    )
    selected_person_api_info = None
    person_card_data = None

    try:
        logger.info(f"Calling Suggest API: {suggest_url}")
        # *** CHANGED: Call _api_req as a method of the session_manager instance ***
        # *** REMOVED: driver and session_manager arguments from the call ***
        suggest_response = session_manager._api_req(
            url=suggest_url,
            method="GET",
            # headers=common_headers, # Headers are mostly handled internally now
            api_description="Person Picker Suggest API",
            referer_url=referer,  # Provide specific referer if needed
            timeout=15,
            use_csrf_token=True,  # Suggest API might need CSRF
        )

        # Process suggest response
        if isinstance(suggest_response, dict) and "error" in suggest_response:
            logger.error(f"Suggest API failed: {suggest_response['error']}")
            return
        # Handle cases where _api_req returns the Response object on error
        if isinstance(suggest_response, requests.Response) and not suggest_response.ok:
            logger.error(
                f"Suggest API request failed with status {suggest_response.status_code}"
            )
            return
        if (
            suggest_response is None
        ):  # Handle None return from _api_req on total failure
            logger.error("Suggest API request failed (returned None).")
            return

        if not isinstance(suggest_response, list) or not suggest_response:
            logger.info("No matches found via Suggest API.")
            # Maybe try a different search API here as a fallback?
            return

        # --- Auto-select the first suggestion ---
        # TODO: Implement selection logic if multiple suggestions are relevant
        if len(suggest_response) > 1:
            logger.info(
                f"Multiple suggestions found ({len(suggest_response)}), auto-selecting the first."
            )
            # Add logic here to display options and let user choose if needed
            # for i, person in enumerate(suggest_response):
            #     logger.info(f"  {i+1}. {person.get('Name', 'N/A')} (ID: {person.get('PersonId', 'N/A')})")
            # ... input choice ...

        selected_person_api_info = suggest_response[0]
        logger.info(
            f"Selected person from API: {selected_person_api_info.get('Name', 'N/A')}"
        )

        # Extract details needed for the next API call (/person-card)
        person_id = selected_person_api_info.get("PersonId")
        # Tree ID might differ if the person is from a different tree? Use the one from the response.
        person_tree_id = selected_person_api_info.get(
            "TreeId", my_tree_id
        )  # Fallback to user's tree

        if not person_id or not person_tree_id:
            logger.error(
                "Could not extract necessary PersonId/TreeId from suggest response."
            )
            logger.debug(f"Suggest Response Item: {selected_person_api_info}")
            return

    except Exception as e:
        logger.error(
            f"Error during /person-picker/suggest API call or processing: {e}",
            exc_info=True,
        )
        return  # Stop if suggest fails

    # --- Call /person-card API ---
    if selected_person_api_info and person_id and person_tree_id:
        person_card_url = f"{base_url}/api/search-results/person-card/tree/{person_tree_id}/person/{person_id}"
        try:
            logger.info(f"Calling Person Card API: {person_card_url}")
            # *** CHANGED: Call _api_req as a method of the session_manager instance ***
            # *** REMOVED: driver and session_manager arguments from the call ***
            person_card_response = session_manager._api_req(
                url=person_card_url,
                method="GET",
                # headers=common_headers, # Headers handled internally
                api_description="Person Card API",
                referer_url=referer,  # Maintain context if needed
                timeout=15,
                use_csrf_token=True,  # Person card API might need CSRF
            )

            if (
                isinstance(person_card_response, dict)
                and "error" in person_card_response
            ):
                logger.error(f"Person Card API failed: {person_card_response['error']}")
                # Continue? Or stop? Data will be incomplete. Stop for now.
                return
            # Handle cases where _api_req returns the Response object on error
            if (
                isinstance(person_card_response, requests.Response)
                and not person_card_response.ok
            ):
                logger.error(
                    f"Person Card API request failed with status {person_card_response.status_code}"
                )
                return
            if person_card_response is None:  # Handle None return from _api_req
                logger.error("Person Card API request failed (returned None).")
                return

            if not isinstance(person_card_response, dict):
                logger.error(
                    "Person card API did not return a valid dictionary result."
                )
                logger.debug(f"Person Card Response Type: {type(person_card_response)}")
                logger.debug(
                    f"Person Card Response Content: {str(person_card_response)[:500]}..."
                )
                return

            person_card_data = person_card_response  # Store the valid data

        except Exception as e:
            logger.error(
                f"Error during /search-results/person-card API call or processing: {e}",
                exc_info=True,
            )
            # Continue without person card data? Or stop? Stop for now.
            return

    # --- Display Results from Person Card API ---
    if person_card_data:
        logger.info("\n--- Individual Details (API - Person Card) ---")
        logger.info(f" Name: {person_card_data.get('name', 'Unknown')}")
        # Check for birth/death date and place details
        birth_info = person_card_data.get("birth", "")
        death_info = person_card_data.get("death", "")
        logger.info(f"   Birth: {birth_info if birth_info else '(N/A)'}")
        logger.info(f"   Death: {death_info if death_info else '(N/A)'}")
        logger.info(f"   Person ID: {person_card_data.get('personId', '(N/A)')}")
        logger.info(
            f"   Tree ID: {person_card_data.get('treeId', '(N/A)')}"
        )  # Often included

        # Parents
        logger.info("\n Parents:")
        father = person_card_data.get("father")
        mother = person_card_data.get("mother")
        if father:
            logger.info(
                f"  - Father: {father.get('name', '')} {father.get('lifeSpan', '')}"
            )
        if mother:
            logger.info(
                f"  - Mother: {mother.get('name', '')} {mother.get('lifeSpan', '')}"
            )
        if not (father or mother):
            logger.info("  (None found in API data)")

        # Spouse(s) - Person card usually shows only one selected spouse
        logger.info("\n Spouse(s):")
        # Key might be 'spouse' or 'selectedSpouse', check response structure if needed
        spouse = person_card_data.get("selectedSpouse") or person_card_data.get(
            "spouse"
        )
        if spouse and isinstance(spouse, dict):  # Ensure spouse data is a dictionary
            logger.info(
                f"  - Spouse: {spouse.get('name', '')} {spouse.get('lifeSpan', '')}"
            )
            # Children with this spouse are often nested
            logger.info("\n Children (with this spouse):")
            children = spouse.get("children", [])
            if children:
                for child in children:
                    logger.info(
                        f"  - Child: {child.get('name', '')} {child.get('lifeSpan', '')}"
                    )
            else:
                logger.info("  (None found for this spouse in API data)")
        elif spouse:  # Handle cases where spouse might be something else unexpected
            logger.warning(
                f"Spouse data found but not in expected dictionary format: {spouse}"
            )
            logger.info("  (Could not parse spouse details)")
        else:
            logger.info("  (None found in API data)")

        # Note: Person card API might not show siblings directly.

    else:
        logger.info("\nNo detailed person card data could be retrieved from the API.")


# --- Main Execution ---


def main():
    """Main execution flow for Action 11 (API Report)."""
    logger.info("--- Action 11: API Report Starting ---")

    if not API_UTILS_AVAILABLE:
        logger.critical(
            "API utilities (from 'utils' module) are unavailable. Cannot proceed."
        )
        sys.exit(1)

    try:
        # Execute the main API report handler
        handle_api_report()

    except KeyboardInterrupt:
        logger.warning("\nOperation interrupted by user.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in main: {e}", exc_info=True)
    finally:
        # Attempt to close Selenium driver if it was opened by session_manager
        if session_manager and getattr(session_manager, "driver_live", False):
            logger.info("Attempting to clean up browser session...")
            try:
                # Use a quit method if available in the session manager implementation
                if hasattr(session_manager, "close_sess") and callable(
                    session_manager.close_sess
                ):
                    # close_sess handles driver.quit() internally
                    session_manager.close_sess(
                        keep_db=True
                    )  # Keep DB pool if other actions might run
                    logger.info(
                        "Browser session closed via session_manager.close_sess()."
                    )
                elif (
                    hasattr(session_manager, "driver")
                    and session_manager.driver
                    and hasattr(session_manager.driver, "quit")
                ):
                    session_manager.driver.quit()  # Fallback direct quit
                    logger.info("Browser session closed directly.")
                else:
                    logger.info("No active driver or quit method found to close.")
            except Exception as close_err:
                logger.error(
                    f"Error closing browser session: {close_err}", exc_info=False
                )

        logger.info("--- Action 11: API Report Finished ---")


if __name__ == "__main__":
    # Check dependencies before running main
    if API_UTILS_AVAILABLE:
        main()
    else:
        logger.critical(
            "Exiting: Required API utilities (utils module / api_utils) not available."
        )
        sys.exit(1)
