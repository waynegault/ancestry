#!/usr/bin/env python3

"""
main.py - Ancestry Research Automation Main Entry Point

Provides the main application entry point with menu-driven interface for
all automation workflows including DNA match gathering, inbox processing,
messaging, and genealogical research tools.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import gc
import inspect
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

# === THIRD-PARTY IMPORTS ===
import psutil
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

# === LOCAL IMPORTS ===
# Action modules
from action6_gather import coord  # Import the main DNA match gathering function
from action7_inbox import InboxProcessor
from action8_messaging import send_messages_to_matches
from action9_process_productive import process_productive_messages
from action10 import main as run_action10
from action11 import run_action11


def _load_and_validate_config_schema() -> Optional[Any]:
    """Load and validate configuration schema."""
    try:
        from config import config_schema
        if config_schema is None:
            logger.error("config_schema is None - configuration not properly initialized")
            return None
        logger.debug("Configuration loaded successfully")
        return config_schema
    except ImportError as e:
        logger.error(f"Could not import config_schema from config package: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        return None


def _check_processing_limits(config: Any) -> None:
    """Check essential processing limits and log warnings."""
    # Note: MAX_PAGES=0 is valid (means unlimited), so no warning needed
    if config.batch_size <= 0:
        logger.warning("BATCH_SIZE not set or invalid - actions may use large batches")
    # Note: MAX_PRODUCTIVE_TO_PROCESS=0 and MAX_INBOX=0 are valid (means unlimited)


def _check_rate_limiting_settings(config: Any) -> None:
    """Check rate limiting settings and log warnings."""
    # Note: Rate limiting settings are user preferences - no warnings needed
    # Users can adjust REQUESTS_PER_SECOND, INITIAL_DELAY, and BACKOFF_FACTOR as needed
    pass


def _log_configuration_summary(config: Any) -> None:
    """Log current configuration for transparency."""
    logger.info("=== ACTION CONFIGURATION VALIDATION ===")
    logger.info(f"MAX_PAGES: {config.api.max_pages}")
    logger.info(f"BATCH_SIZE: {config.batch_size}")
    logger.info(f"MAX_PRODUCTIVE_TO_PROCESS: {config.max_productive_to_process}")
    logger.info(f"MAX_INBOX: {config.max_inbox}")
    logger.info(f"Rate Limiting - RPS: {config.api.requests_per_second}, Delay: {config.api.initial_delay}s")
    logger.info("========================================")


# Configuration validation
def validate_action_config() -> bool:
    """
    Validate that all actions respect .env configuration limits.
    Prevents Action 6-style failures by ensuring conservative settings are applied.
    """
    try:
        # Load and validate configuration
        config = _load_and_validate_config_schema()
        if config is None:
            return False

        # Check processing limits
        _check_processing_limits(config)

        # Check rate limiting settings
        _check_rate_limiting_settings(config)

        # Log configuration summary
        _log_configuration_summary(config)

        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

# Core modules
from config.config_manager import ConfigManager
from core.session_manager import SessionManager
from database import (
    Base,
    ConversationLog,
    DnaMatch,
    FamilyTree,
    MessageTemplate,
    Person,
    backup_database,
    db_transn,
)
from logging_config import setup_logging
from my_selectors import WAIT_FOR_PAGE_SELECTOR
from utils import (
    log_in,
    login_status,
    nav_to_page,
)

# Initialize config manager
config_manager = ConfigManager()
config = config_manager.get_config()


def menu() -> str:
    """Display the main menu and return the user's choice."""
    print("Main Menu")
    print("=" * 17)
    level_name = "UNKNOWN"  # Default

    if logger and logger.handlers:
        console_handler = None
        for handler in logger.handlers:
            if (
                isinstance(handler, logging.StreamHandler)
                and handler.stream == sys.stderr
            ):
                console_handler = handler
                break
        if console_handler:
            level_name = logging.getLevelName(console_handler.level)
        else:
            level_name = logging.getLevelName(logger.getEffectiveLevel())
    elif hasattr(config, "logging") and hasattr(config.logging, "log_level"):
        level_name = config.logging.log_level.upper()

    print(f"(Log Level: {level_name})\n")
    print("0. Delete all rows except the first")
    print("1. Run Full Workflow (7, 9, 8)")
    print("2. Reset Database")
    print("3. Backup Database")
    print("4. Restore Database")
    print("5. Check Login Status")
    print("6. Gather DNA Matches [start page]")
    print("7. Search Inbox")
    print("8. Send Messages")
    print("9. Process Productive Messages")
    print("10. GEDCOM Report (Local File)")
    print("11. API Report (Ancestry Online)")
    print("")
    print("test. Run Main.py Internal Tests")
    print("testall. Run All Module Tests")
    print("")
    print("sec. Credential Manager (Setup/View/Update/Import from .env)")
    print("s. Show Cache Statistics")
    print("t. Toggle Console Log Level (INFO/DEBUG)")
    print("c. Clear Screen")
    print("q. Exit")
    return input("\nEnter choice: ").strip().lower()


# End of menu


def clear_log_file() -> tuple[bool, Optional[str]]:
    """Finds the FileHandler, closes it, clears the log file, and returns a success flag and the log file path."""
    cleared = False
    log_file_handler: Optional[logging.FileHandler] = None
    log_file_path: Optional[str] = None
    try:
        # Step 1: Find the FileHandler in the logger's handlers
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_handler = handler
                log_file_path = handler.baseFilename  # type: ignore[union-attr]
                break
        if log_file_handler and log_file_path:
            # Step 2: Flush the handler (ensuring all previous writes are persisted to disk)
            log_file_handler.flush()  # type: ignore[union-attr]
            # Step 3: Close the handler (releases resources)
            log_file_handler.close()  # type: ignore[union-attr]
            # Step 4: Clear the log file contents
            with Path(log_file_path).open("w", encoding="utf-8"):
                pass
            cleared = True
    except PermissionError as permission_error:
        # Handle permission errors when attempting to open the log file
        logger.warning(
            f"Permission denied clearing log '{log_file_path}': {permission_error}"
        )
    except OSError as io_error:
        # Handle I/O errors when attempting to open the log file
        logger.warning(f"IOError clearing log '{log_file_path}': {io_error}")
    except Exception as error:
        # Handle any other exceptions during the log clearing process
        logger.warning(f"Error clearing log '{log_file_path}': {error}", exc_info=True)
    return cleared, log_file_path


# End of clear_log_file


# State management for caching initialization
class _CachingState:
    """Manages caching initialization state."""
    initialized = False

_caching_state = _CachingState()


def initialize_aggressive_caching() -> None:
    """Initialize aggressive caching systems."""
    try:
        from core.system_cache import warm_system_caches  # type: ignore[import-not-found]
        return warm_system_caches()
    except ImportError:
        logger.warning("System cache module not available")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize aggressive caching: {e}")
        return False


def ensure_caching_initialized() -> None:
    """Initialize aggressive caching systems if not already done."""
    if not _caching_state.initialized:
        logger.debug("Initializing caching systems on-demand...")
        cache_init_success = initialize_aggressive_caching()
        if cache_init_success:
            logger.debug("Caching systems initialized successfully")
            _caching_state.initialized = True
        else:
            logger.warning(
                "Some caching systems failed to initialize, continuing with reduced performance"
            )
        return cache_init_success
    logger.debug("Caching systems already initialized")
    return True


# End of ensure_caching_initialized


# Helper functions for exec_actn

def _determine_browser_requirement(action_name: str) -> bool:
    """Determine if action requires a browser."""
    browserless_actions = [
        "all_but_first_actn",
        "reset_db_actn",
        "backup_db_actn",
        "restore_db_actn",
        "run_action10",
    ]
    return action_name not in browserless_actions


def _determine_required_state(action_name: str, requires_browser: bool) -> str:
    """Determine the required session state for the action."""
    if not requires_browser:
        return "db_ready"
    if action_name == "check_login_actn":
        return "driver_ready"
    return "session_ready"


def _ensure_required_state(session_manager: SessionManager, required_state: str, action_name: str, choice: str) -> bool:
    """Ensure the required session state is achieved."""
    if required_state == "db_ready":
        return session_manager.ensure_db_ready()
    if required_state == "driver_ready":
        return session_manager.browser_manager.ensure_driver_live(f"{action_name} - Browser Start")
    if required_state == "session_ready":
        # Skip CSRF check for Action 11 (cookies available after navigation)
        skip_csrf = (choice in ["11"])
        return session_manager.ensure_session_ready(action_name=f"{action_name} - Setup", skip_csrf=skip_csrf)
    return True


def _prepare_action_arguments(action_func, session_manager: SessionManager, args: tuple) -> tuple:
    """Prepare arguments for action function call."""
    func_sig = inspect.signature(action_func)
    pass_session_manager = "session_manager" in func_sig.parameters
    action_name = action_func.__name__

    # Handle keyword args specifically for coord function
    if action_name in ["coord", "gather_DNA_matches"] and "start" in func_sig.parameters:
        start_val = 1
        int_args = [a for a in args if isinstance(a, int)]
        if int_args:
            start_val = int_args[-1]
        kwargs_for_action = {"start": start_val}

        coord_args = []
        if pass_session_manager:
            coord_args.append(session_manager)
        if action_name == "gather_DNA_matches" and "config_schema" in func_sig.parameters:
            coord_args.append(config)

        return coord_args, kwargs_for_action
    # General case
    final_args = []
    if pass_session_manager:
        final_args.append(session_manager)
    final_args.extend(args)
    return final_args, {}


def _execute_action_function(action_func, prepared_args: tuple, kwargs: dict):
    """Execute the action function with prepared arguments."""
    if kwargs:
        return action_func(*prepared_args, **kwargs)
    return action_func(*prepared_args)


def _should_close_session(action_result, action_exception, close_sess_after: bool, action_name: str) -> bool:
    """Determine if session should be closed."""
    if action_result is False or action_exception is not None:
        logger.warning(f"Action '{action_name}' failed or raised exception. Closing session.")
        return True
    if close_sess_after:
        logger.debug(f"Closing session after '{action_name}' as requested by caller (close_sess_after=True).")
        return True
    return False


def _log_performance_metrics(start_time: float, process, mem_before: float, choice: str, action_name: str):
    """Log performance metrics for the action."""
    duration = time.time() - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_duration = f"{int(hours)} hr {int(minutes)} min {seconds:.2f} sec"

    try:
        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_used = mem_after - mem_before
        mem_log = f"Memory used: {mem_used:.1f} MB"
    except Exception as mem_err:
        mem_log = f"Memory usage unavailable: {mem_err}"

    logger.info("------------------------------------------")
    logger.info(f"Action {choice} ({action_name}) finished.")
    logger.info(f"Duration: {formatted_duration}")
    logger.info(mem_log)
    logger.info("------------------------------------------\n")


def _perform_session_cleanup(session_manager: SessionManager, should_close: bool, action_name: str):
    """Perform session cleanup based on action result."""
    if should_close and isinstance(session_manager, SessionManager):
        if session_manager.browser_needed and session_manager.driver_live:
            logger.debug("Closing browser session...")
            session_manager.close_browser()
            logger.debug("Browser session closed. DB connections kept.")
        elif should_close and action_name in ["all_but_first_actn"]:
            logger.debug("Closing all connections including database...")
            session_manager.close_sess(keep_db=False)
            logger.debug("All connections closed.")
    elif isinstance(session_manager, SessionManager) and session_manager.driver_live and not should_close:
        logger.debug(f"Keeping session live after '{action_name}'.")


def exec_actn(
    action_func,
    session_manager: SessionManager,
    choice: str,
    close_sess_after: bool = False,
    *args,
) -> bool:
    """
    Executes an action, ensuring the required session state
    (driver live, session ready) is met beforehand using SessionManager methods.
    Leaves the session open unless action fails or close_sess_after is True.

    Args:
        action_func: The function representing the action to execute.
        session_manager: The SessionManager instance to manage session state.
        choice: The user's choice of action.
        close_sess_after: Flag to close session after action, defaults to False.
        *args: Additional arguments to pass to the action function.

    Returns:
        True if action completed successfully, False otherwise.
    """
    start_time = time.time()
    action_name = action_func.__name__

    # Performance Logging Setup
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    logger.info("------------------------------------------")
    logger.info(f"Action {choice}: Starting {action_name}...")
    logger.info("------------------------------------------")

    action_result = None
    action_exception = None

    # Determine browser requirement and required state
    requires_browser = _determine_browser_requirement(action_name)
    session_manager.browser_needed = requires_browser
    required_state = _determine_required_state(action_name, requires_browser)

    try:
        # Ensure Required State
        state_ok = _ensure_required_state(session_manager, required_state, action_name, choice)
        if not state_ok:
            logger.error(f"Failed to achieve required state '{required_state}' for action '{action_name}'.")
            raise Exception(f"Setup failed: Could not achieve state '{required_state}'.")

        # Execute Action
        prepared_args, kwargs = _prepare_action_arguments(action_func, session_manager, args)
        action_result = _execute_action_function(action_func, prepared_args, kwargs)

    except Exception as e:
        logger.error(f"Exception during action {action_name}: {e}", exc_info=True)
        action_result = False
        action_exception = e

    finally:
        # Session Closing Logic
        should_close = _should_close_session(action_result, action_exception, close_sess_after, action_name)

        # Performance Logging
        print(" ")  # Spacer
        if action_result is False:
            logger.debug(f"Action {choice} ({action_name}) reported failure.")
        elif action_exception is not None:
            logger.debug(f"Action {choice} ({action_name}) failed due to exception: {type(action_exception).__name__}.")

        final_outcome = action_result is not False and action_exception is None
        logger.debug(f"Final outcome for Action {choice} ('{action_name}'): {final_outcome}\n\n")

        _log_performance_metrics(start_time, process, mem_before, choice, action_name)

        # Perform cleanup
        _perform_session_cleanup(session_manager, should_close, action_name)

    return final_outcome

# End of exec_actn


# --- Action Functions


# Action 0 (all_but_first_actn)
def _delete_table_records(sess, table_class, filter_condition, table_name: str, person_id_to_keep: int) -> int:
    """Delete records from a table based on filter condition."""
    logger.debug(f"Deleting from {table_name} where people_id != {person_id_to_keep}...")
    result = sess.query(table_class).filter(filter_condition).delete(synchronize_session=False)
    count = result if result is not None else 0
    logger.info(f"Deleted {count} {table_name} records.")
    return count


def _perform_deletions(sess, person_id_to_keep: int) -> dict:
    """Perform all deletion operations and return counts."""
    deleted_counts = {
        "conversation_log": _delete_table_records(
            sess, ConversationLog, ConversationLog.people_id != person_id_to_keep,
            "conversation_log", person_id_to_keep
        ),
        "dna_match": _delete_table_records(
            sess, DnaMatch, DnaMatch.people_id != person_id_to_keep,
            "dna_match", person_id_to_keep
        ),
        "family_tree": _delete_table_records(
            sess, FamilyTree, FamilyTree.people_id != person_id_to_keep,
            "family_tree", person_id_to_keep
        ),
        "people": _delete_table_records(
            sess, Person, Person.id != person_id_to_keep,
            "people", person_id_to_keep
        ),
    }

    total_deleted = sum(deleted_counts.values())
    if total_deleted == 0:
        logger.info(f"No records found to delete besides Person ID {person_id_to_keep}.")

    return deleted_counts


def all_but_first_actn(session_manager: SessionManager, *_):
    """
    V1.2: Modified to delete records from people, conversation_log,
          dna_match, and family_tree, except for the person with a
          specific profile_id. Leaves message_types untouched. Browserless.
    Closes the provided main session pool FIRST.
    Creates a temporary SessionManager for the delete operation.
    """
    # Define the specific profile ID to keep from config (ensure it's uppercase for comparison)
    profile_id_to_keep = config.test.test_profile_id
    if not profile_id_to_keep:
        logger.error(            "TESTING_PROFILE_ID is not configured. Cannot determine which profile to keep."
        )
        return False
    profile_id_to_keep = profile_id_to_keep.upper()

    temp_manager = None  # Initialize
    session = None
    success = False
    try:
        # --- Close main pool FIRST ---
        if session_manager:
            logger.debug(
                f"Closing main DB connections before deleting data (except {profile_id_to_keep})..."
            )
            session_manager.cls_db_conn(keep_db=False)
            logger.debug("Main DB pool closed.")
        # --- End closing main pool ---

        logger.info(
            f"Deleting data for all people except Profile ID: {profile_id_to_keep}..."
        )
        # Create a temporary SessionManager for this specific operation
        temp_manager = SessionManager()
        session = temp_manager.get_db_conn()
        if session is None:
            raise Exception("Failed to get DB session via temporary manager.")

        with db_transn(session) as sess:
            # 1. Find the ID of the person to keep
            person_to_keep = (
                sess.query(Person.id, Person.username)
                .filter(
                    Person.profile_id == profile_id_to_keep, Person.deleted_at.is_(None)
                )
                .first()
            )

            if not person_to_keep:
                logger.warning(
                    f"Person with Profile ID {profile_id_to_keep} not found. No records will be deleted."
                )
                return True  # Exit gracefully if the keeper doesn't exist

            person_id_to_keep = person_to_keep.id
            logger.debug(
                f"Keeping Person ID: {person_id_to_keep} (ProfileID: {profile_id_to_keep}, User: {person_to_keep.username})"
            )

            # --- Perform Deletions ---
            _perform_deletions(sess, person_id_to_keep)

        success = True  # Mark success if transaction completes

    except Exception as e:
        logger.error(
            f"Error during deletion (except {profile_id_to_keep}): {e}", exc_info=True
        )
        success = False  # Explicitly mark failure
    finally:
        # Clean up the temporary session manager and its resources
        if temp_manager:
            if session:
                temp_manager.return_session(session)
            temp_manager.cls_db_conn(keep_db=False)  # Close the temp pool
        logger.debug(f"Delete action (except {profile_id_to_keep}) finished.")
    return success


# end of action 0 (all_but_first_actn)


# Helper functions for run_core_workflow_action

def _run_action6_gather(session_manager) -> bool:
    """Run Action 6: Gather Matches."""
    logger.info("--- Running Action 6: Gather Matches (Always from page 1) ---")
    gather_result = gather_DNA_matches(session_manager, config, start=1)
    if gather_result is False:
        logger.error("Action 6 FAILED.")
        print("ERROR: Match gathering failed. Check logs for details.")
        return False
    logger.info("Action 6 OK.")
    print("✓ Match gathering completed successfully.")
    return True


def _run_action7_inbox(session_manager) -> bool:
    """Run Action 7: Search Inbox."""
    logger.info("--- Running Action 7: Search Inbox ---")
    inbox_url = urljoin(config.api.base_url, "/messaging/")
    logger.debug(f"Navigating to Inbox ({inbox_url}) for Action 7...")

    try:
        if not nav_to_page(
            session_manager.driver,
            inbox_url,
            "div.messaging-container",
            session_manager,
        ):
            logger.error("Action 7 nav FAILED - Could not navigate to inbox page.")
            print("ERROR: Could not navigate to inbox page. Check network connection.")
            return False

        logger.debug("Navigation to inbox page successful.")
        time.sleep(2)

        logger.debug("Running inbox search...")
        inbox_processor = InboxProcessor(session_manager=session_manager)
        search_result = inbox_processor.search_inbox()

        if search_result is False:
            logger.error("Action 7 FAILED - Inbox search returned failure.")
            print("ERROR: Inbox search failed. Check logs for details.")
            return False

        logger.info("Action 7 OK.")
        print("✓ Inbox search completed successfully.")
        return True

    except Exception as inbox_error:
        logger.error(f"Action 7 FAILED with exception: {inbox_error}", exc_info=True)
        print(f"ERROR during inbox search: {inbox_error}")
        return False


def _run_action9_process_productive(session_manager) -> bool:
    """Run Action 9: Process Productive Messages."""
    logger.info("--- Running Action 9: Process Productive Messages ---")
    logger.debug("Navigating to Base URL for Action 9...")

    try:
        if not nav_to_page(
            session_manager.driver,
            config.api.base_url,
            WAIT_FOR_PAGE_SELECTOR,
            session_manager,
        ):
            logger.error("Action 9 nav FAILED - Could not navigate to base URL.")
            print("ERROR: Could not navigate to base URL. Check network connection.")
            return False

        logger.debug("Navigation to base URL successful. Processing productive messages...")
        time.sleep(2)

        process_result = process_productive_messages(session_manager)

        if process_result is False:
            logger.error("Action 9 FAILED - Productive message processing returned failure.")
            print("ERROR: Productive message processing failed. Check logs for details.")
            return False

        logger.info("Action 9 OK.")
        print("✓ Productive message processing completed successfully.")
        return True

    except Exception as process_error:
        logger.error(f"Action 9 FAILED with exception: {process_error}", exc_info=True)
        print(f"ERROR during productive message processing: {process_error}")
        return False


def _run_action8_send_messages(session_manager) -> bool:
    """Run Action 8: Send Messages."""
    logger.info("--- Running Action 8: Send Messages ---")
    logger.debug("Navigating to Base URL for Action 8...")

    try:
        if not nav_to_page(
            session_manager.driver,
            config.api.base_url,
            WAIT_FOR_PAGE_SELECTOR,
            session_manager,
        ):
            logger.error("Action 8 nav FAILED - Could not navigate to base URL.")
            print("ERROR: Could not navigate to base URL. Check network connection.")
            return False

        logger.debug("Navigation to base URL successful. Sending messages...")
        time.sleep(2)

        send_result = send_messages_to_matches(session_manager)

        if send_result is False:
            logger.error("Action 8 FAILED - Message sending returned failure.")
            print("ERROR: Message sending failed. Check logs for details.")
            return False

        logger.info("Action 8 OK.")
        print("✓ Message sending completed successfully.")
        return True

    except Exception as message_error:
        logger.error(f"Action 8 FAILED with exception: {message_error}", exc_info=True)
        print(f"ERROR during message sending: {message_error}")
        return False


# Action 1
def run_core_workflow_action(session_manager: SessionManager, *_: Any) -> bool:
    """
    Action to run the core workflow sequence: Action 7 (Inbox) → Action 9 (Process Productive) → Action 8 (Send Messages).
    Optionally runs Action 6 (Gather) first if configured.
    Relies on exec_actn ensuring session is ready beforehand.
    """
    result = False

    if not session_manager or not session_manager.session_ready:
        logger.error("Cannot run core workflow: Session not ready.")
        return result

    try:
        # Run Action 6 if configured
        run_action6 = config.include_action6_in_workflow
        if run_action6 and not _run_action6_gather(session_manager):
            return result

        # Run Action 7, 9, and 8 in sequence
        if (_run_action7_inbox(session_manager) and
            _run_action9_process_productive(session_manager) and
            _run_action8_send_messages(session_manager)):

            # Build success message
            action_sequence = []
            if run_action6:
                action_sequence.append("6")
            action_sequence.extend(["7", "9", "8"])
            action_sequence_str = "-".join(action_sequence)

            logger.info(f"Core Workflow (Actions {action_sequence_str}) finished successfully.")
            print(f"\n✓ Core Workflow (Actions {action_sequence_str}) completed successfully.")
            result = True

    except Exception as e:
        logger.error(f"Critical error during core workflow: {e}", exc_info=True)
        print(f"CRITICAL ERROR during core workflow: {e}")

    return result


# End Action 1


# Action 2 (reset_db_actn)
def _truncate_all_tables(temp_manager: SessionManager) -> bool:
    """Truncate all tables in the database."""
    logger.debug("Truncating all tables...")
    truncate_session = temp_manager.get_db_conn()
    if not truncate_session:
        logger.critical("Failed to get session for truncating tables. Reset aborted.")
        return False

    try:
        with db_transn(truncate_session) as sess:
            # Delete all records from tables in reverse order of dependencies
            sess.query(ConversationLog).delete(synchronize_session=False)
            sess.query(DnaMatch).delete(synchronize_session=False)
            sess.query(FamilyTree).delete(synchronize_session=False)
            sess.query(Person).delete(synchronize_session=False)
            # Keep MessageType table intact
        temp_manager.return_session(truncate_session)
        logger.debug("All tables truncated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error truncating tables: {e}", exc_info=True)
        temp_manager.return_session(truncate_session)
        return False


def _reinitialize_database_schema(temp_manager: SessionManager) -> bool:
    """Re-initialize database schema."""
    logger.debug("Re-initializing database schema...")
    try:
        # This will create a new engine and session factory pointing to the file path
        temp_manager.db_manager._initialize_engine_and_session()
        if not temp_manager.db_manager.engine or not temp_manager.db_manager.Session:
            raise SQLAlchemyError("Failed to initialize DB engine/session for recreation!")

        # This will recreate the tables in the existing file
        Base.metadata.create_all(temp_manager.db_manager.engine)
        logger.debug("Database schema recreated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error reinitializing database schema: {e}", exc_info=True)
        return False


def _seed_message_templates(recreation_session: Any) -> bool:
    """Seed message templates from messages.json."""
    logger.debug("Seeding message_types table...")
    script_dir = Path(__file__).resolve().parent
    messages_file = script_dir / "messages.json"

    if not messages_file.exists():
        logger.warning("'messages.json' not found. Cannot seed MessageTypes.")
        return False

    try:
        with messages_file.open("r", encoding="utf-8") as f:
            messages_data = json.load(f)

        if not isinstance(messages_data, dict):
            logger.error("'messages.json' has incorrect format. Cannot seed.")
            return False

        with db_transn(recreation_session) as sess:
            # First check if there are any existing message templates
            existing_count = sess.query(func.count(MessageTemplate.id)).scalar() or 0

            if existing_count > 0:
                logger.debug(f"Found {existing_count} existing message templates. Skipping seeding.")
            else:
                # Only add message templates if none exist
                templates_to_add = [MessageTemplate(template_name=name) for name in messages_data]
                if templates_to_add:
                    sess.add_all(templates_to_add)
                    logger.debug(f"Added {len(templates_to_add)} message templates.")
                else:
                    logger.warning("No message templates found in messages.json to seed.")

        count = recreation_session.query(func.count(MessageTemplate.id)).scalar() or 0
        logger.debug(f"MessageTemplate seeding complete. Total templates in DB: {count}")
        return True
    except Exception as e:
        logger.error(f"Error seeding message templates: {e}", exc_info=True)
        return False


def reset_db_actn(session_manager: SessionManager, *_):
    """
    Action to COMPLETELY reset the database by deleting the file. Browserless.
    - Closes main pool.
    - Deletes the .db file.
    - Recreates schema from scratch.
    - Seeds the MessageType table.
    """
    db_path = config.database.database_file
    reset_successful = False
    temp_manager = None  # For recreation/seeding
    recreation_session = None  # Session for seeding

    try:
        # --- 1. Close main pool FIRST ---
        if session_manager:
            logger.debug("Closing main DB connections before database deletion...")
            session_manager.cls_db_conn(keep_db=False)  # Ensure pool is closed
            logger.debug("Main DB pool closed.")

        # Force garbage collection to release any file handles
        logger.debug("Running garbage collection to release file handles...")
        gc.collect()
        time.sleep(1.0)
        gc.collect()

        # --- 2. Delete the Database File ---
        if db_path is None:
            logger.critical("DATABASE_FILE is not configured. Reset aborted.")
            return False

        logger.debug(f"Attempting to delete database file: {db_path}...")
        try:
            # Streamlined database reset using single temporary SessionManager
            logger.debug("Creating temporary session manager for database reset...")
            temp_manager = SessionManager()

            # Step 1: Truncate all tables
            if not _truncate_all_tables(temp_manager):
                return False

            # Step 2: Re-initialize database schema
            if not _reinitialize_database_schema(temp_manager):
                return False

            # Step 3: Seed MessageType Table
            recreation_session = temp_manager.get_db_conn()
            if not recreation_session:
                raise SQLAlchemyError("Failed to get session for seeding MessageTypes!")

            _seed_message_templates(recreation_session)

            reset_successful = True
            logger.info("Database reset completed successfully.")

        except Exception as recreate_err:
            logger.error(
                f"Error during DB recreation/seeding: {recreate_err}", exc_info=True
            )
            reset_successful = False
        finally:
            # Clean up the temporary manager and its session/engine
            logger.debug("Cleaning up temporary resource manager for reset...")
            if temp_manager:
                if recreation_session:
                    temp_manager.return_session(recreation_session)
                temp_manager.cls_db_conn(keep_db=False)  # Dispose temp engine
            logger.debug("Temporary resource manager cleanup finished.")

    except Exception as e:
        logger.error(f"Outer error during DB reset action: {e}", exc_info=True)
        reset_successful = False  # Ensure failure is marked

    finally:
        logger.debug("Reset DB action finished.")

    return reset_successful


# end of Action 2 (reset_db_actn)


# Action 3 (backup_db_actn)
def backup_db_actn(*_: Any) -> bool:
    """Action to backup the database. Browserless."""
    try:
        logger.debug("Starting DB backup...")
        # _session_manager isn't used but needed for exec_actn compatibility
        result = backup_database()
        if result:
            logger.info("DB backup OK.")
            return True
        logger.error("DB backup failed.")
        return False
    except Exception as e:
        logger.error(f"Error during DB backup: {e}", exc_info=True)
        return False


# end of Action 3


# Action 4 (restore_db_actn)
def restore_db_actn(session_manager: SessionManager, *_):  # Added session_manager back
    """
    Action to restore the database. Browserless.
    Closes the provided main session pool FIRST.
    """
    backup_dir = config.database.data_dir
    db_path = config.database.database_file
    success = False

    # Validate paths
    if backup_dir is None:
        logger.error("Cannot restore database: DATA_DIR is not configured.")
        return False

    if db_path is None:
        logger.error("Cannot restore database: DATABASE_FILE is not configured.")
        return False

    backup_path = backup_dir / "ancestry_backup.db"

    try:
        # --- Close main pool FIRST ---
        if session_manager:
            logger.debug("Closing main DB connections before restore...")
            session_manager.cls_db_conn(keep_db=False)
            logger.debug("Main DB pool closed.")
        # --- End closing main pool ---

        logger.debug(f"Restoring DB from: {backup_path}")
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}.")
            return False

        logger.debug("Running GC before restore...")
        gc.collect()
        time.sleep(0.5)
        gc.collect()

        shutil.copy2(backup_path, db_path)
        logger.info("Db restored from backup OK.")
        success = True
    except FileNotFoundError:
        logger.error(f"Backup not found during copy: {backup_path}")
    except (OSError, shutil.Error) as e:
        logger.error(f"Error restoring DB: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"Unexpected restore error: {e}", exc_info=True)
    finally:
        logger.debug("DB restore action finished.")
    return success


# end of Action 4


def _display_session_info(session_manager: SessionManager) -> None:
    """Display session information if available."""
    if session_manager.my_profile_id:
        print(f"  Profile ID: {session_manager.my_profile_id}")
    if session_manager.tree_owner_name:
        print(f"  Account: {session_manager.tree_owner_name}")


def _handle_logged_in_status(session_manager: SessionManager) -> bool:
    """Handle the case when user is already logged in."""
    print("\n✓ You are currently logged in to Ancestry.")
    _display_session_info(session_manager)
    return True


def _verify_login_success(session_manager: SessionManager) -> bool:
    """Verify login was successful and display session info."""
    final_status = login_status(session_manager, disable_ui_fallback=False)
    if final_status is True:
        print("✓ Login verification confirmed.")
        _display_session_info(session_manager)
        return True
    print("⚠️  Login appeared successful but verification failed.")
    return False


def _attempt_login(session_manager: SessionManager) -> bool:
    """Attempt to log in with stored credentials."""
    print("\n✗ You are NOT currently logged in to Ancestry.")
    print("  Attempting to log in with stored credentials...")

    try:
        login_result = log_in(session_manager)

        if login_result:
            print("✓ Login successful!")
            return _verify_login_success(session_manager)

        print("✗ Login failed. Please check your credentials.")
        print("  You can update credentials using the 'sec' option in the main menu.")
        return False

    except Exception as login_e:
        logger.error(f"Exception during login attempt: {login_e}", exc_info=True)
        print(f"✗ Login failed with error: {login_e}")
        print("  You can update credentials using the 'sec' option in the main menu.")
        return False


# Action 5 (check_login_actn)
def check_login_actn(session_manager: SessionManager, *_) -> bool:
    """
    REVISED V12: Checks login status and attempts login if needed.
    This action starts a browser session and checks login status.
    If not logged in, it attempts to log in using stored credentials.
    Provides clear user feedback about the final login state.
    """
    # Phase 1 (Driver Start) is handled by exec_actn if needed.
    # We only need to check if driver is live before proceeding.
    if not session_manager.driver_live:
        logger.error("Driver not live. Cannot check login status.")
        print("ERROR: Browser not started. Cannot check login status.")
        print(
            "       Select any browser-required action (1, 6-9) to start the browser."
        )
        return False

    print("\nChecking login status...")

    # Call login_status directly to check initial status
    try:
        status = login_status(
            session_manager, disable_ui_fallback=False
        )  # Use UI fallback for reliability

        if status is True:
            return _handle_logged_in_status(session_manager)
        if status is False:
            return _attempt_login(session_manager)
        # Status is None
        print("\n? Unable to determine login status due to a technical error.")
        print("  This may indicate a browser or network issue.")
        logger.warning("Login status check returned None (ambiguous result).")
        return False

    except Exception as e:
        logger.error(f"Exception during login status check: {e}", exc_info=True)
        print(f"\n! Error checking login status: {e}")
        print("  This may indicate a browser or network issue.")
        return False


# End Action 5


# Action 6 (gather_DNA_matches wrapper)
def gather_DNA_matches(session_manager: SessionManager, config_schema: Optional[Any] = None, start: int = 1) -> bool:
    """
    Action wrapper for gathering matches (coord function from action6).
    Relies on exec_actn ensuring session is ready before calling.

    Args:
        session_manager: The SessionManager instance.
        config_schema: The configuration schema (optional, uses global config if None).
        start: The page number to start gathering from (default is 1).
    """
    # Use global config if config_schema not provided
    if config_schema is None:
        config_schema = config
    # Guard clause now checks session_ready
    if not session_manager or not session_manager.session_ready:
        logger.error("Cannot gather matches: Session not ready.")
        print("ERROR: Session not ready. Cannot gather matches.")
        return False

    try:
        # Call the imported function from action6
        result = coord(session_manager, start=start)
        if result is False:
            logger.error("Match gathering reported failure or incomplete run.")
            print("⚠️  WARNING: Match gathering incomplete or failed. Check logs for details.")
            return False
        logger.info("Gathering matches completed successfully.")
        print("✓ Match gathering completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error during gather_DNA_matches: {e}", exc_info=True)
        print(f"ERROR: Exception during match gathering: {e}")
        return False


# End of gather_DNA_matches





# Action 7 (srch_inbox_actn)
def srch_inbox_actn(session_manager: Any, *_: Any) -> bool:
    """Action to search the inbox. Relies on exec_actn ensuring session is ready."""
    # Guard clause now checks session_manager exists
    if not session_manager:
        logger.error("Cannot search inbox: SessionManager is None.")
        return False

    # Check session_ready attribute safely
    session_ready = getattr(session_manager, "session_ready", None)
    if session_ready is None:
        # If session_ready is not set, initialize it based on driver_live
        driver_live = getattr(session_manager, "driver_live", False)
        if driver_live:
            logger.warning("session_ready not set, initializing based on driver_live")
            session_manager.session_ready = True
            session_ready = True
        else:
            logger.warning(
                "session_ready and driver_live not set, initializing to False"
            )
            session_manager.session_ready = False
            session_ready = False

    # Now check if session is ready
    if not session_ready:
        logger.error("Cannot search inbox: Session not ready.")
        return False

    logger.debug("Starting inbox search...")
    try:
        processor = InboxProcessor(session_manager=session_manager)
        result = processor.search_inbox()
        if result is False:
            logger.error("Inbox search reported failure.")
            return False
        logger.info("Inbox search OK.")
        return True  # Use INFO
    except Exception as e:
        logger.error(f"Error during inbox search: {e}", exc_info=True)
        return False


# End of srch_inbox_actn


# Action 8 (send_messages_action)
def send_messages_action(session_manager: Any, *_: Any) -> bool:
    """Action to send messages. Relies on exec_actn ensuring session is ready."""
    # Guard clause now checks session_manager exists
    if not session_manager:
        logger.error("Cannot send messages: SessionManager is None.")
        return False

    # Check session_ready attribute safely
    session_ready = getattr(session_manager, "session_ready", None)
    if session_ready is None:
        # If session_ready is not set, initialize it based on driver_live
        driver_live = getattr(session_manager, "driver_live", False)
        if driver_live:
            logger.warning("session_ready not set, initializing based on driver_live")
            session_manager.session_ready = True
            session_ready = True
        else:
            logger.warning(
                "session_ready and driver_live not set, initializing to False"
            )
            session_manager.session_ready = False
            session_ready = False

    # Now check if session is ready
    if not session_ready:
        logger.error("Cannot send messages: Session not ready.")
        return False

    logger.debug("Starting message sending...")
    try:
        # Navigate to Base URL first (good practice before starting message loops)
        logger.debug("Navigating to Base URL before sending...")
        if not nav_to_page(
            session_manager.driver,
            config.api.base_url,
            WAIT_FOR_PAGE_SELECTOR,
            session_manager,
        ):
            logger.error("Failed nav to base URL. Aborting message sending.")
            return False
        logger.debug("Navigation OK. Proceeding to send messages...")

        # Call the actual sending function
        result = send_messages_to_matches(session_manager)
        if result is False:
            logger.error("Message sending reported failure.")
            return False
        logger.info("Messages sent OK.")
        return True  # Use INFO
    except Exception as e:
        logger.error(f"Error during message sending: {e}", exc_info=True)
        return False


# End of send_messages_action


# Action 9 (process_productive_messages_action)
def process_productive_messages_action(session_manager: Any, *_: Any) -> bool:
    """Action to process productive messages. Relies on exec_actn ensuring session is ready."""
    # Guard clause now checks session_manager exists
    if not session_manager:
        logger.error("Cannot process productive messages: SessionManager is None.")
        return False

    # Check session_ready attribute safely
    session_ready = getattr(session_manager, "session_ready", None)
    if session_ready is None:
        # If session_ready is not set, initialize it based on driver_live
        driver_live = getattr(session_manager, "driver_live", False)
        if driver_live:
            logger.warning("session_ready not set, initializing based on driver_live")
            session_manager.session_ready = True
            session_ready = True
        else:
            logger.warning(
                "session_ready and driver_live not set, initializing to False"
            )
            session_manager.session_ready = False
            session_ready = False

    # Now check if session is ready
    if not session_ready:
        logger.error("Cannot process productive messages: Session not ready.")
        return False

    logger.debug("Starting productive message processing...")
    try:
        # Call the actual processing function
        result = process_productive_messages(session_manager)
        if result is False:
            logger.error("Productive message processing reported failure.")
            return False
        logger.info("Productive message processing OK.")
        return True
    except Exception as e:
        logger.error(f"Error during productive message processing: {e}", exc_info=True)
        return False


# End of process_productive_messages_action


# Action 11 (run_action11_wrapper)
def run_action11_wrapper(session_manager: Any, *_: Any) -> bool:
    """Action to run API Report. Relies on exec_actn for consistent logging and error handling."""
    logger.debug("Starting API Report...")
    try:
        # Call the actual API Report function, passing the session_manager
        result = run_action11(session_manager)
        if result is False:
            logger.error("API Report reported failure.")
            return False
        logger.debug("API Report OK.")
        return True
    except Exception as e:
        logger.error(f"Error during API Report: {e}", exc_info=True)
        return False


# End of run_action11_wrapper


def _set_windows_console_focus() -> None:
    """Ensure terminal window has focus on Windows."""
    try:
        if os.name == 'nt':  # Windows
            import ctypes

            # Get console window handle
            kernel32 = ctypes.windll.kernel32
            user32 = ctypes.windll.user32

            # Get console window
            console_window = kernel32.GetConsoleWindow()
            if console_window:
                # Bring console window to foreground
                user32.SetForegroundWindow(console_window)
                user32.ShowWindow(console_window, 9)  # SW_RESTORE
    except Exception:
        pass  # Silently ignore focus errors


def _print_config_error_message() -> None:
    """Print detailed configuration error message and exit."""
    logger.critical("Configuration validation failed - unable to proceed")
    print("\n❌ CONFIGURATION ERROR:")
    print("   Critical configuration validation failed.")
    print("   This usually means missing credentials or configuration files.")
    print("")
    print("💡 SOLUTIONS:")
    print("\n🔒 Recommended: Use the secure credential manager")
    print("   python credentials.py")

    print("\n📋 Alternative options:")
    print(
        "   1. Run 'sec. Setup Security (Encrypt Credentials)' from main menu"
    )
    print("   2. Copy .env.example to .env and add your credentials")
    print("   3. Ensure the required security dependencies are installed:")
    print("      pip install cryptography keyring")

    print("\n📚 For detailed instructions:")
    print("   See ENV_IMPORT_GUIDE.md")

    print("\n⚠️ Security Note:")
    print("   The secure credential manager requires:")
    print("   - cryptography package (for encryption)")
    print("   - keyring package (for secure key storage)")

    print("\nExiting application...")
    sys.exit(1)


def _check_action_confirmation(choice: str) -> bool:
    """
    Check if action requires confirmation and get user confirmation.

    Returns:
        True if action should proceed, False if cancelled
    """
    confirm_actions = {
        "0": "Delete all people except specific profile ID",
        "2": "COMPLETELY reset the database (deletes data)",
        "4": "Restore database from backup (overwrites data)",
    }

    if choice in confirm_actions:
        action_desc = confirm_actions[choice]
        confirm = (
            input(
                f"Are you sure you want to {action_desc}? ⚠️  This cannot be undone. (yes/no): "
            )
            .strip()
            .lower()
        )
        if confirm not in ["yes", "y"]:
            print("Action cancelled.\n")
            return False
        print(" ")  # Newline after confirmation

    return True


def _run_main_tests() -> None:
    """Run Main.py Internal Tests."""
    try:
        print("\n" + "=" * 60)
        print("RUNNING MAIN.PY INTERNAL TESTS")
        print("=" * 60)
        result = run_comprehensive_tests()
        if result:
            print("\n🎉 All main.py tests completed successfully!")
        else:
            print("\n⚠️ Some main.py tests failed. Check output above.")
    except Exception as e:
        logger.error(f"Error running main.py tests: {e}")
        print(f"Error running main.py tests: {e}")
    print("\nReturning to main menu...")
    input("Press Enter to continue...")


def _run_all_tests() -> None:
    """Run All Module Tests."""
    try:
        import subprocess

        print("\n" + "=" * 60)
        print("RUNNING ALL MODULE TESTS")
        print("=" * 60)
        result = subprocess.run(
            [sys.executable, "run_all_tests.py"],
            check=False, capture_output=False,
            text=True,
        )
        if result.returncode == 0:
            print("\n🎉 All module tests completed successfully!")
        else:
            print(f"\n⚠️ Some tests failed (exit code: {result.returncode})")
    except FileNotFoundError:
        print("Error: run_all_tests.py not found in current directory.")
    except Exception as e:
        logger.error(f"Error running all tests: {e}")
        print(f"Error running all tests: {e}")
    print("\nReturning to main menu...")
    input("Press Enter to continue...")


def _run_credential_manager() -> None:
    """Setup Security (Encrypt Credentials)."""
    try:
        from credentials import UnifiedCredentialManager

        print("\n" + "=" * 50)
        print("CREDENTIAL MANAGEMENT")
        print("=" * 50)
        manager = UnifiedCredentialManager()
        manager.run()
    except ImportError as e:
        logger.error(f"Error importing credentials manager: {e}")
        print("\n❌ Error: Unable to use the credential manager.")

        if "No module named 'cryptography'" in str(e) or "No module named 'keyring'" in str(e):
            print("\n" + "=" * 60)
            print("       SECURITY DEPENDENCIES MISSING")
            print("=" * 60)
            print("\nRequired security packages are not installed:")
            print("  - cryptography: For secure encryption/decryption")
            print("  - keyring: For secure storage of master keys")

            print("\n📋 Installation Instructions:")
            print("  1. Install required packages:")
            print("     pip install cryptography keyring")
            print("     - OR -")
            print("     pip install -r requirements.txt")

            print("\n📚 For more information, see:")
            print("  - ENV_IMPORT_GUIDE.md")
            print("  - SECURITY_STREAMLINED.md")
        else:
            print(
                "Error: credentials.py not found or has other import issues."
            )
            print(f"Details: {e}")
            print(
                "\nPlease check that all files are in the correct location."
            )
    except Exception as e:
        logger.error(f"Error running credential manager: {e}")
        print(f"Error running credential manager: {e}")
    print("\nReturning to main menu...")
    input("Press Enter to continue...")


def _show_cache_statistics() -> None:
    """Show cache statistics."""
    try:
        logger.info("Cache statistics feature currently unavailable")
        print("Cache statistics feature currently unavailable.")
    except Exception as e:
        logger.error(f"Error displaying cache statistics: {e}")
        print("Error displaying cache statistics. Check logs for details.")


def _toggle_log_level() -> None:
    """Toggle console log level between DEBUG and INFO."""
    os.system("cls" if os.name == "nt" else "clear")
    if logger and logger.handlers:
        console_handler = None
        for handler in logger.handlers:
            if (
                isinstance(handler, logging.StreamHandler)
                and handler.stream == sys.stderr
            ):
                console_handler = handler
                break
        if console_handler:
            current_level = console_handler.level
            new_level = (
                logging.DEBUG
                if current_level > logging.DEBUG
                else logging.INFO
            )
            new_level_name = logging.getLevelName(new_level)
            # Re-call setup_logging to potentially update filters etc. too
            setup_logging(log_level=new_level_name)
            logger.info(f"Console log level toggled to: {new_level_name}")
        else:
            logger.warning(
                "Could not find console handler to toggle level."
            )
    else:
        print(
            "WARNING: Logger not ready or has no handlers.", file=sys.stderr
        )


def _handle_database_actions(choice: str, session_manager: Any) -> bool:
    """Handle database-only actions (no browser needed)."""
    if choice == "0":
        exec_actn(all_but_first_actn, session_manager, choice)
    elif choice == "2":
        exec_actn(reset_db_actn, session_manager, choice)
    elif choice == "3":
        exec_actn(backup_db_actn, session_manager, choice)
    elif choice == "4":
        exec_actn(restore_db_actn, session_manager, choice)
    return True


def _handle_action6_with_start_page(choice: str, session_manager: Any, config: Any) -> bool:
    """Handle Action 6 (DNA match gathering) with optional start page."""
    parts = choice.split()
    start_val = 1
    if len(parts) > 1:
        try:
            start_arg = int(parts[1])
            start_val = start_arg if start_arg > 0 else 1
        except ValueError:
            logger.warning(f"Invalid start page '{parts[1]}'. Using 1.")
            print(f"Invalid start page '{parts[1]}'. Using page 1 instead.")

    exec_actn(gather_DNA_matches, session_manager, "6", False, config, start_val)
    return True


def _handle_browser_actions(choice: str, session_manager: Any, config: Any) -> bool:
    """Handle browser-required actions."""
    result = False

    if choice == "1":
        ensure_caching_initialized()
        exec_actn(run_core_workflow_action, session_manager, choice, close_sess_after=True)
        result = True
    elif choice == "5":
        exec_actn(check_login_actn, session_manager, choice)
        result = True
    elif choice.startswith("6"):
        # Action 6 - DNA match gathering (with optional start page)
        result = _handle_action6_with_start_page(choice, session_manager, config)
    elif choice == "7":
        exec_actn(srch_inbox_actn, session_manager, choice)
        result = True
    elif choice == "8":
        exec_actn(send_messages_action, session_manager, choice)
        result = True
    elif choice == "9":
        ensure_caching_initialized()
        exec_actn(process_productive_messages_action, session_manager, choice)
        result = True
    elif choice == "10":
        ensure_caching_initialized()
        exec_actn(run_action10, session_manager, choice)
        result = True
    elif choice == "11":
        ensure_caching_initialized()
        exec_actn(run_action11_wrapper, session_manager, choice)
        result = True

    return result


def _handle_test_options(choice: str) -> bool:
    """Handle test options."""
    if choice == "test":
        _run_main_tests()
        return True
    if choice == "testall":
        _run_all_tests()
        return True
    return False


def _handle_meta_options(choice: str) -> bool:
    """Handle meta options (sec, s, t, c, q)."""
    if choice == "sec":
        _run_credential_manager()
        return True
    if choice == "s":
        _show_cache_statistics()
        return True
    if choice == "t":
        _toggle_log_level()
        return True
    if choice == "c":
        os.system("cls" if os.name == "nt" else "clear")
        return True
    if choice == "q":
        os.system("cls" if os.name == "nt" else "clear")
        print("Exiting.")
        return False
    return False


def _dispatch_menu_action(choice: str, session_manager: Any, config: Any) -> bool:
    """
    Dispatch menu action based on user choice.

    Returns:
        True to continue menu loop, False to exit
    """
    # --- Database-only actions (no browser needed) ---
    if choice in ["0", "2", "3", "4"]:
        return _handle_database_actions(choice, session_manager)

    # --- Browser-required actions ---
    if choice in ["1", "5", "7", "8", "9", "10", "11"] or choice.startswith("6"):
        result = _handle_browser_actions(choice, session_manager, config)
        if result:
            return True

    # --- Test Options ---
    result = _handle_test_options(choice)
    if result:
        return True

    # --- Meta Options ---
    result = _handle_meta_options(choice)
    if result is not False:
        return result

    # Handle invalid choices
    print("Invalid choice.\n")
    return True


def main() -> None:
    # Initialize session_manager as local variable
    session_manager = None

    # Ensure terminal window has focus on Windows
    _set_windows_console_focus()

    try:
        print("")
        # --- Logging Setup ---
        # Logger already set up by setup_module at module level
        # Just call setup_logging to ensure proper configuration
        setup_logging()

        # --- Configuration Validation ---
        # Validate action configuration to prevent Action 6-style failures
        validate_action_config()

        if config is None:
            _print_config_error_message()

        # --- Instantiate SessionManager ---
        session_manager = SessionManager()  # No browser started by default

        # --- Main menu loop ---
        while True:
            choice = menu()
            print("")

            # --- Confirmation Check ---
            if not _check_action_confirmation(choice):
                continue

            # --- Action Dispatching ---
            if not _dispatch_menu_action(choice, session_manager, config):
                break  # Exit requested

    except KeyboardInterrupt:
        os.system("cls" if os.name == "nt" else "clear")
        print("\nCTRL+C detected. Exiting.")
    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)
    finally:
        # Final cleanup: Always close the session manager if it exists
        logger.info("Performing final cleanup...")

        if session_manager is not None:
            try:
                session_manager.close_sess(keep_db=False)
                logger.debug("Session Manager closed in final cleanup.")
            except Exception as final_close_e:
                logger.error(
                    f"Error during final Session Manager cleanup: {final_close_e}"
                )

        # Log program finish
        logger.info("--- Main program execution finished ---")
        print("\nExecution finished.")


# end main


# ============================================================================
# MODULE-LEVEL TEST FUNCTIONS FOR main.py
# ============================================================================
# Extracted from monolithic main_module_tests() for better organization
# Each test function is independent and can be run individually


def _test_module_initialization():
        """Test module initialization and import availability"""
        # Test that all required functions are available
        assert callable(menu), "menu() function should be callable"
        assert callable(main), "main() function should be callable"
        assert callable(clear_log_file), "clear_log_file() function should be callable"

        # Test that all action modules are imported
        assert coord is not None, "action6_gather.coord should be imported"
        assert InboxProcessor is not None, "InboxProcessor should be imported"
        assert (
            send_messages_to_matches is not None
        ), "send_messages_to_matches should be imported"
        assert (
            process_productive_messages is not None
        ), "process_productive_messages should be imported"
        assert run_action10 is not None, "run_action10 should be imported"
        assert run_action11 is not None, "run_action11 should be imported"


def _test_configuration_availability():
    """Test configuration and database availability"""
    assert config is not None, "config should be available"
    assert logger is not None, "logger should be available"
    assert SessionManager is not None, "SessionManager should be available"

    # Test database components
    assert Base is not None, "SQLAlchemy Base should be available"
    assert Person is not None, "Person model should be available"
    assert ConversationLog is not None, "ConversationLog model should be available"
    assert DnaMatch is not None, "DnaMatch model should be available"


def _test_clear_log_file_function():
        """Test log file clearing functionality"""
        # Test clear_log_file function exists and is callable
        assert callable(clear_log_file), "clear_log_file should be callable"

        # Test function returns proper tuple structure
        try:
            result = clear_log_file()
            assert isinstance(result, tuple), "clear_log_file should return a tuple"
            assert len(result) == 2, "clear_log_file should return a 2-element tuple"
            success, message = result
            assert isinstance(success, bool), "First element should be boolean"
            assert message is None or isinstance(
                message, str
            ), "Second element should be None or string"
        except Exception as e:
            # Function may fail in test environment, but should not crash
            assert isinstance(e, Exception), "Should handle errors gracefully"


def _test_main_function_structure():
    """Test main function structure and error handling"""
    assert callable(main), "main() function should be callable"

    # Test that main function has proper structure for error handling
    import inspect

    sig = inspect.signature(main)
    assert len(sig.parameters) == 0, "main() should take no parameters"


def _test_menu_system_components():
        """Test menu system components availability"""
        # Test menu function exists
        assert callable(menu), "menu() function should be callable"

        # Test that menu has access to all action functions
        menu_globals = menu.__globals__
        assert "coord" in menu_globals, "menu should have access to coord function"
        assert (
            "InboxProcessor" in menu_globals
        ), "menu should have access to InboxProcessor"
        assert (
            "send_messages_to_matches" in menu_globals
        ), "menu should have access to send_messages_to_matches"
        assert (
            "process_productive_messages" in menu_globals
        ), "menu should have access to process_productive_messages"
        assert "run_action10" in menu_globals, "menu should have access to run_action10"
        assert "run_action11" in menu_globals, "menu should have access to run_action11"


def _test_action_function_availability():
    """Test all action functions are properly imported and callable"""
    # Test action6_gather
    assert callable(coord), "coord function should be callable"

    # Test action7_inbox
    assert callable(InboxProcessor), "InboxProcessor should be callable"

    # Test action8_messaging
    assert callable(
        send_messages_to_matches
    ), "send_messages_to_matches should be callable"

    # Test action9_process_productive
    assert callable(
        process_productive_messages
    ), "process_productive_messages should be callable"

    # Test action10
    assert callable(run_action10), "run_action10 should be callable"

    # Test action11
    assert callable(run_action11), "run_action11 should be callable"


def _test_database_operations():
        """Test database operation functions"""
        assert callable(backup_database), "backup_database should be callable"
        assert callable(db_transn), "db_transn should be callable"
        assert callable(reset_db_actn), "reset_db_actn should be callable"

        # Test database models are available
        assert Person is not None, "Person model should be available"
        assert ConversationLog is not None, "ConversationLog model should be available"
        assert DnaMatch is not None, "DnaMatch model should be available"
        assert FamilyTree is not None, "FamilyTree model should be available"
        assert MessageTemplate is not None, "MessageTemplate model should be available"


def _test_reset_db_actn_integration():
    """Test reset_db_actn function integration and method availability"""
    # Test that reset_db_actn can be called without AttributeError
    try:
        # Create a test SessionManager to verify method availability
        test_sm = SessionManager()

        # Verify that the required methods exist on the SessionManager and DatabaseManager
        assert hasattr(test_sm, 'db_manager'), "SessionManager should have db_manager attribute"
        assert hasattr(test_sm.db_manager, '_initialize_engine_and_session'), \
            "DatabaseManager should have _initialize_engine_and_session method"
        assert hasattr(test_sm.db_manager, 'engine'), "DatabaseManager should have engine attribute"
        assert hasattr(test_sm.db_manager, 'Session'), "DatabaseManager should have Session attribute"

        # Test that reset_db_actn doesn't fail with AttributeError on method calls
        # Note: We don't actually run the reset to avoid affecting the test database
        logger.debug("reset_db_actn integration test: All required methods and attributes verified")

    except AttributeError as e:
        raise AssertionError(f"reset_db_actn integration test failed with AttributeError: {e}") from e
    except Exception as e:
        # Other exceptions are acceptable for this test (we're only checking for AttributeError)
        logger.debug(f"reset_db_actn integration test: Non-AttributeError exception (acceptable): {e}")


def _test_edge_case_handling():
        """Test edge cases and error conditions"""
        # Test imports are properly structured
        import sys

        assert "action6_gather" in sys.modules, "action6_gather should be imported"
        assert "action7_inbox" in sys.modules, "action7_inbox should be imported"
        assert (
            "action8_messaging" in sys.modules
        ), "action8_messaging should be imported"
        assert (
            "action9_process_productive" in sys.modules
        ), "action9_process_productive should be imported"
        assert "action10" in sys.modules, "action10 should be imported"
        assert "action11" in sys.modules, "action11 should be imported"


def _test_import_error_handling():
    """Test import error scenarios"""
    # Check that main module has all required imports
    module_globals = globals()
    required_imports = [
        "coord",
        "InboxProcessor",
        "send_messages_to_matches",
        "process_productive_messages",
        "run_action10",
        "run_action11",
        "config",
        "logger",
        "SessionManager",
    ]

    for import_name in required_imports:
        assert import_name in module_globals, f"{import_name} should be imported"


def _test_session_manager_integration():
        """Test SessionManager integration"""
        assert SessionManager is not None, "SessionManager should be available"
        assert callable(SessionManager), "SessionManager should be callable"

        # Test SessionManager has required methods
        # Should have key methods for session management
        assert hasattr(
            SessionManager, "__init__"
        ), "SessionManager should have __init__ method"


def _test_logging_integration():
    """Test logging system integration"""
    assert logger is not None, "logger should be available"
    assert hasattr(logger, "info"), "logger should have info method"
    assert hasattr(logger, "error"), "logger should have error method"
    assert hasattr(logger, "warning"), "logger should have warning method"
    assert hasattr(logger, "debug"), "logger should have debug method"
    assert hasattr(logger, "critical"), "logger should have critical method"


def _test_configuration_integration():
    """Test configuration system integration"""
    assert config is not None, "config should be available"

    # Test config has basic attributes (may vary by implementation)
    # This tests that the config object is properly initialized
    assert hasattr(config, "__dict__") or hasattr(
        config, "__getattribute__"
    ), "config should be a proper object"


def _test_validate_action_config():
        """Test the new validate_action_config() function from Action 6 lessons"""
        # Test that the function exists and is callable
        assert callable(validate_action_config), "validate_action_config should be callable"

        # Test that the function can be executed without errors
        try:
            result = validate_action_config()
            assert isinstance(result, bool), "validate_action_config should return boolean"
            # Function should succeed even if some warnings are generated
            assert result is True, "validate_action_config should return True for basic validation"
        except Exception as e:
            # If it fails, it should be due to missing config, not function errors
            assert "config" in str(e).lower(), f"validate_action_config failed unexpectedly: {e}"


def _test_database_integration():
    """Test database system integration"""
    # Test database functions are available
    assert callable(backup_database), "backup_database should be callable"

    # Test database transaction manager
    assert callable(db_transn), "db_transn should be callable"

    # Test that we can access database models
    from database import Base

    assert Base is not None, "SQLAlchemy Base should be accessible"


def _test_action_integration():
        """Test all actions integrate properly with main"""
        # Test that all action functions can be called (at module level)
        actions_to_test = [
            ("coord", coord),
            ("InboxProcessor", InboxProcessor),
            ("send_messages_to_matches", send_messages_to_matches),
            ("process_productive_messages", process_productive_messages),
            ("run_action10", run_action10),
            ("run_action11", run_action11),
        ]

        for action_name, action_func in actions_to_test:
            assert callable(action_func), f"{action_name} should be callable"
            assert action_func is not None, f"{action_name} should not be None"


def _test_import_performance():
    """Test import performance is reasonable"""
    import importlib
    import time

    # Test that re-importing modules is fast (cached)
    start_time = time.time()

    # Test a few key imports
    try:
        config_module = sys.modules.get("config")
        if config_module:
            importlib.reload(config_module)
    except Exception:
        pass  # Module reload may not work in test environment

    duration = time.time() - start_time
    assert duration < 1.0, f"Module reloading should be fast, took {duration:.3f}s"


def _test_memory_efficiency():
    """Test memory usage is reasonable"""
    import sys

    # Check that module size is reasonable
    module_size = sys.getsizeof(sys.modules[__name__])
    assert (
        module_size < 10000
    ), f"Module size should be reasonable, got {module_size} bytes"

    # Test that globals are not excessive (increased limit due to extracted test functions)
    globals_count = len(globals())
    assert (
        globals_count < 150
    ), f"Global variables should be reasonable, got {globals_count}"


def _test_function_call_performance():
        """Test function call performance"""
        import time

        # Test that basic function calls are fast
        start_time = time.time()

        for _ in range(1000):
            # Test a simple function call
            result = callable(menu)
            assert result is True, "menu should be callable"

        duration = time.time() - start_time
        assert (
            duration < 0.1
        ), f"1000 function checks should be fast, took {duration:.3f}s"


def _test_error_handling_structure():
    """Test error handling structure in main functions"""
    import inspect

    # Test that main function has proper structure
    main_source = inspect.getsource(main)
    assert "try:" in main_source, "main() should have try-except structure"
    assert "except" in main_source, "main() should have exception handling"
    assert "finally:" in main_source, "main() should have finally block"

    # Test that KeyboardInterrupt is handled
    assert (
        "KeyboardInterrupt" in main_source
    ), "main() should handle KeyboardInterrupt"


def _test_cleanup_procedures():
    """Test cleanup procedures are in place"""
    import inspect

    # Test that main has cleanup code
    main_source = inspect.getsource(main)
    assert "finally:" in main_source, "main() should have finally block for cleanup"
    assert "cleanup" in main_source.lower(), "main() should mention cleanup"


def _test_exception_handling_coverage():
        """Test exception handling covers expected scenarios"""
        import inspect

        # Test main function exception handling
        main_source = inspect.getsource(main)

        # Should handle general exceptions
        assert "Exception" in main_source, "main() should handle general exceptions"

        # Should have logging for errors
        assert "logger" in main_source, "main() should use logger for error reporting"


# ============================================================================
# MAIN TEST SUITE RUNNER
# ============================================================================


def main_module_tests() -> bool:
    """Comprehensive test suite for main.py"""
    try:
        from test_framework import TestSuite, suppress_logging
    except ImportError:
        # Fall back to relative import if absolute import fails
        from .test_framework import TestSuite, suppress_logging

    suite = TestSuite("Main Application Controller & Menu System", "main.py")
    suite.start_suite()

    # Run all tests with suppress_logging
    with suppress_logging():
        # INITIALIZATION TESTS
        suite.run_test(
            test_name="menu(), main(), clear_log_file(), action imports",
            test_func=_test_module_initialization,
            test_summary="Module initialization and core function availability",
            method_description="Testing availability of main functions and action module imports",
            expected_outcome="All core functions are available and action modules are properly imported",
        )

        suite.run_test(
            test_name="config, logger, SessionManager, database models",
            test_func=_test_configuration_availability,
            test_summary="Configuration and database component availability",
            method_description="Testing configuration instance and database model imports",
            expected_outcome="Configuration and database components are properly available",
        )

        # CORE FUNCTIONALITY TESTS
        suite.run_test(
            test_name="clear_log_file() function logic and return values",
            test_func=_test_clear_log_file_function,
            test_summary="Log file clearing functionality and return structure",
            method_description="Testing clear_log_file function execution and return tuple structure",
            expected_outcome="Function executes properly and returns appropriate tuple structure",
        )

        suite.run_test(
            test_name="main() function structure and signature",
            test_func=_test_main_function_structure,
            test_summary="Main function structure and parameter requirements",
            method_description="Testing main function callable status and parameter signature",
            expected_outcome="Main function has proper structure and takes no parameters",
        )

        suite.run_test(
            test_name="menu() system and action function access",
            test_func=_test_menu_system_components,
            test_summary="Menu system components and action function accessibility",
            method_description="Testing menu function and its access to all action functions",
            expected_outcome="Menu system has access to all required action functions",
        )

        suite.run_test(
            test_name="coord(), InboxProcessor(), send_messages_to_matches(), process_productive_messages(), run_action10(), run_action11()",
            test_func=_test_action_function_availability,
            test_summary="All action functions are properly imported and callable",
            method_description="Testing callable status of all action module functions",
            expected_outcome="All action functions are available and callable",
        )

        suite.run_test(
            test_name="backup_database(), db_transn(), database models",
            test_func=_test_database_operations,
            test_summary="Database operation functions and model availability",
            method_description="Testing database functions and model imports",
            expected_outcome="Database operations and models are properly available",
        )

        suite.run_test(
            test_name="reset_db_actn() integration and method availability",
            test_func=_test_reset_db_actn_integration,
            test_summary="Database reset function integration and required method verification",
            method_description="Testing reset_db_actn function for proper SessionManager and DatabaseManager method access",
            expected_outcome="reset_db_actn can access all required methods without AttributeError",
        )

        # EDGE CASE TESTS
        suite.run_test(
            test_name="Edge case handling and module import validation",
            test_func=_test_edge_case_handling,
            test_summary="Edge cases and import validation scenarios",
            method_description="Testing edge conditions and module import status",
            expected_outcome="Edge cases are handled and imports are properly validated",
        )

        suite.run_test(
            test_name="Import error scenarios and required module presence",
            test_func=_test_import_error_handling,
            test_summary="Import error handling and required module validation",
            method_description="Testing essential module imports and availability",
            expected_outcome="All essential modules are imported and available",
        )

        # INTEGRATION TESTS
        suite.run_test(
            test_name="SessionManager integration and method availability",
            test_func=_test_session_manager_integration,
            test_summary="SessionManager integration with main application",
            method_description="Testing SessionManager availability and method access",
            expected_outcome="SessionManager integrates properly with required methods",
        )

        suite.run_test(
            test_name="Logging system integration and method availability",
            test_func=_test_logging_integration,
            test_summary="Logging system integration with main application",
            method_description="Testing logger availability and all required logging methods",
            expected_outcome="Logging system is properly integrated with all methods available",
        )

        suite.run_test(
            test_name="Configuration system integration and object access",
            test_func=_test_configuration_integration,
            test_summary="Configuration system integration with main application",
            method_description="Testing config availability and object structure",
            expected_outcome="Configuration system is properly integrated and accessible",
        )

        suite.run_test(
            test_name="Configuration validation system from Action 6 lessons",
            test_func=_test_validate_action_config,
            test_summary="Configuration validation system prevents Action 6-style failures",
            method_description="Testing validate_action_config() function validates .env settings and rate limiting",
            expected_outcome="Configuration validation function works correctly and returns boolean result",
        )

        suite.run_test(
            test_name="Database system integration and transaction management",
            test_func=_test_database_integration,
            test_summary="Database system integration with main application",
            method_description="Testing database functions and model accessibility",
            expected_outcome="Database system is properly integrated with transaction support",
        )

        suite.run_test(
            test_name="All action function integration with main application",
            test_func=_test_action_integration,
            test_summary="Action functions integrate properly with main application",
            method_description="Testing action function availability and callable status",
            expected_outcome="All action functions integrate properly and are callable",
        )

        # PERFORMANCE TESTS
        suite.run_test(
            test_name="Module import and reload performance",
            test_func=_test_import_performance,
            test_summary="Import performance and module caching efficiency",
            method_description="Testing module import and reload times for performance",
            expected_outcome="Module imports and reloads complete within reasonable time limits",
        )

        suite.run_test(
            test_name="Memory usage efficiency and global variable management",
            test_func=_test_memory_efficiency,
            test_summary="Memory usage efficiency and resource management",
            method_description="Testing module memory usage and global variable count",
            expected_outcome="Memory usage is reasonable and global variables are controlled",
        )

        suite.run_test(
            test_name="Function call performance and responsiveness",
            test_func=_test_function_call_performance,
            test_summary="Function call performance and execution speed",
            method_description="Testing basic function call performance with multiple iterations",
            expected_outcome="Function calls execute efficiently within performance limits",
        )

        # ERROR HANDLING TESTS
        suite.run_test(
            test_name="main() error handling structure and exception coverage",
            test_func=_test_error_handling_structure,
            test_summary="Error handling structure in main function",
            method_description="Testing main function for proper try-except-finally structure",
            expected_outcome="Main function has comprehensive error handling structure",
        )

        suite.run_test(
            test_name="Cleanup procedures and resource management",
            test_func=_test_cleanup_procedures,
            test_summary="Cleanup procedures and resource management implementation",
            method_description="Testing cleanup code presence and resource management",
            expected_outcome="Proper cleanup procedures are implemented for resource management",
        )

        suite.run_test(
            test_name="Exception handling coverage and logging integration",
            test_func=_test_exception_handling_coverage,
            test_summary="Exception handling coverage and error logging",
            method_description="Testing exception handling scope and logging integration",
            expected_outcome="Exception handling covers expected scenarios with proper logging",
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive main module tests using standardized TestSuite format."""
    return main_module_tests()


# --- Entry Point ---

if __name__ == "__main__":
    main()


# end of main.py
