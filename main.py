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
from typing import Any, Callable, Optional
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
    if config.api.max_pages <= 0:
        logger.warning("MAX_PAGES not set or invalid - actions may process unlimited pages")
    if config.batch_size <= 0:
        logger.warning("BATCH_SIZE not set or invalid - actions may use large batches")
    if config.max_productive_to_process <= 0:
        logger.warning("MAX_PRODUCTIVE_TO_PROCESS not set - actions may process unlimited items")
    if config.max_inbox <= 0:
        logger.warning("MAX_INBOX not set - actions may process unlimited inbox items")


def _check_rate_limiting_settings(config: Any) -> None:
    """Check rate limiting settings and log warnings."""
    if config.api.requests_per_second > 1.0:
        logger.warning(f"requests_per_second ({config.api.requests_per_second}) may be too aggressive - consider ≤1.0")
    if config.api.retry_backoff_factor < 2.0:
        logger.warning(f"retry_backoff_factor ({config.api.retry_backoff_factor}) may be too low - consider ≥2.0")
    if config.api.initial_delay < 1.0:
        logger.warning(f"initial_delay ({config.api.initial_delay}) may be too short - consider ≥1.0")


def _log_configuration_summary(config: Any) -> None:
    """Log current configuration for transparency (debug level for clean startup)."""
    logger.debug("=== ACTION CONFIGURATION VALIDATION ===")
    logger.debug(f"MAX_PAGES: {config.api.max_pages}")
    logger.debug(f"BATCH_SIZE: {config.batch_size}")
    logger.debug(f"MAX_PRODUCTIVE_TO_PROCESS: {config.max_productive_to_process}")
    logger.debug(f"MAX_INBOX: {config.max_inbox}")
    logger.debug(
        f"⚡ Rate Limiting Config - Workers: {config.api.thread_pool_workers}, "
        f"RPS: {config.api.requests_per_second}, InitialDelay: {config.api.initial_delay}s, "
        f"MaxDelay: {config.api.max_delay}s, Backoff: {config.api.backoff_factor}"
    )
    logger.debug("========================================")


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
    print("6. Gather Matches [start page]")
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
        result = session_manager.ensure_db_ready()
        if not result:
            logger.error(f"Failed to ensure database ready for {action_name}")
        return result

    if required_state == "driver_ready":
        result = session_manager.browser_manager.ensure_driver_live(f"{action_name} - Browser Start")
        if not result:
            logger.error(f"Failed to ensure driver live for {action_name}")
            print(f"\n✗ Failed to start browser for action: {action_name}")
            print("  Please check the log file for detailed error messages.")
        return result

    if required_state == "session_ready":
        skip_csrf = (choice == "11")
        result = session_manager.ensure_session_ready(action_name=f"{action_name} - Setup", skip_csrf=skip_csrf)
        if not result:
            logger.error(f"Failed to ensure session ready for {action_name}")
        return result

    return True


def _prepare_action_arguments(action_func: Callable, session_manager: SessionManager, args: tuple) -> tuple:
    """Prepare arguments for action function call."""
    func_sig = inspect.signature(action_func)
    pass_session_manager = "session_manager" in func_sig.parameters
    action_name = action_func.__name__

    # Handle keyword args specifically for coord function
    if action_name in ["coord", "coord_action"] and "start" in func_sig.parameters:
        # Extract start value, preserving None for checkpoint auto-resume
        start_val = None
        config_arg = None
        
        for arg in args:
            if isinstance(arg, int):
                start_val = arg
            elif arg is not None and not isinstance(arg, int):
                # First non-integer, non-None arg is config_schema
                if config_arg is None:
                    config_arg = arg
        
        # If no start_val found in args, use default of 1 (not None)
        # This maintains backward compatibility when no args are passed
        if start_val is None and None not in args:
            start_val = 1
        
        kwargs_for_action = {"start": start_val}

        coord_args = []
        if pass_session_manager:
            coord_args.append(session_manager)
        if action_name == "coord_action" and "config_schema" in func_sig.parameters:
            # Use the config_schema passed in args, or fall back to global config
            coord_args.append(config_arg if config_arg is not None else config)

        return coord_args, kwargs_for_action
    # General case
    final_args = []
    if pass_session_manager:
        final_args.append(session_manager)
    final_args.extend(args)
    return final_args, {}


def _execute_action_function(action_func: Callable, prepared_args: tuple, kwargs: dict) -> Any:
    """Execute the action function with prepared arguments."""
    if kwargs:
        return action_func(*prepared_args, **kwargs)
    return action_func(*prepared_args)


def _should_close_session(action_result: Any, action_exception: Optional[Exception], close_sess_after: bool, action_name: str) -> bool:
    """Determine if session should be closed."""
    if action_result is False or action_exception is not None:
        logger.warning(f"Action '{action_name}' failed or raised exception. Closing session.")
        return True
    if close_sess_after:
        logger.debug(f"Closing session after '{action_name}' as requested by caller (close_sess_after=True).")
        return True
    return False


def _log_performance_metrics(start_time: float, process: psutil.Process, mem_before: float, choice: str, action_name: str) -> None:
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


def _perform_session_cleanup(session_manager: SessionManager, should_close: bool, action_name: str) -> None:
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
    action_func: Callable,
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
def _delete_table_records(sess: Any, table_class: Any, filter_condition: Any, table_name: str, person_id_to_keep: int) -> int:
    """Delete records from a table based on filter condition."""
    logger.debug(f"Deleting from {table_name} where people_id != {person_id_to_keep}...")
    result = sess.query(table_class).filter(filter_condition).delete(synchronize_session=False)
    count = result if result is not None else 0
    logger.info(f"Deleted {count} {table_name} records.")
    return count


def _perform_deletions(sess: Any, person_id_to_keep: int) -> dict:
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


def all_but_first_actn(session_manager: SessionManager, *_) -> bool:
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

def _run_action6_gather(session_manager: SessionManager) -> bool:
    """Run Action 6: Gather Matches."""
    logger.info("--- Running Action 6: Gather Matches (Always from page 1) ---")
    gather_result = coord_action(session_manager, config, start=1)
    if gather_result is False:
        logger.error("Action 6 FAILED.")
        return False
    logger.info("Action 6 OK.")
    print("✓ Match gathering completed successfully.")
    return True


def _run_action7_inbox(session_manager: SessionManager) -> bool:
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
            return False

        logger.debug("Navigation to inbox page successful.")
        time.sleep(2)

        logger.debug("Running inbox search...")
        inbox_processor = InboxProcessor(session_manager=session_manager)
        search_result = inbox_processor.search_inbox()

        if search_result is False:
            logger.error("Action 7 FAILED - Inbox search returned failure.")
            return False

        logger.info("Action 7 OK.")
        print("✓ Inbox search completed successfully.")
        return True

    except Exception as inbox_error:
        logger.error(f"Action 7 FAILED with exception: {inbox_error}", exc_info=True)
        return False


def _run_action9_process_productive(session_manager: SessionManager) -> bool:
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
            return False

        logger.debug("Navigation to base URL successful. Processing productive messages...")
        time.sleep(2)

        process_result = process_productive_messages(session_manager)

        if process_result is False:
            logger.error("Action 9 FAILED - Productive message processing returned failure.")
            return False

        logger.info("Action 9 OK.")
        print("✓ Productive message processing completed successfully.")
        return True

    except Exception as process_error:
        logger.error(f"Action 9 FAILED with exception: {process_error}", exc_info=True)
        return False


def _run_action8_send_messages(session_manager: SessionManager) -> bool:
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
            return False

        logger.debug("Navigation to base URL successful. Sending messages...")
        time.sleep(2)

        send_result = send_messages_to_matches(session_manager)

        if send_result is False:
            logger.error("Action 8 FAILED - Message sending returned failure.")
            return False

        logger.info("Action 8 OK.")
        print("✓ Message sending completed successfully.")
        return True

    except Exception as message_error:
        logger.error(f"Action 8 FAILED with exception: {message_error}", exc_info=True)
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

    return result


# End Action 1


# Action 2 (reset_db_actn)
def _truncate_all_tables_direct(session: Any, db_manager: Any) -> bool:
    """Truncate all tables in the database using a direct session."""
    try:
        with db_transn(session) as sess:
            # Delete all records from tables in reverse order of dependencies
            # Use a try-except for each table in case it doesn't exist
            from sqlalchemy.exc import OperationalError

            tables_to_truncate = [
                (ConversationLog, "conversation_log"),
                (DnaMatch, "dna_match"),
                (FamilyTree, "family_tree"),
                (Person, "people")
            ]

            for table_class, table_name in tables_to_truncate:
                try:
                    deleted_count = sess.query(table_class).delete(synchronize_session=False)
                    logger.debug(f"Truncated {table_name}: {deleted_count} rows deleted")
                except OperationalError as op_err:
                    if "no such table" in str(op_err):
                        logger.debug(f"Table {table_name} does not exist, skipping truncation")
                    else:
                        raise
            # Keep MessageType table intact
        db_manager.return_session(session)
        logger.debug("All tables truncated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error truncating tables: {e}", exc_info=True)
        db_manager.return_session(session)
        return False


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
            # Use a try-except for each table in case it doesn't exist
            from sqlalchemy.exc import OperationalError

            tables_to_truncate = [
                (ConversationLog, "conversation_log"),
                (DnaMatch, "dna_match"),
                (FamilyTree, "family_tree"),
                (Person, "people")
            ]

            for table_class, table_name in tables_to_truncate:
                try:
                    deleted_count = sess.query(table_class).delete(synchronize_session=False)
                    logger.debug(f"Truncated {table_name}: {deleted_count} rows deleted")
                except OperationalError as op_err:
                    if "no such table" in str(op_err):
                        logger.debug(f"Table {table_name} does not exist, skipping truncation")
                    else:
                        raise
            # Keep MessageType table intact
        temp_manager.return_session(truncate_session)
        logger.debug("All tables truncated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error truncating tables: {e}", exc_info=True)
        temp_manager.return_session(truncate_session)
        return False


def _reinitialize_database_schema_direct(db_manager: Any) -> bool:
    """Re-initialize database schema using DatabaseManager directly."""
    logger.debug("Re-initializing database schema...")
    try:
        # Re-initialize engine and session if needed
        db_manager._initialize_engine_and_session()
        if not db_manager.engine or not db_manager.Session:
            raise SQLAlchemyError("Failed to initialize DB engine/session for recreation!")

        # This will recreate the tables
        Base.metadata.create_all(db_manager.engine)
        logger.debug("Database schema recreated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error reinitializing database schema: {e}", exc_info=True)
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
                # Note: MessageTemplate requires template_key, message_content, template_category, and tree_status
                templates_to_add = []
                for template_key, template_content in messages_data.items():
                    # Parse template key: "In_Tree-Initial" -> category="In_Tree", tree_status="in_tree"
                    parts = template_key.split('-')
                    tree_status_prefix = parts[0] if parts else "Universal"

                    # Determine tree_status from the prefix (In_Tree -> in_tree, Out_Tree -> out_tree)
                    tree_status = tree_status_prefix.lower().replace('_', '_') if tree_status_prefix else "universal"

                    # Category is the message type (e.g., "Initial", "Follow-up", etc.)
                    category = parts[1] if len(parts) > 1 else "initial"

                    templates_to_add.append(MessageTemplate(
                        template_key=template_key,
                        message_content=template_content if isinstance(template_content, str) else str(template_content),
                        template_category=category,
                        tree_status=tree_status
                    ))

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


def _close_main_db_pool(session_manager: SessionManager) -> None:
    """Close main database pool and force garbage collection."""
    if session_manager:
        logger.debug("Closing main DB connections before database deletion...")
        session_manager.cls_db_conn(keep_db=False)
        logger.debug("Main DB pool closed.")

    # Force garbage collection to release any file handles
    logger.debug("Running garbage collection to release file handles...")
    gc.collect()
    time.sleep(1.0)
    gc.collect()


def _perform_database_reset_operations(temp_db_manager: Any) -> tuple[bool, Any]:
    """
    Perform database reset operations: truncate, reinitialize, and seed.

    Returns:
        Tuple of (success, recreation_session)
    """
    recreation_session = None

    try:
        # Ensure database is ready
        if not temp_db_manager.ensure_ready():
            logger.error("Failed to ensure temporary database manager ready")
            return False, None

        # Step 1: Truncate all tables
        logger.debug("Truncating all tables...")
        truncate_session = temp_db_manager.get_session()
        if not truncate_session:
            logger.critical("Failed to get session for truncating tables. Reset aborted.")
            return False, None

        if not _truncate_all_tables_direct(truncate_session, temp_db_manager):
            return False, None

        # Step 2: Re-initialize database schema
        if not _reinitialize_database_schema_direct(temp_db_manager):
            return False, None

        # Step 3: Seed MessageType Table
        recreation_session = temp_db_manager.get_session()
        if not recreation_session:
            raise SQLAlchemyError("Failed to get session for seeding MessageTypes!")

        _seed_message_templates(recreation_session)
        logger.info("Database reset completed successfully.")
        return True, recreation_session

    except Exception as recreate_err:
        logger.error(f"Error during DB recreation/seeding: {recreate_err}", exc_info=True)
        return False, recreation_session


def _cleanup_temp_db_manager(temp_db_manager: Any, recreation_session: Any) -> None:
    """Clean up the temporary database manager and its engine."""
    logger.debug("Cleaning up temporary database manager for reset...")
    if temp_db_manager:
        if recreation_session:
            temp_db_manager.return_session(recreation_session)
        temp_db_manager.close_connections()
    logger.debug("Temporary database manager cleanup finished.")


def reset_db_actn(session_manager: SessionManager, *_) -> bool:
    """
    Action to COMPLETELY reset the database by deleting the file. Browserless.
    - Closes main pool.
    - Deletes the .db file.
    - Recreates schema from scratch.
    - Seeds the MessageType table.
    """
    db_path = config.database.database_file

    if db_path is None:
        logger.critical("DATABASE_FILE is not configured. Reset aborted.")
        return False

    try:
        # Close main pool
        _close_main_db_pool(session_manager)

        # Create temporary database manager
        logger.debug(f"Attempting to reset database file: {db_path}...")
        logger.debug("Creating temporary database manager for database reset...")
        from core.database_manager import DatabaseManager
        temp_db_manager = DatabaseManager(str(db_path))

        # Perform reset operations
        reset_successful, recreation_session = _perform_database_reset_operations(temp_db_manager)

        # Cleanup
        _cleanup_temp_db_manager(temp_db_manager, recreation_session)

        return reset_successful

    except Exception as e:
        logger.error(f"Outer error during DB reset action: {e}", exc_info=True)
        return False

    finally:
        logger.debug("Reset DB action finished.")


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
def restore_db_actn(session_manager: SessionManager, *_) -> bool:  # Added session_manager back
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
    print("\n✗ You are NOT currently logged in to Ancestry.\n\n")
    print("  Attempting to log in with stored credentials...")
    logger.debug("Skipping redundant API verification - already confirmed not logged in")

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
        print("  You can update credentials using the 'sec' option in the main menu.")
        return False


# Action 5 (check_login_actn)
def check_login_actn(session_manager: SessionManager, *_) -> bool:
    """
    REVISED V13: Checks login status and attempts login if needed.
    This action starts a browser session and checks login status.
    If not logged in, it attempts to log in using stored credentials.
    Provides clear user feedback about the final login state.

    Note: Browser startup is handled by exec_actn based on _determine_required_state
    returning "driver_ready" for this action. This ensures browser is live before
    this function executes.
    """
    # Driver should already be live (started by exec_actn's _ensure_required_state)
    # But add defensive check with better error message
    if not session_manager.driver_live:
        logger.error("Driver not live after setup. This indicates browser startup failed.")
        print("  Please check:")
        print("  1. ChromeDriver is properly installed")
        print("  2. Chrome browser is installed and up-to-date")
        print("  3. No firewall blocking browser startup")
        print("  4. Check the log file for detailed error messages")
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
        print("  This may indicate a browser or network issue.")
        return False


# End Action 5


# Action 6 (coord_action wrapper)
def coord_action(session_manager: SessionManager, config_schema: Optional[Any] = None, start: int = 1) -> bool:
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
        return False

    print(f"Gathering DNA Matches from page {start}...")
    try:
        # Call the imported function from action6
        result = coord(session_manager, config_schema, start=start)
        if result is False:
            logger.error("Match gathering reported failure.")
            return False
        logger.info("Gathering matches OK.")
        print("✓ Match gathering completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error during coord_action: {e}", exc_info=True)
        return False


# End of coord_action


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
    print("   Refer to README.md (Security setup & credential management)")

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
    """Run Main.py Internal Tests - Tests moved to external test suite."""
    print("\n" + "=" * 60)
    print("MAIN.PY TESTS")
    print("=" * 60)
    print("\n📋 Note: main.py tests have been removed to prevent recursion.")
    print("   main.py is the application entry point and should not test itself.")
    print("   All other modules (58 modules) are tested by run_all_tests.py")
    print("\n   To run all module tests, use menu option 'testall' or run:")
    print("   python run_all_tests.py")
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

            if os.name != "nt":  # Not Windows
                print("\n  For Linux/macOS users, you may also need:")
                print("     pip install keyrings.alt")
                print(
                    "     Some Linux distributions may require: sudo apt-get install python3-dbus"
                )

            print("\n📚 For more information, review:")
            print("  - README.md (Security setup & credential management)")
            print("  - Run: python credentials.py --interactive for guided setup")
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
    print("\nReturning to main menu...")
    input("Press Enter to continue...")


def _show_cache_statistics() -> None:
    """Show cache statistics."""
    try:
        logger.info("Cache statistics feature currently unavailable")
        print("Cache statistics feature currently unavailable.")
    except Exception as e:
        logger.error(f"Error displaying cache statistics: {e}")


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
    # Use None to indicate "auto-resume from checkpoint if available"
    # Only use explicit page number when user provides it
    start_val = None
    if len(parts) > 1:
        try:
            start_arg = int(parts[1])
            start_val = start_arg if start_arg > 0 else 1
            print(f"Starting DNA match gathering from page {start_val}...")
        except ValueError:
            logger.warning(f"Invalid start page '{parts[1]}'. Using auto-resume.")
            print(f"Invalid start page '{parts[1]}'. Will auto-resume from checkpoint if available.")
    else:
        print("Starting DNA match gathering (will auto-resume from checkpoint if available)...")

    exec_actn(coord_action, session_manager, "6", False, config, start_val)
    return True


def _handle_browser_actions(choice: str, session_manager: Any, config: Any) -> bool:
    """Handle browser-required actions."""
    result = False

    if choice == "1":
        ensure_caching_initialized()
        exec_actn(run_core_workflow_action, session_manager, choice, close_sess_after=True)
        result = True
    elif choice == "5":
        exec_actn(check_login_actn, session_manager, choice, close_sess_after=True)
        result = True
    elif choice.startswith("6"):
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

    # Prevent system sleep during entire session
    from utils import prevent_system_sleep, restore_system_sleep
    sleep_state = prevent_system_sleep()

    try:
        print("")
        # --- Logging Setup ---
        # Logger already set up by setup_module at module level
        # Use INFO for clean startup, DEBUG for detailed troubleshooting
        setup_logging(log_level="INFO")

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
        # Restore normal sleep behavior
        restore_system_sleep(sleep_state)

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


# --- Entry Point ---

if __name__ == "__main__":
    main()


# end of main.py
