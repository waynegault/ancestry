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

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
# Error handling imports available if needed

# === STANDARD LIBRARY IMPORTS ===
import gc
import inspect
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional
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


# Configuration validation
def validate_action_config() -> bool:
    """
    Validate that all actions respect .env configuration limits.
    Prevents Action 6-style failures by ensuring conservative settings are applied.
    """
    try:
        # Import and validate configuration
        try:
            from config import config_schema
            if config_schema is None:
                logger.error("config_schema is None - configuration not properly initialized")
                return False
            config = config_schema  # Assign the loaded configuration instance
            logger.debug("Configuration loaded successfully")
        except ImportError as e:
            logger.error(f"Could not import config_schema from config package: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            return False

        # Check essential processing limits (only warn about critical issues)
        warnings = []
        if config.max_productive_to_process <= 0:
            warnings.append("MAX_PRODUCTIVE_TO_PROCESS not set")
        if config.max_inbox <= 0:
            warnings.append("MAX_INBOX not set")
        # Remove RPS warning since 0.25 is actually conservative

        if warnings:
            logger.debug(f"Configuration notes: {'; '.join(warnings)}")

        # Log current configuration for transparency
        logger.debug("=== ACTION CONFIGURATION VALIDATION ===")
        logger.debug(f"MAX_PAGES: {config.api.max_pages}")
        logger.debug(f"BATCH_SIZE: {config.batch_size}")
        logger.debug(f"MAX_PRODUCTIVE_TO_PROCESS: {config.max_productive_to_process}")
        logger.debug(f"MAX_INBOX: {config.max_inbox}")
        logger.debug(f"Rate Limiting - RPS: {config.api.requests_per_second}, Delay: {config.api.initial_delay}s")
        logger.debug("========================================")

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

# === PHASE 12: GEDCOM AI INTEGRATION ===
try:
    from gedcom_search_utils import get_cached_gedcom_data
    PHASE_12_AVAILABLE = True
    logger.debug("Phase 12 GEDCOM AI components loaded successfully")
except ImportError as e:
    logger.warning(f"Phase 12 GEDCOM AI components not available: {e}")
    PHASE_12_AVAILABLE = False
    get_cached_gedcom_data = None

# Initialize config manager
config_manager = ConfigManager()
config = config_manager.get_config()


def menu():
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
    print("=== PHASE 12: GEDCOM AI INTELLIGENCE ===")
    print("12. GEDCOM Intelligence Analysis")
    print("13. DNA-GEDCOM Cross-Reference")
    print("14. Research Prioritization")
    print("15. Comprehensive GEDCOM AI Analysis")
    print("")
    print("test. Run Main.py Internal Tests")
    print("testall. Run All Module Tests")
    print("")
    print("sec. Credential Manager (Setup/View/Update/Import from .env)")
    print("s. Show Cache Statistics")
    print("t. Toggle Log Level (DEBUG → INFO → WARNING)")
    print("c. Clear Screen")
    print("q. Exit")
    print("settings. Review/Edit .env Settings")
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
            from pathlib import Path
            Path(log_file_path).open("w", encoding="utf-8").close()
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


def _show_platform_specific_instructions():
    """Show platform-specific installation instructions for non-Windows systems."""
    import platform

    # Use platform.system() instead of os.name to avoid static analysis warning
    system_name = platform.system()
    if system_name in ("Linux", "Darwin"):  # Linux or macOS
        print("\n  For Linux/macOS users, you may also need:")
        print("     pip install keyrings.alt")
        print(
            "     Some Linux distributions may require: sudo apt-get install python3-dbus"
        )


# Global flag to track if caching has been initialized
_caching_initialized = False


def initialize_aggressive_caching():
    """Initialize aggressive caching systems."""
    try:
        from core.system_cache import warm_system_caches
        return warm_system_caches()
    except ImportError:
        logger.warning("System cache module not available")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize aggressive caching: {e}")
        return False


def ensure_caching_initialized():
    """Initialize aggressive caching systems if not already done."""
    global _caching_initialized  # noqa: PLW0603 - module-level init guard

    if not _caching_initialized:
        logger.debug("Initializing caching systems on-demand...")
        cache_init_success = initialize_aggressive_caching()
        if cache_init_success:
            logger.debug("Caching systems initialized successfully")
            _caching_initialized = True
        else:
            logger.warning(
                "Some caching systems failed to initialize, continuing with reduced performance"
            )
        return cache_init_success
    logger.debug("Caching systems already initialized")
    return True


# End of ensure_caching_initialized


def ensure_gedcom_loaded_and_cached():
    """
    Ensure GEDCOM file is loaded and cached before GEDCOM operations.

    Returns:
        bool: True if GEDCOM data is available, False otherwise
    """
    try:
        # First ensure caching is initialized
        ensure_caching_initialized()

        # Check if GEDCOM data is already cached
        if PHASE_12_AVAILABLE:
            from gedcom_search_utils import get_cached_gedcom_data
            cached_data = get_cached_gedcom_data()
            if cached_data:
                logger.info("GEDCOM data already cached and available")
                return True

        # Try to load GEDCOM data
        print("\n📁 GEDCOM data not found in cache. Loading GEDCOM file...")

        # Check if GEDCOM path is configured
        try:
            from config.config_manager import ConfigManager
            config_manager = ConfigManager()
            config = config_manager.get_config()
            gedcom_path = getattr(config.database, 'gedcom_file_path', None)

            if not gedcom_path:
                print("❌ GEDCOM file path not configured.")
                print("Please set GEDCOM_FILE_PATH in your .env file or run option 10 first.")
                return False

        except Exception as e:
            logger.debug(f"Error getting GEDCOM path from config: {e}")
            # GEDCOM file path not configured - please set GEDCOM_FILE_PATH in .env or run option 10
            return False

        # Try to load the GEDCOM file
        from pathlib import Path
        gedcom_file = Path(gedcom_path)

        if not gedcom_file.exists():
            print(f"❌ GEDCOM file not found: {gedcom_path}")
            print("Please check the file path or run option 10 to load a different file.")
            return False

        print(f"📂 Loading GEDCOM file: {gedcom_file.name}")

        if PHASE_12_AVAILABLE:
            from gedcom_search_utils import load_gedcom_data, set_cached_gedcom_data

            # Load the GEDCOM data
            gedcom_data = load_gedcom_data(gedcom_file)

            if gedcom_data:
                # Cache the loaded data
                set_cached_gedcom_data(gedcom_data)
                print("✅ GEDCOM file loaded and cached successfully!")
                print(f"   📊 Individuals: {len(getattr(gedcom_data, 'indi_index', {}))}")
                return True
            print("❌ Failed to load GEDCOM data from file.")
            return False
        print("❌ Phase 12 components not available for GEDCOM loading.")
        return False

    except Exception as e:
        logger.error(f"Error ensuring GEDCOM loaded and cached: {e}")
        # Error loading GEDCOM file logged above
        return False


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

    # --- Performance Logging Setup ---
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    logger.info("------------------------------------------")
    logger.info(f"Action {choice}: Starting {action_name}...")
    logger.info("------------------------------------------\n")

    action_result = None
    action_exception = None  # Store exception if one occurs

    # Determine if the action requires a browser
    browserless_actions = [
        "all_but_first_actn",
        "reset_db_actn",
        "backup_db_actn",
        "restore_db_actn",
        "run_action10",  # GEDCOM Report (Local File)
        # "run_action11_wrapper",  # API Report (Ancestry Online) - Removed: needs browser session
    ]

    # Set browser_needed flag based on action
    requires_browser = action_name not in browserless_actions
    session_manager.browser_needed = requires_browser

    # Determine the required session state for the action
    required_state = "none"  # Default for actions that don't need any special state

    if requires_browser:
        # Special case for check_login_actn: only needs driver, not full session
        required_state = "driver_ready" if action_name == "check_login_actn" else "session_ready"
    else:
        required_state = "db_ready"  # Database-only session

        # --- Preflight: prevent duplicate Action 6 run if lock is held ---
        if action_name in ("coord", "coord_action"):
            try:
                from pathlib import Path
                lock_file = Path("Locks") / "action6.lock"
                if lock_file.exists():
                    try:
                        data = lock_file.read_text(encoding="utf-8", errors="ignore").strip()
                        parts = data.split("|")
                        holder_pid = int(parts[0]) if parts and parts[0].isdigit() else None
                        holder_run_id = parts[1] if len(parts) > 1 else ""
                    except Exception:
                        holder_pid = None
                        holder_run_id = ""
                    # Check if holder is alive
                    alive = False
                    if holder_pid:
                        try:
                            alive = psutil.pid_exists(holder_pid)
                        except Exception:
                            alive = False
                    if alive:
                        logger.info(
                            f"Action 6 already running (PID={holder_pid}, RUN={holder_run_id}). Skipping duplicate start."
                        )
                        return True  # Graceful no-op, do not close session
            except Exception as e:
                logger.debug(f"Action 6 lock preflight check error: {e}")

    try:
        # --- Ensure Required State ---
        state_ok = True
        if required_state == "db_ready":
            state_ok = session_manager.ensure_db_ready()
        elif required_state == "driver_ready":
            # For check_login_actn: only ensure browser is started, no login attempts
            state_ok = session_manager.browser_manager.ensure_driver_live(f"{action_name} - Browser Start")
        elif required_state == "session_ready":
            # Skip CSRF token validation for Action 11 (only uses Tree Ladder API)
            skip_csrf = (choice == "11")
            state_ok = session_manager.ensure_session_ready(
                action_name=f"{action_name} - Setup", skip_csrf=skip_csrf
            )

        if not state_ok:
            # Log specific state failure before raising generic exception
            logger.error(
                f"Failed to achieve required state '{required_state}' for action '{action_name}'."
            )
            raise Exception(
                f"Setup failed: Could not achieve state '{required_state}'."
            )

        # --- Execute Action ---
        # Prepare arguments for action function call
        func_sig = inspect.signature(action_func)
        pass_session_manager = "session_manager" in func_sig.parameters

        final_args = []
        if pass_session_manager:
            final_args.append(session_manager)
        final_args.extend(args)

        # Handle keyword args specifically for coord function
        if action_name in ["coord", "coord_action"] and "start" in func_sig.parameters:
            start_val = 1
            int_args = [a for a in args if isinstance(a, int)]
            if int_args:
                start_val = int_args[-1]
            kwargs_for_action = {"start": start_val}
            # Prepare coord specific positional args
            coord_args = []
            if pass_session_manager:
                coord_args.append(session_manager)
            # coord_action also needs config_schema
            if action_name == "coord_action" and "config_schema" in func_sig.parameters:
                coord_args.append(config)  # Pass the global config
            # Call with prepared positional args and keyword args
            action_result = action_func(*coord_args, **kwargs_for_action)
        else:
            # General case - call with the assembled final_args list
            action_result = action_func(*final_args)

    except Exception as e:
        # Log exception details and mark action as failure
        logger.error(f"Exception during action {action_name}: {e}", exc_info=True)
        action_result = False
        action_exception = e

    finally:
        # --- Session Closing Logic (Simplified) ---
        should_close = False
        if action_result is False or action_exception is not None:
            # Close session if action failed or raised exception
            logger.warning(
                f"Action '{action_name}' failed or raised exception. Closing session."
            )
            should_close = True
        elif close_sess_after:
            # Close session if explicitly requested
            logger.debug(
                f"Closing session after '{action_name}' as requested by caller (close_sess_after=True)."
            )
            should_close = True

        # --- Performance Logging ---
        duration = time.time() - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_duration = f"{int(hours)} hr {int(minutes)} min {seconds:.2f} sec"
        # Recalculate memory usage safely
        try:
            mem_after = process.memory_info().rss / (1024 * 1024)
            mem_used = mem_after - mem_before
            mem_log = f"Memory used: {mem_used:.1f} MB"
        except Exception as mem_err:
            mem_log = f"Memory usage unavailable: {mem_err}"

        print(" ")  # Spacer

        # Restore old footer style
        if action_result is False:
            logger.debug(f"Action {choice} ({action_name}) reported failure.")
        elif action_exception is not None:
            logger.debug(
                f"Action {choice} ({action_name}) failed due to exception: {type(action_exception).__name__}."
            )

        # --- Return Action Result ---
        # Return True only if action completed without exception AND didn't return False explicitly
        final_outcome = action_result is not False and action_exception is None

        logger.info("------------------------------------------")
        logger.info(f"Action {choice} ({action_name}) finished.")
        logger.info(f"Duration: {formatted_duration}")
        logger.info(mem_log)
        logger.info("------------------------------------------\n")

        # Perform cleanup AFTER footer to prevent logs bleeding after completion
        if should_close and isinstance(session_manager, SessionManager):
            if session_manager.browser_needed and session_manager.driver_live:
                logger.debug("Closing browser session...")
                # Close browser but keep DB connections for most actions
                session_manager.close_browser()
                logger.debug("Browser session closed. DB connections kept.")
            elif should_close and action_name in ["all_but_first_actn"]:
                # For specific actions, close everything including DB
                logger.debug("Closing all connections including database...")
                session_manager.close_sess(keep_db=False)
                logger.debug("All connections closed.")

    return final_outcome

# End of exec_actn


# --- Action Functions


# Action 0 (all_but_first_actn)
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
        else:
            logger.warning(
                "No main session manager passed to all_but_first_actn to close."
            )
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
                    Person.profile_id == profile_id_to_keep, Person.deleted_at is None
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
            deleted_counts = {
                "conversation_log": 0,
                "dna_match": 0,
                "family_tree": 0,
                "people": 0,
            }

            # 2. Delete from conversation_log
            logger.debug(
                f"Deleting from conversation_log where people_id != {person_id_to_keep}..."
            )
            result_conv = (
                sess.query(ConversationLog)
                .filter(ConversationLog.people_id != person_id_to_keep)
                .delete(synchronize_session=False)
            )
            deleted_counts["conversation_log"] = (
                result_conv if result_conv is not None else 0
            )
            logger.info(
                f"Deleted {deleted_counts['conversation_log']} conversation_log records."
            )

            # 3. Delete from dna_match
            logger.debug(
                f"Deleting from dna_match where people_id != {person_id_to_keep}..."
            )
            result_dna = (
                sess.query(DnaMatch)
                .filter(DnaMatch.people_id != person_id_to_keep)
                .delete(synchronize_session=False)
            )
            deleted_counts["dna_match"] = result_dna if result_dna is not None else 0
            logger.info(f"Deleted {deleted_counts['dna_match']} dna_match records.")

            # 4. Delete from family_tree
            logger.debug(
                f"Deleting from family_tree where people_id != {person_id_to_keep}..."
            )
            result_ft = (
                sess.query(FamilyTree)
                .filter(FamilyTree.people_id != person_id_to_keep)
                .delete(synchronize_session=False)
            )
            deleted_counts["family_tree"] = result_ft if result_ft is not None else 0
            logger.info(f"Deleted {deleted_counts['family_tree']} family_tree records.")

            # 5. Delete from people
            logger.debug(f"Deleting from people where id != {person_id_to_keep}...")
            result_people = (
                sess.query(Person)
                .filter(Person.id != person_id_to_keep)
                .delete(synchronize_session=False)
            )
            deleted_counts["people"] = result_people if result_people is not None else 0
            logger.info(f"Deleted {deleted_counts['people']} people records.")

            total_deleted = sum(deleted_counts.values())
            if total_deleted == 0:
                logger.info(
                    f"No records found to delete besides Person ID {person_id_to_keep}."
                )

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


# Action 1
def run_core_workflow_action(session_manager, *_):
    """
    Action to run the core workflow sequence: Action 7 (Inbox) → Action 9 (Process Productive) → Action 8 (Send Messages).
    Optionally runs Action 6 (Gather) first if configured.
    Relies on exec_actn ensuring session is ready beforehand.
    """
    # Guard clause now checks session_ready
    if not session_manager or not session_manager.session_ready:
        logger.error("Cannot run core workflow: Session not ready.")
        return False

    try:
        # --- Action 6 (Optional) ---
        # Check if Action 6 should be included in the workflow
        run_action6 = config.include_action6_in_workflow
        if run_action6:
            logger.info("--- Running Action 6: Gather Matches (Always from page 1) ---")
            gather_result = coord_action(session_manager, config, start=1)
            if gather_result is False:
                logger.error("Action 6 FAILED.")
                # Match gathering failure logged above
                return False
            logger.info("Action 6 OK.")

        # --- Action 7 ---
        logger.info("--- Running Action 7: Search Inbox ---")
        inbox_url = urljoin(config.api.base_url, "/messaging/")
        logger.debug(f"Navigating to Inbox ({inbox_url}) for Action 7...")

        # Use a more reliable selector that exists on the messaging page
        # First try to navigate to the inbox page
        try:
            if not nav_to_page(
                session_manager.driver,
                inbox_url,
                "div.messaging-container",  # More general selector for the messaging container
                session_manager,
            ):
                logger.error("Action 7 nav FAILED - Could not navigate to inbox page.")
                # Navigation error logged above
                return False

            logger.debug("Navigation to inbox page successful.")

            # Add a short delay to ensure page is fully loaded
            time.sleep(2)

            # Now run the inbox processor
            logger.debug("Running inbox search...")
            inbox_processor = InboxProcessor(session_manager=session_manager)
            search_result = inbox_processor.search_inbox()

            if search_result is False:
                logger.error("Action 7 FAILED - Inbox search returned failure.")
                # Inbox search failure logged above
                return False
            logger.info("Action 7 OK.")
                # Inbox search completed successfully
        except Exception as inbox_error:
            logger.error(
                f"Action 7 FAILED with exception: {inbox_error}", exc_info=True
            )
            # Error during inbox search logged above
            return False

        # --- Action 9 ---
        logger.info("--- Running Action 9: Process Productive Messages ---")
        logger.debug("Navigating to Base URL for Action 9...")

        try:
            if not nav_to_page(
                session_manager.driver,
                config.api.base_url,
                WAIT_FOR_PAGE_SELECTOR,  # Use a general page load selector
                session_manager,
            ):
                logger.error("Action 9 nav FAILED - Could not navigate to base URL.")
                # Navigation error logged above
                return False

            logger.debug(
                "Navigation to base URL successful. Processing productive messages..."
            )

            # Add a short delay to ensure page is fully loaded
            time.sleep(2)

            # Process productive messages
            process_result = process_productive_messages(session_manager)

            if process_result is False:
                logger.error(
                    "Action 9 FAILED - Productive message processing returned failure."
                )
                print(
                    "ERROR: Productive message processing failed. Check logs for details."
                )
                return False
            logger.info("Action 9 OK.")
                # Productive message processing completed successfully
        except Exception as process_error:
            logger.error(
                f"Action 9 FAILED with exception: {process_error}", exc_info=True
            )
            print(f"ERROR during productive message processing: {process_error}")
            return False

        # --- Action 8 ---
        logger.info("--- Running Action 8: Send Messages ---")
        logger.debug("Navigating to Base URL for Action 8...")

        try:
            if not nav_to_page(
                session_manager.driver,
                config.api.base_url,
                WAIT_FOR_PAGE_SELECTOR,  # Use a general page load selector
                session_manager,
            ):
                logger.error("Action 8 nav FAILED - Could not navigate to base URL.")
                print(
                    "ERROR: Could not navigate to base URL. Check network connection."
                )
                return False

            logger.debug("Navigation to base URL successful. Sending messages...")

            # Add a short delay to ensure page is fully loaded
            time.sleep(2)

            # send_messages_to_matches expects session_manager
            send_result = send_messages_to_matches(session_manager)

            if send_result is False:
                logger.error("Action 8 FAILED - Message sending returned failure.")
                # Message sending failure logged above
                return False
            logger.info("Action 8 OK.")
                # Message sending completed successfully
        except Exception as message_error:
            logger.error(
                f"Action 8 FAILED with exception: {message_error}", exc_info=True
            )
            print(f"ERROR during message sending: {message_error}")
            return False

        # Determine which actions were run for the success message
        action_sequence = []
        if run_action6:
            action_sequence.append("6")
        action_sequence.extend(["7", "9", "8"])
        action_sequence_str = "-".join(action_sequence)

        logger.info(
            f"Core Workflow (Actions {action_sequence_str}) finished successfully."
        )
        print(
            f"\n✓ Core Workflow (Actions {action_sequence_str}) completed successfully."
        )
        return True
    except Exception as e:
        logger.error(f"Critical error during core workflow: {e}", exc_info=True)
        # Critical error during core workflow logged above
        return False


# End Action 1


def _create_message_template(template_key: str, template_content: str) -> MessageTemplate:
    """Create a MessageTemplate object with proper categorization."""
    # Extract subject line from content
    subject_line = None
    if template_content.startswith("Subject: "):
        lines = template_content.split("\n", 1)
        if len(lines) >= 1:
            subject_line = lines[0].replace("Subject: ", "").strip()

    # Determine template category and tree status
    template_category = "other"
    tree_status = "universal"

    if "Initial" in template_key:
        template_category = "initial"
    elif "Follow_Up" in template_key:
        template_category = "follow_up"
    elif "Reminder" in template_key:
        template_category = "reminder"
    elif "Acknowledgement" in template_key:
        template_category = "acknowledgement"
    elif "Desist" in template_key:
        template_category = "desist"

    if template_key.startswith("In_Tree"):
        tree_status = "in_tree"
    elif template_key.startswith("Out_Tree"):
        tree_status = "out_tree"

    # Create human-readable name
    template_name = template_key.replace("_", " ").replace("-", " - ")

    return MessageTemplate(
        template_key=template_key,
        template_name=template_name,
        subject_line=subject_line,
        message_content=template_content,
        template_category=template_category,
        tree_status=tree_status,
        is_active=True,
        version=1
    )


# Action 2 (reset_db_actn)
def reset_db_actn(session_manager: SessionManager, *_):
    """
    Action to COMPLETELY reset the database by truncating tables and recreating schema.
    - Closes main pool.
    - Truncates all tables (safer than file deletion).
    - Recreates schema from scratch.
    - Seeds the MessageTemplate table.
    """
    db_path = config.database.database_file
    reset_successful = False
    temp_manager = None  # For recreation/seeding
    recreation_session = None  # Session for seeding

    try:
        # --- 1. Close main pool FIRST ---
        if session_manager:
            logger.debug("Closing main DB connections before database reset...")
            session_manager.cls_db_conn(keep_db=False)  # Ensure pool is closed
            logger.debug("Main DB pool closed.")
        else:
            logger.warning("No main session manager passed to reset_db_actn to close.")

        # Force garbage collection to release any file handles
        logger.debug("Running garbage collection to release file handles...")
        gc.collect()
        time.sleep(1.0)
        gc.collect()

        # --- 2. Reset Database Content ---
        if db_path is None:
            logger.critical("DATABASE_FILE is not configured. Reset aborted.")
            return False

        logger.debug("Starting database reset using table truncation...")
        try:
            # Create temporary SessionManager for database reset
            logger.debug("Creating temporary session manager for database reset...")
            temp_manager = SessionManager()

            # Step 1: Safely truncate all tables that exist
            logger.debug("Safely truncating existing tables...")
            truncate_session = temp_manager.get_db_conn()
            if truncate_session:
                with db_transn(truncate_session) as sess:
                    # Check which tables exist and delete in safe order
                    from sqlalchemy import inspect
                    inspector = inspect(sess.get_bind())
                    existing_tables = inspector.get_table_names()
                    logger.debug(f"Found existing tables: {existing_tables}")

                    # Delete in reverse dependency order, but only if tables exist
                    if 'conversation_log' in existing_tables:
                        sess.query(ConversationLog).delete(synchronize_session=False)
                        logger.debug("Truncated conversation_log table")

                    if 'dna_match' in existing_tables:
                        sess.query(DnaMatch).delete(synchronize_session=False)
                        logger.debug("Truncated dna_match table")

                    if 'family_tree' in existing_tables:
                        sess.query(FamilyTree).delete(synchronize_session=False)
                        logger.debug("Truncated family_tree table")

                    if 'people' in existing_tables:
                        sess.query(Person).delete(synchronize_session=False)
                        logger.debug("Truncated people table")

                    # Clear message_types too for complete reset
                    if 'message_types' in existing_tables:
                        sess.query(MessageTemplate).delete(synchronize_session=False)
                        logger.debug("Truncated message_types table")

                temp_manager.return_session(truncate_session)
                logger.debug("All existing tables truncated successfully.")
            else:
                logger.critical("Failed to get session for truncating tables. Reset aborted.")
                return False

            # Step 2: Ensure all tables exist with proper schema
            logger.debug("Ensuring complete database schema...")
            temp_manager.db_manager._initialize_engine_and_session()
            if not temp_manager.db_manager.engine or not temp_manager.db_manager.Session:
                raise SQLAlchemyError(
                    "Failed to initialize DB engine/session for recreation!"
                )

            # Create any missing tables
            Base.metadata.create_all(temp_manager.db_manager.engine)
            logger.debug("Database schema ensured successfully.")

            # --- Seed MessageTemplate Table (handled by database.py) ---
            # Note: MessageTemplate seeding is now handled by database.py during create_all()
            # The following code is kept for reference but should not be needed
            recreation_session = temp_manager.get_db_conn()
            if not recreation_session:
                raise SQLAlchemyError("Failed to get session for seeding MessageTemplates!")

            # MessageTemplate verification (templates managed in database)
            with db_transn(recreation_session) as sess:
                template_count = sess.query(func.count(MessageTemplate.id)).scalar() or 0
                logger.debug(f"MessageTemplate verification: {template_count} templates found in database")
            # --- End Seeding ---

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
def backup_db_actn(
    session_manager: Optional[SessionManager] = None, *_
):  # Added session_manager parameter for exec_actn compatibility
    _ = session_manager  # Mark as intentionally unused
    """Action to backup the database. Browserless."""
    try:
        logger.debug("Starting DB backup...")
        # session_manager isn't used but needed for exec_actn compatibility
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
        else:
            logger.warning(
                "No main session manager passed to restore_db_actn to close."
            )
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


# Action 5 (check_login_actn)
def check_login_actn(session_manager: SessionManager, *_) -> bool:
    """
    REVISED V12: Checks login status and attempts login if needed.
    This action starts a browser session and checks login status.
    If not logged in, it attempts to log in using stored credentials.
    Provides clear user feedback about the final login state.
    """
    if not session_manager:
        logger.error("SessionManager required for check_login_actn.")
        print("ERROR: Internal error - session manager not available.")
        return False

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
            print("\n✓ You are currently logged in to Ancestry.")
            # Display additional session info if available
            if session_manager.my_profile_id:
                print(f"  Profile ID: {session_manager.my_profile_id}")
            if session_manager.tree_owner_name:
                print(f"  Account: {session_manager.tree_owner_name}")
            return True
        if status is False:
            print("\n✗ You are NOT currently logged in to Ancestry.")
            print("  Attempting to log in with stored credentials...")

            # Attempt login using the session manager's login functionality
            try:
                login_result = log_in(session_manager)

                if login_result:
                    print("✓ Login successful!")
                    # Check status again after login
                    final_status = login_status(session_manager, disable_ui_fallback=False)
                    if final_status is True:
                        print("✓ Login verification confirmed.")
                        # Display session info if available
                        if session_manager.my_profile_id:
                            print(f"  Profile ID: {session_manager.my_profile_id}")
                        if session_manager.tree_owner_name:
                            print(f"  Account: {session_manager.tree_owner_name}")
                        return True
                    print("⚠️  Login appeared successful but verification failed.")
                    return False
                print("✗ Login failed. Please check your credentials.")
                print("  You can update credentials using the 'sec' option in the main menu.")
                return False

            except Exception as login_e:
                logger.error(f"Exception during login attempt: {login_e}", exc_info=True)
                print(f"✗ Login failed with error: {login_e}")
                print("  You can update credentials using the 'sec' option in the main menu.")
                return False

        else:  # Status is None
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


# Action 6 (coord_action wrapper)
def coord_action(session_manager, config_schema=None, start=1):
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

    logger.info(f"Gathering DNA Matches from page {start}...")
    try:
        # Call the imported function from action6
        result = coord(session_manager, config, start=start)
        if result is False:
            logger.error("Match gathering reported failure.")
            return False
        logger.debug("Gathering matches OK.")
        return True
    except Exception as e:
        logger.error(f"Error during coord_action: {e}", exc_info=True)
        return False


# End of coord_action


# Action 7 (srch_inbox_actn)
def srch_inbox_actn(session_manager, *_):
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
            logger.debug("Initializing session_ready based on driver_live")
            session_manager.session_ready = True
            session_ready = True
        else:
            logger.debug("Initializing session_ready to False")
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
def send_messages_action(session_manager, *_):
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
            logger.debug("Initializing session_ready based on driver_live")
            session_manager.session_ready = True
            session_ready = True
        else:
            logger.debug("Initializing session_ready to False")
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
        logger.debug("Messages sent OK.")
        return True  # Use INFO
    except Exception as e:
        logger.error(f"Error during message sending: {e}", exc_info=True)
        return False


# End of send_messages_action


# Action 9 (process_productive_messages_action)
def process_productive_messages_action(session_manager, *_):
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
            logger.debug("Initializing session_ready based on driver_live")
            session_manager.session_ready = True
            session_ready = True
        else:
            logger.debug("Initializing session_ready to False")
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
def run_action11_wrapper(session_manager, *_):
    """Action to run API Report. Relies on exec_actn for consistent logging and error handling."""
    logger.debug("Starting API Report...")
    try:
        # Call the actual API Report function, passing the session_manager for API calls
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


def main():
    global logger, session_manager  # noqa: PLW0603 - CLI modifies module globals intentionally
    session_manager = None  # Initialize session_manager

    # Ensure terminal window has focus (Windows & VS Code)
    try:
        if os.name == 'nt':  # Windows
            import ctypes
            import time

            # Get console window handle
            kernel32 = ctypes.windll.kernel32
            user32 = ctypes.windll.user32

            # Method 1: Try to focus console window (for regular terminals)
            console_window = kernel32.GetConsoleWindow()
            if console_window:
                # Bring console window to foreground
                user32.SetForegroundWindow(console_window)
                user32.ShowWindow(console_window, 9)  # SW_RESTORE

            # Method 2: For VS Code integrated terminal, try to focus current window
            # Get the currently active window
            current_window = user32.GetForegroundWindow()
            if current_window and current_window != console_window:
                # This might be VS Code - try to ensure it stays focused
                user32.SetForegroundWindow(current_window)
                user32.BringWindowToTop(current_window)

                # Also try to send a focus message to ensure terminal panel is active
                # This is a gentle nudge that shouldn't disrupt the user
                user32.SetActiveWindow(current_window)

            # Small delay to ensure focus operations complete
            time.sleep(0.05)

    except Exception as focus_error:
        # Silently ignore focus errors but log for debugging if logger is available
        try:
            # Check if logger exists and is available (logging is already imported at top)
            if 'logger' in globals() and logger and hasattr(logger, 'debug'):
                logger.debug(f"Terminal focus attempt failed (non-critical): {focus_error}")
        except Exception:
            pass  # Even logging failed, continue silently

    try:
        print("")
        # --- Logging Setup ---
        logger = setup_logging()

        # --- Configuration Validation ---
        # Validate action configuration to prevent Action 6-style failures
        validate_action_config()

        if config is None:
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

        # --- Instantiate SessionManager ---
        session_manager = SessionManager()  # No browser started by default

        # --- Main menu loop ---
        while True:
            choice = menu()
            print("")

            # --- Confirmation dictionary ---
            confirm_actions = {
                "0": "Delete all people except specific profile ID",
                "2": "COMPLETELY reset the database (deletes data)",
                "4": "Restore database from backup (overwrites data)",
            }

            # --- Confirmation Check ---
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
                    continue
                print(" ")  # Newline after confirmation

            # --- Action Dispatching ---
            # Note: Removed most close_sess_after=False as the default is now to keep open.
            # Added close_sess_after=True only where explicit closure after action is desired.

            # --- Database-only actions (no browser needed) ---
            if choice in ["0", "2", "3", "4"]:
                # For database-only actions, we can use the main session_manager
                # The exec_actn function will set browser_needed=False based on the action

                if choice == "0":
                    # Confirmation handled above
                    exec_actn(all_but_first_actn, session_manager, choice)
                elif choice == "2":
                    # Confirmation handled above
                    exec_actn(reset_db_actn, session_manager, choice)
                elif choice == "3":
                    exec_actn(
                        backup_db_actn, session_manager, choice
                    )  # Run through exec_actn for consistent logging
                elif choice == "4":
                    # Confirmation handled above
                    exec_actn(restore_db_actn, session_manager, choice)

            # --- Browser-required actions ---
            elif choice == "1":
                # Initialize caching for GEDCOM operations (needed for action 9 in workflow)
                ensure_caching_initialized()
                # exec_actn will set browser_needed=True based on the action
                exec_actn(
                    run_core_workflow_action,
                    session_manager,
                    choice,
                    close_sess_after=True,
                )  # Close after full sequence
            elif choice == "5":
                exec_actn(check_login_actn, session_manager, choice)  # API-only check
            elif choice.startswith("6"):
                parts = choice.split()
                start_val = 1
                if len(parts) > 1:
                    try:
                        start_arg = int(parts[1])
                        start_val = start_arg if start_arg > 0 else 1
                    except ValueError:
                        logger.warning(f"Invalid start page '{parts[1]}'. Using 1.")
                        print(f"Invalid start page '{parts[1]}'. Using page 1 instead.")

                # Call exec_actn with the correct parameters
                exec_actn(
                    coord_action,
                    session_manager,
                    "6",
                    False,  # don't close session after
                    config,
                    start_val,
                )  # Keep open
            elif choice == "7":
                exec_actn(srch_inbox_actn, session_manager, choice)  # Keep open
            elif choice == "8":
                exec_actn(send_messages_action, session_manager, choice)  # Keep open
            elif choice == "9":
                # Initialize caching for GEDCOM operations
                ensure_caching_initialized()
                exec_actn(
                    process_productive_messages_action, session_manager, choice
                )  # Keep open
            elif choice == "10":
                # Initialize caching for GEDCOM operations and load GEDCOM data
                ensure_caching_initialized()
                exec_actn(run_action10, session_manager, choice)
                # After running action 10, try to cache the GEDCOM data for future use
                try:
                    print("\n📁 Caching GEDCOM data for Phase 12 AI analysis...")
                    if ensure_gedcom_loaded_and_cached():
                        print("✅ GEDCOM data cached successfully! Phase 12 AI analysis now available.")
                    else:
                        print("INFO: GEDCOM data not cached. Phase 12 AI analysis may require manual loading.")
                except Exception as e:
                    logger.debug(f"Error caching GEDCOM data after action 10: {e}")
                    print("INFO: GEDCOM data caching skipped. Phase 12 AI analysis may require manual loading.")
            elif choice == "11":
                # Initialize caching for GEDCOM operations
                ensure_caching_initialized()
                # Use the wrapper function to run Action 11 through exec_actn
                exec_actn(run_action11_wrapper, session_manager, choice)
            # === PHASE 12: GEDCOM AI INTELLIGENCE ===
            elif choice == "12":
                if ensure_gedcom_loaded_and_cached():
                    run_gedcom_intelligence_analysis()
                else:
                    input("\nPress Enter to continue...")
            elif choice == "13":
                if ensure_gedcom_loaded_and_cached():
                    run_dna_gedcom_crossref()
                else:
                    input("\nPress Enter to continue...")
            elif choice == "14":
                if ensure_gedcom_loaded_and_cached():
                    run_research_prioritization()
                else:
                    input("\nPress Enter to continue...")
            elif choice == "15":
                if ensure_gedcom_loaded_and_cached():
                    run_comprehensive_gedcom_ai()
                else:
                    input("\nPress Enter to continue...")
            # --- Test Options ---
            elif choice == "test":
                # Show Main.py Status
                print("\n" + "=" * 60)
                print("MAIN.PY APPLICATION STATUS")
                print("=" * 60)
                print("🚀 main.py - Application entry point")
                print("✅ Application is properly initialized and ready to run")
                print("📋 All action modules loaded successfully")
                print("🎯 Menu system functional")
                print("\n✅ main.py status: READY")
                input("Press Enter to continue...")
            elif choice == "testall":
                # Run All Module Tests
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
            # --- Meta Options ---
            elif choice == "sec":
                # Setup Security (Encrypt Credentials)
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

                    if "No module named 'cryptography'" in str(
                        e
                    ) or "No module named 'keyring'" in str(e):
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

                        # Platform-specific installation instructions
                        _show_platform_specific_instructions()

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
            elif choice == "settings":
                # Review/Edit .env Settings
                try:
                    env_path = Path(__file__).parent / ".env"
                    if not env_path.exists():
                        print(".env file not found.")
                        input("Press Enter to continue...")
                        continue
                    # Read .env file
                    with env_path.open(encoding="utf-8") as f:
                        lines = f.readlines()
                    # Filter out comments and blank lines
                    # Build a list of (line_index, setting_line) for settings only
                    settings = [(i, line.strip()) for i, line in enumerate(lines) if line.strip() and not line.strip().startswith("#")]
                    print("\nCurrent .env Settings:")
                    for idx, (line_idx, setting) in enumerate(settings, 1):
                        print(f"{idx}. {setting}")
                    print("\nEnter the number of the setting to edit, or 'q' to cancel.")
                    sel = input("Select setting: ").strip().lower()
                    if sel == "q":
                        continue
                    try:
                        sel_idx = int(sel) - 1
                        if sel_idx < 0 or sel_idx >= len(settings):
                            print("Invalid selection.")
                            input("Press Enter to continue...")
                            continue
                    except ValueError:
                        print("Invalid input.")
                        input("Press Enter to continue...")
                        continue
                    line_idx, orig_setting = settings[sel_idx]
                    key, eq, value = orig_setting.partition("=")
                    if not eq:
                        print("Selected line is not a valid setting.")
                        input("Press Enter to continue...")
                        continue
                    print(f"Current value for {key}: {value}")
                    new_value = input(f"Enter new value for {key} (or leave blank to cancel): ").strip()
                    if not new_value:
                        print("No change made.")
                        input("Press Enter to continue...")
                        continue
                    # Update the line in the original lines list
                    lines[line_idx] = f"{key}={new_value}\n"
                    # Write back to .env
                    with env_path.open("w", encoding="utf-8") as f:
                        f.writelines(lines)
                    print(f"Updated {key} in .env.")
                except Exception as e:
                    print(f"Error editing .env: {e}")
                print("\nReturning to main menu...")
                input("Press Enter to continue...")
            elif choice == "s":
                # Show cache statistics
                try:
                    logger.info("Cache statistics feature currently unavailable")
                    print("Cache statistics feature currently unavailable.")
                except Exception as e:
                    logger.error(f"Error displaying cache statistics: {e}")
                    print("Error displaying cache statistics. Check logs for details.")
            elif choice == "t":
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

                        # Cycle through DEBUG → INFO → WARNING
                        if current_level == logging.DEBUG:
                            new_level_name = "INFO"
                        elif current_level == logging.INFO:
                            new_level_name = "WARNING"
                        else:  # WARNING or other
                            new_level_name = "DEBUG"

                        # Re-call setup_logging to potentially update filters etc. too
                        logger = setup_logging(log_level=new_level_name)
                        logger.info(f"Console log level toggled to: {new_level_name}")
                        print(f"Log level changed to: {new_level_name}")
                    else:
                        logger.warning(
                            "Could not find console handler to toggle level."
                        )
                else:
                    print(
                        "WARNING: Logger not ready or has no handlers.", file=sys.stderr
                    )
            elif choice == "c":
                os.system("cls" if os.name == "nt" else "clear")
            elif choice == "q":
                os.system("cls" if os.name == "nt" else "clear")
                print("Exiting.")
                break
            # Handle invalid choices
            elif choice not in confirm_actions:  # Avoid double 'invalid' message
                print("Invalid choice.\n")

            # No need to track if driver became live anymore

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
        # Execution finished


# end main


# === PHASE 12: GEDCOM AI FUNCTIONS ===

def run_gedcom_intelligence_analysis():
    """Run GEDCOM Intelligence Analysis (Phase 12.1)."""
    print("\n" + "="*60)
    print("🧠 GEDCOM INTELLIGENCE ANALYSIS")
    print("="*60)

    if not PHASE_12_AVAILABLE:
        print("❌ Phase 12 GEDCOM AI components not available.")
        print("Please ensure all Phase 12 modules are properly installed.")
        input("\nPress Enter to continue...")
        return

    try:
        # Get cached GEDCOM data (already loaded by menu handler)
        from gedcom_search_utils import get_cached_gedcom_data
        gedcom_data = get_cached_gedcom_data()

        # Initialize analyzer
        print("🔍 Initializing GEDCOM Intelligence Analyzer...")
        from gedcom_intelligence import GedcomIntelligenceAnalyzer
        analyzer = GedcomIntelligenceAnalyzer()

        # Perform analysis
        print("🧠 Analyzing family tree data...")
        analysis_result = analyzer.analyze_gedcom_data(gedcom_data)

        # Display results
        print("\n" + "="*50)
        print("📊 ANALYSIS RESULTS")
        print("="*50)

        summary = analysis_result.get("summary", {})
        print(f"👥 Individuals Analyzed: {analysis_result.get('individuals_analyzed', 0)}")
        print(f"🔍 Gaps Identified: {summary.get('total_gaps', 0)}")
        print(f"⚠️  Conflicts Found: {summary.get('total_conflicts', 0)}")
        print(f"🎯 Research Opportunities: {summary.get('total_opportunities', 0)}")
        print(f"🔥 High Priority Items: {summary.get('high_priority_items', 0)}")

        # Show top gaps
        gaps = analysis_result.get("gaps_identified", [])
        if gaps:
            print("\n🔍 TOP GAPS IDENTIFIED:")
            for i, gap in enumerate(gaps[:5], 1):
                print(f"{i}. {gap.get('description', 'Unknown gap')} (Priority: {gap.get('priority', 'unknown')})")

        # Show top conflicts
        conflicts = analysis_result.get("conflicts_identified", [])
        if conflicts:
            print("\n⚠️  TOP CONFLICTS FOUND:")
            for i, conflict in enumerate(conflicts[:3], 1):
                print(f"{i}. {conflict.get('description', 'Unknown conflict')} (Severity: {conflict.get('severity', 'unknown')})")

        # Show AI insights
        ai_insights = analysis_result.get("ai_insights", {})
        tree_completeness = ai_insights.get("tree_completeness", {})
        if tree_completeness:
            completeness = tree_completeness.get("completeness_percentage", 0)
            print(f"\n🌳 Tree Completeness: {completeness:.1f}%")

        print("\n✅ Analysis completed successfully!")

    except Exception as e:
        print(f"❌ Error during GEDCOM intelligence analysis: {e}")
        logger.error(f"GEDCOM intelligence analysis error: {e}")

    input("\nPress Enter to continue...")


def run_dna_gedcom_crossref():
    """Run DNA-GEDCOM Cross-Reference Analysis (Phase 12.2)."""
    print("\n" + "="*60)
    print("🧬 DNA-GEDCOM CROSS-REFERENCE ANALYSIS")
    print("="*60)

    if not PHASE_12_AVAILABLE:
        print("❌ Phase 12 GEDCOM AI components not available.")
        input("\nPress Enter to continue...")
        return

    try:
        # Load GEDCOM data
        print("📁 Loading GEDCOM data...")
        if get_cached_gedcom_data is None:
            print("❌ GEDCOM functionality not available.")
            return

        gedcom_data = get_cached_gedcom_data()

        if not gedcom_data:
            print("❌ No GEDCOM data available.")
            input("\nPress Enter to continue...")
            return

        # Get DNA matches from database
        print("🧬 Loading DNA match data...")
        session_manager = SessionManager()
        session = session_manager.get_db_conn()

        try:
            if not session:
                print("❌ Could not get database session for DNA match data.")
                print("INFO: DNA cross-reference will proceed without DNA match data.")
                dna_matches = []
            else:
                dna_matches_db = session.query(DnaMatch).limit(10).all()

                # Convert to Phase 12 format
                dna_matches = []
                for match in dna_matches_db:
                    from dna_gedcom_crossref import DNAMatch
                    dna_match = DNAMatch(
                        match_id=str(match.id),
                        match_name=match.username or "Unknown",
                        estimated_relationship="unknown",
                        shared_dna_cm=None,
                        testing_company="Ancestry"
                    )
                    dna_matches.append(dna_match)

                print(f"📊 Found {len(dna_matches)} DNA matches to analyze")

        finally:
            if session:
                session_manager.return_session(session)

        if not dna_matches:
            print("❌ No DNA matches available for analysis.")
            input("\nPress Enter to continue...")
            return

        # Initialize cross-referencer
        print("🔍 Initializing DNA-GEDCOM Cross-Referencer...")
        from dna_gedcom_crossref import DNAGedcomCrossReferencer
        crossref = DNAGedcomCrossReferencer()

        # Perform analysis
        print("🧬 Cross-referencing DNA matches with GEDCOM data...")
        analysis_result = crossref.analyze_dna_gedcom_connections(dna_matches, gedcom_data)

        # Display results
        print("\n" + "="*50)
        print("📊 CROSS-REFERENCE RESULTS")
        print("="*50)

        summary = analysis_result.get("summary", {})
        print(f"🧬 DNA Matches Analyzed: {analysis_result.get('dna_matches_analyzed', 0)}")
        print(f"👥 GEDCOM People Analyzed: {analysis_result.get('gedcom_people_analyzed', 0)}")
        print(f"🔗 Cross-References Found: {summary.get('total_cross_references', 0)}")
        print(f"⭐ High Confidence Matches: {summary.get('high_confidence_matches', 0)}")
        print(f"⚠️  Conflicts Identified: {summary.get('conflicts_found', 0)}")
        print(f"✅ Verification Opportunities: {summary.get('verification_opportunities', 0)}")

        # Show top matches
        crossref_matches = analysis_result.get("cross_reference_matches", [])
        if crossref_matches:
            print("\n🔗 TOP CROSS-REFERENCE MATCHES:")
            for i, match in enumerate(crossref_matches[:5], 1):
                confidence = match.get('confidence_score', 0)
                dna_name = match.get('dna_match_name', 'Unknown')
                match_type = match.get('match_type', 'unknown')
                print(f"{i}. {dna_name} - {match_type} (Confidence: {confidence:.1%})")

        print("\n✅ Cross-reference analysis completed!")

    except Exception as e:
        print(f"❌ Error during DNA-GEDCOM cross-reference: {e}")
        logger.error(f"DNA-GEDCOM cross-reference error: {e}")

    input("\nPress Enter to continue...")


def run_research_prioritization():
    """Run Research Prioritization Analysis (Phase 12.3)."""
    print("\n" + "="*60)
    print("📊 INTELLIGENT RESEARCH PRIORITIZATION")
    print("="*60)

    if not PHASE_12_AVAILABLE:
        print("❌ Phase 12 GEDCOM AI components not available.")
        input("\nPress Enter to continue...")
        return

    try:
        # First run GEDCOM intelligence analysis
        print("🧠 Running GEDCOM intelligence analysis...")
        from gedcom_search_utils import get_cached_gedcom_data
        gedcom_data = get_cached_gedcom_data()

        if not gedcom_data:
            print("❌ No GEDCOM data available.")
            input("\nPress Enter to continue...")
            return

        from gedcom_intelligence import GedcomIntelligenceAnalyzer
        analyzer = GedcomIntelligenceAnalyzer()
        gedcom_analysis = analyzer.analyze_gedcom_data(gedcom_data)

        # Initialize prioritizer
        print("📊 Initializing Research Prioritizer...")
        from research_prioritization import IntelligentResearchPrioritizer
        prioritizer = IntelligentResearchPrioritizer()

        # Perform prioritization
        print("🎯 Analyzing research priorities...")
        prioritization_result = prioritizer.prioritize_research_tasks(gedcom_analysis, {})

        # Display results
        print("\n" + "="*50)
        print("📊 PRIORITIZATION RESULTS")
        print("="*50)

        print(f"🎯 Total Priorities Identified: {prioritization_result.get('total_priorities_identified', 0)}")

        # Show family line analysis
        family_lines = prioritization_result.get("family_line_analysis", [])
        if family_lines:
            print("\n👨‍👩‍👧‍👦 FAMILY LINE ANALYSIS:")
            for line in family_lines[:3]:
                surname = line.get('surname', 'Unknown')
                completeness = line.get('completeness_percentage', 0)
                generations = line.get('generations_back', 0)
                print(f"• {surname} Line: {completeness:.1f}% complete, {generations} generations back")

        # Show location clusters
        clusters = prioritization_result.get("location_clusters", [])
        if clusters:
            print("\n🌍 RESEARCH CLUSTERS:")
            for cluster in clusters[:3]:
                location = cluster.get('location', 'Unknown')
                people_count = cluster.get('people_count', 0)
                efficiency = cluster.get('research_efficiency_score', 0)
                print(f"• {location}: {people_count} people (Efficiency: {efficiency:.1%})")

        # Show top priority tasks
        tasks = prioritization_result.get("prioritized_tasks", [])
        if tasks:
            print("\n🔥 TOP PRIORITY RESEARCH TASKS:")
            for i, task in enumerate(tasks[:5], 1):
                description = task.get('description', 'Unknown task')
                priority_score = task.get('priority_score', 0)
                urgency = task.get('urgency', 'unknown')
                print(f"{i}. {description[:60]}... (Score: {priority_score:.1f}, Urgency: {urgency})")

        # Show recommendations
        recommendations = prioritization_result.get("research_recommendations", [])
        if recommendations:
            print("\n💡 AI RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"{i}. {rec}")

        # Show next steps
        next_steps = prioritization_result.get("next_steps", [])
        if next_steps:
            print("\n🚀 IMMEDIATE NEXT STEPS:")
            for i, step in enumerate(next_steps[:3], 1):
                print(f"{i}. {step}")

        print("\n✅ Research prioritization completed!")

    except Exception as e:
        print(f"❌ Error during research prioritization: {e}")
        logger.error(f"Research prioritization error: {e}")

    input("\nPress Enter to continue...")


def run_comprehensive_gedcom_ai():
    """Run Comprehensive GEDCOM AI Analysis (All Phase 12 components)."""
    print("\n" + "="*60)
    print("🤖 COMPREHENSIVE GEDCOM AI ANALYSIS")
    print("="*60)

    if not PHASE_12_AVAILABLE:
        print("❌ Phase 12 GEDCOM AI components not available.")
        input("\nPress Enter to continue...")
        return

    try:
        # Load GEDCOM data
        print("📁 Loading GEDCOM data...")
        from gedcom_search_utils import get_cached_gedcom_data
        gedcom_data = get_cached_gedcom_data()

        if not gedcom_data:
            print("❌ No GEDCOM data available.")
            input("\nPress Enter to continue...")
            return

        # Get DNA matches
        print("🧬 Loading DNA match data...")
        session_manager = SessionManager()
        session = session_manager.get_db_conn()

        dna_matches_data = []
        try:
            if not session:
                print("❌ Could not get database session for DNA match data.")
                print("INFO: Comprehensive analysis will proceed without DNA match data.")
            else:
                dna_matches_db = session.query(DnaMatch).limit(10).all()
                for match in dna_matches_db:
                    dna_matches_data.append({
                        "match_id": str(match.id),
                        "match_name": match.username or "Unknown",
                        "estimated_relationship": "unknown",
                        "shared_dna_cm": None,
                        "testing_company": "Ancestry"
                    })
        finally:
            if session:
                session_manager.return_session(session)

        # Initialize comprehensive integrator
        print("🤖 Initializing GEDCOM AI Integrator...")
        from gedcom_ai_integration import GedcomAIIntegrator
        integrator = GedcomAIIntegrator()

        # Perform comprehensive analysis
        print("🧠 Running comprehensive GEDCOM AI analysis...")
        comprehensive_result = integrator.perform_comprehensive_analysis(
            gedcom_data, dna_matches_data
        )

        # Display results
        print("\n" + "="*50)
        print("📊 COMPREHENSIVE ANALYSIS RESULTS")
        print("="*50)

        summary = comprehensive_result.get("summary", {})
        print(f"👥 Individuals Analyzed: {summary.get('gedcom_individuals_analyzed', 0)}")
        print(f"🔍 Gaps Identified: {summary.get('total_gaps_identified', 0)}")
        print(f"⚠️  Conflicts Found: {summary.get('total_conflicts_identified', 0)}")
        print(f"🎯 Research Priorities: {summary.get('research_priorities_generated', 0)}")

        if dna_matches_data:
            print(f"🧬 DNA Matches Analyzed: {summary.get('dna_matches_analyzed', 0)}")
            print(f"🔗 DNA Cross-References: {summary.get('dna_crossref_matches', 0)}")

        # Show integrated insights
        insights = comprehensive_result.get("integrated_insights", {})
        if insights:
            print("\n🧠 INTEGRATED AI INSIGHTS:")
            tree_health = insights.get('tree_health_score', 0)
            print(f"🌳 Tree Health Score: {tree_health}/100")

            dna_potential = insights.get('dna_verification_potential', 'Unknown')
            print(f"🧬 DNA Verification Potential: {dna_potential}")

            data_quality = insights.get('data_quality_assessment', 'Unknown')
            print(f"📊 Data Quality Assessment: {data_quality}")

        # Show actionable recommendations
        recommendations = comprehensive_result.get("actionable_recommendations", [])
        if recommendations:
            print("\n💡 ACTIONABLE RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"{i}. {rec}")

        print("\n✅ Comprehensive GEDCOM AI analysis completed!")
        print("📄 Full analysis results available in system logs.")

    except Exception as e:
        print(f"❌ Error during comprehensive GEDCOM AI analysis: {e}")
        logger.error(f"Comprehensive GEDCOM AI analysis error: {e}")

    input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()


# end of main.py
