#!/usr/bin/env python3

# main.py

# --- Standard library imports ---
import gc
import inspect  # Keep standard inspect
import json
import logging
import os
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path
from urllib.parse import urljoin
from typing import Optional, Any, Tuple

# --- Third-party imports ---
from selenium.webdriver.remote.remote_connection import RemoteConnection
import urllib3.poolmanager
import psutil
import urllib3
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
)  # Added WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from sqlalchemy import create_engine, event, func, inspect as sa_inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

# --- Local application imports ---
from action6_gather import coord as coord_action_func, nav_to_list
from action7_inbox import InboxProcessor
from action8_messaging import send_messages_to_matches
from action9_process_productive import process_productive_messages

# Ensure that run_action10 is defined in action10.py or remove this line if not needed
try:
    from action10 import run_action10
except ImportError:
    print("WARNING: 'run_action10' is not defined in 'action10'. Please verify.")
    run_action10 = None
from action11 import run_action11
from chromedriver import cleanup_webdrv
from config import config_instance, selenium_config
from database import (
    backup_database,
    Base,
    db_transn,
    delete_database,
    MessageType,
    Person,
    ConversationLog,
    DnaMatch,
    FamilyTree,
)
import database
from logging_config import logger, setup_logging
from my_selectors import (
    INBOX_CONTAINER_SELECTOR,
    MATCH_ENTRY_SELECTOR,
    WAIT_FOR_PAGE_SELECTOR,
)
from utils import (
    SessionManager,
    is_elem_there,
    log_in,
    login_status,
    nav_to_page,
    retry,
)


def menu():
    """Display the main menu and return the user's choice."""
    print("Main Menu")
    print("=" * 17)
    level_name = "UNKNOWN"  # Default

    if logger and logger.handlers:
        try:
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
        except Exception as e:
            print(f"DEBUG: Error getting console handler level: {e}", file=sys.stderr)
            level_name = "ERROR"
    elif "config_instance" in globals() and hasattr(config_instance, "LOG_LEVEL"):
        level_name = config_instance.LOG_LEVEL.upper()

    print(f"(Log Level: {level_name})\n")
    print("0. Delete all rows except the first")
    print("1. Run Actions 6, 7, and 8 Sequentially")
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
    print("t. Toggle Console Log Level (INFO/DEBUG)")
    print("c. Clear Screen")
    print("q. Exit")
    choice = input("\nEnter choice: ").strip().lower()
    return choice


# End of menu


def clear_log_file() -> Tuple[bool, Optional[str]]:
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
            with open(log_file_path, "w", encoding="utf-8") as f:
                pass
            cleared = True
    except PermissionError as permission_error:
        # Handle permission errors when attempting to open the log file
        logger.warning(
            f"Permission denied clearing log '{log_file_path}': {permission_error}"
        )
    except IOError as io_error:
        # Handle I/O errors when attempting to open the log file
        logger.warning(f"IOError clearing log '{log_file_path}': {io_error}")
    except Exception as error:
        # Handle any other exceptions during the log clearing process
        logger.warning(f"Error clearing log '{log_file_path}': {error}", exc_info=True)
    return cleared, log_file_path


# End of clear_log_file


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

    # Determine the required session state for the action
    required_state = "none"  # Default for browserless actions
    # Actions that are DB-only and do NOT require browser or session
    browserless_actions = [
        "all_but_first_actn",
        "reset_db_actn",
        "backup_db_actn",
        "restore_db_actn",
        "run_action10",  # Action 10 does not require session
    ]
    if action_name in browserless_actions:
        required_state = "none"
    elif action_name not in ["all_but_first_actn"]:
        required_state = "session_ready"

    try:
        # --- Ensure Required State ---
        state_ok = True
        if required_state == "driver_live":
            state_ok = session_manager.ensure_driver_live(
                action_name=f"{action_name} - Setup"
            )
        elif required_state == "session_ready":
            state_ok = session_manager.ensure_session_ready(
                action_name=f"{action_name} - Setup"
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
        func_sig = inspect.signature(action_func)
        pass_config = "config_instance" in func_sig.parameters
        pass_session_manager = "session_manager" in func_sig.parameters

        final_args = []
        if pass_session_manager:
            final_args.append(session_manager)
        if pass_config:
            final_args.append(config_instance)
        final_args.extend(args)

        # Handle keyword args specifically for coord_action_func
        if (
            action_name in ("coord_action_func", "coord_action")
            and "start" in func_sig.parameters
        ):
            start_val = 1
            int_args = [a for a in args if isinstance(a, int)]
            if int_args:
                start_val = int_args[-1]
            kwargs_for_action = {"start": start_val}
            # Prepare coord_action_func specific positional args

            coord_args = []
            if pass_session_manager:
                coord_args.append(session_manager)
            if pass_config:
                coord_args.append(config_instance)
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

        # Perform close if needed and possible
        if (
            should_close
            and isinstance(session_manager, SessionManager)
            and session_manager.driver_live
        ):
            logger.debug(f"Closing browser session...")
            # Keep DB pool for browserless actions if closing due to error, else close pool
            session_manager.close_sess(keep_db=(action_name in ["all_but_first_actn"]))
            logger.debug(
                f"Browser session closed. DB Pool status: {'Kept' if action_name in ['all_but_first_actn'] else 'Closed'}."
            )
        # Log if session is kept open
        elif (
            isinstance(session_manager, SessionManager)
            and session_manager.driver_live
            and not should_close
        ):
            logger.debug(f"Keeping session live after '{action_name}'.")

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

        logger.info("------------------------------------------")
        logger.info(f"Action {choice} ({action_name}) finished.")
        logger.info(f"Duration: {formatted_duration}")
        logger.info(mem_log)
        logger.info("------------------------------------------\n")

    # --- Return Action Result ---
    # Return True only if action completed without exception AND didn't return False explicitly
    final_outcome = action_result is not False and action_exception is None
    logger.debug(
        f"Final outcome for Action {choice} ('{action_name}'): {final_outcome}"
    )
    return final_outcome


# End of exec_actn


# --- Action Functions


# Action 0 (all_but_first_actn)
def all_but_first_actn(session_manager: SessionManager, *args):
    """
    V1.2: Modified to delete records from people, conversation_log,
          dna_match, and family_tree, except for the person with a
          specific profile_id. Leaves message_types untouched. Browserless.
    Closes the provided main session pool FIRST.
    Creates a temporary SessionManager for the delete operation.
    """
    # Define the specific profile ID to keep (ensure it's uppercase for comparison)
    profile_id_to_keep = "08FA6E79-0006-0000-0000-000000000000".upper()

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
                .filter(Person.profile_id == profile_id_to_keep)
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
def run_actions_6_7_8_action(session_manager, *args):
    """
    Action to run actions 6, 7, and 8 sequentially.
    Relies on exec_actn ensuring session is ready beforehand.
    """
    # Guard clause now checks session_ready
    if not session_manager or not session_manager.session_ready:
        logger.error("Cannot run sequential actions: Session not ready.")
        return False

    all_successful = True
    try:
        # --- Action 6 ---
        logger.info("--- Running Action 6: Gather Matches (Always from page 1) ---")
        # coord_action_func expects session_manager, config_instance, start=...
        # config_instance is needed
        gather_result = coord_action_func(session_manager, config_instance, start=1)
        if gather_result is False:
            logger.error("Action 6 FAILED.")
            return False
        else:
            logger.info("Action 6 OK.")

        # --- Action 7 ---
        logger.info("--- Running Action 7: Search Inbox ---")
        inbox_url = urljoin(config_instance.BASE_URL, "/messaging/")
        logger.debug(f"Navigating to Inbox ({inbox_url}) for Action 7...")
        # Wait for a specific element indicating inbox has loaded
        if not nav_to_page(
            session_manager.driver,
            inbox_url,
            "div[data-testid='conversation-list-item']",  # Selector for a conversation item
            session_manager,
        ):
            logger.error("Action 7 nav FAILED.")
            return False
        logger.debug("Nav OK. Running search...")
        inbox_processor = InboxProcessor(session_manager=session_manager)
        search_result = inbox_processor.search_inbox()
        if search_result is False:
            logger.error("Action 7 FAILED.")
            return False
        else:
            logger.info("Action 7 OK.")

        # --- Action 8 ---
        logger.info("--- Running Action 8: Send Messages ---")
        logger.debug("Navigating to Base URL for Action 8...")
        if not nav_to_page(
            session_manager.driver,
            config_instance.BASE_URL,
            WAIT_FOR_PAGE_SELECTOR,  # Use a general page load selector
            session_manager,
        ):
            logger.error("Action 8 nav FAILED.")
            return False
        logger.debug("Nav OK. Sending messages...")
        # send_messages_to_matches expects session_manager
        send_result = send_messages_to_matches(session_manager)
        if send_result is False:
            logger.error("Action 8 FAILED.")
            return False
        else:
            logger.info("Action 8 OK.")

        logger.info("Sequential Actions 6-7-8 finished successfully.")
        return True
    except Exception as e:
        logger.error(
            f"Critical error during sequential actions 6-7-8: {e}", exc_info=True
        )
        return False


# End Action 1


# Action 2 (reset_db_actn)
def reset_db_actn(session_manager: SessionManager, *args):
    """
    Action to COMPLETELY reset the database by deleting the file. Browserless.
    - Closes main pool.
    - Deletes the .db file.
    - Recreates schema from scratch.
    - Seeds the MessageType table.
    """
    db_path = config_instance.DATABASE_FILE
    reset_successful = False
    temp_manager = None  # For recreation/seeding
    recreation_session = None  # Session for seeding

    try:
        # --- 1. Close main pool FIRST ---
        if session_manager:
            logger.debug("Closing main DB connections before database deletion...")
            session_manager.cls_db_conn(keep_db=False)  # Ensure pool is closed
            logger.debug("Main DB pool closed.")
        else:
            logger.warning("No main session manager passed to reset_db_actn to close.")

        # --- 2. Delete the Database File ---
        logger.debug(f"Attempting to delete database file: {db_path}...")
        try:
            # Call delete_database function from the database module
            database.delete_database(None, db_path)  # Pass None for session_manager
            logger.debug(f"Database file '{db_path.name}' deleted successfully.")
        except Exception as del_err:
            logger.critical(
                f"Failed to delete database file '{db_path.name}'. Reset aborted.",
                exc_info=True,
            )
            return False  # Critical failure if deletion fails

        # --- 3. Re-initialize DB Schema and Seed ---
        logger.debug("Re-initializing database schema and seeding MessageTypes...")
        # Use a temporary SessionManager to handle creation on the now non-existent file path
        temp_manager = SessionManager()
        try:
            # This will create a new engine and session factory pointing to the file path
            temp_manager._initialize_db_engine_and_session()
            if not temp_manager.engine or not temp_manager.Session:
                raise SQLAlchemyError(
                    "Failed to initialize DB engine/session for recreation!"
                )

            # This will create the tables in the new, empty file
            Base.metadata.create_all(temp_manager.engine)
            logger.debug("New database file created and tables schema applied.")

            # --- Seed MessageType Table ---
            recreation_session = temp_manager.get_db_conn()
            if not recreation_session:
                raise SQLAlchemyError("Failed to get session for seeding MessageTypes!")

            logger.debug("Seeding message_types table...")
            script_dir = Path(__file__).resolve().parent
            messages_file = script_dir / "messages.json"
            if messages_file.exists():
                with messages_file.open("r", encoding="utf-8") as f:
                    messages_data = json.load(f)
                if isinstance(messages_data, dict):
                    # Use the session from the temporary manager
                    with db_transn(recreation_session) as sess:
                        types_to_add = [
                            MessageType(type_name=name)
                            for name in messages_data
                            # No need to check existence in an empty DB
                        ]
                        if types_to_add:
                            sess.add_all(types_to_add)
                            logger.debug(f"Added {len(types_to_add)} message types.")
                        else:
                            logger.warning(
                                "No message types found in messages.json to seed."
                            )
                    count = (
                        recreation_session.query(func.count(MessageType.id)).scalar()
                        or 0
                    )
                    logger.debug(
                        f"MessageType seeding OK. Total types in new DB: {count}"
                    )
                else:
                    logger.error("'messages.json' has incorrect format. Cannot seed.")
            else:
                logger.warning(f"'messages.json' not found. Cannot seed MessageTypes.")
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
    session_manager: Optional[SessionManager], *args
):  # Added session_manager back (Optional)
    """Action to backup the database. Browserless."""
    try:
        logger.debug("Starting DB backup...")
        # session_manager isn't strictly needed but kept for signature consistency
        backup_database()
        logger.info("DB backup OK.")
        return True
    except Exception as e:
        logger.error(f"Error during DB backup: {e}", exc_info=True)
        return False


# end of Action 3


# Action 4 (restore_db_actn)
def restore_db_actn(
    session_manager: SessionManager, *args
):  # Added session_manager back
    """
    Action to restore the database. Browserless.
    Closes the provided main session pool FIRST.
    """
    backup_dir = config_instance.DATA_DIR
    backup_path = backup_dir / "ancestry_backup.db"
    db_path = config_instance.DATABASE_FILE
    success = False
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
        logger.info(f"Db restored from backup OK.")
        success = True
    except FileNotFoundError:
        logger.error(f"Backup not found during copy: {backup_path}")
    except (OSError, IOError, shutil.Error) as e:
        logger.error(f"Error restoring DB: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"Unexpected restore error: {e}", exc_info=True)
    finally:
        logger.debug("DB restore action finished.")
    return success


# end of Action 4


# Action 5 (check_login_actn)
def check_login_actn(session_manager: SessionManager, *args) -> bool:
    """
    Action 5: Ensure browser and session readiness (start driver, login, identifiers).
    """
    if not session_manager:
        logger.error("SessionManager required for check_login_actn.")
        return False

    logger.debug("Checking driver and session readiness (Action 5)...")

    # Phase 1: Ensure WebDriver is initialized and on base URL
    if not session_manager.ensure_driver_live(action_name="check_login_actn - Setup"):
        logger.error("Failed to start browser session.")
        return False

    # Phase 2: Ensure full session readiness (login, CSRF, identifiers)
    if not session_manager.ensure_session_ready(action_name="check_login_actn - Setup"):
        logger.error("Failed to authenticate with Ancestry.")
        return False

    logger.info("Session is ready. Login verification successful.")
    return True


# End of check_login_actn


# Action 6 (coord_action wrapper)
def coord_action(session_manager, config_instance, start=1):
    """
    Action wrapper for gathering matches (coord_action_func).
    Relies on exec_actn ensuring session is ready before calling.
    """
    # Guard clause now checks session_ready
    if not session_manager or not session_manager.session_ready:
        logger.error("Cannot gather matches: Session not ready.")
        return False

    logger.debug(f"Gathering DNA Matches from page {start}...")
    try:
        # Call the imported function (coord_action_func from action6_gather)
        result = coord_action_func(session_manager, config_instance, start=start)
        if result is False:
            logger.error("Match gathering reported failure.")
            return False
        else:
            logger.info("Gathering matches OK.")
            return True  # Use INFO for success
    except Exception as e:
        logger.error(f"Error during coord_action: {e}", exc_info=True)
        return False


# End of coord_action


# Action 7 (srch_inbox_actn)
def srch_inbox_actn(session_manager, *args):
    """Action to search the inbox. Relies on exec_actn ensuring session is ready."""
    # Guard clause now checks session_ready
    if not session_manager or not session_manager.session_ready:
        logger.error("Cannot search inbox: Session not ready.")
        return False

    logger.debug("Starting inbox search...")
    try:
        processor = InboxProcessor(session_manager=session_manager)
        result = processor.search_inbox()
        if result is False:
            logger.error("Inbox search reported failure.")
            return False
        else:
            logger.info("Inbox search OK.")
            return True  # Use INFO
    except Exception as e:
        logger.error(f"Error during inbox search: {e}", exc_info=True)
        return False


# End of srch_inbox_actn


# Action 8 (send_messages_action)
def send_messages_action(session_manager, *args):
    """Action to send messages. Relies on exec_actn ensuring session is ready."""
    # Guard clause now checks session_ready
    if not session_manager or not session_manager.session_ready:
        logger.error("Cannot send messages: Session not ready.")
        return False

    logger.debug("Starting message sending...")
    try:
        # Navigate to Base URL first (good practice before starting message loops)
        logger.debug("Navigating to Base URL before sending...")
        if not nav_to_page(
            session_manager.driver,
            config_instance.BASE_URL,
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
        else:
            logger.info("Messages sent OK.")
            return True  # Use INFO
    except Exception as e:
        logger.error(f"Error during message sending: {e}", exc_info=True)
        return False


# End of send_messages_action


# Action 9
# end of Action 9


def main():
    global logger, session_manager  # Ensure global logger can be modified
    session_manager = None  # Initialize session_manager
    was_driver_live = False  # Track if driver was ever live

    try:
        os.system("cls" if os.name == "nt" else "clear")
        print("")
        # --- Logging Setup ---
        try:
            from config import config_instance  # Local import for clarity

            db_file_path = config_instance.DATABASE_FILE
            log_filename = db_file_path.with_suffix(".log").name
            log_level_to_set = getattr(config_instance, "LOG_LEVEL", "INFO").upper()
            logger = setup_logging(log_file=log_filename, log_level=log_level_to_set)
        except Exception as log_setup_e:
            print(f"CRITICAL: Logging setup error: {log_setup_e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            logging.basicConfig(
                level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
            )
            logger = logging.getLogger("logger_fallback")
            logger.warning("Using fallback logging.")

        # --- Instantiate SessionManager ---
        session_manager = SessionManager()

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
                else:
                    print(" ")  # Newline after confirmation

            # --- Action Dispatching ---
            # Note: Removed most close_sess_after=False as the default is now to keep open.
            # Added close_sess_after=True only where explicit closure after action is desired.
            if choice == "0":
                # Confirmation handled above
                exec_actn(all_but_first_actn, session_manager, choice)
                # Recreate manager as DB structure might have changed significantly
                # Although all_but_first closes the *main* pool, it uses a temp one.
                # Recreating ensures the main manager points to the potentially recreated DB file correctly.
                # session_manager.close_sess(keep_db=False) # Ensure old manager state is cleared
                # session_manager = SessionManager() # This might be too disruptive if other state needed?
                # For now, assume the action handled DB closure correctly and main manager state is okay.
            elif choice == "1":
                exec_actn(
                    run_actions_6_7_8_action,
                    session_manager,
                    choice,
                    close_sess_after=True,
                )  # Close after full sequence
            elif choice == "2":
                # Confirmation handled above
                exec_actn(reset_db_actn, session_manager, choice)
                # Recreate manager after full reset
                session_manager.close_sess(
                    keep_db=False
                )  # Ensure old manager state is cleared
                session_manager = SessionManager()
            elif choice == "3":
                exec_actn(backup_db_actn, session_manager, choice)
            elif choice == "4":
                # Confirmation handled above
                exec_actn(restore_db_actn, session_manager, choice)
                # Recreate manager after restore
                session_manager.close_sess(
                    keep_db=False
                )  # Ensure old manager state is cleared
                session_manager = SessionManager()
            elif choice == "5":
                exec_actn(check_login_actn, session_manager, choice)  # Keep open
            elif choice.startswith("6"):
                parts = choice.split()
                start_val = 1
                if len(parts) > 1:
                    try:
                        start_arg = int(parts[1])
                        start_val = start_arg if start_arg > 0 else 1
                    except ValueError:
                        logger.warning(f"Invalid start page '{parts[1]}'. Using 1.")
                # Call exec_actn correctly passing config_instance and start_val as part of *args
                exec_actn(
                    coord_action,
                    session_manager,
                    "6",
                    False,
                    config_instance,
                    start_val,
                )  # Keep open
            elif choice == "7":
                exec_actn(srch_inbox_actn, session_manager, choice)  # Keep open
            elif choice == "8":
                exec_actn(send_messages_action, session_manager, choice)  # Keep open
            elif choice == "9":
                exec_actn(
                    process_productive_messages, session_manager, choice
                )  # Keep open
            elif choice == "10":
                exec_actn(run_action10, session_manager, choice)
            elif choice == "11":
                exec_actn(run_action11, session_manager, choice)
            # --- Meta Options ---
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
                        new_level = (
                            logging.DEBUG
                            if current_level > logging.DEBUG
                            else logging.INFO
                        )
                        new_level_name = logging.getLevelName(new_level)
                        # Re-call setup_logging to potentially update filters etc. too
                        logger = setup_logging(log_level=new_level_name)
                        logger.info(f"Console log level toggled to: {new_level_name}")
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
            else:
                # Handle invalid choices
                if choice not in confirm_actions:  # Avoid double 'invalid' message
                    print("Invalid choice.\n")

            # --- Track if driver became live ---
            if (
                session_manager
                and hasattr(session_manager, "driver_live")
                and session_manager.driver_live
            ):
                was_driver_live = (
                    True  # Track if driver was successfully started at any point
                )

    except KeyboardInterrupt:
        os.system("cls" if os.name == "nt" else "clear")
        print("\nCTRL+C detected. Exiting.")
    except Exception as e:
        # Log critical error using the global logger if available
        if "logger" in globals() and logger:
            logger.critical(f"Critical error in main: {e}", exc_info=True)
        else:  # Fallback print if logger failed
            print(f"CRITICAL ERROR (no logger): {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    finally:
        # Final cleanup: Always close the session manager if it exists
        logger_present = "logger" in globals() and logger is not None
        sm_present = "session_manager" in locals() and session_manager is not None

        if logger_present:
            logger.info("Performing final cleanup...")
        else:
            print("Performing final cleanup...", file=sys.stderr)

        if sm_present:
            try:
                # Close session, including DB pool, regardless of keep_db flags used earlier
                if session_manager is not None:
                    session_manager.close_sess(keep_db=False)
                if logger_present:
                    logger.debug("Session Manager closed in final cleanup.")
                else:
                    print("Session Manager closed.", file=sys.stderr)
            except Exception as final_close_e:
                if logger_present:
                    logger.error(
                        f"Error during final Session Manager cleanup: {final_close_e}"
                    )
                else:
                    print(
                        f"Error during final Session Manager cleanup: {final_close_e}",
                        file=sys.stderr,
                    )
        elif logger_present:
            logger.debug("Session Manager was None during final cleanup.")

        # Log program finish
        if logger_present:
            logger.info("--- Main program execution finished ---")
        else:
            print(
                "--- Main program execution finished (logger unavailable) ---",
                file=sys.stderr,
            )
        print("\nExecution finished.")


# end main

# --- Entry Point ---
if __name__ == "__main__":
    main()


# end of main.py
