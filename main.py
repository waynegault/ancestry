#!/usr/bin/env python3

# main.py

# --- Standard library imports ---
import gc
import inspect
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urljoin

# --- Third-party imports ---
import psutil
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

# --- Local application imports ---
# Action modules
from action6_gather import coord  # Import the main DNA match gathering function
from action7_inbox import InboxProcessor
from action8_messaging import send_messages_to_matches
from action9_process_productive import process_productive_messages
from action10 import run_action10
from action11 import run_action11

# Core modules
from config import config_instance
from database import (
    backup_database,
    Base,
    db_transn,
    MessageType,
    Person,
    ConversationLog,
    DnaMatch,
    FamilyTree,
)
from logging_config import logger, setup_logging
from my_selectors import WAIT_FOR_PAGE_SELECTOR
from utils import (
    SessionManager,
    login_status,
    nav_to_page,
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
            with open(log_file_path, "w", encoding="utf-8"):
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

    logger.info("\n------------------------------------------")
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
        "run_action11_wrapper",  # API Report (Ancestry Online)
    ]

    # Set browser_needed flag based on action
    requires_browser = action_name not in browserless_actions
    session_manager.browser_needed = requires_browser

    # Determine the required session state for the action
    required_state = "none"  # Default for actions that don't need any special state

    if requires_browser:
        required_state = "session_ready"  # Full session with browser
    else:
        required_state = "db_ready"  # Database-only session

    try:
        # --- Ensure Required State ---
        state_ok = True
        if required_state == "db_ready":
            state_ok = session_manager.ensure_db_ready()
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
        # Prepare arguments for action function call
        func_sig = inspect.signature(action_func)
        pass_config = "config_instance" in func_sig.parameters
        pass_session_manager = "session_manager" in func_sig.parameters

        final_args = []
        if pass_session_manager:
            final_args.append(session_manager)
        if pass_config:
            final_args.append(config_instance)
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
        if should_close and isinstance(session_manager, SessionManager):
            if session_manager.browser_needed and session_manager.driver_live:
                logger.debug(f"Closing browser session...")
                # Close browser but keep DB connections for most actions
                session_manager.close_browser()
                logger.debug(f"Browser session closed. DB connections kept.")
            elif should_close and action_name in ["all_but_first_actn"]:
                # For specific actions, close everything including DB
                logger.debug(f"Closing all connections including database...")
                session_manager.close_sess(keep_db=False)
                logger.debug(f"All connections closed.")
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
def all_but_first_actn(session_manager: SessionManager, *_):
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
                .filter(
                    Person.profile_id == profile_id_to_keep, Person.deleted_at == None
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
def run_actions_6_7_8_action(session_manager, *_):
    """
    Action to run actions 6, 7, and 8 sequentially.
    Relies on exec_actn ensuring session is ready beforehand.
    """
    # Guard clause now checks session_ready
    if not session_manager or not session_manager.session_ready:
        logger.error("Cannot run sequential actions: Session not ready.")
        return False

    try:
        # --- Action 6 ---
        logger.info("--- Running Action 6: Gather Matches (Always from page 1) ---")
        print("Starting DNA match gathering from page 1...")
        # Call the coord_action function which wraps the coord function
        gather_result = coord_action(session_manager, config_instance, start=1)
        if gather_result is False:
            logger.error("Action 6 FAILED.")
            print("ERROR: Match gathering failed. Check logs for details.")
            return False
        else:
            logger.info("Action 6 OK.")
            print("✓ Match gathering completed successfully.")

        # --- Action 7 ---
        logger.info("--- Running Action 7: Search Inbox ---")
        inbox_url = urljoin(config_instance.BASE_URL, "/messaging/")
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
                print(
                    "ERROR: Could not navigate to inbox page. Check network connection."
                )
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
                print("ERROR: Inbox search failed. Check logs for details.")
                return False
            else:
                logger.info("Action 7 OK.")
                print("✓ Inbox search completed successfully.")
        except Exception as inbox_error:
            logger.error(
                f"Action 7 FAILED with exception: {inbox_error}", exc_info=True
            )
            print(f"ERROR during inbox search: {inbox_error}")
            return False

        # --- Action 8 ---
        logger.info("--- Running Action 8: Send Messages ---")
        logger.debug("Navigating to Base URL for Action 8...")

        try:
            if not nav_to_page(
                session_manager.driver,
                config_instance.BASE_URL,
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
                print("ERROR: Message sending failed. Check logs for details.")
                return False
            else:
                logger.info("Action 8 OK.")
                print("✓ Message sending completed successfully.")
        except Exception as message_error:
            logger.error(
                f"Action 8 FAILED with exception: {message_error}", exc_info=True
            )
            print(f"ERROR during message sending: {message_error}")
            return False

        logger.info("Sequential Actions 6-7-8 finished successfully.")
        print("\n✓ All sequential actions (6-7-8) completed successfully.")
        return True
    except Exception as e:
        logger.error(
            f"Critical error during sequential actions 6-7-8: {e}", exc_info=True
        )
        print(f"CRITICAL ERROR during sequential actions: {e}")
        return False


# End Action 1


# Action 2 (reset_db_actn)
def reset_db_actn(session_manager: SessionManager, *_):
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
            # Instead of using delete_database, we'll truncate all tables
            # This is a safer approach that doesn't require deleting the file
            logger.debug("Creating a temporary session manager to truncate tables...")
            temp_truncate_manager = SessionManager()
            truncate_session = temp_truncate_manager.get_db_conn()

            if truncate_session:
                logger.debug("Truncating all tables...")
                with db_transn(truncate_session) as sess:
                    # Delete all records from tables in reverse order of dependencies
                    sess.query(ConversationLog).delete(synchronize_session=False)
                    sess.query(DnaMatch).delete(synchronize_session=False)
                    sess.query(FamilyTree).delete(synchronize_session=False)
                    sess.query(Person).delete(synchronize_session=False)
                    # Keep MessageType table intact

                # Close the temporary session manager
                temp_truncate_manager.cls_db_conn(keep_db=False)
                logger.debug("All tables truncated successfully.")
            else:
                logger.critical(
                    "Failed to get session for truncating tables. Reset aborted."
                )
                return False
        except Exception:
            logger.critical(
                f"Failed to reset database tables. Reset aborted.",
                exc_info=True,
            )
            return False  # Critical failure if deletion fails

        # --- 3. Re-initialize DB Schema ---
        logger.debug("Re-initializing database schema...")
        # Use a temporary SessionManager to handle recreation
        temp_manager = SessionManager()
        try:
            # This will create a new engine and session factory pointing to the file path
            temp_manager._initialize_db_engine_and_session()
            if not temp_manager.engine or not temp_manager.Session:
                raise SQLAlchemyError(
                    "Failed to initialize DB engine/session for recreation!"
                )

            # This will recreate the tables in the existing file
            Base.metadata.create_all(temp_manager.engine)
            logger.debug("Database schema recreated successfully.")

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
                        # First check if there are any existing message types
                        existing_count = (
                            sess.query(func.count(MessageType.id)).scalar() or 0
                        )

                        if existing_count > 0:
                            logger.debug(
                                f"Found {existing_count} existing message types. Skipping seeding."
                            )
                        else:
                            # Only add message types if none exist
                            types_to_add = [
                                MessageType(type_name=name) for name in messages_data
                            ]
                            if types_to_add:
                                sess.add_all(types_to_add)
                                logger.debug(
                                    f"Added {len(types_to_add)} message types."
                                )
                            else:
                                logger.warning(
                                    "No message types found in messages.json to seed."
                                )

                    count = (
                        recreation_session.query(func.count(MessageType.id)).scalar()
                        or 0
                    )
                    logger.debug(
                        f"MessageType seeding complete. Total types in DB: {count}"
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
    session_manager: Optional[SessionManager] = None, *_
):  # Added session_manager parameter for exec_actn compatibility
    """Action to backup the database. Browserless."""
    try:
        logger.debug("Starting DB backup...")
        # session_manager isn't used but needed for exec_actn compatibility
        result = backup_database()
        if result:
            logger.info("DB backup OK.")
            return True
        else:
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
    backup_dir = config_instance.DATA_DIR
    db_path = config_instance.DATABASE_FILE
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
def check_login_actn(session_manager: SessionManager, *_) -> bool:
    """
    REVISED V7: Checks login status using login_status and provides clear user feedback.
    Relies on exec_actn to ensure driver is live (Phase 1) if needed.
    Does NOT attempt login itself. Does NOT trigger ensure_session_ready. Keeps session open.
    Improved error handling and user feedback.
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

    # Call login_status directly to check
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
        elif status is False:
            print("\n✗ You are NOT currently logged in to Ancestry.")
            print("  Select option 1, 6, 7, 8, 9, or 11 to trigger automatic login.")
            return False
        else:  # Status is None
            print("\n? Unable to determine login status due to a technical error.")
            print(
                "  Try selecting option 1, 6, 7, 8, 9, or 11 to trigger automatic login."
            )
            logger.warning("Login status check returned None (ambiguous result).")
            return False
    except Exception as e:
        logger.error(f"Exception during login status check: {e}", exc_info=True)
        print(f"\n! Error checking login status: {e}")
        print("  Try selecting option 1, 6, 7, 8, 9, or 11 to trigger automatic login.")
        return False


# End Action 5


# Action 6 (coord_action wrapper)
def coord_action(session_manager, config_instance, start=1):
    """
    Action wrapper for gathering matches (coord function from action6).
    Relies on exec_actn ensuring session is ready before calling.

    Args:
        session_manager: The SessionManager instance.
        config_instance: The configuration instance.
        start: The page number to start gathering from (default is 1).
        *args: Additional arguments that might be passed by exec_actn (ignored).
        **kwargs: Additional keyword arguments that might be passed by exec_actn (ignored).
    """
    # Guard clause now checks session_ready
    if not session_manager or not session_manager.session_ready:
        logger.error("Cannot gather matches: Session not ready.")
        print("ERROR: Session not ready. Cannot gather matches.")
        return False

    print(f"Gathering DNA Matches from page {start}...")
    try:
        # Call the imported function from action6
        result = coord(session_manager, config_instance, start=start)
        if result is False:
            logger.error("Match gathering reported failure.")
            print("ERROR: Match gathering failed. Check logs for details.")
            return False
        else:
            logger.info("Gathering matches OK.")
            print("✓ Match gathering completed successfully.")
            return True
    except Exception as e:
        logger.error(f"Error during coord_action: {e}", exc_info=True)
        print(f"ERROR: Exception during match gathering: {e}")
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
        else:
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
        else:
            logger.info("Productive message processing OK.")
            return True
    except Exception as e:
        logger.error(f"Error during productive message processing: {e}", exc_info=True)
        return False


# End of process_productive_messages_action


# Action 11 (run_action11_wrapper)
def run_action11_wrapper(session_manager, *_):
    """Action to run API Report. Relies on exec_actn for consistent logging and error handling."""
    # Note: session_manager is not used but is required for exec_actn compatibility
    logger.debug("Starting API Report...")
    try:
        # Call the actual API Report function
        result = run_action11()
        if result is False:
            logger.error("API Report reported failure.")
            return False
        else:
            logger.info("API Report OK.")
            return True
    except Exception as e:
        logger.error(f"Error during API Report: {e}", exc_info=True)
        return False


# End of run_action11_wrapper


def main():
    global logger, session_manager  # Ensure global logger can be modified
    session_manager = None  # Initialize session_manager

    try:
        print("")
        # --- Logging Setup ---
        logger = setup_logging()

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
                else:
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
                    # Recreate manager after full reset
                    session_manager.close_sess(keep_db=False)
                    session_manager = SessionManager()
                elif choice == "3":
                    exec_actn(
                        backup_db_actn, session_manager, choice
                    )  # Run through exec_actn for consistent logging
                elif choice == "4":
                    # Confirmation handled above
                    exec_actn(restore_db_actn, session_manager, choice)
                    # Recreate manager after restore
                    session_manager.close_sess(keep_db=False)
                    session_manager = SessionManager()

            # --- Browser-required actions ---
            elif choice == "1":
                # exec_actn will set browser_needed=True based on the action
                exec_actn(
                    run_actions_6_7_8_action,
                    session_manager,
                    choice,
                    close_sess_after=True,
                )  # Close after full sequence
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
                        print(f"Invalid start page '{parts[1]}'. Using page 1 instead.")

                print(f"Starting DNA match gathering from page {start_val}...")
                # Call exec_actn with the correct parameters
                exec_actn(
                    coord_action,
                    session_manager,
                    "6",
                    False,  # don't close session after
                    config_instance,
                    start_val,
                )  # Keep open
            elif choice == "7":
                exec_actn(srch_inbox_actn, session_manager, choice)  # Keep open
            elif choice == "8":
                exec_actn(send_messages_action, session_manager, choice)  # Keep open
            elif choice == "9":
                exec_actn(
                    process_productive_messages_action, session_manager, choice
                )  # Keep open
            elif choice == "10":
                exec_actn(run_action10, session_manager, choice)
            elif choice == "11":
                # Use the wrapper function to run Action 11 through exec_actn
                exec_actn(run_action11_wrapper, session_manager, choice)
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
        print("\nExecution finished.")


# end main

# --- Entry Point ---
if __name__ == "__main__":
    main()


# end of main.py
