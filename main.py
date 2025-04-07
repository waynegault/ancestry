#!/usr/bin/env python3

# main.py

# --- Standard library imports ---
import gc
import inspect as std_inspect
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
from typing import Optional, Any  # Added Any

# --- Third-party imports  ---
from selenium.webdriver.remote.remote_connection import RemoteConnection
import urllib3.poolmanager
import psutil
import urllib3
from selenium.common.exceptions import TimeoutException
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
from chromedriver import cleanup_webdrv
from config import config_instance, selenium_config
from database import (
    backup_database,
    Base,
    db_transn,
    delete_database,
    InboxStatus,
    MessageType,
    Person,
)
from logging_config import logger, setup_logging
from my_selectors import (
    INBOX_CONTAINER_SELECTOR,
    MATCH_ENTRY_SELECTOR,
    WAIT_FOR_PAGE_SELECTOR,
)
from utils import (  # Ensure SessionManager is imported
    SessionManager,
    is_elem_there,
    log_in,  # Keep log_in import for now, SessionManager uses it internally
    login_status,  # Keep login_status import, SessionManager uses it internally
    nav_to_page,
    retry,
)


def menu():
    """Display the main menu and return the user's choice."""
    print("Main Menu")
    print("=" * 13)
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
    print("1. Run Actions 6, 7, and 8 Sequentially")
    print("2. Reset Database")
    print("3. Backup Database")
    print("4. Restore Database")
    print("5. Check Login Status")  # Updated Menu Text
    print("6. Gather Matches [start page]")
    print("7. Search Inbox")
    print("8. Send Messages")
    print("9. Delete all rows except the first")
    print("")
    print("t. Toggle Console Log Level (INFO/DEBUG)")
    print("c. Clear Screen")
    print("q. Exit")
    choice = input("\nEnter choice: ").strip().lower()
    return choice


# End of menu


def clear_log_file():
    """Finds the FileHandler, closes it, clears the log file."""
    global logger
    cleared = False
    handler_to_reopen = None
    log_path = None
    try:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler_to_reopen = handler
                log_path = handler.baseFilename
                break
        if handler_to_reopen and log_path:
            if (
                hasattr(handler_to_reopen, "flush")
                and hasattr(handler_to_reopen, "close")
                and hasattr(handler_to_reopen, "stream")
                and handler_to_reopen.stream
            ):
                handler_to_reopen.flush()
                handler_to_reopen.close()
                with open(log_path, "w", encoding="utf-8") as f:
                    pass
                cleared = True
            else:
                logger.warning(
                    f"FileHandler '{log_path}' invalid/closed. Skipping clear."
                )
    except PermissionError as pe:
        logger.warning(f"Permission denied clearing log '{log_path}': {pe}")
    except IOError as e:
        logger.warning(f"IOError clearing log '{log_path}': {e}")
    except Exception as e:
        logger.warning(f"Error clearing log '{log_path}': {e}", exc_info=True)
    return cleared, log_path


# End of clear_log_file


def exec_actn(action_func, session_manager, choice, close_sess=True, *args):
    """
    V3.5 REVISED: Executes action, manages session start/stop using new phases.
    Handles browserless actions and Action 5 specifically.
    Skips Phase 2 (ensure_session_ready) for Action 5.
    Restores original header/footer style.
    Correctly passes session_manager to browserless actions needing it for cleanup.
    """
    import inspect  # Local import

    log_cleared, cleared_log_path = clear_log_file()
    start_time = time.time()
    action_name = action_func.__name__
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    # --- Restore Header Style ---
    logger.info("--------------------------------------")
    logger.info(f"Action {choice}: Starting {action_name}...")
    logger.info("--------------------------------------\n")
    # --- End Restore Header Style ---

    action_result = None
    session_started_by_exec = False
    session_made_ready_by_exec = False  # Keep this flag for potential future use

    browserless_actions = [
        "reset_db_actn",
        "backup_db_actn",
        "restore_db_actn",
        "all_but_first_actn",
    ]

    try:
        needs_browser = action_name not in browserless_actions
        # Determine if Phase 2 (readiness) is needed - skip for Action 5
        needs_ready_session = (
            action_name not in browserless_actions and action_name != "check_login_actn"
        )

        if needs_browser and session_manager:
            # --- Phase 1: Ensure Driver is Live ---
            if not session_manager.driver_live:
                logger.info(
                    f"Driver not live for {action_name}. Starting driver (Phase 1)..."
                )
                start_ok = session_manager.start_sess(
                    action_name=f"{action_name} - Driver Start"
                )
                if not start_ok:
                    raise Exception("Driver Start Failed (Phase 1)")
                session_started_by_exec = True
                logger.info(f"Driver started successfully for {action_name}.")
            else:
                logger.debug(f"Driver already live for {action_name}.")

            # --- Phase 2: Ensure Session is Ready (if needed) ---
            if needs_ready_session:  # Check the flag determined above
                if not session_manager.session_ready:
                    logger.info(
                        f"Session not ready for {action_name}. Ensuring readiness (Phase 2)..."
                    )
                    ready_ok = session_manager.ensure_session_ready(
                        action_name=f"{action_name} - Session Ready"
                    )
                    if not ready_ok:
                        raise Exception("Session Readiness Failed (Phase 2)")
                    session_made_ready_by_exec = True
                    logger.info(f"Session ready for {action_name}.")
                else:
                    logger.debug(f"Session already ready for {action_name}.")
            elif action_name == "check_login_actn":
                logger.debug(
                    f"Skipping Phase 2 (ensure_session_ready) for {action_name}."
                )

        elif needs_browser and not session_manager:
            raise Exception("SessionManager Missing")
        else:
            logger.debug(f"Action {action_name} is browserless.")

        # Execute the action function
        func_sig = inspect.signature(action_func)
        action_args_to_pass = list(args)
        pass_config = "config_instance" in func_sig.parameters
        pass_session_manager = "session_manager" in func_sig.parameters

        # Prepare arguments, ensuring session_manager is passed if needed
        final_args = []
        if pass_session_manager:
            final_args.append(session_manager)
        if pass_config:
            # Ensure config_instance is only added if not already in args
            # (This logic might be simplified depending on how args are always passed)
            if not any(isinstance(a, type(config_instance)) for a in args):
                final_args.append(config_instance)
        final_args.extend(args)  # Add remaining args

        # --- Corrected: Handle keyword args for coord_action specifically ---
        target_func_name = action_func.__name__
        if hasattr(action_func, "__wrapped__"):
            target_func_name = action_func.__wrapped__.__name__

        if target_func_name == "coord_action_func" and "start" in func_sig.parameters:
            start_val = 1
            # Find the integer start value if provided in *args
            int_args = [a for a in args if isinstance(a, int)]
            if int_args:
                start_val = int_args[-1]  # Assume last int is start page

            kwargs_for_action = {"start": start_val}
            # Rebuild final_args for coord, ensuring session_manager and config are first if needed
            coord_args = []
            if pass_session_manager:
                coord_args.append(session_manager)
            if pass_config:
                coord_args.append(config_instance)
            action_result = action_func(*coord_args, **kwargs_for_action)
        else:
            # General case - pass the assembled final_args list
            action_result = action_func(*final_args)

    except Exception as e:
        logger.error(f"Exception during action {action_name}: {e}", exc_info=True)
        action_result = False

    finally:
        # Session Closing Logic
        exec_should_close_browser = False
        if session_manager and session_manager.driver_live:
            if session_started_by_exec and close_sess:
                # If we started the driver in this exec_actn call AND close_sess is True
                exec_should_close_browser = True
                logger.debug(
                    f"Closing session started by {action_name} (close_sess=True)."
                )
            elif action_name != "check_login_actn" and close_sess:
                # If session existed before, but close_sess is True AND it's not Action 5
                exec_should_close_browser = True
                logger.debug(
                    f"Closing pre-existing session after {action_name} (close_sess=True)."
                )
            elif action_name == "check_login_actn":
                # Explicitly keep Action 5 open (unless close_sess was False passed externally - unlikely)
                logger.debug(f"Keeping session live after {action_name} (Action 5).")
            elif not close_sess:
                logger.debug(
                    f"Keeping session live after {action_name} (close_sess=False)."
                )

        if exec_should_close_browser:
            logger.debug(f"Closing browser session after {action_name}...")
            # Pass keep_db based on whether the action was browserless (keep DB for those)
            session_manager.close_sess(keep_db=(action_name in browserless_actions))
            logger.debug(
                f"Browser session closed. DB Pool status: {'Kept' if action_name in browserless_actions else 'Closed'}."
            )
        elif action_name in browserless_actions:
            # If the action was browserless, ensure the DB pool (if held by session_manager) is still closed correctly,
            # unless close_sess was explicitly False (unlikely for browserless).
            # The main 'finally' block in main() handles final cleanup anyway.
            # The action itself should now handle closing the pool it was passed.
            logger.debug(f"No browser session relevant for {action_name}.")

        # Performance Logging
        duration = time.time() - start_time
        # Restore old duration format
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_duration = f"{int(hours)} hr {int(minutes)} min {seconds:.2f} sec"
        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_used = mem_after - mem_before

        # Restore old footer style
        if action_result is False:
            logger.error(
                f"Action {choice} ({action_name}) reported a failure (returned False or exception occurred).\n"
            )

        logger.info("--------------------------------------")
        logger.info(f"Action {choice} ({action_name}) finished.")
        logger.info(f"Duration: {formatted_duration}")
        logger.info(f"Memory used: {mem_used:.1f} MB")
        logger.info("--------------------------------------\n")
        # End Restore old footer style


# End of exec_actn


# --- Action Functions (Update required actions) ---


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


# Action 2 (reset_db_actn) - REVISED
def reset_db_actn(session_manager: SessionManager, *args):  # Added session_manager back
    """
    Action to reset the database. Browserless.
    Closes the provided main session pool FIRST.
    """
    db_path = config_instance.DATABASE_FILE
    reset_successful = False
    try:
        # --- Close main pool FIRST ---
        if session_manager:
            logger.warning("Closing main DB connections before reset attempt...")
            session_manager.cls_db_conn(keep_db=False)
            logger.info("Main DB pool closed.")
        else:
            logger.warning("No main session manager passed to reset_db_actn to close.")
        # --- End closing main pool ---

        logger.debug("Running GC before delete attempt...")
        gc.collect()
        time.sleep(0.5)
        gc.collect()  # Short GC pause

        logger.debug("Attempting database file deletion...")
        # Pass None for session_manager to delete_database as it doesn't need it anymore
        delete_database(None, db_path, max_attempts=5)
        logger.info(f"Database file '{db_path}' deleted/gone.")

        # Re-initialize DB and Seed using a temporary manager
        logger.debug("Re-initializing DB...")
        recreation_manager = SessionManager()  # Create temporary one
        recreation_session = None
        try:
            # Initialize engine/session factory for the temp manager
            recreation_manager._initialize_db_engine_and_session()
            if not recreation_manager.engine or not recreation_manager.Session:
                raise SQLAlchemyError("Failed to re-initialize DB engine/session!")

            # Create tables using the temp manager's engine
            Base.metadata.create_all(recreation_manager.engine)
            logger.debug("Tables created.")

            # Seed message types using the temp manager's session
            recreation_session = recreation_manager.get_db_conn()
            if not recreation_session:
                raise SQLAlchemyError("Failed to get session for seeding!")

            logger.debug("Seeding message_types...")
            script_dir = Path(__file__).resolve().parent
            messages_file = script_dir / "messages.json"
            if messages_file.exists():
                with messages_file.open("r", encoding="utf-8") as f:
                    messages_data = json.load(f)
                if isinstance(messages_data, dict):
                    with db_transn(recreation_session) as sess:  # Use context manager
                        types_to_add = [
                            MessageType(type_name=name)
                            for name in messages_data
                            if not sess.query(MessageType)
                            .filter_by(type_name=name)
                            .first()
                        ]
                        if types_to_add:
                            sess.add_all(types_to_add)
                            logger.debug(f"Added {len(types_to_add)} message types.")
                        else:
                            logger.debug("Message types already exist.")
                    count = (
                        recreation_session.query(func.count(MessageType.id)).scalar()
                        or 0
                    )
                    logger.debug(f"Verification: {count} message types found.")
                else:
                    logger.error("'messages.json' has incorrect format.")
            else:
                logger.warning(
                    f"'messages.json' not found at '{messages_file}', skipping seeding."
                )

            reset_successful = True
            logger.info("Database reset and re-initialization completed OK.")
        except Exception as seed_e:
            logger.error(
                f"Error during DB re-initialization/seeding: {seed_e}", exc_info=True
            )
            reset_successful = False
        finally:  # Cleanup temp manager
            logger.debug("Cleaning up recreation manager resources...")
            if recreation_session:
                recreation_manager.return_session(recreation_session)
            recreation_manager.cls_db_conn(
                keep_db=False
            )  # Ensure temp engine is disposed
            logger.debug("Recreation manager cleanup finished.")

    except PermissionError as pe:
        logger.error(f"DB reset FAILED: Permissions/lock on '{db_path}'. {pe}")
    except Exception as e:
        logger.error(f"Error during DB reset: {e}", exc_info=True)
    finally:
        logger.debug("Reset DB action finished.")
    return reset_successful


# end of Action 2


# Action 3 (backup_db_actn) - No changes needed, browserless.
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


# Action 4 (restore_db_actn) - REVISED
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
            logger.warning("Closing main DB connections before restore...")
            session_manager.cls_db_conn(keep_db=False)
            logger.info("Main DB pool closed.")
        else:
            logger.warning(
                "No main session manager passed to restore_db_actn to close."
            )
        # --- End closing main pool ---

        logger.info(f"Restoring DB from: {backup_path}")
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


# Action 5 (check_login_actn) - REVISED V5
def check_login_actn(session_manager: SessionManager, *args) -> bool:
    """
    REVISED V5: Checks login status using login_status.
    Relies on exec_actn to ensure driver is live (Phase 1) if needed.
    Does NOT attempt login itself. Does NOT trigger ensure_session_ready. Keeps session open.
    """
    if not session_manager:
        logger.error("SessionManager required for check_login_actn.")
        return False

    logger.info("Verifying login status...")

    # Phase 1 (Driver Start) is handled by exec_actn if needed.
    # We only need to check if driver is live before proceeding.
    if not session_manager.driver_live:
        logger.error("Driver not live. Cannot check login status.")
        # It's possible exec_actn failed Phase 1.
        return False

    # Call login_status directly to check
    status = login_status(session_manager)

    if status is True:
        logger.info("Login verification successful (already logged in).")
        # --- REMOVED: Do not call ensure_session_ready here ---
        return True
    elif status is False:
        logger.warning("Login verification failed (user not logged in).")
        return False
    else:  # Status is None
        logger.error("Login verification failed (critical error during check).")
        return False


# End Action 5


# Action 6 (coord_action wrapper) - REVISED
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


# Action 7 (srch_inbox_actn) - REVISED
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


# Action 8 (send_messages_action) - REVISED
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


# Action 9 (all_but_first_actn) - REVISED
def all_but_first_actn(
    session_manager: SessionManager, *args
):  # Added session_manager back
    """
    Action to delete all 'people' rows except the first. Browserless.
    Closes the provided main session pool FIRST.
    Creates a temporary SessionManager for the delete operation.
    """
    temp_manager = None  # Initialize
    session = None
    success = False
    try:
        # --- Close main pool FIRST ---
        if session_manager:
            logger.warning("Closing main DB connections before delete-all-but-first...")
            session_manager.cls_db_conn(keep_db=False)
            logger.info("Main DB pool closed.")
        else:
            logger.warning(
                "No main session manager passed to all_but_first_actn to close."
            )
        # --- End closing main pool ---

        logger.info("Deleting all but first person record...")
        # Create a temporary SessionManager for this specific operation
        temp_manager = SessionManager()
        session = temp_manager.get_db_conn()
        if session is None:
            raise Exception("Failed to get DB session via temporary manager.")

        with db_transn(session) as sess:
            first = sess.query(Person).order_by(Person.id.asc()).first()
            if not first:
                logger.info("People table empty.")
                return True
            logger.debug(f"Keeping ID: {first.id} ({first.username})")
            to_delete = sess.query(Person).filter(Person.id != first.id).all()
            if not to_delete:
                logger.info("Only one person found.")
            else:
                logger.debug(f"Deleting {len(to_delete)} people...")
                for i, person in enumerate(to_delete):
                    # logger.debug(f"Deleting {i+1}/{len(to_delete)}: {person.username} (ID: {person.id})") # Can be verbose
                    sess.delete(person)
                logger.info(f"Deleted {len(to_delete)} people.")
        success = True
    except Exception as e:
        logger.error(f"Error during deletion: {e}", exc_info=True)
    finally:
        if temp_manager:
            if session:
                temp_manager.return_session(session)
            temp_manager.cls_db_conn(keep_db=False)  # Close the temp pool
        logger.debug("Delete action finished.")
    return success


# end of Action 9


def main():
    global logger, session_manager
    session_manager = None
    was_driver_live = False  # Track if driver was ever successfully started

    try:
        os.system("cls" if os.name == "nt" else "clear")
        print("")
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

        session_manager = SessionManager()  # Instantiate AFTER logging setup

        # --- Main menu loop ---
        while True:
            choice = menu()
            print("")
            # --- Action Dispatching ---
            if choice == "1":
                exec_actn(run_actions_6_7_8_action, session_manager, choice)
            elif choice == "2":
                # --- Modified Handling for Action 2 ---
                # Now pass the current session_manager to exec_actn
                exec_actn(reset_db_actn, session_manager, choice, close_sess=False)
                logger.info("Re-initializing main SessionManager after reset...")
                session_manager = SessionManager()  # Recreate for subsequent actions
                # --- End Modified Handling for Action 2 ---
            elif choice == "3":
                # --- Modified Handling for Action 3 ---
                # Pass session_manager (though backup_db_actn might not use it)
                exec_actn(backup_db_actn, session_manager, choice, close_sess=False)
                # --- End Modified Handling for Action 3 ---
            elif choice == "4":
                # --- Modified Handling for Action 4 ---
                # Pass the current session_manager to exec_actn
                exec_actn(restore_db_actn, session_manager, choice, close_sess=False)
                logger.info("Re-initializing main SessionManager after restore...")
                session_manager = SessionManager()  # Recreate for subsequent actions
                # --- End Modified Handling for Action 4 ---
            elif choice == "5":
                # Action 5 now only checks status, keep session open by default
                exec_actn(check_login_actn, session_manager, choice, close_sess=False)
            elif choice.startswith("6"):
                parts = choice.split()
                start_val = 1
                if len(parts) > 1:
                    try:
                        start_arg = int(parts[1])
                        start_val = start_arg if start_arg > 0 else 1
                    except ValueError:
                        logger.warning(f"Invalid start page '{parts[1]}'. Using 1.")
                # Pass session_manager, config_instance and start_val correctly
                exec_actn(
                    coord_action, session_manager, "6", True, config_instance, start_val
                )
            elif choice == "7":
                exec_actn(srch_inbox_actn, session_manager, choice)
            elif choice == "8":
                exec_actn(send_messages_action, session_manager, choice)
            elif choice == "9":
                # --- Modified Handling for Action 9 ---
                # Pass the current session_manager to exec_actn
                exec_actn(all_but_first_actn, session_manager, choice, close_sess=False)
                logger.info(
                    "Re-initializing main SessionManager after delete-all-but-first..."
                )
                session_manager = SessionManager()  # Recreate for subsequent actions
                # --- End Modified Handling for Action 9 ---

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
                        logger = setup_logging(log_level=new_level_name)
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
                print("Invalid choice.\n")

            # --- Check if driver became live during the action ---
            if (
                session_manager  # Check if session_manager exists (wasn't None)
                and hasattr(session_manager, "driver_live")
                and session_manager.driver_live
            ):
                was_driver_live = True

    except KeyboardInterrupt:
        os.system("cls" if os.name == "nt" else "clear")
        print("\nCTRL+C detected. Exiting.")
    except Exception as e:
        if "logger" in globals() and logger:
            logger.critical(f"Critical error in main: {e}", exc_info=True)
        else:
            print(f"CRITICAL ERROR (no logger): {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    finally:
        # Use session_manager if initialized (it might be None after Action 2/4/9)
        if "session_manager" in locals() and session_manager:
            # --- Corrected Check for final cleanup log message ---
            should_log_cleanup = False
            if was_driver_live:  # Check if driver was *ever* live during the run
                should_log_cleanup = True
            elif (
                session_manager
                and hasattr(session_manager, "driver_live")
                and session_manager.driver_live
            ):  # Check if *currently* live
                should_log_cleanup = True
            # --- End Corrected Check ---

            if should_log_cleanup:
                log_msg = "Performing final browser cleanup..."
                if logger:
                    logger.info(log_msg)
                else:
                    print(log_msg, file=sys.stderr)

            # close_sess handles driver quit and DB pool dispose safely
            session_manager.close_sess(
                keep_db=False
            )  # Ensure DB pool is closed on final exit

        elif not ("session_manager" in locals() and session_manager):
            logger.info(
                "Session Manager was None during final cleanup (expected after reset/restore/delete)."
            )

        if "logger" in globals() and logger:
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
