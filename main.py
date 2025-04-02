#!/usr/bin/env python3

# main.py

# --- Imports ---
from calendar import c
from encodings.punycode import T
import os
import sys
import logging # Keep logging import
import traceback
import time
import shutil
import gc
from sqlalchemy import inspect # Keep necessary sqlalchemy imports
import json
import psutil
from pathlib import Path # Keep Path import if used elsewhere
from urllib.parse import urljoin # Import urljoin to fix the error

# Import necessary functions/classes from other modules
from logging_config import setup_logging, logger # Import the global logger instance
from database import (
    delete_database,
    backup_database,
    db_transn,
    Base,
    MessageType,
    InboxStatus,
    Person,
    ConnectionPool
)
from sqlalchemy import create_engine, event, text, func # Ensure func is imported if used directly here (unlikely)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from utils import (
    SessionManager,
    nav_to_page,
    login_status,
    retry,
    is_elem_there,
    log_in,
    # Add other specific utils imports if needed by actions called directly
)
from chromedriver import cleanup_webdrv
# Import action functions/classes
from action6_gather import nav_to_list, coord as coord_action_func 
from config import config_instance, selenium_config 
from action7_inbox import InboxProcessor
from action8_messaging import send_messages_to_matches
# Import Selenium specifics needed by actions (if not fully encapsulated)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
# Import selectors used directly in main or passed to actions
from my_selectors import INBOX_CONTAINER_SELECTOR, WAIT_FOR_PAGE_SELECTOR, MATCH_ENTRY_SELECTOR

# --- Function Definitions (menu, exec_actn, action functions) ---

def menu():
    """Display the main menu and return the user's choice."""
    print("Main Menu")
    print("=" * 13)
    level_name = "UNKNOWN" # Default

    # Get level NAME from the console handler if logger is initialized and has handlers
    if logger and logger.handlers:
        try:
            console_handler = None
            for handler in logger.handlers:
                # Identify the specific console handler targeting stderr
                if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                    console_handler = handler
                    break

            if console_handler:
                level_name = logging.getLevelName(console_handler.level)
            else: # Fallback if no console handler found (less likely with current setup)
                level_name = logging.getLevelName(logger.getEffectiveLevel()) # Use logger's level
        except Exception as e:
            # Use print as logger might be the source of the issue
            print(f"DEBUG: Error retrieving console handler level: {e}", file=sys.stderr)
            level_name = "ERROR" # Error retrieving level

    elif 'config_instance' in globals() and hasattr(config_instance, 'LOG_LEVEL'):
        # If logger not ready, show the configured initial level
        level_name = config_instance.LOG_LEVEL.upper()
    # else: # Logger not ready and config not available yet
    #     level_name remains "UNKNOWN"

    print(f"(Console: {level_name} | File: DEBUG)\n") # Show console and fixed file level
    print("1. Run Actions 6, 7, and 8 Sequentially")
    print("2. Reset Database")
    print("3. Backup Database")
    print("4. Restore Database")
    print("5. Check Login")
    print("6. Gather Matches [start page]")
    print("7. Search Inbox")
    print("8. Send Messages")
    print("9. Delete all rows except the first")
    print("0. Test URL Speed")
    print("")
    print("t. Toggle Console Log Level (INFO/DEBUG)") # Clarify it's console level
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
                # logger.debug(f"Found FileHandler for path: {log_path}") # Log before closing/clearing
                break # Assume only one file handler

        if handler_to_reopen and log_path:
            # print(f"DEBUG: Attempting to close handler and clear log file: {log_path}") # Use print for immediate feedback
            handler_to_reopen.flush()
            handler_to_reopen.close() # Close the stream
            # logger needs to remove the handler *temporarily* only if open('w') fails otherwise
            # logger.removeHandler(handler_to_reopen) # Maybe not needed if FileHandler handles this

            # Clear the file by opening in write mode
            with open(log_path, 'w', encoding='utf-8') as f:
                pass # Just opening in 'w' truncates the file
            cleared = True
            # print(f"DEBUG: Log file cleared: {log_path}")

            # The FileHandler should automatically reopen the stream on the next log message
            # because delay=False by default. If issues arise, explicitly re-adding might
            # be needed, but try this first.
            # Example of re-adding if needed:
            # logger.addHandler(handler_to_reopen) # Re-attach the handler instance

        # else:
            # print("DEBUG: No FileHandler found, skipping log clear.")

    except PermissionError as pe:
         # Use print because logger might be in a weird state
         print(f"WARNING: Permission denied clearing log file '{log_path}'. Log may not be fresh. Error: {pe}", file=sys.stderr)
    except IOError as e:
        print(f"WARNING: IOError clearing log file '{log_path}'. Log may not be fresh. Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: Unexpected error clearing log file '{log_path}'. Log may not be fresh. Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Print traceback for unexpected errors

    return cleared, log_path
# End of clear_log_file

def exec_actn(action_func, session_manager, choice, close_sess=True, *args):
    # --- NEW: Clear log file at the start of executing an action ---
    log_cleared, cleared_log_path = clear_log_file()
    start_time = time.time()
    action_name = action_func.__name__
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    logger.info("--------------------------------------")
    logger.info(f"Action {choice}: Starting {action_name}...")
    logger.info("--------------------------------------\n")

    action_result = None
    session_started_by_exec = False

    try:
        if session_manager and not session_manager.session_active and action_name != "check_login_actn":
            logger.debug(f"Starting session for action: {action_name}\n")
            start_ok, _ = session_manager.start_sess(action_name=action_name)
            if not start_ok:
                logger.critical(f"Failed to start session for action {action_name}. Aborting action.")
                raise Exception("Session start failed")
            session_started_by_exec = True
        elif session_manager and action_name != "check_login_actn":
             logger.debug(f"Session already active for action: {action_name}.")
        # Execute the action function
        action_result = action_func(session_manager, *args)

    except Exception as e:
        logger.error(f"Exception during action {action_name}: {e}", exc_info=True)
        action_result = False

    finally:
        # --- Session Closing Logic (remains the same) ---
        exec_should_close = False
        if session_manager and session_manager.session_active:
            if session_started_by_exec and close_sess and action_name != "check_login_actn":
                exec_should_close = True
        if exec_should_close:
            session_manager.close_sess()
        elif session_manager and session_manager.session_active:
             if action_name == "check_login_actn":
                  logger.warning(f"Session remains active after {action_name}, which should close it internally.")
             elif not close_sess:
                  logger.debug(f"Keeping session active after {action_name} as requested (close_sess=False).")
             elif not session_started_by_exec:
                  logger.debug(f"Keeping session active after {action_name} (was already active).")
        elif session_manager and not session_manager.session_active:
             if action_name == "check_login_actn":
                  logger.debug(f"Session correctly closed by {action_name} internally.")

        # --- Performance Logging (remains the same) ---
        duration = time.time() - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_duration = f"{int(hours)} hr {int(minutes)} min {seconds:.2f} sec"
        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_used = mem_after - mem_before

        print(" ") # Spacer

        if action_result is False:
            logger.error(f"Action {choice} ({action_name}) reported a failure (returned False).\n")

        logger.info("--------------------------------------")
        logger.info(f"Action {choice} ({action_name}) finished.")
        logger.info(f"Duration: {formatted_duration}")
        logger.info(f"Memory used by action: {mem_used:.1f} MB")
        logger.info("--------------------------------------\n")
# End of exec_actn

# --- Action function definitions ---

# Action 0
def test_url_speed_action(session_manager, *args): # Add *args for consistency
    """Action to test the loading speed of two URLs and compare them."""
    # Ensure session is active before proceeding
    if not session_manager or not session_manager.driver:
         logger.error("Cannot test URL speed: Session/Driver not available.")
         return False

    # --- Inner function definition ---
    def test_url_speed(driver, url, target_selector):
        start_time = time.time()
        success = False
        try:
            # Navigate first, waiting only for body initially
            if not nav_to_page(driver, url, selector="body", session_manager=session_manager):
                logger.error(f"Initial navigation to {url} failed.")
                return time.time() - start_time, False

            # --- MODIFICATION: Increase timeout ---
            wait_timeout = 60 # Increased from 30
            # --- END MODIFICATION ---

            # Use a standard wait from config with the increased timeout
            wait = selenium_config.default_wait(driver, timeout=wait_timeout)

            # Log the selector and timeout being used
            logger.debug(f"Waiting up to {wait_timeout}s for selector '{target_selector}' at {url}")

            # Wait for the specific target element's visibility
            wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, target_selector)))
            success = True
            logger.debug(f"Target element '{target_selector}' found at {url}.")
            return time.time() - start_time, success
        except TimeoutException:
             logger.warning(f"Timeout waiting for target element '{target_selector}' at {url}.")
             return time.time() - start_time, False
        except Exception as e:
            logger.error(f"Error loading/verifying {url}: {e}", exc_info=True)
            return time.time() - start_time, False
    # --- End inner function ---

    # URLs and selector to test (selector remains the same for now)
    url1 = "https://www.ancestry.co.uk/messaging/?p=0725C8FC-0006-0000-0000-000000000000&testguid1=FB609BA5-5A0D-46EE-BF18-C300D8DE5AB7&testguid2=795155EB-3345-4079-8BBD-65ED4C79CF4D"
    url2 = "https://www.ancestry.co.uk/messaging/?p=0725C8FC-0006-0000-0000-000000000000"
    target_text_selector = "#message-box"

    try:
        logger.info(f"Testing URL 1: {url1}")
        time1, success1 = test_url_speed(session_manager.driver, url1, target_text_selector)
        logger.log(logging.INFO if success1 else logging.WARNING, f"URL 1 {'loaded successfully' if success1 else 'failed/timed out'} in {time1:.2f} seconds.")

        logger.info(f"\nTesting URL 2: {url2}")
        time2, success2 = test_url_speed(session_manager.driver, url2, target_text_selector)
        logger.log(logging.INFO if success2 else logging.WARNING, f"URL 2 {'loaded successfully' if success2 else 'failed/timed out'} in {time2:.2f} seconds.")

        logger.info("\n--- Comparison ---")
        logger.info(f"URL 1 Load Time: {time1:.2f}s (Success: {success1})")
        logger.info(f"URL 2 Load Time: {time2:.2f}s (Success: {success2})")

        if success1 and success2:
            diff = abs(time1 - time2)
            if diff < 0.5: # Consider them roughly equal if difference is small
                 logger.info("Both URLs loaded successfully in approximately the same time.")
            elif time1 < time2:
                logger.info(f"URL 1 (with extra params) was faster by {diff:.2f} seconds.")
            else:
                logger.info(f"URL 2 (without extra params) was faster by {diff:.2f} seconds.")
        else:
            logger.warning("One or both URLs failed to load completely; speed comparison may be unreliable.")

        return True # Indicate action completed, even if URLs failed

    except Exception as e:
        logger.error(f"Error during URL speed test action: {e}", exc_info=True)
        return False # Indicate action failed
# End of test_url_speed_action

# Action 1
@retry() # Keep retry if desired for the whole sequence
def run_actions_6_7_8_action(session_manager, *args): # Add *args
    """Action to run actions 6, 7, and 8 sequentially."""
    # Ensure session is active
    if not session_manager or not session_manager.driver or not session_manager.session_active:
         logger.error("Cannot run sequential actions: Session not active.")
         return False

    all_successful = True # Track overall success

    try:
        logger.info("--- Starting Action 6: Gather Matches ---")
        # Navigate to the correct starting page for gathering
        if not session_manager.nav_to_list(): # Use the specific method
            logger.error("Action 6 prerequisite failed: Cannot navigate to DNA matches list.")
            return False # Fail the sequence if navigation fails

        # Execute the core gathering logic (imported as coord_action_func)
        gather_result = coord_action_func(session_manager, config_instance) # Pass necessary args
        if gather_result is False: # Check return value
            logger.error("Action 6: Gather Matches reported failure.")
            all_successful = False
        else:
             logger.info("Action 6: Gather Matches completed.")

        # Proceed only if previous action was okay (optional)
        if not all_successful and False: # Set second condition to True to stop on failure
             logger.warning("Skipping subsequent actions due to failure in Action 6.")
             return False


        logger.info("--- Starting Action 7: Search Inbox ---")
        inbox_url = urljoin(config_instance.BASE_URL, "connect/messagecenter/folder/inbox")
        if not nav_to_page(session_manager.driver, inbox_url, INBOX_CONTAINER_SELECTOR, session_manager):
            logger.error("Action 7 prerequisite failed: Cannot navigate to the inbox page.")
            return False

        inbox_processor = InboxProcessor(session_manager=session_manager) # Pass session manager
        # search_inbox should ideally return True/False
        search_result = inbox_processor.search_inbox() # Call method on instance
        if search_result is False:
             logger.error("Action 7: Search Inbox reported failure.")
             all_successful = False
        else:
            logger.info("Action 7: Search Inbox completed.")

        # Proceed only if previous action was okay (optional)
        if not all_successful and False: # Set second condition to True to stop on failure
             logger.warning("Skipping subsequent actions due to failure in Action 7.")
             return False


        logger.info("--- Starting Action 8: Send Messages ---")
        # Navigate to a suitable page before starting messaging (e.g., base URL)
        if not nav_to_page(session_manager.driver, config_instance.BASE_URL, WAIT_FOR_PAGE_SELECTOR, session_manager):
            logger.error("Action 8 prerequisite failed: Cannot navigate to base URL.")
            return False

        # Execute messaging logic
        send_result = send_messages_to_matches(session_manager) # Pass session manager
        if send_result is False:
            logger.error("Action 8: Send Messages reported failure.")
            all_successful = False
        else:
            logger.info("Action 8: Send Messages completed.")


        logger.info("Sequential Actions 6, 7, and 8 sequence finished.")
        return all_successful # Return overall success status

    except Exception as e:
        logger.error(f"Error during sequential actions 6-7-8: {e}", exc_info=True)
        return False # Indicate failure on exception
# End Action 1

# Action 2
def reset_db_actn(*args): 
    """
    Action to reset the database (delete and recreate) and seed message_types.
    Handles its own temporary SessionManager for the reset process.
    Returns True on success, False on failure.
    """
    db_path = config_instance.DATABASE_FILE # Path object
    # Use a dedicated, temporary SessionManager for the reset operation
    reset_manager = SessionManager()
    session = None # Initialize session
    success = False

    try:
        logger.debug("Resetting database...")

        # 1. Delete database using the temporary manager to close connections
        logger.debug(f"Attempting to delete database file: {db_path}")
        delete_database(reset_manager, db_path, max_attempts=5) # Pass the temp manager
        logger.debug(f"Database file deletion successful (or file did not exist).")

        # --- Re-initialize Engine/Pool within the temporary manager ---
        logger.debug("Re-initializing database engine and connection pool for reset manager...")
        # cls_db_conn was called by delete_database, re-create engine/pool
        reset_manager.engine = create_engine(f"sqlite:///{str(db_path)}")
        @event.listens_for(reset_manager.engine, "connect")
        def connect(dbapi_connection, connection_record): # Listener attached to reset_manager's engine
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        reset_manager.Session = sessionmaker(bind=reset_manager.engine)
        reset_manager._db_conn_pool = ConnectionPool(str(db_path), pool_size=config_instance.DB_POOL_SIZE)

        # 2. Get a session from the temporary manager's pool
        session = reset_manager.get_db_conn()
        if not session:
             raise Exception("Failed to get database session after re-initialization!")

        # 3. Create tables using the temporary manager's engine
        logger.debug("Creating new tables...")
        Base.metadata.create_all(reset_manager.engine)
        logger.debug("Tables created successfully.")

        # 4. Seed message_types using the temporary manager's session
        logger.debug("Seeding message_types table...")
        # --- Use explicit commit/rollback for seeding diagnosis ---
        try:
            messages_file = Path("messages.json")
            if messages_file.exists():
                with messages_file.open("r", encoding="utf-8") as f:
                    messages_data = json.load(f)

                if isinstance(messages_data, dict):
                    message_types_to_add = []
                    for type_name in messages_data.keys():
                         if not session.query(MessageType).filter_by(type_name=type_name).first():
                             message_types_to_add.append(MessageType(type_name=type_name)) 
                    if message_types_to_add:
                        session.add_all(message_types_to_add)
                        logger.debug(f"Adding {len(message_types_to_add)} new message types...")
                        session.flush() # Flush before commit
                        session.commit() # Explicit commit
                        logger.debug("Seeding commit successful.")
                        # Verify seeding within the same session
                        count_after = session.query(func.count(MessageType.id)).scalar()
                        logger.debug(f"Verification: Found {count_after} message types after seeding commit.")
                    else:
                        logger.info("All message types already exist.")
                else:
                     logger.error("Format error: messages.json is not a dictionary.")
            else:
                 logger.warning("'messages.json' not found, skipping seeding.")
            success = True # Mark success if seeding finishes
        except (FileNotFoundError, json.JSONDecodeError, SQLAlchemyError) as seed_e:
            logger.error(f"Error during seeding: {seed_e}", exc_info=True)
            if session and session.is_active: session.rollback()
            success = False # Mark failure
        except Exception as seed_e_unexp:
            logger.critical(f"Unexpected error during seeding: {seed_e_unexp}", exc_info=True)
            if session and session.is_active: session.rollback()
            success = False # Mark failure
        # --- End explicit commit/rollback block ---

        if success: logger.info("Database reset successfully.")
        else: logger.error("Database reset successful, but seeding failed.")


    except Exception as e:
        logger.error(f"Error during database reset process: {e}", exc_info=True)
        success = False

    finally:
        logger.debug("Performing final cleanup for reset_db_actn...")
        # Always clean up the temporary reset_manager
        if 'reset_manager' in locals() and reset_manager:
            if session:
                reset_manager.return_session(session)
            reset_manager.cls_db_conn()
        logger.debug("Database reset action finished.")

    return success
# end of Action 2

# Action 3
def backup_db_actn(session_manager, *args): # Add *args
    """Action to backup the database."""
    # session_manager might not be needed if backup_database is standalone
    try:
        logger.debug("Starting database backup...")
        backup_database() # Call the standalone function from database.py
        logger.debug("Database backup completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error during database backup action: {e}", exc_info=True)
        return False
# end of Action 3

# Action 4
def restore_db_actn(session_manager, *args): # Add *args
    """Action to restore the database from the backup."""
    backup_dir = config_instance.DATA_DIR # Path object
    backup_path = backup_dir / "ancestry_backup.db" # Construct path
    db_path = config_instance.DATABASE_FILE # Path object
    success = False

    # Create a temporary manager if needed, mainly to close connections
    temp_manager = False
    if session_manager is None:
         logger.debug("Creating temporary SessionManager for database restore.")
         session_manager = SessionManager()
         temp_manager = True

    try:
        logger.debug(f"Attempting to restore database from: {backup_path}")

        # Check if backup file exists
        if not backup_path.exists():
             logger.error(f"Backup file not found: {backup_path}. Cannot restore.")
             return False # Indicate failure

        # Ensure connections are closed before overwriting file
        logger.debug("Closing existing database connections before restore...")
        session_manager.cls_db_conn()
        gc.collect() # Garbage collect
        time.sleep(1) # Small delay

        # Perform the copy
        shutil.copy2(backup_path, db_path) # Works with Path objects
        logger.info(f"Db restored from '{backup_path}' to '{db_path}'.")
        success = True

    except FileNotFoundError: # Should be caught by exists() check, but keep as safety
        logger.error(f"Backup file not found during copy: {backup_path}")
    except (OSError, IOError, shutil.Error) as e: # Catch potential copy errors
        logger.error(f"Error restoring database file: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"Unexpected critical error during database restore: {e}", exc_info=True)

    finally:
        # Clean up temporary manager if created
        if temp_manager and session_manager:
             session_manager.cls_db_conn() # Ensure its connections are closed too
             logger.debug("Temporary SessionManager cleaned up after restore.")
        logger.debug("Database restore action finished.")

    return success
# end of Action 4

# Action 5
def check_login_actn(session_manager, *args):
    """
    Action to verify that a session can be started and the user can be logged in.
    Relies entirely on start_sess for the verification logic.
    Closes the session immediately after the check.
    """
    if not session_manager:
         logger.error("SessionManager instance required for check_login_actn.")
         return False # Indicate failure: No manager provided

    logger.debug("Verifying session start and login capability...")
    login_verification_successful = False

    try:
        # Call start_sess. Its success (True) implies login is verified.
        # Pass a specific action name for clarity in logs if desired.
        start_ok, _ = session_manager.start_sess(action_name="Login Verification (Action 5)")

        if start_ok:
            logger.info("Login verification successful (session started, user logged in).")
            login_verification_successful = True
        else:
            # start_sess failed, meaning session couldn't start or login failed.
            logger.error("Login verification failed (unable to start session or log in).")
            login_verification_successful = False

    except Exception as e:
        logger.error(f"Unexpected error during login verification action: {e}", exc_info=True)
        login_verification_successful = False # Treat exception as failure

    finally:
        # Always ensure the session is closed after this specific action runs.
        # Check if a driver instance was potentially created, even if start_sess failed partway.
        if session_manager and session_manager.driver:
            logger.debug("Closing session after login verification action.")
            session_manager.close_sess() # close_sess sets driver to None and session_active to False
        elif session_manager:
             # Ensure session_active is False if no driver was created or start_sess failed early
             session_manager.session_active = False
             logger.debug("Session was not active or driver not created; ensuring state is inactive.")
        logger.debug("Login verification action finished.")

    return login_verification_successful # Return True if verification succeeded, False otherwise
# End Action 5

# Action 6
def coord_action(session_manager, config_instance, start_page=1): # Renamed internal call
    """Action wrapper for gathering matches (coord)."""
    if not session_manager or not session_manager.driver or not session_manager.session_active:
         logger.error("Cannot gather matches: Session not active.")
         return False

    logger.debug(f"Gathering DNA Matches from page {start_page}...\n")
    try:
        # Call the imported gathering function
        result = coord_action_func(session_manager, config_instance, start_page)
        if result is False: # Check return value
             logger.error("Match gathering (coord) reported failure.")
             return False
        else:
             logger.debug("Successfully gathered matches.")
             return True
    except Exception as e:
        logger.error(f"Error during coord_action wrapper: {e}", exc_info=True)
        return False
# End of coord_action6


# Action 7
def srch_inbox_actn(session_manager, *args): # Add *args
    """Action to search the inbox for messages using the API."""
    # Ensure session is active for API calls, but no UI navigation needed here.
    if not session_manager or not session_manager.is_sess_valid(): # Check session validity
         logger.error("Cannot search inbox: Session not active or invalid.")
         return False

    logger.debug("11. Starting inbox search...")
    try:
        # --- REMOVED NAVIGATION STEP ---
        # inbox_url = urljoin(config_instance.BASE_URL, "connect/messagecenter/folder/inbox")
        # if not nav_to_page(session_manager.driver, inbox_url, INBOX_CONTAINER_SELECTOR, session_manager):
        #     logger.error("Failed to navigate to inbox page. Aborting search action.")
        #     return False
        # --- END REMOVED NAVIGATION STEP ---

        inbox_processor = InboxProcessor(session_manager=session_manager)
        # The search_inbox method now handles API calls and DB interaction directly.
        # It requires session_manager for API calls (_api_req) and DB access.
        result = inbox_processor.search_inbox() # Call the method

        if result is False:
             logger.error("Inbox search reported failure.")
             return False
        else:
             logger.debug("Inbox search completed.")
             return True

    except Exception as e:
        logger.error(f"Error during inbox search action: {e}", exc_info=True)
        return False
# End of srch_inbox_actn

# Action 8
def send_messages_action(session_manager, *args): # Add *args
    """Action to send messages to DNA matches."""
    if not session_manager or not session_manager.driver or not session_manager.session_active:
         logger.error("Cannot send messages: Session not active.")
         return False

    logger.info("Starting message sending process...")
    try:
        # Navigate to a neutral page first (e.g., base URL)
        if not nav_to_page(session_manager.driver, config_instance.BASE_URL, WAIT_FOR_PAGE_SELECTOR, session_manager):
            logger.error("Failed to navigate to base URL before sending messages. Aborting.")
            return False

        # Call the messaging function
        result = send_messages_to_matches(session_manager)
        if result is False:
             logger.error("Message sending process reported failure.")
             return False
        else:
             logger.info("Message sending process completed.")
             return True

    except Exception as e:
        logger.error(f"Error during message sending action: {e}", exc_info=True)
        return False
# End of send_messages_action

# Action 9
def all_but_first_actn(session_manager, *args): # Add *args
    """Action to delete all 'people' rows except the first, using cascading deletes."""
    # Create a temporary manager if None provided (for DB connection)
    temp_manager = False
    if session_manager is None:
        logger.debug("Creating temporary SessionManager for delete action.")
        session_manager = SessionManager()
        temp_manager = True

    session = None
    success = False
    try:
        logger.info("Deleting all but the first row)...")
        session = session_manager.get_db_conn()
        if session is None:
            raise Exception("Failed to get a database session.")

        with db_transn(session) as sess: # Use context manager and alias
            # Find the first person's ID
            first_person = sess.query(Person).order_by(Person.id.asc()).first()
            if not first_person:
                logger.info("Table 'people' is empty. Nothing to delete.")
                return True # Considered success

            first_person_id = first_person.id
            logger.debug(f"Keeping person with ID: {first_person_id} ({first_person.username})")

            # --- MODIFICATION: Use Loop Delete instead of Bulk Delete ---
            # Comment out or remove the bulk delete line:
            # deleted_count = sess.query(Person).filter(Person.id != first_person_id).delete(synchronize_session=False)
            # logger.info(f"Deleted {deleted_count} rows from 'people' table (cascades initiated).")

            # Use loop method (slower but ensures individual cascade triggers via ORM):
            people_to_delete = sess.query(Person).filter(Person.id != first_person_id).all()
            count = 0
            total_to_delete = len(people_to_delete) # Get total count for logging

            if not people_to_delete:
                 logger.info("Only one person found, nothing else to delete.")
            else:
                 logger.debug(f"Found {total_to_delete} people to delete...")
                 # Iterate through the list to delete each person object
                 for i, person in enumerate(people_to_delete):
                     logger.debug(f"Deleting person {i+1}/{total_to_delete}: {person.username} (ID: {person.id})")
                     sess.delete(person) # ORM delete triggers cascade handling
                     count += 1
                     # Optional: Flush periodically for large deletes to manage memory/transaction size
                     # if count % 100 == 0:
                     #     logger.debug(f"Flushing after deleting {count} records...")
                     #     sess.flush()
                 logger.info(f"Deleted {count} people.")
            # --- END MODIFICATION ---

        success = True

    except Exception as e:
        logger.error(f"Error during deletion action: {e}", exc_info=True)
        # Rollback is handled by db_transn context manager on exception

    finally:
        logger.debug("Performing cleanup for delete action...")
        if session_manager:
            if session:
                session_manager.return_session(session)
            # Only close connections if a temporary manager was used
            if temp_manager:
                 session_manager.cls_db_conn()
                 logger.debug("Temporary SessionManager connections closed.")
        logger.debug("Delete action finished.")

    return success
# end of Action 9

def main():
    global logger, session_manager # Declare globals to modify/use them

    session_manager = None # Initialize global session_manager

    try:
        # Initial screen clear and blank line
        os.system("cls" if os.name == "nt" else "clear")
        print("")
        try:
            # --- INITIAL LOGGING SETUP ---
            # Ensure config_instance is imported before this point
            from config import config_instance # Keep import for other uses (DB path)
            db_file_path = config_instance.DATABASE_FILE
            # Construct log file name relative to DB file
            log_filename = db_file_path.with_suffix(".log").name

            # >>> MODIFICATION START: Force initial level to INFO <<<
            # Directly call setup_logging requesting "INFO" or omit log_level
            # to use the function's internal default. Explicitly setting "INFO" here
            # makes the intention clear and overrides any config value for the initial setup.
            logger = setup_logging(
                log_file=log_filename,
                log_level="INFO" # Explicitly set initial level to INFO
            )
            # >>> MODIFICATION END <<<

            # Log the level obtained from config *after* initial setup for user info, if needed.
            config_log_level = getattr(config_instance, 'LOG_LEVEL', 'Not Set')
            # logger.debug(f"Configured LOG_LEVEL (ignored for initial setup): {config_log_level}")


        except Exception as log_setup_e:
             # Use print as logger setup failed
             print(f"CRITICAL: Error during logging setup: {log_setup_e}", file=sys.stderr)
             traceback.print_exc(file=sys.stderr)
             # Basic fallback logging if setup fails catastrophically
             logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
             logger = logging.getLogger("logger_fallback") # Use a different name for fallback
             logger.warning("Using fallback basic logging due to setup error.")


        # Instantiate SessionManager AFTER logging is ready
        # Assuming SessionManager uses the global logger implicitly or is passed it
        session_manager = SessionManager() # Assign to global variable

        # --- Main menu loop ---
        while True:
            choice = menu() # Display menu showing current console level
            print("") # Spacer

            # --- Action Dispatching ---
            # (Keep existing action dispatching logic for choices 0-9)
            # Example for one action:
            if choice == "0":
                exec_actn(test_url_speed_action, session_manager, choice, close_sess=True)
            # ... other choices 1-9 ...
            elif choice == "1":
                exec_actn(run_actions_6_7_8_action, session_manager, choice, close_sess=False) # Keep session open after sequence
            elif choice == "2":
                exec_actn(reset_db_actn, None, choice, close_sess=False)
                if session_manager and hasattr(session_manager, '_initialize_db_pool'):
                    logger.debug("Re-initializing main SessionManager DB pool after reset.")
                    pass  # Removed call to undefined method _initialize_db_pool
            elif choice == "3":
                 exec_actn(backup_db_actn, None, choice, close_sess=False)
            elif choice == "4":
                exec_actn(restore_db_actn, session_manager, choice, close_sess=False)
                if session_manager and hasattr(session_manager, '_initialize_db_pool'):
                     logger.debug("Re-initializing main SessionManager DB pool after restore.")
                     pass  # Removed call to undefined method _initialize_db_pool
            elif choice == "5":
                 exec_actn(check_login_actn, session_manager, choice, close_sess=False) # Handles its own close
            elif choice.startswith("6"):
                 parts = choice.split()
                 start_page = 1
                 if len(parts) > 1:
                    try:
                        start_page = int(parts[1])
                        if start_page <= 0: raise ValueError("Start page must be positive")
                    except ValueError as e:
                        print(f"Invalid start page specified: {e}. Using page 1.")
                        start_page = 1
                 exec_actn(coord_action, session_manager, "6", True, config_instance, start_page)
            elif choice == "7":
                exec_actn(srch_inbox_actn, session_manager, choice, close_sess=True)
            elif choice == "8":
                exec_actn(send_messages_action, session_manager, choice, close_sess=True)
            elif choice == "9":
                exec_actn(all_but_first_actn, None, choice, close_sess=False)
                if session_manager and hasattr(session_manager, '_initialize_db_pool'):
                     logger.debug("Re-initializing main SessionManager DB pool after delete.")
                     logger.warning("Reinitialization of the database pool is not implemented. Ensure the database pool is properly managed.")

            # --- Meta Options ---
            elif choice == "t":
                # --- LOG LEVEL TOGGLE LOGIC ---
                os.system("cls" if os.name == "nt" else "clear")
                if logger and logger.handlers:
                    console_handler = None
                    for handler in logger.handlers:
                        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                            console_handler = handler
                            break

                    if console_handler:
                        current_level = console_handler.level
                        # Toggle between INFO and DEBUG
                        new_level = logging.DEBUG if current_level > logging.DEBUG else logging.INFO
                        new_level_name = logging.getLevelName(new_level)

                        # Call setup_logging again to update the console handler's level
                        # This is the intended mechanism based on logging_config.py
                        # It returns the same logger instance, but modifies the handler level
                        setup_logging(log_level=new_level_name)


                    else:
                         logger.warning("Could not find console handler to toggle level.")
                else:
                     print("WARNING: Logger not ready, cannot toggle level.", file=sys.stderr)
                # --- END LOG LEVEL TOGGLE LOGIC ---

            elif choice == "c":
                os.system("cls" if os.name == "nt" else "clear")
            elif choice == "q":
                os.system("cls" if os.name == "nt" else "clear")
                if logger: logger.info("Exiting program by user choice.")
                print("Exiting the program.\n")
                break
            else:
                print("Invalid choice. Please try again.\n")

    # --- Exception Handling & Cleanup ---
    except KeyboardInterrupt:
        os.system("cls" if os.name == "nt" else "clear")
        print("\nCTRL+C detected. Exiting gracefully.")
        if logger: logger.warning("CTRL+C detected. Initiating graceful exit.")
    except Exception as e:
        # Ensure logger exists before trying to log the critical error
        if logger:
            logger.critical(f"Critical error in main execution block: {e}", exc_info=True)
        else: # Logger failed or not initialized
            print(f"CRITICAL ERROR (logging unavailable): {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    finally:
        # Use the global session_manager for cleanup
        if session_manager: # Check if it was successfully initialized
             active_before_final_close = session_manager.session_active
             if active_before_final_close :
                  if logger: logger.info("Performing final session cleanup (browser driver)...")
                  session_manager.close_sess() # Closes browser and sets inactive
                  if logger: logger.info("Browser session cleanup finished.")
             # Always ensure DB connections are closed
             if logger: logger.debug("Performing final DB connection cleanup...")
             session_manager.cls_db_conn()
             if logger: logger.debug("Final DB connection cleanup finished.")

        if logger: logger.info("--- Main program execution finished ---")
        else: print("--- Main program execution finished (logger unavailable) ---", file=sys.stderr)

        print("\nExecution finished.")
# end main

# --- Entry Point ---
if __name__ == "__main__":
    main()