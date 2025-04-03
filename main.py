# File: main.py

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

# --- Third-party imports  ---
from selenium.webdriver.remote.remote_connection import RemoteConnection
import urllib3.poolmanager 
import psutil
import urllib3


from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
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
    ConnectionPool,
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
from utils import (
    SessionManager,
    is_elem_there,
    log_in,
    login_status,
    nav_to_page,
    retry,
)

# +++ START urllib3 PoolManager Scheme Patch (Risky - Use with Caution) +++
try:
    import urllib3
    import urllib3.poolmanager
    print("INFO: Preparing to patch urllib3.poolmanager.PoolManager.pool_classes_by_scheme default maxsize (Risky).", file=sys.stderr)

    # Get the pool class used for HTTP
    http_pool_cls = urllib3.poolmanager.PoolManager.pool_classes_by_scheme.get('http')

    if http_pool_cls:
        # Modify the default arguments stored or used by this class if possible
        # This is highly dependent on urllib3 internal structure
        # Option A: Modify if it uses connection_pool_kw like PoolManager itself (less likely)
        # if hasattr(http_pool_cls, 'connection_pool_kw') and isinstance(http_pool_cls.connection_pool_kw, dict):
        #    http_pool_cls.connection_pool_kw['maxsize'] = 10

        # Option B: Modify default maxsize if it's a direct attribute (less likely)
        # if hasattr(http_pool_cls, 'default_maxsize'):
        #    http_pool_cls.default_maxsize = 10

        # Option C: If the class itself is stored, we might patch its __init__ (similar to previous failed attempt but more targeted)
        # This seems most plausible if the scheme lookup returns the class type
        if isinstance(http_pool_cls, type) and issubclass(http_pool_cls, urllib3.connectionpool.HTTPConnectionPool):
            original_init = http_pool_cls.__init__

            def patched_http_init(self, *args, **kwargs):
                pool_maxsize = kwargs.get('maxsize')
                if pool_maxsize is None or pool_maxsize == 1:
                     # print(f"INFO: Patching {http_pool_cls.__name__} maxsize from {pool_maxsize or 'default (1)'} to 10.", file=sys.stderr) # Less verbose
                     kwargs['maxsize'] = 10
                original_init(self, *args, **kwargs)

            http_pool_cls.__init__ = patched_http_init
            print(f"INFO: Applied __init__ patch to {http_pool_cls.__name__} found via pool_classes_by_scheme.", file=sys.stderr)
        else:
            print("WARNING: Could not determine how to patch the HTTP pool class found via scheme.", file=sys.stderr)

    else:
        print("WARNING: Could not find 'http' scheme in urllib3.poolmanager.PoolManager.pool_classes_by_scheme.", file=sys.stderr)

except Exception as patch_e:
    print(f"ERROR: Failed to apply urllib3 scheme patch: {patch_e}", file=sys.stderr)
# +++ END urllib3 PoolManager Scheme Patch +++

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

    print(f"(Log Level: {level_name})\n") # Show console and fixed file level
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
                break # Assume only one file handler

        if handler_to_reopen and log_path:
            # --- Ensure handler is valid before using ---
            if hasattr(handler_to_reopen, 'flush') and hasattr(handler_to_reopen, 'close') and hasattr(handler_to_reopen, 'stream') and handler_to_reopen.stream:
                handler_to_reopen.flush()
                handler_to_reopen.close() # Close the stream
                # Clear the file by opening in write mode
                with open(log_path, 'w', encoding='utf-8') as f:
                    pass # Just opening in 'w' truncates the file
                cleared = True
                # FileHandler should reopen automatically on next log message
            else:
                 print(f"WARNING: FileHandler for '{log_path}' seems invalid or closed already. Skipping clear.", file=sys.stderr)


    except PermissionError as pe:
        print(f"WARNING: Permission denied clearing log file '{log_path}'. Log may not be fresh. Error: {pe}", file=sys.stderr)
    except IOError as e:
        print(f"WARNING: IOError clearing log file '{log_path}'. Log may not be fresh. Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: Unexpected error clearing log file '{log_path}'. Log may not be fresh. Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Print traceback for unexpected errors

    return cleared, log_path
# End of clear_log_file

def exec_actn(action_func, session_manager, choice, close_sess=True, *args):
    """
    V3 REVISED: Executes an action function, managing session start/stop
    and logging performance. Passes config_instance explicitly if needed.
    """
    import inspect # Local import for signature check

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
    browserless_actions = ["reset_db_actn", "backup_db_actn", "restore_db_actn", "all_but_first_actn"]

    try:
        # Start session ONLY if needed by the action AND not already active
        if session_manager and \
           not session_manager.session_active and \
           action_name not in browserless_actions and \
           action_name != "check_login_actn":

            logger.debug(f"Starting browser session for action: {action_name}\n")
            start_ok, _ = session_manager.start_sess(action_name=action_name)
            if not start_ok:
                logger.critical(f"Failed to start browser session for action {action_name}. Aborting action.")
                raise Exception("Browser Session start failed")
            session_started_by_exec = True
            logger.debug(f"Browser session successfully started by exec_actn for {action_name}.")
        elif session_manager and action_name not in browserless_actions and action_name != "check_login_actn":
            logger.debug(f"Browser session already active for action: {action_name}.")
        elif action_name in browserless_actions:
            logger.debug(f"Action '{action_name}' does not require browser start via exec_actn.")

        # Execute the action function
        # --- Pass config_instance if needed by the action function signature ---
        func_sig = inspect.signature(action_func)
        action_args_to_pass = list(args) # Convert tuple to list for potential modification
        pass_config = False
        if 'config_instance' in func_sig.parameters:
             pass_config = True
             # Prepend config_instance if it's not already the first arg in *args
             # This assumes config_instance is expected AFTER session_manager
             # If the wrapped function expects config_instance first, adjust logic
             if not action_args_to_pass or action_args_to_pass[0] != config_instance:
                  action_args_to_pass.insert(0, config_instance)


        # Check if start_page needs to be passed (specific to coord_action)
        if action_name == 'coord_action' and 'start_page' in func_sig.parameters:
            # Assume start_page is the last argument if passed via *args
            if len(args) > 0 and isinstance(args[-1], int):
                start_page_val = args[-1]
                if pass_config:
                     # config is already inserted at [0], start_page is at [1]
                     action_result = action_func(session_manager, action_args_to_pass[0], start_page=start_page_val)
                else:
                     # No config, start_page is the first arg in *args
                     action_result = action_func(session_manager, start_page=start_page_val)
            else: # Default start page if not passed
                 if pass_config:
                      action_result = action_func(session_manager, action_args_to_pass[0], start_page=1)
                 else:
                      action_result = action_func(session_manager, start_page=1)
        else:
            # General case for other actions
            if pass_config:
                action_result = action_func(session_manager, *action_args_to_pass)
            else:
                action_result = action_func(session_manager, *args) # Pass original args


    except Exception as e:
        logger.error(f"Exception during action {action_name}: {e}", exc_info=True)
        action_result = False

    finally:
        # Session Closing Logic
        exec_should_close_browser = False
        if session_manager and session_manager.session_active:
            if session_started_by_exec and close_sess:
                exec_should_close_browser = True

        if exec_should_close_browser:
            logger.debug(f"Closing browser session started by exec_actn for {action_name}.")
            session_manager.close_sess()
            logger.debug("Browser session closed.")
        elif session_manager and session_manager.session_active:
            if action_name == "check_login_actn":
                 logger.warning(f"Session remains active after {action_name}, which should close it.")
            elif not close_sess:
                 logger.debug(f"Keeping browser session active after {action_name} as requested (close_sess=False).")
            elif not session_started_by_exec:
                 logger.debug(f"Keeping browser session active after {action_name} (was already active).")
            elif action_name in browserless_actions:
                 logger.debug(f"Browser session was not started by exec_actn for {action_name}, keeping active if running.")

        # Performance Logging
        duration = time.time() - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_duration = f"{int(hours)} hr {int(minutes)} min {seconds:.2f} sec"
        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_used = mem_after - mem_before

        # print(" ") # Spacer

        if action_result is False:
            logger.error(f"Action {choice} ({action_name}) reported a failure (returned False or exception occurred).\n")

        logger.info("--------------------------------------")
        logger.info(f"Action {choice} ({action_name}) finished.")
        logger.info(f"Duration: {formatted_duration}")
        logger.info(f"Memory used by action: {mem_used:.1f} MB")
        logger.info("--------------------------------------\n")
# End of exec_actn

# --- Action function definitions ---

# Action 0
def test_url_speed_action(session_manager, *args):
    """Action to test the loading speed of two URLs and compare them."""
    if not session_manager or not session_manager.driver:
         logger.error("Cannot test URL speed: Session/Driver not available.")
         return False

    def test_url_speed(driver, url, target_selector):
        start_time = time.time()
        success = False
        try:
            if not nav_to_page(driver, url, selector="body", session_manager=session_manager):
                logger.error(f"Initial navigation to {url} failed.")
                return time.time() - start_time, False
            wait_timeout = 60
            wait = selenium_config.default_wait(driver, timeout=wait_timeout)
            logger.debug(f"Waiting up to {wait_timeout}s for selector '{target_selector}' at {url}")
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

    url1 = "https://www.ancestry.co.uk/messaging/?p=0725C8FC-0006-0000-0000-000000000000&testguid1=FB609BA5-5A0D-46EE-BF18-C300D8DE5AB7&testguid2=795155EB-3345-4079-8BBD-65ED4C79CF4D"
    url2 = "https://www.ancestry.co.uk/messaging/?p=0725C8FC-0006-0000-0000-000000000000"
    target_text_selector = "#message-box"

    try:
        logger.info(f"Testing URL 1: {url1}")
        time1, success1 = test_url_speed(session_manager.driver, url1, target_text_selector)
        logger.log(logging.INFO if success1 else logging.WARNING, f"URL 1 {'loaded' if success1 else 'failed'} in {time1:.2f}s.")

        logger.info(f"\nTesting URL 2: {url2}")
        time2, success2 = test_url_speed(session_manager.driver, url2, target_text_selector)
        logger.log(logging.INFO if success2 else logging.WARNING, f"URL 2 {'loaded' if success2 else 'failed'} in {time2:.2f}s.")

        logger.info("\n--- Comparison ---")
        logger.info(f"URL 1 Load Time: {time1:.2f}s (Success: {success1})")
        logger.info(f"URL 2 Load Time: {time2:.2f}s (Success: {success2})")

        if success1 and success2:
            diff = abs(time1 - time2)
            if diff < 0.5: logger.info("Both URLs loaded in approx same time.")
            elif time1 < time2: logger.info(f"URL 1 faster by {diff:.2f}s.")
            else: logger.info(f"URL 2 faster by {diff:.2f}s.")
        else: logger.warning("URL failure; comparison unreliable.")
        return True
    except Exception as e:
        logger.error(f"Error during URL speed test: {e}", exc_info=True)
        return False
# End of test_url_speed_action

# Action 1
@retry()
def run_actions_6_7_8_action(session_manager, *args):
    """Action to run actions 6, 7, and 8 sequentially."""
    if not session_manager or not session_manager.driver or not session_manager.session_active:
         logger.error("Cannot run sequential actions: Session not active.")
         return False

    all_successful = True
    try:
        logger.info("--- Starting Action 6: Gather Matches ---")
        # Ensure navigation is attempted, but proceed cautiously if it fails initially
        if not nav_to_list(session_manager): # nav_to_list is from action6_gather
            logger.warning("Initial navigation to match list failed in sequential run. Action 6 will attempt navigation again.")
            # Don't return False here, let coord_action_func handle its own navigation needs
        # Call coord_action_func directly, passing config_instance
        gather_result = coord_action_func(session_manager, config_instance) # Use renamed imported function
        if gather_result is False: logger.error("Action 6 reported failure."); all_successful = False
        else: logger.info("Action 6 completed.")

        # Optional early exit if critical step fails
        # if not all_successful: logger.warning("Skipping Actions 7 & 8 due to failure in Action 6."); return False

        logger.info("--- Starting Action 7: Search Inbox ---")
        inbox_url = urljoin(config_instance.BASE_URL, "connect/messagecenter/folder/inbox")
        if not nav_to_page(session_manager.driver, inbox_url, INBOX_CONTAINER_SELECTOR, session_manager):
            logger.error("Action 7 prerequisite failed: Cannot navigate to inbox. Skipping Action 7.")
            all_successful = False # Mark failure but continue to Action 8 if needed
        else:
            inbox_processor = InboxProcessor(session_manager=session_manager)
            search_result = inbox_processor.search_inbox()
            if search_result is False: logger.error("Action 7 reported failure."); all_successful = False
            else: logger.info("Action 7 completed.")

        # Optional early exit
        # if not all_successful: logger.warning("Skipping Action 8 due to failure in Action 7."); return False

        logger.info("--- Starting Action 8: Send Messages ---")
        # Navigate to a neutral page before starting messaging
        if not nav_to_page(session_manager.driver, config_instance.BASE_URL, WAIT_FOR_PAGE_SELECTOR, session_manager):
            logger.error("Action 8 prerequisite failed: Cannot navigate to base URL. Skipping Action 8.")
            all_successful = False
        else:
            send_result = send_messages_to_matches(session_manager)
            if send_result is False: logger.error("Action 8 reported failure."); all_successful = False
            else: logger.info("Action 8 completed.")

        logger.info("Sequential Actions 6-7-8 finished.")
        return all_successful
    except Exception as e:
        logger.error(f"Error during sequential actions 6-7-8: {e}", exc_info=True)
        return False
# End Action 1


# Action 2
def reset_db_actn(session_manager, *args):
    """
    Action to reset the database and seed message_types.
    Ensures main session manager connections are closed BEFORE attempting deletion.
    """
    db_path = config_instance.DATABASE_FILE
    reset_successful = False
    original_session_closed_here = False

    # --- STEP 1: Ensure main session manager connections are closed ---
    if session_manager and session_manager._db_conn_pool: # Check if pool exists
        logger.warning("Main session manager has an active DB pool. Closing connections before reset attempt...")
        try:
            session_manager.cls_db_conn() # Close pool and dispose engine
            original_session_closed_here = True
            logger.debug("Main session DB connections closed successfully.")
            # Add a brief pause and GC just in case
            gc.collect()
            time.sleep(0.5)
        except Exception as close_err:
            logger.error(f"Failed to close main session DB connections before reset: {close_err}", exc_info=True)
            # Proceed with caution, deletion might still fail
    elif session_manager:
        logger.debug("Main session manager DB pool already seems closed or uninitialized.")
    else:
        logger.warning("No main session manager provided to reset_db_actn.")
        # Create a temporary manager just for the deletion attempt, but this is less ideal
        session_manager = SessionManager() # Create temporary one

    # --- STEP 2: Attempt Database Deletion ---
    try:
        logger.debug("Attempting database file deletion...")
        # Pass the potentially temporary or already-closed session_manager
        # delete_database will try to close its connections again (harmless if already done)
        delete_database(session_manager, db_path, max_attempts=5)
        logger.debug(f"Database file '{db_path}' deleted successfully (or was already gone).")

        # --- STEP 3: Re-initialize DB and Seed ---
        # Use a new, clean SessionManager instance specifically for re-creation
        recreation_manager = SessionManager()
        recreation_session = None
        try:
            logger.debug("Re-initializing DB engine/pool for recreation manager...")
            # Use the string path for create_engine URI
            recreation_manager.engine = create_engine(f"sqlite:///{str(db_path)}")
            @event.listens_for(recreation_manager.engine, "connect")
            def enable_foreign_keys(dbapi_conn, conn_rec):
                cursor = dbapi_conn.cursor()
                try: cursor.execute("PRAGMA foreign_keys=ON")
                finally: cursor.close()

            recreation_manager.Session = sessionmaker(bind=recreation_manager.engine)
            # Use string path for ConnectionPool
            recreation_manager._db_conn_pool = ConnectionPool(str(db_path), pool_size=config_instance.DB_POOL_SIZE)

            logger.debug("Creating tables...")
            Base.metadata.create_all(recreation_manager.engine)
            logger.debug("Tables created.")

            recreation_session = recreation_manager.get_db_conn()
            if not recreation_session:
                raise SQLAlchemyError("Failed to get session after DB re-initialization!")

            logger.debug("Seeding message_types...")
            try:
                # Define messages_file path relative to the main script or use absolute path if needed
                # Assuming messages.json is in the same directory as main.py
                script_dir = Path(__file__).resolve().parent # Use resolve() for robustness
                messages_file = script_dir / "messages.json"

                if messages_file.exists():
                    with messages_file.open("r", encoding="utf-8") as f:
                        messages_data = json.load(f)

                    if isinstance(messages_data, dict):
                        # Use the recreation_session for seeding
                        with db_transn(recreation_session) as sess:
                            types_to_add = []
                            for name in messages_data:
                                exists = sess.query(MessageType).filter_by(type_name=name).first()
                                if not exists:
                                    types_to_add.append(MessageType(type_name=name))

                            if types_to_add:
                                logger.debug(f"Adding {len(types_to_add)} new message types...")
                                sess.add_all(types_to_add)
                                logger.debug("Seeding commit OK.")
                            else:
                                logger.info("Message types already exist in the new database.")
                        # Verify count after commit
                        count = recreation_session.query(func.count(MessageType.id)).scalar()
                        logger.debug(f"Verification: {count} message types found in DB.")
                    else:
                        logger.error("'messages.json' has incorrect format (expected dictionary).")
                else:
                    logger.warning(f"'messages.json' not found at '{messages_file}', skipping message type seeding.")

                reset_successful = True # Mark success only after seeding attempt
                logger.info("Reset and re-initialization completed OK.\n")

            except (FileNotFoundError, json.JSONDecodeError, SQLAlchemyError) as seed_e:
                 logger.error(f"Error seeding message_types after reset: {seed_e}\n", exc_info=True)
                 reset_successful = False # Failed during seeding
            except Exception as seed_e_unexp:
                 logger.critical(f"Unexpected seeding error after reset: {seed_e_unexp}\n", exc_info=True)
                 reset_successful = False # Failed during seeding

        finally:
            # Clean up the recreation_manager
            logger.debug("Cleaning up recreation manager resources...")
            if 'recreation_manager' in locals() and recreation_manager:
                if recreation_session:
                    recreation_manager.return_session(recreation_session)
                recreation_manager.cls_db_conn() # Close pool and dispose engine
            logger.debug("Recreation manager cleanup finished.")

    except PermissionError as pe:
        logger.error(f"Database reset FAILED: Could not delete file '{db_path}' due to permissions/lock.", exc_info=False)
        logger.debug(f"PermissionError details: {pe}")
        reset_successful = False
    except Exception as e:
        logger.error(f"Error during DB reset process: {e}", exc_info=True)
        reset_successful = False

    finally:
        # --- STEP 4: Potentially Re-initialize Main Session ---
        # If the original session was closed by *this* function,
        # we should probably leave it closed, as the caller (exec_actn)
        # expects browserless actions not to mess with the main session state.
        # The next action requiring DB will re-initialize it if needed.
        if original_session_closed_here:
             logger.debug("Original main session was closed by reset_db_actn. Leaving it closed.")
             # It's important that subsequent actions correctly handle getting a new DB connection
             # if session_manager._db_conn_pool is None.

        logger.debug("Reset DB action finished.")

    return reset_successful
# end of Action 2

# Action 3
def backup_db_actn(session_manager, *args):
    """Action to backup the database."""
    try: logger.debug("Starting DB backup..."); backup_database(); logger.debug("DB backup OK."); return True
    except Exception as e: logger.error(f"Error during DB backup: {e}", exc_info=True); return False
# end of Action 3

# Action 4
def restore_db_actn(session_manager, *args):
    """Action to restore the database from the backup."""
    backup_dir = config_instance.DATA_DIR; backup_path = backup_dir / "ancestry_backup.db"
    db_path = config_instance.DATABASE_FILE; success = False
    if not session_manager: logger.error("SessionManager required for restore."); return False
    try:
        logger.debug(f"Restoring DB from: {backup_path}")
        if not backup_path.exists(): logger.error(f"Backup not found: {backup_path}."); return False
        logger.debug("Closing connections before restore..."); session_manager.cls_db_conn(); gc.collect(); time.sleep(1)
        shutil.copy2(backup_path, db_path); logger.info(f"Db restored from backup.\n"); success = True
    except FileNotFoundError: logger.error(f"Backup not found during copy: {backup_path}\n")
    except (OSError, IOError, shutil.Error) as e: logger.error(f"Error restoring DB: {e}", exc_info=True)
    except Exception as e: logger.critical(f"Unexpected restore error: {e}\n", exc_info=True)
    finally: logger.debug("DB restore action finished.")
    return success
# end of Action 4

# Action 5
def check_login_actn(session_manager, *args):
    """Action to verify session start and login capability."""
    if not session_manager: logger.error("SessionManager required."); return False
    logger.debug("Verifying login capability..."); login_ok = False
    try:
        start_ok, _ = session_manager.start_sess(action_name="Login Verification (Action 5)")
        if start_ok: logger.info("Login verification successful.\n"); login_ok = True
        else: logger.error("Login verification failed.\n"); login_ok = False
    except Exception as e: logger.error(f"Error during login verification: {e}\n", exc_info=True); login_ok = False
    finally:
        if session_manager and session_manager.driver: logger.debug("Closing session after check."); session_manager.close_sess()
        elif session_manager: session_manager.session_active = False
        logger.debug("Login check action finished.")
    return login_ok
# End Action 5

# Action 6
# Define coord_action wrapper that accepts config_instance and start_page from exec_actn args
def coord_action(session_manager, config_instance, start_page=1):
    """Action wrapper for gathering matches (coord), correctly receiving config_instance and start_page."""
    if not session_manager or not session_manager.driver or not session_manager.session_active: logger.error("Cannot gather: Session not active."); return False
    logger.debug(f"Gathering DNA Matches from page {start_page}...\n")
    try:
        # Call the imported and renamed coord_action_func directly
        # It already expects session_manager, config_instance, and start_page
        result = coord_action_func(session_manager, config_instance, start_page)
        if result is False: logger.error("Match gathering reported failure."); return False
        else: logger.debug("Gathering matches OK."); return True
    except Exception as e: logger.error(f"Error during coord_action: {e}", exc_info=True); return False
# End of coord_action6


# Action 7
def srch_inbox_actn(session_manager, *args):
    """Action to search the inbox for messages using the API."""
    if not session_manager or not session_manager.is_sess_valid(): logger.error("Cannot search inbox: Session invalid."); return False
    logger.debug("Starting inbox search (API)...");
    try:
        processor = InboxProcessor(session_manager=session_manager)
        result = processor.search_inbox()
        if result is False: logger.error("Inbox search reported failure."); return False
        else: logger.debug("Inbox search OK."); return True
    except Exception as e: logger.error(f"Error during inbox search: {e}", exc_info=True); return False
# End of srch_inbox_actn

# Action 8
def send_messages_action(session_manager, *args):
    """Action to send messages to DNA matches."""
    if not session_manager or not session_manager.driver or not session_manager.session_active: logger.error("Cannot send messages: Session not active."); return False
    logger.info("Starting message sending...")
    try:
        if not nav_to_page(session_manager.driver, config_instance.BASE_URL, WAIT_FOR_PAGE_SELECTOR, session_manager): logger.error("Failed nav to base URL. Aborting."); return False
        result = send_messages_to_matches(session_manager)
        if result is False: logger.error("Message sending reported failure."); return False
        else: logger.info("Message sending OK."); return True
    except Exception as e: logger.error(f"Error during message sending: {e}", exc_info=True); return False
# End of send_messages_action

# Action 9
def all_but_first_actn(session_manager, *args):
    """Action to delete all 'people' rows except the first."""
    if not session_manager: logger.error("SessionManager required."); return False
    session = None; success = False
    try:
        logger.info("Deleting all but first person record..."); session = session_manager.get_db_conn()
        if session is None: raise Exception("Failed to get DB session.")
        with db_transn(session) as sess:
            first = sess.query(Person).order_by(Person.id.asc()).first()
            if not first: logger.info("People table empty."); return True
            logger.debug(f"Keeping ID: {first.id} ({first.username})")
            to_delete = sess.query(Person).filter(Person.id != first.id).all()
            count = 0; total = len(to_delete)
            if not to_delete: logger.info("Only one person found.")
            else:
                logger.debug(f"Deleting {total} people...")
                for i, person in enumerate(to_delete):
                    logger.debug(f"Deleting {i+1}/{total}: {person.username} (ID: {person.id})")
                    sess.delete(person); count += 1
                logger.info(f"Deleted {count} people.")
        success = True
    except Exception as e: logger.error(f"Error during deletion: {e}", exc_info=True)
    finally:
        logger.debug("Cleaning up delete action...");
        if session_manager and session: session_manager.return_session(session)
        logger.debug("Delete action finished.")
    return success
# end of Action 9

def main():
    # Import inspect locally within main if needed (renamed due to conflict)
    import inspect as std_inspect

    global logger, session_manager # Declare globals

    session_manager = None # Initialize global session_manager

    try:
        # Clear screen and print initial spacer
        os.system("cls" if os.name == "nt" else "clear"); print("")

        # --- Setup Logging ---
        try:
            from config import config_instance # Keep local import for clarity
            db_file_path = config_instance.DATABASE_FILE
            log_filename = db_file_path.with_suffix(".log").name
            # Pass log_level from config if available
            log_level_to_set = getattr(config_instance, 'LOG_LEVEL', 'INFO').upper()
            logger = setup_logging(log_file=log_filename, log_level=log_level_to_set)
            # logger.info(f"Initial logger level configured to: {log_level_to_set}") # Log initial level
        except Exception as log_setup_e:
            print(f"CRITICAL: Logging setup error: {log_setup_e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
            logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
            logger = logging.getLogger("logger_fallback"); logger.warning("Using fallback logging.")

        # Instantiate SessionManager AFTER logging setup AND logger level adjustment
        session_manager = SessionManager() # Assign to global variable

        # --- Main menu loop ---
        while True:
            choice = menu(); print("") # Spacer

            # --- Action Dispatching ---
            if choice == "0": exec_actn(test_url_speed_action, session_manager, choice, close_sess=True)
            elif choice == "1": exec_actn(run_actions_6_7_8_action, session_manager, choice, close_sess=False) # Keep session open after sequence
            elif choice == "2": exec_actn(reset_db_actn, session_manager, choice, close_sess=False) # DB actions don't need browser closed
            elif choice == "3": exec_actn(backup_db_actn, session_manager, choice, close_sess=False) # DB actions don't need browser closed
            elif choice == "4": exec_actn(restore_db_actn, session_manager, choice, close_sess=False) # DB actions don't need browser closed
            elif choice == "5": exec_actn(check_login_actn, session_manager, choice, close_sess=True) # Check login should close its own session
            elif choice.startswith("6"):
                parts = choice.split()
                start_page = 1
                if len(parts) > 1:
                    try:
                        start_page_arg = int(parts[1]) # Convert
                        if start_page_arg <= 0:
                            raise ValueError("Start page must be positive")
                        start_page = start_page_arg # Assign only if valid
                    except ValueError as e:
                        print(f"Invalid start page: {e}. Using page 1.")
                        start_page = 1
                # Pass start_page using *args mechanism correctly handled by exec_actn
                exec_actn(coord_action, session_manager, "6", True, start_page) # Pass start_page as the last positional arg

            elif choice == "7": exec_actn(srch_inbox_actn, session_manager, choice, close_sess=True) # Close browser after inbox search
            elif choice == "8": exec_actn(send_messages_action, session_manager, choice, close_sess=True) # Close browser after sending
            elif choice == "9": exec_actn(all_but_first_actn, session_manager, choice, close_sess=False) # DB action

            # --- Meta Options ---
            elif choice == "t":
                os.system("cls" if os.name == "nt" else "clear")
                if logger and logger.handlers:
                    console_handler = None
                    # Find the specific handler for stderr
                    for handler in logger.handlers:
                        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                            console_handler = handler
                            break
                    if console_handler:
                        current_level = console_handler.level
                        # Toggle between DEBUG and INFO
                        new_level = logging.DEBUG if current_level > logging.DEBUG else logging.INFO
                        new_level_name = logging.getLevelName(new_level)
                        # Re-run setup_logging to apply the new level globally (including console)
                        # Ensure logger instance is reassigned if setup_logging potentially returns a new one
                        logger = setup_logging(log_level=new_level_name)
                        logger.info(f"Console log level toggled to: {new_level_name}") # Log the change
                    else:
                        logger.warning("Could not find console handler to toggle level.")
                else:
                    print("WARNING: Logger not ready or has no handlers.", file=sys.stderr)

            elif choice == "c": os.system("cls" if os.name == "nt" else "clear")
            elif choice == "q": os.system("cls" if os.name == "nt" else "clear"); print("Exiting."); break
            else: print("Invalid choice.\n")

    # --- Exception Handling & Cleanup ---
    except KeyboardInterrupt: os.system("cls" if os.name == "nt" else "clear"); print("\nCTRL+C detected. Exiting.")
    except Exception as e:
        # Use logger if available, otherwise print to stderr
        if 'logger' in globals() and logger:
             logger.critical(f"Critical error in main: {e}", exc_info=True)
        else:
             print(f"CRITICAL ERROR (no logger): {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
    finally:
        # Use session_manager if it was successfully initialized
        if 'session_manager' in locals() and session_manager:
            active_before = session_manager.session_active
            if active_before:
                 if logger: logger.info("Performing final browser cleanup...")
                 else: print("Performing final browser cleanup...", file=sys.stderr)
                 session_manager.close_sess()
            # Ensure DB connections are closed even if browser wasn't active
            if logger: logger.debug("Performing final DB cleanup...")
            else: print("Performing final DB cleanup...", file=sys.stderr)
            session_manager.cls_db_conn()

        # Log final message using logger if available
        if 'logger' in globals() and logger:
            logger.info("--- Main program execution finished ---")
        else:
            print("--- Main program execution finished (logger unavailable) ---", file=sys.stderr)
        print("\nExecution finished.")
# end main

# --- Entry Point ---
if __name__ == "__main__":
    main()