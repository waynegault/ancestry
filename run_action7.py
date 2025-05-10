#!/usr/bin/env python3

"""
run_action7.py - Script to run Action 7 (Inbox Processor) with proper initialization

This script initializes the SessionManager, starts the browser, ensures the session is ready,
and then runs the InboxProcessor to search the inbox.
"""

import sys
import time
from typing import Optional

from action7_inbox import InboxProcessor
from utils import SessionManager
from config import config_instance
from logging_config import logger, setup_logging


def run_action7():
    """
    Runs Action 7 (Inbox Processor) with proper initialization.

    Returns:
        bool: True if the action completed successfully, False otherwise.
    """
    # Setup logging
    logger = setup_logging()

    print("\n=== Running Action 7: Inbox Processor ===\n")

    # Initialize SessionManager
    session_manager = SessionManager()

    try:
        # Start the browser session
        print("Starting browser session...")
        if not session_manager.start_sess():
            logger.error("Failed to start browser session.")
            return False

        # Ensure session is ready (this will handle login if needed)
        print("Ensuring session is ready (may trigger login)...")
        if not session_manager.ensure_session_ready(action_name="Action 7 Setup"):
            logger.error("Failed to ensure session is ready.")
            return False

        # Check if my_profile_id is set
        if not session_manager.my_profile_id:
            logger.warning(
                "Profile ID not set after session initialization. Using default from config."
            )
            # Set profile ID manually from config
            session_manager.my_profile_id = config_instance.TESTING_PROFILE_ID
            if not session_manager.my_profile_id:
                logger.error("No profile ID available in config. Cannot proceed.")
                return False

        print(f"Session ready. Profile ID: {session_manager.my_profile_id}")

        # Initialize InboxProcessor
        print("Initializing InboxProcessor...")
        inbox_processor = InboxProcessor(session_manager=session_manager)

        # Run search_inbox
        print("Starting inbox search...\n")
        result = inbox_processor.search_inbox()

        if result:
            print("\n✓ Inbox search completed successfully.")
        else:
            print("\n✗ Inbox search failed. Check the logs for details.")

        return result
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return False
    except Exception as e:
        logger.error(f"Error during Action 7 execution: {e}", exc_info=True)
        print(f"\n✗ Error during execution: {e}")
        return False
    finally:
        # Close the session
        print("\nClosing session...")
        if session_manager:
            session_manager.close_sess(keep_db=True)
        print("Session closed.")


if __name__ == "__main__":
    # Check for command line arguments
    verbose = False
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["verbose", "-v", "--verbose"]:
        verbose = True
        print("Verbose mode enabled.")

    # Run Action 7
    success = run_action7()

    # Exit with appropriate status code
    sys.exit(0 if success else 1)
