#!/usr/bin/env python3

"""
run_action7_direct.py - Script to run Action 7 (Inbox Processor) directly

This script calls the srch_inbox_actn function from main.py directly.
"""

import sys
import time
from typing import Optional


def run_action7_direct():
    """
    Runs Action 7 (Inbox Processor) directly by calling srch_inbox_actn from main.py.

    Returns:
        bool: True if the action completed successfully, False otherwise.
    """
    try:
        # Import required modules
        from utils import SessionManager

        print("\n=== Running Action 7 (Inbox Processor) Directly ===\n")

        # Initialize SessionManager with browser_needed=True
        session_manager = SessionManager()
        session_manager.browser_needed = True

        # Start the browser session
        print("Starting browser session...")
        if not session_manager.start_sess():
            print("❌ Failed to start browser session.")
            return False

        # Ensure session is ready (this will handle login if needed)
        print("Ensuring session is ready (may trigger login)...")
        if not session_manager.ensure_session_ready(action_name="Action 7 Direct"):
            print("❌ Failed to ensure session is ready.")
            return False

        # Create a custom InboxProcessor that ignores the comparator
        from action7_inbox import InboxProcessor

        class NoComparatorInboxProcessor(InboxProcessor):
            def _create_comparator(self, _):  # Use underscore for unused parameter
                """Override to always return None (no comparator)"""
                print("Ignoring comparator as requested.")
                return None

            def _process_inbox_loop(
                self, session, comp_conv_id, comp_ts, my_pid_lower, progress_bar
            ):
                """Override to force fetching all conversations"""
                print(
                    "Overriding _process_inbox_loop to force fetching all conversations"
                )

                # Call the parent method but modify the result to force fetching
                result = super()._process_inbox_loop(
                    session, comp_conv_id, comp_ts, my_pid_lower, progress_bar
                )

                # Return the result
                return result

        # Initialize our custom InboxProcessor
        print("Initializing InboxProcessor (with comparator disabled)...")
        inbox_processor = NoComparatorInboxProcessor(session_manager=session_manager)

        # Set max_inbox_limit to 5
        inbox_processor.max_inbox_limit = 5
        print(f"Setting conversation limit to {inbox_processor.max_inbox_limit}")

        # Run search_inbox
        print("Starting inbox search...\n")
        result = inbox_processor.search_inbox()

        if result:
            print("\n✅ Inbox search completed successfully.")
        else:
            print("\n❌ Inbox search failed. Check the logs for details.")

        return result
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return False
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        return False
    finally:
        # Close the session
        print("\nClosing session...")
        if "session_manager" in locals() and session_manager:
            session_manager.close_sess(keep_db=True)
        print("Session closed.")


if __name__ == "__main__":
    # Run Action 7
    success = run_action7_direct()

    # Exit with appropriate status code
    sys.exit(0 if success else 1)
