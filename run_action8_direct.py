#!/usr/bin/env python3

"""
run_action8_direct.py - Script to run Action 8 (Send Messages) directly

This script calls the send_messages_to_matches function from action8_messaging.py directly.
"""

import sys
from utils import SessionManager
from action8_messaging import send_messages_to_matches

def run_action8_direct():
    """
    Runs Action 8 (Send Messages) directly by calling send_messages_to_matches from action8_messaging.py.
    
    Returns:
        bool: True if the action completed successfully, False otherwise.
    """
    try:
        print("\n=== Running Action 8 (Send Messages) Directly ===\n")
        
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
        if not session_manager.ensure_session_ready(action_name="Action 8 Direct"):
            print("❌ Failed to ensure session is ready.")
            return False
        
        # Run send_messages_to_matches
        print("Starting message sending...\n")
        result = send_messages_to_matches(session_manager)
        
        if result:
            print("\n✅ Message sending completed successfully.")
        else:
            print("\n❌ Message sending failed. Check the logs for details.")
        
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
        if 'session_manager' in locals() and session_manager:
            session_manager.close_sess(keep_db=True)
        print("Session closed.")

if __name__ == "__main__":
    # Run Action 8
    success = run_action8_direct()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
