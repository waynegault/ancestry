#!/usr/bin/env python3

"""
run_action9_direct.py - Script to run Action 9 (Process Productive Messages) directly

This script calls the process_productive_messages function from action9_process_productive.py directly.
"""

import sys
from utils import SessionManager
from action9_process_productive import process_productive_messages

def run_action9_direct():
    """
    Runs Action 9 (Process Productive Messages) directly by calling process_productive_messages.
    
    Returns:
        bool: True if the action completed successfully, False otherwise.
    """
    try:
        print("\n=== Running Action 9 (Process Productive Messages) Directly ===\n")
        print("Action 9 processes messages that have been classified as 'PRODUCTIVE' by the AI.")
        print("It performs the following tasks:")
        print(" 1. Finds people with unprocessed productive messages")
        print(" 2. Uses AI to extract information from these messages")
        print(" 3. Creates Microsoft To-Do tasks for follow-up actions")
        print(" 4. Sends acknowledgement messages to the senders")
        print(" 5. Archives the conversations after processing")
        print("\nStarting Action 9...\n")
        
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
        if not session_manager.ensure_session_ready(action_name="Action 9 Direct"):
            print("❌ Failed to ensure session is ready.")
            return False
        
        # Run process_productive_messages
        print("Starting productive message processing...\n")
        result = process_productive_messages(session_manager)
        
        if result:
            print("\n✅ Productive message processing completed successfully.")
        else:
            print("\n❌ Productive message processing failed. Check the logs for details.")
        
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
    # Run Action 9
    success = run_action9_direct()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
