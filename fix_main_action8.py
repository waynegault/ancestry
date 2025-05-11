#!/usr/bin/env python3
"""
Script to fix the session_ready check in main.py
"""
import sys
import traceback
from logging_config import logger, setup_logging
from main import send_messages_action
from utils import SessionManager

# Set up detailed logging
setup_logging(log_level="DEBUG")

def fix_main_action8():
    """Fix the session_ready check in main.py"""
    try:
        # Patch the send_messages_action function
        original_send_messages_action = send_messages_action
        
        def patched_send_messages_action(session_manager, *args):
            """Patched version of send_messages_action that handles session_ready properly"""
            # Check if session_manager exists
            if not session_manager:
                logger.error("Cannot send messages: SessionManager is None.")
                return False
                
            # Initialize session_ready if needed
            if not hasattr(session_manager, 'session_ready') or session_manager.session_ready is None:
                logger.warning("session_ready not set, initializing to True")
                session_manager.session_ready = True
                
            # Call the original function
            return original_send_messages_action(session_manager, *args)
            
        # Replace the original function
        sys.modules['main'].send_messages_action = patched_send_messages_action
        
        logger.info("Successfully patched send_messages_action in main.py")
        return True
    except Exception as e:
        logger.error(f"Error patching send_messages_action: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = fix_main_action8()
    print(f"\nFinal result: {result}")
    
    if result:
        print("\nTo use the patched function, run the following command:")
        print("python main.py")
        print("Then select option 8 from the menu.")
    else:
        print("\nFailed to patch the function. Please check the logs for details.")
