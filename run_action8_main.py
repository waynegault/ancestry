#!/usr/bin/env python3
"""
Script to run action 8 directly from main.py
"""
import sys
import traceback
from logging_config import logger, setup_logging
from main import exec_actn, send_messages_action
from utils import SessionManager

# Set up detailed logging
setup_logging(log_level="DEBUG")

def run_action8_main():
    """Run action 8 directly from main.py"""
    try:
        logger.info("Initializing SessionManager...")
        session_manager = SessionManager()
        
        # Set required properties for testing
        logger.info("Setting required properties on SessionManager...")
        session_manager.driver_live = True
        session_manager.session_ready = True
        
        logger.info("Running action 8 from main.py...")
        result = exec_actn(send_messages_action, session_manager, "8")
        logger.info(f"Action 8 result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error running action 8: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = run_action8_main()
    print(f"\nFinal result: {result}")
