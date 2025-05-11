#!/usr/bin/env python3
"""
Test script for action8_messaging.py that mimics how it's called from main.py
"""
import sys
import traceback
import logging
from utils import SessionManager
import action8_messaging
from action8_messaging import send_messages_to_matches
from logging_config import logger, setup_logging
from config import config_instance
from main import send_messages_action, nav_to_page, WAIT_FOR_PAGE_SELECTOR

# Set up detailed logging
setup_logging(log_level="DEBUG")

def test_action8_main():
    """Test action8_messaging.py as it would be called from main.py"""
    try:
        # Initialize SessionManager
        logger.info("Step 1: Initializing SessionManager...")
        session_manager = SessionManager()
        
        # Set required properties for testing
        logger.info("Step 2: Setting required properties on SessionManager...")
        session_manager.my_profile_id = config_instance.TESTING_PROFILE_ID
        session_manager.my_tree_id = config_instance.MY_TREE_ID
        session_manager.driver_live = True
        session_manager.session_ready = True
        
        # Mock the login_status function to always return True
        logger.info("Step 3: Mocking login_status function...")
        original_login_status = action8_messaging.login_status
        
        def mock_login_status(*args, **kwargs):
            logger.info("Mock login_status called - returning True")
            return True
            
        action8_messaging.login_status = mock_login_status
        
        # Mock the nav_to_page function
        logger.info("Step 4: Mocking nav_to_page function...")
        original_nav_to_page = sys.modules['main'].nav_to_page
        
        def mock_nav_to_page(*args, **kwargs):
            logger.info("Mock nav_to_page called - returning True")
            return True
            
        sys.modules['main'].nav_to_page = mock_nav_to_page
        
        try:
            # Call the action as it would be called from main.py
            logger.info("Step 5: Calling send_messages_action...")
            result = send_messages_action(session_manager)
            
            logger.info(f"Step 6: Function result: {result}")
            return result
        finally:
            # Restore the original functions
            logger.info("Step 7: Restoring original functions...")
            action8_messaging.login_status = original_login_status
            sys.modules['main'].nav_to_page = original_nav_to_page
            
    except Exception as e:
        logger.error(f"Error testing action8_messaging: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_action8_main()
    print(f"\nFinal result: {result}")
