#!/usr/bin/env python3
"""
Test script for action8_messaging.py that mocks the browser session
"""
import sys
import traceback
import logging
from unittest import mock
from utils import SessionManager
import action8_messaging
from action8_messaging import send_messages_to_matches
from logging_config import logger, setup_logging
from config import config_instance
from main import exec_actn, send_messages_action, nav_to_page, WAIT_FOR_PAGE_SELECTOR

# Set up detailed logging
setup_logging(log_level="DEBUG")

def test_action8_exec():
    """Test action8_messaging.py with mocked browser session"""
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
        
        # Create a mock driver
        logger.info("Step 3: Creating mock driver...")
        mock_driver = mock.MagicMock()
        mock_driver.current_url = config_instance.BASE_URL
        mock_driver.window_handles = ["mock_handle"]
        mock_driver.get_cookies.return_value = [
            {"name": "ANCSESSIONID", "value": "mock_session_id"},
            {"name": "SecureATT", "value": "mock_secure_att"},
        ]
        session_manager.driver = mock_driver
        
        # Mock the login_status function to always return True
        logger.info("Step 4: Mocking login_status function...")
        original_login_status = action8_messaging.login_status
        
        def mock_login_status(*args, **kwargs):
            logger.info("Mock login_status called - returning True")
            return True
            
        action8_messaging.login_status = mock_login_status
        
        # Mock the nav_to_page function
        logger.info("Step 5: Mocking nav_to_page function...")
        original_nav_to_page = sys.modules['main'].nav_to_page
        
        def mock_nav_to_page(*args, **kwargs):
            logger.info("Mock nav_to_page called - returning True")
            return True
            
        sys.modules['main'].nav_to_page = mock_nav_to_page
        
        # Mock the ensure_session_ready function
        logger.info("Step 6: Mocking ensure_session_ready function...")
        original_ensure_session_ready = session_manager.ensure_session_ready
        
        def mock_ensure_session_ready(*args, **kwargs):
            logger.info("Mock ensure_session_ready called - returning True")
            session_manager.session_ready = True
            return True
            
        session_manager.ensure_session_ready = mock_ensure_session_ready
        
        try:
            # Call the action through exec_actn
            logger.info("Step 7: Calling exec_actn with send_messages_action...")
            result = exec_actn(send_messages_action, session_manager, "8")
            
            logger.info(f"Step 8: Function result: {result}")
            return result
        finally:
            # Restore the original functions
            logger.info("Step 9: Restoring original functions...")
            action8_messaging.login_status = original_login_status
            sys.modules['main'].nav_to_page = original_nav_to_page
            session_manager.ensure_session_ready = original_ensure_session_ready
            
    except Exception as e:
        logger.error(f"Error testing action8_messaging: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_action8_exec()
    print(f"\nFinal result: {result}")
