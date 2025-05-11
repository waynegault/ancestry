#!/usr/bin/env python3
"""
Test script for action8_messaging.py to identify issues.
"""
import sys
import traceback
import logging
from utils import SessionManager
import action8_messaging
from action8_messaging import send_messages_to_matches
from logging_config import logger, setup_logging
from config import config_instance

# Set up more detailed logging
setup_logging(log_level="DEBUG")


def test_action8():
    """Test action8_messaging.py"""
    try:
        # Initialize SessionManager
        logger.info("Initializing SessionManager...")
        session_manager = SessionManager()

        # Set required properties for testing
        logger.info("Setting required properties on SessionManager...")
        session_manager.my_profile_id = config_instance.TESTING_PROFILE_ID
        session_manager.my_tree_id = config_instance.MY_TREE_ID
        session_manager.driver_live = True
        session_manager.session_ready = True

        # Mock the login_status function to always return True
        logger.info("Mocking login_status function...")
        original_login_status = action8_messaging.login_status

        def mock_login_status(*args, **kwargs):
            logger.info("Mock login_status called - returning True")
            return True

        action8_messaging.login_status = mock_login_status

        try:
            # Test send_messages_to_matches function
            logger.info("Testing send_messages_to_matches function...")
            result = send_messages_to_matches(session_manager)

            logger.info(f"Function result: {result}")
            return result
        finally:
            # Restore the original login_status function
            action8_messaging.login_status = original_login_status

    except Exception as e:
        logger.error(f"Error testing action8_messaging: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_action8()
