#!/usr/bin/env python3
"""
Debug script for action8_messaging.py to identify the exact point of failure.
"""
import sys
import traceback
import logging
from utils import SessionManager
import action8_messaging
from action8_messaging import send_messages_to_matches
from logging_config import logger, setup_logging
from config import config_instance

# Set up detailed logging
setup_logging(log_level="DEBUG")

def debug_action8():
    """Debug action8_messaging.py with detailed step-by-step logging"""
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
        
        # Mock the message templates
        logger.info("Step 4: Checking message templates...")
        if not action8_messaging.MESSAGE_TEMPLATES:
            logger.warning("MESSAGE_TEMPLATES is empty, creating mock templates...")
            # Create mock templates for all required keys
            mock_templates = {}
            for key in action8_messaging.MESSAGE_TYPES_ACTION8.keys():
                mock_templates[key] = f"Mock template for {key}"
            mock_templates["Productive_Reply_Acknowledgement"] = "Mock template for Productive_Reply_Acknowledgement"
            action8_messaging.MESSAGE_TEMPLATES = mock_templates
        
        try:
            # Test send_messages_to_matches function
            logger.info("Step 5: Testing send_messages_to_matches function...")
            
            # Add detailed logging to track execution
            original_prefetch = action8_messaging._prefetch_messaging_data
            
            def debug_prefetch(*args, **kwargs):
                logger.info("Executing _prefetch_messaging_data...")
                try:
                    result = original_prefetch(*args, **kwargs)
                    logger.info(f"_prefetch_messaging_data returned: {result}")
                    return result
                except Exception as e:
                    logger.error(f"Error in _prefetch_messaging_data: {e}")
                    traceback.print_exc()
                    raise
            
            action8_messaging._prefetch_messaging_data = debug_prefetch
            
            # Run the function
            logger.info("Step 6: Executing send_messages_to_matches...")
            result = send_messages_to_matches(session_manager)
            
            logger.info(f"Step 7: Function result: {result}")
            return result
        finally:
            # Restore the original functions
            logger.info("Step 8: Restoring original functions...")
            action8_messaging.login_status = original_login_status
            action8_messaging._prefetch_messaging_data = original_prefetch
            
    except Exception as e:
        logger.error(f"Error testing action8_messaging: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = debug_action8()
    print(f"\nFinal result: {result}")
