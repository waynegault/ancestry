#!/usr/bin/env python3
"""
Debug script for main.py action 8 to identify the exact point of failure.
"""
import sys
import traceback
import logging
from utils import SessionManager
import action8_messaging
from logging_config import logger, setup_logging
from config import config_instance
import main

# Set up detailed logging
setup_logging(log_level="DEBUG")

def debug_main_action8():
    """Debug main.py action 8 with detailed step-by-step logging"""
    try:
        # Initialize SessionManager
        logger.info("Step 1: Initializing SessionManager...")
        session_manager = SessionManager()
        
        # Add debug logging to key functions
        logger.info("Step 2: Adding debug logging to key functions...")
        
        # Patch the send_messages_to_matches function
        original_send_messages = action8_messaging.send_messages_to_matches
        
        def debug_send_messages(*args, **kwargs):
            logger.info("Debug: Entering send_messages_to_matches...")
            try:
                # Log the arguments
                if args:
                    logger.info(f"Debug: send_messages_to_matches args: {args}")
                if kwargs:
                    logger.info(f"Debug: send_messages_to_matches kwargs: {kwargs}")
                
                # Call the original function
                result = original_send_messages(*args, **kwargs)
                logger.info(f"Debug: send_messages_to_matches result: {result}")
                return result
            except Exception as e:
                logger.error(f"Debug: Error in send_messages_to_matches: {e}")
                logger.error(traceback.format_exc())
                raise
                
        action8_messaging.send_messages_to_matches = debug_send_messages
        
        # Patch the safe_column_value function
        original_safe_column_value = action8_messaging.safe_column_value
        
        def debug_safe_column_value(obj, attr_name, default=None):
            try:
                logger.info(f"Debug: safe_column_value called with obj={type(obj)}, attr_name={attr_name}, default={default}")
                
                # Check if the attribute exists
                has_attr = hasattr(obj, attr_name)
                logger.info(f"Debug: hasattr(obj, {attr_name}) = {has_attr}")
                
                if has_attr:
                    # Get the attribute value
                    value = getattr(obj, attr_name)
                    logger.info(f"Debug: getattr(obj, {attr_name}) = {value} (type: {type(value)})")
                    
                    # Call the original function
                    result = original_safe_column_value(obj, attr_name, default)
                    logger.info(f"Debug: safe_column_value result: {result} (type: {type(result)})")
                    return result
                else:
                    logger.info(f"Debug: Attribute {attr_name} not found, returning default: {default}")
                    return default
            except Exception as e:
                logger.error(f"Debug: Error in safe_column_value: {e}")
                logger.error(traceback.format_exc())
                return default
                
        action8_messaging.safe_column_value = debug_safe_column_value
        
        # Patch the send_messages_action function
        original_send_messages_action = main.send_messages_action
        
        def debug_send_messages_action(*args, **kwargs):
            logger.info("Debug: Entering send_messages_action...")
            try:
                # Log the arguments
                if args:
                    logger.info(f"Debug: send_messages_action args: {args}")
                if kwargs:
                    logger.info(f"Debug: send_messages_action kwargs: {kwargs}")
                
                # Call the original function
                result = original_send_messages_action(*args, **kwargs)
                logger.info(f"Debug: send_messages_action result: {result}")
                return result
            except Exception as e:
                logger.error(f"Debug: Error in send_messages_action: {e}")
                logger.error(traceback.format_exc())
                raise
                
        main.send_messages_action = debug_send_messages_action
        
        try:
            # Call the action directly
            logger.info("Step 3: Calling send_messages_action...")
            result = main.send_messages_action(session_manager)
            
            logger.info(f"Step 4: Function result: {result}")
            return result
        finally:
            # Restore the original functions
            logger.info("Step 5: Restoring original functions...")
            action8_messaging.send_messages_to_matches = original_send_messages
            action8_messaging.safe_column_value = original_safe_column_value
            main.send_messages_action = original_send_messages_action
            
    except Exception as e:
        logger.error(f"Error testing main action 8: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    result = debug_main_action8()
    print(f"\nFinal result: {result}")
