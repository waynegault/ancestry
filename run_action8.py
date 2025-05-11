#!/usr/bin/env python3
"""
Script to run action 8 directly from main.py
"""
import sys
import traceback
from logging_config import logger, setup_logging
from main import exec_actn

# Set up detailed logging
setup_logging(log_level="DEBUG")

def run_action8():
    """Run action 8 directly from main.py"""
    try:
        logger.info("Running action 8 from main.py...")
        result = exec_actn(8)
        logger.info(f"Action 8 result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error running action 8: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = run_action8()
    print(f"\nFinal result: {result}")
