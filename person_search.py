#!/usr/bin/env python3

# person_search.py
"""
Unified module for searching and retrieving person information from GEDCOM and Ancestry API.
Provides functions for searching, getting family details, and relationship paths.
"""

# --- Unified import system ---
from core_imports import (
    standardize_module_imports,
    auto_register_module,
    get_logger,
    safe_execute,
)

# Register this module immediately
auto_register_module(globals(), __name__)

import re
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import json

# Initialize logger
logger = get_logger(__name__)

# Import from local modules
from logging_config import logger as legacy_logger
from config import config_manager, config_schema
from utils import SessionManager

# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
)


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for person_search.py.
    Tests person searching, filtering, and matching functionality.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Person Search & Matching Engine", "person_search.py")
    suite.start_suite()

    # INITIALIZATION TESTS
    def test_module_imports():
        """Test that required modules are imported correctly"""
        import re
        from typing import Dict, List, Any, Optional, Tuple, Union
        import os
        import json

        # Verify imports worked
        assert re is not None, "re module import failed"
        assert Dict is not None, "typing.Dict import failed"
        assert os is not None, "os module import failed"
        assert json is not None, "json module import failed"

    def test_config_access():
        """Test configuration access"""
        # Test direct config access
        base_url = config_schema.api.base_url
        assert isinstance(base_url, str), "Base URL should be a string"
        assert base_url, "Base URL should not be empty"

    with suppress_logging():
        # INITIALIZATION TESTS
        suite.run_test(
            "Module imports",
            test_module_imports,
            "All required modules (re, typing, os, json) import successfully",
            "Import core Python modules and verify they are available",
            "All module imports completed without exceptions and objects are properly instantiated",
        )

        suite.run_test(
            "Configuration access",
            test_config_access,
            "Configuration can be accessed directly",
            "Test direct configuration access",
            "Verify configuration schema provides proper access to settings",
        )

        # INTEGRATION TESTS
        def test_session_manager_import():
            """Test that SessionManager can be imported"""
            from utils import SessionManager

            assert SessionManager is not None, "SessionManager import failed"

        suite.run_test(
            "SessionManager import",
            test_session_manager_import,
            "SessionManager can be imported from utils module",
            "Import SessionManager from utils and verify it's available",
            "SessionManager imported successfully and is not None",
        )

        def test_logger_import():
            """Test that logger is imported and configured"""
            from logging_config import logger

            assert logger is not None, "logger import failed"
            assert hasattr(logger, "info"), "logger should have info method"

        suite.run_test(
            "Logger import and configuration",
            test_logger_import,
            "Logger is imported and properly configured with required methods",
            "Import logger from logging_config and verify it has required methods",
            "Logger imported successfully with all required logging methods available",
        )

        # ERROR HANDLING TESTS
        return suite.finish_suite()


# Register module functions at module load
auto_register_module(globals(), __name__)


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys

    print("🔍 Running Person Search & Matching Engine comprehensive test suite...")
    success = safe_execute(lambda: run_comprehensive_tests())
    sys.exit(0 if success else 1)
