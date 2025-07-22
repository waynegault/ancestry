"""
Standardized Imports Module - Single Source of Truth

This module provides a single, consistent way to import core functionality
across the entire Ancestry project, eliminating the 5+ different import
patterns currently scattered throughout the codebase.

Usage in any module:
    from standard_imports import *

    # Everything is now available:
    logger.info("Standardized logging")
    register_function("my_func", my_function)
    # etc.

This replaces inconsistent patterns like:
- from core_imports import auto_register_module, get_logger, ...
- try: from core_imports import ... except ImportError: ...
- from logging_config import logger
- Various fallback patterns
"""

# === CORE UNIFIED IMPORTS ===
from core_imports import (
    # Core functionality
    auto_register_module,
    get_logger,
    standardize_module_imports,
    safe_execute,
    # Function registry
    register_function,
    get_function,
    is_function_available,
    call_function,
    get_available_functions,
    register_many,
    # Utilities
    get_project_root,
    import_context,
    cleanup_registry,
    get_stats,
    ensure_imports,
)

# === STANDARD LIBRARY IMPORTS (COMMONLY NEEDED) ===
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple


# === STANDARDIZED LOGGER ===
# Single logger pattern for all modules
def get_standard_logger(module_name: str) -> logging.Logger:
    """Get a standardized logger for any module."""
    return get_logger(module_name)


# === AUTO-REGISTRATION HELPER ===
def setup_module(module_globals: Dict[str, Any], module_name: str) -> logging.Logger:
    """
    One-line module setup that handles everything:
    - Auto-registration
    - Logger setup
    - Import standardization

    Usage at top of any module:
        from standard_imports import setup_module
        logger = setup_module(globals(), __name__)
    """
    auto_register_module(module_globals, module_name)
    standardize_module_imports()
    return get_standard_logger(module_name)


# === ERROR HANDLING UTILITIES ===
def safe_import(module_name: str, fallback=None):
    """Safely import a module with fallback."""
    try:
        import importlib

        return importlib.import_module(module_name)
    except ImportError:
        return fallback


def safe_import_from(module_name: str, item_name: str, fallback=None):
    """Safely import an item from a module with fallback."""
    try:
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, item_name, fallback)
    except (ImportError, AttributeError):
        return fallback


# === TESTING INTEGRATION ===
def get_unified_test_framework():
    """Get the unified test framework, with fallback to individual tests."""
    try:
        from test_framework_unified import StandardTestFramework

        return StandardTestFramework
    except ImportError:
        # Fallback to legacy test framework
        return None


# === EXPORTS ===
# Make everything available for "from standard_imports import *"
__all__ = [
    # Core functionality
    "auto_register_module",
    "get_logger",
    "standardize_module_imports",
    "safe_execute",
    # Function registry
    "register_function",
    "get_function",
    "is_function_available",
    "call_function",
    "get_available_functions",
    "register_many",
    # Utilities
    "get_project_root",
    "import_context",
    "cleanup_registry",
    "get_stats",
    "ensure_imports",
    # Standardized helpers
    "setup_module",
    "get_standard_logger",
    "safe_import",
    "safe_import_from",
    "get_unified_test_framework",
    # Common types
    "Dict",
    "List",
    "Any",
    "Optional",
    "Callable",
    "Union",
    "Tuple",
    # Common modules
    "os",
    "sys",
    "time",
    "logging",
    "Path",
]

# === USAGE EXAMPLES ===

"""
BEFORE (Inconsistent patterns across codebase):

# Pattern 1 - Verbose and error-prone
from core_imports import (
    auto_register_module,
    get_logger, 
    standardize_module_imports,
    register_function,
    get_function,
    is_function_available,
)
auto_register_module(globals(), __name__)
logger = get_logger(__name__)

# Pattern 2 - Try/except fallbacks everywhere  
try:
    from core_imports import register_function
except ImportError:
    register_function = None

# Pattern 3 - Mixed logger sources
from logging_config import logger

# Pattern 4 - Manual fallbacks
try:
    from core_imports import auto_register_module
    pass  # Already registered at line 169
except ImportError:
    pass

AFTER (Single standardized pattern):

# Option 1 - Full control
from standard_imports import *
logger = setup_module(globals(), __name__)

# Option 2 - Individual imports 
from standard_imports import setup_module, register_function, safe_execute
logger = setup_module(globals(), __name__)

# Option 3 - Minimal
from standard_imports import setup_module
logger = setup_module(globals(), __name__)
# Everything else available via standard_imports if needed
"""

# Self-test
if __name__ == "__main__":
    # Test that all imports work
    logger = setup_module(globals(), __name__)
    logger.info("✅ Standard imports module loaded successfully")

    # Test function registration
    def test_function():
        return "test_result"

    register_function("test_standard_imports", test_function)
    assert is_function_available("test_standard_imports")
    assert get_function("test_standard_imports")() == "test_result"

    logger.info("✅ All standard imports functionality verified")
    print("Standard imports module ready for use across the codebase!")
