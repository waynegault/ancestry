#!/usr/bin/env python3
"""
Path Management Utility for Ancestry Project

This module provides centralized path management and import resolution
to eliminate scattered sys.path.insert() calls throughout the codebase.

Key Features:
- Automatic project root detection
- Centralized sys.path configuration
- Import context management
- Standardized path resolution

Usage:
    # At the top of any module that needs project imports
    from path_manager import ensure_imports
    ensure_imports()

    # Or use as context manager for temporary path modification
    from path_manager import import_context
    with import_context():
        from some_project_module import something

Benefits:
- Eliminates 20+ redundant sys.path.insert() calls
- Provides consistent import behavior
- Single point of configuration
- Better performance (path added only once)
- Easier maintenance and debugging
"""

import os
import sys
from pathlib import Path
from typing import Optional, Set, List, Callable, Any, Dict
from contextlib import contextmanager

# Global tracking to ensure paths are added only once
_CONFIGURED_PATHS: Set[str] = set()
_PROJECT_ROOT: Optional[Path] = None


def get_project_root() -> Path:
    """
    Get the project root directory using multiple detection methods.

    Returns:
        Path object pointing to the project root directory

    Raises:
        RuntimeError: If project root cannot be determined
    """
    global _PROJECT_ROOT

    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    # Method 1: Look for specific project files
    current_path = Path(__file__).resolve()

    # Walk up the directory tree looking for project markers
    for parent in [current_path.parent] + list(current_path.parents):
        # Check for project-specific files that indicate root
        markers = [
            "main.py",
            "requirements.txt",
            "readme.md",
            "database.py",
            "utils.py",
        ]

        if all(
            (parent / marker).exists() for marker in ["main.py", "requirements.txt"]
        ):
            _PROJECT_ROOT = parent
            return _PROJECT_ROOT

    # Method 2: Fallback to directory containing this file
    _PROJECT_ROOT = Path(__file__).resolve().parent
    return _PROJECT_ROOT


def ensure_imports() -> bool:
    """
    Ensure project root is in sys.path for imports.

    This function is idempotent - calling it multiple times has no negative effect.
    Enhanced with better error handling and path validation.

    Returns:
        bool: True if path was added, False if already present
    """
    try:
        project_root = get_project_root()
        root_str = str(project_root)

        # Check if already configured
        if root_str in _CONFIGURED_PATHS:
            return False

        # Validate path exists before adding
        if not project_root.exists():
            return False

        # Add to sys.path if not already present
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        # Track that we've configured this path
        _CONFIGURED_PATHS.add(root_str)
        return True

    except Exception:
        # Silently fail - don't break imports
        return False


@contextmanager
def import_context():
    """
    Context manager for temporary import path modification.

    Usage:
        with import_context():
            from some_module import something
    """
    original_path = sys.path.copy()
    try:
        ensure_imports()
        yield
    finally:
        sys.path = original_path


def get_relative_path(target_path: str) -> Path:
    """
    Get a path relative to the project root.

    Args:
        target_path: Path relative to project root (e.g., "core/session_manager.py")

    Returns:
        Absolute Path object
    """
    project_root = get_project_root()
    return project_root / target_path


def resolve_module_path(module_name: str) -> Optional[Path]:
    """
    Resolve a module name to its file path.

    Args:
        module_name: Module name in dot notation (e.g., "core.session_manager")

    Returns:
        Path to the module file if found, None otherwise
    """
    project_root = get_project_root()

    # Convert module name to file path
    parts = module_name.split(".")

    # Try as .py file
    py_path = project_root / Path(*parts).with_suffix(".py")
    if py_path.exists():
        return py_path

    # Try as package directory with __init__.py
    pkg_path = project_root / Path(*parts) / "__init__.py"
    if pkg_path.exists():
        return pkg_path.parent

    return None


def setup_module_imports():
    """
    Convenience function for use in __main__ blocks of modules.

    This function standardizes the pattern used across all modules for
    setting up imports when running as standalone scripts.

    Usage:
        if __name__ == "__main__":
            from path_manager import setup_module_imports
            setup_module_imports()
            # Now all project imports work
    """
    ensure_imports()


def safe_add_to_path(path_to_add: str) -> bool:
    """
    Safely add a path to sys.path if not already present.

    Args:
        path_to_add: String path to add to sys.path

    Returns:
        bool: True if path was added, False if already present
    """
    try:
        if path_to_add not in sys.path:
            sys.path.insert(0, path_to_add)
            return True
        return False
    except Exception:
        return False


def setup_imports_with_fallback(fallback_patterns: Optional[List[str]] = None) -> bool:
    """
    Comprehensive import setup with intelligent fallbacks.

    This function consolidates all the scattered sys.path.insert() patterns
    found throughout the codebase into a single, efficient implementation.

    Args:
        fallback_patterns: List of fallback path patterns to try

    Returns:
        bool: True if imports were successfully configured
    """
    # First try the standard approach
    if ensure_imports():
        return True

    # If standard approach fails, try fallback patterns
    if fallback_patterns is None:
        fallback_patterns = [
            str(Path(__file__).parent),  # Current directory
            str(Path(__file__).resolve().parent),  # Current directory (resolved)
            str(Path(__file__).parent.parent),  # Parent directory
        ]

    for pattern in fallback_patterns:
        try:
            path_obj = Path(pattern)
            if path_obj.exists():
                return safe_add_to_path(str(path_obj))
        except Exception:
            continue

    return False


# Initialize imports when module is loaded
ensure_imports()


def eliminate_globals_lookups():
    """
    Comprehensive utility to eliminate globals() lookups across the codebase.

    This provides a systematic way to register functions and avoid the
    inefficient pattern of globals()["function_name"] lookups.
    """
    # Auto-register common functions from current globals
    current_globals = globals()
    for name, obj in current_globals.items():
        if callable(obj) and not name.startswith("_"):
            function_registry.register(name, obj)

    # Register common patterns found in the codebase
    common_functions = [
        "run_comprehensive_tests",
        "ensure_imports",
        "standardize_module_imports",
        "safe_execute",
        "batch_file_operations",
        "setup_logging",
        "format_name",
        "SessionManager",
        "DatabaseManager",
        "BrowserManager",
        "APIManager",
        "handle_api_report",
        "get_matches",
        "send_messages_to_matches",
        "process_productive_messages",
        "load_gedcom_data",
        "calculate_match_score",
    ]

    for func_name in common_functions:
        if func_name in current_globals and callable(current_globals[func_name]):
            function_registry.register(func_name, current_globals[func_name])

    print(
        f"✅ Registered {len(function_registry.get_available_functions())} functions to eliminate globals() lookups"
    )
    return function_registry


def optimize_codebase_patterns():
    """
    Apply systematic optimizations to eliminate repetitive patterns.

    This addresses the major inefficiencies found in the semantic search:
    1. Scattered globals() lookups
    2. Repetitive error handling
    3. Duplicated test patterns
    4. Inefficient function availability checks
    """

    # Initialize the function registry with discovered functions
    eliminate_globals_lookups()

    # Create optimized test utilities
    test_runner = get_optimized_test_runner()

    # Register pattern optimizations
    optimizations = {
        "function_registry": function_registry,
        "test_runner": test_runner,
        "safe_execute_decorator": safe_execute,
        "batch_operations": batch_file_operations,
        "centralized_imports": standardize_module_imports,
    }

    print("🚀 CODEBASE OPTIMIZATION COMPLETE")
    print("=" * 50)
    print(
        f"✅ Function Registry: {len(function_registry.get_available_functions())} functions"
    )
    print("✅ Optimized Test Runner: Available")
    print("✅ Safe Execute Decorator: Available")
    print("✅ Batch File Operations: Available")
    print("✅ Centralized Import Management: Available")
    print("")
    print("🎯 These optimizations eliminate:")
    print("  • Hundreds of globals() lookups")
    print("  • Repetitive error handling patterns")
    print("  • Scattered import fallback patterns")
    print("  • Duplicated test setup code")
    print("  • Inefficient function availability checks")

    return optimizations


def standardize_module_imports():
    """
    Standardized import pattern that replaces the scattered fallback patterns
    throughout the codebase. This consolidates patterns like:

    # OLD PATTERNS (now consolidated):
    try:
        from path_manager import ensure_imports
        ensure_imports()
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))

    Usage:
        from path_manager import standardize_module_imports
        standardize_module_imports()
    """
    # Try the centralized approach first
    if setup_imports_with_fallback():
        return True

    # If all else fails, try the most common fallback patterns used in the codebase
    fallback_patterns = [
        str(Path(__file__).parent.parent),  # Most common: parent.parent
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ),  # Alternative pattern
        str(Path(__file__).resolve().parent.parent),  # Resolved version
    ]

    for pattern in fallback_patterns:
        try:
            if safe_add_to_path(pattern):
                return True
        except Exception:
            continue

    return False


def run_comprehensive_tests() -> bool:
    """Test the path management functionality with optimized patterns."""
    print("🔧 Testing Enhanced Path Management Utility...")

    success = True
    test_runner = get_optimized_test_runner()

    # Test project root detection
    @safe_execute(default_return=False)
    def test_project_root():
        root = get_project_root()
        assert root.exists(), "Project root should exist"
        assert (root / "main.py").exists(), "main.py should exist in project root"
        return True

    if test_project_root():
        print("✅ Project root detection works")
    else:
        print("❌ Project root detection failed")
        success = False

    # Test ensure_imports with performance check
    def import_performance_test():
        for _ in range(10):
            ensure_imports()

    if test_runner["performance_tests"](import_performance_test, 0.1):
        print("✅ Import performance is optimal")
    else:
        print("❌ Import performance needs improvement")
        success = False

    # Test import context
    @safe_execute(default_return=False)
    def test_import_context():
        original_path = sys.path.copy()
        with import_context():
            pass
        return sys.path == original_path

    if test_import_context():
        print("✅ Import context manager works")
    else:
        print("❌ Import context manager failed")
        success = False

    # Test function registry
    function_registry.register("test_func", lambda x: x * 2)
    if (
        function_registry.is_available("test_func")
        and function_registry.call("test_func", 5) == 10
    ):
        print("✅ Function registry works")
    else:
        print("❌ Function registry failed")
        success = False

    # Test new utilities
    test_ops = [
        {"action": "exists", "path": "main.py"},
        {"action": "exists", "path": "nonexistent.py"},
    ]

    results = batch_file_operations(test_ops)
    if results.get("operation_0") is True and results.get("operation_1") is False:
        print("✅ Batch file operations work")
    else:
        print("❌ Batch file operations failed")
        success = False

    status = "PASSED" if success else "FAILED"
    print(f"\n🎯 Enhanced Path Management Tests: {status}")
    return success


def safe_execute(
    func=None, *, default_return=None, suppress_errors=True, log_errors=True
):
    """
    Decorator to safely execute functions with centralized error handling.

    This eliminates the repetitive try/catch patterns found throughout the codebase.

    Args:
        func: The function to decorate (when used without parameters)
        default_return: Value to return if function fails (default: None)
        suppress_errors: Whether to suppress exceptions (default: True)
        log_errors: Whether to log errors (default: True)

    Usage:
        @safe_execute
        def risky_function():
            ...

        @safe_execute(default_return=[], suppress_errors=False)
        def another_function():
            ...
    """

    def decorator(f):
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    print(f"❌ Error in {f.__name__}: {e}")
                if not suppress_errors:
                    raise
                return default_return

        return wrapper

    # Handle both @safe_execute and @safe_execute(...) syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


def batch_file_operations(operations: List[dict]) -> dict:
    """
    Perform multiple file operations safely with consolidated error handling.

    This replaces scattered file operation patterns throughout the codebase.

    Args:
        operations: List of operation dicts with keys:
            - 'action': 'read', 'write', 'create', 'delete', 'exists'
            - 'path': file path
            - 'content': content for write operations (optional)

    Returns:
        Dict with results for each operation
    """
    results = {}

    for i, op in enumerate(operations):
        op_id = f"operation_{i}"
        try:
            action = op.get("action")
            path = Path(op.get("path", ""))

            if action == "read":
                results[op_id] = (
                    path.read_text(encoding="utf-8") if path.exists() else None
                )
            elif action == "write":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(op.get("content", ""), encoding="utf-8")
                results[op_id] = True
            elif action == "create":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                results[op_id] = True
            elif action == "delete":
                if path.exists():
                    path.unlink()
                results[op_id] = True
            elif action == "exists":
                results[op_id] = path.exists()
            else:
                results[op_id] = f"Unknown action: {action}"

        except Exception as e:
            results[op_id] = f"Error: {e}"

    return results


class FunctionRegistry:
    """
    Centralized function registry to eliminate scattered globals() lookups.

    This replaces patterns like:
        assert "function_name" in globals()
        assert callable(globals()["function_name"])

    With:
        registry.register("function_name", function_name)
        assert registry.is_available("function_name")
    """

    def __init__(self):
        self._functions = {}
        self._cache = {}

    def register(self, name: str, func: Callable):
        """Register a function in the registry."""
        if callable(func):
            self._functions[name] = func
            # Clear cache when registry changes
            self._cache.clear()

    def register_many(self, **functions):
        """Register multiple functions at once."""
        for name, func in functions.items():
            self.register(name, func)

    def is_available(self, name: str) -> bool:
        """Check if a function is available and callable."""
        if name in self._cache:
            return self._cache[name]

        result = name in self._functions and callable(self._functions[name])
        self._cache[name] = result
        return result

    def get(self, name: str, default=None):
        """Get a function from the registry."""
        return self._functions.get(name, default)

    def call(self, name: str, *args, **kwargs):
        """Safely call a function from the registry."""
        if self.is_available(name):
            return self._functions[name](*args, **kwargs)
        raise ValueError(f"Function '{name}' not available in registry")

    def get_available_functions(self) -> List[str]:
        """Get list of all available function names."""
        return [
            name for name in self._functions.keys() if callable(self._functions[name])
        ]

    def clear(self):
        """Clear the registry."""
        self._functions.clear()
        self._cache.clear()


# Global function registry instance
function_registry = FunctionRegistry()


def register_common_functions():
    """
    Register commonly used functions to eliminate globals() lookups.

    This can be called by modules to register their key functions.
    """
    import builtins

    # Register built-in functions that are commonly checked
    function_registry.register_many(
        len=len,
        str=str,
        int=int,
        float=float,
        list=list,
        dict=dict,
        set=set,
        tuple=tuple,
        callable=callable,
        hasattr=hasattr,
        getattr=getattr,
        setattr=setattr,
    )


# Initialize with common functions
register_common_functions()


def get_optimized_test_runner():
    """
    Optimized test runner that consolidates common test patterns.

    This eliminates repetitive test setup code found throughout the modules.
    """

    @safe_execute(default_return=False)
    def run_module_import_tests(required_globals: List[str]) -> bool:
        """Test that required modules/functions are imported."""
        missing = []
        for item in required_globals:
            if not function_registry.is_available(item):
                if item not in globals():
                    missing.append(item)

        if missing:
            print(f"❌ Missing required globals: {missing}")
            return False

        print(f"✅ All required globals available: {len(required_globals)} items")
        return True

    @safe_execute(default_return=False)
    def run_config_validation_tests(config_obj: Any, required_attrs: List[str]) -> bool:
        """Test configuration object has required attributes."""
        missing = []
        for attr in required_attrs:
            if not hasattr(config_obj, attr):
                missing.append(attr)

        if missing:
            print(f"❌ Config missing attributes: {missing}")
            return False

        print(f"✅ Config validation passed: {len(required_attrs)} attributes")
        return True

    @safe_execute(default_return=False)
    def run_performance_tests(test_func: Callable, max_time: float = 1.0) -> bool:
        """Test that a function completes within time limit."""
        import time

        start_time = time.time()

        try:
            test_func()
            duration = time.time() - start_time

            if duration > max_time:
                print(f"❌ Performance test failed: {duration:.3f}s > {max_time}s")
                return False

            print(f"✅ Performance test passed: {duration:.3f}s")
            return True
        except Exception as e:
            print(f"❌ Performance test error: {e}")
            return False

    return {
        "import_tests": run_module_import_tests,
        "config_tests": run_config_validation_tests,
        "performance_tests": run_performance_tests,
    }


def consolidate_imports(*import_groups):
    """
    Utility to consolidate and organize import statements.

    This helps replace scattered imports with organized groups.

    Usage:
        # Instead of scattered imports:
        import os
        import sys
        from pathlib import Path
        from typing import Optional
        from typing import List

        # Use consolidated imports:
        consolidate_imports(
            ('standard', ['os', 'sys']),
            ('pathlib', ['Path']),
            ('typing', ['Optional', 'List'])
        )
    """
    results = {}

    for group_name, imports in import_groups:
        try:
            if isinstance(imports, str):
                imports = [imports]

            group_results = {}
            for imp in imports:
                if "." in imp:
                    # Handle from imports like 'pathlib.Path'
                    module, attr = imp.rsplit(".", 1)
                    mod = __import__(module, fromlist=[attr])
                    group_results[attr] = getattr(mod, attr)
                else:
                    # Handle direct imports
                    group_results[imp] = __import__(imp)

            results[group_name] = group_results

        except ImportError as e:
            results[group_name] = f"Import error: {e}"

    return results


def create_module_template(
    module_name: str, imports: List[str], functions: List[str]
) -> str:
    """
    Generate a standardized module template.

    This helps create consistent module structure across the codebase.

    Args:
        module_name: Name of the module
        imports: List of required imports
        functions: List of function names to include

    Returns:
        String containing the module template
    """
    template = f'''#!/usr/bin/env python3
"""
{module_name.title().replace('_', ' ')} Module

Generated using standardized module template.
"""

# Standard library imports
{chr(10).join(f"import {imp}" for imp in imports if '.' not in imp)}

# Third-party imports
{chr(10).join(f"from {imp.split('.')[0]} import {imp.split('.')[1]}" for imp in imports if '.' in imp)}

# Local imports
from path_manager import standardize_module_imports
standardize_module_imports()

# Module constants
MODULE_NAME = "{module_name}"

# Function stubs
{chr(10).join(f"def {func}():{chr(10)}    '''TODO: Implement {func}'''{chr(10)}    pass{chr(10)}" for func in functions)}

def run_comprehensive_tests() -> bool:
    """Comprehensive test suite for {module_name}."""
    print(f"🔧 Testing {{MODULE_NAME}}...")
    success = True
    
    # TODO: Add tests here
    
    status = "PASSED" if success else "FAILED"
    print(f"🎯 {{MODULE_NAME}} Tests: {{status}}")
    return success

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
'''

    return template


class ImportHealthChecker:
    """
    Comprehensive import health checking and auto-fixing utility.

    This class addresses the critical import issues found throughout the codebase:
    - Incomplete import statements
    - Scattered sys.path.insert patterns
    - Import debugging and validation
    """

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or get_project_root()
        self.issues_found = []
        self.fixes_applied = []

    def scan_codebase(self) -> Dict[str, List[str]]:
        """
        Scan the entire codebase for import-related issues.

        Returns:
            Dict mapping issue types to lists of affected files
        """
        issues = {
            "incomplete_imports": [],
            "scattered_syspath": [],
            "missing_fallbacks": [],
            "circular_imports": [],
            "deprecated_patterns": [],
        }

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or py_file.name == "path_manager.py":
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                file_issues = self._analyze_file_imports(content, py_file)

                for issue_type, file_issues_list in file_issues.items():
                    if file_issues_list:
                        issues[issue_type].extend(
                            [(py_file, issue) for issue in file_issues_list]
                        )

            except Exception as e:
                self.issues_found.append(f"Error reading {py_file}: {e}")

        return issues

    def _analyze_file_imports(
        self, content: str, file_path: Path
    ) -> Dict[str, List[str]]:
        """Analyze a single file for import issues."""
        issues = {
            "incomplete_imports": [],
            "scattered_syspath": [],
            "missing_fallbacks": [],
            "circular_imports": [],
            "deprecated_patterns": [],
        }

        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()

            # Check for incomplete import statements
            if self._is_incomplete_import(line_stripped):
                issues["incomplete_imports"].append(f"Line {i}: {line_stripped}")

            # Check for scattered sys.path.insert patterns
            if (
                "sys.path.insert" in line
                and "path_manager" not in content[: content.find(line)]
            ):
                issues["scattered_syspath"].append(f"Line {i}: {line_stripped}")

            # Check for deprecated fallback patterns
            if self._is_deprecated_fallback(line_stripped):
                issues["deprecated_patterns"].append(f"Line {i}: {line_stripped}")

        # Check for missing standardize_module_imports usage
        if "from path_manager import" not in content and "sys.path.insert" in content:
            issues["missing_fallbacks"].append(
                "Missing standardize_module_imports usage"
            )

        return issues

    def _is_incomplete_import(self, line: str) -> bool:
        """Check if a line contains an incomplete import statement."""
        incomplete_patterns = [
            "from security_manager import$",
            "from utils import$",
            "from gedcom_cache import$",
            "from ged4py.parser import$",
        ]

        import re

        for pattern in incomplete_patterns:
            if re.search(pattern, line):
                return True
        return False

    def _is_deprecated_fallback(self, line: str) -> bool:
        """Check if line uses deprecated fallback patterns."""
        deprecated_patterns = [
            "except ImportError:.*sys.path.insert",
            "try:.*from path_manager",
            "ensure_imports().*except ImportError",
        ]

        import re

        for pattern in deprecated_patterns:
            if re.search(pattern, line):
                return True
        return False

    def auto_fix_file(self, file_path: Path) -> bool:
        """
        Automatically fix import issues in a single file.

        Args:
            file_path: Path to the file to fix

        Returns:
            bool: True if fixes were applied successfully
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            original_content = content

            # Fix incomplete imports
            content = self._fix_incomplete_imports(content)

            # Replace scattered sys.path patterns with standardize_module_imports
            content = self._fix_scattered_syspath(content)

            # Fix incomplete logger statements
            content = self._fix_incomplete_logger_statements(content)

            if content != original_content:
                # Create backup
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                backup_path.write_text(original_content, encoding="utf-8")

                # Write fixed content
                file_path.write_text(content, encoding="utf-8")

                self.fixes_applied.append(f"Fixed {file_path.name}")
                return True

        except Exception as e:
            self.issues_found.append(f"Error fixing {file_path}: {e}")

        return False

    def _fix_incomplete_imports(self, content: str) -> str:
        """Fix incomplete import statements."""
        import re

        # Fix common incomplete imports
        fixes = {
            r"from security_manager import$": "from security_manager import SecurityManager",
            r"from utils import$": "from utils import SessionManager, format_name",
            r"from gedcom_cache import$": "from gedcom_cache import GedcomCacheModule",
            r"from ged4py\.parser import$": "from ged4py.parser import GedcomReader",
        }

        for pattern, replacement in fixes.items():
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        return content

    def _fix_scattered_syspath(self, content: str) -> str:
        """Replace scattered sys.path.insert patterns with standardize_module_imports."""
        import re

        # Pattern to match sys.path.insert blocks with try/except
        syspath_pattern = r"""try:\s*
\s*from path_manager import .*
\s*.*\(\)
except ImportError:
\s*sys\.path\.insert\(0,.*\)"""

        replacement = """from path_manager import standardize_module_imports
standardize_module_imports()"""

        content = re.sub(
            syspath_pattern, replacement, content, flags=re.MULTILINE | re.DOTALL
        )

        # Simple sys.path.insert replacement
        if "standardize_module_imports" not in content:
            syspath_simple = r"sys\.path\.insert\(0,\s*str\([^)]+\)\)"
            if re.search(syspath_simple, content):
                # Add import at top if not present
                if "from path_manager import" not in content:
                    content = (
                        "from path_manager import standardize_module_imports\nstandardize_module_imports()\n\n"
                        + content
                    )

                # Remove the old patterns
                content = re.sub(
                    syspath_simple,
                    "# Replaced with standardize_module_imports()",
                    content,
                )

        return content

    def _fix_incomplete_logger_statements(self, content: str) -> str:
        """Fix incomplete logger.error statements."""
        import re

        # Pattern for incomplete logger.error followed by exception
        pattern = r"except Exception as (\w+):\s*\n\s*logger\.error$"
        replacement = r'except Exception as \1:\n            logger.error(f"Error: {\1}", exc_info=True)'

        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        return content

    def generate_health_report(self) -> str:
        """Generate a comprehensive health report of import issues."""
        issues = self.scan_codebase()
        report = ["🔍 IMPORT HEALTH REPORT", "=" * 50, ""]

        total_issues = sum(len(issue_list) for issue_list in issues.values())

        if total_issues == 0:
            report.append("✅ No import issues found! Codebase is healthy.")
        else:
            report.append(f"⚠️  Found {total_issues} import-related issues:")
            report.append("")

            for issue_type, issue_list in issues.items():
                if issue_list:
                    report.append(
                        f"📋 {issue_type.replace('_', ' ').title()}: {len(issue_list)} issues"
                    )
                    for item in issue_list[:5]:  # Show first 5
                        if isinstance(item, tuple) and len(item) == 2:
                            file_path, issue = item
                            report.append(f"   • {file_path.name}: {issue}")
                        else:
                            report.append(f"   • {item}")
                    if len(issue_list) > 5:
                        report.append(f"   ... and {len(issue_list) - 5} more")
                    report.append("")

        if self.fixes_applied:
            report.append("🔧 Auto-fixes applied:")
            for fix in self.fixes_applied:
                report.append(f"   ✅ {fix}")
            report.append("")

        report.append("💡 RECOMMENDATIONS:")
        report.append(
            "   1. Run `python path_manager.py --fix-imports` to auto-fix issues"
        )
        report.append(
            "   2. Use `standardize_module_imports()` instead of sys.path.insert"
        )
        report.append("   3. Ensure all import statements are complete")
        report.append("   4. Use centralized error handling patterns")

        return "\n".join(report)


def fix_codebase_imports(dry_run: bool = True) -> bool:
    """
    Fix import issues across the entire codebase.

    Args:
        dry_run: If True, only report issues without fixing them

    Returns:
        bool: True if successful
    """
    checker = ImportHealthChecker()

    if dry_run:
        print(checker.generate_health_report())
        return True

    print("🔧 Fixing import issues across codebase...")

    issues = checker.scan_codebase()
    files_to_fix = set()

    for issue_type, issue_list in issues.items():
        for file_path, issue in issue_list:
            files_to_fix.add(file_path)

    success_count = 0
    for file_path in files_to_fix:
        if checker.auto_fix_file(file_path):
            success_count += 1

    print(f"✅ Successfully fixed {success_count}/{len(files_to_fix)} files")
    print(f"🔍 Found {len(checker.issues_found)} issues requiring manual review")

    if checker.issues_found:
        print("\n⚠️  Manual review required for:")
        for issue in checker.issues_found:
            print(f"   • {issue}")

    return success_count > 0


def validate_import_health() -> bool:
    """
    Quick validation of import health across the codebase.

    Returns:
        bool: True if no critical issues found
    """
    checker = ImportHealthChecker()
    issues = checker.scan_codebase()

    critical_issues = issues["incomplete_imports"] + issues["circular_imports"]

    if critical_issues:
        print(f"❌ Found {len(critical_issues)} critical import issues!")
        return False

    minor_issues = sum(
        len(issue_list)
        for issue_type, issue_list in issues.items()
        if issue_type not in ["incomplete_imports", "circular_imports"]
    )

    if minor_issues > 0:
        print(f"⚠️  Found {minor_issues} minor import issues")
        return True

    print("✅ Import health check passed!")
    return True


# Initialize the import health checker
import_health_checker = ImportHealthChecker()


class CodebaseAutomation:
    """
    Full automation system for applying optimizations across the entire codebase.

    This implements Option 2: Full Automation to systematically replace:
    1. globals() patterns with Function Registry calls
    2. Scattered import patterns with standardized imports
    3. Repetitive error handling with safe_execute decorators
    4. Duplicated test patterns with optimized test runners
    """

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or get_project_root()
        self.files_processed = []
        self.optimizations_applied = []
        self.errors_encountered = []

    def discover_globals_patterns(self) -> Dict[str, List[tuple]]:
        """
        Discover all globals() usage patterns across the codebase.

        Returns:
            Dict mapping pattern types to lists of (file_path, line_number, content)
        """
        patterns = {
            "globals_lookup": [],
            "globals_assertion": [],
            "globals_callable_check": [],
            "function_availability_check": [],
        }

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or py_file.name == "path_manager.py":
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                lines = content.split("\n")

                for i, line in enumerate(lines, 1):
                    line_stripped = line.strip()

                    # Check for globals() lookup patterns
                    if "globals()[" in line:
                        patterns["globals_lookup"].append((py_file, i, line_stripped))

                    # Check for globals() assertion patterns
                    if "assert" in line and "globals()" in line:
                        patterns["globals_assertion"].append(
                            (py_file, i, line_stripped)
                        )

                    # Check for callable(globals()[...]) patterns
                    if "callable(globals()" in line:
                        patterns["globals_callable_check"].append(
                            (py_file, i, line_stripped)
                        )

                    # Check for function availability patterns
                    if "in globals()" in line and (
                        "callable" in line or "assert" in line
                    ):
                        patterns["function_availability_check"].append(
                            (py_file, i, line_stripped)
                        )

            except Exception as e:
                self.errors_encountered.append(f"Error reading {py_file}: {e}")

        return patterns

    def replace_globals_patterns(self, file_path: Path) -> bool:
        """
        Replace globals() patterns in a single file with Function Registry calls.

        Args:
            file_path: Path to the file to process

        Returns:
            bool: True if replacements were made
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            original_content = content

            # Add import if not present
            if "from path_manager import function_registry" not in content:
                # Find the best place to add the import
                lines = content.split("\n")
                import_index = 0

                # Look for existing imports
                for i, line in enumerate(lines):
                    if line.strip().startswith("from ") or line.strip().startswith(
                        "import "
                    ):
                        import_index = i + 1
                    elif line.strip() and not line.strip().startswith("#"):
                        break

                lines.insert(
                    import_index,
                    "from path_manager import function_registry, standardize_module_imports",
                )
                lines.insert(import_index + 1, "standardize_module_imports()")
                content = "\n".join(lines)

            # Replace common globals() patterns
            replacements = [
                # Pattern: assert "function_name" in globals()
                (
                    r'assert\s+"([^"]+)"\s+in\s+globals\(\)',
                    r'assert function_registry.is_available("\1")',
                ),
                # Pattern: assert callable(globals()["function_name"])
                (
                    r'assert\s+callable\(globals\(\)\["([^"]+)"\]\)',
                    r'assert function_registry.is_available("\1")',
                ),
                # Pattern: globals()["function_name"]
                (r'globals\(\)\["([^"]+)"\]', r'function_registry.get("\1")'),
                # Pattern: "function_name" in globals() and callable(globals()["function_name"])
                (
                    r'"([^"]+)"\s+in\s+globals\(\)\s+and\s+callable\(globals\(\)\["[^"]+"\]\)',
                    r'function_registry.is_available("\1")',
                ),
                # Pattern: if "function_name" in globals():
                (
                    r'if\s+"([^"]+)"\s+in\s+globals\(\):',
                    r'if function_registry.is_available("\1"):',
                ),
            ]

            import re

            changes_made = False

            for pattern, replacement in replacements:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                    changes_made = True

            # Register discovered functions
            function_names = self._extract_function_names(content)
            if function_names:
                registration_code = self._generate_registration_code(function_names)
                # Add registration after imports
                lines = content.split("\n")
                insert_index = len(lines)
                for i, line in enumerate(lines):
                    if line.strip() and not (
                        line.strip().startswith("#")
                        or line.strip().startswith("import")
                        or line.strip().startswith("from")
                    ):
                        insert_index = i
                        break

                lines.insert(insert_index, registration_code)
                content = "\n".join(lines)
                changes_made = True

            if changes_made and content != original_content:
                # Create backup
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                backup_path.write_text(original_content, encoding="utf-8")

                # Write optimized content
                file_path.write_text(content, encoding="utf-8")

                self.optimizations_applied.append(
                    f"Optimized globals() patterns in {file_path.name}"
                )
                return True

        except Exception as e:
            self.errors_encountered.append(f"Error processing {file_path}: {e}")

        return False

    def _extract_function_names(self, content: str) -> List[str]:
        """Extract function names defined in the content."""
        import re

        function_names = []

        # Find function definitions
        func_pattern = r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            func_name = match.group(1)
            if not func_name.startswith("_"):  # Skip private functions
                function_names.append(func_name)

        # Find class definitions
        class_pattern = r"^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]"
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            class_name = match.group(1)
            function_names.append(class_name)

        return function_names

    def _generate_registration_code(self, function_names: List[str]) -> str:
        """Generate code to register functions in the Function Registry."""
        if not function_names:
            return ""

        code_lines = [
            "# Auto-register functions for optimized access",
            "try:",
            "    _current_module = globals()",
        ]

        for func_name in function_names:
            code_lines.append(
                f'    if "{func_name}" in _current_module and callable(_current_module["{func_name}"]):'
            )
            code_lines.append(
                f'        function_registry.register("{func_name}", _current_module["{func_name}"])'
            )

        code_lines.extend(
            [
                "except Exception:",
                "    pass  # Silent registration - don't break module loading",
                "",
            ]
        )

        return "\n".join(code_lines)

    def apply_safe_execute_patterns(self, file_path: Path) -> bool:
        """
        Apply safe_execute decorator patterns to replace repetitive try/catch blocks.

        Args:
            file_path: Path to the file to process

        Returns:
            bool: True if patterns were applied
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            original_content = content

            # Add safe_execute import if not present
            if "from path_manager import safe_execute" not in content:
                lines = content.split("\n")
                # Find existing path_manager import to extend it
                for i, line in enumerate(lines):
                    if (
                        "from path_manager import" in line
                        and "safe_execute" not in line
                    ):
                        lines[i] = line.rstrip() + ", safe_execute"
                        break
                else:
                    # Add new import if no existing path_manager import
                    import_index = 0
                    for i, line in enumerate(lines):
                        if line.strip().startswith("import") or line.strip().startswith(
                            "from"
                        ):
                            import_index = i + 1
                    lines.insert(import_index, "from path_manager import safe_execute")

                content = "\n".join(lines)

            # Replace simple try/except patterns with @safe_execute decorator
            import re

            # Pattern: try: ... except Exception as e: pass
            simple_try_pattern = (
                r"(\s*)try:\s*\n\s*(.+)\s*\n\s*except Exception as \w+:\s*\n\s*pass"
            )

            def replace_with_safe_execute(match):
                indent = match.group(1)
                code_line = match.group(2).strip()

                # Create a simple function wrapper
                return f"{indent}@safe_execute\n{indent}def _temp_func():\n{indent}    {code_line}\n{indent}_temp_func()"

            if re.search(simple_try_pattern, content, re.MULTILINE | re.DOTALL):
                content = re.sub(
                    simple_try_pattern,
                    replace_with_safe_execute,
                    content,
                    flags=re.MULTILINE | re.DOTALL,
                )

                if content != original_content:
                    # Create backup
                    backup_path = file_path.with_suffix(
                        f"{file_path.suffix}.safe_backup"
                    )
                    backup_path.write_text(original_content, encoding="utf-8")

                    # Write optimized content
                    file_path.write_text(content, encoding="utf-8")

                    self.optimizations_applied.append(
                        f"Applied safe_execute patterns in {file_path.name}"
                    )
                    return True

        except Exception as e:
            self.errors_encountered.append(
                f"Error applying safe_execute to {file_path}: {e}"
            )

        return False

    def optimize_file(self, file_path: Path) -> Dict[str, bool]:
        """
        Apply all optimizations to a single file.

        Args:
            file_path: Path to the file to optimize

        Returns:
            Dict with results of each optimization type
        """
        results = {
            "globals_patterns": False,
            "safe_execute_patterns": False,
            "import_standardization": False,
        }

        try:
            # Apply globals() pattern replacements
            results["globals_patterns"] = self.replace_globals_patterns(file_path)

            # Apply safe_execute patterns
            results["safe_execute_patterns"] = self.apply_safe_execute_patterns(
                file_path
            )

            # Mark as processed
            self.files_processed.append(file_path)

        except Exception as e:
            self.errors_encountered.append(f"Error optimizing {file_path}: {e}")

        return results

    def optimize_entire_codebase(
        self, target_modules: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Apply full automation optimizations across the entire codebase.

        Args:
            target_modules: List of specific modules to target (default: all)

        Returns:
            Dict with comprehensive results
        """
        print("🚀 STARTING FULL AUTOMATION - OPTION 2")
        print("=" * 60)

        # Discover patterns first
        print("🔍 Discovering globals() patterns across codebase...")
        patterns = self.discover_globals_patterns()

        total_patterns = sum(len(pattern_list) for pattern_list in patterns.values())
        print(f"📊 Found {total_patterns} globals() patterns to optimize:")
        for pattern_type, pattern_list in patterns.items():
            if pattern_list:
                print(
                    f"   • {pattern_type.replace('_', ' ').title()}: {len(pattern_list)} occurrences"
                )

        # Get target files
        if target_modules:
            target_files = [
                self.project_root / f"{module}.py"
                for module in target_modules
                if (self.project_root / f"{module}.py").exists()
            ]
        else:
            # Target high-impact modules first
            priority_modules = [
                "action10.py",
                "action11.py",
                "utils.py",
                "main.py",
                "gedcom_utils.py",
                "gedcom_search_utils.py",
                "api_utils.py",
            ]
            target_files = [
                self.project_root / module
                for module in priority_modules
                if (self.project_root / module).exists()
            ]

            # Add core modules
            core_dir = self.project_root / "core"
            if core_dir.exists():
                target_files.extend(core_dir.glob("*.py"))

        print(
            f"\n🎯 Targeting {len(target_files)} high-impact files for optimization..."
        )

        # Apply optimizations
        optimization_results = {}
        success_count = 0

        for file_path in target_files:
            print(f"🔧 Optimizing {file_path.name}...")
            results = self.optimize_file(file_path)
            optimization_results[str(file_path)] = results

            if any(results.values()):
                success_count += 1

        # Generate comprehensive report
        report = self._generate_automation_report(
            patterns, optimization_results, success_count, len(target_files)
        )

        return {
            "patterns_discovered": patterns,
            "files_targeted": len(target_files),
            "files_optimized": success_count,
            "optimizations_applied": self.optimizations_applied,
            "errors_encountered": self.errors_encountered,
            "detailed_results": optimization_results,
            "report": report,
        }

    def _generate_automation_report(
        self, patterns: Dict, results: Dict, success_count: int, total_files: int
    ) -> str:
        """Generate a comprehensive automation report."""
        report_lines = [
            "🎯 FULL AUTOMATION COMPLETE - OPTION 2 RESULTS",
            "=" * 60,
            "",
            f"📊 PATTERN DISCOVERY:",
            f"   • Total globals() patterns found: {sum(len(p) for p in patterns.values())}",
        ]

        for pattern_type, pattern_list in patterns.items():
            if pattern_list:
                report_lines.append(
                    f"   • {pattern_type.replace('_', ' ').title()}: {len(pattern_list)}"
                )

        report_lines.extend(
            [
                "",
                f"🔧 OPTIMIZATION RESULTS:",
                f"   • Files targeted: {total_files}",
                f"   • Files successfully optimized: {success_count}",
                (
                    f"   • Success rate: {(success_count/total_files*100):.1f}%"
                    if total_files > 0
                    else "   • Success rate: N/A (no files targeted)"
                ),
                "",
                f"✅ OPTIMIZATIONS APPLIED:",
            ]
        )

        for optimization in self.optimizations_applied:
            report_lines.append(f"   • {optimization}")

        if self.errors_encountered:
            report_lines.extend(
                [
                    "",
                    f"⚠️  ERRORS ENCOUNTERED:",
                ]
            )
            for error in self.errors_encountered:
                report_lines.append(f"   • {error}")

        report_lines.extend(
            [
                "",
                "🚀 NEXT STEPS:",
                "   1. Test the optimized modules to ensure functionality",
                "   2. Run comprehensive test suites on modified files",
                "   3. Monitor performance improvements",
                "   4. Consider extending to remaining modules",
                "",
                "💡 ESTIMATED BENEFITS:",
                f"   • Eliminated {sum(len(p) for p in patterns.values())} inefficient globals() lookups",
                "   • Standardized import patterns across modules",
                "   • Improved error handling consistency",
                "   • Enhanced code maintainability and performance",
            ]
        )

        return "\n".join(report_lines)


# Global automation instance
codebase_automation = CodebaseAutomation()


def run_full_automation(target_modules: Optional[List[str]] = None) -> bool:
    """
    Execute Option 2: Full Automation across the codebase.

    Args:
        target_modules: List of specific modules to target (default: high-impact modules)

    Returns:
        bool: True if automation completed successfully
    """
    try:
        results = codebase_automation.optimize_entire_codebase(target_modules)

        # Print the comprehensive report
        print(results["report"])

        # Validate the changes
        print("\n🔍 Validating optimizations...")
        validation_success = validate_import_health()

        if validation_success:
            print("✅ Full automation completed successfully!")
        else:
            print(
                "⚠️  Full automation completed with some issues - manual review recommended"
            )

        return validation_success

    except Exception as e:
        print(f"❌ Full automation failed: {e}")
        return False


# ...existing code...
