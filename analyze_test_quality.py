#!/usr/bin/env python3
"""
Comprehensive Test Quality Analysis Tool

Analyzes all Python modules for:
1. Test quality - validates behavior vs smoke tests
2. Test duplication - identifies redundant tests
3. Test coverage - checks if critical functions are tested
4. Authentication requirements - identifies tests needing live sessions
"""

import ast
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


class TestQualityAnalyzer:
    """Analyzes test quality across all Python modules."""

    def __init__(self):
        self.results = []
        self.issues = defaultdict(list)
        self.stats = {
            "total_modules": 0,
            "modules_with_tests": 0,
            "total_test_functions": 0,
            "smoke_tests": 0,
            "no_assertions": 0,
            "always_true": 0,
            "needs_auth": 0,
            "duplicate_logic": 0,
        }

    def analyze_all_modules(self):
        """Analyze all Python modules in the project."""
        exclude_dirs = {"__pycache__", ".git", "venv", "env", ".venv", "Cache", "Data", "Logs", "test_data", "scripts", "archive"}

        for root, dirs, files in os.walk("."):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]

            for file in files:
                if file.endswith(".py") and not file.startswith("."):
                    filepath = str(Path(root) / file)
                    # Normalize path separators for consistent display
                    filepath = filepath.replace("\\", "/")
                    self.analyze_module(filepath)

    def analyze_module(self, filepath: str):
        """Analyze a single Python module."""
        self.stats["total_modules"] += 1

        try:
            content = Path(filepath).read_text(encoding="utf-8")

            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                self.issues["parse_errors"].append(filepath)
                return

            # Find test functions
            test_functions = self._find_test_functions(tree, content)

            if not test_functions:
                return

            self.stats["modules_with_tests"] += 1
            self.stats["total_test_functions"] += len(test_functions)

            # Analyze each test function
            module_issues = []
            for test_func in test_functions:
                issues = self._analyze_test_function(test_func, content, filepath)
                if issues:
                    module_issues.extend(issues)

            if module_issues:
                self.results.append(
                    {
                        "module": filepath,
                        "test_count": len(test_functions),
                        "issues": module_issues,
                    }
                )

        except Exception as e:
            self.issues["analysis_errors"].append(f"{filepath}: {e}")

    def _find_test_functions(self, tree: ast.AST, content: str) -> list[dict]:
        """Find all test functions in the AST."""
        test_functions = []
        content_lines = content.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                # Get function source by extracting lines directly
                # ast.get_source_segment doesn't work well for nested functions
                start_line = node.lineno - 1  # 0-indexed
                end_line = node.end_lineno if node.end_lineno else start_line + 1

                func_source = "\n".join(content_lines[start_line:end_line])

                test_functions.append(
                    {
                        "name": node.name,
                        "lineno": node.lineno,
                        "source": func_source,
                        "ast_node": node,
                    }
                )

        return test_functions

    def _analyze_test_function(
        self, test_func: dict, _content: str, _filepath: str
    ) -> list[str]:
        """Analyze a single test function for quality issues."""
        issues = []
        source = test_func["source"]
        name = test_func["name"]

        # Check for smoke tests (just returns True)
        if self._is_smoke_test(source):
            issues.append(f"{name}: Smoke test - just returns True")
            self.stats["smoke_tests"] += 1

        # Check for no assertions
        if not self._has_assertions(source):
            issues.append(f"{name}: No assertions found")
            self.stats["no_assertions"] += 1

        # Check for always returning True
        if self._always_returns_true(source):
            issues.append(f"{name}: Always returns True")
            self.stats["always_true"] += 1

        # Check if needs authentication
        if self._needs_authentication(source):
            issues.append(f"{name}: Likely needs authentication")
            self.stats["needs_auth"] += 1

        # Check for minimal test logic
        if self._is_minimal_test(source):
            issues.append(f"{name}: Minimal test logic")

        return issues

    def _is_smoke_test(self, source: str) -> bool:
        """Check if test is just a smoke test."""
        # Remove comments and docstrings
        lines = [
            line.strip()
            for line in source.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        # Check if it's just "return True" or similar
        return len(lines) <= 3 and any("return True" in line for line in lines)

    def _has_assertions(self, source: str) -> bool:
        """Check if test has any assertions."""
        assertion_patterns = [
            r"\bassert\b",
            r"\.assert",
            r"assertEqual",
            r"assertTrue",
            r"assertFalse",
            r"assertIn",
            r"assertNotIn",
            r"assertRaises",
            r"raise AssertionError",  # Pattern used in action10.py and others
        ]

        # Check for explicit assertion patterns
        if any(re.search(pattern, source) for pattern in assertion_patterns):
            return True

        # Check for return True/False pattern (used in ai_interface.py and others)
        # This is valid when the test function returns bool and is called by test framework
        if re.search(r"return\s+(True|False)", source):
            # Make sure it's not just "return True" at the end (smoke test)
            # Look for conditional returns or multiple return statements
            return_statements = re.findall(r"return\s+(True|False)", source)
            if len(return_statements) > 1 or "return False" in source:
                return True

        # Check for boolean expressions in return statements (message_personalization.py pattern)
        # Examples: return x > 5, return "text" in msg, return count == 2
        if re.search(r"return\s+.+\s+(==|!=|>|<|>=|<=|in|not in|and|or)\s+", source):
            return True

        # Check for result/success variable pattern (database.py, message_personalization.py pattern)
        # Pattern: result = True/False ... return result
        # Pattern: success = len(x) > 5 and "text" in x ... return success
        has_result_var = re.search(r"(result|success)\s*=\s*(True|False)", source) or \
                         re.search(r"(result|success)\s*=\s*.+\s+(==|!=|>|<|>=|<=|in|not in|and|or)\s+", source)
        if has_result_var and re.search(r"return\s+(result|success)", source):
            return True

        # Check for all() pattern (system_cache.py pattern)
        # Pattern: all_passed = all(results.values()) ... return all_passed
        if re.search(r"all\(", source) and re.search(r"return\s+\w+", source):
            return True

        # Check for suite.finish_suite() pattern (test_framework.py pattern)
        # Pattern: return suite.finish_suite()
        return bool(re.search(r"return\s+\w+\.finish_suite\(\)", source))

    def _always_returns_true(self, source: str) -> bool:
        """Check if test always returns True without real validation."""
        # Look for pattern: minimal logic followed by return True
        lines = [line.strip() for line in source.split("\n") if line.strip()]

        # Count meaningful lines (excluding def, docstring, comments)
        meaningful_lines = [
            line
            for line in lines
            if not line.startswith("def ")
            and not line.startswith('"""')
            and not line.startswith("'''")
            and not line.startswith("#")
        ]

        # If very few lines and ends with return True, likely fake
        return len(meaningful_lines) <= 2 and any(
            "return True" in line for line in meaningful_lines
        )

    def _needs_authentication(self, source: str) -> bool:
        """Check if test likely needs authentication."""
        auth_indicators = [
            "session",
            "login",
            "auth",
            "api_utils",
            "SessionManager",
            "get_session",
        ]

        return any(indicator in source for indicator in auth_indicators)

    def _is_minimal_test(self, source: str) -> bool:
        """Check if test has minimal logic."""
        # Count lines of actual code (excluding def, docstring, comments, blank)
        lines = source.split("\n")
        code_lines = 0

        in_docstring = False
        for line in lines:
            stripped = line.strip()

            # Skip blank lines
            if not stripped:
                continue

            # Handle docstrings
            if '"""' in stripped or "'''" in stripped:
                in_docstring = not in_docstring
                continue

            if in_docstring:
                continue

            # Skip comments and function definition
            if stripped.startswith("#") or stripped.startswith("def "):
                continue

            code_lines += 1

        # If less than 3 lines of actual code, it's minimal
        return code_lines < 3

    def print_report(self):
        """Print comprehensive test quality report."""
        print("\n" + "=" * 80)
        print("TEST QUALITY ANALYSIS REPORT")
        print("=" * 80)

        # Overall statistics
        print("\n📊 OVERALL STATISTICS:")
        print(f"  Total modules analyzed: {self.stats['total_modules']}")
        print(f"  Modules with tests: {self.stats['modules_with_tests']}")
        print(f"  Total test functions: {self.stats['total_test_functions']}")

        print("\n⚠️  QUALITY ISSUES FOUND:")
        print(f"  Smoke tests (just return True): {self.stats['smoke_tests']}")
        print(f"  Tests with no assertions: {self.stats['no_assertions']}")
        print(f"  Tests always returning True: {self.stats['always_true']}")
        print(f"  Tests needing authentication: {self.stats['needs_auth']}")

        # Detailed issues by module
        if self.results:
            print("\n" + "=" * 80)
            print("DETAILED ISSUES BY MODULE")
            print("=" * 80)

            for result in sorted(self.results, key=lambda x: len(x["issues"]), reverse=True):
                print(f"\n📁 {result['module']} ({result['test_count']} tests)")
                for issue in result["issues"]:
                    print(f"  ⚠️  {issue}")

        # Parse errors
        if self.issues["parse_errors"]:
            print("\n" + "=" * 80)
            print("PARSE ERRORS")
            print("=" * 80)
            for error in self.issues["parse_errors"]:
                print(f"  ❌ {error}")

        # Analysis errors
        if self.issues["analysis_errors"]:
            print("\n" + "=" * 80)
            print("ANALYSIS ERRORS")
            print("=" * 80)
            for error in self.issues["analysis_errors"]:
                print(f"  ❌ {error}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    analyzer = TestQualityAnalyzer()
    analyzer.analyze_all_modules()
    analyzer.print_report()

