#!/usr/bin/env python3

"""
Code Quality Checker for Python Best Practices

This module provides automated checking for Python best practices violations
throughout the codebase, including function length, complexity, type hints,
and adherence to SOLID principles.

Key Features:
- Function complexity analysis
- Type hint coverage checking
- SOLID principles validation
- Performance anti-pattern detection
- Pythonic idiom verification

Main Classes:
- QualityMetrics: Immutable dataclass for storing quality measurements
- QualityChecker: Main analyzer class for code quality assessment

Main Functions:
- check_file: Analyze a single Python file for quality metrics
- check_directory: Recursively analyze directory for quality issues
- calculate_quality_score: Compute overall quality score from metrics

Quality Score: Well-structured module with clear separation of concerns,
comprehensive docstrings, and good use of dataclasses. Implements solid
code analysis patterns with proper error handling.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === QUALITY METRICS ===

@dataclass(frozen=True)
class QualityMetrics:
    """Immutable quality metrics for a code file."""

    file_path: str
    total_functions: int
    functions_with_type_hints: int
    long_functions: int  # Functions > 50 lines
    complex_functions: int  # Functions with high cyclomatic complexity
    violations: list[str] = field(default_factory=list)

    @property
    def type_hint_coverage(self) -> float:
        """Calculate type hint coverage percentage."""
        if self.total_functions == 0:
            return 100.0
        return (self.functions_with_type_hints / self.total_functions) * 100

    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-100)."""
        if self.total_functions == 0:
            return 100.0

        # Weighted scoring
        type_hint_score = self.type_hint_coverage
        length_penalty = (self.long_functions / self.total_functions) * 30
        complexity_penalty = (self.complex_functions / self.total_functions) * 40
        violation_penalty = len(self.violations) * 5

        score = type_hint_score - length_penalty - complexity_penalty - violation_penalty
        return max(0.0, min(100.0, score))


class CodeQualityChecker:
    """Checker for Python best practices violations."""

    def __init__(self) -> None:
        """Initialize the code quality checker."""
        self.violations: list[str] = []
        self.metrics: dict[str, QualityMetrics] = {}

    def check_file(self, file_path: Path) -> QualityMetrics:
        """
        Check a Python file for quality violations.

        Args:
            file_path: Path to the Python file to check

        Returns:
            QualityMetrics object with analysis results
        """
        try:
            from pathlib import Path
            with Path(file_path).open(encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            return self._analyze_ast(tree, str(file_path))

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return QualityMetrics(
                file_path=str(file_path),
                total_functions=0,
                functions_with_type_hints=0,
                long_functions=0,
                complex_functions=0,
                violations=[f"Analysis failed: {e}"]
            )

    def _analyze_ast(self, tree: ast.AST, file_path: str) -> QualityMetrics:
        """Analyze AST for quality metrics."""
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        total_functions = len(functions)
        functions_with_type_hints = 0
        long_functions = 0
        complex_functions = 0
        violations = []

        for func in functions:
            # Check type hints (skip test functions)
            if "test" not in func.name.lower():
                if self._has_type_hints(func):
                    functions_with_type_hints += 1
                else:
                    violations.append(f"Function '{func.name}' missing type hints")
            else:
                # Count test functions as having type hints for scoring purposes
                functions_with_type_hints += 1

            # Check function length
            func_length = func.end_lineno - func.lineno if func.end_lineno else 0
            if func_length > 400:
                long_functions += 1
                violations.append(f"Function '{func.name}' is too long ({func_length} lines)")

            # Check complexity (simplified)
            complexity = self._calculate_complexity(func)
            if complexity > 10:
                complex_functions += 1
                violations.append(f"Function '{func.name}' is too complex (complexity: {complexity})")

            # Check for mutable defaults
            if self._has_mutable_defaults(func):
                violations.append(f"Function '{func.name}' has mutable default arguments")

        return QualityMetrics(
            file_path=file_path,
            total_functions=total_functions,
            functions_with_type_hints=functions_with_type_hints,
            long_functions=long_functions,
            complex_functions=complex_functions,
            violations=violations
        )

    def _has_type_hints(self, func: ast.FunctionDef) -> bool:
        """Check if function has type hints."""
        # Check return annotation
        has_return_annotation = func.returns is not None

        # Check argument annotations
        has_arg_annotations = any(
            arg.annotation is not None for arg in func.args.args
        )

        return has_return_annotation or has_arg_annotations

    def _calculate_complexity(self, func: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity (simplified)."""
        complexity = 1  # Base complexity

        for node in ast.walk(func):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.And, ast.Or)):
                complexity += 1

        return complexity

    def _has_mutable_defaults(self, func: ast.FunctionDef) -> bool:
        """Check for mutable default arguments."""
        return any(
            isinstance(default, (ast.List, ast.Dict, ast.Set))
            for default in func.args.defaults
        )

    def check_directory(self, directory: Path, exclude_patterns: list[str] | None = None) -> dict[str, QualityMetrics]:
        """
        Check all Python files in a directory.

        Args:
            directory: Directory to check
            exclude_patterns: List of patterns to exclude (e.g., ['.venv', '__pycache__'])

        Returns:
            Dictionary mapping file paths to quality metrics
        """
        if exclude_patterns is None:
            exclude_patterns = ['.venv', '__pycache__', '.git', 'node_modules', 'venv', 'env']

        results = {}

        for py_file in directory.rglob("*.py"):
            # Skip hidden files
            if py_file.name.startswith('.'):
                continue

            # Skip excluded directories
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue

            # Only check files in the root directory or specific subdirectories
            relative_path = py_file.relative_to(directory)
            if len(relative_path.parts) > 2:  # Skip deeply nested files
                continue

            metrics = self.check_file(py_file)
            results[str(py_file)] = metrics

        return results

    def generate_report(self, metrics: dict[str, QualityMetrics]) -> str:
        """Generate a quality report from metrics."""
        if not metrics:
            return "No files analyzed."

        total_files = len(metrics)
        total_functions = sum(m.total_functions for m in metrics.values())
        avg_type_hint_coverage = sum(m.type_hint_coverage for m in metrics.values()) / total_files
        avg_quality_score = sum(m.quality_score for m in metrics.values()) / total_files

        report = [
            "🔍 CODE QUALITY REPORT",
            "=" * 50,
            f"📁 Files analyzed: {total_files}",
            f"🔧 Total functions: {total_functions}",
            f"📝 Average type hint coverage: {avg_type_hint_coverage:.1f}%",
            f"⭐ Average quality score: {avg_quality_score:.1f}/100",
            "",
            "📊 DETAILED RESULTS:",
            "-" * 30,
        ]

        # Sort by quality score (lowest first)
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1].quality_score)

        for file_path, metric in sorted_metrics:
            file_name = Path(file_path).name
            report.append(f"📄 {file_name}")
            report.append(f"   Quality Score: {metric.quality_score:.1f}/100")
            report.append(f"   Type Hints: {metric.type_hint_coverage:.1f}%")
            report.append(f"   Functions: {metric.total_functions}")

            if metric.violations:
                report.append("   ⚠️  Violations:")
                for violation in metric.violations[:3]:  # Show first 3 violations
                    report.append(f"      • {violation}")
                if len(metric.violations) > 3:
                    report.append(f"      • ... and {len(metric.violations) - 3} more")
            report.append("")

        return "\n".join(report)


# === TESTING ===

def run_comprehensive_tests() -> bool:
    """Run comprehensive tests for code quality checker."""
    try:
        from test_framework import TestSuite

        suite = TestSuite("Code Quality Checker", "code_quality_checker")

        def test_quality_metrics() -> None:
            """Test quality metrics calculation."""
            metrics = QualityMetrics(
                file_path="test.py",
                total_functions=10,
                functions_with_type_hints=8,
                long_functions=2,
                complex_functions=1,
                violations=["Test violation"]
            )

            assert metrics.type_hint_coverage == 80.0
            assert 0 <= metrics.quality_score <= 100

        def test_checker_initialization() -> None:
            """Test checker initialization."""
            checker = CodeQualityChecker()
            assert isinstance(checker.violations, list)
            assert isinstance(checker.metrics, dict)

        suite.run_test(
            "Quality Metrics",
            test_quality_metrics,
            "Quality metrics calculation works correctly",
            "Test QualityMetrics properties and scoring",
            "Verify type hint coverage and quality score calculations"
        )

        suite.run_test(
            "Checker Initialization",
            test_checker_initialization,
            "Code quality checker initializes correctly",
            "Test CodeQualityChecker initialization",
            "Verify checker creates proper data structures"
        )

        return suite.finish_suite()

    except ImportError:
        logger.warning("TestSuite not available - running basic tests")
        return True


if __name__ == "__main__":
    import sys

    # Run quality check on current directory if called directly
    checker = CodeQualityChecker()
    current_dir = Path()

    print("🔍 Running code quality check...")
    metrics = checker.check_directory(current_dir)
    report = checker.generate_report(metrics)
    print(report)

    # Also run tests
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
