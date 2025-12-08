#!/usr/bin/env python3
"""Audit helper for enforcing standardized import patterns."""

from __future__ import annotations

import ast
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)

_SKIP_DIR_NAMES = {".git", "Cache", "Data", "Logs", "__pycache__", "node_modules", ".venv"}
_SKIP_FILES = set()


@dataclass(frozen=True)
class SetupModuleIssue:
    """Represents a discovered issue with setup_module usage."""

    path: str
    line: int
    issue_type: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} [{self.issue_type}] {self.detail}"


def _should_skip(path: Path) -> bool:
    """Determine whether a path should be skipped during scanning."""
    parts = set(path.parts)
    if parts.intersection(_SKIP_DIR_NAMES):
        return True
    return path.name in _SKIP_FILES


def _resolve_call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        value = _resolve_call_name(func.value)
        if value:
            return f"{value}.{func.attr}"
        return func.attr
    return None


def _is_globals_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "globals"
        and not node.args
        and not node.keywords
    )


def _is_dunder_name(node: ast.AST, expected: str) -> bool:
    return isinstance(node, ast.Name) and node.id == expected


def _collect_setup_calls(tree: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            call_name = _resolve_call_name(node.func)
            if call_name and call_name.endswith("setup_module"):
                calls.append(node)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return calls


def _parse_python_file(path: Path) -> ast.AST | None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Skipping %s due to read error: %s", path, exc)
        return None

    try:
        return ast.parse(source)
    except SyntaxError as exc:
        logger.warning("Skipping %s due to parse error: %s", path, exc)
        return None


def _relative_repo_path(path: Path, base_path: Path) -> str:
    try:
        return str(path.relative_to(base_path))
    except ValueError:
        return str(path)


def _record_duplicate_call_issue(
    rel_path: str,
    setup_calls: list[ast.Call],
) -> list[SetupModuleIssue]:
    if len(setup_calls) <= 1:
        return []
    return [
        SetupModuleIssue(
            path=rel_path,
            line=setup_calls[1].lineno,
            issue_type="duplicate_call",
            detail="Multiple setup_module invocations detected",
        )
    ]


def _validate_setup_call(rel_path: str, call: ast.Call) -> list[SetupModuleIssue]:
    issues: list[SetupModuleIssue] = []

    if len(call.args) < 2:
        issues.append(
            SetupModuleIssue(
                path=rel_path,
                line=call.lineno,
                issue_type="invalid_args",
                detail="setup_module expects globals() and __name__",
            )
        )
        return issues

    if not _is_globals_call(call.args[0]):
        issues.append(
            SetupModuleIssue(
                path=rel_path,
                line=call.lineno,
                issue_type="invalid_globals",
                detail="First argument should be globals()",
            )
        )

    if not _is_dunder_name(call.args[1], "__name__"):
        issues.append(
            SetupModuleIssue(
                path=rel_path,
                line=call.lineno,
                issue_type="invalid_module_name",
                detail="Second argument should be __name__",
            )
        )

    return issues


def _scan_file_for_issues(path: Path, base_path: Path) -> list[SetupModuleIssue]:
    tree = _parse_python_file(path)
    if tree is None:
        return []

    setup_calls = _collect_setup_calls(tree)
    if not setup_calls:
        return []

    rel_path = _relative_repo_path(path, base_path)
    issues = _record_duplicate_call_issue(rel_path, setup_calls)
    for call in setup_calls:
        issues.extend(_validate_setup_call(rel_path, call))
    return issues


def scan_for_setup_module_issues(root: Path | None = None) -> list[SetupModuleIssue]:
    """Scan the repository for duplicate or malformed setup_module calls."""
    base_path = (root or Path.cwd()).resolve()
    issues: list[SetupModuleIssue] = []

    for path in base_path.rglob("*.py"):
        if _should_skip(path):
            continue
        issues.extend(_scan_file_for_issues(path, base_path))

    return issues


# === TESTING SUPPORT ===


def _create_temp_file(directory: Path, name: str, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / name
    file_path.write_text(content, encoding="utf-8")
    return file_path


def import_audit_module_tests() -> bool:
    suite = TestSuite("Import Standardization Audit", "import_audit.py")

    def test_detects_duplicates() -> None:
        with TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            _create_temp_file(
                tmp_dir,
                "dup.py",
                "from standard_imports import setup_module\n"
                "logger = setup_module(globals(), __name__)\n"
                "logger = setup_module(globals(), __name__)\n",
            )
            issues = scan_for_setup_module_issues(tmp_dir)
            assert any(issue.issue_type == "duplicate_call" for issue in issues)

    def test_detects_invalid_arguments() -> None:
        with TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            _create_temp_file(
                tmp_dir,
                "bad_args.py",
                "from standard_imports import setup_module\nlogger = setup_module(globals(), __file__)\n",
            )
            issues = scan_for_setup_module_issues(tmp_dir)
            assert any(issue.issue_type == "invalid_module_name" for issue in issues)

    def test_repository_is_clean() -> None:
        issues = scan_for_setup_module_issues(Path.cwd())
        assert not issues, f"Unexpected setup_module issues: {issues}"

    suite.run_test(
        "Duplicate detection",
        test_detects_duplicates,
        "Scanner flags multiple setup_module calls",
        "Write temporary file with duplicate calls and scan",
        "Duplicate usage should be reported",
    )
    suite.run_test(
        "Argument validation",
        test_detects_invalid_arguments,
        "Scanner validates globals()/__name__ arguments",
        "Write file with incorrect arguments",
        "Invalid signatures should be reported",
    )
    suite.run_test(
        "Repository is compliant",
        test_repository_is_clean,
        "Real repository has no setup_module issues",
        "Scan current working directory",
        "Repository should be free of duplicates",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(import_audit_module_tests)


if __name__ == "__main__":
    tests_passed = run_comprehensive_tests()
    repo_issues = scan_for_setup_module_issues()
    if repo_issues:
        for issue in repo_issues:
            print(f"❌ {issue}")
        sys.exit(1)
    sys.exit(0 if tests_passed else 1)
