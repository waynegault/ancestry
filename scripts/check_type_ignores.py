#!/usr/bin/env python3
"""Repository guard that fails if any `type: ignore` directives are present."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_framework import TestSuite, create_standard_test_runner

EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "node_modules", "env", "venv"}
TYPE_IGNORE_DIRECTIVE = "type: " + "ignore"


def iter_python_files(root: Path) -> Iterable[Path]:
    """Yield repository Python files excluding virtualenv and cache folders."""
    for py_file in root.rglob("*.py"):
        if any(part in EXCLUDED_PARTS for part in py_file.parts):
            continue
        yield py_file


def scan_file_for_type_ignores(py_file: Path) -> list[tuple[int, str]]:
    """Return every line/ snippet in *py_file* that uses `type: ignore`."""
    matches: list[tuple[int, str]] = []
    contents = py_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx, line in enumerate(contents, start=1):
        comment_index = line.find("#")
        if comment_index == -1:
            continue
        if "type: ignore" in line[comment_index:].lower():
            matches.append((idx, line.strip() or "<empty>"))
    return matches


def main(repo_root: Path | None = None) -> int:
    """Scan *repo_root* and emit a non-zero exit code if violations exist."""
    if repo_root is None:
        repo_root = REPO_ROOT
    violations: list[tuple[Path, int, str]] = []

    for py_file in iter_python_files(repo_root):
        for line_no, snippet in scan_file_for_type_ignores(py_file):
            violations.append((py_file.relative_to(repo_root), line_no, snippet))

    if violations:
        print("❌ Found disallowed `type: ignore` directives:")
        for rel_path, line_no, snippet in violations:
            print(f"  {rel_path}:{line_no}: {snippet}")
        print(f"Total occurrences: {len(violations)}")
        return 1

    print("✅ No `type: ignore` directives detected.")
    return 0


def _create_file(root: Path, relative: str, contents: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def _run_main_with_structure(structure: dict[str, str]) -> tuple[int, str]:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        for rel_path, contents in structure.items():
            _create_file(root, rel_path, contents)
        output = StringIO()
        with redirect_stdout(output):
            exit_code = main(root)
    return exit_code, output.getvalue()


def _test_iter_python_files_skips_excluded_folders() -> bool:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        included = _create_file(root, "pkg/keep.py", "print('ok')")
        excluded = root / "__pycache__"
        excluded.mkdir()
        _create_file(excluded, "skip.py", "print('skip')")
        _create_file(root / "node_modules", "ignored.py", "print('no')")
        discovered = {path.relative_to(root) for path in iter_python_files(root)}
    assert discovered == {included.relative_to(root)}
    return True


def _test_scan_file_for_type_ignores_detects_comment_usage() -> bool:
    with TemporaryDirectory() as tmp_dir:
        directive_line = f"value = 1  # {TYPE_IGNORE_DIRECTIVE.upper()}[attr-defined]\\n"
        path = _create_file(
            Path(tmp_dir),
            "sample.py",
            directive_line + "print('type: ignore in string')\\n",
        )
        matches = scan_file_for_type_ignores(path)
    assert matches and matches[0][0] == 1
    assert "type: ignore" in matches[0][1].lower()
    return True


def _test_main_reports_detected_violations() -> bool:
    exit_code, output = _run_main_with_structure({"bad.py": f"value = 1  # {TYPE_IGNORE_DIRECTIVE}"})
    assert exit_code == 1
    assert "bad.py:1" in output
    assert "Total occurrences: 1" in output
    return True


def _test_main_succeeds_without_violations() -> bool:
    exit_code, output = _run_main_with_structure({"good.py": "value = 1\n"})
    assert exit_code == 0
    assert "No `type: ignore` directives detected" in output
    return True


def module_tests() -> bool:
    suite = TestSuite("scripts.check_type_ignores", "scripts/check_type_ignores.py")
    suite.run_test(
        "Iter skips excluded parts",
        _test_iter_python_files_skips_excluded_folders,
        "Ensures iter_python_files ignores cache and dependency folders.",
    )
    suite.run_test(
        "Scan detects comment",
        _test_scan_file_for_type_ignores_detects_comment_usage,
        "Ensures scan_file_for_type_ignores captures directives.",
    )
    suite.run_test(
        "Main reports violations",
        _test_main_reports_detected_violations,
        "Ensures non-zero exit and report when violations exist.",
    )
    suite.run_test(
        "Main succeeds clean",
        _test_main_succeeds_without_violations,
        "Ensures zero exit and success message when repository is clean.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


def _should_run_module_tests() -> bool:
    return os.environ.get("RUN_MODULE_TESTS") == "1"


if __name__ == "__main__":
    if _should_run_module_tests():
        success = run_comprehensive_tests()
        raise SystemExit(0 if success else 1)
    raise SystemExit(main())
