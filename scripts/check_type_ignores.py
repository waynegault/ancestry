#!/usr/bin/env python3
"""Repository guard that fails if any `type: ignore` directives are present."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "node_modules", "env", "venv"}


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


def main() -> int:
    """Scan the repository and emit a non-zero exit code if violations exist."""
    repo_root = Path(__file__).resolve().parents[1]
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


if __name__ == "__main__":
    raise SystemExit(main())
