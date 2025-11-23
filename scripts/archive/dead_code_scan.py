#!/usr/bin/env python3
from __future__ import annotations

"""Lightweight dead-code scanner for the Ancestry project.

This script performs a conservative static pass over all Python files in the
repository and flags function definitions that appear only once by name across
the entire tree (i.e., the definition itself).

It is intentionally simple and conservative:
 - Only top-level functions and methods directly under classes are considered.
 - Dunder functions and common test helpers (``test_*``, ``*_tests``) are
     ignored.
 - A candidate is reported only when its name occurs exactly once as a
     standalone word across all source files (definition only).

Results are written to ``Cache/dead_code_candidates.json`` with entries of the
form::

        {
                "name": "_some_helper",
                "file": "action7_inbox.py",
                "lineno": 123,
                "kind": "module_function" | "method",
                "owner": "ClassName" | null,
                "total_refs": 1
        }

This file is intended as input for human review before any deletions.
"""

import ast
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS: set[str] = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    "Cache",
    "Data",
    "Logs",
}

SKIP_FILES: set[str] = {
    "scripts/dead_code_scan.py",
}


@dataclass
class SymbolInfo:
    name: str
    file: str  # path relative to ROOT, POSIX-style
    lineno: int
    kind: str  # "module_function" | "method" | "class"
    owner: str | None


def iter_py_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # prune directories in-place for efficiency
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = Path(dirpath) / name
            yield path


def load_file_texts(py_files: Iterable[Path]) -> dict[Path, str]:
    texts: dict[Path, str] = {}
    for path in py_files:
        try:
            texts[path] = path.read_text(encoding="utf-8")
        except OSError:
            # Skip unreadable files but keep going.
            continue
    return texts


def _parse_ast_safely(path: Path, text: str) -> ast.Module | None:
    try:
        return ast.parse(text, filename=str(path))
    except SyntaxError:
        # Skip files with syntax errors (e.g., incomplete work in progress).
        return None


def _collect_symbols_from_tree(tree: ast.Module, rel_path: str) -> list[SymbolInfo]:
    defs: list[SymbolInfo] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defs.append(
                SymbolInfo(
                    name=node.name,
                    file=rel_path,
                    lineno=node.lineno,
                    kind="module_function",
                    owner=None,
                )
            )
        elif isinstance(node, ast.ClassDef):
            defs.append(
                SymbolInfo(
                    name=node.name,
                    file=rel_path,
                    lineno=node.lineno,
                    kind="class",
                    owner=None,
                )
            )

            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef):
                    defs.append(
                        SymbolInfo(
                            name=body_item.name,
                            file=rel_path,
                            lineno=body_item.lineno,
                            kind="method",
                            owner=node.name,
                        )
                    )

    return defs


def collect_symbol_defs(file_texts: dict[Path, str]) -> list[SymbolInfo]:
    defs: list[SymbolInfo] = []

    for path, text in file_texts.items():
        rel = path.relative_to(ROOT).as_posix()
        if rel in SKIP_FILES:
            continue

        tree = _parse_ast_safely(path, text)
        if tree is None:
            continue

        defs.extend(_collect_symbols_from_tree(tree, rel))

    return defs


def compute_usage_counts(symbols: list[SymbolInfo], file_texts: dict[Path, str]) -> dict[str, int]:
    # Precompute a big string with file contents for simple regex searching.
    all_text = "\n".join(file_texts.values())

    counts: dict[str, int] = {}

    for sym in symbols:
        # We use a conservative whole-word regex to avoid substring matches.
        pattern = rf"\b{re.escape(sym.name)}\b"
        matches = re.findall(pattern, all_text)
        counts[sym.name] = len(matches)

    return counts


def _should_skip_name(name: str) -> bool:
    if name.startswith("__") and name.endswith("__"):
        return True
    if name.startswith("test_"):
        return True
    return name.endswith("_tests")


def find_dead_code_candidates() -> list[dict[str, Any]]:
    py_files = list(iter_py_files(ROOT))
    file_texts = load_file_texts(py_files)
    symbols = collect_symbol_defs(file_texts)

    usage_counts = compute_usage_counts(symbols, file_texts)

    candidates: list[dict[str, Any]] = []
    for sym in symbols:
        if _should_skip_name(sym.name):
            continue

        total_refs = usage_counts.get(sym.name, 0)
        if total_refs == 1:
            candidates.append(
                {
                    "name": sym.name,
                    "file": sym.file,
                    "lineno": sym.lineno,
                    "kind": sym.kind,
                    "owner": sym.owner,
                    "total_refs": total_refs,
                }
            )

    return candidates


def main() -> None:
    candidates = find_dead_code_candidates()

    out_path = ROOT / "Cache" / "dead_code_candidates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    analyzed_files = len(list(iter_py_files(ROOT)))
    print(f"[dead_code_scan] Analyzed {analyzed_files} files, found {len(candidates)} candidate functions.")
    print(f"[dead_code_scan] Results written to: {out_path}")


if __name__ == "__main__":
    main()
