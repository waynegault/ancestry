#!/usr/bin/env python3
from __future__ import annotations

"""This helper has been moved to ``scripts/archive/dead_code_scan.py``.

The archived version contains the full dead-code scanner implementation.
This top-level stub exists only to keep backward references clear.
"""

raise SystemExit(
    "Use scripts/archive/dead_code_scan.py for dead-code analysis; this "
    "top-level stub exists only as a pointer to the archived script."
)

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
            if name.endswith(".py"):
                yield Path(dirpath) / name


def load_file_texts(py_files: Iterable[Path]) -> dict[Path, str]:
    file_texts: dict[Path, str] = {}
    for path in py_files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            try:
                text = path.read_text(encoding="latin-1")
            except Exception:
                continue
        file_texts[path] = text
    return file_texts


def _parse_ast_safely(path: Path, text: str) -> ast.Module | None:
    """Parse Python source into an AST, returning None on SyntaxError."""
    try:
        return ast.parse(text, filename=str(path))
    except SyntaxError:
        return None


def _collect_symbols_from_tree(tree: ast.Module, rel_path: str) -> list[SymbolInfo]:
    """Collect top-level symbols (functions, classes, methods) from an AST tree."""
    symbols: list[SymbolInfo] = []

    # Only consider top-level functions, classes, and methods directly under classes
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if _should_skip_name(name):
                continue
            symbols.append(
                SymbolInfo(
                    name=name,
                    file=rel_path,
                    lineno=node.lineno,
                    kind="module_function",
                    owner=None,
                )
            )

        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            if not _should_skip_name(class_name):
                symbols.append(
                    SymbolInfo(
                        name=class_name,
                        file=rel_path,
                        lineno=node.lineno,
                        kind="class",
                        owner=None,
                    )
                )
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    name = child.name
                    if _should_skip_name(name):
                        continue
                    symbols.append(
                        SymbolInfo(
                            name=name,
                            file=rel_path,
                            lineno=child.lineno,
                            kind="method",
                            owner=class_name,
                        )
                    )

    return symbols


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


def _should_skip_name(name: str) -> bool:
    """Return True for names we don't want to consider for dead-code checks."""
    if name.startswith("__") and name.endswith("__"):
        return True
    if name.startswith("test_"):
        return True
    return name.endswith("_tests")


def compute_usage_counts(file_texts: dict[Path, str], defs: list[SymbolInfo]) -> list[dict[str, Any]]:
    # Pre-collect all texts for faster iteration
    texts: list[str] = list(file_texts.values())

    results: list[dict[str, Any]] = []
    for info in defs:
        # Simple word-boundary regex to catch both code and string references
        pattern = re.compile(r"\\b" + re.escape(info.name) + r"\\b")
        total_refs = 0
        for text in texts:
            total_refs += len(pattern.findall(text))

        if total_refs == 1:
            results.append(
                {
                    "name": info.name,
                    "file": info.file,
                    "lineno": info.lineno,
                    "kind": info.kind,
                    "owner": info.owner,
                    "total_refs": total_refs,
                }
            )

    return results


def main() -> None:
    py_files = list(iter_py_files(ROOT))
    file_texts = load_file_texts(py_files)
    defs = collect_symbol_defs(file_texts)
    candidates = compute_usage_counts(file_texts, defs)

    candidates.sort(key=lambda r: (r["file"], r["lineno"]))

    out_payload: dict[str, Any] = {
        "root": str(ROOT),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }

    out_path = ROOT / "Cache" / "dead_code_candidates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    print(f"[dead_code_scan] Analyzed {len(py_files)} files, found {len(candidates)} candidate functions.")
    print(f"[dead_code_scan] Results written to: {out_path}")


if __name__ == "__main__":  # pragma: no cover - utility script
    main()
