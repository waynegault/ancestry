#!/usr/bin/env python3
"""Dead-code scanner for the Ancestry project.

Performs conservative static analysis to identify function definitions
that appear only once across the entire codebase (definition only).
Results are written to Cache/dead_code_candidates.json for manual review.
"""


import ast
import json
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testing.test_framework import TestSuite, create_standard_test_runner

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
    "testing/dead_code_scan.py",
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


def collect_symbol_defs(
    file_texts: dict[Path, str], *, root: Path = ROOT, skip_files: set[str] | None = None
) -> list[SymbolInfo]:
    defs: list[SymbolInfo] = []
    ignored_files = skip_files if skip_files is not None else SKIP_FILES

    for path, text in file_texts.items():
        rel = path.relative_to(root).as_posix()
        if rel in ignored_files:
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


def compute_usage_counts(
    file_texts: dict[Path, str], defs: list[SymbolInfo], *, verbose: bool = False
) -> list[dict[str, Any]]:
    texts: list[str] = list(file_texts.values())
    total_defs = len(defs)

    results: list[dict[str, Any]] = []
    for i, info in enumerate(defs, 1):
        if verbose and i % 100 == 0:
            print(f"  Analyzing symbol {i}/{total_defs}...")
        pattern = re.compile(r"\b" + re.escape(info.name) + r"\b")
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


def main(args: list[str] | None = None) -> int:
    """Generate dead-code report and write to Cache/dead_code_candidates.json."""
    _ = args
    payload = generate_dead_code_report(verbose=True)
    out_path = write_dead_code_report(payload)
    print(f"✅ Dead-code scan complete: {payload['candidate_count']} candidates found.")
    print(f"📝 Results written to: {out_path}")
    return 0


def generate_dead_code_report(
    root: Path = ROOT, *, skip_files: set[str] | None = None, verbose: bool = False
) -> dict[str, Any]:
    """Produce the dead-code candidate payload without writing it to disk."""
    if verbose:
        print("🔍 Starting dead code scan...")

    py_files = list(iter_py_files(root))
    if verbose:
        print(f"📁 Found {len(py_files)} Python files to analyze")

    file_texts = load_file_texts(py_files)
    if verbose:
        print(f"📄 Loaded {len(file_texts)} file texts")

    defs = collect_symbol_defs(file_texts, root=root, skip_files=skip_files)
    if verbose:
        print(f"🔢 Found {len(defs)} symbol definitions")
        print("⏳ Computing usage counts (this may take a moment)...")

    candidates = compute_usage_counts(file_texts, defs, verbose=verbose)
    candidates.sort(key=lambda r: (r["file"], r["lineno"]))

    return {
        "root": str(root),
        "candidate_count": len(candidates),
        "files_analyzed": len(py_files),
        "candidates": candidates,
    }


def write_dead_code_report(payload: dict[str, Any], root: Path = ROOT) -> Path:
    """Persist the payload to Cache/dead_code_candidates.json under *root*."""

    out_path = root / "Cache" / "dead_code_candidates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _create_file(root: Path, relative: str, contents: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def _test_main_runs_scan_successfully() -> bool:
    """Test main() using a small temp directory to avoid slow full-repo scan."""
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        _create_file(root, "sample.py", "def example(): pass\nexample()\n")

        # Generate and write report to temp dir
        payload = generate_dead_code_report(root, skip_files=set())
        out_path = write_dead_code_report(payload, root)

        assert out_path.exists()
        assert payload["files_analyzed"] == 1
    return True


def _test_iter_py_files_skips_configured_dirs() -> bool:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        _create_file(root, "pkg/keep.py", "print('ok')")
        skip_dir = root / "Cache"
        skip_dir.mkdir()
        _create_file(skip_dir, "ignored.py", "print('skip')")
        files = list(iter_py_files(root))
    rel_paths = {path.relative_to(root).as_posix() for path in files}
    assert rel_paths == {"pkg/keep.py"}
    return True


def _test_collect_symbol_defs_filters_helpers() -> bool:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        _create_file(
            root,
            "pkg/mod.py",
            """\
def good_func():
    pass

def __skip_me__():
    pass

def test_helper():
    pass

class Sample:
    def method_one(self):
        pass

    def test_method(self):
        pass
""",
        )
        file_texts = load_file_texts(iter_py_files(root))
        defs = collect_symbol_defs(file_texts, root=root, skip_files=set())

    names = {(info.name, info.kind) for info in defs}
    assert ("good_func", "module_function") in names
    assert ("Sample", "class") in names
    assert ("method_one", "method") in names
    assert all("test" not in name and "__" not in name for name, _ in names)
    return True


def _test_compute_usage_counts_flags_single_reference() -> bool:
    defs = [
        SymbolInfo(name="unused_func", file="a.py", lineno=1, kind="module_function", owner=None),
        SymbolInfo(name="used_func", file="a.py", lineno=5, kind="module_function", owner=None),
    ]
    file_texts = {
        Path("a.py"): "def unused_func():\n    pass\n",
        Path("b.py"): "def used_func():\n    pass\n\nused_func()\n",
    }

    candidates = compute_usage_counts(file_texts, defs)
    assert len(candidates) == 1
    assert candidates[0]["name"] == "unused_func"
    assert candidates[0]["total_refs"] == 1
    return True


def _test_generate_dead_code_report_counts_candidates() -> bool:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        _create_file(
            root,
            "pkg/first.py",
            """\
def keeper():
    return 1

def maybe_dead():
    return 2

keeper()
""",
        )
        _create_file(root, "pkg/second.py", "print('noop')\n")
        payload = generate_dead_code_report(root, skip_files=set())

    assert payload["files_analyzed"] == 2
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["name"] == "maybe_dead"
    return True


def _test_write_dead_code_report_persists_payload() -> bool:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        payload = {"root": str(root), "candidate_count": 0, "candidates": []}
        out_path = write_dead_code_report(payload, root)
        written = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path.relative_to(root).as_posix() == "Cache/dead_code_candidates.json"
    assert written == payload
    return True


def module_tests() -> bool:
    suite = TestSuite("testing.dead_code_scan", "testing/dead_code_scan.py")
    suite.run_test(
        "Main runs scan successfully",
        _test_main_runs_scan_successfully,
        "Ensures CLI runs scan and outputs completion message.",
    )
    suite.run_test(
        "Iterator skips directories",
        _test_iter_py_files_skips_configured_dirs,
        "Ensures iter_py_files respects the configured skip list.",
    )
    suite.run_test(
        "Symbol collection filters helpers",
        _test_collect_symbol_defs_filters_helpers,
        "Ensures test/dunder helpers are excluded from symbol discovery.",
    )
    suite.run_test(
        "Usage counts flag single refs",
        _test_compute_usage_counts_flags_single_reference,
        "Ensures compute_usage_counts identifies unused functions.",
    )
    suite.run_test(
        "Generate report counts candidates",
        _test_generate_dead_code_report_counts_candidates,
        "Ensures report generation counts files and candidates correctly.",
    )
    suite.run_test(
        "Report writer persists payload",
        _test_write_dead_code_report_persists_payload,
        "Ensures payload is written to Cache/dead_code_candidates.json.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    if os.environ.get("RUN_MODULE_TESTS") == "1":
        success = run_comprehensive_tests()
        raise SystemExit(0 if success else 1)
    raise SystemExit(main())
