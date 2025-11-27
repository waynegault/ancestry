#!/usr/bin/env python3
"""Analyze Python imports across the project to detect circular dependencies.

This script:
1. Scans all Python files in the project
2. Extracts import statements
3. Builds a dependency graph
4. Detects circular import chains
5. Outputs a report with recommendations

Usage:
    python scripts/analyze_imports.py
    python scripts/analyze_imports.py --output graph.json
    python scripts/analyze_imports.py --check  # Exit code 1 if circular imports found
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Directories to skip during analysis
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    "Cache",
    "Logs",
    "Data",
    "test_data",
    "test_examples",
}

# Files to skip
SKIP_FILES = {
    "setup.py",
    "conftest.py",
}


def get_python_files(root: Path) -> list[Path]:
    """Get all Python files in the project, excluding specified directories."""
    python_files: list[Path] = []
    for path in root.rglob("*.py"):
        # Skip if any parent directory is in SKIP_DIRS
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        # Skip specified files
        if path.name in SKIP_FILES:
            continue
        python_files.append(path)
    return sorted(python_files)


def get_module_name(file_path: Path, root: Path) -> str:
    """Convert a file path to a module name."""
    relative = file_path.relative_to(root)
    parts = list(relative.parts)
    # Remove .py extension
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    # Handle __init__.py
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else ""


def extract_imports(file_path: Path) -> list[str]:
    """Extract all import statements from a Python file."""
    imports: list[str] = []
    try:
        with file_path.open(encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    return imports


def is_local_import(import_name: str, local_modules: set[str]) -> bool:
    """Check if an import is a local project import."""
    # Check if it's a direct match
    if import_name in local_modules:
        return True
    # Check if it's a submodule of a local module
    parts = import_name.split(".")
    for i in range(len(parts)):
        prefix = ".".join(parts[: i + 1])
        if prefix in local_modules:
            return True
    return False


def build_dependency_graph(files: list[Path], root: Path) -> tuple[dict[str, list[str]], set[str]]:
    """Build a dependency graph from Python files."""
    # First, collect all local module names
    local_modules: set[str] = set()
    for file_path in files:
        module_name = get_module_name(file_path, root)
        if module_name:
            local_modules.add(module_name)
            # Also add parent packages
            parts = module_name.split(".")
            for i in range(len(parts)):
                local_modules.add(".".join(parts[: i + 1]))

    # Build the dependency graph
    graph: dict[str, list[str]] = defaultdict(list)
    for file_path in files:
        module_name = get_module_name(file_path, root)
        if not module_name:
            continue
        imports = extract_imports(file_path)
        for imp in imports:
            if is_local_import(imp, local_modules):
                # Normalize to base module
                base_import = imp.split(".")[0]
                if base_import != module_name.split(".")[0]:
                    graph[module_name].append(imp)

    return dict(graph), local_modules


def find_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    """Find all cycles in the dependency graph using DFS."""
    cycles: list[list[str]] = []
    visited: set[str] = set()
    rec_stack: set[str] = set()
    path: list[str] = []

    def dfs(node: str) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            # Normalize neighbor to top-level module for cycle detection
            neighbor_base = neighbor.split(".")[0]
            if neighbor_base not in visited:
                dfs(neighbor_base)
            elif neighbor_base in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor_base)
                cycle = [*path[cycle_start:], neighbor_base]
                if cycle not in cycles:
                    cycles.append(cycle)

        path.pop()
        rec_stack.remove(node)

    for node in graph:
        if node not in visited:
            dfs(node)

    return cycles


def analyze_imports(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Analyze imports and return a report."""
    files = get_python_files(root)
    graph, local_modules = build_dependency_graph(files, root)
    cycles = find_cycles(graph)

    # Calculate module statistics
    import_counts = {module: len(deps) for module, deps in graph.items()}
    imported_by: dict[str, list[str]] = defaultdict(list)
    for module, deps in graph.items():
        for dep in deps:
            imported_by[dep.split(".")[0]].append(module)

    # Find most imported modules (high fan-in)
    high_fan_in = sorted(
        ((mod, len(importers)) for mod, importers in imported_by.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    # Find modules with most imports (high fan-out)
    high_fan_out = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_files": len(files),
        "total_modules": len(local_modules),
        "total_edges": sum(len(deps) for deps in graph.values()),
        "cycles": cycles,
        "cycle_count": len(cycles),
        "high_fan_in": high_fan_in,
        "high_fan_out": high_fan_out,
        "graph": graph,
    }


def print_report(report: dict[str, Any]) -> None:
    """Print a human-readable report."""
    print("=" * 60)
    print("Import Analysis Report")
    print("=" * 60)
    print()
    print(f"Total Python files analyzed: {report['total_files']}")
    print(f"Total modules identified: {report['total_modules']}")
    print(f"Total import edges: {report['total_edges']}")
    print()

    if report["cycles"]:
        print("⚠️  CIRCULAR IMPORTS DETECTED:")
        print("-" * 40)
        for i, cycle in enumerate(report["cycles"], 1):
            print(f"  {i}. {' → '.join(cycle)}")
        print()
    else:
        print("✅ No circular imports detected!")
        print()

    print("Top 10 Most Imported Modules (high fan-in):")
    print("-" * 40)
    for mod, count in report["high_fan_in"]:
        print(f"  {mod}: imported by {count} modules")
    print()

    print("Top 10 Modules with Most Imports (high fan-out):")
    print("-" * 40)
    for mod, count in report["high_fan_out"]:
        print(f"  {mod}: imports {count} modules")
    print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze Python imports for circular dependencies")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output graph to JSON file",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with code 1 if circular imports found (for CI)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output errors and warnings",
    )
    args = parser.parse_args()

    report = analyze_imports()

    if not args.quiet:
        print_report(report)

    if args.output:
        # Save graph to JSON (excluding non-serializable parts)
        output_data = {
            "total_files": report["total_files"],
            "total_modules": report["total_modules"],
            "total_edges": report["total_edges"],
            "cycles": report["cycles"],
            "cycle_count": report["cycle_count"],
            "high_fan_in": report["high_fan_in"],
            "high_fan_out": report["high_fan_out"],
            "graph": report["graph"],
        }
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"Graph saved to {args.output}")

    if args.check and report["cycles"]:
        print(f"❌ Found {len(report['cycles'])} circular import(s)!", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
