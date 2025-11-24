#!/usr/bin/env python3
"""Maintenance script for docs/code_graph.json."""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_framework import TestSuite, create_standard_test_runner

DEFAULT_GRAPH_RELATIVE = Path("docs/code_graph.json")


def load_graph(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_graph(path: Path, graph: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)


def remove_node(graph: dict[str, Any], node_id: str) -> bool:
    nodes = graph.get("nodes", [])
    initial_count = len(nodes)
    graph["nodes"] = [n for n in nodes if n.get("id") != node_id]

    if len(graph["nodes"]) == initial_count:
        print(f"Node '{node_id}' not found.")
        return False

    edges = graph.get("links", [])
    graph["links"] = [e for e in edges if e.get("source") != node_id and e.get("target") != node_id]

    print(f"Removed node '{node_id}' and associated edges.")
    return True


def resolve_graph_path(search_root: Path | None = None) -> Path:
    """Locate docs/code_graph.json relative to *search_root* or CWD."""
    base = search_root or Path.cwd()
    candidates = [base / DEFAULT_GRAPH_RELATIVE, base / ".." / DEFAULT_GRAPH_RELATIVE]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("docs/code_graph.json not found.")


def main(args: list[str] | None = None, search_root: Path | None = None) -> int:
    args = args if args is not None else sys.argv[1:]
    if len(args) < 2:
        print("Usage: python maintain_code_graph.py <remove_node> <node_id>")
        return 1

    command, target = args[0], args[1]

    try:
        graph_path = resolve_graph_path(search_root)
    except FileNotFoundError:
        print("Error: docs/code_graph.json not found.")
        return 1

    graph = load_graph(graph_path)

    if command == "remove_node":
        if remove_node(graph, target):
            save_graph(graph_path, graph)
            print("Graph updated successfully.")
        return 0

    print(f"Unknown command: {command}")
    return 1


def _write_graph(root: Path, graph: dict[str, Any]) -> Path:
    graph_path = root / DEFAULT_GRAPH_RELATIVE
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return graph_path


def _run_main(args: list[str], root: Path | None = None) -> tuple[int, str]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = main(args, search_root=root)
    return exit_code, buffer.getvalue()


def _test_remove_node_eliminates_node_and_edges() -> bool:
    graph = {
        "nodes": [{"id": "keep"}, {"id": "drop"}],
        "links": [
            {"source": "keep", "target": "other"},
            {"source": "drop", "target": "keep"},
            {"source": "other", "target": "drop"},
        ],
    }

    assert remove_node(graph, "drop") is True
    assert graph["nodes"] == [{"id": "keep"}]
    assert graph["links"] == [{"source": "keep", "target": "other"}]
    return True


def _test_remove_node_handles_missing_entries() -> bool:
    graph = {"nodes": [{"id": "keep"}], "links": []}
    assert remove_node(graph, "absent") is False
    assert graph["nodes"] == [{"id": "keep"}]
    return True


def _test_resolve_graph_path_falls_back_to_parent() -> bool:
    with TemporaryDirectory() as tmp_dir:
        repo_root = Path(tmp_dir)
        scripts_dir = repo_root / "scripts"
        scripts_dir.mkdir()
        expected = _write_graph(repo_root, {"nodes": [], "links": []}).resolve()
        resolved = resolve_graph_path(scripts_dir)
    assert resolved == expected
    return True


def _test_main_reports_usage_on_missing_args() -> bool:
    exit_code, output = _run_main([], None)
    assert exit_code == 1
    assert "Usage:" in output
    return True


def _test_main_handles_missing_graph_file() -> bool:
    with TemporaryDirectory() as tmp_dir:
        repo_root = Path(tmp_dir)
        exit_code, output = _run_main(["remove_node", "a"], repo_root)
    assert exit_code == 1
    assert "Error: docs/code_graph.json not found." in output
    return True


def _test_main_remove_node_updates_graph_file() -> bool:
    with TemporaryDirectory() as tmp_dir:
        repo_root = Path(tmp_dir)
        _write_graph(
            repo_root,
            {
                "nodes": [{"id": "keep"}, {"id": "drop"}],
                "links": [
                    {"source": "keep", "target": "drop"},
                    {"source": "drop", "target": "keep"},
                ],
            },
        )
        exit_code, output = _run_main(["remove_node", "drop"], repo_root)
        graph_path = (repo_root / DEFAULT_GRAPH_RELATIVE).resolve()
        updated = load_graph(graph_path)

    assert exit_code == 0
    assert "Graph updated successfully." in output
    assert updated["nodes"] == [{"id": "keep"}]
    assert updated["links"] == []
    return True


def _test_main_handles_unknown_command() -> bool:
    with TemporaryDirectory() as tmp_dir:
        repo_root = Path(tmp_dir)
        _write_graph(repo_root, {"nodes": [], "links": []})
        exit_code, output = _run_main(["noop", "target"], repo_root)
    assert exit_code == 1
    assert "Unknown command" in output
    return True


def module_tests() -> bool:
    suite = TestSuite("scripts.maintain_code_graph", "scripts/maintain_code_graph.py")
    suite.run_test(
        "Remove node trims edges",
        _test_remove_node_eliminates_node_and_edges,
        "Ensures remove_node deletes the node and its connected edges.",
    )
    suite.run_test(
        "Remove node missing",
        _test_remove_node_handles_missing_entries,
        "Ensures remove_node gracefully handles absent node ids.",
    )
    suite.run_test(
        "Resolve path fallback",
        _test_resolve_graph_path_falls_back_to_parent,
        "Ensures graph resolution finds docs/ when running from scripts/.",
    )
    suite.run_test(
        "Usage enforcement",
        _test_main_reports_usage_on_missing_args,
        "Ensures CLI emits usage guidance when args are missing.",
    )
    suite.run_test(
        "Missing graph file",
        _test_main_handles_missing_graph_file,
        "Ensures CLI reports missing docs/code_graph.json.",
    )
    suite.run_test(
        "Successful node removal",
        _test_main_remove_node_updates_graph_file,
        "Ensures CLI removes nodes and persists the updated graph.",
    )
    suite.run_test(
        "Unknown command",
        _test_main_handles_unknown_command,
        "Ensures CLI rejects unsupported commands.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    if os.environ.get("RUN_MODULE_TESTS") == "1":
        success = run_comprehensive_tests()
        raise SystemExit(0 if success else 1)
    raise SystemExit(main())
