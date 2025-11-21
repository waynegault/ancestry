#!/usr/bin/env python3
"""
Maintenance script for docs/code_graph.json.
Allows removing stale nodes and edges programmatically.
"""
import json
import sys
from pathlib import Path
from typing import Any


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

    # Remove edges connected to this node
    edges = graph.get("links", [])
    graph["links"] = [
        e for e in edges if e.get("source") != node_id and e.get("target") != node_id
    ]

    print(f"Removed node '{node_id}' and associated edges.")
    return True


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python maintain_code_graph.py <remove_node> <node_id>")
        sys.exit(1)

    command = sys.argv[1]
    target = sys.argv[2]

    graph_path = Path("docs/code_graph.json")
    if not graph_path.exists():
        # Try looking up one level if running from scripts/
        graph_path = Path("../docs/code_graph.json")
        if not graph_path.exists():
            print("Error: docs/code_graph.json not found.")
            sys.exit(1)

    graph = load_graph(graph_path)

    if command == "remove_node":
        if remove_node(graph, target):
            save_graph(graph_path, graph)
            print("Graph updated successfully.")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
