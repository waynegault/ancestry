#!/usr/bin/env python3
"""
Script to scan the codebase and update docs/code_graph.json.
Preserves existing manual metadata for nodes that still exist.
Adds new nodes for found files, classes, and functions.
Removes nodes for code that no longer exists.
"""

import sys
from pathlib import Path

# Add repo root to path to allow imports if needed, though this script uses safe parsing
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_venv() -> None:
    """Ensure running in venv, auto-restart if needed."""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        return

    venv_python = REPO_ROOT / '.venv' / 'Scripts' / 'python.exe'
    if not venv_python.exists():
        venv_python = REPO_ROOT / '.venv' / 'bin' / 'python'
        if not venv_python.exists():
            print("⚠️  WARNING: Not running in virtual environment")
            return

    import os as _os

    print(f"🔄 Re-running with venv Python: {venv_python}")
    _os.chdir(REPO_ROOT)
    _os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])


_ensure_venv()

import ast
import json
import os
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from typing import Any, cast

from testing.test_framework import TestSuite, create_standard_test_runner

GRAPH_PATH = REPO_ROOT / "docs" / "code_graph.json"


DEFAULT_DOCUMENTATION: dict[str, str] = {
    "primary": "readme.md - Comprehensive project documentation",
    "codebase": "docs/visualize_code_graph.html - Interactive code visualization",
    "generator": "scripts/update_code_graph.py - Regenerates docs/code_graph.json",
    "maintenance": "scripts/maintain_code_graph.py - Small maintenance operations on docs/code_graph.json",
    "testing": "testing/test_framework.py - TestSuite framework and standard test runner",
    "monitoring": "docs/grafana/ - Grafana dashboards and setup scripts",
    "development": ".github/copilot-instructions.md - AI development guidelines",
    "tasks": "todo.md - Implementation roadmap and production readiness status",
    "audit": "docs/specs/mission_execution_spec.md - Mission scope, gaps, acceptance criteria",
}

# Directories to exclude from scanning
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".vscode",
    ".github",
    "build",
    "dist",
    "site-packages",
    "node_modules",
    "Logs",
    "Cache",
    "Data",
    "test_data",
}

# Files to exclude
SKIP_FILES = {
    "__init__.py",
    "setup.py",
}


class CodeGraphUpdater:
    def __init__(self, root_dir: Path, graph_path: Path):
        self.root_dir = root_dir
        self.graph_path = graph_path
        self.current_nodes: dict[str, dict[str, Any]] = {}
        self.scanned_nodes: dict[str, dict[str, Any]] = {}
        self.generated_links: list[dict[str, str]] = []

    def load_current_graph(self) -> None:
        """Loads existing graph to preserve metadata."""
        if not self.graph_path.exists():
            print(f"Graph file not found at {self.graph_path}, creating new.")
            self.current_nodes = {}
            return

        with self.graph_path.open(encoding="utf-8") as f:
            data = json.load(f)
            # Store nodes by ID for easy lookup
            for node in data.get("nodes", []):
                self.current_nodes[node["id"]] = node

    def scan_codebase(self) -> None:
        """Walks the codebase and parses Python files."""
        print(f"Scanning from {self.root_dir}...")
        for root, dirs, files in os.walk(self.root_dir):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

            for file in files:
                if not file.endswith(".py") or file in SKIP_FILES:
                    continue

                file_path = Path(root) / file
                self._parse_file(file_path)

    def _parse_file(self, file_path: Path):
        rel_path = file_path.relative_to(self.root_dir).as_posix()
        file_id = f"file:{rel_path}"

        try:
            with file_path.open(encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content)
        except Exception as e:
            print(f"Error parsing {rel_path}: {e}")
            return

        docstring = ast.get_docstring(tree)

        # Add File Node
        self.scanned_nodes[file_id] = {
            "id": file_id,
            "type": "file",
            "name": file_path.name,
            "path": rel_path,
            "summary": docstring.split('\n')[0] if docstring else "Python source file.",
            "mechanism": "Python module.",
            "quality": "new",  # Default for new nodes
            "concerns": [],
            "opportunities": [],
            "tests": None,  # or "Pending check"
            "notes": "",
        }

        self._visit_nodes(tree, rel_path)

    def _visit_nodes(self, tree: ast.AST, rel_path: str):
        """Visits nodes using GraphVisitor and generates links."""
        module_name = rel_path.replace("/", ".").replace(".py", "")
        file_id = f"file:{rel_path}"

        class GraphVisitor(ast.NodeVisitor):
            def __init__(self, scanner: "CodeGraphUpdater"):
                self.scanner = scanner
                self.current_class = None
                self.current_function = None

            def visit_ClassDef(self, node: ast.ClassDef):
                node_id = f"class:{module_name}.{node.name}"
                doc = ast.get_docstring(node)
                self.scanner.scanned_nodes[node_id] = {
                    "id": node_id,
                    "type": "class",
                    "name": node.name,
                    "path": rel_path,
                    "summary": doc.split('\n')[0] if doc else f"Class {node.name}.",
                    "mechanism": "Class definition.",
                    "quality": "new",
                    "concerns": [],
                    "opportunities": [],
                    "tests": None,
                    "notes": "",
                }
                # Link: file defines class
                self.scanner.generated_links.append(
                    {
                        "source": file_id,
                        "target": node_id,
                        "kind": "defines",
                    }
                )

                prev_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = prev_class

            def visit_FunctionDef(self, node: ast.FunctionDef):
                if self.current_class:
                    # Method
                    if node.name.startswith("__") and node.name != "__init__":
                        return

                    node_id = f"method:{self.current_class}.{node.name}"
                    node_type = "method"
                    name_display = f"{self.current_class}.{node.name}"
                    # Link: class defines method
                    class_id = f"class:{module_name}.{self.current_class}"
                    self.scanner.generated_links.append(
                        {
                            "source": class_id,
                            "target": node_id,
                            "kind": "defines",
                        }
                    )
                else:
                    # Top-level function
                    node_id = f"function:{module_name}.{node.name}"
                    node_type = "function"
                    name_display = f"{module_name}.{node.name}"
                    # Link: file defines function
                    self.scanner.generated_links.append(
                        {
                            "source": file_id,
                            "target": node_id,
                            "kind": "defines",
                        }
                    )

                doc = ast.get_docstring(node)
                self.scanner.scanned_nodes[node_id] = {
                    "id": node_id,
                    "type": node_type,
                    "name": name_display,
                    "path": rel_path,
                    "summary": doc.split('\n')[0] if doc else f"{node_type.capitalize()} {node.name}.",
                    "mechanism": "Function/Method logic.",
                    "quality": "new",
                    "concerns": [],
                    "opportunities": [],
                    "tests": None,
                    "notes": "",
                }

        GraphVisitor(self).visit(tree)

    def _merge_nodes(self) -> tuple[list[dict[str, Any]], int, int, int]:
        final_nodes: list[dict[str, Any]] = []
        updated_count = 0
        new_count = 0

        sorted_scanned_ids = sorted(self.scanned_nodes.keys())
        preserve_fields = ["mechanism", "quality", "concerns", "opportunities", "tests", "notes", "summary"]

        for node_id in sorted_scanned_ids:
            new_node = self.scanned_nodes[node_id]
            if node_id in self.current_nodes:
                current = self.current_nodes[node_id]
                merged = new_node.copy()
                for field in preserve_fields:
                    if current.get(field) and current.get(field) != "new":
                        merged[field] = current[field]
                final_nodes.append(merged)
                updated_count += 1
                continue

            if new_node["summary"] == "Python source file.":
                new_node["summary"] = f"Module {new_node['name']}"
            final_nodes.append(new_node)
            new_count += 1

        removed_count = sum(1 for old_id in self.current_nodes if old_id not in self.scanned_nodes)
        return final_nodes, updated_count, new_count, removed_count

    def _build_output_data(
        self,
        final_nodes: list[dict[str, Any]],
        new_count: int,
        removed_count: int,
    ) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "metadata": {
                "schemaVersion": "1.1",
                "generatedAt": now_iso,
                "lastUpdated": now_iso,
                "description": "Code structure knowledge graph for Ancestry research automation platform.",
                "scope": "Complete project codebase analysis including core/*, actions, utilities, and AI integration layers.",
                "recentUpdates": f"Automated update: {new_count} added, {removed_count} removed.",
                "documentation": DEFAULT_DOCUMENTATION,
            },
            "nodes": final_nodes,
            "links": self.generated_links,
        }

    def _apply_existing_metadata(
        self,
        output_data: dict[str, Any],
        *,
        new_count: int,
        updated_count: int,
        removed_count: int,
    ) -> None:
        if not self.graph_path.exists():
            return

        with self.graph_path.open(encoding="utf-8") as f:
            orig_data = json.load(f)
        if "metadata" not in orig_data:
            return

        output_data["metadata"] = orig_data["metadata"]
        now_iso = datetime.now(timezone.utc).isoformat()
        output_data["metadata"]["generatedAt"] = now_iso
        output_data["metadata"]["lastUpdated"] = now_iso
        output_data["metadata"]["scope"] = (
            "Complete project codebase analysis including core/*, actions, utilities, and AI integration layers."
        )
        output_data["metadata"]["recentUpdates"] = (
            f"{datetime.now().strftime('%Y-%m-%d')}: Automated scan. {new_count} new, {updated_count} updated, {removed_count} removed. {len(self.generated_links)} links generated."
        )

        existing_docs = output_data["metadata"].get("documentation", {})
        merged_docs: dict[str, str] = dict(existing_docs) if isinstance(existing_docs, dict) else {}
        merged_docs.update(DEFAULT_DOCUMENTATION)
        output_data["metadata"]["documentation"] = merged_docs

    def update_graph(self) -> None:
        self.load_current_graph()

        # Override simple scan with visitor scan
        self.scan_codebase()

        final_nodes, updated_count, new_count, removed_count = self._merge_nodes()

        print(
            f"Graph update complete: {len(final_nodes)} nodes (Updated: {updated_count}, New: {new_count}, Removed: {removed_count})"
        )

        output_data = self._build_output_data(final_nodes, new_count=new_count, removed_count=removed_count)

        print(f"Generated {len(self.generated_links)} links from code structure.")

        self._apply_existing_metadata(
            output_data,
            new_count=new_count,
            updated_count=updated_count,
            removed_count=removed_count,
        )

        with self.graph_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)


def _assert_metadata_documentation_merges_and_overrides() -> None:
    with TemporaryDirectory() as tmp:
        root_dir = Path(tmp)
        (root_dir / "pkg").mkdir(parents=True, exist_ok=True)
        (root_dir / "pkg" / "sample.py").write_text(
            """\
def hello(name: str) -> str:
    return f\"Hello {name}\"
""",
            encoding="utf-8",
        )

        graph_path = root_dir / "docs" / "code_graph.json"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "schemaVersion": "1.1",
                        "documentation": {
                            "testing": "test_examples/README.md - BAD STALE PATH",
                            "extra": "keep-me",
                        },
                    },
                    "nodes": [],
                    "links": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        updater = CodeGraphUpdater(root_dir=root_dir, graph_path=graph_path)
        updater.update_graph()

        updated = json.loads(graph_path.read_text(encoding="utf-8"))
        docs = updated.get("metadata", {}).get("documentation", {})
        assert isinstance(docs, dict)

        docs_typed = cast(dict[str, str], docs)
        assert docs_typed.get("testing") == DEFAULT_DOCUMENTATION["testing"]
        assert docs_typed.get("extra") == "keep-me"
        assert docs_typed.get("generator") == DEFAULT_DOCUMENTATION["generator"]


def module_tests() -> bool:
    suite = TestSuite("scripts.update_code_graph", "scripts/update_code_graph.py")
    suite.start_suite()
    suite.run_test(
        test_name="Merges documentation metadata",
        test_func=_assert_metadata_documentation_merges_and_overrides,
        test_summary="Ensure docs/code_graph.json metadata stays accurate",
        functions_tested="CodeGraphUpdater.update_graph",
        method_description="Generate graph in a temp repo with stale metadata, then assert canonical keys overwrite stale values while preserving extras",
        expected_outcome="documentation.testing points to testing/test_framework.py; extra keys preserved",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    updater = CodeGraphUpdater(REPO_ROOT, GRAPH_PATH)
    updater.update_graph()
