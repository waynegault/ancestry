#!/usr/bin/env python3
"""
Script to scan the codebase and update docs/code_graph.json.
Preserves existing manual metadata for nodes that still exist.
Adds new nodes for found files, classes, and functions.
Removes nodes for code that no longer exists.
"""

import ast
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add repo root to path to allow imports if needed, though this script uses safe parsing
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GRAPH_PATH = REPO_ROOT / "docs" / "code_graph.json"

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
        self.current_nodes: Dict[str, Dict[str, Any]] = {}
        self.scanned_nodes: Dict[str, Dict[str, Any]] = {}

    def load_current_graph(self):
        """Loads existing graph to preserve metadata."""
        if not self.graph_path.exists():
            print(f"Graph file not found at {self.graph_path}, creating new.")
            self.current_nodes = {}
            return

        with self.graph_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Store nodes by ID for easy lookup
            for node in data.get("nodes", []):
                self.current_nodes[node["id"]] = node

    def scan_codebase(self):
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
            with open(file_path, "r", encoding="utf-8") as f:
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
        """Visits ClassDef and FunctionDef nodes."""

        # Helper to get full name including parent classes
        def get_full_name(node, parent_name=None):
            if parent_name:
                return f"{parent_name}.{node.name}"
            # For top level functions/classes in a file, usually we refer to them as module.name
            # But the existing graph schema seems to use file path or just module path.
            # Looking at existing nodes: "class:main._CachingState", "function:main.initialize_aggressive_caching"
            # So it uses dotted module path.
            module_path = rel_path.replace("/", ".").replace(".py", "")
            return f"{module_path}.{node.name}"

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                node_id = f"class:{get_full_name(node)}"
                doc = ast.get_docstring(node)
                self.scanned_nodes[node_id] = {
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

                # Scan methods
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and not child.name.startswith("__"):  # Skip dunder methods
                        # existing schema uses method:ClassName.method_name for methods?
                        # Let's check existing: "method:BrowserManager.start_browser"
                        # Ah, existing graph uses "method:BrowserManager.start_browser" without module prefix sometimes?
                        # Let's check more closely: "method:BrowserManager.start_browser" path is "core/browser_manager.py"
                        # But ID doesn't have module prefix. Let's try to infer from existing if possible, or stick to a convention.
                        # Wait, "function:main.initialize_aggressive_caching" HAS module prefix.
                        # "method:BrowserManager.start_browser" DOES NOT have module prefix in ID? "BrowserManager" is the class.
                        # Let's verify if there are duplicate class names in different files.
                        # It is safer to include module prefix if possible, but let's try to match existing style if it's unique.
                        # Existing ID: "method:BrowserManager.start_browser"
                        # The class ID is "class:core.browser_manager.BrowserManager".
                        # It seems inconsistent.
                        # I will use "method:{ClassName}.{method_name}" as the ID format, which seems to match the "method:BrowserManager.start_browser" example.

                        method_id = f"method:{node.name}.{child.name}"
                        child_doc = ast.get_docstring(child)
                        self.scanned_nodes[method_id] = {
                            "id": method_id,
                            "type": "method",
                            "name": f"{node.name}.{child.name}",
                            "path": rel_path,
                            "summary": child_doc.split('\n')[0] if child_doc else f"Method {child.name}.",
                            "mechanism": "Method.",
                            "quality": "new",
                            "concerns": [],
                            "opportunities": [],
                            "tests": None,
                            "notes": "",
                        }

            elif isinstance(node, ast.FunctionDef):
                # Only process top-level functions here (already processed class methods inside ClassDef block but ast.walk visits all)
                # We need to know if it's top-level.
                # Doing a manual traversal is better than ast.walk for parent tracking, but for quick implementation:
                # We can skip if it's inside a class.
                # Actually, implementing a NodeVisitor is cleaner.
                pass

    def _visit_nodes_recursive(self, rel_path: str):
        """Uses a proper NodeVisitor to handle hierarchy."""

        module_name = rel_path.replace("/", ".").replace(".py", "")

        class GraphVisitor(ast.NodeVisitor):
            def __init__(self, scanner):
                self.scanner = scanner
                self.current_class = None

            def visit_ClassDef(self, node):
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

                prev_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = prev_class

            def visit_FunctionDef(self, node):
                if self.current_class:
                    # Method
                    if node.name.startswith("__") and node.name != "__init__":
                        # Skip dunders except init, or maybe skip all? User said "thoroughly review".
                        # Existing graph seems to skip most dunders.
                        return

                    node_id = f"method:{self.current_class}.{node.name}"
                    node_type = "method"
                    name_display = f"{self.current_class}.{node.name}"
                else:
                    # Top-level function
                    node_id = f"function:{module_name}.{node.name}"
                    node_type = "function"
                    name_display = f"{module_name}.{node.name}"

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
                # No need to visit children of functions usually (inner functions are rarely documented in graph)

        with open(self.root_dir / rel_path, "r", encoding="utf-8") as f:
            content = f.read()
        try:
            tree = ast.parse(content)
            GraphVisitor(self).visit(tree)
        except Exception:
            pass

    def update_graph(self):
        self.load_current_graph()

        # Override simple scan with visitor scan
        print(f"Parsing files from {self.root_dir}...")
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for file in files:
                if not file.endswith(".py") or file in SKIP_FILES:
                    continue
                file_path = Path(root) / file
                rel_path = file_path.relative_to(self.root_dir).as_posix()

                # Add file node first
                file_id = f"file:{rel_path}"
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        doc = ast.get_docstring(ast.parse(f.read()))
                except Exception:
                    doc = ""

                self.scanned_nodes[file_id] = {
                    "id": file_id,
                    "type": "file",
                    "name": file_path.name,  # Use filename as name
                    "path": rel_path,
                    "summary": doc.split('\n')[0] if doc else "Python source file.",
                    "mechanism": "Module.",
                    "quality": "new",
                    "concerns": [],
                    "opportunities": [],
                    "tests": None,
                    "notes": "",
                }

                self._visit_nodes_recursive(rel_path)

        # Merge
        final_nodes = []
        updated_count = 0
        new_count = 0
        removed_count = 0

        # Current nodes that were found in scan -> Update but keep metadata
        # Current nodes NOT found in scan -> Remove (or move to Removed category if complex, but lets remove for now)

        # Sort scanned nodes by ID to maintain order
        sorted_scanned_ids = sorted(self.scanned_nodes.keys())

        for node_id in sorted_scanned_ids:
            new_node = self.scanned_nodes[node_id]
            if node_id in self.current_nodes:
                # Merge: keep manual fields from current
                current = self.current_nodes[node_id]
                merged = new_node.copy()
                # Fields to preserve from manual entry being "Truth"
                preserve_fields = ["mechanism", "quality", "concerns", "opportunities", "tests", "notes", "summary"]
                for field in preserve_fields:
                    if current.get(field) and current.get(field) != "new":  # Don't overwrite if it was just "new"
                        merged[field] = current[field]
                final_nodes.append(merged)
                updated_count += 1
            else:
                # improved summary for totally new files if empty
                if new_node["summary"] == "Python source file.":
                    new_node["summary"] = f"Module {new_node['name']}"
                final_nodes.append(new_node)
                new_count += 1

        # Calculate Removed
        for old_id in self.current_nodes:
            if old_id not in self.scanned_nodes:
                removed_count += 1

        print(
            f"Graph update complete: {len(final_nodes)} nodes (Updated: {updated_count}, New: {new_count}, Removed: {removed_count})"
        )

        # Construct final JSON
        output_data = {
            "metadata": {
                "schemaVersion": "1.1",
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
                "description": "Code structure knowledge graph for Ancestry research automation platform.",
                "scope": "Complete project codebase analysis.",
                "recentUpdates": f"Automated update: {new_count} added, {removed_count} removed.",
                "documentation": self.current_nodes.get("metadata", {}).get(
                    "documentation", {}
                ),  # Try to preserve, but accessing flat nodes dict won't work.
                # Need to load metadata separately.
            },
            "nodes": final_nodes,
            "links": [],  # Would need separate pass for links. For now, empty or preserve?
            # Existing script shows links exist.
            # Ideally we preserve links if both source/target exist.
        }

        # Recover metadata and links from original file if possible
        if self.graph_path.exists():
            with self.graph_path.open("r", encoding="utf-8") as f:
                orig_data = json.load(f)
                if "metadata" in orig_data:
                    output_data["metadata"] = orig_data["metadata"]
                    output_data["metadata"]["lastUpdated"] = datetime.now(timezone.utc).isoformat()
                    output_data["metadata"]["recentUpdates"] = (
                        f"{datetime.now().strftime('%Y-%m-%d')}: Automated scan. {new_count} new, {updated_count} updated, {removed_count} removed."
                    )

                if "links" in orig_data:
                    # Filter links where both nodes still exist
                    valid_ids = set(self.scanned_nodes.keys())
                    valid_links = [
                        l for l in orig_data["links"] if l.get("source") in valid_ids and l.get("target") in valid_ids
                    ]
                    output_data["links"] = valid_links
                    print(f"Preserved {len(valid_links)} valid links.")

        with self.graph_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    updater = CodeGraphUpdater(REPO_ROOT, GRAPH_PATH)
    updater.update_graph()
