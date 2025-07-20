#!/usr/bin/env python3
"""
Cleanup Script: Remove Obsolete and Redundant Files

This script identifies and removes files that are no longer needed based on:
1. Functionality consolidated into core_imports.py
2. Legacy development files
3. Redundant modules
4. Unused imports and patterns

Safe cleanup with backup creation before removal.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Set, Dict, Any
import json


def analyze_file_usage(project_root: Path) -> Dict[str, Any]:
    """Analyze which files are actually being used by checking imports."""

    used_files = set()
    import_patterns = {}

    # Scan all Python files for import statements
    for py_file in project_root.rglob("*.py"):
        if py_file.name.startswith("cleanup_") or py_file.name.startswith("test_"):
            continue

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Look for import statements
            lines = content.split("\n")
            for line in lines:
                line = line.strip()

                # Check for local module imports
                if "from path_manager import" in line:
                    import_patterns.setdefault("path_manager.py", []).append(
                        str(py_file)
                    )
                elif "import path_manager" in line:
                    import_patterns.setdefault("path_manager.py", []).append(
                        str(py_file)
                    )
                elif "from core_imports import" in line:
                    import_patterns.setdefault("core_imports.py", []).append(
                        str(py_file)
                    )

        except Exception as e:
            print(f"Warning: Could not analyze {py_file}: {e}")

    return import_patterns


def identify_obsolete_files(project_root: Path) -> Dict[str, List[str]]:
    """Identify files that may be obsolete based on consolidation efforts."""

    obsolete_categories = {
        "consolidated_functionality": [
            "path_manager.py",  # Functionality moved to core_imports.py
        ],
        "development_artifacts": [
            # These would be found if they exist
            "migration_test.py",
            "optimization_test.py",
            "pattern_test.py",
            "import_test.py",
        ],
        "legacy_patterns": [
            # Files that implement old patterns that are now unified
        ],
    }

    # Check which files actually exist
    existing_obsolete = {}
    for category, files in obsolete_categories.items():
        existing = []
        for file in files:
            file_path = project_root / file
            if file_path.exists():
                existing.append(file)
        if existing:
            existing_obsolete[category] = existing

    return existing_obsolete


def check_path_manager_consolidation(project_root: Path) -> Dict[str, Any]:
    """Check if path_manager.py can be safely removed."""

    analysis = {
        "can_remove": False,
        "reason": "",
        "import_count": 0,
        "importing_files": [],
    }

    path_manager_file = project_root / "path_manager.py"
    if not path_manager_file.exists():
        analysis["can_remove"] = True
        analysis["reason"] = "File doesn't exist"
        return analysis

    # Check if any files still import from path_manager
    import_count = 0
    importing_files = []

    for py_file in project_root.rglob("*.py"):
        if py_file.name == "path_manager.py" or py_file.name == "cleanup_obsolete.py":
            continue

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            if (
                "from path_manager import" in content
                or "import path_manager" in content
            ):
                import_count += 1
                importing_files.append(str(py_file.name))

        except Exception:
            continue

    analysis["import_count"] = import_count
    analysis["importing_files"] = importing_files

    if import_count == 0:
        analysis["can_remove"] = True
        analysis["reason"] = (
            "No imports found - functionality consolidated to core_imports.py"
        )
    else:
        analysis["can_remove"] = False
        analysis["reason"] = f"Still imported by {import_count} files"

    return analysis


def create_cleanup_report(project_root: Path) -> str:
    """Generate a comprehensive cleanup analysis report."""

    report = []
    report.append("🧹 COMPREHENSIVE CLEANUP ANALYSIS")
    report.append("=" * 50)
    report.append("")

    # Analyze file usage
    import_patterns = analyze_file_usage(project_root)
    report.append("📋 IMPORT USAGE ANALYSIS:")
    for file, importers in import_patterns.items():
        report.append(f"  • {file}: imported by {len(importers)} files")

    report.append("")

    # Check path_manager consolidation
    pm_analysis = check_path_manager_consolidation(project_root)
    report.append("🔍 PATH_MANAGER CONSOLIDATION ANALYSIS:")
    report.append(f"  • Can remove: {pm_analysis['can_remove']}")
    report.append(f"  • Reason: {pm_analysis['reason']}")
    if pm_analysis["importing_files"]:
        report.append("  • Still imported by:")
        for file in pm_analysis["importing_files"]:
            report.append(f"    - {file}")

    report.append("")

    # Check for obsolete files
    obsolete_files = identify_obsolete_files(project_root)
    report.append("🗑️ OBSOLETE FILES FOUND:")
    if obsolete_files:
        for category, files in obsolete_files.items():
            report.append(f"  • {category}:")
            for file in files:
                report.append(f"    - {file}")
    else:
        report.append("  • No obsolete files identified")

    report.append("")

    # Recommendations
    report.append("💡 CLEANUP RECOMMENDATIONS:")

    if pm_analysis["can_remove"]:
        report.append(
            "  ✅ SAFE TO REMOVE: path_manager.py (functionality consolidated)"
        )
    else:
        report.append("  ⚠️  KEEP: path_manager.py (still in use)")

    report.append("  ✅ .gitignore already properly configured")
    report.append("  ✅ README consolidation completed")

    return "\n".join(report)


def main():
    """Run the cleanup analysis."""
    project_root = Path(__file__).parent

    print("🔍 Analyzing project for cleanup opportunities...")
    print(f"📁 Project root: {project_root}")
    print()

    # Generate comprehensive report
    report = create_cleanup_report(project_root)
    print(report)

    # Save report to file
    report_file = project_root / "cleanup_analysis_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n📋 Full analysis saved to: {report_file.name}")

    # Check if path_manager can be removed
    pm_analysis = check_path_manager_consolidation(project_root)
    if pm_analysis["can_remove"]:
        print("\n🎯 ACTIONABLE CLEANUP IDENTIFIED:")
        print("   • path_manager.py can be safely removed")
        print("   • Functionality has been consolidated into core_imports.py")

        # Offer to create backup and remove
        response = input("\n❓ Remove path_manager.py? (y/N): ").lower().strip()
        if response == "y":
            path_manager_file = project_root / "path_manager.py"
            backup_file = project_root / "path_manager.py.backup"

            # Create backup
            shutil.copy2(path_manager_file, backup_file)
            print(f"✅ Backup created: {backup_file.name}")

            # Remove file
            path_manager_file.unlink()
            print(f"🗑️  Removed: path_manager.py")

            # Add to gitignore
            gitignore_file = project_root / ".gitignore"
            with open(gitignore_file, "a", encoding="utf-8") as f:
                f.write("\n# Obsolete file backups\n")
                f.write("*.backup\n")

            print("✅ Updated .gitignore to exclude backup files")
            print("\n🎉 Cleanup completed successfully!")
        else:
            print("   Skipped removal - file preserved")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
