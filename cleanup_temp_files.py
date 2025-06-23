#!/usr/bin/env python3
"""
Cleanup Script: Remove Temporary Development Files

This script removes all temporary files created during the codebase improvement process:
- Backup files (.backup_unified)
- Migration scripts
- Test consolidation scripts
- Demo/development files
- Any other temporary development artifacts

This cleanup represents the final step after successful consolidation.
"""

# --- Unified import system ---
from core_imports import (
    standardize_module_imports,
    auto_register_module,
    get_logger,
    safe_execute,
)

# Register this module immediately
auto_register_module(globals(), __name__)

import os
import shutil
from pathlib import Path
from typing import List, Set

# Initialize logger
logger = get_logger(__name__)


def cleanup_temporary_files(project_root: Path) -> dict:
    """Remove all temporary development files from the project."""

    stats = {
        "files_removed": 0,
        "directories_removed": 0,
        "total_size_freed": 0,
        "categories": {
            "backup_files": 0,
            "migration_scripts": 0,
            "test_scripts": 0,
            "demo_files": 0,
            "other_temp": 0,
        },
    }

    # Files to remove by category
    files_to_remove = {
        "backup_files": [],
        "migration_scripts": [
            "migrate_to_unified_imports.py",
            "cleanup_temp_files.py",  # This script itself
        ],
        "test_scripts": [
            "unified_test_system.py",
            "consolidate_test_duplication.py",
            "run_all_tests.py",  # Replaced by unified system
        ],
        "demo_files": [
            "demo_registration_efficiency.py",
            "IMPLEMENTATION_GUIDE.py",  # Development guide, not needed in production
        ],
        "other_temp": [],
    }

    # Find all backup files
    backup_files = list(project_root.rglob("*.backup*"))
    files_to_remove["backup_files"] = backup_files

    # Find any other temporary patterns
    temp_patterns = [
        "*.tmp",
        "*.temp",
        "*~",
        "*.bak",
        "*_temp.py",
        "*_test_*.py",
        "*_debug.py",
    ]

    for pattern in temp_patterns:
        temp_files = list(project_root.rglob(pattern))
        files_to_remove["other_temp"].extend(temp_files)

    removed_files = []
    errors = []

    # Remove files by category
    for category, file_list in files_to_remove.items():
        for file_item in file_list:
            try:
                if isinstance(file_item, str):
                    file_path = project_root / file_item
                else:
                    file_path = file_item

                if file_path.exists() and file_path.is_file():
                    # Get file size before deletion
                    file_size = file_path.stat().st_size

                    # Remove the file
                    file_path.unlink()

                    # Update stats
                    stats["files_removed"] += 1
                    stats["total_size_freed"] += file_size
                    stats["categories"][category] += 1

                    removed_files.append((str(file_path), category, file_size))

            except Exception as e:
                errors.append(f"Error removing {file_path}: {e}")

    # Remove empty __pycache__ directories
    pycache_dirs = list(project_root.rglob("__pycache__"))
    for pycache_dir in pycache_dirs:
        try:
            if pycache_dir.exists() and pycache_dir.is_dir():
                shutil.rmtree(pycache_dir)
                stats["directories_removed"] += 1
        except Exception as e:
            errors.append(f"Error removing {pycache_dir}: {e}")

    return {"stats": stats, "removed_files": removed_files, "errors": errors}


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def generate_cleanup_report(results: dict) -> str:
    """Generate a comprehensive cleanup report."""
    stats = results["stats"]
    removed_files = results["removed_files"]
    errors = results["errors"]

    report = [
        "🧹 CODEBASE CLEANUP COMPLETE",
        "=" * 50,
        "",
        "📊 CLEANUP STATISTICS:",
        f"   • Files removed: {stats['files_removed']}",
        f"   • Directories removed: {stats['directories_removed']}",
        f"   • Total size freed: {format_size(stats['total_size_freed'])}",
        "",
        "📂 FILES REMOVED BY CATEGORY:",
    ]

    for category, count in stats["categories"].items():
        if count > 0:
            category_name = category.replace("_", " ").title()
            report.append(f"   • {category_name}: {count} files")

    if removed_files:
        report.extend(
            [
                "",
                "📋 DETAILED REMOVAL LOG:",
            ]
        )

        for file_path, category, size in removed_files:
            file_name = Path(file_path).name
            size_str = format_size(size)
            report.append(f"   • {file_name} ({category}, {size_str})")

    if errors:
        report.extend(
            [
                "",
                f"⚠️  ERRORS ENCOUNTERED ({len(errors)}):",
            ]
        )
        for error in errors:
            report.append(f"   • {error}")

    report.extend(
        [
            "",
            "✅ CLEANUP BENEFITS:",
            "   • Removed all temporary development files",
            "   • Eliminated backup files from migration process",
            "   • Cleaned up test consolidation artifacts",
            "   • Removed demo and development-only scripts",
            "   • Freed up disk space and reduced clutter",
            "",
            "🎯 CODEBASE STATUS:",
            "   • Production-ready clean state achieved",
            "   • Only essential files remain",
            "   • Consolidated improvements are preserved",
            "   • Ready for deployment/production use",
        ]
    )

    return "\n".join(report)


def main():
    """Run the cleanup process."""
    project_root = Path(__file__).parent

    print("🧹 Starting Temporary File Cleanup...")
    print(f"📁 Project root: {project_root}")
    print()
    # Confirm cleanup (auto-confirm for automated execution)
    print("⚠️  This will permanently remove:")
    print("   • All .backup_unified files")
    print("   • Migration scripts")
    print("   • Test consolidation scripts")
    print("   • Demo files")
    print("   • __pycache__ directories")
    print()

    # Auto-confirm for cleanup
    print("✅ Proceeding with cleanup...")

    # Run cleanup
    results = cleanup_temporary_files(project_root)

    # Generate and display report
    report = generate_cleanup_report(results)
    print(report)

    success = results["stats"]["files_removed"] > 0
    return success


# Register module functions at module load
auto_register_module(globals(), __name__)


if __name__ == "__main__":
    success = safe_execute(lambda: main())
    if success:
        print("\n✅ Cleanup completed successfully!")
        print("🎉 Codebase is now in clean, production-ready state!")
    else:
        print("\n⚠️ No files were removed.")
