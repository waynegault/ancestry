#!/usr/bin/env python3
"""
Phase 1 Cleanup Script - Version 2
More careful approach to standardizing imports without corrupting files
"""

import os
import re
import sys
from pathlib import Path


def read_file_safe(filepath):
    """Safely read file with proper encoding handling"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return None


def write_file_safe(filepath, content):
    """Safely write file with proper encoding"""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"❌ Error writing {filepath}: {e}")
        return False


def has_core_imports(content):
    """Check if file already uses core_imports"""
    return "from core_imports import" in content


def get_shebang(content):
    """Extract shebang line if present"""
    if content.startswith("#!"):
        return content.split("\n")[0] + "\n"
    return ""


def get_docstring(content):
    """Extract module docstring if present"""
    lines = content.split("\n")
    in_docstring = False
    docstring_lines = []
    quote_type = None

    for i, line in enumerate(lines):
        if not in_docstring:
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote_type = stripped[:3]
                in_docstring = True
                docstring_lines.append(line)
                if stripped.count(quote_type) >= 2 and len(stripped) > 3:
                    # Single line docstring
                    return "\n".join(docstring_lines) + "\n"
                continue
        else:
            docstring_lines.append(line)
            if quote_type in line:
                return "\n".join(docstring_lines) + "\n"

    return ""


def remove_duplicate_auto_register_calls(content):
    """Remove duplicate auto_register_module calls"""
    lines = content.split("\n")
    auto_register_pattern = r"auto_register_module\(globals\(\),\s*__name__\)"

    found_auto_register = False
    cleaned_lines = []

    for line in lines:
        if re.search(auto_register_pattern, line):
            if not found_auto_register:
                cleaned_lines.append(line)
                found_auto_register = True
            # Skip duplicate calls
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def standardize_file_imports(filepath):
    """Standardize imports in a single file carefully"""
    print(f"Processing: {filepath}")

    # Skip critical system files
    filename = os.path.basename(filepath)
    if filename in ["core_imports.py", "run_all_tests.py"]:
        print(f"  ⏭️  Skipping system file: {filename}")
        return False

    content = read_file_safe(filepath)
    if content is None:
        return False

    # Skip files that don't need core_imports
    if (
        not has_core_imports(content)
        and "def " not in content
        and "class " not in content
    ):
        print(f"  ⏭️  Skipping simple script: {filename}")
        return False

    original_content = content

    # Extract shebang and docstring
    shebang = get_shebang(content)
    docstring = get_docstring(content)

    # Remove duplicates only
    content = remove_duplicate_auto_register_calls(content)

    # If no changes made, skip
    if content == original_content:
        print(f"  ✅ No changes needed")
        return False

    # Write back the file
    if write_file_safe(filepath, content):
        print(f"  ✅ Cleaned duplicate auto_register_module calls")
        return True
    else:
        return False


def main():
    """Main cleanup process"""
    print("Phase 1 Cleanup v2 - Conservative Import Standardization")
    print("=" * 60)

    # Get project root
    project_root = Path(__file__).parent

    # Find Python files to process
    python_files = []

    # Main directory Python files
    for filepath in project_root.glob("*.py"):
        if filepath.name not in [
            "phase1_cleanup.py",
            "phase1_cleanup_v2.py",
            "test_phase1.py",
        ]:
            python_files.append(filepath)

    # Core directory
    core_dir = project_root / "core"
    if core_dir.exists():
        for filepath in core_dir.glob("*.py"):
            python_files.append(filepath)

    # Config directory
    config_dir = project_root / "config"
    if config_dir.exists():
        for filepath in config_dir.glob("*.py"):
            python_files.append(filepath)

    print(f"Found {len(python_files)} Python files to process")
    print()

    # Process each file
    modified_count = 0
    error_count = 0

    for filepath in sorted(python_files):
        try:
            if standardize_file_imports(filepath):
                modified_count += 1
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
            error_count += 1

    print()
    print("=" * 60)
    print(f"Phase 1 Cleanup v2 Complete!")
    print(f"Files processed: {len(python_files)}")
    print(f"Files modified: {modified_count}")
    print(f"Errors: {error_count}")
    print(
        f"Success rate: {((len(python_files) - error_count) / len(python_files)) * 100:.1f}%"
    )

    if modified_count > 0:
        print("\n⚠️  Run tests to verify changes: python run_all_tests.py")


if __name__ == "__main__":
    main()
