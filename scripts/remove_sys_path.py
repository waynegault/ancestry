import os
import pathlib
import re


def should_remove_line(line):
    # Remove sys.path.insert or sys.path.append lines that look like path setup
    return ("sys.path.insert" in line or "sys.path.append" in line) and any(
        x in line for x in ["0", "1", "root", "parent", "dirname", "abspath", "..", "Path", "cwd"]
    )


def process_file(filepath):
    path = pathlib.Path(filepath)
    with path.open('r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    removed_count = 0
    for line in lines:
        if should_remove_line(line):
            removed_count += 1
            continue
        new_lines.append(line)

    if removed_count > 0:
        print(f"Removing {removed_count} lines from {filepath}")
        with path.open('w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    return False


def main():
    root_dir = pathlib.Path(__file__).resolve().parent.parent
    print(f"Scanning {root_dir}...")

    total_files_changed = 0

    for root, _, files in os.walk(root_dir):
        if ".venv" in root or "node_modules" in root or ".git" in root:
            continue

        for file in files:
            if file.endswith(".py"):
                filepath = pathlib.Path(root) / file
                if file == "remove_sys_path.py":  # Don't edit self
                    continue
                if process_file(filepath):
                    total_files_changed += 1

    print(f"Total files changed: {total_files_changed}")


if __name__ == "__main__":
    main()
