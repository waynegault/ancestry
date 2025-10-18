"""Remove trailing whitespace from files."""

from pathlib import Path

files_to_fix = [
    "dna_ethnicity_utils.py",
    "setup_ethnicity_tracking.py",
    "backfill_ethnicity_data.py",
]

for filename in files_to_fix:
    filepath = Path(filename)
    if not filepath.exists():
        print(f"Skipping {filename} - not found")
        continue
    
    content = filepath.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    
    fixed_lines = []
    changes = 0
    for i, line in enumerate(lines, 1):
        # Remove trailing whitespace but keep the newline
        if line.endswith(('\n', '\r\n')):
            stripped = line.rstrip() + '\n'
            if stripped != line:
                changes += 1
            fixed_lines.append(stripped)
        else:
            # Last line without newline
            stripped = line.rstrip()
            if stripped != line:
                changes += 1
            fixed_lines.append(stripped)
    
    if changes > 0:
        filepath.write_text(''.join(fixed_lines), encoding="utf-8")
        print(f"✅ Fixed {changes} lines in {filename}")
    else:
        print(f"✓ No changes needed in {filename}")

print("\nDone!")

