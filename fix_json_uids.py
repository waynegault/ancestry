import json
import os
import pathlib


def fix_dashboard(file_path: str) -> None:
    print(f"Processing {file_path}...")
    with pathlib.Path(file_path).open('r', encoding='utf-8') as f:
        content = f.read()

    # Replace variables with hardcoded UIDs
    new_content = content.replace('${DS_PROMETHEUS}', 'ancestry-prometheus')
    new_content = new_content.replace('${DS_SQLITE}', 'ancestry-sqlite')

    # Load as JSON to verify validity and remove __inputs if present (though I already did that via regex for one part)
    try:
        data = json.loads(new_content)
        if '__inputs' in data:
            print(f"Removing __inputs from {file_path}")
            del data['__inputs']

        # Write back formatted JSON
        with pathlib.Path(file_path).open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully updated {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {file_path}: {e}")


base_path = pathlib.Path(r"c:\Users\wayne\GitHub\Python\Projects\Ancestry\docs\grafana")
fix_dashboard(str(base_path / "ancestry_overview.json"))
fix_dashboard(str(base_path / "system_performance.json"))
