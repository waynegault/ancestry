import subprocess
import sys
import os
from pathlib import Path


def main():
    project_root = Path(__file__).parent.resolve()
    report_path = project_root / "linter_report_root.txt"

    print(f"Project root: {project_root}")
    print(f"Report path: {report_path}")

    cmd = [sys.executable, "-m", "ruff", "check", "--select", "ARG,PLW,PTH,B,SIM", "."]
    print(f"Running command: {cmd}")

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=project_root)

        content = f"Return Code: {result.returncode}\n"
        content += "--- STDOUT ---\n"
        content += result.stdout
        content += "\n--- STDERR ---\n"
        content += result.stderr

        with report_path.open("w", encoding="utf-8") as f:
            f.write(content)

        print(f"Successfully wrote linter report to: {report_path}")
        print("--- CONTENT PREVIEW ---")
        print(content[:500])

    except Exception as e:
        print(f"Failed: {e}")


if __name__ == "__main__":
    main()
