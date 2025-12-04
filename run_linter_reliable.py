import subprocess
import sys
import os
from pathlib import Path


def main():
    # Resolve Logs directory relative to project root, similar to logging_config.py
    # Assuming this script is in the project root
    project_root = Path(__file__).parent.resolve()
    logs_dir = project_root / "Logs"
    logs_dir.mkdir(exist_ok=True)

    report_path = logs_dir / "linter_report.txt"

    print(f"Project root: {project_root}")
    print(f"Logs directory: {logs_dir}")
    print(f"Report path: {report_path}")

    cmd = [sys.executable, "-m", "ruff", "check", "--select", "ARG,PLW,PTH,B,SIM", "."]
    print(f"Running command: {cmd}")

    try:
        # Run ruff and capture output
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=project_root)

        # Write to file with explicit encoding
        with report_path.open("w", encoding="utf-8") as f:
            f.write(f"Return Code: {result.returncode}\n")
            f.write("--- STDOUT ---\n")
            f.write(result.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

        print(f"Successfully wrote linter report to: {report_path}")

    except Exception as e:
        print(f"Failed to run linter or write report: {e}")
        # Try to write error to a fallback file in current dir
        try:
            with Path("linter_panic.txt").open("w", encoding="utf-8") as f:
                f.write(str(e))
        except:
            pass


if __name__ == "__main__":
    main()
