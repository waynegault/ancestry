import subprocess
import sys
from pathlib import Path


def main():
    cmd = [sys.executable, "-m", "ruff", "check", "--select", "ARG,PLW,PTH,B,SIM", "."]
    print(f"Running: {cmd}")
    try:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=Path.cwd())
        output_path = Path(
            r"C:\Users\wayne\.gemini\antigravity\brain\d79efbac-f0da-4f43-889c-f0b6f37357da\linter_report.txt"
        )
        with output_path.open("w", encoding="utf-8") as f:
            f.write(res.stdout)
            f.write(res.stderr)
        print(f"Done. Wrote to {output_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
