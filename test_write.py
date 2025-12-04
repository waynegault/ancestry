import os
from pathlib import Path


def main():
    project_root = Path(__file__).parent.resolve()
    logs_dir = project_root / "Logs"
    logs_dir.mkdir(exist_ok=True)

    test_file = logs_dir / "test_write.txt"

    print(f"Writing to: {test_file}")

    try:
        with test_file.open("w", encoding="utf-8") as f:
            f.write("Hello from test_write.py")
        print("Write successful")
    except Exception as e:
        print(f"Write failed: {e}")


if __name__ == "__main__":
    main()
