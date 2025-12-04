import os
import sys
import subprocess
from pathlib import Path

# Configure env vars BEFORE importing logging_config
os.environ["LOG_DIR"] = "."
os.environ["LOG_FILE"] = "linter_debug.txt"

# Now import
try:
    from core.logging_config import setup_logging
except ImportError:
    # Fix path if needed
    sys.path.append(str(Path(__file__).parent.resolve()))
    from core.logging_config import setup_logging

import logging


def main():
    # Setup logging to write to linter_debug.txt in current dir
    logger = setup_logging(log_level="INFO")

    logger.info("Starting linter check via logging infrastructure...")

    cmd = [sys.executable, "-m", "ruff", "check", "--select", "ARG,PLW,PTH,B,SIM", "."]
    logger.info(f"Running command: {cmd}")

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=Path.cwd())

        logger.info(f"Return Code: {result.returncode}")
        logger.info("--- STDOUT ---")
        logger.info(result.stdout)
        logger.info("--- STDERR ---")
        logger.info(result.stderr)

    except Exception as e:
        logger.error(f"Failed to run linter: {e}")


if __name__ == "__main__":
    main()
