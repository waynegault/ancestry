#!/usr/bin/env python3
"""
LM Studio Auto-Start Demo

Demonstrates the automatic LM Studio startup functionality.
Run this to test if LM Studio auto-start is working correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import config_schema
from lm_studio_manager import create_manager_from_config


def main():
    print("\n" + "="*60)
    print("LM Studio Auto-Start Demo")
    print("="*60 + "\n")

    print("Configuration:")
    print(f"  Path: {config_schema.api.lm_studio_path}")
    print(f"  Auto-start: {config_schema.api.lm_studio_auto_start}")
    print(f"  Timeout: {config_schema.api.lm_studio_startup_timeout}s")
    print(f"  API URL: {config_schema.api.local_llm_base_url}")
    print()

    # Create manager from config
    manager = create_manager_from_config(config_schema)

    # Try to ensure LM Studio is ready
    print("Ensuring LM Studio is ready...")
    success, error_msg = manager.ensure_ready()

    print()
    if success:
        print("✅ SUCCESS: LM Studio is running and API is ready!")
        print("\nYou can now use AI features with local_llm provider.")
    else:
        print("❌ FAILED: LM Studio could not be started or is not ready")
        print(f"\nError: {error_msg}")
        print("\nTroubleshooting:")
        print("1. Verify LM_STUDIO_PATH in .env points to correct executable")
        print("2. Check that LM Studio can start manually")
        print("3. Ensure no firewall is blocking localhost:1234")
        print("4. Load a model in LM Studio after it starts")
        return 1

    print("\n" + "="*60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
