#!/usr/bin/env python3
"""
Deploy Grafana Dashboards
Forces an update of all Grafana dashboards from the docs/grafana directory.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logging
from performance.grafana_checker import ensure_dashboards_imported


def main():
    setup_logging()
    print("🚀 Deploying Grafana dashboards...")

    try:
        success = ensure_dashboards_imported(force=True)
        if success:
            print("✅ Dashboards deployed successfully!")
            sys.exit(0)
        else:
            print("❌ Failed to deploy dashboards (Grafana might not be running)")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error deploying dashboards: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
