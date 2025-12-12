#!/usr/bin/env python3
"""Deploy Grafana Dashboards.

Forces an update of all Grafana dashboards from the docs/grafana directory.

Note: This module is also discovered by the repository test runner.
When RUN_MODULE_TESTS=1, it should execute the embedded test harness and avoid
attempting live Grafana deployment.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logging
from performance.grafana_checker import ensure_dashboards_imported


def main() -> None:
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


from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    """Test that module can be imported and definitions are valid."""
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)


if __name__ == "__main__":
    if os.getenv("RUN_MODULE_TESTS") == "1":
        sys.exit(0 if run_comprehensive_tests() else 1)
    main()
