#!/usr/bin/env python3
"""Deploy Grafana dashboards from JSON files to a running Grafana instance.

This script automates the deployment of pre-configured dashboards to Grafana.
It reads dashboard JSON files from docs/grafana/ and imports them via the
Grafana HTTP API.

Usage:
    python scripts/deploy_dashboards.py
    python scripts/deploy_dashboards.py --url http://localhost:3000 --token <api_token>

Requirements:
    - Grafana must be running and accessible
    - API token with dashboard creation permissions
    - requests library (pip install requests)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from testing.test_framework import TestSuite

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Default Grafana configuration
DEFAULT_GRAFANA_URL = "http://localhost:3000"
DEFAULT_API_TOKEN = os.environ.get("GRAFANA_API_TOKEN", "")

# Dashboard files to deploy
DASHBOARD_DIR = PROJECT_ROOT / "docs" / "grafana"
DASHBOARD_FILES = [
    "ancestry_overview.json",
    "code_quality.json",
    "database_summary.json",
    "genealogy_insights.json",
    "system_performance.json",
]


def load_dashboard(file_path: Path) -> dict[str, Any]:
    """Load a dashboard JSON file.

    Args:
        file_path: Path to the dashboard JSON file.

    Returns:
        Dashboard configuration dictionary.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    with Path(file_path).open(encoding="utf-8") as f:
        return json.load(f)


def prepare_dashboard_payload(dashboard: dict[str, Any]) -> dict[str, Any]:
    """Prepare dashboard for Grafana API import.

    Grafana's import API expects a specific format with the dashboard
    wrapped in a payload object.

    Args:
        dashboard: The dashboard configuration.

    Returns:
        API-ready payload dictionary.
    """
    # Remove __requires as it's not needed for import
    dashboard_copy = dashboard.copy()
    dashboard_copy.pop("__requires", None)

    # Ensure the dashboard has required fields
    if "id" not in dashboard_copy:
        dashboard_copy["id"] = None  # Let Grafana assign an ID

    return {
        "dashboard": dashboard_copy,
        "overwrite": True,  # Overwrite existing dashboards with same uid
        "message": "Deployed via scripts/deploy_dashboards.py",
    }


def deploy_dashboard(
    grafana_url: str,
    api_token: str,
    dashboard_path: Path,
) -> tuple[bool, str]:
    """Deploy a single dashboard to Grafana.

    Args:
        grafana_url: Base URL of Grafana instance.
        api_token: API token for authentication.
        dashboard_path: Path to dashboard JSON file.

    Returns:
        Tuple of (success, message).
    """
    if requests is None:
        return False, "requests library not installed (pip install requests)"

    try:
        dashboard = load_dashboard(dashboard_path)
        payload = prepare_dashboard_payload(dashboard)

        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

        url = f"{grafana_url.rstrip('/')}/api/dashboards/db"
        response = requests.post(url, json=payload, headers=headers, timeout=30)

        if response.status_code == 200:
            result = response.json()
            return True, f"Deployed: {result.get('url', 'success')}"
        error = response.json().get("message", response.text)
        return False, f"Failed ({response.status_code}): {error}"

    except FileNotFoundError:
        return False, f"File not found: {dashboard_path}"
    except (json.JSONDecodeError, requests.exceptions.RequestException, Exception) as e:
        error_type = type(e).__name__
        return False, f"{error_type}: {e}"


def deploy_all_dashboards(
    grafana_url: str,
    api_token: str,
    dashboard_dir: Path | None = None,
) -> dict[str, tuple[bool, str]]:
    """Deploy all configured dashboards to Grafana.

    Args:
        grafana_url: Base URL of Grafana instance.
        api_token: API token for authentication.
        dashboard_dir: Optional override for dashboard directory.

    Returns:
        Dictionary mapping dashboard names to (success, message) tuples.
    """
    results: dict[str, tuple[bool, str]] = {}
    base_dir = dashboard_dir or DASHBOARD_DIR

    for filename in DASHBOARD_FILES:
        filepath = base_dir / filename
        success, message = deploy_dashboard(grafana_url, api_token, filepath)
        results[filename] = (success, message)

        status = "✅" if success else "❌"
        logger.info(f"{status} {filename}: {message}")

    return results


def check_grafana_connection(grafana_url: str, api_token: str) -> tuple[bool, str]:
    """Check if Grafana is accessible and token is valid.

    Args:
        grafana_url: Base URL of Grafana instance.
        api_token: API token for authentication.

    Returns:
        Tuple of (success, message).
    """
    if requests is None:
        return False, "requests library not installed"

    try:
        headers = {"Authorization": f"Bearer {api_token}"}
        url = f"{grafana_url.rstrip('/')}/api/org"
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            org = response.json()
            return True, f"Connected to org: {org.get('name', 'unknown')}"
        if response.status_code == 401:
            return False, "Invalid API token"
        return False, f"Unexpected response: {response.status_code}"

    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {grafana_url}"
    except Exception as e:
        return False, f"Error: {e}"


def main() -> int:
    """Main entry point for dashboard deployment."""
    parser = argparse.ArgumentParser(description="Deploy Grafana dashboards from JSON files")
    parser.add_argument(
        "--url",
        default=DEFAULT_GRAFANA_URL,
        help=f"Grafana URL (default: {DEFAULT_GRAFANA_URL})",
    )
    parser.add_argument(
        "--token",
        default=DEFAULT_API_TOKEN,
        help="Grafana API token (or set GRAFANA_API_TOKEN env var)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check connection, don't deploy",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.token:
        logger.error("No API token provided. Set GRAFANA_API_TOKEN or use --token")
        return 1

    # Check connection
    logger.info(f"Checking connection to {args.url}...")
    connected, msg = check_grafana_connection(args.url, args.token)

    if not connected:
        logger.error(f"Connection failed: {msg}")
        return 1

    logger.info(f"✅ {msg}")

    if args.check_only:
        return 0

    # Deploy dashboards
    logger.info(f"Deploying {len(DASHBOARD_FILES)} dashboards...")
    results = deploy_all_dashboards(args.url, args.token)

    # Summary
    success_count = sum(1 for success, _ in results.values() if success)
    total_count = len(results)

    logger.info(f"\n📊 Deployment Summary: {success_count}/{total_count} successful")

    return 0 if success_count == total_count else 1


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def _test_load_dashboard() -> bool:
    """Test loading a dashboard JSON file."""
    dashboard_path = DASHBOARD_DIR / "ancestry_overview.json"
    if not dashboard_path.exists():
        return True  # Skip if file doesn't exist

    dashboard = load_dashboard(dashboard_path)
    return "panels" in dashboard


def _test_prepare_payload() -> bool:
    """Test preparing dashboard payload for API."""
    sample_dashboard = {
        "__requires": [{"type": "panel", "id": "stat"}],
        "panels": [{"id": 1, "type": "stat"}],
        "title": "Test Dashboard",
    }

    payload = prepare_dashboard_payload(sample_dashboard)

    # __requires should be removed
    if "__requires" in payload.get("dashboard", {}):
        return False

    # Should have required fields
    if "dashboard" not in payload:
        return False
    return "overwrite" in payload


def _test_dashboard_files_exist() -> bool:
    """Test that all expected dashboard files exist."""
    for filename in DASHBOARD_FILES:
        filepath = DASHBOARD_DIR / filename
        if not filepath.exists():
            logger.warning(f"Dashboard file missing: {filepath}")
            return False
    return True


def _test_dashboards_valid_json() -> bool:
    """Test that all dashboard files are valid JSON."""
    for filename in DASHBOARD_FILES:
        filepath = DASHBOARD_DIR / filename
        if not filepath.exists():
            continue
        try:
            load_dashboard(filepath)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Invalid JSON in {filename}: {e}")
            return False
    return True


def module_tests() -> bool:
    """Run module tests."""
    suite = TestSuite("Deploy Dashboards", __file__)

    suite.run_test("Load dashboard JSON", _test_load_dashboard)
    suite.run_test("Prepare API payload", _test_prepare_payload)
    suite.run_test("Dashboard files exist", _test_dashboard_files_exist)
    suite.run_test("Dashboards are valid JSON", _test_dashboards_valid_json)

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Standard test runner entry point."""
    return module_tests()


if __name__ == "__main__":
    # Check if running as tests (via RUN_MODULE_TESTS env var or --test flag)
    if os.environ.get("RUN_MODULE_TESTS") == "1" or "--test" in sys.argv:
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    else:
        sys.exit(main())
