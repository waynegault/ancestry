"""
Grafana Setup Checker and Automated Installer
Verifies Grafana installation and triggers automated setup if needed
"""

# === PATH SETUP FOR PACKAGE IMPORTS ===
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import logging
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

logger = logging.getLogger(__name__)

# Paths
GRAFANA_INSTALL_PATH = Path(r"C:\Program Files\GrafanaLabs\grafana")
GRAFANA_CLI = GRAFANA_INSTALL_PATH / "bin" / "grafana-cli.exe"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETUP_SCRIPT = PROJECT_ROOT / "docs" / "grafana" / "setup_grafana.ps1"


def is_grafana_installed() -> bool:
    """Check if Grafana is installed by looking for grafana-cli.exe"""
    return GRAFANA_CLI.exists()


def is_grafana_running() -> bool:
    """Check if Grafana service is accessible on port 3000"""
    try:
        response = urllib.request.urlopen("http://localhost:3000/api/health", timeout=2)
        return response.status == 200
    except Exception:
        return False


def are_plugins_installed() -> tuple[bool, bool]:
    """
    Check if required plugins are installed
    Returns: (sqlite_installed, plugins_accessible)
    """
    if not is_grafana_installed():
        return False, False

    plugins_dir = GRAFANA_INSTALL_PATH / "data" / "plugins"

    # Check for SQLite plugin
    sqlite_plugin = plugins_dir / "frser-sqlite-datasource"
    sqlite_installed = sqlite_plugin.exists()

    return sqlite_installed, True


def check_grafana_status() -> dict[str, bool]:
    """
    Comprehensive Grafana status check
    Returns dict with installation, service, and plugin status
    """
    status = {
        "installed": is_grafana_installed(),
        "running": False,
        "sqlite_plugin": False,
        "plugins_accessible": False,
        "ready": False,
    }

    if status["installed"]:
        status["running"] = is_grafana_running()
        status["sqlite_plugin"], status["plugins_accessible"] = are_plugins_installed()
        status["ready"] = status["running"] and status["sqlite_plugin"]

    return status


def run_automated_setup(silent: bool = False) -> bool:
    """
    Execute the PowerShell setup script with admin privileges

    Args:
        silent: If True, suppress output

    Returns:
        True if setup succeeded, False otherwise
    """
    if not SETUP_SCRIPT.exists():
        logger.error(f"Setup script not found: {SETUP_SCRIPT}")
        return False

    logger.info("Launching Grafana automated setup...")

    try:
        # Determine available PowerShell executable
        powershell_cmd = "pwsh" if shutil.which("pwsh") else "powershell"

        # PowerShell command to run script as administrator
        ps_args = [
            powershell_cmd,
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            f"Start-Process {powershell_cmd} -Verb RunAs -ArgumentList "
            f"'-ExecutionPolicy Bypass -File \"{SETUP_SCRIPT}\" "
            f"{'-Silent' if silent else ''}' -Wait",
        ]

        result = subprocess.run(
            ps_args,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,  # 5 minute timeout
        )

        if result.returncode == 0:
            logger.info("✓ Grafana setup completed successfully")
            return True
        logger.warning(f"Setup script returned non-zero exit code: {result.returncode}")
        if result.stderr:
            logger.debug(f"Setup errors: {result.stderr}")
        return False

    except subprocess.TimeoutExpired:
        logger.error("Setup script timeout (5 minutes)")
        return False
    except Exception as e:
        logger.error(f"Failed to run setup script: {e}")
        return False


def prompt_user_for_setup() -> bool:
    """
    Interactive prompt asking user if they want to run automated setup
    Returns True if user agrees, False otherwise
    """
    print("\n" + "=" * 70)
    print("📊 GRAFANA METRICS VISUALIZATION")
    print("=" * 70)
    print("\n⚠️  Grafana is not fully configured")
    print("\nAutomated setup will:")
    print("  • Download and install Grafana")
    print("  • Install SQLite and Node Graph plugins")
    print("  • Configure Prometheus and SQLite data sources")
    print("  • Import 3 dashboards (Performance, Genealogy, Code Quality)")
    print("  • Requires Administrator privileges")
    print("\nEstimated time: 5-10 minutes")
    print("\n" + "=" * 70)

    response = input("\nRun automated setup now? (y/n): ").strip().lower()
    return response in {'y', 'yes'}


def ensure_grafana_ready(auto_setup: bool = False, silent: bool = False) -> bool:
    """
    Main function to ensure Grafana is ready for use

    Args:
        auto_setup: If True, automatically run setup without prompting
        silent: If True, suppress output during setup

    Returns:
        True if Grafana is ready (or user declined setup), False if setup failed
    """
    status = check_grafana_status()

    # If everything is ready, return success
    if status["ready"]:
        logger.debug("✓ Grafana is installed and ready")
        return True

    # Log what's missing
    if not status["installed"]:
        logger.info("Grafana is not installed")
    elif not status["running"]:
        logger.info("Grafana service is not running")
    elif not status["sqlite_plugin"]:
        logger.info("SQLite plugin is not installed")

    # Decide whether to run setup
    should_setup = auto_setup or (not silent and prompt_user_for_setup())

    if should_setup:
        success = run_automated_setup(silent=silent)
        if success:
            # Re-check status after setup
            import time

            time.sleep(5)  # Give services time to stabilize
            new_status = check_grafana_status()
            return new_status["ready"]
        logger.warning("Grafana setup failed. You can run it manually:")
        logger.warning(f"  PowerShell (as Admin): {SETUP_SCRIPT}")
        return False
    logger.info("Skipping Grafana setup. Run manually when ready:")
    logger.info(f"  PowerShell (as Admin): {SETUP_SCRIPT}")
    return True  # Return True to not block app startup


def get_status_message() -> str:
    """
    Get a human-readable status message with emoji indicator
    Returns status string suitable for menu display
    """
    status = check_grafana_status()

    if status["ready"]:
        return "✅ Grafana Ready (http://localhost:3000)"
    if status["installed"] and status["running"]:
        return "⚠️  Grafana Running (plugins need setup)"
    if status["installed"]:
        return "⚠️  Grafana Installed (service not running)"
    return "❌ Grafana Not Installed (run setup)"


def ensure_dashboards_imported(force: bool = False) -> bool:
    """
    Check if dashboards are imported and import them if missing or forced
    Returns True if all dashboards are present or successfully imported
    """
    if not is_grafana_running():
        logger.debug("Grafana not running, skipping dashboard check")
        return False

    dashboards_dir = PROJECT_ROOT / "docs" / "grafana"
    required_dashboards = [
        ("ancestry-overview", "ancestry_overview.json"),
        ("ancestry-performance", "system_performance.json"),
        ("ancestry-genealogy", "genealogy_insights.json"),
        ("ancestry-code-quality", "code_quality.json"),
    ]

    import base64

    # Check and import each dashboard
    base64_auth = base64.b64encode(b"admin:ancestry").decode("ascii")
    headers = {
        "Authorization": f"Basic {base64_auth}",
        "Content-Type": "application/json",
    }

    imported_count = 0
    for uid, filename in required_dashboards:
        should_import = force
        if not should_import:
            try:
                # Check if dashboard exists
                check_req = urllib.request.Request(f"http://localhost:3000/api/dashboards/uid/{uid}", headers=headers)
                urllib.request.urlopen(check_req, timeout=2)
                logger.debug(f"Dashboard {uid} already exists")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    should_import = True
                else:
                    logger.debug(f"Error checking dashboard {uid}: {e}")

        if should_import:
            # Dashboard missing or forced update, try to import
            logger.info(f"Importing dashboard: {filename}")
            dashboard_path = dashboards_dir / filename
            if not dashboard_path.exists():
                logger.warning(f"Dashboard file not found: {dashboard_path}")
                continue

            try:
                with dashboard_path.open(encoding="utf-8") as f:
                    dashboard_json = json.load(f)

                import_payload = {
                    "dashboard": dashboard_json,
                    "overwrite": True,
                    "inputs": [
                        {
                            "name": "DS_PROMETHEUS",
                            "type": "datasource",
                            "pluginId": "prometheus",
                            "value": "Prometheus",
                        },
                        {
                            "name": "DS_SQLITE",
                            "type": "datasource",
                            "pluginId": "frser-sqlite-datasource",
                            "value": "SQLite",
                        },
                    ],
                }

                import_req = urllib.request.Request(
                    "http://localhost:3000/api/dashboards/import",
                    data=json.dumps(import_payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                urllib.request.urlopen(import_req, timeout=5)
                logger.info(f"✓ Successfully imported {filename}")
                imported_count += 1
            except Exception as import_error:
                logger.warning(f"Failed to import {filename}: {import_error}")

    if imported_count > 0:
        logger.info(f"Imported {imported_count} dashboard(s)")

    return True


def _test_check_grafana_status_handles_installation_flag() -> None:
    with (
        mock.patch(f"{__name__}.is_grafana_installed", return_value=False),
        mock.patch(f"{__name__}.is_grafana_running") as running_mock,
        mock.patch(f"{__name__}.are_plugins_installed") as plugins_mock,
    ):
        status = check_grafana_status()

    assert status == {
        "installed": False,
        "running": False,
        "sqlite_plugin": False,
        "plugins_accessible": False,
        "ready": False,
    }
    running_mock.assert_not_called()
    plugins_mock.assert_not_called()


def _test_check_grafana_status_ready_state() -> None:
    with (
        mock.patch(f"{__name__}.is_grafana_installed", return_value=True),
        mock.patch(f"{__name__}.is_grafana_running", return_value=True),
        mock.patch(f"{__name__}.are_plugins_installed", return_value=(True, True)),
    ):
        status = check_grafana_status()

    assert status["ready"] is True
    assert status["sqlite_plugin"] is True


def _test_get_status_message_variants() -> None:
    ready_status = {
        "installed": True,
        "running": True,
        "sqlite_plugin": True,
        "plugins_accessible": True,
        "ready": True,
    }
    running_status = {**ready_status, "ready": False}
    installed_status = {
        "installed": True,
        "running": False,
        "sqlite_plugin": False,
        "plugins_accessible": False,
        "ready": False,
    }
    missing_status = {
        "installed": False,
        "running": False,
        "sqlite_plugin": False,
        "plugins_accessible": False,
        "ready": False,
    }

    cases = [
        (ready_status, "✅ Grafana Ready (http://localhost:3000)"),
        (running_status, "⚠️  Grafana Running (plugins need setup)"),
        (installed_status, "⚠️  Grafana Installed (service not running)"),
        (missing_status, "❌ Grafana Not Installed (run setup)"),
    ]

    for provided_status, expected in cases:
        with mock.patch(f"{__name__}.check_grafana_status", return_value=provided_status):
            assert get_status_message() == expected


def _test_ensure_grafana_ready_runs_setup_when_needed() -> None:
    initial_status = {
        "installed": True,
        "running": False,
        "sqlite_plugin": False,
        "plugins_accessible": False,
        "ready": False,
    }
    final_status = {**initial_status, "running": True, "sqlite_plugin": True, "plugins_accessible": True, "ready": True}

    with (
        mock.patch(f"{__name__}.check_grafana_status", side_effect=[initial_status, final_status]) as status_mock,
        mock.patch(f"{__name__}.run_automated_setup", return_value=True) as setup_mock,
        mock.patch("time.sleep", return_value=None) as sleep_mock,
    ):
        assert ensure_grafana_ready(auto_setup=True, silent=True) is True

    setup_mock.assert_called_once_with(silent=True)
    sleep_mock.assert_called_once()
    assert status_mock.call_count == 2


def _test_ensure_grafana_ready_skips_when_user_declines() -> None:
    status = {
        "installed": True,
        "running": False,
        "sqlite_plugin": False,
        "plugins_accessible": False,
        "ready": False,
    }

    with (
        mock.patch(f"{__name__}.check_grafana_status", return_value=status),
        mock.patch(f"{__name__}.prompt_user_for_setup", return_value=False) as prompt_mock,
    ):
        assert ensure_grafana_ready(auto_setup=False, silent=False) is True

    prompt_mock.assert_called_once()


def grafana_checker_module_tests() -> bool:
    from testing.test_framework import TestSuite

    suite = TestSuite("grafana_checker", "grafana_checker.py")

    suite.run_test(
        "Status short-circuits when not installed",
        _test_check_grafana_status_handles_installation_flag,
        "Verifies that check_grafana_status avoids network calls when Grafana is not installed.",
    )

    suite.run_test(
        "Status ready state",
        _test_check_grafana_status_ready_state,
        "Ensures ready flag is set only when running and plugins are available.",
    )

    suite.run_test(
        "Status message variants",
        _test_get_status_message_variants,
        "Confirms get_status_message returns the correct text for each state.",
    )

    suite.run_test(
        "Automated setup path",
        _test_ensure_grafana_ready_runs_setup_when_needed,
        "Validates ensure_grafana_ready triggers setup and re-checks status when auto_setup is True.",
    )

    suite.run_test(
        "User decline path",
        _test_ensure_grafana_ready_skips_when_user_declines,
        "Ensures ensure_grafana_ready respects user choice and does not block startup.",
    )

    return suite.finish_suite()


# Use centralized test runner utility from test_utilities
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(grafana_checker_module_tests)


if __name__ == "__main__":
    # Run tests when executed directly
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
