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
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

import requests

logger = logging.getLogger(__name__)

# Paths
GRAFANA_INSTALL_PATH = Path(r"C:\Program Files\GrafanaLabs\grafana")
GRAFANA_CLI = GRAFANA_INSTALL_PATH / "bin" / "grafana-cli.exe"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETUP_SCRIPT = PROJECT_ROOT / "docs" / "grafana" / "setup_grafana.ps1"


def _grafana_base() -> str:
    return os.getenv("GRAFANA_BASE_URL", "http://localhost:3000")


def _grafana_auth() -> tuple[str, str]:
    return os.getenv("GRAFANA_USER", "admin"), os.getenv("GRAFANA_PASSWORD", "admin")


def _grafana_api_auth_headers() -> dict[str, str]:
    """Return auth headers for Grafana HTTP API.

    Supports either basic auth (GRAFANA_USER/GRAFANA_PASSWORD) or a bearer token
    (GRAFANA_API_TOKEN).
    """

    token = os.getenv("GRAFANA_API_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}

    import base64

    grafana_user, grafana_pass = _grafana_auth()
    base64_auth = base64.b64encode(f"{grafana_user}:{grafana_pass}".encode("ascii")).decode("ascii")
    return {"Authorization": f"Basic {base64_auth}"}


def _log_grafana_auth_failure(grafana_base: str) -> None:
    logger.warning(
        "Grafana API auth failed for %s. Set GRAFANA_USER/GRAFANA_PASSWORD or GRAFANA_API_TOKEN.",
        grafana_base,
    )


def is_grafana_installed() -> bool:
    """Check if Grafana is installed by looking for grafana-cli.exe"""
    return GRAFANA_CLI.exists()


def is_grafana_running() -> bool:
    """Check if Grafana service is accessible on port 3000"""
    try:
        grafana_base = _grafana_base()
        response = urllib.request.urlopen(f"{grafana_base}/api/health", timeout=2)
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

    grafana_base = os.getenv("GRAFANA_BASE_URL", "http://localhost:3000")

    if status["ready"]:
        return f"✅ Grafana Ready ({grafana_base})"
    if status["installed"] and status["running"]:
        return "⚠️  Grafana Running (plugins need setup)"
    if status["installed"]:
        return "⚠️  Grafana Installed (service not running)"
    return "❌ Grafana Not Installed (run setup)"


def _upsert_datasource(grafana_base: str, payload: Mapping[str, object]) -> bool:
    payload_dict = dict(payload)
    headers = {"Content-Type": "application/json", **_grafana_api_auth_headers()}
    session = requests.Session()
    success = False
    try:
        existing = session.get(
            f"{grafana_base}/api/datasources/name/{payload_dict['name']}",
            headers=headers,
            timeout=5,
        )
        if existing.status_code in {401, 403}:
            _log_grafana_auth_failure(grafana_base)
            return False
        if existing.status_code == 200:
            ds_id = existing.json().get("id")
            if ds_id:
                updated = session.put(
                    f"{grafana_base}/api/datasources/{ds_id}",
                    headers=headers,
                    data=json.dumps(payload_dict),
                    timeout=5,
                )
                if updated.status_code in {401, 403}:
                    _log_grafana_auth_failure(grafana_base)
                    return False
                if updated.status_code in {200, 201}:
                    success = True
                else:
                    logger.debug("Grafana data source update failed (%s): %s", updated.status_code, updated.text)
                return success

        if existing.status_code not in {404, 200}:
            logger.debug(
                "Grafana data source lookup failed (%s): %s",
                existing.status_code,
                existing.text,
            )
        else:
            created = session.post(
                f"{grafana_base}/api/datasources",
                headers=headers,
                data=json.dumps(payload_dict),
                timeout=5,
            )
            if created.status_code in {401, 403}:
                _log_grafana_auth_failure(grafana_base)
                return False
            if created.status_code in {200, 201, 409}:
                success = True
            else:
                logger.debug("Grafana data source create failed (%s): %s", created.status_code, created.text)
    except Exception as exc:
        logger.debug("Data source upsert failed: %s", exc)
        success = False

    return success


def ensure_data_sources_configured(
    prometheus_url: str | None = None,
    sqlite_path: str | None = None,
) -> bool:
    """Ensure required Grafana data sources exist (Prometheus + SQLite)."""

    if not is_grafana_running():
        logger.debug("Grafana not running; skipping data source configuration")
        return False

    grafana_base = _grafana_base()

    prom_url = prometheus_url or os.getenv("GRAFANA_PROM_URL") or os.getenv("PROMETHEUS_URL", "http://localhost:9091")
    sqlite_db = sqlite_path or os.getenv("GRAFANA_SQLITE_PATH") or str(PROJECT_ROOT / "Data" / "ancestry.db")

    success = True

    prom_payload = {
        "name": "Prometheus",
        "type": "prometheus",
        "url": prom_url,
        "access": "proxy",
        "basicAuth": False,
        "isDefault": True,
    }
    if not _upsert_datasource(grafana_base, prom_payload):
        success = False

    if Path(sqlite_db).exists():
        sqlite_payload = {
            "name": "SQLite",
            "type": "frser-sqlite-datasource",
            "url": "",
            "access": "proxy",
            "basicAuth": False,
            "jsonData": {"path": sqlite_db},
        }
        if not _upsert_datasource(grafana_base, sqlite_payload):
            success = False
    else:
        logger.debug("SQLite database not found at %s; skipping SQLite data source", sqlite_db)

    return success


def _dashboard_should_import(
    *,
    grafana_base: str,
    headers: Mapping[str, str],
    uid: str,
    force: bool,
) -> tuple[bool, bool]:
    if force:
        return True, True

    try:
        check_req = urllib.request.Request(f"{grafana_base}/api/dashboards/uid/{uid}", headers=dict(headers))
        urllib.request.urlopen(check_req, timeout=2)
        logger.debug("Dashboard %s already exists", uid)
        return False, True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return True, True
        if e.code in {401, 403}:
            _log_grafana_auth_failure(grafana_base)
            return False, False
        logger.debug("Error checking dashboard %s: %s", uid, e)
        return False, False
    except Exception as e:
        logger.debug("Error checking dashboard %s: %s", uid, e)
        return False, False


def _import_dashboard(
    *,
    grafana_base: str,
    headers: Mapping[str, str],
    dashboards_dir: Path,
    filename: str,
) -> bool:
    logger.info("Importing dashboard: %s", filename)
    dashboard_path = dashboards_dir / filename
    if not dashboard_path.exists():
        logger.warning("Dashboard file not found: %s", dashboard_path)
        return False

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
            f"{grafana_base}/api/dashboards/import",
            data=json.dumps(import_payload).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        urllib.request.urlopen(import_req, timeout=5)
        logger.info("✓ Successfully imported %s", filename)
        return True
    except urllib.error.HTTPError as import_error:
        if import_error.code in {401, 403}:
            _log_grafana_auth_failure(grafana_base)
        logger.warning("Failed to import %s: %s", filename, import_error)
        return False
    except Exception as import_error:
        logger.warning("Failed to import %s: %s", filename, import_error)
        return False


def ensure_dashboards_imported(force: bool = False) -> bool:
    """Check if dashboards are imported and import them if missing or forced.

    Returns True if all dashboards are present or successfully imported.
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
        ("ancestry-database", "database_summary.json"),
    ]

    grafana_base = _grafana_base()
    headers = {
        **_grafana_api_auth_headers(),
        "Content-Type": "application/json",
    }

    success = True
    imported_count = 0
    for uid, filename in required_dashboards:
        should_import, check_ok = _dashboard_should_import(
            grafana_base=grafana_base,
            headers=headers,
            uid=uid,
            force=force,
        )
        if not check_ok:
            success = False

        if should_import and _import_dashboard(
            grafana_base=grafana_base,
            headers=headers,
            dashboards_dir=dashboards_dir,
            filename=filename,
        ):
            imported_count += 1
        elif should_import:
            success = False

    if imported_count > 0:
        logger.info("Imported %s dashboard(s)", imported_count)

    return success


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

    grafana_base = os.getenv("GRAFANA_BASE_URL", "http://localhost:3000")

    cases = [
        (ready_status, f"✅ Grafana Ready ({grafana_base})"),
        (running_status, "⚠️  Grafana Running (plugins need setup)"),
        (installed_status, "⚠️  Grafana Installed (service not running)"),
        (missing_status, "❌ Grafana Not Installed (run setup)"),
    ]

    for provided_status, expected in cases:
        with mock.patch(f"{__name__}.check_grafana_status", return_value=provided_status):
            assert get_status_message() == expected


def _test_upsert_datasource_fails_on_auth_error() -> None:
    fake_session = mock.Mock()
    fake_response = mock.Mock(status_code=401, text="Invalid username or password")
    fake_session.get.return_value = fake_response

    with mock.patch(f"{__name__}.requests.Session", return_value=fake_session):
        assert _upsert_datasource("http://localhost:3000", {"name": "Prometheus"}) is False


def _test_ensure_dashboards_imported_fails_on_auth_error() -> None:
    from email.message import Message

    auth_error = urllib.error.HTTPError(
        url="http://localhost:3000/api/dashboards/uid/ancestry-overview",
        code=401,
        msg="Unauthorized",
        hdrs=Message(),
        fp=None,
    )

    with (
        mock.patch(f"{__name__}.is_grafana_running", return_value=True),
        mock.patch(f"{__name__}._grafana_base", return_value="http://localhost:3000"),
        mock.patch(f"{__name__}.urllib.request.urlopen", side_effect=auth_error),
    ):
        assert ensure_dashboards_imported(force=False) is False


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
        "Datasource upsert fails on auth error",
        _test_upsert_datasource_fails_on_auth_error,
        "Ensures Grafana API datasource configuration returns False on 401/403 instead of reporting success.",
    )

    suite.run_test(
        "Dashboard import fails on auth error",
        _test_ensure_dashboards_imported_fails_on_auth_error,
        "Ensures dashboard checks return False on 401/403 so the CLI can prompt for credentials/token.",
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
