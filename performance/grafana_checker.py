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

import functools
import json
import logging
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any
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


@functools.lru_cache(maxsize=1)
def _grafana_api_auth_headers() -> dict[str, str]:
    """Return auth headers for Grafana HTTP API.

    Supports either basic auth (GRAFANA_USER/GRAFANA_PASSWORD) or a bearer token
    (GRAFANA_API_TOKEN).
    """
    token = os.getenv("GRAFANA_API_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}

    import base64

    # Try configured credentials first
    grafana_user, grafana_pass = _grafana_auth()
    candidates = [(grafana_user, grafana_pass)]

    # If using defaults, also try 'ancestry' password
    if grafana_user == "admin" and grafana_pass == "admin":
        candidates.append(("admin", "ancestry"))

    grafana_base = _grafana_base()

    for user, pwd in candidates:
        base64_auth = base64.b64encode(f"{user}:{pwd}".encode("ascii")).decode("ascii")
        headers = {"Authorization": f"Basic {base64_auth}"}

        # Verify credentials
        try:
            resp = requests.get(f"{grafana_base}/api/org", headers=headers, timeout=2)
            if resp.status_code == 200:
                return headers
        except Exception:
            continue

    # Fallback to default if all fail (will likely fail later)
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

    logger.info(f"Launching Grafana automated setup from: {SETUP_SCRIPT}")

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
        if result.stdout:
            print("--- PowerShell Output ---")
            print(result.stdout)
            print("-------------------------")
        if result.stderr:
            print("--- PowerShell Error ---")
            print(result.stderr)
            print("------------------------")
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


def reset_admin_password(password: str = "ancestry") -> bool:
    """Reset Grafana admin password using grafana-cli.

    Args:
        password: New password for the admin account.

    Returns:
        True if password was reset successfully.
    """
    if not GRAFANA_CLI.exists():
        logger.error(f"grafana-cli not found: {GRAFANA_CLI}")
        return False

    logger.info(f"Resetting Grafana admin password to '{password}'...")
    try:
        result = subprocess.run(
            [str(GRAFANA_CLI), "admin", "reset-admin-password", password],
            cwd=str(GRAFANA_INSTALL_PATH),
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(result.stdout.strip())
        logger.info("Password reset successfully.")
        # Clear cached auth headers since credentials changed
        _grafana_api_auth_headers.cache_clear()
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to reset password: {e}")
        if e.stdout:
            logger.error(e.stdout)
        if e.stderr:
            logger.error(e.stderr)
        return False


def delete_all_datasources() -> bool:
    """Delete all Grafana datasources.

    Useful for a clean slate before re-deploying datasource configurations.

    Returns:
        True if all datasources were deleted (or none existed).
    """
    grafana_base = _grafana_base()
    headers = {"Content-Type": "application/json", **_grafana_api_auth_headers()}

    try:
        resp = requests.get(f"{grafana_base}/api/datasources", headers=headers, timeout=10)
        if resp.status_code != 200:
            logger.error(f"Failed to list datasources: {resp.status_code}")
            return False

        datasources = resp.json()
        if not datasources:
            logger.info("No datasources to delete.")
            return True

        all_ok = True
        for ds in datasources:
            name = ds.get("name", "?")
            uid = ds.get("uid")
            ds_id = ds.get("id")
            logger.info(f"Deleting datasource: {name} (UID: {uid})")

            # Try UID first, fall back to numeric ID
            del_resp = requests.delete(f"{grafana_base}/api/datasources/uid/{uid}", headers=headers, timeout=10)
            if del_resp.status_code != 200 and ds_id:
                del_resp = requests.delete(f"{grafana_base}/api/datasources/{ds_id}", headers=headers, timeout=10)

            if del_resp.status_code == 200:
                logger.info(f"  Deleted: {name}")
            else:
                logger.error(f"  Failed to delete {name}: {del_resp.status_code} {del_resp.text}")
                all_ok = False

        return all_ok
    except Exception as exc:
        logger.error(f"Error deleting datasources: {exc}")
        return False


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

    prom_url = prometheus_url or os.getenv("GRAFANA_PROM_URL") or os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    sqlite_db = sqlite_path or os.getenv("GRAFANA_SQLITE_PATH") or str(PROJECT_ROOT / "Data" / "ancestry.db")

    success = True

    prom_payload = {
        "name": "Prometheus",
        "type": "prometheus",
        "uid": "ancestry-prometheus",
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
            "uid": "ancestry-sqlite",
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


def _sqlite_set_default(container: dict[str, Any], key: str, value: Any, allow_default: bool = False) -> bool:
    existing = container.get(key)
    if existing is None or (allow_default and existing == "default"):
        container[key] = value
        return True
    return False


def _patch_sqlite_target(target: dict[str, Any], sqlite_ds: dict[str, Any]) -> bool:
    updated = _sqlite_set_default(target, "datasource", sqlite_ds, allow_default=True)
    raw_sql = target.get("rawSql")
    if not raw_sql:
        return updated

    updated |= _sqlite_set_default(target, "queryText", raw_sql)
    updated |= _sqlite_set_default(target, "rawQueryText", raw_sql)
    updated |= _sqlite_set_default(target, "rawQuery", True)
    updated |= _sqlite_set_default(target, "queryType", target.get("format", "table") or "table")
    updated |= _sqlite_set_default(target, "timeColumns", [])
    return updated


def _patch_sqlite_panel(panel: dict[str, Any], sqlite_ds: dict[str, Any]) -> bool:
    modified = _sqlite_set_default(panel, "datasource", sqlite_ds, allow_default=True)

    for target in panel.get("targets", []):
        if _patch_sqlite_target(target, sqlite_ds):
            modified = True

    for child in panel.get("panels", []) or []:
        if _patch_sqlite_panel(child, sqlite_ds):
            modified = True

    return modified


def _patch_sqlite_targets(dashboard: dict[str, Any]) -> bool:
    """Ensure SQLite panels carry required plugin fields.

    The frser-sqlite plugin ignores `rawSql` unless `rawQuery` and `queryText`
    are present. We also seed `rawQueryText`, `queryType`, and `timeColumns`
    to mirror exported dashboards so panels render without manual edits.
    """

    sqlite_ds = {"type": "frser-sqlite-datasource", "uid": "ancestry-sqlite"}
    modified = False

    for panel in dashboard.get("panels", []):
        if _patch_sqlite_panel(panel, sqlite_ds):
            modified = True

    return modified


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

        _patch_sqlite_targets(dashboard_json)

        import_payload = {
            "dashboard": dashboard_json,
            "overwrite": True,
            # Inputs are no longer needed as we use fixed UIDs in the dashboard JSON
            "inputs": [],
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

    # Make sure the SQLite plugin is enabled so code-quality dashboards can run queries
    _ensure_sqlite_plugin_enabled()

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

    # Ensure code_structure table is populated for the Code Quality dashboard
    _ensure_code_structure_table()

    # Some SQLite panels require the plugin-specific queryText field; patch dashboards if missing
    _ensure_sqlite_querytext_fields([uid for uid, _ in required_dashboards])

    return success


def _ensure_sqlite_plugin_enabled() -> None:
    """Enable the SQLite plugin if it is installed but disabled."""

    grafana_base = _grafana_base()
    headers = _grafana_api_auth_headers()

    try:
        resp = requests.get(
            f"{grafana_base}/api/plugins/frser-sqlite-datasource/settings",
            headers=headers,
            timeout=5,
        )
        if resp.status_code != 200:
            logger.debug("Unable to query SQLite plugin settings (status=%s)", resp.status_code)
            return

        settings = resp.json()
        if settings.get("enabled") is True:
            return

        enable_resp = requests.post(
            f"{grafana_base}/api/plugins/frser-sqlite-datasource/settings",
            headers={**headers, "Content-Type": "application/json"},
            json={"enabled": True},
            timeout=5,
        )
        if enable_resp.status_code == 200:
            logger.info("Enabled frser-sqlite-datasource plugin for Grafana")
        else:
            logger.debug(
                "Failed to enable SQLite plugin (status=%s): %s",
                enable_resp.status_code,
                enable_resp.text,
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Error enabling SQLite plugin: %s", exc)


def _ensure_sqlite_querytext_fields(dashboard_uids: list[str]) -> None:
    """Ensure Grafana panels include queryText for SQLite queries.

    The frser-sqlite plugin expects a `queryText` field. Some imports only retain
    `rawSql`, which results in empty panels. This patches dashboards in-place to
    mirror `rawSql` into `queryText` when missing and forces overwrite via API.
    """

    grafana_base = _grafana_base()
    headers = _grafana_api_auth_headers()
    json_headers = {**headers, "Content-Type": "application/json"}

    for uid in dashboard_uids:
        try:
            resp = requests.get(f"{grafana_base}/api/dashboards/uid/{uid}", headers=headers, timeout=5)
            if resp.status_code != 200:
                logger.debug("Skipping queryText patch for %s (status=%s)", uid, resp.status_code)
                continue

            payload = resp.json()
            dashboard = payload.get("dashboard")
            if not dashboard:
                continue

            if not _patch_sqlite_targets(dashboard):
                continue

            save_payload = {"dashboard": dashboard, "overwrite": True}
            save_resp = requests.post(
                f"{grafana_base}/api/dashboards/db",
                headers=json_headers,
                data=json.dumps(save_payload),
                timeout=5,
            )
            if save_resp.status_code in {200, 202}:
                logger.info("Patched dashboard %s to include queryText for SQLite panels", uid)
            else:
                logger.debug(
                    "Failed to patch dashboard %s (status=%s): %s",
                    uid,
                    save_resp.status_code,
                    save_resp.text,
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Error patching dashboard %s: %s", uid, exc)


def _ensure_code_structure_table() -> None:
    """Ensure the code_structure table exists and is populated from code_graph.json."""
    db_path = PROJECT_ROOT / "Data" / "ancestry.db"
    json_path = PROJECT_ROOT / "docs" / "code_graph.json"

    if not json_path.exists():
        logger.warning("code_graph.json not found at %s", json_path)
        return

    import sqlite3

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='code_structure'")
        table_exists = cursor.fetchone() is not None

        # Always refresh the table to ensure it matches the JSON
        # This is fast enough for 7k rows and ensures consistency
        if table_exists:
            cursor.execute("DROP TABLE code_structure")

        logger.info("Populating code_structure table from %s", json_path)

        with Path(json_path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("nodes", [])

        cursor.execute("""
            CREATE TABLE code_structure (
                id TEXT PRIMARY KEY,
                type TEXT,
                name TEXT,
                path TEXT,
                start_line INTEGER,
                end_line INTEGER,
                complexity INTEGER,
                docstring_length INTEGER,
                function_count INTEGER,
                class_count INTEGER,
                import_count INTEGER,
                data TEXT
            )
        """)

        # Batch insert for performance
        batch_data: list[tuple[Any, ...]] = []
        for node in nodes:
            batch_data.append(
                (
                    node.get("id"),
                    node.get("type"),
                    node.get("name"),
                    node.get("path"),  # Map path from JSON to path column
                    node.get("start_line"),
                    node.get("end_line"),
                    node.get("complexity", 0),
                    node.get("docstring_length", 0),
                    node.get("function_count", 0),
                    node.get("class_count", 0),
                    node.get("import_count", 0),
                    json.dumps(node),  # Store full node JSON in data column
                )
            )

        cursor.executemany(
            """
            INSERT INTO code_structure (
                id, type, name, path, start_line, end_line,
                complexity, docstring_length, function_count, class_count, import_count, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            batch_data,
        )

        conn.commit()
        logger.info("Successfully populated code_structure table with %d rows", len(nodes))

        conn.close()
    except Exception as e:
        logger.error("Failed to ensure code_structure table: %s", e)


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
