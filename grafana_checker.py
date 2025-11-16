"""
Grafana Setup Checker and Automated Installer
Verifies Grafana installation and triggers automated setup if needed
"""

import logging
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

# Paths
GRAFANA_INSTALL_PATH = r"C:\Program Files\GrafanaLabs\grafana"
GRAFANA_CLI = os.path.join(GRAFANA_INSTALL_PATH, "bin", "grafana-cli.exe")
PROJECT_ROOT = Path(__file__).parent
SETUP_SCRIPT = PROJECT_ROOT / "docs" / "grafana" / "setup_grafana.ps1"


def is_grafana_installed() -> bool:
    """Check if Grafana is installed by looking for grafana-cli.exe"""
    return os.path.exists(GRAFANA_CLI)


def is_grafana_running() -> bool:
    """Check if Grafana service is accessible on port 3000"""
    try:
        response = urllib.request.urlopen("http://localhost:3000/api/health", timeout=2)
        return response.status == 200
    except Exception:
        return False


def are_plugins_installed() -> Tuple[bool, bool]:
    """
    Check if required plugins are installed
    Returns: (sqlite_installed, plugins_accessible)
    """
    if not is_grafana_installed():
        return False, False
    
    plugins_dir = os.path.join(GRAFANA_INSTALL_PATH, "data", "plugins")
    
    # Check for SQLite plugin
    sqlite_plugin = os.path.join(plugins_dir, "frser-sqlite-datasource")
    sqlite_installed = os.path.exists(sqlite_plugin)
    
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
        # PowerShell command to run script as administrator
        ps_args = [
            "powershell.exe",
            "-ExecutionPolicy", "Bypass",
            "-Command",
            f"Start-Process PowerShell -Verb RunAs -ArgumentList "
            f"'-ExecutionPolicy Bypass -File \"{SETUP_SCRIPT}\" "
            f"{'-Silent' if silent else ''}' -Wait"
        ]
        
        result = subprocess.run(
            ps_args,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("✓ Grafana setup completed successfully")
            return True
        else:
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
    print("\n" + "="*70)
    print("📊 GRAFANA METRICS VISUALIZATION")
    print("="*70)
    print("\n⚠️  Grafana is not fully configured")
    print("\nAutomated setup will:")
    print("  • Download and install Grafana")
    print("  • Install SQLite and Node Graph plugins")
    print("  • Configure Prometheus and SQLite data sources")
    print("  • Import 3 dashboards (Performance, Genealogy, Code Quality)")
    print("  • Requires Administrator privileges")
    print("\nEstimated time: 5-10 minutes")
    print("\n" + "="*70)
    
    response = input("\nRun automated setup now? (y/n): ").strip().lower()
    return response in ('y', 'yes')


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
        else:
            logger.warning("Grafana setup failed. You can run it manually:")
            logger.warning(f"  PowerShell (as Admin): {SETUP_SCRIPT}")
            return False
    else:
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
    elif status["installed"] and status["running"]:
        return "⚠️  Grafana Running (plugins need setup)"
    elif status["installed"]:
        return "⚠️  Grafana Installed (service not running)"
    else:
        return "❌ Grafana Not Installed (run setup)"


# Module-level test
def run_comprehensive_tests() -> bool:
    """Test the Grafana checker functionality"""
    print("\n" + "="*70)
    print("GRAFANA STATUS CHECK")
    print("="*70 + "\n")
    
    status = check_grafana_status()
    
    print(f"Grafana Installed:     {'✓' if status['installed'] else '✗'}")
    print(f"Service Running:       {'✓' if status['running'] else '✗'}")
    print(f"SQLite Plugin:         {'✓' if status['sqlite_plugin'] else '✗'}")
    print(f"Plugins Accessible:    {'✓' if status['plugins_accessible'] else '✗'}")
    print(f"\nOverall Status:        {'✅ READY' if status['ready'] else '❌ NOT READY'}")
    
    print(f"\nStatus Message: {get_status_message()}")
    
    print("\n" + "="*70 + "\n")
    
    return True


if __name__ == "__main__":
    # Run tests when executed directly
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
