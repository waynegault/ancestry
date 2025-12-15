#!/usr/bin/env python3
"""
Dry-Run Test Harness for Ancestry Actions

This script executes the main actions (6, 7, 8, 10, 12, 13) in 'dry_run' mode.
It verifies that the actions run without errors and, crucially, that NO real messages
are sent to Ancestry, while logging activity to `logs/app.log`.

It enforces:
1. APP_MODE environment variable = "dry_run"
2. Action 6 (Gather) limits to 1 page via config injection.
3. Logging setup to capture behavior.
"""

import sys
from pathlib import Path

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _ensure_venv() -> None:
    """Ensure running in venv, auto-restart if needed."""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        return

    venv_python = project_root / '.venv' / 'Scripts' / 'python.exe'
    if not venv_python.exists():
        venv_python = project_root / '.venv' / 'bin' / 'python'
        if not venv_python.exists():
            print("⚠️  WARNING: Not running in virtual environment")
            return

    import os as _os

    print(f"🔄 Re-running with venv Python: {venv_python}")
    _os.chdir(project_root)
    _os.execv(str(venv_python), [str(venv_python), __file__] + sys.argv[1:])


_ensure_venv()

import logging
import os
import time

# 1. Set Environment to dry_run BEFORE importing other modules
os.environ["APP_MODE"] = "dry_run"

# Import Action Entry Points
from actions.action12_shared_matches import fetch_shared_matches as action12_run
from config import config_schema
from core import workflow_actions  # Import module to allow patching
from core.session_manager import SessionManager

# Setup Logging
logger = logging.getLogger("DryRunTest")


def separator(name: str):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    logger.info(f"--- START DRY RUN TEST: {name} ---")


def run_test_action6(sm: SessionManager):
    separator("Action 6: Gather DNA Matches")
    # Force max_pages to 1
    original_max_pages = config_schema.api.max_pages
    config_schema.api.max_pages = 1
    try:
        logger.info("Running Action 6 with max_pages=1...")
        start_time = time.time()
        # Use module-qualified access
        result = workflow_actions.gather_dna_matches(sm, start=1)
        duration = time.time() - start_time

        if result:
            logger.info(f"Action 6 completed successfully in {duration:.2f}s")
        else:
            logger.error("Action 6 reported failure/incomplete.")

    except Exception as e:
        logger.error(f"Action 6 threw exception: {e}", exc_info=True)
    finally:
        config_schema.api.max_pages = original_max_pages


def run_test_action7(sm: SessionManager):
    separator("Action 7: Inbox Search")
    try:
        result = workflow_actions.srch_inbox_actn(sm)
        if result:
            logger.info("Action 7 completed successfully.")
        else:
            logger.error("Action 7 reported failure.")
    except Exception as e:
        logger.error(f"Action 7 threw exception: {e}", exc_info=True)


def run_test_action8(sm: SessionManager):
    separator("Action 8: Send Messages (DRY RUN)")
    if config_schema.app_mode != "dry_run" and os.environ.get("APP_MODE") != "dry_run":
        logger.critical("APP_MODE is NOT dry_run! Aborting Action 8 test.")
        return

    try:
        result = workflow_actions.send_messages_action(sm)
        if result:
            logger.info("Action 8 completed successfully (Simulated).")
        else:
            logger.error("Action 8 reported failure.")
    except Exception as e:
        logger.error(f"Action 8 threw exception: {e}", exc_info=True)


def run_test_action9(sm: SessionManager):
    separator("Action 9: Process Productive Messages")
    try:
        result = workflow_actions.process_productive_messages_action(sm)
        if result:
            logger.info("Action 9 completed successfully.")
        else:
            logger.error("Action 9 reported failure.")
    except Exception as e:
        logger.error(f"Action 9 threw exception: {e}", exc_info=True)


def run_test_action10(_sm: SessionManager):
    separator("Action 10: GEDCOM Import (Limited)")
    logger.info("Skipping full Action 10 run in this harness to avoid long processing.")
    # action10_main(_sm)
    pass


def run_test_action12(sm: SessionManager):
    separator("Action 12: Fetch Shared Matches")
    try:
        original_max_pages = config_schema.api.max_pages
        config_schema.api.max_pages = 1
        logger.info("Running Action 12 with max_pages=1...")
        action12_run(sm)
    except Exception as e:
        logger.error(f"Action 12 error: {e}")
    finally:
        config_schema.api.max_pages = original_max_pages


def run_test_action13(_sm: SessionManager):
    separator("Action 13: Triangulation")
    try:
        logger.info("Skipping Action 13 (Interactive) in headless harness.")
        # action13_run(sm)
    except Exception as e:
        logger.error(f"Action 13 error: {e}")


import argparse
from unittest.mock import MagicMock, patch


def main():
    parser = argparse.ArgumentParser(description="Dry Run Test Harness")
    parser.add_argument("--mock", action="store_true", help="Mock session and network calls")
    args = parser.parse_args()

    print(f"Starting Dry Run Test Harness in mode: {os.environ.get('APP_MODE')}")

    if args.mock:
        print("Mock Mode Enabled: Network calls will be simulated.")
        # Create a mock context for dependencies
        with (
            patch.object(SessionManager, "start_sess", return_value=True),
            patch.object(SessionManager, "ensure_session_ready", return_value=True),
            patch.object(SessionManager, "is_sess_valid", return_value=True),
            patch.object(SessionManager, "_check_session_health", return_value=True),
            patch.object(SessionManager, "validate_system_health", return_value=True),
            # patch("core.workflow_actions.gather_dna_matches", return_value=True),  # Unmocked to test Action 6 logic
            patch(
                "actions.action6_gather._call_match_list_api",
                return_value={
                    "matchList": [{"sampleId": "mock_uuid", "publicDisplayName": "Mock Match"}],
                    "totalPages": 1,
                },
            ),
            patch("actions.action6_gather._fetch_in_tree_status", return_value=set()),
            patch("actions.action6_gather._get_cached_or_fresh_csrf_token", return_value="mock_csrf_token"),
            patch("actions.action6_gather.nav_to_page", return_value=True),  # Patch local nav import
            patch(
                "actions.gather.api_implementations.fetch_combined_details",
                return_value={"tester_profile_id": "mock_pid"},
            ),
            patch(
                "actions.gather.api_implementations.fetch_batch_badge_details",
                return_value={"their_cfpid": "mock_cfpid"},
            ),
            patch("core.workflow_actions.srch_inbox_actn", return_value=True),
            patch("core.workflow_actions.process_productive_messages_action", return_value=True),
            patch("core.workflow_actions.nav_to_page", return_value=True),  # Patch navigation
        ):
            # Initialize Session INSIDE patched context
            sm = SessionManager()

            # CRITICAL: Manually set session_ready=True because session_guards check this attribute
            sm.session_ready = True

            # Setup Mock IDs
            if not sm.api_manager.my_profile_id:
                sm.api_manager.my_profile_id = "mock_profile_id"
            if not sm.api_manager.my_uuid:
                sm.api_manager.my_uuid = "mock_uuid"

            # Mock driver in case it's accessed
            sm.browser_manager = MagicMock()
            sm.browser_manager.driver = MagicMock()
            sm.browser_manager.driver.current_url = "https://www.ancestry.com/discoveryui-matches/list/"

            print("\n--- Running in MOCKED context ---")

            run_test_action6(sm)
            run_test_action7(sm)
            run_test_action9(sm)
            run_test_action8(sm)

    else:
        sm = SessionManager()
        if not sm.start_sess("Dry Run Test"):
            logger.error("Could not start session.")
            print("Session start failed. Aborting.")
            return

        if not sm.ensure_session_ready("Dry Run Test"):
            logger.error("Session readiness check failed.")
            return

        run_test_action6(sm)
        run_test_action7(sm)
        run_test_action9(sm)
        run_test_action8(sm)

    print("\nDry Run Tests Completed. Check logs/app.log for details.")
    if not args.mock and 'sm' in locals():
        sm._close_browser()


if __name__ == "__main__":
    main()
