#!/usr/bin/env python3
"""Production guard checker for messaging safety flags.

Exit code 0 if safe, 1 if any guard fails. Designed for quick CLI use.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _ensure_venv() -> None:
    """Ensure running in venv, auto-restart if needed."""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        return

    venv_python = _PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'
    if not venv_python.exists():
        venv_python = _PROJECT_ROOT / '.venv' / 'bin' / 'python'
        if not venv_python.exists():
            print("⚠️  WARNING: Not running in virtual environment")
            return

    import os as _os

    print(f"🔄 Re-running with venv Python: {venv_python}")
    _os.chdir(_PROJECT_ROOT)
    _os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])


_ensure_venv()

from config import config_schema


def main() -> int:
    issues: list[str] = []

    app_mode = getattr(config_schema, "app_mode", "development")
    if app_mode != "production":
        issues.append(f"APP_MODE is '{app_mode}' (expected 'production' for live sends)")

    if getattr(config_schema, "emergency_stop_enabled", False):
        issues.append("EMERGENCY_STOP is enabled")

    if app_mode == "production" and not getattr(config_schema, "dry_run_verified", False):
        issues.append("DRY_RUN_VERIFIED is false; run a full dry-run before production")

    if (
        app_mode == "production"
        and getattr(config_schema, "auto_approve_enabled", False)
        and not getattr(config_schema, "allow_production_auto_approve", False)
    ):
        issues.append("Auto-approval enabled without ALLOW_PRODUCTION_AUTO_APPROVE=true")

    max_inbox = getattr(config_schema, "max_inbox", 0)
    max_send = getattr(config_schema, "max_send_per_run", 0)
    if max_inbox == 0 or max_send == 0:
        issues.append("MAX_INBOX and/or MAX_SEND_PER_RUN is 0 (unbounded or halted)")

    if issues:
        print("Production guard FAILED:")
        for i, issue in enumerate(issues, start=1):
            print(f"  {i}. {issue}")
        return 1

    print("Production guard OK: all safety flags satisfied for production messaging.")
    print(
        f"APP_MODE={app_mode}, DRY_RUN_VERIFIED={getattr(config_schema, 'dry_run_verified', False)}, "
        f"AUTO_APPROVE_ENABLED={getattr(config_schema, 'auto_approve_enabled', False)}, "
        f"ALLOW_PRODUCTION_AUTO_APPROVE={getattr(config_schema, 'allow_production_auto_approve', False)}, "
        f"MAX_INBOX={max_inbox}, MAX_SEND_PER_RUN={max_send}, "
        f"EMERGENCY_STOP={getattr(config_schema, 'emergency_stop_enabled', False)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
