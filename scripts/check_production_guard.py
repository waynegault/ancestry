#!/usr/bin/env python3
"""Production guard checker for messaging safety flags.

Exit code 0 if safe, 1 if any guard fails. Designed for quick CLI use.
"""

from __future__ import annotations

import sys

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
