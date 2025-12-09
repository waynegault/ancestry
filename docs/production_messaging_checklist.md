# Production Messaging Checklist

Use this checklist before attempting any large-scale outreach (e.g., 16k DNA match invites). Defaults are intentionally conservative: `AUTO_APPROVE_ENABLED` is off and `MAX_INBOX` caps candidate pull volume.

## 0) Configuration

- Set `APP_MODE=production` for real sends; `APP_MODE=dry_run` for rehearsal.
- Keep `AUTO_APPROVE_ENABLED=false` until manual quality is verified; enable only after spot-checking.
- Set `MAX_INBOX` to the exact per-run cap (start small: 20–50). `MAX_INBOX=0` is not recommended for production.
- Set `MAX_SEND_PER_RUN` to a hard ceiling (start at 20–50); final sends are capped by `min(MAX_INBOX, MAX_SEND_PER_RUN)`.
- Keep `EMERGENCY_STOP=false`; set to `true` only to halt all messaging immediately.
- Keep `AUTO_APPROVE_ENABLED=false`; if you must enable it in production, also set `ALLOW_PRODUCTION_AUTO_APPROVE=true` explicitly.
- Confirm `REQUESTS_PER_SECOND=0.3` and do not raise without validation.
- Ensure `MIN_MESSAGE_INTERVAL` implicitly stays at 8 weeks in production mode.
- Require `DRY_RUN_VERIFIED=true` before any production send; leave it `false`/unset otherwise (run aborts in production if not set).
- Optional: set `PER_RECIPIENT_BACKOFF_SECONDS` for an extra throttle in testing/dry_run.

## 1) Dry Run (full list)

- Run Action 8 with `APP_MODE=dry_run` against the full candidate list.
- Export/review: message drafts, auto-approval counts, and any DESIST/blocked detections.
- Fix templates or prompts before proceeding.

## 2) Manual Review Pilot

- Set `AUTO_APPROVE_ENABLED=false` (default). Run with `MAX_INBOX=20–50`.
- Manually review the first 100–200 drafts in the approval queue.
- Confirm no first-touch messages auto-send; confirm DESIST/BLOCKED are skipped.

## 3) Controlled Rollout

- Gradually raise `MAX_INBOX` (e.g., 50 → 200 per run) only after quality is proven.
- If enabling auto-approval, raise the confidence threshold rather than volume; monitor queue stats each run.

## 4) Monitoring & Alerts

- Check `Logs/app.log` for 429s, auth refresh, and critical alerts.
- Review approval queue stats: pending vs. auto-approved; investigate any spikes.
- Daily review DESIST/OPT_OUT responses; halt on complaints.
- Run `python scripts/check_production_guard.py` to validate safety flags before production sends.

## 5) Emergency Stop & Rollback

- To stop sending: set `AUTO_APPROVE_ENABLED=false`, set `MAX_INBOX=0`, or switch `APP_MODE=dry_run`.
- Backup the database before changing modes or thresholds.
- Document any incidents and reset counters before resuming.
