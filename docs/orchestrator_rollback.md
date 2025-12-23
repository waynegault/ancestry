# MessageSendOrchestrator Rollback Procedure

## Overview

This document describes the rollback procedure for the Unified Message Send Orchestrator in case of issues during production deployment.

## Pre-Deployment Checklist

Before enabling the orchestrator in production:

1. **Database Backup** - Take a full backup of the production database
   ```powershell
   # Create timestamped backup
   python main.py  # Select option 3: Backup Database
   ```

2. **Verify Shadow Mode Logs** - Ensure shadow mode has run for at least 1 week with no discrepancies
   ```powershell
   python -m messaging.shadow_mode_analyzer --report
   ```

3. **Document Baseline Metrics** - Record current send volumes, success rates, and error patterns

## Feature Flags

The orchestrator uses a two-level feature flag system:

### Master Switch
```env
# .env file
ENABLE_UNIFIED_SEND_ORCHESTRATOR=false  # Default: disabled
```

### Per-Action Flags (requires master switch to be True)
```env
ORCHESTRATOR_ACTION8=false   # Action 8: Generic sequences
ORCHESTRATOR_ACTION9=false   # Action 9: Custom replies
ORCHESTRATOR_ACTION11=false  # Action 11: Approved drafts
```

## Rollback Levels

### Level 1: Per-Action Rollback (Fastest)

If issues are detected with a specific action, disable only that action's orchestrator:

```env
# Keep master switch enabled, disable specific action
ENABLE_UNIFIED_SEND_ORCHESTRATOR=true
ORCHESTRATOR_ACTION8=false  # Rollback Action 8 only
ORCHESTRATOR_ACTION9=true
ORCHESTRATOR_ACTION11=true
```

**Impact**: Only affects the specific action; other actions continue using orchestrator.

### Level 2: Full Orchestrator Rollback

If issues affect all actions or are systemic:

```env
# Disable master switch - all actions revert to legacy code paths
ENABLE_UNIFIED_SEND_ORCHESTRATOR=false
```

**Impact**: All actions revert to their original, tested legacy code paths.

### Level 3: Emergency Stop

If immediate stop of ALL messaging is required:

```powershell
# Set app mode to read-only (stops all sends across entire system)
# In .env:
APP_MODE=research
```

**Impact**: All message sending stops immediately. Only read operations allowed.

## Rollback Steps

### Immediate Rollback (< 5 minutes)

1. **Edit .env file**:
   ```powershell
   notepad .env
   # Set ENABLE_UNIFIED_SEND_ORCHESTRATOR=false
   ```

2. **Restart the application** (if running as a service):
   ```powershell
   # For development/manual runs - just restart the script
   # For production service - restart the service
   ```

3. **Verify rollback**:
   ```powershell
   # Check logs for legacy code path activation
   Select-String -Path Logs\app.log -Pattern "Orchestrator disabled by feature flag" | Select-Object -Last 5
   ```

### Post-Rollback Verification

1. **Confirm legacy paths are active**:
   ```powershell
   # Should see "Orchestrator disabled" in logs
   Select-String -Path Logs\app.log -Pattern "legacy|disabled" | Select-Object -Last 10
   ```

2. **Verify sends are working**:
   ```powershell
   # Check for successful sends
   Select-String -Path Logs\app.log -Pattern "Message sent successfully" | Select-Object -Last 5
   ```

3. **Monitor for continued issues**:
   ```powershell
   # Watch for errors
   Get-Content Logs\app.log -Wait | Select-String "ERROR|Exception"
   ```

## Database Restoration

If database corruption is suspected (extremely rare):

1. **Stop the application immediately**

2. **Restore from backup**:
   ```powershell
   # Locate backup
   Get-ChildItem Data\backups\ | Sort-Object LastWriteTime -Descending | Select-Object -First 5

   # Copy backup to restore
   Copy-Item "Data\backups\ancestry_backup_YYYYMMDD_HHMMSS.db" -Destination "Data\ancestry.db"
   ```

3. **Restart application with orchestrator disabled**

## Post-Rollback Analysis

After stabilizing the system:

1. **Collect logs** for the incident period:
   ```powershell
   # Copy relevant log sections
   Get-Content Logs\app.log | Where-Object { $_ -match "YYYY-MM-DD HH:" }
   ```

2. **Check audit trail** for send decisions:
   ```powershell
   # Review recent audit entries
   Get-Content Logs\send_audit.jsonl | Select-Object -Last 50
   ```

3. **Run shadow mode analysis** to identify discrepancies:
   ```powershell
   python -m messaging.shadow_mode_analyzer --report
   ```

4. **Document findings** in an incident report

## Rollout Recovery

After resolving the issue, to re-enable the orchestrator:

1. **Deploy the fix** (code changes, config changes, etc.)

2. **Re-enable shadow mode first**:
   ```env
   ENABLE_UNIFIED_SEND_ORCHESTRATOR=false
   ENABLE_SHADOW_MODE=true
   ```

3. **Run shadow mode for 24-48 hours** minimum

4. **Verify no discrepancies** in shadow mode report

5. **Gradually re-enable** using per-action flags:
   - Enable Action 11 first (lowest volume)
   - Monitor for 24 hours
   - Enable Action 9
   - Monitor for 24 hours
   - Enable Action 8 (highest volume)
   - Monitor for 72 hours

6. **Full cutover** after successful gradual rollout

## Support Contacts

For production issues:
- Check the [Troubleshooting Guide](troubleshooting.md)
- Review [Reply Flow Architecture](reply_flow_architecture.md)
- Consult the [Production Messaging Checklist](production_messaging_checklist.md)

---

*Last Updated: 2025-12-23*
