# Troubleshooting Guide

## Common Errors and Solutions

### Session & Authentication Errors

#### Error: "Session not ready" or "Cannot perform action"
**Cause**: Browser session not initialized or expired.

**Solution**:
```powershell
# Check session status via main menu
python main.py  # Option 5: Check Login Status

# Force session refresh if needed
# The session manager will automatically attempt recovery
```

**Prevention**: The `exec_actn()` wrapper automatically calls `ensure_session_ready()` before each action.

---

#### Error: "401 Unauthorized" or "403 Forbidden"
**Cause**: Session cookies expired during long-running operation.

**Solution**:
```powershell
# Clear browser cache and re-login
python main.py  # Option 5: Check Login Status

# If persistent, delete session cache
Remove-Item Cache\session_state\* -Force
```

**Prevention**: Action 6 has proactive session health monitoring that refreshes at 25-minute mark.

---

### Rate Limiting Errors

#### Error: "429 Too Many Requests"
**Cause**: Exceeded Ancestry API rate limits (72-second penalty).

**Solution**:
```powershell
# Check for 429 errors in log
(Select-String -Path Logs\app.log -Pattern "429 error").Count

# Wait for penalty period to expire (72 seconds)
# Then resume with lower RPS setting
```

**Prevention**:
- Keep `REQUESTS_PER_SECOND=0.3` in `.env`
- Never exceed 50 pages without validating zero 429 errors
- Monitor: `Get-Content Logs\app.log -Wait | Select-String "429|rate"`

---

### Database Errors

#### Error: "UNIQUE constraint failed: people.uuid"
**Cause**: UUID case sensitivity mismatch (was fixed Oct 2025).

**Solution**:
```powershell
# Check for duplicate UUIDs
sqlite3 Data/ancestry.db "SELECT uuid, COUNT(*) FROM people GROUP BY uuid HAVING COUNT(*) > 1"

# If duplicates exist, backup and clean
python main.py  # Option 3: Backup Database
```

**Prevention**: All UUID lookups now use `.upper()` for case-insensitive matching.

---

#### Error: "OperationalError: database is locked"
**Cause**: Multiple processes accessing SQLite database simultaneously.

**Solution**:
```powershell
# Find and close other Python processes
Get-Process python* | Select-Object Id, ProcessName

# If stuck, restart and wait
Stop-Process -Name python -Force
Start-Sleep -Seconds 5
python main.py
```

**Prevention**: Only run one main.py instance at a time.

---

### AI Integration Errors

#### Error: "AI provider returned empty response"
**Cause**: API quota exceeded or network timeout.

**Solution**:
```powershell
# Check AI telemetry for patterns
Get-Content Logs\prompt_experiments.jsonl -Tail 20 | ConvertFrom-Json | Select-Object timestamp, parse_success

# Verify API key is set
echo $env:GOOGLE_AI_API_KEY
```

**Prevention**:
- Monitor quality scores: `python prompt_telemetry.py --stats`
- Set up fallback provider in config

---

#### Error: "JSON parsing failed" from AI response
**Cause**: AI returned malformed JSON or unexpected format.

**Solution**:
```powershell
# Check recent AI responses
Get-Content Logs\prompt_experiments.jsonl -Tail 10

# Review prompt for issues
python -c "import json; print(json.dumps(json.load(open('ai/ai_prompts.json'))['intent_classification'], indent=2))"
```

**Prevention**: Quality regression gate blocks prompts with >5 point median drop.

---

### MS Graph Errors

#### Error: "MS_GRAPH_CLIENT_ID not found"
**Cause**: Missing environment variable.

**Solution**:
```powershell
# Add to .env file
echo "MS_GRAPH_CLIENT_ID=your-client-id" >> .env

# Or set environment variable
$env:MS_GRAPH_CLIENT_ID = "your-client-id"
```

---

#### Error: "Device flow timeout"
**Cause**: Browser authentication took too long (15 minute timeout).

**Solution**:
1. Retry the action
2. Complete browser authentication faster
3. Check internet connectivity

**Prevention**: Complete authentication promptly when device code appears.

---

### Action-Specific Errors

#### Action 6: "Circuit breaker TRIPPED"
**Cause**: 5 consecutive failures during DNA match gathering.

**Solution**:
```powershell
# Check what triggered the failures
Select-String -Path Logs\app.log -Pattern "Circuit breaker" | Select-Object -Last 10

# Common causes: session expiry, network issues
# Fix underlying issue and retry
```

**Prevention**: Circuit breaker prevents 15,000+ wasted API calls on persistent failures.

---

#### Action 7: "No inbox messages found"
**Cause**: Inbox page structure changed or selector mismatch.

**Solution**:
```powershell
# Check CSS selectors are still valid
python -c "from browser.css_selectors import INBOX_SELECTORS; print(INBOX_SELECTORS)"

# Verify inbox is accessible manually in browser
```

---

#### Action 11: "Draft not found or already sent"
**Cause**: Draft was already processed or deleted.

**Solution**:
```powershell
# Check draft status
sqlite3 Data/ancestry.db "SELECT id, status, created_at FROM draft_replies ORDER BY created_at DESC LIMIT 10"

# Duplicate prevention checks 5-minute idempotency window
```

---

### Startup Errors

#### Error: "Module not found" or Import errors
**Cause**: Missing dependencies or corrupted virtualenv.

**Solution**:
```powershell
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or recreate virtual environment
python -m venv .venv --clear
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

#### Error: "Configuration validation failed"
**Cause**: Invalid or missing config values.

**Solution**:
```powershell
# Check config validation
python -c "from config.config_manager import get_config_manager; cm = get_config_manager(); print(cm.validate())"

# Review .env file for missing required values
```

---

## Diagnostic Commands

### Quick Health Check
```powershell
# Run health check
python main.py  # Option 7: System Health Check

# Or via code
python -c "from core.health_check import HealthCheckRunner; HealthCheckRunner().run_all()"
```

### Log Analysis
```powershell
# Recent errors
Select-String -Path Logs\app.log -Pattern "ERROR|CRITICAL" | Select-Object -Last 20

# Specific action logs
Select-String -Path Logs\app.log -Pattern "Action 6|Action 7|Action 11" | Select-Object -Last 30

# Performance issues
Select-String -Path Logs\app.log -Pattern "Duration:|Elapsed:" | Select-Object -Last 20
```

### Database Integrity
```powershell
# Check database
sqlite3 Data/ancestry.db "PRAGMA integrity_check"

# Record counts
sqlite3 Data/ancestry.db "SELECT 'people' as tbl, COUNT(*) FROM people UNION SELECT 'dna_matches', COUNT(*) FROM dna_matches"
```

### Test Suite
```powershell
# Run all tests
python run_all_tests.py --fast

# Single module test
python -m actions.action6_gather

# Check quality score
python testing/code_quality_checker.py
```

## Getting Help

1. Check this troubleshooting guide
2. Review `Logs/app.log` for detailed error context
3. Run `python run_all_tests.py` to verify system health
4. Check `todo.md` for known issues and workarounds
