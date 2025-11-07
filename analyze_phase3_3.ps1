# Phase 3.3 Post-Test Analysis
# Analyzes logs after Action 6 run to verify rate limiter performance

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase 3.3: Post-Test Analysis" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check for 429 errors
$count429 = (Select-String -Path Logs\app.log -Pattern "429|Too Many Requests" -AllMatches).Count
if ($count429 -eq 0) {
    Write-Host "✅ PASS: Zero 429 errors detected" -ForegroundColor Green
} else {
    Write-Host "❌ FAIL: $count429 rate limit errors found" -ForegroundColor Red
}

# Check for rate adaptations
$rateDecreases = (Select-String -Path Logs\app.log -Pattern "decreased rate|Decreased rate" -AllMatches).Count
$rateIncreases = (Select-String -Path Logs\app.log -Pattern "increased rate|Increased rate|100 successes" -AllMatches).Count
Write-Host "📊 Rate Adaptations: ↓$rateDecreases decreases, ↑$rateIncreases increases" -ForegroundColor Cyan

# Check for watchdog triggers
$watchdogTriggers = (Select-String -Path Logs\app.log -Pattern "Watchdog triggered|force.*restart" -AllMatches).Count
if ($watchdogTriggers -eq 0) {
    Write-Host "✅ PASS: Zero watchdog triggers (no hangs)" -ForegroundColor Green
} else {
    Write-Host "⚠️  INFO: $watchdogTriggers watchdog triggers (timeout protection activated)" -ForegroundColor Yellow
}

# Check completion status
$completionLine = Select-String -Path Logs\app.log -Pattern "Action 6 completed|Processing complete|final_success" -AllMatches | Select-Object -Last 1
if ($completionLine) {
    Write-Host "✅ Action 6 completed successfully" -ForegroundColor Green
} else {
    Write-Host "⚠️  Action 6 may not have completed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Rate Limiter Final Metrics:" -ForegroundColor Cyan
Write-Host "----------------------------"
Select-String -Path Logs\app.log -Pattern "Rate Limiter Performance|Total Requests:|429 Errors:|Current Rate:|Rate Adjustments:|Average Wait Time:" | Select-Object -Last 6 | ForEach-Object {
    Write-Host $_.Line -ForegroundColor White
}

Write-Host ""
Write-Host "📝 Key Indicators for Success:" -ForegroundColor Yellow
Write-Host "  1. Zero 429 errors ✓" -ForegroundColor $(if ($count429 -eq 0) { "Green" } else { "Red" })
Write-Host "  2. Smooth rate adaptation (gradual changes)" -ForegroundColor Yellow
Write-Host "  3. No oscillation (no rapid up/down changes)" -ForegroundColor Yellow
Write-Host "  4. Stable final rate" -ForegroundColor Yellow
Write-Host ""

# Extract and display the final rate if available
$finalRate = Select-String -Path Logs\app.log -Pattern "Current Rate:\s+(\d+\.\d+)" -AllMatches | Select-Object -Last 1
if ($finalRate -and $finalRate.Matches.Groups.Count -gt 1) {
    $rate = $finalRate.Matches.Groups[1].Value
    Write-Host "Final Rate: $rate req/s" -ForegroundColor Cyan
}
