# Phase 3.3 Monitoring Script
# Watches for rate limiting behavior during Action 6 test

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase 3.3: AdaptiveRateLimiter Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Monitoring Logs\app.log for:" -ForegroundColor Yellow
Write-Host "  - 429 errors (should be ZERO)" -ForegroundColor Yellow
Write-Host "  - Rate adaptations (on_429_error, on_success)" -ForegroundColor Yellow
Write-Host "  - Fill rate changes" -ForegroundColor Yellow
Write-Host "  - Watchdog triggers (should be ZERO)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Gray
Write-Host ""

# Watch log file for key events
Get-Content Logs\app.log -Wait | ForEach-Object {
    $line = $_
    
    # Highlight 429 errors (BAD - should not happen)
    if ($line -match "429|rate limit|Rate Limit") {
        Write-Host $line -ForegroundColor Red
    }
    # Highlight rate adaptations (GOOD - shows learning)
    elseif ($line -match "on_429_error|on_success|fill_rate|req/s|AdaptiveRateLimiter") {
        Write-Host $line -ForegroundColor Green
    }
    # Highlight watchdog triggers (NEUTRAL - validates timeout protection)
    elseif ($line -match "watchdog|timeout|force.*restart|🚨") {
        Write-Host $line -ForegroundColor Magenta
    }
    # Highlight page progress
    elseif ($line -match "Processing page|Page \d+/\d+|Progress:") {
        Write-Host $line -ForegroundColor Cyan
    }
    # Show important events
    elseif ($line -match "ERROR|WARNING|⚠️|✅|❌") {
        Write-Host $line -ForegroundColor Yellow
    }
}
