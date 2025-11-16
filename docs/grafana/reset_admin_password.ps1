# Reset Grafana Admin Password to "ancestry"
# Run this script as Administrator

$ErrorActionPreference = "Stop"

$GRAFANA_BIN = "C:\Program Files\GrafanaLabs\grafana\bin"
$NEW_PASSWORD = "ancestry"

Write-Host "`n[Grafana Password Reset]" -ForegroundColor Cyan
Write-Host "Setting admin password to: $NEW_PASSWORD`n" -ForegroundColor Yellow

# Check if running as Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: Administrator privileges required" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'`n" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Stop Grafana service
Write-Host "Stopping Grafana service..." -ForegroundColor Yellow
Stop-Service -Name "Grafana" -Force
Start-Sleep -Seconds 2
Write-Host "  ✓ Service stopped`n" -ForegroundColor Green

# Reset admin password using grafana-cli
Write-Host "Resetting admin password..." -ForegroundColor Yellow
Set-Location $GRAFANA_BIN

try {
    & .\grafana-cli.exe admin reset-admin-password $NEW_PASSWORD 2>&1 | Out-Null
    Write-Host "  ✓ Password reset successful`n" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Password reset failed: $_`n" -ForegroundColor Red
    
    # Try to start service anyway
    Write-Host "Starting Grafana service..." -ForegroundColor Yellow
    Start-Service -Name "Grafana"
    Read-Host "Press Enter to exit"
    exit 1
}

# Start Grafana service
Write-Host "Starting Grafana service..." -ForegroundColor Yellow
Start-Service -Name "Grafana"
Start-Sleep -Seconds 3

if ((Get-Service -Name "Grafana").Status -eq "Running") {
    Write-Host "  ✓ Service started`n" -ForegroundColor Green
    Write-Host "="*60 -ForegroundColor Cyan
    Write-Host "SUCCESS! You can now log in to Grafana:" -ForegroundColor Green
    Write-Host "  URL:      http://localhost:3000" -ForegroundColor Yellow
    Write-Host "  Username: admin" -ForegroundColor Yellow
    Write-Host "  Password: ancestry" -ForegroundColor Yellow
    Write-Host "="*60 -ForegroundColor Cyan
} else {
    Write-Host "  ✗ Service failed to start`n" -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to exit"
