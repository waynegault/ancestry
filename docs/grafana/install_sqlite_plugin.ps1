# Install Grafana SQLite Plugin with Administrator Privileges
# This script must be run as Administrator

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Grafana SQLite Plugin Installer" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "❌ ERROR: This script requires Administrator privileges`n" -ForegroundColor Red
    Write-Host "To run with admin rights:" -ForegroundColor Yellow
    Write-Host "   1. Right-click PowerShell" -ForegroundColor Yellow
    Write-Host "   2. Select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host "   3. Navigate to: cd 'C:\Users\wayne\GitHub\Python\Projects\Ancestry'" -ForegroundColor Yellow
    Write-Host "   4. Run: .\docs\grafana\install_sqlite_plugin.ps1`n" -ForegroundColor Yellow
    
    Read-Host "Press Enter to exit"
    exit 1
}

# Define Grafana paths
$grafanaPath = "C:\Program Files\GrafanaLabs\grafana"
$grafanaBin = "$grafanaPath\bin"
$grafanaCli = "$grafanaBin\grafana-cli.exe"
$pluginsDir = "$grafanaPath\data\plugins"

# Verify Grafana is installed
if (-not (Test-Path $grafanaCli)) {
    Write-Host "❌ ERROR: Grafana not found at: $grafanaPath`n" -ForegroundColor Red
    Write-Host "Please install Grafana first:" -ForegroundColor Yellow
    Write-Host "   https://grafana.com/grafana/download?platform=windows`n" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✓ Grafana found at: $grafanaPath" -ForegroundColor Green

# Ensure plugins directory exists with proper permissions
if (-not (Test-Path $pluginsDir)) {
    Write-Host "`nCreating plugins directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $pluginsDir -Force | Out-Null
}

# Grant write permissions to plugins directory
Write-Host "`nSetting permissions on plugins directory..." -ForegroundColor Yellow
$acl = Get-Acl $pluginsDir
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    "Users", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow"
)
$acl.SetAccessRule($rule)
Set-Acl $pluginsDir $acl

Write-Host "✓ Permissions configured" -ForegroundColor Green

# Stop Grafana service if running
Write-Host "`nChecking Grafana service status..." -ForegroundColor Yellow
$service = Get-Service -Name "Grafana" -ErrorAction SilentlyContinue

if ($service -and $service.Status -eq "Running") {
    Write-Host "Stopping Grafana service..." -ForegroundColor Yellow
    Stop-Service -Name "Grafana" -Force
    Start-Sleep -Seconds 2
    Write-Host "✓ Service stopped" -ForegroundColor Green
} else {
    Write-Host "✓ Service not running" -ForegroundColor Green
}

# Install SQLite plugin
Write-Host "`nInstalling SQLite datasource plugin..." -ForegroundColor Yellow
Set-Location $grafanaBin

try {
    & .\grafana-cli.exe plugins install frser-sqlite-datasource
    Write-Host "`n✓ Plugin installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "`n❌ ERROR: Plugin installation failed" -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
    Read-Host "`nPress Enter to exit"
    exit 1
}

# Start Grafana service
Write-Host "`nStarting Grafana service..." -ForegroundColor Yellow
Start-Service -Name "Grafana"
Start-Sleep -Seconds 3

$service = Get-Service -Name "Grafana"
if ($service.Status -eq "Running") {
    Write-Host "✓ Grafana service started successfully" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning: Grafana service did not start properly" -ForegroundColor Yellow
}

# Verify plugin installation
Write-Host "`nVerifying installation..." -ForegroundColor Yellow
$pluginPath = "$pluginsDir\frser-sqlite-datasource"

if (Test-Path $pluginPath) {
    Write-Host "✓ Plugin files verified at: $pluginPath" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning: Plugin directory not found" -ForegroundColor Yellow
}

# Final instructions
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "   1. Open Grafana: http://localhost:3000" -ForegroundColor White
Write-Host "   2. Login (default: admin/admin)" -ForegroundColor White
Write-Host "   3. Add SQLite data source:" -ForegroundColor White
Write-Host "      • Configuration → Data Sources → Add data source" -ForegroundColor White
Write-Host "      • Select 'SQLite'" -ForegroundColor White
Write-Host "      • Path: C:\Users\wayne\GitHub\Python\Projects\Ancestry\Data\ancestry.db" -ForegroundColor White
Write-Host "   4. Import dashboards from docs\grafana\ directory`n" -ForegroundColor White

Read-Host "Press Enter to exit"
