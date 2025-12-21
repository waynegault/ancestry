# Automated Grafana Setup Script for Ancestry Research Platform
# This script installs and configures Grafana with all required plugins and data sources

param(
    [switch]$Silent = $false,
    [switch]$SkipDownload = $false
)

$ErrorActionPreference = "Stop"

# Configuration
$GRAFANA_VERSION = "11.3.1"
$GRAFANA_INSTALLER_URL = "https://dl.grafana.com/enterprise/release/grafana-enterprise-$GRAFANA_VERSION.windows-amd64.msi"
$GRAFANA_INSTALL_PATH = "C:\Program Files\GrafanaLabs\grafana"
$GRAFANA_BIN = "$GRAFANA_INSTALL_PATH\bin"
$GRAFANA_DATA = "$GRAFANA_INSTALL_PATH\data"
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
$DB_PATH = "$PROJECT_ROOT\Data\ancestry.db"
$DASHBOARDS_PATH = "$PROJECT_ROOT\docs\grafana"

# Color output functions
function Write-Step { param($msg) if (-not $Silent) { Write-Host "`n[STEP] $msg" -ForegroundColor Cyan } }
function Write-Success { param($msg) if (-not $Silent) { Write-Host "  ✓ $msg" -ForegroundColor Green } }
function Write-Info { param($msg) if (-not $Silent) { Write-Host "  → $msg" -ForegroundColor Yellow } }
function Write-Error-Custom { param($msg) Write-Host "  ✗ $msg" -ForegroundColor Red }

# Check if running as Administrator
function Test-Administrator {
    $user = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($user)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Download file with progress
function Download-File {
    param($url, $output)

    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $url -OutFile $output -UseBasicParsing
        $ProgressPreference = 'Continue'
        return $true
    } catch {
        Write-Error-Custom "Download failed: $_"
        return $false
    }
}

# Main installation workflow
function Install-Grafana {
    Write-Step "Automated Grafana Setup"

    # 1. Check admin privileges
    if (-not (Test-Administrator)) {
        Write-Error-Custom "Administrator privileges required"
        Write-Info "Please run: Right-click PowerShell → 'Run as Administrator'"
        Write-Info "Then execute: .\docs\grafana\setup_grafana.ps1"
        return $false
    }
    Write-Success "Running with Administrator privileges"

    # 2. Check if Grafana already installed
    if (Test-Path "$GRAFANA_BIN\grafana-cli.exe") {
        Write-Success "Grafana already installed at: $GRAFANA_INSTALL_PATH"

        # Verify service exists
        $service = Get-Service -Name "Grafana" -ErrorAction SilentlyContinue
        if ($service) {
            Write-Success "Grafana service registered"

            # Ensure service is running
            if ($service.Status -ne "Running") {
                Write-Info "Starting Grafana service..."
                Start-Service -Name "Grafana"
                Start-Sleep -Seconds 3
            }

            if ((Get-Service -Name "Grafana").Status -eq "Running") {
                Write-Success "Grafana service is running"
            }
        }

        return $true
    }

    # 3. Download Grafana installer
    if (-not $SkipDownload) {
        Write-Step "Downloading Grafana v$GRAFANA_VERSION"
        $installerPath = "$env:TEMP\grafana-installer.msi"

        if (Download-File -url $GRAFANA_INSTALLER_URL -output $installerPath) {
            Write-Success "Downloaded to: $installerPath"
        } else {
            return $false
        }

        # 4. Install Grafana silently
        Write-Step "Installing Grafana..."
        Write-Info "This may take 2-3 minutes..."

        try {
            $msiArgs = "/i `"$installerPath`" /quiet /norestart"
            Start-Process msiexec.exe -ArgumentList $msiArgs -Wait -NoNewWindow
            Write-Success "Grafana installed successfully"

            # Clean up installer
            Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Error-Custom "Installation failed: $_"
            return $false
        }
    }

    # 5. Wait for installation to complete
    Write-Info "Verifying installation..."
    $maxAttempts = 10
    $attempt = 0

    while ($attempt -lt $maxAttempts) {
        if (Test-Path "$GRAFANA_BIN\grafana-cli.exe") {
            Write-Success "Installation verified"
            break
        }
        $attempt++
        Start-Sleep -Seconds 2
    }

    if ($attempt -eq $maxAttempts) {
        Write-Error-Custom "Installation verification timeout"
        return $false
    }

    # 6. Start Grafana service
    Write-Step "Starting Grafana service"

    try {
        $service = Get-Service -Name "Grafana" -ErrorAction SilentlyContinue
        if (-not $service) {
            Write-Error-Custom "Grafana service not found after installation"
            return $false
        }

        if ($service.Status -ne "Running") {
            Start-Service -Name "Grafana"
            Start-Sleep -Seconds 5
        }

        # Set to start automatically
        Set-Service -Name "Grafana" -StartupType Automatic

        $service = Get-Service -Name "Grafana"
        if ($service.Status -eq "Running") {
            Write-Success "Grafana service started (http://localhost:3000)"
        } else {
            Write-Error-Custom "Service failed to start"
            return $false
        }
    } catch {
        Write-Error-Custom "Service startup failed: $_"
        return $false
    }

    return $true
}

# Install plugins
function Install-Plugins {
    Write-Step "Installing Grafana plugins"

    # Ensure Grafana is installed
    if (-not (Test-Path "$GRAFANA_BIN\grafana-cli.exe")) {
        Write-Error-Custom "Grafana not found. Run installation first."
        return $false
    }

    # Stop service for plugin installation
    Write-Info "Stopping Grafana service for plugin installation..."
    Stop-Service -Name "Grafana" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2

    # Set permissions on plugins directory
    $pluginsDir = "$GRAFANA_DATA\plugins"
    if (-not (Test-Path $pluginsDir)) {
        New-Item -ItemType Directory -Path $pluginsDir -Force | Out-Null
    }

    $acl = Get-Acl $pluginsDir
    $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
        "Users", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow"
    )
    $acl.SetAccessRule($rule)
    Set-Acl $pluginsDir $acl

    Set-Location $GRAFANA_BIN

    # Install SQLite plugin
    Write-Info "Installing SQLite datasource plugin..."
    try {
        & .\grafana-cli.exe plugins install frser-sqlite-datasource 2>&1 | Out-Null
        Write-Success "SQLite plugin installed"
    } catch {
        Write-Error-Custom "SQLite plugin installation failed: $_"
    }

    # Install Node Graph plugin (for code architecture visualization)
    Write-Info "Installing Node Graph panel plugin..."
    try {
        & .\grafana-cli.exe plugins install nodeGraph 2>&1 | Out-Null
        Write-Success "Node Graph plugin installed"
    } catch {
        Write-Info "Node Graph plugin may already be built-in (OK)"
    }

    # Restart service
    Write-Info "Restarting Grafana service..."
    Start-Service -Name "Grafana"
    Start-Sleep -Seconds 5

    if ((Get-Service -Name "Grafana").Status -eq "Running") {
        Write-Success "Plugins installed and service restarted"
        return $true
    } else {
        Write-Error-Custom "Service failed to restart after plugin installation"
        return $false
    }
}

# Configure data sources via Grafana API
function Configure-DataSources {
    Write-Step "Configuring data sources"

    # Wait for Grafana API to be ready
    Write-Info "Waiting for Grafana API..."
    $maxAttempts = 30
    $attempt = 0

    while ($attempt -lt $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Success "Grafana API ready"
                break
            }
        } catch {
            # API not ready yet
        }
        $attempt++
        Start-Sleep -Seconds 2
    }

    if ($attempt -eq $maxAttempts) {
        Write-Error-Custom "Grafana API timeout"
        return $false
    }

    # Try both default and custom credentials
    $credentials = @("admin:admin", "admin:ancestry")
    $headers = $null

    foreach ($cred in $credentials) {
        try {
            $base64Auth = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($cred))
            $testHeaders = @{
                "Authorization" = "Basic $base64Auth"
                "Content-Type" = "application/json"
            }
            $response = Invoke-RestMethod -Uri "http://localhost:3000/api/org" -Headers $testHeaders -ErrorAction Stop
            $headers = $testHeaders
            Write-Success "Authenticated with credentials: $($cred.Split(':')[0])"
            break
        } catch {
            # Try next credential
        }
    }

    if (-not $headers) {
        Write-Error-Custom "Could not authenticate with any credentials"
        return $false
    }

    # Check if Prometheus data source exists
    Write-Info "Configuring Prometheus data source..."
    try {
        $prometheusDatasource = @{
            name = "Prometheus"
            type = "prometheus"
            url = "http://localhost:9090"
            access = "proxy"
            isDefault = $true
        } | ConvertTo-Json

        $response = Invoke-RestMethod -Uri "http://localhost:3000/api/datasources" `
            -Method Post -Headers $headers -Body $prometheusDatasource -ErrorAction SilentlyContinue

        Write-Success "Prometheus data source configured"
    } catch {
        if ($_.Exception.Response.StatusCode -eq 409) {
            Write-Success "Prometheus data source already exists"
        } else {
            Write-Info "Prometheus data source: Manual configuration needed"
        }
    }

    # Check if SQLite data source exists
    Write-Info "Configuring SQLite data source..."
    try {
        $sqliteDatasource = @{
            name = "SQLite"
            type = "frser-sqlite-datasource"
            jsonData = @{
                path = $DB_PATH
            }
        } | ConvertTo-Json

        $response = Invoke-RestMethod -Uri "http://localhost:3000/api/datasources" `
            -Method Post -Headers $headers -Body $sqliteDatasource -ErrorAction SilentlyContinue

        Write-Success "SQLite data source configured"
    } catch {
        if ($_.Exception.Response.StatusCode -eq 409) {
            Write-Success "SQLite data source already exists"
        } else {
            Write-Info "SQLite data source: Manual configuration needed"
        }
    }

    return $true
}

# Import dashboards
function Import-Dashboards {
    Write-Step "Importing dashboards"

    $dashboards = @(
        @{file="ancestry_overview.json"; name="Ancestry Automation Overview"},
        @{file="system_performance.json"; name="System Performance & Health"},
        @{file="genealogy_insights.json"; name="Genealogy Research Insights"},
        @{file="code_quality.json"; name="Code Quality & Architecture"},
        @{file="database_summary.json"; name="Database Summary"}
    )

    # Try both default and custom credentials
    $credentials = @("admin:admin", "admin:ancestry")
    $headers = $null

    foreach ($cred in $credentials) {
        try {
            $base64Auth = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($cred))
            $testHeaders = @{
                "Authorization" = "Basic $base64Auth"
                "Content-Type" = "application/json"
            }
            $response = Invoke-RestMethod -Uri "http://localhost:3000/api/org" -Headers $testHeaders -ErrorAction SilentlyContinue
            $headers = $testHeaders
            break
        } catch {
            # Try next credential
        }
    }

    if (-not $headers) {
        Write-Info "Could not authenticate for dashboard import"
        return $false
    }

    foreach ($dash in $dashboards) {
        $dashPath = Join-Path $DASHBOARDS_PATH $dash.file

        if (-not (Test-Path $dashPath)) {
            Write-Info "$($dash.name): File not found, skipping"
            continue
        }

        try {
            $dashboardJson = Get-Content $dashPath -Raw | ConvertFrom-Json

            $importPayload = @{
                dashboard = $dashboardJson
                overwrite = $true
                inputs = @(
                    @{name="DS_PROMETHEUS"; type="datasource"; pluginId="prometheus"; value="Prometheus"},
                    @{name="DS_SQLITE"; type="datasource"; pluginId="frser-sqlite-datasource"; value="SQLite"}
                )
            } | ConvertTo-Json -Depth 20

            $response = Invoke-RestMethod -Uri "http://localhost:3000/api/dashboards/import" `
                -Method Post -Headers $headers -Body $importPayload -ErrorAction Stop

            Write-Success "$($dash.name) imported"
        } catch {
            Write-Info "$($dash.name): $_"
        }
    }

    return $true
}

# Main execution
function Main {
    if (-not $Silent) {
        Write-Host "`n" + ("="*70) -ForegroundColor Cyan
        Write-Host "   Ancestry Research Platform - Grafana Automated Setup" -ForegroundColor Cyan
        Write-Host ("="*70) + "`n" -ForegroundColor Cyan
    }

    $success = $true

    # Install Grafana
    if (-not (Install-Grafana)) {
        $success = $false
    }

    # Install plugins
    if ($success -and -not (Install-Plugins)) {
        $success = $false
    }

    # Configure data sources
    if ($success) {
        Configure-DataSources | Out-Null
    }

    # Import dashboards
    if ($success) {
        Import-Dashboards | Out-Null
    }

    if (-not $Silent) {
        Write-Host "`n" + ("="*70) -ForegroundColor Cyan
        if ($success) {
            Write-Host "   ✅ Setup Complete!" -ForegroundColor Green
            Write-Host ("="*70) -ForegroundColor Cyan
            Write-Host "`n  Grafana is ready at: http://localhost:3000" -ForegroundColor Yellow
            Write-Host "  Default login: admin / admin" -ForegroundColor Yellow
            Write-Host "  ⚠️  Run 'reset_admin_password.ps1' to change password to 'ancestry'`n" -ForegroundColor Yellow
            Write-Host "  ⚠️  Run 'reset_admin_password.ps1' to change password to 'ancestry'`n" -ForegroundColor Yellow
        } else {
            Write-Host "   ⚠️  Setup completed with warnings" -ForegroundColor Yellow
            Write-Host ("="*70) -ForegroundColor Cyan
            Write-Host "`n  Please check manual configuration steps in QUICK_START.md`n" -ForegroundColor Yellow
        }
    }

    return $success
}

# Run if not being dot-sourced
if ($MyInvocation.InvocationName -ne '.') {
    $result = Main
    if (-not $Silent) {
        Read-Host "`nPress Enter to exit"
    }
    exit $(if ($result) { 0 } else { 1 })
}
