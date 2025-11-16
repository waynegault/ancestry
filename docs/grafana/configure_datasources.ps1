# Configure Grafana Data Sources for Ancestry Platform
# Run this after logging into Grafana to set up Prometheus and SQLite

$ErrorActionPreference = "Stop"

$GRAFANA_URL = "http://localhost:3000"
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
$DB_PATH = "$PROJECT_ROOT\Data\ancestry.db"

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "   Grafana Data Source Configuration" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# Prompt for credentials
Write-Host "Enter Grafana credentials:" -ForegroundColor Yellow
$username = Read-Host "Username (default: admin)"
if ([string]::IsNullOrWhiteSpace($username)) { $username = "admin" }

$securePassword = Read-Host "Password" -AsSecureString
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
$password = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

# Create auth header
$base64Auth = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("${username}:${password}"))
$headers = @{
    "Authorization" = "Basic $base64Auth"
    "Content-Type" = "application/json"
}

Write-Host "`nTesting authentication..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$GRAFANA_URL/api/org" -Headers $headers -ErrorAction Stop
    Write-Host "  ✓ Authenticated successfully`n" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Authentication failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "`nPlease check your credentials and try again.`n" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Function to get existing datasource by name
function Get-DataSource {
    param($name)
    try {
        $datasources = Invoke-RestMethod -Uri "$GRAFANA_URL/api/datasources" -Headers $headers -ErrorAction Stop
        return $datasources | Where-Object { $_.name -eq $name }
    } catch {
        return $null
    }
}

# Function to delete datasource
function Remove-DataSource {
    param($id)
    try {
        Invoke-RestMethod -Uri "$GRAFANA_URL/api/datasources/$id" -Method Delete -Headers $headers -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Configure Prometheus Data Source
Write-Host "Configuring Prometheus data source..." -ForegroundColor Yellow

$existingPrometheus = Get-DataSource -name "Prometheus"
if ($existingPrometheus) {
    Write-Host "  → Removing existing Prometheus data source..." -ForegroundColor Gray
    Remove-DataSource -id $existingPrometheus.id | Out-Null
}

$prometheusDatasource = @{
    name = "Prometheus"
    type = "prometheus"
    url = "http://localhost:9090"
    access = "proxy"
    isDefault = $true
    jsonData = @{
        httpMethod = "POST"
        timeInterval = "5s"
    }
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$GRAFANA_URL/api/datasources" `
        -Method Post -Headers $headers -Body $prometheusDatasource -ErrorAction Stop
    Write-Host "  ✓ Prometheus data source configured (http://localhost:9090)" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Failed to configure Prometheus: $($_.Exception.Message)" -ForegroundColor Red
}

# Configure SQLite Data Source
Write-Host "`nConfiguring SQLite data source..." -ForegroundColor Yellow

# Check if database exists
if (-not (Test-Path $DB_PATH)) {
    Write-Host "  ⚠  Database not found at: $DB_PATH" -ForegroundColor Yellow
    Write-Host "  → Data source will be configured but may show 'No Data' until database is populated`n" -ForegroundColor Gray
} else {
    Write-Host "  → Database found: $DB_PATH" -ForegroundColor Gray
}

$existingSQLite = Get-DataSource -name "SQLite"
if ($existingSQLite) {
    Write-Host "  → Removing existing SQLite data source..." -ForegroundColor Gray
    Remove-DataSource -id $existingSQLite.id | Out-Null
}

$sqliteDatasource = @{
    name = "SQLite"
    type = "frser-sqlite-datasource"
    access = "proxy"
    jsonData = @{
        path = $DB_PATH
    }
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$GRAFANA_URL/api/datasources" `
        -Method Post -Headers $headers -Body $sqliteDatasource -ErrorAction Stop
    Write-Host "  ✓ SQLite data source configured ($DB_PATH)" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Failed to configure SQLite: $($_.Exception.Message)" -ForegroundColor Red
}

# Verify data sources
Write-Host "`nVerifying data sources..." -ForegroundColor Yellow
try {
    $datasources = Invoke-RestMethod -Uri "$GRAFANA_URL/api/datasources" -Headers $headers -ErrorAction Stop
    Write-Host "  Configured data sources:" -ForegroundColor Gray
    foreach ($ds in $datasources) {
        $defaultMark = if ($ds.isDefault) { " (default)" } else { "" }
        Write-Host "    • $($ds.name) - $($ds.type)$defaultMark" -ForegroundColor White
    }
} catch {
    Write-Host "  ✗ Could not verify data sources" -ForegroundColor Red
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "   ✅ Data Source Configuration Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Refresh your Grafana dashboards (Ctrl+R)" -ForegroundColor White
Write-Host "  2. Check dashboard settings → Data source = 'Prometheus' or 'SQLite'" -ForegroundColor White
Write-Host "  3. If no data appears, run your application to generate metrics`n" -ForegroundColor White

Read-Host "Press Enter to exit"
