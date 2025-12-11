param(
    [string]$PrometheusExe = $Env:PROMETHEUS_EXE_PATH ?? 'C:\\Programs\\Prometheus\\prometheus.exe',
    [int]$PrometheusPort = 9090
)

$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$promConfig = Join-Path $root 'observability/prometheus/prometheus.yml'
$exporterScript = Join-Path $root 'Cache/run_metrics_exporter_once.py'

if (-not (Test-Path $promConfig)) {
    throw "Prometheus config not found at $promConfig"
}
if (-not (Test-Path $PrometheusExe)) {
    throw "Prometheus executable not found at $PrometheusExe (override with PROMETHEUS_EXE_PATH or -PrometheusExe)"
}
if (-not (Test-Path $exporterScript)) {
    throw "Exporter seed script not found at $exporterScript"
}

$promArgs = @("--config.file=$promConfig", "--web.listen-address=localhost:$PrometheusPort")

$promProc = $null
$exporterProc = $null

function Wait-PrometheusReady {
    param(
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $ready = $false
    while (-not $ready -and (Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri "http://localhost:$PrometheusPort/-/ready" -UseBasicParsing -TimeoutSec 3
            if ($resp.StatusCode -eq 200) { $ready = $true; break }
        } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $ready) { throw "Prometheus did not become ready within $TimeoutSeconds seconds" }
}

function Wait-AncestryTargetUp {
    param(
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $content = Invoke-WebRequest -Uri "http://localhost:$PrometheusPort/api/v1/targets" -UseBasicParsing -TimeoutSec 3 | Select-Object -ExpandProperty Content
            $json = $content | ConvertFrom-Json
            $targets = $json.data.activeTargets
            if ($targets) {
                foreach ($t in $targets) {
                    if ($t.labels.job -eq 'ancestry' -and $t.health -eq 'up') { return }
                }
            }
        } catch { }
        Start-Sleep -Seconds 2
    }
    throw "Ancestry target did not reach UP within $TimeoutSeconds seconds"
}

try {
    Write-Host "Starting Prometheus..." -ForegroundColor Cyan
    $promProc = Start-Process -FilePath $PrometheusExe -ArgumentList $promArgs -PassThru -WindowStyle Hidden
    Wait-PrometheusReady

    Write-Host "Starting metrics exporter and seeding metrics..." -ForegroundColor Cyan
    $exporterProc = Start-Process -FilePath 'python' -ArgumentList $exporterScript -WorkingDirectory $root -PassThru -WindowStyle Hidden

    Write-Host "Waiting for ancestry target to be UP..." -ForegroundColor Cyan
    Wait-AncestryTargetUp

    Write-Host "Running Prometheus smoke tests..." -ForegroundColor Cyan
    & python -m testing.test_prometheus_smoke
    $exitCode = $LASTEXITCODE
}
finally {
    if ($exporterProc -and -not $exporterProc.HasExited) {
        Write-Host "Stopping exporter (PID $($exporterProc.Id))" -ForegroundColor Yellow
        Stop-Process -Id $exporterProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($promProc -and -not $promProc.HasExited) {
        Write-Host "Stopping Prometheus (PID $($promProc.Id))" -ForegroundColor Yellow
        Stop-Process -Id $promProc.Id -Force -ErrorAction SilentlyContinue
    }
}

if ($exitCode -ne $null) {
    exit $exitCode
}
