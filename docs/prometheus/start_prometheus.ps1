<#
Starts a local Prometheus server using docs/prometheus/prometheus.yml
Scrapes the app's exporter at 127.0.0.1:9001 and serves Prometheus at 9090 by default.

Usage:
  pwsh ./docs/prometheus/start_prometheus.ps1

Assumptions:
  - prometheus.exe is either alongside this script OR at C:\Programs\Prometheus\prometheus.exe
  - docs/prometheus/prometheus.yml exists (added in repo)
#>

[CmdletBinding()]
param(
    [string]$PrometheusExe,
    [string]$Config = "$PSScriptRoot/prometheus.yml",
    [int]$Port = 9090
)

$ErrorActionPreference = 'Stop'

function Resolve-PrometheusPath {
    param([string]$Override)
    if ($Override) {
        return (Resolve-Path $Override).Path
    }
    $candidates = @()
    if ($PSScriptRoot) {
        $candidates += Join-Path -Path $PSScriptRoot -ChildPath 'prometheus.exe'
    }
    $candidates += 'C:/Programs/Prometheus/prometheus.exe'
    foreach ($path in $candidates) {
        if (Test-Path $path) { return (Resolve-Path $path).Path }
    }
    throw "prometheus.exe not found. Provide -PrometheusExe or place it in docs/prometheus or C:/Programs/Prometheus."
}

try {
    $exePath = Resolve-PrometheusPath -Override $PrometheusExe
    $cfgPath = (Resolve-Path $Config).Path

    Write-Host "Using prometheus.exe: $exePath" -ForegroundColor Cyan
    Write-Host "Using config:       $cfgPath" -ForegroundColor Cyan
    Write-Host "Serving on:         http://localhost:$Port" -ForegroundColor Cyan

    $args = @("--config.file=$cfgPath", "--web.listen-address=:${Port}")
    $proc = Start-Process -FilePath $exePath -ArgumentList $args -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 2

    if ($proc.HasExited) {
        throw "Prometheus exited immediately (code $($proc.ExitCode)). Check paths and config."
    }

    Write-Host "✅ Prometheus started (PID $($proc.Id))." -ForegroundColor Green
    Write-Host "UI: http://localhost:$Port" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop if running in a terminal session." -ForegroundColor Yellow
}
catch {
    Write-Error $_
    exit 1
}
