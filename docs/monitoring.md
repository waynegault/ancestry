# Monitoring and Metrics Guide

This guide explains how to enable the Prometheus metrics exporter, scrape metrics with Prometheus, and visualize them with Grafana.

## 1. Enable the metrics exporter

1. Add the following entries to your `.env` file (or override via environment variables):
   ```env
   PROMETHEUS_METRICS_ENABLED=true
   PROMETHEUS_METRICS_HOST=127.0.0.1
   PROMETHEUS_METRICS_PORT=9000
   PROMETHEUS_METRICS_NAMESPACE=ancestry
   ```
2. Run the automation (`python main.py`) or any action entry point. The exporter boots automatically when metrics are enabled.
3. Verify that the exporter is listening:
   ```powershell
   curl http://localhost:9000/metrics
   ```
   You should see metric names such as `ancestry_session_uptime_seconds`.

### Standalone exporter for testing

A lightweight CLI is available when you need to expose the `/metrics` endpoint without running an action loop:
```powershell
python observability/metrics_exporter.py --serve --host 127.0.0.1 --port 9000
```
Press `Ctrl+C` to stop the server when finished.

## 2. Prometheus configuration

Add a scrape job to your Prometheus configuration file (`prometheus.yml`):
```yaml
scrape_configs:
  - job_name: "ancestry-automation"
    static_configs:
      - targets: ["127.0.0.1:9000"]
```
Reload Prometheus and confirm that the job transitions to the `UP` state.

## 3. Grafana dashboard

1. Import the starter dashboard at `docs/grafana/ancestry_overview.json` into Grafana.
2. Select your Prometheus data source during import.
3. The dashboard renders:
   - API latency (P95) using `ancestry_api_action_latency_seconds`
   - Cache hit ratio from `ancestry_cache_hit_ratio`
   - Circuit breaker state and trips from `ancestry_circuit_breaker_state`
   - Action throughput via `ancestry_action_processed_total`

Feel free to duplicate panels and tailor them to your environment.

## 4. Troubleshooting

- **Exporter does not start**: confirm `PROMETHEUS_METRICS_ENABLED=true` and that port `9100` is available.
- **No metrics in Prometheus**: run an action flow so instrumentation emits samples, or use the standalone exporter while triggering code paths manually.
- **High cardinality warnings**: endpoint labels are sanitized automatically (`/foo/bar` becomes `foo_bar`). If you add new metrics, keep label sets small and bounded.

## 5. Next steps

- Wire the exporter into your CI environment to capture regression trends.
- Add alerts around latency histograms and circuit breaker trips to detect production issues quickly.
