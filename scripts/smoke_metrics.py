import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config_schema import ObservabilityConfig
from observability.metrics_exporter import (
    apply_observability_settings,
    start_metrics_exporter,
)
from observability.metrics_registry import configure_metrics, metrics, reset_metrics

cfg = ObservabilityConfig(
    enable_prometheus_metrics=True,
    metrics_export_host="127.0.0.1",
    metrics_export_port=9001,
    metrics_namespace="ancestry",
    auto_start_prometheus=False,
)

apply_observability_settings(cfg)
reset_metrics()
configure_metrics(cfg)

started = start_metrics_exporter(cfg.metrics_export_host, cfg.metrics_export_port)
if not started:
    print(f"Failed to start exporter on {cfg.metrics_export_host}:{cfg.metrics_export_port}")
    sys.exit(1)

print(f"✅ Exporter started on {cfg.metrics_export_host}:{cfg.metrics_export_port}")
print("Emitting smoke metrics every 5s. Press Ctrl+C to stop.")

try:
    while True:
        # Increment some metrics to show activity in dashboards
        m = metrics()
        m.api_requests.inc(endpoint="smoke_test", method="GET", result="success")
        m.database_rows.inc(operation="select", amount=10)
        m.action_processed.inc(action="smoke_test", result="success")
        m.cache_operations.inc(service="ancestry", endpoint="smoke_test", operation="hit")

        # Add some duration metrics
        m.api_latency.observe(endpoint="smoke_test", status_family="2xx", seconds=0.15)
        m.database_query_latency.observe(operation="select", seconds=0.05)
        m.ai_quality.observe(provider="google", prompt_key="smoke_test", variant="default", score=85.0)

        time.sleep(5)
except KeyboardInterrupt:
    print("\nStopping exporter...")
