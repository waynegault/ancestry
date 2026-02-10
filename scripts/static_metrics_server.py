"""Serve a static Prometheus metrics payload for testing.

Exposes ancestry_api_requests_total with a non-zero sample so Prometheus smoke tests can pass without
running the full app. Intended for local/dev use only.
"""


import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.venv_bootstrap import ensure_venv

ensure_venv(project_root=_PROJECT_ROOT)

import http.server
import socketserver
import threading
import time

METRIC_HEADER = "# HELP ancestry_api_requests_total API request counter\n# TYPE ancestry_api_requests_total counter\n"

_counter = [5]


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/metrics"):
            _counter[0] += 1
            body = (
                f"{METRIC_HEADER}"
                f"ancestry_api_requests_total{{endpoint=\"combined_details\",method=\"GET\",result=\"success\"}} {_counter[0]}\n"
            )
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    @staticmethod
    def log_message(format: str, *args: object) -> None:  # pragma: no cover - silence default logging
        _ = (format, args)


def main() -> None:
    with socketserver.TCPServer(("0.0.0.0", 9001), Handler) as httpd:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print("Static metrics server running on http://0.0.0.0:9001/metrics")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        httpd.shutdown()


if __name__ == "__main__":
    main()
