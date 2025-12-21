"""Helper utilities backing the interactive CLI actions in main.py.

The helpers live outside main.py so the entry point can focus on session
orchestration while these routines handle log maintenance, analytics views,
metrics dashboards, and other ancillary utilities.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections.abc import Mapping
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from logging import StreamHandler
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional, Protocol, TextIO, cast
from unittest import mock
from urllib import request as urllib_request

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from core.session_manager import SessionManager

from core.logging_config import setup_logging
from testing.test_framework import TestSuite, create_standard_test_runner


class GrafanaCheckerProtocol(Protocol):
    """Protocol describing the optional grafana checker helpers."""

    def ensure_dashboards_imported(self, force: bool = False) -> bool: ...

    def ensure_data_sources_configured(
        self, prometheus_url: str | None = None, sqlite_path: str | None = None
    ) -> bool: ...

    def check_grafana_status(self) -> Mapping[str, Any]: ...

    def ensure_grafana_ready(self, *, auto_setup: bool = False, silent: bool = True) -> None: ...


class LogMaintenanceMixin:
    """Mixin for log maintenance operations."""

    _logger: logging.Logger

    def clear_log_file(self) -> tuple[bool, Optional[str]]:
        """Clear the active log file by flushing/closing the handler."""

        cleared = False
        log_file_handler: Optional[logging.FileHandler] = None
        log_file_path: Optional[str] = None

        # Check instance logger first
        handlers = self._logger.handlers
        # Fallback to root logger if no handlers found on instance logger
        if not handlers:
            handlers = logging.getLogger().handlers

        try:
            for handler in handlers:
                if isinstance(handler, logging.FileHandler):
                    log_file_handler = handler
                    log_file_path = handler.baseFilename
                    break
            if log_file_handler is not None and log_file_path is not None:
                log_file_handler.flush()
                log_file_handler.close()
                with Path(log_file_path).open("w", encoding="utf-8"):
                    pass
                cleared = True
        except PermissionError as permission_error:
            self._logger.warning("Permission denied clearing log '%s': %s", log_file_path, permission_error)
        except OSError as io_error:
            self._logger.warning("IOError clearing log '%s': %s", log_file_path, io_error)
        except Exception as error:  # pragma: no cover - defensive logging only
            self._logger.warning("Error clearing log '%s': %s", log_file_path, error, exc_info=True)
        return cleared, log_file_path

    def toggle_log_level(self) -> None:
        """Toggle console log level between DEBUG and INFO."""

        os.system("cls" if os.name == "nt" else "clear")

        # Check instance logger first, then fallback to root logger
        target_logger = self._logger
        handlers = target_logger.handlers
        if not handlers:
            target_logger = logging.getLogger()
            handlers = target_logger.handlers

        if handlers:
            console_handler: Optional[StreamHandler[TextIO]] = None
            for handler in handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler_typed = cast(logging.StreamHandler[TextIO], handler)
                    # Check if it's a console handler (stderr or stdout)
                    # We relax the check to include any stream handler that isn't a file handler
                    # effectively, but checking stream name or type is safer.
                    # Also check if stream is wrapped (e.g. by colorama)
                    stream = handler_typed.stream
                    is_stderr = stream == sys.stderr
                    is_stdout = stream == sys.stdout
                    # Some environments wrap stderr, so we check if it looks like a console stream
                    is_console = is_stderr or is_stdout or getattr(stream, "name", "") in {"<stderr>", "<stdout>"}

                    if is_console:
                        console_handler = handler_typed
                        break

            if console_handler:
                current_level = console_handler.level
                # If level is NOTSET (0), it inherits. We assume effective level if possible,
                # but handler.level is what we set.
                # If it's 0, we treat it as INFO for toggling purposes (toggle to DEBUG)
                effective_level = current_level if current_level != logging.NOTSET else logging.INFO

                new_level = logging.DEBUG if effective_level > logging.DEBUG else logging.INFO
                new_level_name = logging.getLevelName(new_level)

                setup_logging(log_level=new_level_name, allow_env_override=False)
                self._logger.info("Console log level toggled to: %s", new_level_name)
            else:
                self._logger.warning("Could not find console handler to toggle level. Handlers found: %s", handlers)
        else:
            print("WARNING: Logger not ready or has no handlers.", file=sys.stderr)

    def clear_app_log_menu(self) -> None:
        """Menu action to clear the application log file."""
        cleared, log_path = self.clear_log_file()
        if cleared:
            print(f"\n✅ Log file cleared: {log_path}\n")
        elif log_path:
            print(f"\n❌ Failed to clear log file: {log_path}\n")
        else:
            print("\n⚠️  No log file handler found.\n")
        input("Press Enter to continue...")


class TestRunnerMixin:
    """Mixin for test execution operations."""

    _logger: logging.Logger

    def run_main_tests(self) -> None:
        """Run the relocated main.py test suite."""

        try:
            main_module = importlib.import_module("main")
            run_main_suite = getattr(main_module, "run_comprehensive_tests", None)
            if not callable(run_main_suite):
                raise AttributeError("main.run_comprehensive_tests not available")
            print("\n" + "=" * 60)
            print("RUNNING MAIN.PY INTERNAL TESTS")
            print("=" * 60)
            result = run_main_suite()
            if result:
                print("\n🎉 All main.py tests completed successfully!")
            else:
                print("\n⚠️ Some main.py tests failed. Check output above.")
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.error("Error running main.py tests: %s", exc)
            print(f"Error running main.py tests: {exc}")
        print("\nReturning to main menu...")
        input("Press Enter to continue...")

    def run_all_tests(self) -> None:
        """Invoke the full run_all_tests.py orchestrator via subprocess."""

        try:
            print("\n" + "=" * 60)
            print("RUNNING ALL MODULE TESTS")
            print("=" * 60)
            result = subprocess.run(
                [sys.executable, "run_all_tests.py"],
                check=False,
                capture_output=False,
                text=True,
            )
            if result.returncode == 0:
                print("\n🎉 All module tests completed successfully!")
            else:
                print(f"\n⚠️ Some tests failed (exit code: {result.returncode})")
        except FileNotFoundError:
            print("Error: run_all_tests.py not found in current directory.")
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.error("Error running all tests: %s", exc)
            print(f"Error running all tests: {exc}")
        print("\nReturning to main menu...")
        input("Press Enter to continue...")

    @staticmethod
    def clear_test_cache() -> None:
        """Clear the run_all_tests result cache to force retesting all modules."""
        cache_file = Path("Cache/test_results_cache.json")
        if cache_file.exists():
            try:
                cache_file.unlink()
                print("\n✅ Test result cache cleared successfully.")
                print("   Next test run will retest all modules.\n")
            except OSError as e:
                print(f"\n❌ Failed to clear test cache: {e}\n")
        else:
            print("\n📭 No test cache found (already empty).\n")
        input("Press Enter to continue...")


class AnalyticsMixin:
    """Mixin for visualization and analytics operations."""

    _logger: logging.Logger
    _grafana_checker: Optional[GrafanaCheckerProtocol]

    def open_graph_visualization(self) -> None:
        """Launch the local graph visualization web server."""

        server: Optional[ThreadingHTTPServer] = None
        try:
            docs_dir = Path(__file__).resolve().parents[1] / "docs"
            graph_json = docs_dir / "code_graph.json"
            graph_html = docs_dir / "visualize_code_graph.html"

            if not graph_json.exists():
                print("⚠️  docs/code_graph.json not found. Run: python scripts/update_code_graph.py")
                input("\nPress Enter to return...")
                return

            if not graph_html.exists():
                print("⚠️  docs/visualize_code_graph.html is missing. Regenerate the viewer or pull latest docs.")
                input("\nPress Enter to return...")
                return

            root_dir = docs_dir
            preferred_port = 8765

            local_logger = self._logger

            class GraphRequestHandler(SimpleHTTPRequestHandler):
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    super().__init__(*args, directory=str(root_dir), **kwargs)

                def log_message(self, format: str, *args: Any) -> None:
                    client_host, client_port = getattr(self, "client_address", ("?", "?"))
                    local_logger.debug(
                        "Graph server %s:%s - %s",
                        client_host,
                        client_port,
                        format % args,
                    )

            for candidate_port in range(preferred_port, preferred_port + 20):
                try:
                    server = ThreadingHTTPServer(("127.0.0.1", candidate_port), GraphRequestHandler)
                    preferred_port = candidate_port
                    break
                except OSError:
                    continue

            if server is None:
                print("❌ Unable to start the graph visualization server (no open ports).")
                self._logger.error("Graph visualization server failed to start: no open ports available")
                input("\nPress Enter to continue...")
                return

            server_thread = threading.Thread(
                target=server.serve_forever,
                name="GraphVisualizationServer",
                daemon=True,
            )
            server_thread.start()

            url = f"http://127.0.0.1:{preferred_port}/visualize_code_graph.html"
            print("\n" + "=" * 70)
            print("CODE GRAPH VISUALIZATION")
            print("=" * 70)
            print(f"Serving from: {root_dir}")
            print(f"Graph source: {graph_json}")
            print(f"URL: {url}")
            print("\nPress Enter when you are finished exploring the visualization.")
            print("=" * 70)

            try:
                webbrowser.open(url, new=1)
            except webbrowser.Error as browser_err:
                self._logger.warning("Unable to open browser automatically: %s", browser_err)
                print("⚠️  Please open the URL manually in your browser.")

            input("\nPress Enter to stop the visualization server and return to the menu...")

        except Exception as graph_error:  # pragma: no cover - defensive
            self._logger.error("Error running graph visualization: %s", graph_error, exc_info=True)
            print(f"Error running graph visualization: {graph_error}")
            input("\nPress Enter to continue...")
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
                self._logger.info("Graph visualization server stopped")

    def show_analytics_dashboard(self) -> None:
        """Display the conversation analytics dashboard."""

        try:
            from core.session_manager import SessionManager
            from observability.conversation_analytics import print_analytics_dashboard

            print("\n" + "=" * 80)
            print("LOADING ANALYTICS DASHBOARD")
            print("=" * 80)

            sm = SessionManager()
            db_session = sm.db_manager.get_session()

            if not db_session:
                print("✗ Failed to get database session")
                self._logger.error("Failed to get database session for analytics")
                return

            print_analytics_dashboard(db_session)

        except Exception as exc:  # pragma: no cover - depends on optional deps
            self._logger.error("Error displaying analytics dashboard: %s", exc, exc_info=True)
            print(f"Error displaying analytics dashboard: {exc}")

        print("\nReturning to main menu...")
        input("Press Enter to continue...")

    def show_metrics_report(self) -> None:
        """Open Grafana dashboards or raw metrics depending on availability."""

        try:
            from config.config_manager import get_config_manager
            from observability.metrics_registry import (
                configure_metrics,
                get_metrics_status,
            )
        except Exception as exc:  # pragma: no cover - optional module missing
            self._logger.error("Unable to import metrics registry: %s", exc)
            print("\n⚠️  Metrics registry unavailable")
            return

        try:
            print("\n" + "=" * 70)
            print("📊 GRAFANA METRICS DASHBOARD")
            print("=" * 70)

            cfg = get_config_manager().get_observability_config()
            configure_metrics(cfg)
            status = get_metrics_status()
            metrics_url = self._build_metrics_url(cfg)

            self._print_metrics_config(cfg)
            if not self._metrics_ready(status):
                return

            prom_port = 9090
            if not self._is_prometheus_running(prom_port):
                self._print_prometheus_setup_hint(prom_port, metrics_url)
                webbrowser.open(metrics_url)
                return

            grafana_base = "http://localhost:3000"
            if not self._is_grafana_running(grafana_base):
                self._print_grafana_setup_hint(metrics_url)
                webbrowser.open(metrics_url)
                return

            self._open_grafana_dashboards(grafana_base)

        except Exception as exc:  # pragma: no cover - browser/IO issues
            self._logger.error("Error opening Grafana: %s", exc, exc_info=True)
            print(f"\n⚠️  Error: {exc}")
            print("\n" + "=" * 70 + "\n")

    @staticmethod
    def _print_metrics_config(cfg: Any) -> None:
        browse_host = AnalyticsMixin._browser_metrics_host(cfg.metrics_export_host)
        browse_hint = ""
        if browse_host != cfg.metrics_export_host:
            browse_hint = f" (browse via http://{browse_host}:{cfg.metrics_export_port})"

        print(
            f"\nCurrent config → enabled={cfg.enable_prometheus_metrics} "
            f"host={cfg.metrics_export_host}:{cfg.metrics_export_port} "
            f"namespace={cfg.metrics_namespace}{browse_hint}"
        )

    @staticmethod
    def _metrics_ready(status: dict[str, Any]) -> bool:
        if not status.get("config_enabled"):
            print("\n⚠️  Metrics collection is DISABLED in configuration")
            print("\nTo enable metrics:")
            print("  1. Add to .env: PROMETHEUS_METRICS_ENABLED=true")
            print("  2. Optionally configure: PROMETHEUS_METRICS_PORT=9001")
            print("  3. Restart the application")
            print("\n" + "=" * 70 + "\n")
            return False

        if not status.get("prometheus_available"):
            print("\n⚠️  Prometheus client library is missing — metrics cannot start")
            print("   Install: pip install prometheus-client")
            import_error = status.get("import_error")
            if import_error:
                print(f"   Import error: {import_error}")
            print("\n" + "=" * 70 + "\n")
            return False

        if not status.get("enabled"):
            print("\n⚠️  Metrics are configured but not active (unexpected state)")
            print("   Check logs for Prometheus errors and retry.")
            print("\n" + "=" * 70 + "\n")
            return False

        return True

    @staticmethod
    def _is_prometheus_running(port: int = 9090) -> bool:
        try:
            urllib_request.urlopen(f"http://localhost:{port}/-/ready", timeout=1)
            return True
        except Exception:
            return False

    @staticmethod
    def _print_prometheus_setup_hint(port: int, metrics_url: str) -> None:
        print(f"\n⚠️  Prometheus is NOT running on http://localhost:{port}")
        print("\n💡 Start Prometheus:")
        print("   pwsh ./docs/prometheus/start_prometheus.ps1")
        print("   (or run prometheus.exe with --config.file=docs/prometheus/prometheus.yml)")
        print("\nWhen running, ensure Grafana data source points to http://localhost:9090")
        print(f"\n📊 Opening raw app metrics instead at {metrics_url}")
        print("\n" + "=" * 70 + "\n")

    @staticmethod
    def _is_grafana_running(grafana_base: str) -> bool:
        try:
            urllib_request.urlopen(grafana_base, timeout=1)
            return True
        except Exception:
            return False

    @staticmethod
    def _print_grafana_setup_hint(metrics_url: str) -> None:
        print("\n⚠️  Grafana is NOT running on http://localhost:3000")
        print("\n💡 Setup Instructions:")
        print("   1. Install Prometheus (https://prometheus.io/docs/introduction/first_steps/)")
        print("      Use docs/prometheus/prometheus.yml and run: prometheus --config.file=prometheus.yml")
        print("   2. Install Grafana: https://grafana.com/grafana/download")
        print("   3. Start Prometheus (default: http://localhost:9090) and Grafana")
        print("   4. Login at http://localhost:3000 (default: admin/admin)")
        print("   5. Add Prometheus data source → http://localhost:9090")
        print("   6. Import dashboard: docs/grafana/ancestry_overview.json")
        print(f"\n📊 For now, opening raw metrics at {metrics_url}")
        print("\n" + "=" * 70 + "\n")

    @staticmethod
    def _browser_metrics_host(host: str) -> str:
        """Return a loopback host suitable for browser navigation."""

        normalized = (host or "").strip()
        if normalized in {"0.0.0.0", "::", "[::]"}:
            return "127.0.0.1"
        return normalized or "127.0.0.1"

    @staticmethod
    def _build_metrics_url(cfg: Any) -> str:
        host = AnalyticsMixin._browser_metrics_host(cfg.metrics_export_host)
        return f"http://{host}:{cfg.metrics_export_port}/metrics"

    def _open_grafana_dashboards(self, grafana_base: str) -> None:
        print("\n✅ Grafana is running!")
        print("🔍 Checking dashboards...")

        if self._grafana_checker:
            try:
                self._grafana_checker.ensure_dashboards_imported()
            except Exception as import_err:  # pragma: no cover - optional dependency
                self._logger.debug("Dashboard auto-import check: %s", import_err)

        overview_url = f"{grafana_base}/d/ancestry-overview"
        system_perf_url = f"{grafana_base}/d/ancestry-performance"
        genealogy_url = f"{grafana_base}/d/ancestry-genealogy"
        code_quality_url = f"{grafana_base}/d/ancestry-code-quality"
        database_url = f"{grafana_base}/d/ancestry-database"

        print("🌐 Opening dashboards:")
        print(f"   1. Platform Overview:           {overview_url}")
        print(f"   2. System Performance & Health: {system_perf_url}")
        print(f"   3. Genealogy Research Insights: {genealogy_url}")
        print(f"   4. Code Quality & Architecture: {code_quality_url}")
        print(f"   5. Database Summary:            {database_url}")
        print("\n💡 If dashboards show 'Not found', run: l")
        print(
            "💡 If panels show 'No data': ensure Prometheus is scraping http://127.0.0.1:9001 (see docs/prometheus/prometheus.yml) and Grafana data source points to http://localhost:9090."
        )
        print("\n" + "=" * 70 + "\n")

        webbrowser.open(overview_url)
        time.sleep(0.3)
        webbrowser.open(system_perf_url)
        time.sleep(0.3)
        webbrowser.open(genealogy_url)
        time.sleep(0.3)
        webbrowser.open(code_quality_url)
        time.sleep(0.3)
        webbrowser.open(database_url)

    def run_grafana_setup(self) -> None:
        """Run grafana checker setup flow if helper module is present."""

        if self._grafana_checker:
            status = self._grafana_checker.check_grafana_status()
            if status["ready"]:
                grafana_url = os.getenv("GRAFANA_BASE_URL", "http://localhost:3000")
                grafana_user = os.getenv("GRAFANA_USER", "admin")
                grafana_pass = os.getenv("GRAFANA_PASSWORD", "admin")
                grafana_token_set = bool(os.getenv("GRAFANA_API_TOKEN"))

                print("\n✅ Grafana is running!")
                print(f"   Dashboard URL: {grafana_url}")
                if grafana_token_set:
                    print("   API auth: GRAFANA_API_TOKEN is set")
                else:
                    print(f"   Credentials: {grafana_user} / {grafana_pass}")
                print("\n📊 Checking dashboards...")
                data_sources_ok = self._grafana_checker.ensure_data_sources_configured()
                dashboards_ok = self._grafana_checker.ensure_dashboards_imported()

                if data_sources_ok and dashboards_ok:
                    print("\n✅ Dashboard and data source checks complete!")
                else:
                    print("\n⚠️  Dashboard/data source setup could not be verified.")
                    print(
                        "   💡 Set GRAFANA_USER/GRAFANA_PASSWORD or GRAFANA_API_TOKEN so the app can access the Grafana HTTP API."
                    )
                print("\n📊 Available Dashboards:")
                print(f"   • Overview:    {grafana_url}/d/ancestry-overview")
                print(f"   • Performance: {grafana_url}/d/ancestry-performance")
                print(f"   • Genealogy:   {grafana_url}/d/ancestry-genealogy")
                print(f"   • Code Quality: {grafana_url}/d/ancestry-code-quality")
                print(f"   • Database:    {grafana_url}/d/ancestry-database")
                print(
                    "\n💡 If dashboards are empty, data sources are re-applied automatically. "
                    "Verify Prometheus is reachable at http://localhost:9090 if panels stay blank.\n"
                )
            else:
                self._grafana_checker.ensure_grafana_ready(auto_setup=False, silent=False)
        else:
            print("\n⚠️  Grafana checker module not available")
            print("Ensure grafana_checker.py is in the project root directory\n")

    def open_grafana_dashboard(self) -> None:
        """Open Grafana dashboard in web browser."""
        grafana_base = os.getenv("GRAFANA_BASE_URL", "http://localhost:3000")
        self._open_grafana_dashboards(grafana_base)


class ReviewQueueMixin:
    """Mixin for review queue operations."""

    _logger: logging.Logger

    def show_review_queue(self, session_manager: Optional[SessionManager] = None) -> None:
        """Display pending drafts for human review."""
        try:
            from core.approval_queue import ApprovalQueueService
            from core.session_manager import SessionManager

            print("\n" + "=" * 70)
            print("📋 REVIEW QUEUE - Pending AI-Generated Drafts")
            print("=" * 70)

            sm = session_manager or SessionManager()
            db_session = cast(Any, sm.db_manager).get_session()

            if not db_session:
                print("✗ Failed to get database session")
                return

            service = ApprovalQueueService(db_session)

            draft_stats = service.get_queue_stats()
            pending_drafts = service.get_pending_queue(limit=10)
            pending_facts = self._get_pending_suggested_facts(db_session, limit=10)

            self._render_review_metrics(db_session, draft_stats)
            self._render_pending_drafts(pending_drafts)
            self._render_pending_facts(pending_facts)
            self._render_contextual_draft_log_preview()

            if not pending_drafts and not pending_facts:
                print("\n✅ Nothing pending in review queues.")
                return

            print("\n" + "-" * 70)
            print("Commands: view <id> | approve <id> | reject <id> <reason> | rewrite <id> <feedback>")
            print("          fact approve <id> | fact reject <id> <reason> | refresh | back/exit/q")
            print("(Type back/exit/q to return to the main menu.)")

            while True:
                result = self._handle_review_command(input("review> ").strip())
                if result == "exit":
                    break
                if result == "refresh":
                    self.show_review_queue()
                    return

        except Exception as exc:
            self._logger.error("Error showing review queue: %s", exc, exc_info=True)
            print(f"Error: {exc}")

    def _handle_review_command(self, command: str) -> str:
        """Handle a single review queue command; returns action keyword."""
        if not command:
            return "continue"

        lowered = command.lower()
        if lowered in {"back", "exit", "q"}:
            return "exit"
        if lowered == "refresh":
            return "refresh"

        tokens = command.split()
        handlers: dict[str, Callable[[list[str]], bool | None]] = {
            "view": lambda t: self._view_draft(int(t[1])) if len(t) >= 2 else None,
            "approve": lambda t: self.approve_draft(int(t[1])) if len(t) >= 2 else None,
            "reject": lambda t: self.reject_draft(int(t[1]), " ".join(t[2:]) if len(t) > 2 else "")
            if len(t) >= 2
            else None,
            "rewrite": lambda t: self._rewrite_draft(int(t[1]), " ".join(t[2:]) if len(t) > 2 else "")
            if len(t) >= 2
            else None,
            "fact": self._handle_fact_command,
        }

        handler = handlers.get(tokens[0])
        if handler is not None:
            handler(tokens)
            return "continue"

        print("Unrecognized command. Type 'back' to exit or 'refresh' to reload queue.")
        return "continue"

    def _handle_fact_command(self, tokens: list[str]) -> None:
        if len(tokens) < 3:
            print("Unknown fact command. Use: fact approve <id> | fact reject <id> <reason>")
            return

        action, fact_id = tokens[1], int(tokens[2])
        if action == "approve":
            self.approve_suggested_fact(fact_id)
            return

        if action == "reject":
            reason = " ".join(tokens[3:]) if len(tokens) > 3 else ""
            self.reject_suggested_fact(fact_id, reason)
            return

        print("Unknown fact command. Use: fact approve <id> | fact reject <id> <reason>")

    def _view_draft(self, draft_id: int) -> bool:
        """View full draft content and conversation context."""
        try:
            from core.database import ConversationLog, DraftReply, Person
            from core.session_manager import SessionManager

            sm = SessionManager()
            db_session = sm.db_manager.get_session()
            if not db_session:
                print("✗ Failed to get database session")
                return False

            # Get the draft
            draft = db_session.query(DraftReply).filter(DraftReply.id == draft_id).first()
            if not draft:
                print(f"❌ Draft {draft_id} not found")
                return False

            # Get person info
            person = db_session.query(Person).filter(Person.id == draft.people_id).first()
            person_name = getattr(person, "display_name", None) or getattr(person, "username", "Unknown")

            # Get conversation history
            conv_logs = (
                db_session.query(ConversationLog)
                .filter(ConversationLog.conversation_id == draft.conversation_id)
                .order_by(ConversationLog.latest_timestamp.asc())
                .limit(10)
                .all()
            )

            print("\n" + "=" * 70)
            print(f"📄 DRAFT #{draft_id} - To: {person_name}")
            print("=" * 70)
            print(f"Status: {draft.status}")
            print(f"Created: {draft.created_at.strftime('%Y-%m-%d %H:%M')}")
            print(f"Conversation ID: {draft.conversation_id}")

            if conv_logs:
                print("\n📨 CONVERSATION HISTORY:")
                print("-" * 50)
                for log in conv_logs:
                    direction = "←" if log.direction.value == "IN" else "→"
                    ts = log.latest_timestamp.strftime("%Y-%m-%d %H:%M")
                    content = log.latest_message_content or "(no content)"
                    # Truncate very long messages
                    if len(content) > 500:
                        content = content[:500] + "..."
                    print(f"\n{direction} [{ts}]")
                    print(f"   {content}")

            print("\n" + "-" * 50)
            print("📝 DRAFT REPLY:")
            print("-" * 50)
            print(draft.content)
            print("-" * 50)

            return True

        except Exception as exc:
            self._logger.error("Error viewing draft: %s", exc, exc_info=True)
            print(f"Error viewing draft: {exc}")
            return False

    def _rewrite_draft(self, draft_id: int, feedback: str) -> bool:
        """Regenerate a draft using AI with user feedback."""
        try:
            from ai.ai_interface import generate_genealogical_reply

            rewrite_context = self._prepare_rewrite_context(draft_id, feedback)
            if rewrite_context is None:
                return False

            db_session, sm, draft, person_name, genealogical_data, conv_context, last_message = rewrite_context

            print(f"🔄 Regenerating draft for {person_name} with feedback: '{feedback}'...")

            new_reply = generate_genealogical_reply(
                conversation_context=conv_context,
                user_last_message=last_message,
                genealogical_data_str=genealogical_data,
                session_manager=sm,
            )

            if not new_reply:
                print("❌ AI failed to generate a new reply")
                return False

            from datetime import datetime, timezone

            draft.content = new_reply
            draft.created_at = datetime.now(timezone.utc)  # Reset creation time
            db_session.commit()

            print("\n" + "=" * 70)
            print(f"✅ DRAFT #{draft_id} REWRITTEN")
            print("=" * 70)
            print("\n📝 NEW DRAFT:")
            print("-" * 50)
            print(new_reply)
            print("-" * 50)
            print("\nUse 'approve', 'reject', or 'rewrite' to continue.")
            return True

        except Exception as exc:
            self._logger.error("Error rewriting draft: %s", exc, exc_info=True)
            print(f"Error rewriting draft: {exc}")
            return False

    def _prepare_rewrite_context(
        self, draft_id: int, feedback: str
    ) -> Optional[tuple[Any, Any, Any, str, str, str, str]]:
        import json

        from core.database import ConversationLog, DraftReply, Person
        from core.session_manager import SessionManager

        if not feedback.strip():
            print("⚠️  Please provide feedback for the rewrite, e.g.: rewrite 1 make it more formal")
            return None

        sm = SessionManager()
        db_session = sm.db_manager.get_session()
        if not db_session:
            print("✗ Failed to get database session")
            return None

        draft = db_session.query(DraftReply).filter(DraftReply.id == draft_id).first()
        if not draft:
            print(f"❌ Draft {draft_id} not found")
            return None

        if draft.status != "PENDING":
            print(f"⚠️  Draft {draft_id} is already {draft.status} - cannot rewrite")
            return None

        person = db_session.query(Person).filter(Person.id == draft.people_id).first()
        if not person:
            print(f"❌ Person for draft {draft_id} not found")
            return None

        person_name = getattr(person, "display_name", None) or getattr(person, "username", "Unknown")

        conv_logs = (
            db_session.query(ConversationLog)
            .filter(ConversationLog.conversation_id == draft.conversation_id)
            .order_by(ConversationLog.latest_timestamp.desc())
            .limit(5)
            .all()
        )

        conversation_context, user_last_message = self._build_rewrite_context(conv_logs)
        if not user_last_message:
            user_last_message = "(No inbound message found)"

        genealogical_data = json.dumps(
            {
                "person_name": person_name,
                "relationship": getattr(person, "relationship", None),
                "shared_dna": getattr(person, "total_cm", None),
            }
        )

        rewrite_context = (
            f"{conversation_context}\n\n"
            f"[REWRITE INSTRUCTIONS]: The previous draft was rejected. "
            f"User feedback: {feedback}\n"
            f"Previous draft that needs improvement:\n{draft.content}"
        )

        return db_session, sm, draft, person_name, genealogical_data, rewrite_context, user_last_message

    @staticmethod
    def _build_rewrite_context(conv_logs: list[Any]) -> tuple[str, str]:
        conversation_context = ""
        user_last_message = ""

        for log in reversed(conv_logs):
            direction = "THEM" if log.direction.value == "IN" else "ME"
            content = log.latest_message_content or ""
            conversation_context += f"[{direction}]: {content}\n"
            if log.direction.value == "IN" and not user_last_message:
                user_last_message = content

        return conversation_context, user_last_message

    def _render_contextual_draft_log_preview(self, limit: int = 5) -> None:
        """Show recent contextual draft log entries (draft-only stub)."""
        try:
            draft_path = Path("Logs/contextual_drafts.jsonl")
            if not draft_path.exists():
                return

            entries: list[dict[str, Any]] = []
            with draft_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                        entries.append(record)
                    except json.JSONDecodeError:
                        continue

            if not entries:
                return

            entries = entries[-limit:]
            print(f"\nContextual Draft Log (last {len(entries)}):")
            for item in entries:
                ts = item.get("timestamp", "?")
                pid = item.get("person_id", "?")
                conf = item.get("confidence", "?")
                reason = item.get("quality_reason", item.get("reason", ""))
                preview = (item.get("draft_text", "") or "")[:80]
                print(f" - {ts} | person {pid} | conf {conf} | {reason} | {preview}...")
        except Exception as exc:  # pragma: no cover - defensive only
            self._logger.debug("Could not render contextual draft log preview: %s", exc)

    @staticmethod
    def _get_pending_suggested_facts(db_session: Any, limit: int = 10) -> list[dict[str, Any]]:
        from sqlalchemy import asc

        from core.database import FactStatusEnum, Person, SuggestedFact

        query = (
            db_session.query(SuggestedFact, Person)
            .join(Person, SuggestedFact.people_id == Person.id)
            .filter(SuggestedFact.status == FactStatusEnum.PENDING)
            .order_by(asc(SuggestedFact.created_at))
            .limit(limit)
        )

        pending: list[dict[str, Any]] = []
        for fact, person in query.all():
            pending.append(
                {
                    "id": fact.id,
                    "person_id": person.id,
                    "person_name": getattr(person, "display_name", None) or getattr(person, "username", "?"),
                    "fact_type": getattr(fact.fact_type, "name", str(fact.fact_type)),
                    "new_value": fact.new_value,
                    "confidence": fact.confidence_score,
                    "created_at": fact.created_at,
                    "status": fact.status,
                }
            )
        return pending

    @staticmethod
    def _render_pending_drafts(pending_drafts: list[Any]) -> None:
        if not pending_drafts:
            print("\n📝 Pending Drafts: none")
            return

        print(f"\n📝 Pending Drafts ({len(pending_drafts)} shown):")
        print("-" * 70)
        for i, draft in enumerate(pending_drafts, 1):
            priority_icons = {"critical": "🔴", "high": "🟠", "normal": "🟡", "low": "🟢"}
            icon = priority_icons.get(getattr(draft.priority, "value", ""), "⚪")
            print(f"\n{i}. [{icon} {draft.priority.value.upper()}] ID: {draft.draft_id}")
            print(f"   To: {draft.person_name} (ID: {draft.person_id})")
            print(f"   Confidence: {draft.ai_confidence}%")
            print(f"   Created: {draft.created_at.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Content preview: {draft.content[:100]}...")

    @staticmethod
    def _render_pending_facts(pending_facts: list[dict[str, Any]]) -> None:
        if not pending_facts:
            print("\n🧾 Pending Suggested Facts: none")
            return

        print(f"\n🧾 Pending Suggested Facts ({len(pending_facts)} shown):")
        print("-" * 70)
        for i, fact in enumerate(pending_facts, 1):
            print(f"\n{i}. ID: {fact['id']} | Person: {fact['person_name']} (ID: {fact['person_id']})")
            print(f"   Type: {fact['fact_type']} | Confidence: {fact.get('confidence', '?')}")
            print(f"   Created: {fact['created_at'].strftime('%Y-%m-%d %H:%M')}")
            print(f"   Value: {fact['new_value'][:120]}...")

    @staticmethod
    def _render_review_metrics(db_session: Any, draft_stats: Any) -> None:
        from sqlalchemy import func, or_

        from core.database import (
            ConversationLog,
            ConversationState,
            ConversationStatusEnum,
            DraftReply,
            FactStatusEnum,
            Person,
            PersonStatusEnum,
            SuggestedFact,
        )

        critical_alerts = (
            db_session.query(func.count(ConversationState.id))
            .filter(or_(ConversationState.status == ConversationStatusEnum.HUMAN_REVIEW, ConversationState.safety_flag))
            .scalar()
            or 0
        )
        opt_outs = (
            db_session.query(func.count(Person.id)).filter(Person.status == PersonStatusEnum.DESIST).scalar() or 0
        )
        facts_pending = (
            db_session.query(func.count(SuggestedFact.id))
            .filter(SuggestedFact.status == FactStatusEnum.PENDING)
            .scalar()
            or 0
        )
        facts_approved = (
            db_session.query(func.count(SuggestedFact.id))
            .filter(SuggestedFact.status == FactStatusEnum.APPROVED)
            .scalar()
            or 0
        )
        facts_rejected = (
            db_session.query(func.count(SuggestedFact.id))
            .filter(SuggestedFact.status == FactStatusEnum.REJECTED)
            .scalar()
            or 0
        )
        approvals = (
            db_session.query(func.count(DraftReply.id))
            .filter(DraftReply.status.in_(["APPROVED", "AUTO_APPROVED", "SENT"]))
            .scalar()
            or 0
        )
        sends = (
            db_session.query(func.count(ConversationLog.id)).filter(ConversationLog.direction == "OUT").scalar() or 0
        )

        print("\n📊 Review Metrics:")
        print(f"   Critical alerts / Human review: {critical_alerts}")
        print(f"   Opt-outs detected: {opt_outs}")
        print(f"   Suggested facts: pending {facts_pending}, approved {facts_approved}, rejected {facts_rejected}")
        print(
            f"   Drafts: pending {draft_stats.pending_count}, auto-approved {draft_stats.auto_approved_count}, "
            f"approved today {draft_stats.approved_today}, rejected today {draft_stats.rejected_today}, expired {draft_stats.expired_count}"
        )
        print(f"   Draft approvals (all time): {approvals}")
        print(f"   Outbound sends (all time): {sends}")

    def approve_suggested_fact(self, fact_id: int) -> bool:
        """Approve a SuggestedFact (mark APPROVED)."""
        try:
            from datetime import datetime, timezone

            from core.database import FactStatusEnum, SuggestedFact
            from core.session_manager import SessionManager

            sm = SessionManager()
            db_session = sm.db_manager.get_session()
            if not db_session:
                print("✗ Failed to get database session")
                return False

            fact = db_session.query(SuggestedFact).filter(SuggestedFact.id == fact_id).first()
            if not fact:
                print(f"❌ SuggestedFact {fact_id} not found")
                return False
            if fact.status != FactStatusEnum.PENDING:
                print(f"⚠️  SuggestedFact {fact_id} is already {fact.status.name}")
                return False

            fact.status = FactStatusEnum.APPROVED
            fact.updated_at = datetime.now(timezone.utc)
            db_session.commit()
            print(f"✅ SuggestedFact {fact_id} approved")
            return True
        except Exception as exc:  # pragma: no cover - defensive only
            self._logger.error("Error approving SuggestedFact: %s", exc, exc_info=True)
            print(f"Error approving fact: {exc}")
            return False

    def reject_suggested_fact(self, fact_id: int, reason: str = "") -> bool:
        """Reject a SuggestedFact (mark REJECTED)."""
        try:
            from datetime import datetime, timezone

            from core.database import FactStatusEnum, SuggestedFact
            from core.session_manager import SessionManager

            sm = SessionManager()
            db_session = sm.db_manager.get_session()
            if not db_session:
                print("✗ Failed to get database session")
                return False

            fact = db_session.query(SuggestedFact).filter(SuggestedFact.id == fact_id).first()
            if not fact:
                print(f"❌ SuggestedFact {fact_id} not found")
                return False
            if fact.status != FactStatusEnum.PENDING:
                print(f"⚠️  SuggestedFact {fact_id} is already {fact.status.name}")
                return False

            fact.status = FactStatusEnum.REJECTED
            fact.updated_at = datetime.now(timezone.utc)
            db_session.commit()
            if reason:
                print(f"❌ SuggestedFact {fact_id} rejected: {reason}")
            else:
                print(f"❌ SuggestedFact {fact_id} rejected")
            return True
        except Exception as exc:  # pragma: no cover - defensive only
            self._logger.error("Error rejecting SuggestedFact: %s", exc, exc_info=True)
            print(f"Error rejecting fact: {exc}")
            return False

    def approve_draft(self, draft_id: int, edited_content: Optional[str] = None) -> bool:
        """Approve a draft for sending."""
        try:
            from core.approval_queue import ApprovalQueueService
            from core.session_manager import SessionManager

            sm = SessionManager()
            db_session = sm.db_manager.get_session()

            if not db_session:
                print("✗ Failed to get database session")
                return False

            service = ApprovalQueueService(db_session)
            result = service.approve(draft_id, reviewer="operator", edited_content=edited_content)

            if result.success:
                print(f"✅ {result.message}")
                return True
            print(f"❌ {result.message}")
            return False

        except Exception as exc:
            self._logger.error("Error approving draft: %s", exc, exc_info=True)
            print(f"Error: {exc}")
            return False

    def reject_draft(self, draft_id: int, reason: str = "") -> bool:
        """Reject a draft."""
        try:
            from core.approval_queue import ApprovalQueueService
            from core.session_manager import SessionManager

            sm = SessionManager()
            db_session = sm.db_manager.get_session()

            if not db_session:
                print("✗ Failed to get database session")
                return False

            service = ApprovalQueueService(db_session)
            result = service.reject(draft_id, reviewer="operator", reason=reason)

            if result.success:
                print(f"✅ {result.message}")
                return True
            print(f"❌ {result.message}")
            return False

        except Exception as exc:
            self._logger.error("Error rejecting draft: %s", exc, exc_info=True)
            print(f"Error: {exc}")
            return False

    def run_dry_run_validation(self, limit: int = 50) -> None:
        """Run dry-run validation against historical conversations."""
        try:
            from core.session_manager import SessionManager
            from scripts.dry_run_validation import DryRunProcessor

            print("\n" + "=" * 70)
            print("🧪 DRY-RUN VALIDATION")
            print("=" * 70)
            print(f"\nProcessing up to {limit} historical conversations...")

            sm = SessionManager()
            db_session = sm.db_manager.get_session()

            if not db_session:
                print("✗ Failed to get database session")
                return

            processor = DryRunProcessor(db_session)
            summary = processor.run(limit=limit)
            processor.print_report()

            # Record metrics for quality tracking
            self._record_validation_metrics(summary)

        except Exception as exc:
            self._logger.error("Error running dry-run validation: %s", exc, exc_info=True)
            print(f"Error: {exc}")

    def _record_validation_metrics(self, summary: Any) -> None:
        """Record validation metrics for quality tracking."""
        try:
            from core.metrics_collector import get_metrics_registry

            registry = get_metrics_registry()
            registry.record_metric("DryRunValidation", "total_conversations", float(summary.total_conversations))
            registry.record_metric("DryRunValidation", "successful_drafts", float(summary.successful_drafts))
            registry.record_metric("DryRunValidation", "avg_confidence", summary.avg_confidence)
            registry.record_metric("DryRunValidation", "opt_outs_detected", float(summary.opt_outs_detected))
            registry.record_metric("DryRunValidation", "errors_encountered", float(summary.errors_encountered))

            self._logger.info("Validation metrics recorded to MetricRegistry")
        except Exception as exc:
            self._logger.debug("Could not record validation metrics: %s", exc)

    def launch_review_web_ui(self, _session_manager: Optional[SessionManager] = None) -> None:
        """Launch the browser-based review queue interface."""
        try:
            from ui.review_server import run_server

            print("\n🌐 Launching Review Queue Web Interface...")
            print("   The browser will open automatically.")
            print("   Press Ctrl+C to stop the server and return to the menu.\n")

            run_server(port=5000, open_browser_on_start=True)

        except ImportError as exc:
            self._logger.error("Failed to import review server: %s", exc)
            print("❌ Error: Could not load web interface.")
            print("   Make sure Flask is installed: pip install flask")
        except KeyboardInterrupt:
            print("\n\n✅ Web server stopped.")
        except Exception as exc:
            self._logger.error("Error launching web review UI: %s", exc, exc_info=True)
            print(f"Error: {exc}")
        # Note: No input prompt - return directly to menu


class CacheStatsMixin:
    """Mixin for cache statistics operations."""

    _logger: logging.Logger
    _CACHE_KIND_ICONS: ClassVar[dict[str, str]] = {
        "disk": "📁",
        "memory": "🧠",
        "session": "🔐",
        "system": "⚙️",
        "gedcom": "🌳",
        "performance": "📊",
        "database": "🗄️",
        "retention": "🧹",
    }

    @staticmethod
    def format_cache_stat_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.2f}"
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, (list, tuple, set)):
            return f"{len(value)} items"
        if isinstance(value, dict):
            preview_items = list(value.items())[:3]
            preview = ", ".join(f"{k}={v}" for k, v in preview_items)
            if len(value) > 3:
                preview += ", ..."
            return f"{{{preview}}}"
        return str(value)

    @staticmethod
    def render_retention_targets(targets: Any) -> bool:
        if not (isinstance(targets, list) and targets and isinstance(targets[0], dict)):
            return False

        print("  Targets:")
        now_ts = time.time()
        for target_item in targets:
            target = cast(dict[str, Any], target_item)
            name = target.get("name", "?")
            files = target.get("files_remaining", target.get("files_scanned", "?"))
            size_bytes = target.get("total_size_bytes", 0)
            size_mb = (size_bytes / (1024 * 1024)) if isinstance(size_bytes, (int, float)) else 0.0
            deleted = target.get("files_deleted", 0)
            run_ts = target.get("run_timestamp")
            if isinstance(run_ts, (int, float)) and run_ts:
                age_minutes = max(0.0, (now_ts - run_ts) / 60)
                age_str = f"{age_minutes:.1f}m ago"
            else:
                age_str = "n/a"
            print(f"    - {name}: {files} files, {size_mb:.2f} MB, removed {deleted} ({age_str})")
        return True

    @staticmethod
    def _should_show_value(value: Any) -> bool:
        """Check if a value should be displayed (non-empty/non-zero)."""
        return value not in (None, "", [], {}, 0, 0.0)

    def _format_stat_line(self, key: str, value: Any) -> str:
        """Format a single stat line for display."""
        label = key.replace("_", " ").title()
        # Summarize large nested structures
        if isinstance(value, dict) and len(str(value)) > 100:
            return f"  {label}: {len(value)} items"
        return f"  {label}: {self.format_cache_stat_value(value)}"

    def render_stat_fields(self, stats: dict[str, Any]) -> bool:
        """Render cache statistics, showing only the most useful metrics."""
        priority_fields = [
            "hit_rate",
            "hits",
            "misses",
            "entries",
            "memory_cache_entries",
            "memory_usage_mb",
            "total_size_mb",
            "size_limit_gb",
            "max_entries",
        ]
        skip_fields = {
            "name",
            "kind",
            "health",
            "targets",
            "entries_utilization",
            "size_compliant",
            "volume",
            "eviction_policy",
            "evictions",
            "module_name",
        }

        shown_keys: set[str] = set()
        shown_any = False

        # Show priority fields first
        for key in priority_fields:
            if key not in stats or key in skip_fields:
                continue
            value = stats[key]
            if self._should_show_value(value):
                print(self._format_stat_line(key, value))
                shown_keys.add(key)
                shown_any = True

        # Show remaining interesting fields
        for key in sorted(stats.keys()):
            if key in skip_fields or key in shown_keys:
                continue
            value = stats[key]
            if self._should_show_value(value):
                print(self._format_stat_line(key, value))
                shown_any = True

        return shown_any

    @staticmethod
    def render_health_stats(health: Any) -> bool:
        if not (isinstance(health, dict) and health):
            return False

        health_dict = cast(dict[str, Any], health)
        score = health_dict.get("overall_score")
        score_str = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
        print(f"  Health Score: {score_str}")
        recommendations = health_dict.get("recommendations")
        if recommendations:
            print(f"  Recommendations: {len(recommendations)}")
        return True

    def print_cache_component(self, component_name: str, stats: dict[str, Any]) -> None:
        icon = self._CACHE_KIND_ICONS.get(stats.get("kind", ""), "🗃️")
        display_name = stats.get("name", component_name).upper()
        kind = stats.get("kind", "unknown")
        print(f"{icon} {display_name} [{kind}]")
        print("-" * 70)

        had_output = False
        had_output |= self.render_retention_targets(stats.get("targets"))
        had_output |= self.render_stat_fields(stats)
        had_output |= self.render_health_stats(stats.get("health"))

        if not had_output:
            print("  No statistics available for this component.")
        print()

    def _show_cache_registry_stats(self) -> bool:
        try:
            from core.cache_registry import get_cache_registry

            registry = get_cache_registry()
            summary = registry.summary()
            component_names = summary.get("registry", {}).get("names", [])
            if not component_names:
                return False

            for component_name in component_names:
                stats = summary.get(component_name, {})
                self.print_cache_component(component_name, stats)

            registry_info = summary.get("registry", {})
            print("REGISTRY OVERVIEW")
            print("-" * 70)
            print(f"  Components: {registry_info.get('components', len(component_names))}")
            print(f"  Registered: {', '.join(component_names)}")
            print()
            return True
        except Exception as exc:  # pragma: no cover - depends on optional modules
            self._logger.error("Failed to display cache registry stats: %s", exc, exc_info=True)
            return False

    def _show_base_cache_stats(self) -> bool:
        try:
            from caching.cache import get_cache_stats

            base_stats = get_cache_stats()
            if base_stats:
                print("📁 DISK CACHE (Base System)")
                print("-" * 70)
                print(f"  Hits: {base_stats.get('hits', 0):,}")
                print(f"  Misses: {base_stats.get('misses', 0):,}")
                print(f"  Hit Rate: {base_stats.get('hit_rate', 0):.1f}%")
                print(f"  Entries: {base_stats.get('entries', 0):,} / {base_stats.get('max_entries', 'N/A')}")
                print(f"  Volume: {base_stats.get('volume', 0):,} bytes")
                print(f"  Cache Dir: {base_stats.get('cache_dir', 'N/A')}")
                print()
                return True
        except Exception as exc:
            self._logger.debug("Could not get base cache stats: %s", exc)
        return False

    def _show_unified_cache_stats(self) -> bool:
        try:
            from caching.cache_manager import get_cache_coordinator

            unified_mgr = get_cache_coordinator()
            comprehensive_stats = unified_mgr.get_comprehensive_stats()
            stats_shown = False

            session_stats = comprehensive_stats.get('session_cache', {})
            if session_stats:
                print("🔐 SESSION CACHE")
                print("-" * 70)
                print(f"  Active Sessions: {session_stats.get('active_sessions', 0)}")
                print(f"  Tracked Sessions: {session_stats.get('tracked_sessions', 0)}")
                print(f"  Component TTL: {session_stats.get('component_ttl', 0)}s")
                print(f"  Session TTL: {session_stats.get('session_ttl', 0)}s")
                print()
                stats_shown = True

            api_stats = comprehensive_stats.get('api_cache', {})
            if api_stats:
                print("🌐 API CACHE")
                print("-" * 70)
                print(f"  Active Sessions: {api_stats.get('active_sessions', 0)}")
                print(f"  Cache Available: {api_stats.get('cache_available', False)}")
                print()
                stats_shown = True

            system_stats = comprehensive_stats.get('system_cache', {})
            if system_stats:
                print("⚙️  SYSTEM CACHE")
                print("-" * 70)
                print(f"  GC Collections: {system_stats.get('gc_collections', 0)}")
                print(f"  Memory Freed: {system_stats.get('memory_freed_mb', 0):.2f} MB")
                print(f"  Peak Memory: {system_stats.get('peak_memory_mb', 0):.2f} MB")
                print(f"  Current Memory: {system_stats.get('current_memory_mb', 0):.2f} MB")
                print()
                stats_shown = True

            return stats_shown
        except Exception as exc:
            self._logger.debug("Could not get unified cache stats: %s", exc)
        return False

    def _show_performance_cache_stats(self) -> bool:
        try:
            from performance.performance_cache import get_cache_stats as get_perf_stats

            perf_stats = get_perf_stats()
            if perf_stats:
                print("📊 PERFORMANCE CACHE (GEDCOM)")
                print("-" * 70)
                print(f"  Memory Entries: {perf_stats.get('memory_entries', 0)}")
                print(f"  Memory Usage: {perf_stats.get('memory_usage_mb', 0):.2f} MB")
                print(f"  Memory Pressure: {perf_stats.get('memory_pressure', 0):.1f}%")
                print(f"  Disk Cache Dir: {perf_stats.get('disk_cache_dir', 'N/A')}")
                print()
                return True
        except Exception as exc:
            self._logger.debug("Could not get performance cache stats: %s", exc)
        return False

    def show_cache_statistics(self) -> None:
        """Display cache statistics across all cache subsystems."""

        try:
            os.system("cls" if os.name == "nt" else "clear")
            print("\n" + "=" * 70)
            print("CACHE STATISTICS")
            print("=" * 70 + "\n")

            stats_collected = self._show_cache_registry_stats()
            if not stats_collected:
                stats_collected = any(
                    [
                        self._show_base_cache_stats(),
                        self._show_unified_cache_stats(),
                        self._show_performance_cache_stats(),
                    ]
                )

            if not stats_collected:
                print("No cache statistics available.")
                print("Caches may not be initialized yet.")

            self._logger.debug("Cache statistics displayed")
            print("=" * 70)

        except Exception as exc:  # pragma: no cover - console I/O
            self._logger.error("Error displaying cache statistics: %s", exc, exc_info=True)
            print("Error displaying cache statistics. Check logs for details.")

        input("\nPress Enter to continue...")


class ConfigMaintenanceMixin:
    """Mixin for configuration and schema operations."""

    _logger: logging.Logger

    def run_config_health_check(self) -> None:
        """Run comprehensive configuration health check and display report."""
        try:
            os.system("cls" if os.name == "nt" else "clear")
            from config.validator import run_health_check

            run_health_check()
            self._logger.debug("Configuration health check completed")
        except ImportError as exc:
            self._logger.error("Configuration validator not available: %s", exc)
            print("Configuration health check module not available.")
        except Exception as exc:  # pragma: no cover - console I/O
            self._logger.error("Error running configuration health check: %s", exc, exc_info=True)
            print("Error running configuration health check. Check logs for details.")

        input("\nPress Enter to continue...")

    def reload_configuration(self) -> None:
        """Hot-reload configuration using ConfigManager."""
        try:
            from config.config_manager import get_config_manager
            from core.feature_flags import bootstrap_feature_flags
            from core.rate_limiter import get_adaptive_rate_limiter

            manager = get_config_manager()
            if manager is None:
                print("\n✗ ConfigManager not available\n")
                return

            manager.reload_config()
            config = manager.get_config()
            flags = bootstrap_feature_flags(config)

            # Refresh rate limiter with updated endpoint profiles
            api_settings = getattr(config, "api", None)
            endpoint_profiles = getattr(api_settings, "endpoint_throttle_profiles", {}) if api_settings else {}
            limiter = get_adaptive_rate_limiter()
            if limiter and endpoint_profiles:
                limiter.configure_endpoint_profiles(endpoint_profiles)
                print(f"   Rate limiter: {len(endpoint_profiles)} endpoint profiles applied")

            print(f"\n✅ Configuration reloaded. Feature flags: {len(flags.get_all_flags())} loaded.\n")
        except ImportError as exc:
            self._logger.error("Configuration reload failed: %s", exc)
            print(f"Error reloading configuration: {exc}")
        except Exception as exc:
            self._logger.error("Error reloading configuration: %s", exc, exc_info=True)
            print(f"Error reloading configuration: {exc}")
        input("Press Enter to continue...")

    def run_config_setup_wizard(self) -> None:
        """Interactive configuration setup wizard (first-run helper)."""
        try:
            from config.config_manager import get_config_manager

            manager = get_config_manager(force_new=True)
            if manager is None:
                print("\n✗ ConfigManager not available\n")
                return

            manager.run_setup_wizard()
            print("\n✅ Setup wizard completed. You may need to restart for changes to take effect.\n")
        except Exception as exc:
            self._logger.error("Error running setup wizard: %s", exc, exc_info=True)
            print(f"Error running setup wizard: {exc}")
        input("Press Enter to continue...")

    def run_schema_migrations_action(self) -> None:
        """Apply pending schema migrations and display status."""

        print("\n" + "=" * 70)
        print("SCHEMA MIGRATIONS")
        print("=" * 70)

        db_manager: Optional[Any] = None
        try:
            from core import schema_migrator
            from core.database_manager import DatabaseManager

            db_manager = DatabaseManager()
            if db_manager is None or not db_manager.ensure_ready():
                print("Unable to initialize database engine. See logs for details.")
                return

            engine = getattr(db_manager, "engine", None)
            if engine is None:
                print("Unable to access database engine instance.")
                return

            registered_migrations = schema_migrator.get_registered_migrations()
            print(f"Registered migrations: {len(registered_migrations)}")

            applied_versions = schema_migrator.apply_pending_migrations(engine)
            installed_versions = schema_migrator.get_applied_versions(engine)

            if applied_versions:
                print(f"\nApplied migrations: {', '.join(applied_versions)}")
            else:
                print("\nNo pending migrations; schema already current.")

            if installed_versions:
                print(f"Installed versions ({len(installed_versions)}): {', '.join(installed_versions)}")
            else:
                print("Installed versions: none recorded.")

            pending_versions = [
                migration.version for migration in registered_migrations if migration.version not in installed_versions
            ]
            if pending_versions:
                print(f"Pending migrations ({len(pending_versions)}): {', '.join(pending_versions)}")
            else:
                print("All registered migrations have been applied.")
        except Exception as exc:  # pragma: no cover - depends on db setup
            self._logger.error("Failed to run schema migrations: %s", exc, exc_info=True)
            print(f"Error applying migrations: {exc}")
        finally:
            if db_manager is not None:
                try:
                    db_manager.close_connections(dispose_engine=True)
                except Exception:
                    self._logger.debug("Failed to close temporary database manager", exc_info=True)
            input("\nPress Enter to continue...")


class SystemMixin:
    """Mixin for system-level operations."""

    @staticmethod
    def clear_screen() -> None:
        os.system("cls" if os.name == "nt" else "clear")

    def exit_application(self) -> bool:
        # Close browser immediately before exiting
        try:
            from core.session_utils import close_cached_session

            close_cached_session(keep_db=False)
        except Exception:
            pass  # Silently ignore cleanup errors

        self.clear_screen()
        print("Exiting.")
        return False


class MainCLIHelpers(
    LogMaintenanceMixin,
    TestRunnerMixin,
    AnalyticsMixin,
    ReviewQueueMixin,
    CacheStatsMixin,
    ConfigMaintenanceMixin,
    SystemMixin,
):
    """Container for log/analytics helper actions used by the main menu."""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        grafana_checker: Optional[GrafanaCheckerProtocol] = None,
    ) -> None:
        self._logger = logger
        self._grafana_checker = grafana_checker


# ------------------------------------------------------------------
# Test helpers (internal use only)
# ------------------------------------------------------------------


def _create_helper_for_tests(
    *,
    log_path: Optional[Path] = None,
    include_stream: bool = False,
) -> tuple[MainCLIHelpers, logging.Logger]:
    logger = logging.getLogger(f"cli-maint-tests-{time.time_ns()}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    if log_path is not None:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    if include_stream:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

    return MainCLIHelpers(logger=logger), logger


def _teardown_test_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _capture_stdout(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = func(*args, **kwargs)
    return result, buffer.getvalue()


def _test_format_cache_stat_value() -> bool:
    assert MainCLIHelpers.format_cache_stat_value(1_234_567) == "1,234,567"
    assert MainCLIHelpers.format_cache_stat_value(3.14159) == "3.14"
    assert MainCLIHelpers.format_cache_stat_value([1, 2, 3]) == "3 items"
    rich_dict = {"a": 1, "b": 2, "c": 3, "d": 4}
    assert MainCLIHelpers.format_cache_stat_value(rich_dict) == "{a=1, b=2, c=3, ...}"
    assert MainCLIHelpers.format_cache_stat_value("ready") == "ready"
    return True


def _test_render_retention_targets() -> bool:
    targets = [
        {
            "name": "Disk Cache",
            "files_remaining": 10,
            "total_size_bytes": 5 * 1024 * 1024,
            "files_deleted": 2,
            "run_timestamp": time.time() - 120,
        }
    ]
    result, output = _capture_stdout(MainCLIHelpers.render_retention_targets, targets)
    assert result is True
    assert "Disk Cache" in output
    assert "10 files" in output
    assert "MB" in output

    result, _ = _capture_stdout(MainCLIHelpers.render_retention_targets, None)
    assert result is False
    return True


def _test_render_stat_and_health_fields() -> bool:
    helper, logger = _create_helper_for_tests()
    try:
        stats = {"hits": 5, "misses": 2, "name": "disk", "kind": "disk"}
        result, output = _capture_stdout(helper.render_stat_fields, stats)
        assert result is True
        assert "Hits: 5" in output
        assert "Misses: 2" in output
        result, _ = _capture_stdout(helper.render_stat_fields, {"name": "only"})
        assert result is False
    finally:
        _teardown_test_logger(logger)

    result, output = _capture_stdout(
        MainCLIHelpers.render_health_stats,
        {"overall_score": 87.5, "recommendations": ["optimize", "trim"]},
    )
    assert result is True
    assert "87.5" in output
    assert "Recommendations: 2" in output

    result, _ = _capture_stdout(MainCLIHelpers.render_health_stats, None)
    assert result is False
    return True


def _test_print_cache_component_output() -> bool:
    helper, logger = _create_helper_for_tests()
    try:
        _, output = _capture_stdout(helper.print_cache_component, "component", {})
        assert "🗃️ COMPONENT" in output
        assert "No statistics available" in output
    finally:
        _teardown_test_logger(logger)
    return True


def _test_clear_log_file_behavior() -> bool:
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "cli.log"
        log_path.write_text("existing line\n", encoding="utf-8")
        helper, logger = _create_helper_for_tests(log_path=log_path)
        try:
            cleared, cleared_path = helper.clear_log_file()
            assert cleared is True
            assert cleared_path == str(log_path)
            assert not log_path.read_text(encoding="utf-8")
        finally:
            _teardown_test_logger(logger)
    return True


def _test_toggle_log_level_switches_levels() -> bool:
    helper, logger = _create_helper_for_tests(include_stream=True)
    try:
        stream_handler = next(
            handler
            for handler in logger.handlers
            if isinstance(handler, logging.StreamHandler)
            and cast(Optional[TextIO], getattr(handler, "stream", None)) is sys.stderr
        )
        stream_handler.setLevel(logging.INFO)

        with (
            mock.patch.object(sys.modules[__name__], "setup_logging") as patched_setup,
            mock.patch("os.system") as mock_system,
        ):
            helper.toggle_log_level()
            patched_setup.assert_called_once()
            kwargs = patched_setup.call_args.kwargs
            assert kwargs["log_level"] == "DEBUG"
            assert kwargs["allow_env_override"] is False
            mock_system.assert_called()

        stream_handler.setLevel(logging.DEBUG)
        with (
            mock.patch.object(sys.modules[__name__], "setup_logging") as patched_setup,
            mock.patch("os.system") as mock_system,
        ):
            helper.toggle_log_level()
            patched_setup.assert_called_once()
            kwargs = patched_setup.call_args.kwargs
            assert kwargs["log_level"] == "INFO"
            assert kwargs["allow_env_override"] is False
            mock_system.assert_called()
    finally:
        _teardown_test_logger(logger)
    return True


def _test_review_queue_renderers() -> bool:
    helper, logger = _create_helper_for_tests()
    try:
        drafts = [
            SimpleNamespace(
                priority=SimpleNamespace(value="high"),
                draft_id=101,
                person_name="Alice",
                person_id=1,
                ai_confidence=92,
                created_at=datetime(2024, 1, 1, 12, 0),
                content="Hello from Alice" * 3,
            ),
            SimpleNamespace(
                priority=SimpleNamespace(value="low"),
                draft_id=202,
                person_name="Bob",
                person_id=2,
                ai_confidence=55,
                created_at=datetime(2024, 1, 2, 9, 30),
                content="Bob follow up" * 3,
            ),
        ]

        _, draft_output = _capture_stdout(helper._render_pending_drafts, drafts)
        assert "Pending Drafts (2 shown)" in draft_output
        assert "ID: 101" in draft_output and "Alice" in draft_output
        assert "Confidence: 92%" in draft_output
        assert "LOW" in draft_output and "ID: 202" in draft_output

        _, no_draft_output = _capture_stdout(helper._render_pending_drafts, [])
        assert "Pending Drafts: none" in no_draft_output

        facts = [
            {
                "id": 11,
                "person_id": 3,
                "person_name": "Carol",
                "fact_type": "Birth",
                "new_value": "Born 1900 in Boston",
                "confidence": 0.87,
                "created_at": datetime(2024, 1, 3, 8, 45),
            }
        ]

        _, fact_output = _capture_stdout(helper._render_pending_facts, facts)
        assert "Pending Suggested Facts (1 shown)" in fact_output
        assert "ID: 11" in fact_output and "Carol" in fact_output
        assert "Born 1900 in Boston" in fact_output

        _, no_fact_output = _capture_stdout(helper._render_pending_facts, [])
        assert "Pending Suggested Facts: none" in no_fact_output

        def _make_scalar_query(value: int) -> mock.Mock:
            query = mock.Mock()
            query.filter.return_value = query
            query.scalar.return_value = value
            return query

        db_session = mock.Mock()
        db_session.query.side_effect = [
            _make_scalar_query(2),
            _make_scalar_query(1),
            _make_scalar_query(6),
            _make_scalar_query(3),
            _make_scalar_query(4),
            _make_scalar_query(8),
            _make_scalar_query(9),
        ]

        draft_stats = SimpleNamespace(
            pending_count=4,
            auto_approved_count=1,
            approved_today=2,
            rejected_today=1,
            expired_count=0,
        )

        _, metrics_output = _capture_stdout(helper._render_review_metrics, db_session, draft_stats)
        assert "Critical alerts / Human review: 2" in metrics_output
        assert "Opt-outs detected: 1" in metrics_output
        assert "Suggested facts: pending 6, approved 3, rejected 4" in metrics_output
        assert "Drafts: pending 4" in metrics_output
        assert "Draft approvals (all time): 8" in metrics_output
        assert "Outbound sends (all time): 9" in metrics_output
    finally:
        _teardown_test_logger(logger)
    return True


# ------------------------------------------------------------------
# Embedded TestSuite
# ------------------------------------------------------------------


def module_tests() -> bool:
    suite = TestSuite("cli.maintenance", "cli/maintenance.py")
    suite.start_suite()

    suite.run_test(
        "Format cache stat values",
        _test_format_cache_stat_value,
        "Ensures value formatter handles common types.",
    )

    suite.run_test(
        "Render retention targets",
        _test_render_retention_targets,
        "Ensures retention targets render summaries and fallbacks.",
    )

    suite.run_test(
        "Render stat and health fields",
        _test_render_stat_and_health_fields,
        "Ensures stat and health helpers print expected lines.",
    )

    suite.run_test(
        "Print cache component fallback",
        _test_print_cache_component_output,
        "Ensures cache component output handles empty stats.",
    )

    suite.run_test(
        "Clear log file handler",
        _test_clear_log_file_behavior,
        "Ensures bound log file handler can be cleared.",
    )

    suite.run_test(
        "Toggle console log level",
        _test_toggle_log_level_switches_levels,
        "Ensures log level toggle switches between INFO and DEBUG.",
    )

    suite.run_test(
        "Render review queue summaries",
        _test_review_queue_renderers,
        "Ensures pending drafts, facts, and metrics render correctly.",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)


__all__ = ["GrafanaCheckerProtocol", "MainCLIHelpers"]
