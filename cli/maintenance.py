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
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from logging import StreamHandler
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional, Protocol, TextIO, cast
from unittest import mock
from urllib import request as urllib_request

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from core.logging_config import setup_logging
from testing.test_framework import TestSuite, create_standard_test_runner


class GrafanaCheckerProtocol(Protocol):
    """Protocol describing the optional grafana checker helpers."""

    def ensure_dashboards_imported(self) -> None: ...

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
            root_dir = Path(__file__).resolve().parent
            preferred_port = 8765

            local_logger = self._logger

            class GraphRequestHandler(SimpleHTTPRequestHandler):
                directory = str(root_dir)

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
            from observability.metrics_registry import is_metrics_enabled
        except Exception as exc:  # pragma: no cover - optional module missing
            self._logger.error("Unable to import metrics registry: %s", exc)
            print("\n⚠️  Metrics registry unavailable")
            return

        try:
            print("\n" + "=" * 70)
            print("📊 GRAFANA METRICS DASHBOARD")
            print("=" * 70)

            if not is_metrics_enabled():
                print("\n⚠️  Metrics collection is DISABLED")
                print("\nTo enable metrics:")
                print("  1. Add to .env: PROMETHEUS_METRICS_ENABLED=true")
                print("  2. Optionally configure: PROMETHEUS_METRICS_PORT=9000")
                print("  3. Restart the application")
                print("\n" + "=" * 70 + "\n")
                return

            grafana_base = "http://localhost:3000"
            try:
                urllib_request.urlopen(grafana_base, timeout=1)
                grafana_running = True
            except Exception:
                grafana_running = False

            if not grafana_running:
                print("\n⚠️  Grafana is NOT running on http://localhost:3000")
                print("\n💡 Setup Instructions:")
                print("   1. Install Grafana: https://grafana.com/grafana/download")
                print("   2. Start Grafana service")
                print("   3. Login at http://localhost:3000 (default: admin/admin)")
                print("   4. Add Prometheus data source → http://localhost:9000")
                print("   5. Import dashboard: docs/grafana/ancestry_overview.json")
                print("\n📊 For now, opening raw metrics at http://localhost:9000/metrics")
                print("\n" + "=" * 70 + "\n")
                webbrowser.open("http://localhost:9000/metrics")
                return

            print("\n✅ Grafana is running!")
            print("🔍 Checking dashboards...")

            if self._grafana_checker:
                try:
                    self._grafana_checker.ensure_dashboards_imported()
                except Exception as import_err:  # pragma: no cover - optional dependency
                    self._logger.debug("Dashboard auto-import check: %s", import_err)

            system_perf_url = f"{grafana_base}/d/ancestry-performance"
            genealogy_url = f"{grafana_base}/d/ancestry-genealogy"
            code_quality_url = f"{grafana_base}/d/ancestry-code-quality"

            print("🌐 Opening dashboards:")
            print(f"   1. System Performance & Health: {system_perf_url}")
            print(f"   2. Genealogy Research Insights: {genealogy_url}")
            print(f"   3. Code Quality & Architecture: {code_quality_url}")
            print("\n💡 If dashboards show 'Not found', run: l")
            print("\n" + "=" * 70 + "\n")

            webbrowser.open(system_perf_url)
            time.sleep(0.5)
            webbrowser.open(genealogy_url)
            time.sleep(0.5)
            webbrowser.open(code_quality_url)

        except Exception as exc:  # pragma: no cover - browser/IO issues
            self._logger.error("Error opening Grafana: %s", exc, exc_info=True)
            print(f"\n⚠️  Error: {exc}")
            print("\n" + "=" * 70 + "\n")

    def run_grafana_setup(self) -> None:
        """Run grafana checker setup flow if helper module is present."""

        if self._grafana_checker:
            status = self._grafana_checker.check_grafana_status()
            if status["ready"]:
                print("\n✅ Grafana is already fully configured and running!")
                print("   Dashboard URL: http://localhost:3000")
                print("   Default credentials: admin / ancestry")
                print("\n📊 Checking dashboards...")
                self._grafana_checker.ensure_dashboards_imported()
                print("\n✅ Dashboard check complete!")
                print("\n📊 Available Dashboards:")
                print("   • Overview:    http://localhost:3000/d/ancestry-overview")
                print("   • Performance: http://localhost:3000/d/ancestry-performance")
                print("   • Genealogy:   http://localhost:3000/d/ancestry-genealogy")
                print("   • Code Quality: http://localhost:3000/d/ancestry-code-quality")
                print("\n💡 If dashboards are empty, configure data sources:")
                print("   Run: .\\docs\\grafana\\configure_datasources.ps1\n")
            else:
                self._grafana_checker.ensure_grafana_ready(auto_setup=False, silent=False)
        else:
            print("\n⚠️  Grafana checker module not available")
            print("Ensure grafana_checker.py is in the project root directory\n")


class ReviewQueueMixin:
    """Mixin for review queue operations."""

    _logger: logging.Logger

    def show_review_queue(self) -> None:
        """Display pending drafts for human review."""
        try:
            from core.approval_queue import ApprovalQueueService
            from core.session_manager import SessionManager
            from core.database import SuggestedFact

            print("\n" + "=" * 70)
            print("📋 REVIEW QUEUE - Pending AI-Generated Drafts")
            print("=" * 70)

            sm = SessionManager()
            db_session = sm.db_manager.get_session()

            if not db_session:
                print("✗ Failed to get database session")
                return

            service = ApprovalQueueService(db_session)

            draft_stats = service.get_queue_stats()
            pending_drafts = service.get_pending_queue(limit=10)
            pending_facts = self._get_pending_suggested_facts(db_session, limit=10)

            self._render_review_metrics(db_session, draft_stats, pending_facts)
            self._render_pending_drafts(pending_drafts)
            self._render_pending_facts(pending_facts, SuggestedFact)
            self._render_contextual_draft_log_preview()

            if not pending_drafts and not pending_facts:
                print("\n✅ Nothing pending in review queues.")
                return

            print("\n" + "-" * 70)
            print(
                "Commands: approve <draft_id> | reject <draft_id> <reason> | "
                "fact approve <id> | fact reject <id> <reason> | refresh | back"
            )

            while True:
                command = input("review> ").strip()
                if not command:
                    continue
                if command.lower() in {"back", "exit", "q"}:
                    break
                if command.lower() == "refresh":
                    self.show_review_queue()
                    return

                tokens = command.split()
                if tokens[0] == "approve" and len(tokens) >= 2:
                    self.approve_draft(int(tokens[1]))
                    continue
                if tokens[0] == "reject" and len(tokens) >= 2:
                    reason = " ".join(tokens[2:]) if len(tokens) > 2 else ""
                    self.reject_draft(int(tokens[1]), reason)
                    continue
                if tokens[0] == "fact" and len(tokens) >= 3:
                    action = tokens[1]
                    fact_id = int(tokens[2])
                    if action == "approve":
                        self.approve_suggested_fact(fact_id)
                    elif action == "reject":
                        reason = " ".join(tokens[3:]) if len(tokens) > 3 else ""
                        self.reject_suggested_fact(fact_id, reason)
                    else:
                        print("Unknown fact command. Use: fact approve <id> | fact reject <id> <reason>")
                    continue

                print("Unrecognized command. Type 'back' to exit or 'refresh' to reload queue.")

        except Exception as exc:
            self._logger.error("Error showing review queue: %s", exc, exc_info=True)
            print(f"Error: {exc}")

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

    def _get_pending_suggested_facts(self, db_session: Any, limit: int = 10) -> list[dict[str, Any]]:
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

    def _render_pending_drafts(self, pending_drafts: list[Any]) -> None:
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

    def _render_pending_facts(self, pending_facts: list[dict[str, Any]], suggested_fact_cls: Any) -> None:
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

    def _render_review_metrics(self, db_session: Any, draft_stats: Any, pending_facts: list[dict[str, Any]]) -> None:
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

    def render_stat_fields(self, stats: dict[str, Any]) -> bool:
        shown_any = False
        for key in sorted(stats.keys()):
            if key in {"name", "kind", "health", "targets"}:
                continue
            value = stats[key]
            if value in (None, "", [], {}):
                continue
            print(f"  {key.replace('_', ' ').title()}: {self.format_cache_stat_value(value)}")
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

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)


__all__ = ["GrafanaCheckerProtocol", "MainCLIHelpers"]
