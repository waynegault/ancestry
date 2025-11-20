"""Helper utilities backing the interactive CLI actions in main.py.

The helpers live outside main.py so the entry point can focus on session
orchestration while these routines handle log maintenance, analytics views,
metrics dashboards, and other ancillary utilities.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Mapping
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from logging import StreamHandler
from pathlib import Path
from typing import Any, ClassVar, Optional, Protocol, TextIO, cast
from urllib import request as urllib_request

from logging_config import setup_logging


class GrafanaCheckerProtocol(Protocol):
    """Protocol describing the optional grafana checker helpers."""

    def ensure_dashboards_imported(self) -> None: ...

    def check_grafana_status(self) -> Mapping[str, Any]: ...

    def ensure_grafana_ready(self, *, auto_setup: bool = False, silent: bool = True) -> None: ...


class MainCLIHelpers:
    """Container for log/analytics helper actions used by the main menu."""

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

    def __init__(
        self,
        *,
        logger: logging.Logger,
        grafana_checker: Optional[GrafanaCheckerProtocol] = None,
    ) -> None:
        self._logger = logger
        self._grafana_checker = grafana_checker

    # ------------------------------------------------------------------
    # Log maintenance helpers
    # ------------------------------------------------------------------

    def clear_log_file(self) -> tuple[bool, Optional[str]]:
        """Clear the active log file by flushing/closing the handler."""

        cleared = False
        log_file_handler: Optional[logging.FileHandler] = None
        log_file_path: Optional[str] = None
        try:
            for handler in self._logger.handlers:
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
            self._logger.warning(
                "Permission denied clearing log '%s': %s", log_file_path, permission_error
            )
        except OSError as io_error:
            self._logger.warning("IOError clearing log '%s': %s", log_file_path, io_error)
        except Exception as error:  # pragma: no cover - defensive logging only
            self._logger.warning(
                "Error clearing log '%s': %s", log_file_path, error, exc_info=True
            )
        return cleared, log_file_path

    # ------------------------------------------------------------------
    # Test runners
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Visualization and analytics helpers
    # ------------------------------------------------------------------

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
                    server = ThreadingHTTPServer(
                        ("127.0.0.1", candidate_port), GraphRequestHandler
                    )
                    preferred_port = candidate_port
                    break
                except OSError:
                    continue

            if server is None:
                print("❌ Unable to start the graph visualization server (no open ports).")
                self._logger.error(
                    "Graph visualization server failed to start: no open ports available"
                )
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
                self._logger.warning(
                    "Unable to open browser automatically: %s", browser_err
                )
                print("⚠️  Please open the URL manually in your browser.")

            input("\nPress Enter to stop the visualization server and return to the menu...")

        except Exception as graph_error:  # pragma: no cover - defensive
            self._logger.error(
                "Error running graph visualization: %s", graph_error, exc_info=True
            )
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
            from conversation_analytics import print_analytics_dashboard
            from core.session_manager import SessionManager

            print("\n" + "=" * 80)
            print("LOADING ANALYTICS DASHBOARD")
            print("=" * 80)

            sm = SessionManager()
            db_session = sm.get_db_conn()

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

    # ------------------------------------------------------------------
    # Cache statistics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_cache_stat_value(value: Any) -> str:
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
    def _render_retention_targets(targets: Any) -> bool:
        if not (isinstance(targets, list) and targets and isinstance(targets[0], dict)):
            return False

        print("  Targets:")
        now_ts = time.time()
        for target in targets:
            name = target.get("name", "?")
            files = target.get("files_remaining", target.get("files_scanned", "?"))
            size_bytes = target.get("total_size_bytes", 0)
            size_mb = (
                (size_bytes / (1024 * 1024)) if isinstance(size_bytes, (int, float)) else 0.0
            )
            deleted = target.get("files_deleted", 0)
            run_ts = target.get("run_timestamp")
            if isinstance(run_ts, (int, float)) and run_ts:
                age_minutes = max(0.0, (now_ts - run_ts) / 60)
                age_str = f"{age_minutes:.1f}m ago"
            else:
                age_str = "n/a"
            print(
                f"    - {name}: {files} files, {size_mb:.2f} MB, removed {deleted} ({age_str})"
            )
        return True

    def _render_stat_fields(self, stats: dict[str, Any]) -> bool:
        shown_any = False
        for key in sorted(stats.keys()):
            if key in {"name", "kind", "health", "targets"}:
                continue
            value = stats[key]
            if value in (None, "", [], {}):
                continue
            print(f"  {key.replace('_', ' ').title()}: {self._format_cache_stat_value(value)}")
            shown_any = True
        return shown_any

    @staticmethod
    def _render_health_stats(health: Any) -> bool:
        if not (isinstance(health, dict) and health):
            return False
        score = health.get("overall_score")
        score_str = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
        print(f"  Health Score: {score_str}")
        recommendations = health.get("recommendations")
        if recommendations:
            print(f"  Recommendations: {len(recommendations)}")
        return True

    def _print_cache_component(self, component_name: str, stats: dict[str, Any]) -> None:
        icon = self._CACHE_KIND_ICONS.get(stats.get("kind", ""), "🗃️")
        display_name = stats.get("name", component_name).upper()
        kind = stats.get("kind", "unknown")
        print(f"{icon} {display_name} [{kind}]")
        print("-" * 70)

        had_output = False
        had_output |= self._render_retention_targets(stats.get("targets"))
        had_output |= self._render_stat_fields(stats)
        had_output |= self._render_health_stats(stats.get("health"))

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
                self._print_cache_component(component_name, stats)

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
            from cache import get_cache_stats

            base_stats = get_cache_stats()
            if base_stats:
                print("📁 DISK CACHE (Base System)")
                print("-" * 70)
                print(f"  Hits: {base_stats.get('hits', 0):,}")
                print(f"  Misses: {base_stats.get('misses', 0):,}")
                print(f"  Hit Rate: {base_stats.get('hit_rate', 0):.1f}%")
                print(
                    f"  Entries: {base_stats.get('entries', 0):,} / {base_stats.get('max_entries', 'N/A')}"
                )
                print(f"  Volume: {base_stats.get('volume', 0):,} bytes")
                print(f"  Cache Dir: {base_stats.get('cache_dir', 'N/A')}")
                print()
                return True
        except Exception as exc:
            self._logger.debug("Could not get base cache stats: %s", exc)
        return False

    def _show_unified_cache_stats(self) -> bool:
        try:
            from cache_manager import get_unified_cache_manager

            unified_mgr = get_unified_cache_manager()
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
            from performance_cache import get_cache_stats as get_perf_stats

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

    # ------------------------------------------------------------------
    # Database/schema utilities
    # ------------------------------------------------------------------

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
                print(
                    f"Installed versions ({len(installed_versions)}): "
                    f"{', '.join(installed_versions)}"
                )
            else:
                print("Installed versions: none recorded.")

            pending_versions = [
                migration.version
                for migration in registered_migrations
                if migration.version not in installed_versions
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
                    self._logger.debug(
                        "Failed to close temporary database manager", exc_info=True
                    )
            input("\nPress Enter to continue...")

    # ------------------------------------------------------------------
    # Logging/metrics toggles
    # ------------------------------------------------------------------

    def toggle_log_level(self) -> None:
        """Toggle console log level between DEBUG and INFO."""

        os.system("cls" if os.name == "nt" else "clear")
        if self._logger and self._logger.handlers:
            console_handler: Optional[StreamHandler[TextIO]] = None
            for handler in self._logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler_typed = cast(logging.StreamHandler[TextIO], handler)
                    if handler_typed.stream == sys.stderr:
                        console_handler = handler_typed
                        break
            if console_handler:
                current_level = console_handler.level
                new_level = logging.DEBUG if current_level > logging.DEBUG else logging.INFO
                new_level_name = logging.getLevelName(new_level)
                setup_logging(log_level=new_level_name, allow_env_override=False)
                self._logger.info("Console log level toggled to: %s", new_level_name)
            else:
                self._logger.warning("Could not find console handler to toggle level.")
        else:
            print("WARNING: Logger not ready or has no handlers.", file=sys.stderr)

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
                    self._logger.debug(
                        "Dashboard auto-import check: %s", import_err
                    )

            system_perf_url = f"{grafana_base}/d/ancestry-performance"
            genealogy_url = f"{grafana_base}/d/ancestry-genealogy"
            code_quality_url = f"{grafana_base}/d/ancestry-code-quality"

            print("🌐 Opening dashboards:")
            print(f"   1. System Performance & Health: {system_perf_url}")
            print(f"   2. Genealogy Research Insights: {genealogy_url}")
            print(f"   3. Code Quality & Architecture: {code_quality_url}")
            print("\n💡 If dashboards show 'Not found', run: setup-grafana")
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

    # ------------------------------------------------------------------
    # Miscellaneous helpers
    # ------------------------------------------------------------------

    @staticmethod
    def clear_screen() -> None:
        os.system("cls" if os.name == "nt" else "clear")

    def exit_application(self) -> bool:
        self.clear_screen()
        print("Exiting.")
        return False


__all__ = ["GrafanaCheckerProtocol", "MainCLIHelpers"]
