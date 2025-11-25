from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, cast
from unittest import mock

from requests.exceptions import ConnectionError
from sqlalchemy.orm import Session as SqlAlchemySession

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from actions.gather.checkpoint import finalize_checkpoint_after_run, load_checkpoint, persist_checkpoint
from actions.gather.metrics import (
    PageProcessingMetrics,
    accumulate_page_metrics,
    collect_total_processed,
    compose_progress_snapshot,
    log_page_completion_summary,
    log_page_start,
    log_timing_breakdown,
)
from actions.gather.persistence import process_batch_lookups as gather_process_batch_lookups
from config import config_schema
from core.error_handling import (
    AuthenticationExpiredError,
    BrowserSessionError,
    MaxApiFailuresExceededError,
    circuit_breaker,
    error_context,
    selenium_retry,
    timeout_protection,
    with_enhanced_recovery,
)
from core.logging_utils import log_action_banner
from core.session_manager import SessionManager
from standard_imports import setup_module
from test_framework import TestSuite, create_standard_test_runner
from utils import log_final_summary, log_starting_position

logger = setup_module(globals(), __name__)

GatherState = dict[str, Any]


@dataclass(frozen=True)
class GatherOrchestratorHooks:
    """Runtime dependencies retained from the legacy module."""

    matches_per_page: int
    relationship_prob_max_per_page: int
    db_error_page_threshold: int
    navigate_and_get_initial_page_data: Callable[
        [SessionManager, int], tuple[Optional[list[dict[str, Any]]], Optional[int], bool]
    ]
    determine_page_processing_range: Callable[[int, int], tuple[int, int]]
    do_batch: Callable[[SessionManager, list[dict[str, Any]], int], tuple[int, int, int, int, PageProcessingMetrics]]
    get_matches: Callable[
        [SessionManager, SqlAlchemySession, int],
        Optional[tuple[list[dict[str, Any]], int]],
    ]
    adjust_delay: Callable[[SessionManager, int], None]
    action_state_cls: Any
    calculate_failure_threshold: Callable[[int], int]


def _initialize_gather_state() -> dict[str, Any]:
    """Initializes counters and state variables for the gathering process."""

    return {
        "total_new": 0,
        "total_updated": 0,
        "total_skipped": 0,
        "total_errors": 0,
        "total_pages_processed": 0,
        "db_connection_errors": 0,
        "final_success": True,
        "matches_on_current_page": [],
        "total_pages_from_api": None,
        "aggregate_metrics": PageProcessingMetrics(),
        "pages_with_metrics": 0,
        "pages_target": 0,
        "total_pages_in_run": 0,
        "last_page_to_process": 0,
        "run_started_at": time.time(),
        "resume_from_checkpoint": False,
        "requested_start_page": None,
        "effective_start_page": 1,
        "last_checkpoint_written_at": None,
        "checkpoint_metadata": None,
    }


def _validate_start_page(start_arg: Any) -> int:
    """Validates and returns the starting page number."""

    try:
        start_page = int(start_arg)
    except (TypeError, ValueError):
        logger.warning("Invalid start page value '%s'. Using default page 1.", start_arg)
        return 1

    if start_page <= 0:
        logger.warning("Invalid start page '%s'. Using default page 1.", start_arg)
        return 1

    return start_page


def _determine_start_page(start_arg: Optional[int]) -> tuple[int, bool, Optional[dict[str, Any]]]:
    """Resolve the effective start page, optionally resuming from checkpoint."""

    if start_arg is not None:
        return _validate_start_page(start_arg), False, None

    checkpoint_data = load_checkpoint()
    if not checkpoint_data:
        return 1, False, None

    resume_page_raw = checkpoint_data.get("next_page")
    last_page_raw = checkpoint_data.get("last_page", resume_page_raw)

    def _coerce(value: Any, *, default: int) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    resume_page = _coerce(resume_page_raw, default=1)
    last_page = _coerce(last_page_raw, default=resume_page)

    if resume_page > last_page:
        logger.debug(
            "Checkpoint requested resume page %s beyond last page %s; starting fresh.",
            resume_page,
            last_page,
        )
        return 1, False, None

    logger.info("Resuming Action 6 from checkpoint page %d (planned last page %d).", resume_page, last_page)
    return resume_page, True, checkpoint_data


@dataclass
class GatherOrchestrator:
    """Coordinates Action 6 using modularized helpers."""

    session_manager: SessionManager
    hooks: GatherOrchestratorHooks

    def coord(self, start: Optional[int] = None) -> bool:
        action_start_time = time.time()

        self._validate_session_state()
        coord_state = self._build_coord_state(start, action_start_time)
        state = cast(dict[str, object], coord_state[0])
        start_page = coord_state[1]
        self._log_action_start(start_page)

        if state.get("resume_from_checkpoint"):
            checkpoint_meta = state.get("checkpoint_metadata")
            if isinstance(checkpoint_meta, dict):
                checkpoint_data: Optional[dict[str, Any]] = cast(dict[str, Any], checkpoint_meta)
            else:
                checkpoint_data = None
            planned_total = checkpoint_data.get("total_pages_in_run") if checkpoint_data else None
            logger.info(
                "Checkpoint resume active: starting at page %d (planned total pages: %s)",
                start_page,
                planned_total if planned_total is not None else "unknown",
            )

        keyboard_interrupt: Optional[KeyboardInterrupt] = None

        try:
            state["final_success"] = self._execute_coord_run(state, start_page)
        except KeyboardInterrupt as interruption:
            logger.warning("Keyboard interrupt detected. Stopping match gathering.")
            state["final_success"] = False
            keyboard_interrupt = interruption
        except Exception as exc:  # pragma: no cover - defensive logging
            state["final_success"] = False
            self._handle_coord_failure(exc)
        finally:
            finalize_checkpoint_after_run(state)
            self._log_final_results(state, action_start_time)

            if keyboard_interrupt is not None:
                logger.info("Re-raising KeyboardInterrupt after cleanup.")
                raise keyboard_interrupt

        return bool(state.get("final_success", False))

    @staticmethod
    def _build_coord_state(
        start: Optional[int],
        action_start_time: float,
    ) -> tuple[dict[str, Any], int]:
        state = _initialize_gather_state()
        state["run_started_at"] = action_start_time
        state["requested_start_page"] = start
        start_page, resumed_from_checkpoint, checkpoint_data = _determine_start_page(start)
        state["effective_start_page"] = start_page
        state["resume_from_checkpoint"] = resumed_from_checkpoint
        state["checkpoint_metadata"] = checkpoint_data
        return state, start_page

    def _execute_coord_run(self, state: GatherState, start_page: int) -> bool:
        page_range = self._prepare_page_range(start_page, state)
        if page_range is None:
            return True

        last_page_to_process, total_pages_in_run = page_range
        initial_matches_for_loop = state["matches_on_current_page"]
        loop_success = self._main_page_processing_loop(
            start_page,
            last_page_to_process,
            total_pages_in_run,
            initial_matches_for_loop,
            state,
        )
        return bool(state.get("final_success", True)) and loop_success

    def _log_action_start(self, start_page: int) -> None:
        app_mode = getattr(config_schema, "app_mode", "production")
        dry_run_enabled = app_mode.lower() == "dry_run"
        raw_max_pages = getattr(config_schema.api, "max_pages", 0)
        requested_max_pages = raw_max_pages if raw_max_pages else "unlimited"
        rel_prob_limit = (
            self.hooks.relationship_prob_max_per_page if self.hooks.relationship_prob_max_per_page > 0 else "unlimited"
        )

        log_action_banner(
            action_name="Gather DNA Matches",
            action_number=6,
            stage="start",
            logger_instance=logger,
            details={
                "start_page": start_page,
                "requested_pages": requested_max_pages,
                "matches_per_page": self.hooks.matches_per_page,
                "mode": app_mode,
                "dry_run": "yes" if dry_run_enabled else "no",
                "rel_prob_limit": rel_prob_limit,
            },
        )
        logger.debug(
            "Action 6 start | start_page=%s | requested_pages=%s | matches_per_page=%s | mode=%s | dry_run=%s | rel_prob_limit=%s",
            start_page,
            requested_max_pages,
            self.hooks.matches_per_page,
            app_mode,
            "yes" if dry_run_enabled else "no",
            rel_prob_limit,
        )
        logger.debug("--- Starting DNA Match Gathering (Action 6) from page %s ---", start_page)

    def _handle_initial_fetch(
        self,
        start_page: int,
        state: GatherState,
    ) -> tuple[int, int, int]:
        initial_matches, total_pages_api, initial_fetch_ok = self.hooks.navigate_and_get_initial_page_data(
            self.session_manager,
            start_page,
        )

        if not initial_fetch_ok or total_pages_api is None:
            state["final_success"] = False
            raise RuntimeError("Initial fetch failed")

        state["total_pages_from_api"] = total_pages_api
        state["matches_on_current_page"] = initial_matches if initial_matches is not None else []
        logger.info("Total pages found: %s", total_pages_api)

        last_page_to_process, total_pages_in_run = self.hooks.determine_page_processing_range(
            total_pages_api,
            start_page,
        )

        if total_pages_in_run <= 0:
            logger.info("No pages to process (Start: %s, End: %s).", start_page, last_page_to_process)
            raise RuntimeError("No pages to process")

        total_matches_estimate = total_pages_in_run * self.hooks.matches_per_page
        state["pages_target"] = total_pages_in_run

        log_starting_position(
            f"Processing {total_pages_in_run} pages from page {start_page} to {last_page_to_process}",
            {
                "Total Pages Available": total_pages_api,
                "Pages to Process": total_pages_in_run,
                "Estimated Matches": total_matches_estimate,
                "Start Page": start_page,
                "End Page": last_page_to_process,
            },
        )

        return total_pages_api, last_page_to_process, total_pages_in_run

    def _prepare_page_range(
        self,
        start_page: int,
        state: GatherState,
    ) -> Optional[tuple[int, int]]:
        try:
            _, last_page_to_process, total_pages_in_run = self._handle_initial_fetch(
                start_page,
                state,
            )
        except RuntimeError as exc:
            if str(exc) == "No pages to process":
                logger.info("No pages available for Action 6. Nothing to process this run.")
                return None
            raise

        state["last_page_to_process"] = last_page_to_process
        state["total_pages_in_run"] = total_pages_in_run
        return last_page_to_process, total_pages_in_run

    def _main_page_processing_loop(
        self,
        start_page: int,
        last_page_to_process: int,
        total_pages_in_run: int,
        initial_matches_on_page: Optional[list[dict[str, Any]]],
        state: GatherState,
    ) -> bool:
        action_state_cls = self.hooks.action_state_cls
        dynamic_threshold = self.hooks.calculate_failure_threshold(total_pages_in_run)
        original_threshold = action_state_cls.critical_api_failure_threshold
        action_state_cls.critical_api_failure_threshold = dynamic_threshold

        if dynamic_threshold != original_threshold:
            logger.info(
                "Action 6: API failure threshold adjusted to %d for %d-page run (baseline %d)",
                dynamic_threshold,
                total_pages_in_run,
                original_threshold,
            )
        else:
            logger.debug(
                "Action 6: API failure threshold remains %d for %d-page run",
                dynamic_threshold,
                total_pages_in_run,
            )

        current_page_num = start_page
        total_matches_estimate_this_run = total_pages_in_run * self.hooks.matches_per_page
        if start_page == 1 and initial_matches_on_page is not None:
            total_matches_estimate_this_run = max(total_matches_estimate_this_run, len(initial_matches_on_page))
        if total_matches_estimate_this_run <= 0:
            total_matches_estimate_this_run = self.hooks.matches_per_page

        logger.info("Estimated matches: %s", total_matches_estimate_this_run)

        loop_final_success = True
        matches_on_page_for_batch = initial_matches_on_page

        while current_page_num <= last_page_to_process:
            current_page_num, loop_final_success = self._process_single_page(
                current_page_num,
                start_page,
                matches_on_page_for_batch,
                state,
                loop_final_success,
            )
            matches_on_page_for_batch = None

            persist_checkpoint(
                next_page=current_page_num,
                last_page_to_process=last_page_to_process,
                total_pages_in_run=total_pages_in_run,
                state=state,
            )

            if not loop_final_success:
                break

        return loop_final_success

    def _process_single_page(
        self,
        current_page_num: int,
        start_page: int,
        matches_on_page_for_batch: Optional[list[dict[str, Any]]],
        state: GatherState,
        loop_final_success: bool,
    ) -> tuple[int, bool]:
        log_page_start(current_page_num, state)

        if not self._check_and_handle_session_health(current_page_num):
            return current_page_num, False

        if not self._validate_session_before_page(current_page_num):
            return current_page_num, False

        matches, should_continue, loop_final_success = self._handle_page_fetch_and_validation(
            current_page_num,
            start_page,
            matches_on_page_for_batch,
            state,
            loop_final_success,
        )

        if should_continue:
            return current_page_num + 1, loop_final_success

        if not matches:
            logger.info("No matches found or processed on page %s.", current_page_num)
            time.sleep(0.2)
            return current_page_num + 1, loop_final_success

        if self._try_fast_skip_page(matches, current_page_num, state):
            return current_page_num + 1, loop_final_success

        page_new, page_updated, page_skipped, page_errors, page_metrics = self.hooks.do_batch(
            self.session_manager,
            matches,
            current_page_num,
        )

        self._update_state_and_progress(state, page_new, page_updated, page_skipped, page_errors)

        progress_snapshot = compose_progress_snapshot(state)

        log_page_completion_summary(
            current_page_num,
            page_new,
            page_updated,
            page_skipped,
            page_errors,
            page_metrics,
            progress_snapshot,
        )

        if page_metrics:
            state["aggregate_metrics"] = state.get("aggregate_metrics", PageProcessingMetrics())
            state["pages_with_metrics"] = int(state.get("pages_with_metrics", 0)) + 1
        accumulate_page_metrics(state, page_metrics)

        self._apply_rate_limiting(current_page_num)
        return current_page_num + 1, loop_final_success

    def _handle_page_fetch_and_validation(
        self,
        current_page_num: int,
        start_page: int,
        matches_on_page_for_batch: Optional[list[dict[str, Any]]],
        state: GatherState,
        loop_final_success: bool,
    ) -> tuple[Optional[list[dict[str, Any]]], bool, bool]:
        if current_page_num == start_page and matches_on_page_for_batch is not None:
            return matches_on_page_for_batch, False, loop_final_success

        db_session = self._get_database_session_with_retry(current_page_num, state)
        if not db_session:
            state["total_errors"] += self.hooks.matches_per_page
            if state["db_connection_errors"] >= self.hooks.db_error_page_threshold:
                logger.critical(
                    "Aborting run due to %s consecutive DB connection failures.",
                    state["db_connection_errors"],
                )
                return None, True, False
            return None, True, loop_final_success

        matches = self._fetch_page_matches(db_session, current_page_num, state)
        if not matches:
            time.sleep(0.2 if loop_final_success else 1.0)
            return None, True, loop_final_success

        return matches, False, loop_final_success

    def _get_database_session_with_retry(
        self,
        current_page_num: int,
        state: GatherState,
        max_retries: int = 3,
    ) -> Optional[SqlAlchemySession]:
        db_session: Optional[SqlAlchemySession] = None
        for retry_attempt in range(max_retries):
            db_session = self.session_manager.get_db_conn()
            if db_session:
                state["db_connection_errors"] = 0
                return db_session
            logger.warning(
                "DB session attempt %s/%s failed for page %s. Retrying in 5s...",
                retry_attempt + 1,
                max_retries,
                current_page_num,
            )
            time.sleep(5)

        state["db_connection_errors"] += 1
        logger.error("Could not get DB session for page %s after %s retries.", current_page_num, max_retries)
        return None

    @staticmethod
    def _update_state_and_progress(
        state: GatherState,
        page_new: int,
        page_updated: int,
        page_skipped: int,
        page_errors: int,
    ) -> None:
        state["total_new"] += page_new
        state["total_updated"] += page_updated
        state["total_skipped"] += page_skipped
        state["total_errors"] += page_errors
        state["total_pages_processed"] += 1
        logger.debug(
            "Page totals: %s new, %s updated, %s skipped, %s errors",
            page_new,
            page_updated,
            page_skipped,
            page_errors,
        )

    def _try_fast_skip_page(
        self,
        matches_on_page: list[dict[str, Any]],
        current_page_num: int,
        state: GatherState,
    ) -> bool:
        if not matches_on_page:
            return False

        quick_db_session = self.session_manager.get_db_conn()
        if not quick_db_session:
            return False

        try:
            uuids_on_page = [m["uuid"].upper() for m in matches_on_page if m.get("uuid")]
            if not uuids_on_page:
                return False

            page_statuses = {"skipped": 0}
            lookup_artifacts = gather_process_batch_lookups(
                quick_db_session,
                matches_on_page,
                current_page_num,
                page_statuses,
            )

            page_skip_count = page_statuses.get("skipped", 0)

            if len(lookup_artifacts.fetch_candidates_uuid) == 0:
                logger.info("%s matches unchanged - fast skip", len(matches_on_page))
                state["total_skipped"] += page_skip_count
                state["total_pages_processed"] += 1
                progress_snapshot = compose_progress_snapshot(state)
                log_page_completion_summary(
                    current_page_num,
                    0,
                    0,
                    page_skip_count,
                    0,
                    None,
                    progress_snapshot,
                )
                return True
            return False
        except Exception as fast_skip_exc:  # pragma: no cover - diagnostics only
            logger.debug(
                "Fast-skip lookup pipeline unavailable for page %s: %s",
                current_page_num,
                fast_skip_exc,
                exc_info=True,
            )
            return False
        finally:
            self.session_manager.return_session(quick_db_session)

    def _fetch_page_matches(
        self,
        db_session: SqlAlchemySession,
        current_page_num: int,
        state: GatherState,
    ) -> Optional[list[dict[str, Any]]]:
        try:
            if not self.session_manager.is_sess_valid():
                raise ConnectionError(f"WebDriver session invalid before get_matches page {current_page_num}.")
            result = self.hooks.get_matches(self.session_manager, db_session, current_page_num)
            if result is None:
                logger.warning("get_matches returned None for page %s. Skipping.", current_page_num)
                state["total_errors"] += self.hooks.matches_per_page
                return []
            matches_on_page, _ = result
            return matches_on_page
        except ConnectionError as conn_e:
            logger.error(
                "ConnectionError get_matches page %s: %s",
                current_page_num,
                conn_e,
                exc_info=False,
            )
            state["total_errors"] += self.hooks.matches_per_page
            return []
        except Exception as get_match_e:
            logger.error(
                "Error get_matches page %s: %s",
                current_page_num,
                get_match_e,
                exc_info=True,
            )
            state["total_errors"] += self.hooks.matches_per_page
            return []
        finally:
            if db_session:
                self.session_manager.return_session(db_session)

    def _apply_rate_limiting(self, current_page_num: int) -> None:
        self.hooks.adjust_delay(self.session_manager, current_page_num)
        limiter = getattr(self.session_manager, "dynamic_rate_limiter", None)
        if limiter is not None and hasattr(limiter, "wait"):
            limiter.wait()

    def _check_and_handle_session_health(self, current_page_num: int) -> bool:
        if not self.session_manager.check_session_health():
            self._handle_session_death(current_page_num)
            return False

        self._attempt_proactive_session_refresh()
        self._check_database_pool_health(current_page_num)
        return True

    @staticmethod
    def _handle_session_death(current_page_num: int) -> None:
        logger.critical(
            "🚨 SESSION DEATH DETECTED at page %s. Immediately halting processing to prevent cascade failures.",
            current_page_num,
        )

    def _attempt_proactive_session_refresh(self) -> None:
        start_time = getattr(self.session_manager, "session_start_time", None)
        if not start_time:
            return

        session_age = time.time() - start_time
        if session_age > 800:
            logger.info("Proactively refreshing session after %.0f seconds to prevent timeout", session_age)
            if self.session_manager.attempt_session_recovery(reason="proactive"):
                logger.info("✅ Proactive session refresh successful")
            else:
                logger.error("❌ Proactive session refresh failed")

    def _check_database_pool_health(self, current_page_num: int) -> None:
        if current_page_num % 25 != 0:
            return

        try:
            db_manager = getattr(self.session_manager, "db_manager", None)
            if db_manager and hasattr(db_manager, "get_performance_stats"):
                stats = db_manager.get_performance_stats()
                active_conns = stats.get("active_connections", 0)
                logger.debug(
                    "Database pool status at page %s: %s active connections",
                    current_page_num,
                    active_conns,
                )
            else:
                logger.debug("Database connection pool check at page %s", current_page_num)
        except Exception as pool_opt_exc:  # pragma: no cover - diagnostics only
            logger.debug("Connection pool check at page %s: %s", current_page_num, pool_opt_exc)

    def _validate_session_before_page(self, current_page_num: int) -> bool:
        if not self.session_manager.is_sess_valid():
            logger.critical(
                "WebDriver session invalid/unreachable before processing page %s. Aborting run.",
                current_page_num,
            )
            return False
        return True

    def _log_final_results(self, state: Mapping[str, Any], action_start_time: float) -> None:
        run_time_seconds = time.time() - action_start_time

        summary = {
            "Pages Scanned": state.get("total_pages_processed", 0),
            "New Matches": state.get("total_new", 0),
            "Updated Matches": state.get("total_updated", 0),
            "Skipped (No Change)": state.get("total_skipped", 0),
            "Errors": state.get("total_errors", 0),
            "Total Processed": collect_total_processed(state),
        }
        log_final_summary(summary, run_time_seconds)
        log_timing_breakdown(state)
        self._emit_rate_limiter_metrics()
        self._emit_action_status(state)

    def _emit_rate_limiter_metrics(self) -> None:
        limiter = getattr(self.session_manager, "rate_limiter", None)
        if not limiter:
            return

        metrics = limiter.get_metrics()
        logger.info("Rate Limiter Performance")
        logger.info("Total Requests:        %s", metrics.total_requests)
        logger.info("429 Errors:            %s", metrics.error_429_count)
        logger.info("Current Rate:          %.3f req/s", metrics.current_fill_rate)
        logger.info("Rate Adjustments:      ↓%s ↑%s", metrics.rate_decreases, metrics.rate_increases)
        logger.info("Average Wait Time:     %.3fs", metrics.avg_wait_time)

        try:
            from rate_limiter import persist_rate_limiter_state

            persist_rate_limiter_state(limiter, metrics)
            logger.debug("Persisted rate limiter state for next run reuse")
        except ImportError:
            logger.debug("Rate limiter persistence unavailable (module import failed)")

    @staticmethod
    def _emit_action_status(state: Mapping[str, Any]) -> None:
        details = {
            "pages": state.get("total_pages_processed", 0),
            "new": state.get("total_new", 0),
            "updated": state.get("total_updated", 0),
            "skipped": state.get("total_skipped", 0),
            "errors": state.get("total_errors", 0),
        }
        stage = "success" if state.get("final_success") else "failure"
        log_action_banner(
            action_name="Gather DNA Matches",
            action_number=6,
            stage=stage,
            logger_instance=logger,
            details=details,
        )

    @staticmethod
    def _handle_coord_failure(exc: Exception) -> None:
        if isinstance(exc, ConnectionError):
            logger.critical("ConnectionError during coord execution: %s", exc, exc_info=True)
            return

        if isinstance(exc, MaxApiFailuresExceededError):
            logger.critical(
                "Halting run due to excessive critical API failures: %s",
                exc,
                exc_info=False,
            )
            return

        logger.error("Critical error during coord execution: %s", exc, exc_info=True)

    @with_enhanced_recovery(max_attempts=3, base_delay=2.0, max_delay=60.0)
    @selenium_retry()
    @circuit_breaker(failure_threshold=3, recovery_timeout=60)
    @timeout_protection(timeout=900)
    @error_context("DNA match gathering coordination")
    def _validate_session_state(self) -> None:
        if (
            not self.session_manager.driver
            or not self.session_manager.driver_live
            or not self.session_manager.session_ready
        ):
            raise BrowserSessionError(
                "WebDriver/Session not ready for DNA match gathering",
                context={
                    "driver_live": self.session_manager.driver_live,
                    "session_ready": self.session_manager.session_ready,
                },
            )
        if not self.session_manager.my_uuid:
            raise AuthenticationExpiredError("Failed to retrieve my_uuid for DNA match gathering")


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


def _make_stub_hooks() -> GatherOrchestratorHooks:
    class _FakeActionState:
        critical_api_failure_threshold = 10

    def _noop_batch(*_args: Any, **_kwargs: Any) -> tuple[int, int, int, int, PageProcessingMetrics]:
        return (0, 0, 0, 0, PageProcessingMetrics())

    def _noop_get_matches(*_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        return ([], 1)

    return GatherOrchestratorHooks(
        matches_per_page=20,
        relationship_prob_max_per_page=5,
        db_error_page_threshold=3,
        navigate_and_get_initial_page_data=lambda _sm, _page: ([], 1, True),
        determine_page_processing_range=lambda _total, start: (start, 1),
        do_batch=_noop_batch,
        get_matches=_noop_get_matches,
        adjust_delay=lambda *_args: None,
        action_state_cls=_FakeActionState,
        calculate_failure_threshold=lambda _pages: 10,
    )


def _test_initialize_helpers() -> bool:
    state = _initialize_gather_state()
    assert isinstance(state, dict)
    required_keys = {"total_new", "total_updated", "total_pages_processed"}
    assert required_keys.issubset(state.keys())

    assert _validate_start_page("7") == 7
    assert _validate_start_page(-2) == 1
    assert _validate_start_page("invalid") == 1
    return True


def _test_checkpoint_resume_logic_round_trip() -> bool:
    import json
    import tempfile

    from actions.gather import checkpoint as checkpoint_module

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_file = Path(tmpdir) / "action6_checkpoint.json"
        plan = checkpoint_module.GatherCheckpointPlan(enabled=True, path=checkpoint_file, max_age_hours=24)
        sample_state = {
            "effective_start_page": 2,
            "requested_start_page": None,
            "total_new": 0,
            "total_updated": 0,
            "total_skipped": 0,
            "total_errors": 0,
            "total_pages_processed": 0,
        }

        with mock.patch.object(checkpoint_module, "checkpoint_settings", return_value=plan):
            checkpoint_module.clear_checkpoint(plan)
            start_page, resumed, payload = _determine_start_page(None)
            assert start_page == 1 and resumed is False and payload is None

            checkpoint_module.write_checkpoint_state(5, 10, 20, sample_state, plan=plan)
            assert checkpoint_file.exists()

            start_page, resumed, payload = _determine_start_page(None)
            assert start_page == 5 and resumed is True
            assert payload is not None and payload.get("last_page") == 10

            override_start, override_resumed, _ = _determine_start_page(3)
            assert override_start == 3 and override_resumed is False

            updated_state = {**sample_state, "total_pages_processed": 1}
            persist_checkpoint(
                next_page=6,
                last_page_to_process=10,
                total_pages_in_run=20,
                state=updated_state,
                plan=plan,
            )
            assert json.loads(checkpoint_file.read_text())["next_page"] == 6

            persist_checkpoint(
                next_page=11,
                last_page_to_process=10,
                total_pages_in_run=20,
                state=updated_state,
                plan=plan,
            )
            assert not checkpoint_file.exists()

    return True


def _test_retry_policy_alignment() -> bool:
    selenium_policy = getattr(getattr(config_schema, "retry_policies", None), "selenium", None)
    if selenium_policy is None:
        return False

    helper_name = getattr(GatherOrchestrator._validate_session_state, "__retry_helper__", None)
    policy_name = getattr(GatherOrchestrator._validate_session_state, "__retry_policy__", None)
    settings = getattr(GatherOrchestrator._validate_session_state, "__retry_settings__", {})

    expected_settings = {
        "max_attempts": selenium_policy.max_attempts,
        "backoff_factor": selenium_policy.backoff_factor,
        "base_delay": selenium_policy.initial_delay_seconds,
        "max_delay": selenium_policy.max_delay_seconds,
    }

    return (
        helper_name == "selenium_retry"
        and policy_name == "selenium"
        and all(settings.get(key) == value for key, value in expected_settings.items())
    )


def _test_log_final_results_emits_summary() -> bool:
    class FakeLimiter:
        @staticmethod
        def get_metrics() -> Any:
            class Metrics:
                total_requests = 0
                error_429_count = 0
                current_fill_rate = 0.0
                rate_decreases = 0
                rate_increases = 0
                avg_wait_time = 0.0

            return Metrics()

    class FakeSessionManager:
        rate_limiter: Any = FakeLimiter()

    orchestrator = GatherOrchestrator(cast(SessionManager, FakeSessionManager()), _make_stub_hooks())
    state = {
        "total_pages_processed": 5,
        "total_new": 7,
        "total_updated": 3,
        "total_skipped": 5,
        "total_errors": 1,
        "final_success": True,
    }

    with (
        mock.patch(__name__ + ".log_final_summary") as log_final_mock,
        mock.patch.object(orchestrator, "_emit_rate_limiter_metrics") as rate_mock,
        mock.patch.object(orchestrator, "_emit_action_status") as status_mock,
    ):
        orchestrator._log_final_results(state, action_start_time=time.time() - 42)

    log_final_mock.assert_called_once()
    summary_arg, runtime_arg = log_final_mock.call_args[0]
    assert summary_arg["Pages Scanned"] == state["total_pages_processed"]
    assert summary_arg["Errors"] == state["total_errors"]
    assert runtime_arg > 0
    rate_mock.assert_called_once()
    status_mock.assert_called_once()
    return True


def _test_determine_start_page_prefers_checkpoint() -> bool:
    checkpoint_payload = {"next_page": 5, "last_page": 10}
    with mock.patch(__name__ + ".load_checkpoint", return_value=checkpoint_payload):
        start_page, resumed, payload = _determine_start_page(None)
    assert start_page == 5
    assert resumed is True
    assert payload == checkpoint_payload
    return True


def _test_orchestrator_coord_invokes_execution() -> bool:
    class FakeSessionManager:
        driver = object()
        driver_live = True
        session_ready = True
        my_uuid = "UUID"
        rate_limiter = None
        dynamic_rate_limiter = None
        session_start_time = None

        @staticmethod
        def is_sess_valid() -> bool:
            return True

        @staticmethod
        def check_session_health() -> bool:
            return True

        @staticmethod
        def get_db_conn() -> None:
            return None

        @staticmethod
        def return_session(_session: Any) -> None:
            return None

    orchestrator = GatherOrchestrator(cast(SessionManager, FakeSessionManager()), _make_stub_hooks())
    fake_state = {"final_success": True}

    with (
        mock.patch.object(orchestrator, "_build_coord_state", return_value=(fake_state, 2)) as build_mock,
        mock.patch.object(orchestrator, "_log_action_start") as log_start_mock,
        mock.patch.object(orchestrator, "_execute_coord_run", return_value=True) as exec_mock,
        mock.patch.object(orchestrator, "_log_final_results") as log_final_mock,
        mock.patch(__name__ + ".finalize_checkpoint_after_run") as finalize_mock,
    ):
        result = orchestrator.coord(start=2)

    build_mock.assert_called_once()
    log_start_mock.assert_called_once()
    exec_mock.assert_called_once_with(fake_state, 2)
    finalize_mock.assert_called_once_with(fake_state)
    log_final_mock.assert_called_once()
    assert result is True
    return True


def module_tests() -> bool:
    suite = TestSuite("actions.gather.orchestrator", "actions/gather/orchestrator.py")
    suite.run_test(
        "State initialization & page validation",
        _test_initialize_helpers,
        "Ensures helper functions create valid state dicts and sanitize start pages.",
    )
    suite.run_test(
        "Checkpoint persistence round-trip",
        _test_checkpoint_resume_logic_round_trip,
        "Validates checkpoint save/load behavior and automatic resume handling.",
    )
    suite.run_test(
        "Checkpoint resume detection",
        _test_determine_start_page_prefers_checkpoint,
        "Ensures the start-page helper honors checkpoint metadata.",
    )
    suite.run_test(
        "coord() orchestrates execution",
        _test_orchestrator_coord_invokes_execution,
        "Ensures the orchestrator wires validation, execution, and cleanup steps in order.",
    )
    suite.run_test(
        "Retry helper alignment",
        _test_retry_policy_alignment,
        "Validates selenium retry metadata is preserved on the session validator.",
    )
    suite.run_test(
        "Final summary accuracy",
        _test_log_final_results_emits_summary,
        "Verifies that _log_final_results emits consistent summary metrics and telemetry.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = module_tests()
    raise SystemExit(0 if success else 1)
