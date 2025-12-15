#!/usr/bin/env python3
"""Action execution helpers extracted from main.py."""

from __future__ import annotations

import inspect
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, cast

import psutil

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import logging

from core.action_registry import ActionMetadata, ActionRequirement, get_action_registry
from core.error_handling import (
    ancestry_api_recovery,
    ancestry_database_recovery,
    ancestry_session_recovery,
)

logger = logging.getLogger(__name__)

ActionCallable = Callable[..., Any]
MetricsProvider = Callable[[], Any]


@dataclass
class _RunnerState:
    config: Any | None = None
    metrics_provider: MetricsProvider | None = None


_runner_state = _RunnerState()

if TYPE_CHECKING:
    from core.session_manager import SessionManager


class DatabaseManagerProtocol(Protocol):
    engine: Any
    Session: Any

    def ensure_ready(self) -> bool: ...

    def _initialize_engine_and_session(self) -> None: ...

    def close_connections(self, dispose_engine: bool = False) -> None: ...


class BrowserManagerProtocol(Protocol):
    browser_needed: bool

    def ensure_driver_live(self, action_name: str) -> bool: ...

    def close_driver(self, reason: Optional[str] = None) -> None: ...


class APIManagerProtocol(Protocol):
    csrf_token: str
    tree_owner_name: Optional[str]
    my_tree_id: str

    def sync_cookies_from_browser(
        self,
        browser_manager: BrowserManagerProtocol,
        *,
        session_manager: SessionManager,
    ) -> bool: ...


def configure_action_runner(*, config: Any | None = None, metrics_provider: MetricsProvider | None = None) -> None:
    """Register shared dependencies used by the action runner."""

    if config is not None:
        _runner_state.config = config
    if metrics_provider is not None:
        _runner_state.metrics_provider = metrics_provider


def parse_menu_choice(choice: str) -> tuple[str, list[str]]:
    """Split raw menu input into the action identifier and trailing arguments."""

    tokens = choice.strip().split()
    if not tokens:
        return "", []
    return tokens[0].lower(), tokens[1:]


def get_action_metadata(action_id: str) -> Optional[ActionMetadata]:
    """Return action metadata for the provided identifier."""

    if not action_id:
        return None
    registry = get_action_registry()
    return registry.get_action(action_id)


def get_database_manager(session_manager: SessionManager) -> Optional[DatabaseManagerProtocol]:
    """Safely retrieve the session's DatabaseManager-like component."""

    return cast(Optional[DatabaseManagerProtocol], getattr(session_manager, "db_manager", None))


def get_browser_manager(session_manager: SessionManager) -> Optional[BrowserManagerProtocol]:
    """Safely retrieve the session's BrowserManager-like component."""

    return cast(Optional[BrowserManagerProtocol], getattr(session_manager, "browser_manager", None))


def get_api_manager(session_manager: SessionManager) -> Optional[APIManagerProtocol]:
    """Safely retrieve the session's APIManager-like component."""

    return cast(Optional[APIManagerProtocol], getattr(session_manager, "api_manager", None))


def _determine_browser_requirement(choice: str, metadata: Optional[ActionMetadata] = None) -> bool:
    if metadata is None:
        action_id, _ = parse_menu_choice(choice)
        metadata = get_action_metadata(action_id)

    if metadata:
        return metadata.browser_requirement != ActionRequirement.NONE

    browserless_choices = {"1", "2", "3", "4", "10"}
    return choice not in browserless_choices


def _determine_required_state(
    choice: str,
    requires_browser: bool,
    metadata: Optional[ActionMetadata] = None,
) -> str:
    if metadata is None:
        action_id, _ = parse_menu_choice(choice)
        metadata = get_action_metadata(action_id)

    if metadata:
        if metadata.browser_requirement == ActionRequirement.NONE:
            return "db_ready"
        if metadata.browser_requirement == ActionRequirement.DRIVER_ONLY:
            return "driver_ready"
        return "session_ready"

    if not requires_browser:
        return "db_ready"
    if choice == "5":
        return "driver_ready"
    return "session_ready"


def _ensure_required_state(
    session_manager: SessionManager,
    required_state: str,
    action_name: str,
    choice: str,
    metadata: Optional[ActionMetadata] = None,
) -> bool:
    """Ensure the required state for action execution with recovery strategies.

    Uses recovery strategies from error_handling module when initial setup fails:
    - db_ready: Uses ancestry_database_recovery on failure
    - driver_ready: Uses ancestry_session_recovery on failure
    - session_ready: Uses ancestry_session_recovery + ancestry_api_recovery on failure
    """
    if required_state == "db_ready":
        return _ensure_db_ready(session_manager, action_name)

    if required_state == "driver_ready":
        return _ensure_driver_ready(session_manager, action_name)

    if required_state == "session_ready":
        return _ensure_session_ready(session_manager, action_name, choice, metadata)

    return True


def _ensure_db_ready(session_manager: SessionManager, action_name: str) -> bool:
    db_manager = get_database_manager(session_manager)
    if db_manager is None:
        logger.error("Database manager unavailable for action '%s'", action_name)
        return False

    result = db_manager.ensure_ready()
    if result:
        return True

    logger.info("🔄 Database setup failed, attempting recovery...")
    if ancestry_database_recovery(session_manager):
        result = db_manager.ensure_ready()
        if result:
            logger.info("✅ Database recovery successful")
    return result


def _ensure_driver_ready(session_manager: SessionManager, action_name: str) -> bool:
    browser_manager = get_browser_manager(session_manager)
    if browser_manager is None:
        logger.error("Browser manager unavailable for action '%s'", action_name)
        return False

    result = browser_manager.ensure_driver_live(f"{action_name} - Browser Start")
    if result:
        return True

    logger.info("🔄 Driver setup failed, attempting session recovery...")
    if ancestry_session_recovery(session_manager):
        result = browser_manager.ensure_driver_live(f"{action_name} - Browser Retry")
        if result:
            logger.info("✅ Driver recovery successful")
    return result


def _ensure_session_ready(
    session_manager: SessionManager,
    action_name: str,
    choice: str,
    metadata: Optional[ActionMetadata],
) -> bool:
    skip_csrf = bool(metadata.skip_csrf_check) if metadata else choice in {"10"}
    if not session_manager._guard_action("session_ready", action_name):
        return False

    result = session_manager.ensure_session_ready(
        action_name=f"{action_name} - Setup",
        skip_csrf=skip_csrf,
    )
    if result:
        return True

    logger.info("🔄 Session setup failed, attempting recovery...")
    if ancestry_session_recovery(session_manager):
        ancestry_api_recovery(session_manager)
        result = session_manager.ensure_session_ready(
            action_name=f"{action_name} - Retry",
            skip_csrf=skip_csrf,
        )
        if result:
            logger.info("✅ Session recovery successful")
    return result


def _prepare_action_arguments(
    action_func: ActionCallable,
    session_manager: SessionManager,
    args: tuple[Any, ...],
) -> tuple[list[Any], dict[str, Any]]:
    func_sig = inspect.signature(action_func)
    pass_session_manager = "session_manager" in func_sig.parameters
    action_name = action_func.__name__

    if action_name in {"coord", "gather_dna_matches"} and "start" in func_sig.parameters:
        start_val: Optional[int] = None
        if args:
            potential_start = args[-1]
            if isinstance(potential_start, int):
                start_val = potential_start if potential_start > 0 else None
            elif potential_start is not None:
                logger.debug(
                    "Ignoring unexpected non-integer start argument %s for %s",
                    potential_start,
                    action_name,
                )
        kwargs_for_action: dict[str, Any] = {"start": start_val}

        coord_args: list[Any] = []
        if pass_session_manager:
            coord_args.append(session_manager)
        if action_name == "gather_dna_matches" and "config_schema" in func_sig.parameters:
            coord_args.append(_runner_state.config)

        return coord_args, kwargs_for_action

    final_args: list[Any] = []
    if pass_session_manager:
        final_args.append(session_manager)
    final_args.extend(args)
    return final_args, {}


def _execute_action_function(
    action_func: ActionCallable,
    prepared_args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    if kwargs:
        return action_func(*prepared_args, **kwargs)
    return action_func(*prepared_args)


@dataclass
class _ActionExecutionContext:
    choice: str
    action_name: str
    start_time: float
    process: psutil.Process
    mem_before: float
    result: Any = None
    exception: BaseException | None = None

    @property
    def succeeded(self) -> bool:
        return self.result is not False and self.exception is None


def _initialize_action_context(action_func: ActionCallable, choice: str) -> _ActionExecutionContext:
    action_name = action_func.__name__
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    logger.info(f"{'=' * 45}")
    logger.info(f"Action {choice}: Starting {action_name}...")
    logger.info(f"{'=' * 45}\n")

    return _ActionExecutionContext(
        choice=choice,
        action_name=action_name,
        start_time=start_time,
        process=process,
        mem_before=mem_before,
    )


def _determine_metrics_label(action_result: Any, final_outcome: bool) -> str:
    result_label = "success" if final_outcome else "failure"

    if isinstance(action_result, str):
        candidate = action_result.lower()
        if candidate == "skipped":
            return candidate

    if isinstance(action_result, tuple):
        typed_result = cast(tuple[Any, ...], action_result)
        if len(typed_result) > 1:
            label_candidate = typed_result[1]
            if isinstance(label_candidate, str):
                normalized = label_candidate.lower()
                if normalized in {"success", "failure", "skipped"}:
                    return normalized

    return result_label


def _record_action_analytics(
    context: _ActionExecutionContext,
    *,
    duration_sec: float,
    mem_used_mb: float | None,
) -> None:
    final_outcome = context.succeeded

    try:
        from observability.analytics import log_event, pop_transient_extras

        extras = pop_transient_extras()
        log_event(
            action_name=context.action_name,
            choice=context.choice,
            success=bool(final_outcome),
            duration_sec=duration_sec,
            mem_used_mb=mem_used_mb,
            extras=extras,
        )
    except Exception as exc:
        logger.debug(f"Analytics logging skipped: {exc}")

    try:
        result_label = _determine_metrics_label(context.result, final_outcome)
        metrics_bundle = _runner_state.metrics_provider() if _runner_state.metrics_provider is not None else None
        if metrics_bundle is not None:
            metrics_bundle.action_processed.inc(context.action_name, result_label)
            metrics_bundle.action_duration.observe(context.action_name, duration_sec)
    except Exception:
        logger.debug("Failed to record action throughput metric", exc_info=True)


def _should_close_session(
    action_result: Any,
    action_exception: BaseException | None,
    close_sess_after: bool,
    action_name: str,
    browser_was_used: bool = False,
) -> bool:
    # Always close browser if it was used, regardless of success/failure
    if browser_was_used:
        if action_result is False or action_exception is not None:
            logger.debug(f"Closing browser after '{action_name}' failed (browser was used).")
        else:
            logger.debug(f"Closing browser after '{action_name}' completed successfully (browser was used).")
        return True
    if close_sess_after:
        if action_result is False or action_exception is not None:
            logger.debug(f"Action '{action_name}' failed but close_sess_after=True, closing session.")
        else:
            logger.debug(f"Closing session after '{action_name}' as requested by caller (close_sess_after=True).")
        return True
    return False


def _log_performance_metrics(
    start_time: float,
    process: psutil.Process,
    mem_before: float,
    choice: str,
    action_name: str,
) -> tuple[float, float | None]:
    duration = time.time() - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_duration = f"{int(hours)} hr {int(minutes)} min {seconds:.2f} sec"

    mem_used: float | None = None
    try:
        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_used = mem_after - mem_before
        mem_log = f"Memory used: {mem_used:.1f} MB"
    except Exception as mem_err:
        mem_log = f"Memory usage unavailable: {mem_err}"

    logger.info(f"{'=' * 45}")
    logger.info(f"Action {choice} ({action_name}) finished.")
    logger.info(f"Duration: {formatted_duration}")
    logger.info(mem_log)
    logger.info(f"{'=' * 45}\n")
    return duration, mem_used


def _perform_session_cleanup(session_manager: SessionManager, should_close: bool, action_name: str) -> None:
    if should_close:
        # Always close browser if driver is live and we're told to close
        if session_manager.browser_manager.driver_live:
            logger.debug("Closing browser session...")
            session_manager.close_sess(keep_db=True)
            logger.debug("Browser session closed. DB connections kept.")
        elif action_name in {"all_but_first_actn"}:
            logger.debug("Closing all connections including database...")
            session_manager.close_sess(keep_db=False)
            logger.debug("All connections closed.")
        return

    if session_manager.browser_manager.driver_live:
        logger.debug(f"Keeping session live after '{action_name}'.")


def _finalize_action_execution(
    context: _ActionExecutionContext,
    session_manager: SessionManager,
    close_sess_after: bool,
) -> bool:
    browser_was_used = session_manager.browser_manager.browser_needed
    should_close = _should_close_session(
        context.result,
        context.exception,
        close_sess_after,
        context.action_name,
        browser_was_used,
    )

    print(" ")

    if context.result is False:
        logger.debug(f"Action {context.choice} ({context.action_name}) reported failure.")
    elif context.exception is not None:
        logger.debug(
            f"Action {context.choice} ({context.action_name}) failed due to exception: "
            f"{type(context.exception).__name__}."
        )

    logger.debug(f"Final outcome for Action {context.choice} ('{context.action_name}'): {context.succeeded}\n")

    duration_sec, mem_used_mb = _log_performance_metrics(
        context.start_time,
        context.process,
        context.mem_before,
        context.choice,
        context.action_name,
    )

    _record_action_analytics(context, duration_sec=duration_sec, mem_used_mb=mem_used_mb)
    _perform_session_cleanup(session_manager, should_close, context.action_name)

    return context.succeeded


def exec_actn(
    action_func: ActionCallable,
    session_manager: SessionManager,
    choice: str,
    close_sess_after: bool = False,
    *args: Any,
) -> bool:
    context = _initialize_action_context(action_func, choice)
    final_outcome = False

    action_metadata = get_action_metadata(choice)

    requires_browser = _determine_browser_requirement(choice, action_metadata)
    session_manager.browser_manager.browser_needed = requires_browser
    required_state = _determine_required_state(choice, requires_browser, action_metadata)

    try:
        state_ok = _ensure_required_state(
            session_manager,
            required_state,
            context.action_name,
            choice,
            action_metadata,
        )
        if not state_ok:
            logger.error(
                "Failed to achieve required state '%s' for action '%s'.",
                required_state,
                context.action_name,
            )
            raise Exception(f"Setup failed: Could not achieve state '{required_state}'.")

        prepared_args, kwargs = _prepare_action_arguments(action_func, session_manager, args)
        context.result = _execute_action_function(action_func, prepared_args, kwargs)

    except Exception as exc:
        logger.error(f"Exception during action {context.action_name}: {exc}", exc_info=True)
        context.result = False
        context.exception = exc

    finally:
        final_outcome = _finalize_action_execution(context, session_manager, close_sess_after)

    return final_outcome


# ============================================================================
# MODULE TESTS
# ============================================================================


def _test_parse_menu_choice_behavior() -> bool:
    action_id, args = parse_menu_choice(" 6   42 extra ")
    assert action_id == "6", "Menu parsing should normalize the action id"
    assert args == ["42", "extra"], "Trailing tokens must be preserved in order"

    empty_choice = parse_menu_choice("   ")
    assert empty_choice == ("", []), "Whitespace-only choices should return empty identifiers"
    return True


def _test_prepare_action_arguments_special_cases() -> bool:
    configure_action_runner(config={"sentinel": True})
    fake_session = cast("SessionManager", object())

    def gather_dna_matches(session_manager: Any, config_schema: Any | None = None, start: int | None = None) -> None:
        del session_manager, config_schema, start

    prepared_args, kwargs = _prepare_action_arguments(gather_dna_matches, fake_session, (7,))
    assert prepared_args == [fake_session, {"sentinel": True}], "Special cases should inject session and config"
    assert kwargs == {"start": 7}, "Start argument should be forwarded through kwargs"

    def regular_action(session_manager: Any, payload: str) -> None:
        del session_manager, payload

    prepared_args, kwargs = _prepare_action_arguments(regular_action, fake_session, ("payload",))
    assert prepared_args == [fake_session, "payload"], "Session manager precedes positional args"
    assert kwargs == {}, "General actions should not inject kwargs"
    return True


def _test_determine_metrics_label() -> bool:
    tuple_result = ("ignored", "skipped")
    assert _determine_metrics_label(tuple_result, True) == "skipped", "Tuple labels should override defaults"
    assert _determine_metrics_label("SKIPPED", False) == "skipped", "String results should normalize case"
    assert _determine_metrics_label("unexpected", False) == "failure", "Fallback should match final outcome"
    return True


def _test_should_close_session_logic() -> bool:
    # Browser was used: ALWAYS close, regardless of success/failure
    assert _should_close_session(False, None, False, "test", True), "Browser usage closes even on failure"
    assert _should_close_session("ok", ValueError("boom"), False, "test", True), (
        "Browser usage closes even on exception"
    )
    assert _should_close_session("ok", None, False, "test", True), "Browser usage triggers close on success"

    # Explicit close_sess_after=True: always close
    assert _should_close_session("ok", None, True, "test", False), "close_sess_after=True closes session"
    assert _should_close_session(False, None, True, "test", False), "close_sess_after=True closes even on failure"

    # No browser, no explicit close: keep session alive
    assert not _should_close_session("ok", None, False, "test", False), "No browser, no close flag keeps session"
    assert not _should_close_session(False, None, False, "test", False), (
        "No browser, failure, no close flag keeps session"
    )
    return True


def action_runner_module_tests() -> bool:
    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite("core/action_runner.py - Action Runner Helpers", "core/action_runner.py")

    with suppress_logging():
        suite.run_test(
            "Menu choice parsing",
            _test_parse_menu_choice_behavior,
            "parse_menu_choice normalizes identifiers and arguments",
            "parse_menu_choice",
            "Ensures whitespace handling and argument preservation",
        )

        suite.run_test(
            "Argument preparation",
            _test_prepare_action_arguments_special_cases,
            "_prepare_action_arguments injects config and session when required",
            "_prepare_action_arguments",
            "Validates coord/gather special-case flow and general path",
        )

        suite.run_test(
            "Metrics label detection",
            _test_determine_metrics_label,
            "_determine_metrics_label prioritizes tuple/string results",
            "_determine_metrics_label",
            "Verifies tuple override, case normalization, and fallback",
        )

        suite.run_test(
            "Session closure logic",
            _test_should_close_session_logic,
            "_should_close_session only closes when explicitly requested on success",
            "_should_close_session",
            "Ensures failures/exceptions keep sessions alive while successes honor the flag",
        )

    return suite.finish_suite()


from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(action_runner_module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
