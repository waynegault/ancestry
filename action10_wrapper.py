import io
import logging
import os
import sys
from collections.abc import Sequence
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Optional, cast

from core.action_runner import (
    get_api_manager,
    get_browser_manager,
)
from core.analytics_helpers import (
    IDNormalizer,
    MatchAnalyzer,
    MatchList,
    SupplementaryHandler,
    load_match_analysis_helpers,
    load_result_row_builders,
    set_comparison_mode_analytics,
)
from core.session_manager import SessionManager
from standard_imports import setup_module
from test_framework import TestSuite, create_standard_test_runner

logger = setup_module(globals(), __name__)

SearchAPIFunc = Callable[["SessionManager", dict[str, Any], int], MatchList]


@dataclass
class _ComparisonConfig:
    """Configuration inputs needed for the GEDCOM/API comparison run."""

    gedcom_path: Optional[Path]
    reference_person_id_raw: Optional[str]
    reference_person_name: Optional[str]
    date_flex: Optional[dict[str, Any]]
    scoring_weights: dict[str, Any]
    max_display_results: int


@dataclass
class _ComparisonResults:
    """Container for search results spanning GEDCOM and API sources."""

    gedcom_data: Any
    gedcom_matches: MatchList
    api_matches: MatchList


def _perform_gedcom_search(
    gedcom_path: Optional[Path],
    criteria: dict[str, Any],
    scoring_weights: dict[str, Any],
    date_flex: Optional[dict[str, Any]],
) -> tuple[Any, MatchList]:
    """Perform GEDCOM search and return data and matches."""

    action10_module = import_module("action10")
    load_gedcom_data_fn = cast(Callable[[Path], Any], getattr(action10_module, "load_gedcom_data"))
    build_filter_criteria = cast(Callable[[dict[str, Any]], Any], getattr(action10_module, "_build_filter_criteria"))
    filter_and_score = cast(Callable[..., MatchList], getattr(action10_module, "filter_and_score_individuals"))

    gedcom_data: Any = None
    gedcom_matches: MatchList = []

    if gedcom_path is not None:
        try:
            gedcom_data = load_gedcom_data_fn(gedcom_path)
            filter_criteria = build_filter_criteria(criteria)
            normalized_date_flex: dict[str, Any] = date_flex or {}
            gedcom_matches = filter_and_score(
                gedcom_data,
                filter_criteria,
                criteria,
                scoring_weights,
                normalized_date_flex,
            )
        except Exception as exc:
            logger.error(f"GEDCOM search failed: {exc}")

    return gedcom_data, gedcom_matches


def _perform_api_search_fallback(
    session_manager: SessionManager,
    criteria: dict[str, Any],
    max_results: int,
) -> MatchList:
    """Perform API search as fallback when GEDCOM has no matches."""

    api_module = import_module("api_search_core")
    search_api = cast(
        SearchAPIFunc,
        getattr(api_module, "search_ancestry_api_for_person"),
    )

    try:
        session_ok = session_manager.ensure_session_ready(
            action_name="GEDCOM/API Search - API Fallback", skip_csrf=False
        )
        if not session_ok:
            logger.error("Could not establish browser session for API search")
            return []

        api_manager = get_api_manager(session_manager)
        browser_manager = get_browser_manager(session_manager)
        if api_manager is None or browser_manager is None:
            logger.error("Browser/API managers unavailable for API fallback search")
            return []

        try:
            synced = api_manager.sync_cookies_from_browser(
                browser_manager,
                session_manager=session_manager,
            )
            if not synced:
                logger.warning("Cookie sync from browser failed, but attempting API search anyway")
        except Exception as sync_err:
            logger.warning(f"Cookie sync error: {sync_err}, but attempting API search anyway")

        return search_api(session_manager, criteria, max_results)
    except Exception as exc:
        logger.error(f"API search failed: {exc}", exc_info=True)
        return []


def _format_table_row(row: Sequence[str], widths: Sequence[int]) -> str:
    """Return padded string for display rows."""

    return " | ".join(col.ljust(width) for col, width in zip(row, widths))


def _compute_table_widths(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> list[int]:
    """Return column widths based on headers and row content."""

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    return widths


def _log_result_table(label: str, rows: list[list[str]], total: int, headers: list[str]) -> None:
    """Log table output for GEDCOM/API matches when debug logging is enabled."""
    widths = _compute_table_widths(rows, headers)
    header_row = _format_table_row(headers, widths)

    logger.debug("")
    logger.debug(f"=== {label} Results (Top {len(rows)} of {total}) ===")
    logger.debug(header_row)
    logger.debug("-" * len(header_row))

    if rows:
        for row in rows:
            logger.debug(_format_table_row(row, widths))
    else:
        logger.debug("(no matches)")


def _display_search_results(gedcom_matches: MatchList, api_matches: MatchList, max_to_show: int) -> None:
    """Display GEDCOM and API search results in tables."""
    try:
        create_row_gedcom, create_row_api = load_result_row_builders()
    except Exception as exc:
        logger.error(f"Unable to load search-result row builders: {exc}", exc_info=True)
        print("Unable to display search results (internal helper unavailable).")
        return

    headers = ["ID", "Name", "Birth", "Birth Place", "Death", "Death Place", "Total"]
    left_rows: list[list[str]] = [create_row_gedcom(m) for m in gedcom_matches[:max_to_show]]
    right_rows: list[list[str]] = [create_row_api(m) for m in api_matches[:max_to_show]]

    if not left_rows and not right_rows:
        print("")
        print("No matches found.")
        return

    if logger.isEnabledFor(logging.DEBUG):
        _log_result_table("GEDCOM", left_rows, len(gedcom_matches), headers)
        _log_result_table("API", right_rows, len(api_matches), headers)
        logger.debug("")
        logger.debug(
            f"Summary: GEDCOM — showing top {len(left_rows)} of {len(gedcom_matches)} total | "
            f"API — showing top {len(right_rows)} of {len(api_matches)} total"
        )


def _load_match_helpers_safely() -> tuple[
    Optional[IDNormalizer], Optional[MatchAnalyzer], Optional[SupplementaryHandler]
]:
    try:
        return load_match_analysis_helpers()
    except Exception as exc:
        logger.error(f"Unable to load match analysis helpers: {exc}", exc_info=True)
        return None, None, None


def _display_gedcom_details(
    gedcom_matches: MatchList,
    gedcom_data: Any,
    reference_person_id_raw: Optional[str],
    reference_person_name: Optional[str],
    normalize_id: Optional[IDNormalizer],
    analyze_top_match_fn: Optional[MatchAnalyzer],
) -> None:
    if not (gedcom_matches and gedcom_data is not None and analyze_top_match_fn):
        return
    try:
        ref_norm = reference_person_id_raw
        if reference_person_id_raw and normalize_id is not None:
            ref_norm = normalize_id(reference_person_id_raw)
        analyze_top_match_fn(
            gedcom_data,
            gedcom_matches[0],
            ref_norm,
            reference_person_name or "Reference Person",
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"GEDCOM family/relationship display failed: {exc}")


def _display_api_details(
    gedcom_matches: MatchList,
    api_matches: MatchList,
    session_manager: SessionManager,
    supplementary_handler: Optional[SupplementaryHandler],
) -> None:
    if not (api_matches and not gedcom_matches and supplementary_handler):
        return
    try:
        supplementary_handler(api_matches[0], session_manager)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"API family/relationship display failed: {exc}")


def _display_detailed_match_info(
    gedcom_matches: MatchList,
    api_matches: MatchList,
    gedcom_data: Any,
    _reference_person_id_raw: Optional[str],
    _reference_person_name: Optional[str],
    session_manager: SessionManager,
) -> None:
    """Display detailed information for top match."""
    normalize_id, analyze_top_match_fn, supplementary_handler = _load_match_helpers_safely()

    _display_gedcom_details(
        gedcom_matches,
        gedcom_data,
        _reference_person_id_raw,
        _reference_person_name,
        normalize_id,
        analyze_top_match_fn,
    )
    _display_api_details(gedcom_matches, api_matches, session_manager, supplementary_handler)


def _collect_comparison_inputs() -> Optional[tuple[_ComparisonConfig, dict[str, Any]]]:
    """Load search criteria plus configuration needed for comparison mode."""

    try:
        from action10 import validate_config
        from search_criteria_utils import get_unified_search_criteria
    except Exception as exc:
        logger.error(f"Side-by-side setup failed: {exc}", exc_info=True)
        return None

    criteria = get_unified_search_criteria()
    if not criteria:
        return None

    (
        gedcom_path,
        reference_person_id_raw,
        reference_person_name,
        date_flex,
        scoring_weights,
        max_display_results,
    ) = validate_config()

    config = _ComparisonConfig(
        gedcom_path=gedcom_path,
        reference_person_id_raw=reference_person_id_raw,
        reference_person_name=reference_person_name,
        date_flex=date_flex,
        scoring_weights=scoring_weights,
        max_display_results=max_display_results,
    )
    return config, criteria


def _execute_comparison_search(
    session_manager: SessionManager,
    *,
    comparison_config: _ComparisonConfig,
    criteria: dict[str, Any],
) -> _ComparisonResults:
    """Run GEDCOM search followed by API fallback when needed."""

    gedcom_data, gedcom_matches = _perform_gedcom_search(
        comparison_config.gedcom_path,
        criteria,
        comparison_config.scoring_weights,
        comparison_config.date_flex,
    )

    api_matches: MatchList = []
    if not gedcom_matches:
        api_matches = _perform_api_search_fallback(
            session_manager,
            criteria,
            comparison_config.max_display_results,
        )
    else:
        logger.debug("Skipping API search because GEDCOM returned matches.")

    return _ComparisonResults(
        gedcom_data=gedcom_data,
        gedcom_matches=gedcom_matches,
        api_matches=api_matches,
    )


def _render_comparison_results(
    session_manager: SessionManager,
    *,
    comparison_config: _ComparisonConfig,
    comparison_results: _ComparisonResults,
) -> None:
    """Display summary tables, detail view, and analytics for comparison mode."""

    _display_search_results(
        comparison_results.gedcom_matches,
        comparison_results.api_matches,
        max_to_show=1,
    )

    _display_detailed_match_info(
        comparison_results.gedcom_matches,
        comparison_results.api_matches,
        comparison_results.gedcom_data,
        comparison_config.reference_person_id_raw,
        comparison_config.reference_person_name,
        session_manager,
    )

    set_comparison_mode_analytics(
        len(comparison_results.gedcom_matches),
        len(comparison_results.api_matches),
    )


def run_gedcom_then_api_fallback(session_manager: SessionManager, *_: Any) -> bool:
    """Action 10: GEDCOM-first search with API fallback; unified presentation (header → family → relationship)."""
    collected = _collect_comparison_inputs()
    if not collected:
        return False

    comparison_config, criteria = collected
    comparison_results = _execute_comparison_search(
        session_manager,
        comparison_config=comparison_config,
        criteria=criteria,
    )

    _render_comparison_results(
        session_manager,
        comparison_config=comparison_config,
        comparison_results=comparison_results,
    )

    return bool(comparison_results.gedcom_matches or comparison_results.api_matches)


_PATCH_SENTINEL = object()


@contextmanager
def _temporary_module(module_name: str, module: ModuleType):
    original = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        yield module
    finally:
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original


@contextmanager
def _patched_globals(replacements: dict[str, Any]):
    previous: dict[str, Any] = {}
    try:
        for attr, value in replacements.items():
            previous[attr] = globals().get(attr, _PATCH_SENTINEL)
            globals()[attr] = value
        yield
    finally:
        for attr in replacements:
            original = previous[attr]
            if original is _PATCH_SENTINEL:
                globals().pop(attr, None)
            else:
                globals()[attr] = original


def _test_table_helpers_compute_widths_and_format_rows() -> bool:
    headers = ["Short", "Longer"]
    rows = [["aaa", "bbb"], ["cccccc", "d"]]
    widths = _compute_table_widths(rows, headers)
    assert widths == [6, 6]
    formatted = _format_table_row(rows[0], widths)
    left, right = formatted.split(" | ")
    assert left == rows[0][0].ljust(widths[0])
    assert right == rows[0][1].ljust(widths[1])
    return True


def _test_perform_gedcom_search_invokes_action10_helpers() -> bool:
    captured: dict[str, Any] = {}

    def fake_load(path: Path) -> dict[str, str]:
        captured["load"] = path
        return {"path": str(path)}

    def fake_build(criteria: dict[str, Any]) -> dict[str, Any]:
        captured["build"] = dict(criteria)
        return {"filtered": True}

    def fake_filter(
        data: Any,
        filter_criteria: dict[str, Any],
        criteria: dict[str, Any],
        scoring_weights: dict[str, Any],
        date_flex: dict[str, Any],
    ) -> MatchList:
        captured["filter"] = (data, filter_criteria, criteria, scoring_weights, date_flex)
        return [{"id": "M1"}]

    fake_module = ModuleType("action10")
    setattr(fake_module, "load_gedcom_data", fake_load)
    setattr(fake_module, "_build_filter_criteria", fake_build)
    setattr(fake_module, "filter_and_score_individuals", fake_filter)

    with _temporary_module("action10", fake_module):
        data, matches = _perform_gedcom_search(
            Path("family.ged"),
            {"name": "Ada"},
            {"score": 1},
            {"years": 2},
        )

    assert data == {"path": "family.ged"}
    assert matches == [{"id": "M1"}]
    assert captured["build"]["name"] == "Ada"
    assert captured["filter"][4] == {"years": 2}
    return True


def _test_perform_api_search_fallback_returns_fallback_matches() -> bool:
    class FakeSessionManager:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def ensure_session_ready(self, *, action_name: str, skip_csrf: bool) -> bool:
            self.calls.append((action_name, skip_csrf))
            return True

    class FakeAPIManager:
        def __init__(self) -> None:
            self.sync_calls: list[tuple[Any, Any]] = []

        def sync_cookies_from_browser(self, browser_manager: Any, session_manager: Any) -> bool:
            self.sync_calls.append((browser_manager, session_manager))
            return True

    fake_session_raw = FakeSessionManager()
    fake_session = cast(SessionManager, fake_session_raw)
    fake_api_manager = FakeAPIManager()
    fake_browser_manager = object()

    def fake_get_api_manager(_session_manager: SessionManager) -> FakeAPIManager:
        return fake_api_manager

    def fake_get_browser_manager(_session_manager: SessionManager) -> Any:
        return fake_browser_manager

    api_calls: list[tuple[dict[str, Any], int]] = []

    def fake_search(_session: SessionManager, criteria: dict[str, Any], max_results: int) -> MatchList:
        api_calls.append((criteria, max_results))
        return [{"id": "API1"}]

    fake_api_module = ModuleType("api_search_core")
    setattr(fake_api_module, "search_ancestry_api_for_person", fake_search)

    with (
        _temporary_module("api_search_core", fake_api_module),
        _patched_globals(
            {
                "get_api_manager": fake_get_api_manager,
                "get_browser_manager": fake_get_browser_manager,
            }
        ),
    ):
        result = _perform_api_search_fallback(fake_session, {"surname": "Jones"}, 3)

    assert result == [{"id": "API1"}]
    assert api_calls == [({"surname": "Jones"}, 3)]
    assert fake_api_manager.sync_calls
    assert fake_session_raw.calls
    return True


def _test_display_search_results_uses_row_builders() -> bool:
    gedcom_matches = [{"id": "G1", "name": "Left"}]
    api_matches = [{"id": "A1", "name": "Right"}]
    left_calls: list[str] = []
    right_calls: list[str] = []

    def build_left(match: dict[str, str]) -> list[str]:
        left_calls.append(match["id"])
        return [match["id"], match["name"]]

    def build_right(match: dict[str, str]) -> list[str]:
        right_calls.append(match["id"])
        return [match["id"], match["name"]]

    with (
        _patched_globals({"load_result_row_builders": lambda: (build_left, build_right)}),
        redirect_stdout(io.StringIO()),
    ):
        _display_search_results(gedcom_matches, api_matches, max_to_show=1)

    assert left_calls == ["G1"]
    assert right_calls == ["A1"]
    return True


def _test_display_search_results_handles_empty_case() -> bool:
    buffer = io.StringIO()
    with (
        _patched_globals({"load_result_row_builders": lambda: (lambda _m: ["id"], lambda _m: ["id"])}),
        redirect_stdout(buffer),
    ):
        _display_search_results([], [], max_to_show=2)
    assert "No matches found." in buffer.getvalue()
    return True


def _test_display_detailed_match_info_invokes_analysis_helpers() -> bool:
    normalize_calls: list[str] = []
    analyze_calls: list[tuple[Any, Any, Any, Any]] = []
    supplementary_calls: list[Any] = []

    def normalize(value: Optional[str]) -> str:
        normalize_calls.append(value or "")
        return f"norm-{value}"

    def analyze(data: Any, match: Any, ref_norm: Any, ref_name: str) -> None:
        analyze_calls.append((data, match, ref_norm, ref_name))

    def supplementary(_match: Any, _session: Any) -> None:
        supplementary_calls.append(True)

    with _patched_globals({"load_match_analysis_helpers": lambda: (normalize, analyze, supplementary)}):
        _display_detailed_match_info(
            [{"id": "G"}],
            [],
            {"data": True},
            "REF1",
            "Person",
            cast(SessionManager, SimpleNamespace()),
        )

    assert normalize_calls == ["REF1"]
    assert analyze_calls and analyze_calls[0][2] == "norm-REF1"
    assert not supplementary_calls
    return True


def _test_display_detailed_match_info_invokes_supplementary_when_needed() -> bool:
    calls: list[str] = []

    def normalize(value: Optional[str]) -> Optional[str]:
        return value

    def analyze(*_args: Any, **_kwargs: Any) -> None:
        calls.append("analyze")

    def supplementary(match: Any, _session: Any) -> None:
        calls.append(f"supplement:{match['id']}")

    with _patched_globals({"load_match_analysis_helpers": lambda: (normalize, analyze, supplementary)}):
        _display_detailed_match_info(
            [],
            [{"id": "API"}],
            None,
            None,
            None,
            cast(SessionManager, SimpleNamespace()),
        )

    assert calls == ["supplement:API"]
    return True


def _test_execute_comparison_search_skips_api_when_matches_exist() -> bool:
    config = _ComparisonConfig(
        gedcom_path=None,
        reference_person_id_raw=None,
        reference_person_name=None,
        date_flex=None,
        scoring_weights={"w": 1},
        max_display_results=1,
    )

    def fake_gedcom(*_args: Any, **_kwargs: Any) -> tuple[Any, MatchList]:
        return ({"data": True}, [{"id": "G1"}])

    fallback_calls: list[Any] = []

    def fake_fallback(*_args: Any, **_kwargs: Any) -> MatchList:
        fallback_calls.append(True)
        return [{"id": "API1"}]

    with _patched_globals(
        {
            "_perform_gedcom_search": fake_gedcom,
            "_perform_api_search_fallback": fake_fallback,
        }
    ):
        results = _execute_comparison_search(
            cast(SessionManager, SimpleNamespace()),
            comparison_config=config,
            criteria={"city": "Boston"},
        )

    assert results.gedcom_matches == [{"id": "G1"}]
    assert results.api_matches == []
    assert not fallback_calls
    return True


def _test_execute_comparison_search_triggers_api_when_needed() -> bool:
    config = _ComparisonConfig(
        gedcom_path=None,
        reference_person_id_raw=None,
        reference_person_name=None,
        date_flex=None,
        scoring_weights={"w": 1},
        max_display_results=1,
    )

    def fake_gedcom(*_args: Any, **_kwargs: Any) -> tuple[Any, MatchList]:
        return (None, [])

    def fake_fallback(*_args: Any, **_kwargs: Any) -> MatchList:
        return [{"id": "API-only"}]

    with _patched_globals(
        {
            "_perform_gedcom_search": fake_gedcom,
            "_perform_api_search_fallback": fake_fallback,
        }
    ):
        results = _execute_comparison_search(
            cast(SessionManager, SimpleNamespace()),
            comparison_config=config,
            criteria={"country": "FR"},
        )

    assert results.gedcom_matches == []
    assert results.api_matches == [{"id": "API-only"}]
    return True


def _test_run_gedcom_then_api_fallback_handles_missing_inputs() -> bool:
    with _patched_globals({"_collect_comparison_inputs": lambda: None}):
        assert run_gedcom_then_api_fallback(cast(SessionManager, SimpleNamespace())) is False
    return True


def _test_run_gedcom_then_api_fallback_renders_results() -> bool:
    config = _ComparisonConfig(
        gedcom_path=None,
        reference_person_id_raw=None,
        reference_person_name=None,
        date_flex=None,
        scoring_weights={"w": 1},
        max_display_results=2,
    )

    comparison = _ComparisonResults(gedcom_data=None, gedcom_matches=[{"id": "G1"}], api_matches=[])
    render_calls: list[tuple[Any, Any]] = []

    def fake_collect() -> tuple[_ComparisonConfig, dict[str, Any]]:
        return config, {"given": "criteria"}

    def fake_execute(_session: SessionManager, **_kwargs: Any) -> _ComparisonResults:
        return comparison

    def fake_render(session_manager: SessionManager, **kwargs: Any) -> None:
        render_calls.append((session_manager, kwargs))

    with _patched_globals(
        {
            "_collect_comparison_inputs": fake_collect,
            "_execute_comparison_search": fake_execute,
            "_render_comparison_results": fake_render,
        }
    ):
        result = run_gedcom_then_api_fallback(cast(SessionManager, SimpleNamespace()))

    assert result is True
    assert len(render_calls) == 1
    assert "comparison_results" in render_calls[0][1]
    return True


def module_tests() -> bool:
    suite = TestSuite("action10_wrapper", "action10_wrapper.py")
    suite.run_test(
        "Table formatting helpers",
        _test_table_helpers_compute_widths_and_format_rows,
        "Ensures column widths and row formatting stay aligned for console tables.",
    )
    suite.run_test(
        "GEDCOM search orchestration",
        _test_perform_gedcom_search_invokes_action10_helpers,
        "Validates GEDCOM search wires through to action10 helpers.",
    )
    suite.run_test(
        "API fallback orchestration",
        _test_perform_api_search_fallback_returns_fallback_matches,
        "Confirms API fallback syncs cookies and returns API results.",
    )
    suite.run_test(
        "Result display builders",
        _test_display_search_results_uses_row_builders,
        "Ensures GEDCOM/API row builders are invoked for each match.",
    )
    suite.run_test(
        "Empty result display",
        _test_display_search_results_handles_empty_case,
        "Ensures empty-result messaging is shown when no matches exist.",
    )
    suite.run_test(
        "Match detail helpers",
        _test_display_detailed_match_info_invokes_analysis_helpers,
        "Validates normalization/analyzer helpers execute for GEDCOM matches.",
    )
    suite.run_test(
        "Supplementary handler",
        _test_display_detailed_match_info_invokes_supplementary_when_needed,
        "Ensures API supplementary handler runs when GEDCOM lacks matches.",
    )
    suite.run_test(
        "Comparison execution (GEDCOM matches)",
        _test_execute_comparison_search_skips_api_when_matches_exist,
        "Ensures API fallback is skipped when GEDCOM already produced matches.",
    )
    suite.run_test(
        "Comparison execution (API fallback)",
        _test_execute_comparison_search_triggers_api_when_needed,
        "Ensures API fallback executes when GEDCOM returns nothing.",
    )
    suite.run_test(
        "Run action handles missing config",
        _test_run_gedcom_then_api_fallback_handles_missing_inputs,
        "Ensures run function exits gracefully without inputs.",
    )
    suite.run_test(
        "Run action renders results",
        _test_run_gedcom_then_api_fallback_renders_results,
        "Ensures happy-path run executes render step and returns truthy value.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


def _should_run_module_tests() -> bool:
    return os.environ.get("RUN_MODULE_TESTS") == "1"


def _print_module_usage() -> int:
    print("action10_wrapper exposes helpers for GEDCOM/API comparison orchestration and has no CLI entry point.")
    print("Set RUN_MODULE_TESTS=1 before execution to run the embedded regression tests.")
    return 0


if __name__ == "__main__":
    if _should_run_module_tests():
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    sys.exit(_print_module_usage())
