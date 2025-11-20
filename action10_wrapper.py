import logging
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Optional, cast

from core.action_runner import (
    get_api_manager,
    get_browser_manager,
)
from core.analytics_helpers import (
    IDNormalizer,
    MatchAnalyzer,
    MatchList,
    RowBuilder,
    SupplementaryHandler,
    load_match_analysis_helpers,
    load_result_row_builders,
    set_comparison_mode_analytics,
)
from core.session_manager import SessionManager
from standard_imports import setup_module

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


def _display_detailed_match_info(
    gedcom_matches: MatchList,
    api_matches: MatchList,
    gedcom_data: Any,
    _reference_person_id_raw: Optional[str],
    _reference_person_name: Optional[str],
    session_manager: SessionManager,
) -> None:
    """Display detailed information for top match."""
    normalize_id: Optional[IDNormalizer] = None
    analyze_top_match_fn: Optional[MatchAnalyzer] = None
    supplementary_handler: Optional[SupplementaryHandler] = None

    try:
        normalize_id, analyze_top_match_fn, supplementary_handler = load_match_analysis_helpers()
    except Exception as exc:
        logger.error(f"Unable to load match analysis helpers: {exc}", exc_info=True)

    try:
        if gedcom_matches and gedcom_data is not None and analyze_top_match_fn is not None:
            if _reference_person_id_raw and normalize_id is not None:
                ref_norm = normalize_id(_reference_person_id_raw)
            else:
                ref_norm = _reference_person_id_raw
            analyze_top_match_fn(
                gedcom_data,
                gedcom_matches[0],
                ref_norm,
                _reference_person_name or "Reference Person",
            )
    except Exception as e:
        logger.error(f"GEDCOM family/relationship display failed: {e}")

    try:
        if api_matches and not gedcom_matches and supplementary_handler is not None:
            supplementary_handler(api_matches[0], session_manager)
    except Exception as e:
        logger.error(f"API family/relationship display failed: {e}")


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

    return bool(
        comparison_results.gedcom_matches or comparison_results.api_matches
    )
