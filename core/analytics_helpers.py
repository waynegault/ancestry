from importlib import import_module
from typing import Any, Callable, Optional, cast

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

MatchRecord = dict[str, Any]
MatchList = list[MatchRecord]
RowBuilder = Callable[[MatchRecord], list[str]]
MatchAnalyzer = Callable[[Any, MatchRecord, Optional[str], str], None]
SupplementaryHandler = Callable[[MatchRecord, Any], None]  # SessionManager is Any to avoid circular import
IDNormalizer = Callable[[Optional[str]], Optional[str]]
AnalyticsExtrasSetter = Callable[[dict[str, Any]], None]


def load_result_row_builders() -> tuple[RowBuilder, RowBuilder]:
    """Load row builder helpers from GEDCOM and API modules with type safety."""

    action10_module = import_module("action10")
    api_search_module = import_module("api_search_core")
    gedcom_builder = cast(RowBuilder, getattr(action10_module, "_create_table_row"))
    api_builder = cast(RowBuilder, getattr(api_search_module, "_create_table_row_for_candidate"))
    return gedcom_builder, api_builder


def load_match_analysis_helpers() -> tuple[IDNormalizer, MatchAnalyzer, SupplementaryHandler]:
    """Load helper functions used when rendering match details."""

    action10_module = import_module("action10")
    api_search_module = import_module("api_search_core")
    normalize_id = cast(IDNormalizer, getattr(action10_module, "normalize_gedcom_id"))
    analyze_top_match = cast(MatchAnalyzer, getattr(action10_module, "analyze_top_match"))
    handle_supplementary = cast(
        SupplementaryHandler,
        getattr(api_search_module, "_handle_supplementary_info_phase"),
    )
    return normalize_id, analyze_top_match, handle_supplementary


def set_comparison_mode_analytics(gedcom_count: int, api_count: int) -> None:
    """Record analytics metadata when comparison mode runs successfully."""

    try:
        analytics_module = import_module("analytics")
    except Exception:
        return

    setter = getattr(analytics_module, "set_transient_extras", None)
    if not callable(setter):
        return

    extras = {
        "comparison_mode": True,
        "gedcom_candidates": gedcom_count,
        "api_candidates": api_count,
    }
    try:
        cast(AnalyticsExtrasSetter, setter)(extras)
    except Exception:
        logger.debug("Analytics extras setter failed", exc_info=True)


def get_metrics_bundle() -> Optional[Any]:
    """Return a metrics bundle if the observability module is available."""
    try:
        metrics_module = import_module("observability.metrics_registry")
        metrics_factory = getattr(metrics_module, "metrics", None)
        if callable(metrics_factory):
            return metrics_factory()
    except Exception:  # pragma: no cover - metrics are optional
        logger.debug("Metrics bundle request failed", exc_info=True)
        return None
    return None
