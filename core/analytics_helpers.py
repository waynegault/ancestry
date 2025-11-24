import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, cast
from unittest import mock

ModulePatch = Any

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from standard_imports import setup_module
from test_framework import TestSuite, create_standard_test_runner

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


# ------------------------------------------------------------------
# Embedded regression tests
# ------------------------------------------------------------------


def _build_fake_module(**attrs: Any) -> SimpleNamespace:
    return SimpleNamespace(**attrs)


def _module_patch(target: str, module_map: dict[str, Any]) -> ModulePatch:
    def _fake_import(name: str) -> Any:
        return module_map[name]

    return mock.patch(target, side_effect=_fake_import)


def _test_load_result_row_builders_returns_callables() -> bool:
    def gedcom_builder(_: MatchRecord) -> list[str]:
        return ["gedcom"]

    def api_builder(_: MatchRecord) -> list[str]:
        return ["api"]

    module_map = {
        "action10": _build_fake_module(_create_table_row=gedcom_builder),
        "api_search_core": _build_fake_module(_create_table_row_for_candidate=api_builder),
    }

    with _module_patch(f"{__name__}.import_module", module_map):
        gedcom_fn, api_fn = load_result_row_builders()

    assert gedcom_fn is gedcom_builder
    assert api_fn is api_builder
    return True


def _test_load_match_analysis_helpers_returns_expected_helpers() -> bool:
    def normalize(value: Optional[str]) -> Optional[str]:
        return value.upper() if value else None

    def analyze(_: Any, __: MatchRecord, ___: Optional[str], ____: str) -> None:
        return None

    def handle(_: MatchRecord, __: Any) -> None:
        return None

    module_map = {
        "action10": _build_fake_module(
            normalize_gedcom_id=normalize,
            analyze_top_match=analyze,
        ),
        "api_search_core": _build_fake_module(
            _handle_supplementary_info_phase=handle,
        ),
    }

    with _module_patch(f"{__name__}.import_module", module_map):
        normalizer, analyzer, supplementary = load_match_analysis_helpers()

    assert normalizer is normalize
    assert analyzer is analyze
    assert supplementary is handle
    return True


def _test_set_comparison_mode_analytics_invokes_setter() -> bool:
    captured: dict[str, Any] = {}

    def setter(extras: dict[str, Any]) -> None:
        captured.update(extras)

    module_map = {"analytics": _build_fake_module(set_transient_extras=setter)}

    with _module_patch(f"{__name__}.import_module", module_map):
        set_comparison_mode_analytics(3, 7)

    assert captured == {
        "comparison_mode": True,
        "gedcom_candidates": 3,
        "api_candidates": 7,
    }
    return True


def _test_set_comparison_mode_analytics_handles_missing_dependencies() -> bool:
    with mock.patch(f"{__name__}.import_module", side_effect=ImportError()):
        set_comparison_mode_analytics(1, 1)

    module_map = {"analytics": _build_fake_module(set_transient_extras=None)}
    with _module_patch(f"{__name__}.import_module", module_map):
        set_comparison_mode_analytics(2, 2)
    return True


def _test_get_metrics_bundle_returns_factory_value() -> bool:
    sentinel = object()

    def metrics_factory() -> Any:
        return sentinel

    module_map = {"observability.metrics_registry": _build_fake_module(metrics=metrics_factory)}

    with _module_patch(f"{__name__}.import_module", module_map):
        result = get_metrics_bundle()

    assert result is sentinel
    return True


def module_tests() -> bool:
    suite = TestSuite("core.analytics_helpers", "core/analytics_helpers.py")

    suite.run_test(
        "Load result row builders",
        _test_load_result_row_builders_returns_callables,
        "Ensures row builders resolve from action10/api modules.",
    )

    suite.run_test(
        "Load match analysis helpers",
        _test_load_match_analysis_helpers_returns_expected_helpers,
        "Ensures match analysis helpers resolve and return expected callables.",
    )

    suite.run_test(
        "Set comparison analytics setter",
        _test_set_comparison_mode_analytics_invokes_setter,
        "Ensures analytics extras setter receives comparison metadata.",
    )

    suite.run_test(
        "Comparison analytics fallback",
        _test_set_comparison_mode_analytics_handles_missing_dependencies,
        "Ensures analytics helper tolerates missing modules or setters.",
    )

    suite.run_test(
        "Metrics bundle factory",
        _test_get_metrics_bundle_returns_factory_value,
        "Ensures metrics bundle returns factory output when available.",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = module_tests()
    import sys as _sys

    _sys.exit(0 if success else 1)
