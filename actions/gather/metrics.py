from __future__ import annotations

import sys
import time
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional
from unittest import mock

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from standard_imports import setup_module
from test_framework import TestSuite, create_standard_test_runner

logger = setup_module(globals(), __name__)

PREFETCH_ENDPOINT_LABELS: dict[str, str] = {
    "combined_details": "Match profile",
    "relationship_prob": "Relationship probability",
    "badge_details": "DNA badge",
    "ladder_details": "Tree ladder",
    "ethnicity": "Ethnicity",
}


@dataclass
class PageProcessingMetrics:
    """Aggregated telemetry for a processed page."""

    total_matches: int = 0
    fetch_candidates: int = 0
    existing_matches: int = 0
    db_seconds: float = 0.0
    prefetch_seconds: float = 0.0
    commit_seconds: float = 0.0
    total_seconds: float = 0.0
    batches: int = 0
    idle_seconds: float = 0.0
    prefetch_breakdown: dict[str, float] = field(default_factory=dict)
    prefetch_call_counts: dict[str, int] = field(default_factory=dict)

    def merge(self, other: PageProcessingMetrics) -> PageProcessingMetrics:
        """Combine metrics from another batch into this aggregate."""

        self.total_matches += other.total_matches
        self.fetch_candidates += other.fetch_candidates
        self.existing_matches += other.existing_matches
        self.db_seconds += other.db_seconds
        self.prefetch_seconds += other.prefetch_seconds
        self.commit_seconds += other.commit_seconds
        self.total_seconds += other.total_seconds
        self.batches += other.batches
        self.idle_seconds += other.idle_seconds
        for endpoint, duration in other.prefetch_breakdown.items():
            self.prefetch_breakdown[endpoint] = self.prefetch_breakdown.get(endpoint, 0.0) + duration
        for endpoint, count in other.prefetch_call_counts.items():
            self.prefetch_call_counts[endpoint] = self.prefetch_call_counts.get(endpoint, 0) + count
        return self

    def has_signal(self) -> bool:
        """Return True if the metrics include timing data worth logging."""

        return any(
            value > 0
            for value in (
                self.total_seconds,
                self.prefetch_seconds,
                self.db_seconds,
                self.commit_seconds,
            )
        )


def _format_brief_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--"

    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"

    minutes, secs = divmod(int(seconds), 60)
    if minutes == 0:
        return f"{seconds:.1f}s"

    hours, mins = divmod(minutes, 60)
    if hours == 0:
        return f"{mins}m {secs:02d}s"

    return f"{hours}h {mins:02d}m"


def format_duration_with_avg(total_seconds: float, denominator: float, unit: str) -> str:
    if total_seconds <= 0:
        return "0.00s"
    if denominator <= 0:
        return f"{total_seconds:.2f}s"

    average = total_seconds / denominator
    if average >= 1.0:
        return f"{total_seconds:.2f}s (avg={average:.2f}s/{unit})"

    average_ms = average * 1000.0
    if average_ms >= 100.0:
        return f"{total_seconds:.2f}s (avg={average_ms:.0f}ms/{unit})"
    return f"{total_seconds:.2f}s (avg={average_ms:.1f}ms/{unit})"


def _iter_endpoint_stats(
    breakdown: dict[str, float],
    counts: dict[str, int],
):
    for endpoint, total in sorted(breakdown.items(), key=lambda item: item[1], reverse=True):
        if total <= 0:
            continue
        count = counts.get(endpoint, 0)
        if count <= 0:
            continue
        yield endpoint, total, count, total / count


def _format_avg_value(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.2f}s"
    return f"{seconds * 1000:.0f}ms"


def _format_endpoint_breakdown(
    breakdown: dict[str, float],
    counts: dict[str, int],
    limit: int | None = 3,
    *,
    style: Literal["inline", "list"] = "inline",
) -> str:
    entries: list[str] = []
    for endpoint, total, count, _avg in _iter_endpoint_stats(breakdown, counts):
        label = PREFETCH_ENDPOINT_LABELS.get(endpoint, endpoint)
        duration_summary = format_duration_with_avg(total, float(count), "call")
        entries.append(f"{label}={duration_summary}")
        if limit and len(entries) >= limit:
            break

    if not entries:
        return ""

    if style == "list":
        return "\n".join(f"- {entry}" for entry in entries)

    return " | ".join(entries)


def detailed_endpoint_lines(
    breakdown: dict[str, float],
    counts: dict[str, int],
) -> list[str]:
    lines: list[str] = []
    for endpoint, total, count, avg in _iter_endpoint_stats(breakdown, counts):
        label = PREFETCH_ENDPOINT_LABELS.get(endpoint, endpoint)
        avg_display = _format_avg_value(avg)
        lines.append(f"{label}: total {total:.2f}s across {count} calls (avg {avg_display})")
    return lines


def _log_timing_snapshot(pages_tracked: int, metrics: PageProcessingMetrics) -> None:
    if pages_tracked <= 1:
        return

    breakdown_limit = 3 if pages_tracked < 10 else None
    snapshot = _format_endpoint_breakdown(
        metrics.prefetch_breakdown,
        metrics.prefetch_call_counts,
        limit=breakdown_limit,
    )
    if not snapshot:
        return

    logger.info(f"Timing snapshot after {pages_tracked} page(s): {snapshot}")


def collect_total_processed(state: Mapping[str, Any]) -> int:
    """Return the total number of matches processed successfully."""

    return int(state.get("total_new", 0)) + int(state.get("total_updated", 0)) + int(state.get("total_skipped", 0))


def log_timing_breakdown_details(
    aggregate_metrics: PageProcessingMetrics,
    pages_with_metrics: int,
    matches_for_avg: int,
    total_processed_matches: int,
) -> None:
    """Emit detailed timing statistics for the run."""

    logger.info("Timing Breakdown")
    logger.info(f"Tracked Pages:        {pages_with_metrics}")
    logger.info(
        "Tracked Matches:      %s",
        aggregate_metrics.total_matches or total_processed_matches,
    )

    if aggregate_metrics.total_seconds and pages_with_metrics:
        avg_page_duration = aggregate_metrics.total_seconds / pages_with_metrics
        logger.info(f"Avg Page Duration:    {avg_page_duration:.2f}s")

    api_per_page = (
        f"{(aggregate_metrics.prefetch_seconds / pages_with_metrics):.2f}s/page"
        if aggregate_metrics.prefetch_seconds
        else "0.00s/page"
    )
    logger.info(
        "API Prefetch Time:    %s (%s)",
        format_duration_with_avg(
            aggregate_metrics.prefetch_seconds,
            float(aggregate_metrics.fetch_candidates),
            "call",
        ),
        api_per_page,
    )
    logger.info(
        "DB Lookup Time:       %s",
        format_duration_with_avg(
            aggregate_metrics.db_seconds,
            matches_for_avg,
            "match",
        ),
    )
    logger.info(
        "Commit Time:          %s",
        format_duration_with_avg(
            aggregate_metrics.commit_seconds,
            matches_for_avg,
            "match",
        ),
    )
    logger.info(
        "Total Processing:     %s",
        format_duration_with_avg(
            aggregate_metrics.total_seconds,
            matches_for_avg,
            "match",
        ),
    )
    if aggregate_metrics.idle_seconds > 0.0:
        logger.info(
            "Pacing Delay:        %s",
            format_duration_with_avg(
                aggregate_metrics.idle_seconds,
                matches_for_avg,
                "match",
            ),
        )
    if aggregate_metrics.fetch_candidates:
        logger.info(
            "API Calls/Page:      %.1f",
            aggregate_metrics.fetch_candidates / pages_with_metrics,
        )
    if aggregate_metrics.total_seconds > 0 and total_processed_matches > 0:
        throughput = total_processed_matches / aggregate_metrics.total_seconds
        logger.info(f"Avg Throughput:      {throughput:.2f} match/s")

    endpoint_lines = detailed_endpoint_lines(
        aggregate_metrics.prefetch_breakdown,
        aggregate_metrics.prefetch_call_counts,
    )
    if endpoint_lines:
        logger.info("API Endpoint Averages:")
        for line in endpoint_lines:
            logger.info(f"  • {line}")


def log_timing_breakdown(state: Mapping[str, Any]) -> None:
    """Log timing metrics when aggregate data is available."""

    aggregate_metrics = state.get("aggregate_metrics")
    pages_with_metrics = int(state.get("pages_with_metrics", 0) or 0)
    if not isinstance(aggregate_metrics, PageProcessingMetrics) or pages_with_metrics <= 0:
        return

    total_processed_matches = collect_total_processed(state)
    matches_for_avg = max(
        aggregate_metrics.total_matches,
        total_processed_matches,
        1,
    )
    log_timing_breakdown_details(
        aggregate_metrics,
        pages_with_metrics,
        matches_for_avg,
        total_processed_matches,
    )


def log_page_start(current_page: int, state: Mapping[str, Any], *, now: Optional[float] = None) -> None:
    pages_done = int(state.get("total_pages_processed", 0))
    pages_target = int(state.get("pages_target") or 0)
    page_index = pages_done + 1

    if pages_target <= 0:
        pages_target = page_index
    pages_target = max(pages_target, page_index)

    run_started_at = state.get("run_started_at")
    reference_time = now or time.time()
    elapsed = (reference_time - float(run_started_at)) if run_started_at is not None else None
    avg_per_page = (elapsed / pages_done) if elapsed and pages_done else None
    pages_remaining = max(pages_target - page_index, 0)
    eta = avg_per_page * pages_remaining if avg_per_page else None
    percent_complete = (pages_done / pages_target * 100.0) if pages_target else 0.0

    tokens = [f"Page {current_page} ({page_index} of {pages_target})", f"{percent_complete:.0f}% complete"]

    if elapsed is not None:
        tokens.append(f"elapsed {_format_brief_duration(elapsed)}")
    if eta is not None:
        tokens.append(f"ETA {_format_brief_duration(eta)}")

    logger.info(" | ".join(tokens))


def compose_progress_snapshot(state: Mapping[str, Any], *, now: Optional[float] = None) -> dict[str, Any]:
    pages_done = int(state.get("total_pages_processed", 0))
    pages_target = int(state.get("pages_target") or 0)
    if pages_target <= 0:
        pages_target = max(pages_done, 1)

    pages_target = max(pages_target, pages_done)

    run_started_at = state.get("run_started_at")
    reference_time = now or time.time()
    elapsed = (reference_time - float(run_started_at)) if run_started_at is not None else None

    avg_per_page = (elapsed / pages_done) if elapsed and pages_done else None
    pages_remaining = max(pages_target - pages_done, 0)
    eta = avg_per_page * pages_remaining if avg_per_page else None

    percent_complete = (pages_done / pages_target * 100.0) if pages_target else 100.0

    return {
        "page_index": max(pages_done, 1),
        "pages_target": pages_target,
        "pages_done": pages_done,
        "percent": percent_complete,
        "elapsed": elapsed,
        "eta": eta,
    }


def log_page_completion_summary(
    page: int,
    page_new: int,
    page_updated: int,
    page_skipped: int,
    page_errors: int,
    metrics: Optional[PageProcessingMetrics],
    progress: Optional[dict[str, Any]] = None,
) -> None:
    lines: list[str] = [f"Page {page} complete:"]

    if progress:
        percent = progress.get("percent", 0.0)
        elapsed = _format_brief_duration(progress.get("elapsed"))
        eta = _format_brief_duration(progress.get("eta"))

        lines.append(f"  - {percent:.0f}% of total downloaded")
        if elapsed != "--":
            lines.append(f"  - took {elapsed}")
        if eta != "--":
            lines.append(f"  - ETA {eta} to full download")

    lines.append(f"  - new={page_new} updated={page_updated} skipped={page_skipped} errors={page_errors}")

    total_processed = page_new + page_updated + page_skipped
    if metrics and metrics.total_seconds:
        avg_rate = (total_processed / metrics.total_seconds) if total_processed and metrics.total_seconds else 0.0
        lines.append(f"  - rate={avg_rate:.2f} match/s")

        breakdown_list = _format_endpoint_breakdown(
            metrics.prefetch_breakdown,
            metrics.prefetch_call_counts,
            style="list",
        )
        if breakdown_list:
            lines.append("  API endpoints (by total time):")
            for entry in breakdown_list.splitlines():
                lines.append(f"    {entry}")
    elif not metrics:
        lines.append("  - metrics unavailable for this page")

    logger.info("\n".join(lines))


def accumulate_page_metrics(state: MutableMapping[str, Any], page_metrics: Optional[PageProcessingMetrics]) -> None:
    if not isinstance(page_metrics, PageProcessingMetrics) or not page_metrics.has_signal():
        return

    aggregate_metrics = state.get("aggregate_metrics")
    if not isinstance(aggregate_metrics, PageProcessingMetrics):
        aggregate_metrics = PageProcessingMetrics()
        state["aggregate_metrics"] = aggregate_metrics

    aggregate_metrics.merge(page_metrics)
    state["pages_with_metrics"] = int(state.get("pages_with_metrics", 0)) + 1

    pages_tracked = state["pages_with_metrics"]
    if pages_tracked in {1, 5} or pages_tracked % 10 == 0:
        _log_timing_snapshot(pages_tracked, aggregate_metrics)


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


def _test_compose_progress_snapshot() -> bool:
    state: dict[str, Any] = {"total_pages_processed": 5, "pages_target": 10, "run_started_at": 0.0}
    snapshot = compose_progress_snapshot(state, now=50.0)
    assert snapshot["percent"] == 50.0
    assert snapshot["eta"] == 50.0  # avg 10s per page, 5 pages remaining
    return True


def _test_accumulate_page_metrics_records_state() -> bool:
    state: dict[str, Any] = {}
    metrics = PageProcessingMetrics(total_seconds=2.0, prefetch_seconds=1.0, db_seconds=0.5, commit_seconds=0.5)
    accumulate_page_metrics(state, metrics)
    assert isinstance(state["aggregate_metrics"], PageProcessingMetrics)
    assert state["pages_with_metrics"] == 1
    return True


def _test_log_page_completion_summary_emits_info() -> bool:
    metrics = PageProcessingMetrics(total_seconds=2.0)
    metrics.prefetch_breakdown["combined_details"] = 1.0
    metrics.prefetch_call_counts["combined_details"] = 2
    progress = {"percent": 25.0, "elapsed": 5.0, "eta": 15.0}

    with mock.patch.object(logger, "info") as info_mock:
        log_page_completion_summary(2, 1, 1, 0, 0, metrics, progress)
        info_mock.assert_called_once()
        message = info_mock.call_args[0][0]
        assert "Page 2 complete" in message
        assert "Match profile" in message
    return True


def _test_collect_total_processed_counts() -> bool:
    state = {"total_new": 3, "total_updated": 4, "total_skipped": 5}
    assert collect_total_processed(state) == 12
    return True


def _test_log_timing_breakdown_with_metrics() -> bool:
    metrics = PageProcessingMetrics(
        total_matches=2,
        fetch_candidates=2,
        db_seconds=0.5,
        prefetch_seconds=0.5,
        commit_seconds=0.5,
        total_seconds=1.5,
    )
    state: dict[str, Any] = {
        "aggregate_metrics": metrics,
        "pages_with_metrics": 1,
        "total_new": 1,
        "total_updated": 1,
        "total_skipped": 0,
    }

    with mock.patch.object(logger, "info") as info_mock:
        log_timing_breakdown(state)
        info_mock.assert_called()
    return True


def module_tests() -> bool:
    suite = TestSuite("actions.gather.metrics", "actions/gather/metrics.py")
    suite.run_test(
        "Compose progress snapshot",
        _test_compose_progress_snapshot,
        "Ensures percent/ETA math mirrors the legacy helper.",
    )
    suite.run_test(
        "Accumulate page metrics",
        _test_accumulate_page_metrics_records_state,
        "Ensures per-page metrics roll into the aggregate container.",
    )
    suite.run_test(
        "Log page completion summary",
        _test_log_page_completion_summary_emits_info,
        "Ensures the structured summary reaches the logger with endpoint labels.",
    )
    suite.run_test(
        "Collect total processed",
        _test_collect_total_processed_counts,
        "Ensures match counters add up correctly.",
    )
    suite.run_test(
        "Log timing breakdown",
        _test_log_timing_breakdown_with_metrics,
        "Ensures timing breakdown emits logger output when metrics exist.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = module_tests()
    raise SystemExit(0 if success else 1)
