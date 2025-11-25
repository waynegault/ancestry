"""Action package entry point.

This package hosts in-progress extractions from the legacy
`action6_gather.py` module so that future refactors can
incrementally migrate orchestration, fetching, checkpointing,
persistence, and metrics helpers without breaking the existing
entry points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .gather.checkpoint import GatherCheckpointPlan, GatherCheckpointService
from .gather.fetch import GatherFetchPlan, GatherFetchService
from .gather.metrics import (
    PageProcessingMetrics,
    accumulate_page_metrics,
    collect_total_processed,
    compose_progress_snapshot,
    detailed_endpoint_lines,
    format_duration_with_avg,
    log_page_completion_summary,
    log_page_start,
    log_timing_breakdown,
    log_timing_breakdown_details,
)
from .gather.orchestrator import GatherOrchestrator
from .gather.persistence import GatherBatchSummary, GatherPersistenceService

if TYPE_CHECKING:
    from action6_gather import coord as gather_coord


def __getattr__(name: str):
    """Lazy import to avoid circular import with action6_gather."""
    if name == "gather_coord":
        from action6_gather import coord

        return coord
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GatherBatchSummary",
    "GatherCheckpointPlan",
    "GatherCheckpointService",
    "GatherFetchPlan",
    "GatherFetchService",
    "GatherOrchestrator",
    "GatherPersistenceService",
    "PageProcessingMetrics",
    "accumulate_page_metrics",
    "collect_total_processed",
    "compose_progress_snapshot",
    "detailed_endpoint_lines",
    "format_duration_with_avg",
    "gather_coord",
    "log_page_completion_summary",
    "log_page_start",
    "log_timing_breakdown",
    "log_timing_breakdown_details",
]
