"""Action 6 gather package scaffolding.

The modules defined here currently delegate to the legacy
`action6_gather.py` helpers while we migrate individual concerns
(fetching, checkpointing, persistence, metrics, orchestration)
into independently testable components.
"""

from __future__ import annotations

from .checkpoint import GatherCheckpointPlan, GatherCheckpointService
from .fetch import GatherFetchPlan, GatherFetchService
from .metrics import (
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
from .orchestrator import GatherOrchestrator
from .prefetch import (
    PrefetchConfig,
    PrefetchHooks,
    PrefetchResult,
    get_prefetched_data_for_match,
    perform_api_prefetches,
)
from .persistence import (
    BatchLookupArtifacts,
    GatherBatchSummary,
    GatherPersistenceService,
    PersistenceHooks,
    needs_ethnicity_refresh,
    prepare_and_commit_batch_data,
    process_batch_lookups,
)

__all__ = [
    "GatherBatchSummary",
    "BatchLookupArtifacts",
    "GatherCheckpointPlan",
    "GatherCheckpointService",
    "GatherFetchPlan",
    "GatherFetchService",
    "GatherOrchestrator",
    "GatherPersistenceService",
    "PersistenceHooks",
    "PrefetchConfig",
    "PrefetchHooks",
    "PrefetchResult",
    "PageProcessingMetrics",
    "accumulate_page_metrics",
    "collect_total_processed",
    "compose_progress_snapshot",
    "detailed_endpoint_lines",
    "format_duration_with_avg",
    "log_page_completion_summary",
    "log_page_start",
    "log_timing_breakdown",
    "log_timing_breakdown_details",
    "needs_ethnicity_refresh",
    "prepare_and_commit_batch_data",
    "process_batch_lookups",
    "get_prefetched_data_for_match",
    "perform_api_prefetches",
]
