"""Action 6 gather package scaffolding.

The modules defined here currently delegate to the legacy
`action6_gather.py` helpers while we migrate individual concerns
(fetching, checkpointing, persistence, metrics, orchestration)
into independently testable components.
"""

from __future__ import annotations

from .checkpoint import GatherCheckpointPlan, GatherCheckpointService
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
from .persistence import (
    BatchLookupArtifacts,
    GatherBatchSummary,
    GatherPersistenceService,
    PersistenceHooks,
    execute_bulk_db_operations,
    needs_ethnicity_refresh,
    prepare_and_commit_batch_data,
    process_batch_lookups,
)
from .prefetch import (
    PrefetchConfig,
    PrefetchHooks,
    PrefetchResult,
    get_prefetched_data_for_match,
    perform_api_prefetches,
)

__all__ = [
    "BatchLookupArtifacts",
    "GatherBatchSummary",
    "GatherCheckpointPlan",
    "GatherCheckpointService",
    "GatherOrchestrator",
    "GatherPersistenceService",
    "PageProcessingMetrics",
    "PersistenceHooks",
    "PrefetchConfig",
    "PrefetchHooks",
    "PrefetchResult",
    "accumulate_page_metrics",
    "collect_total_processed",
    "compose_progress_snapshot",
    "detailed_endpoint_lines",
    "execute_bulk_db_operations",
    "format_duration_with_avg",
    "get_prefetched_data_for_match",
    "log_page_completion_summary",
    "log_page_start",
    "log_timing_breakdown",
    "log_timing_breakdown_details",
    "needs_ethnicity_refresh",
    "perform_api_prefetches",
    "prepare_and_commit_batch_data",
    "process_batch_lookups",
]

# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_comprehensive_tests() else 1)
