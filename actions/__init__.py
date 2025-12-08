"""Action package entry point.

This package contains the main action modules for the Ancestry automation:
- action6_gather: DNA match gathering from Ancestry
- action7_inbox: Inbox processing and message classification
- action8_messaging: Automated messaging to DNA matches
- action9_process_productive: Processing productive conversations
- action10: GEDCOM analysis and genealogical intelligence

It also hosts extractions from action6_gather module for better organization:
- gather/: Checkpoint, fetch, metrics, orchestration, persistence helpers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .gather.checkpoint import GatherCheckpointPlan, GatherCheckpointService
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
    from actions.action6_gather import coord as gather_coord


# Action module lazy imports to avoid circular dependencies
_ACTION_MODULES = {
    "action6_gather": ".action6_gather",
    "action7_inbox": ".action7_inbox",
    "action8_messaging": ".action8_messaging",
    "action9_process_productive": ".action9_process_productive",
    "action10": ".action10",
}


def __getattr__(name: str):
    """Lazy import to avoid circular import with action modules."""
    if name == "gather_coord":
        from actions.action6_gather import coord

        return coord
    if name in _ACTION_MODULES:
        import importlib

        return importlib.import_module(_ACTION_MODULES[name], __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Gather subpackage exports
    "GatherBatchSummary",
    "GatherCheckpointPlan",
    "GatherCheckpointService",
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
