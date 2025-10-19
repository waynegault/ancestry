#!/usr/bin/env python3
"""
common_params.py - Common Parameter Grouping Dataclasses

This module provides dataclasses for grouping commonly-used function parameters
to reduce parameter counts and improve code maintainability.
"""

# === CORE INFRASTRUCTURE ===
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class GraphContext:
    """
    Graph traversal context for genealogical relationship calculations.

    Used in gedcom_utils.py and relationship_utils.py for BFS/DFS operations.
    Accepts both dict[str, list[str]] and dict[str, set[str]] for flexibility.
    """
    id_to_parents: Mapping[str, Collection[str]]
    id_to_children: Mapping[str, Collection[str]]
    current_id: Optional[str] = None
    start_id: Optional[str] = None
    end_id: Optional[str] = None


@dataclass
class RetryContext:
    """
    Retry logic parameters for API calls and error handling.

    Used across multiple modules for exponential backoff retry logic.
    """
    attempt: int
    max_attempts: int
    max_delay: float
    backoff_factor: float = 2.0
    current_delay: float = 1.0
    retries_left: Optional[int] = None
    retry_status_codes: Optional[Union[list[int], set[int]]] = None


@dataclass
class MatchIdentifiers:
    """
    DNA match identification parameters.

    Used in action6_gather.py for processing DNA matches.
    """
    uuid: Optional[str]
    username: str
    in_my_tree: bool
    log_ref_short: str
    profile_id: Optional[str] = None


@dataclass
class ConversationIdentifiers:
    """
    Conversation/messaging identification parameters.

    Used in action7_inbox.py and action8_messaging.py.
    """
    api_conv_id: str
    people_id: Optional[str] = None
    my_pid_lower: Optional[str] = None
    effective_conv_id: Optional[str] = None
    log_prefix: Optional[str] = None


@dataclass
class ApiIdentifiers:
    """
    API-related identification parameters.

    Used in api_utils.py and related modules.
    """
    owner_profile_id: str
    api_person_id: Optional[str] = None
    api_tree_id: Optional[str] = None
    owner_tree_id: Optional[str] = None


@dataclass
class BatchCounters:
    """
    Batch processing counters.

    Used in action6_gather.py and other batch processing modules.
    """
    new: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    total: int = 0
    sent: int = 0
    acked: int = 0


@dataclass
class BatchConfig:
    """
    Batch processing configuration.

    Used in action8_messaging.py and other batch processing modules.
    """
    commit_batch_size: int
    max_memory_mb: int
    max_items: int
    max_messages_to_send: int = 0


@dataclass
class ConversationProcessingContext:
    """
    Context for processing conversations in action7_inbox.py.

    Groups lookup maps and batch data collections.
    """
    existing_persons_map: dict[str, Any]
    existing_conv_logs: dict[tuple, Any]
    conv_log_upserts_dicts: list[dict[str, Any]]
    person_updates: dict[str, Any]
    comp_conv_id: Optional[str] = None
    comp_ts: Optional[Any] = None
    my_pid_lower: Optional[str] = None
    min_aware_dt: Optional[Any] = None


@dataclass
class MessagingBatchData:
    """
    Batch data collections for messaging operations.

    Used in action8_messaging.py for collecting database updates.
    """
    db_logs_to_add_dicts: list[dict[str, Any]]
    person_updates: dict[str, Any]


@dataclass
class ProcessingState:
    """
    Processing state for batch operations.

    Used in action8_messaging.py for tracking processing progress.
    """
    batch_num: int
    progress_bar: Optional[Any] = None
    processed_in_loop: int = 0


@dataclass
class NavigationConfig:
    """
    Navigation configuration for browser navigation operations.

    Used in utils.py for navigate_to_page_with_retry.
    """
    url: str
    selector: str
    target_url_base: str
    signin_page_url_base: str
    unavailability_selectors: dict[str, tuple]
    page_timeout: int
    element_timeout: int


@dataclass
class ProgressIndicatorConfig:
    """
    Configuration for progress indicators.

    Used in core/progress_indicators.py for ProgressIndicator.
    """
    unit: str = "items"
    show_memory: bool = True
    show_rate: bool = True
    update_interval: float = 3.0
    show_bar: bool = True
    log_start: bool = True
    log_finish: bool = True
    leave: bool = True


@dataclass
class PrefetchedData:
    """
    Prefetched API data for person operations.

    Used in action6_gather.py for _prepare_person_operation_data.
    """
    combined_details: Optional[dict[str, Any]] = None
    tree_data: Optional[dict[str, Any]] = None


@dataclass
class RelationshipCalcContext:
    """
    Context for relationship calculation checks.

    Used in action11.py for _log_relationship_calculation_checks.
    """
    can_attempt_calculation: bool
    is_owner: bool
    can_calc_tree_ladder: bool
    can_calc_discovery_api: bool
    owner_tree_id_str: Optional[str]
    selected_tree_id_str: Optional[str]
    owner_profile_id_str: Optional[str]
    selected_global_id_str: Optional[str]
    selected_person_tree_id: Optional[str]
    source_of_ids: str


@dataclass
class MessageContext:
    """
    Message content and metadata for messaging operations.

    Used in action8_messaging.py for message preparation and sending.
    """
    person: Any  # Person object
    message_text: str
    message_to_send_key: str
    template_selection_reason: str
    log_prefix: str


@dataclass
class ConversationState:
    """
    Conversation state tracking for messaging operations.

    Used in action8_messaging.py for conversation management.
    """
    existing_conversation_id: Optional[str] = None
    effective_conv_id: Optional[str] = None
    latest_out_log: Optional[Any] = None  # ConversationLog object
    latest_in_log: Optional[Any] = None  # ConversationLog object


@dataclass
class MessageFlags:
    """
    Message operation flags and status.

    Used in action8_messaging.py for message sending control.
    """
    send_message_flag: bool
    skip_log_reason: str = ""
    message_status: str = ""


@dataclass
class ExtractionExperimentEvent:
    """
    Telemetry event data for extraction experiments.

    Used in prompt_telemetry.py for record_extraction_experiment_event.
    """
    variant_label: str
    prompt_key: str
    parse_success: bool
    prompt_version: Optional[str] = None
    extracted_data: Optional[dict[str, Any]] = None
    suggested_tasks: Optional[Any] = None
    raw_response_text: Optional[str] = None
    user_id: Optional[str] = None
    error: Optional[str] = None
    quality_score: Optional[float] = None
    component_coverage: Optional[float] = None
    anomaly_summary: Optional[str] = None


@dataclass
class SearchCriteria:
    """
    Search criteria for person/match searches.

    Used in api_search_utils.py and action10.py.
    """
    search_name: str
    field_name: Optional[str] = None
    test_name: Optional[str] = None
    candidate_data: Optional[dict[str, Any]] = None
    name_flex: Optional[str] = None


@dataclass
class RequestConfig:
    """
    HTTP request configuration parameters.

    Used in utils.py for _api_req and related functions.
    """
    url: str
    method: str = "GET"
    headers: Optional[dict[str, str]] = None
    referer_url: Optional[str] = None
    use_csrf_token: bool = False
    add_default_origin: bool = False
    timeout: Optional[int] = None
    allow_redirects: bool = True
    data: Optional[dict[str, Any]] = None
    json_data: Optional[dict[str, Any]] = None
    json: Optional[dict[str, Any]] = None
    force_text_response: bool = False
    cookie_jar: Optional[Any] = None


# ==============================================
# Comprehensive Test Suite
# ==============================================

def _test_graph_context_initialization() -> bool:
    """Test GraphContext initialization."""
    id_to_parents = {"person1": ["parent1", "parent2"]}
    id_to_children = {"parent1": ["person1", "person2"]}

    ctx = GraphContext(
        id_to_parents=id_to_parents,
        id_to_children=id_to_children,
        current_id="person1"
    )

    assert ctx.current_id == "person1", "Should store current_id"
    assert ctx.id_to_parents == id_to_parents, "Should store id_to_parents"
    assert ctx.id_to_children == id_to_children, "Should store id_to_children"
    return True


def _test_retry_context_initialization() -> bool:
    """Test RetryContext initialization."""
    ctx = RetryContext(
        attempt=1,
        max_attempts=3,
        max_delay=10.0,
        backoff_factor=2.0
    )

    assert ctx.attempt == 1, "Should store attempt"
    assert ctx.max_attempts == 3, "Should store max_attempts"
    assert ctx.max_delay == 10.0, "Should store max_delay"
    assert ctx.backoff_factor == 2.0, "Should store backoff_factor"
    return True


def _test_match_identifiers_initialization() -> bool:
    """Test MatchIdentifiers initialization."""
    identifiers = MatchIdentifiers(
        uuid="uuid-123",
        username="testuser",
        in_my_tree=True,
        log_ref_short="REF001",
        profile_id="12345"
    )

    assert identifiers.uuid == "uuid-123", "Should store uuid"
    assert identifiers.username == "testuser", "Should store username"
    assert identifiers.in_my_tree is True, "Should store in_my_tree"
    assert identifiers.profile_id == "12345", "Should store profile_id"
    return True


def _test_progress_indicator_config_initialization() -> bool:
    """Test ProgressIndicatorConfig initialization."""
    config = ProgressIndicatorConfig(
        unit="items",
        show_memory=True,
        show_rate=True
    )

    assert config.unit == "items", "Should store unit"
    assert config.show_memory is True, "Should store show_memory"
    assert config.show_rate is True, "Should store show_rate"
    return True


def _test_search_criteria_initialization() -> bool:
    """Test SearchCriteria initialization."""
    criteria = SearchCriteria(
        search_name="John Doe",
        field_name="name"
    )

    assert criteria.search_name == "John Doe", "Should store search_name"
    assert criteria.field_name == "name", "Should store field_name"
    return True


def _test_request_config_initialization() -> bool:
    """Test RequestConfig initialization."""
    config = RequestConfig(
        url="https://example.com/api",
        method="POST",
        use_csrf_token=True
    )

    assert config.url == "https://example.com/api", "Should store url"
    assert config.method == "POST", "Should store method"
    assert config.use_csrf_token is True, "Should store use_csrf_token"
    return True


def _test_dataclass_defaults() -> bool:
    """Test that dataclass defaults work correctly."""
    # RetryContext defaults
    ctx = RetryContext(attempt=1, max_attempts=3, max_delay=10.0)
    assert ctx.backoff_factor == 2.0, "Should have default backoff_factor"
    assert ctx.current_delay == 1.0, "Should have default current_delay"

    # RequestConfig defaults
    config = RequestConfig(url="https://example.com")
    assert config.method == "GET", "Should have default method"
    assert config.allow_redirects is True, "Should have default allow_redirects"

    return True


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for common_params.py.
    Tests parameter dataclass initialization and defaults.
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "Common Parameters & Dataclass Definitions",
            "common_params.py"
        )
        suite.start_suite()

        suite.run_test(
            "GraphContext Initialization",
            _test_graph_context_initialization,
            "GraphContext initializes with correct graph data",
            "Test GraphContext creation with parent/child mappings",
            "Test genealogical graph context setup",
        )

        suite.run_test(
            "RetryContext Initialization",
            _test_retry_context_initialization,
            "RetryContext initializes with retry parameters",
            "Test RetryContext creation with retry settings",
            "Test retry logic parameter setup",
        )

        suite.run_test(
            "MatchIdentifiers Initialization",
            _test_match_identifiers_initialization,
            "MatchIdentifiers initializes with DNA match data",
            "Test MatchIdentifiers creation with match data",
            "Test DNA match identifier setup",
        )

        suite.run_test(
            "ProgressIndicatorConfig Initialization",
            _test_progress_indicator_config_initialization,
            "ProgressIndicatorConfig initializes with display options",
            "Test ProgressIndicatorConfig creation",
            "Test progress indicator configuration",
        )

        suite.run_test(
            "SearchCriteria Initialization",
            _test_search_criteria_initialization,
            "SearchCriteria initializes with search parameters",
            "Test SearchCriteria creation with search data",
            "Test search criteria setup",
        )

        suite.run_test(
            "RequestConfig Initialization",
            _test_request_config_initialization,
            "RequestConfig initializes with HTTP request parameters",
            "Test RequestConfig creation with request settings",
            "Test HTTP request configuration",
        )

        suite.run_test(
            "Dataclass Defaults",
            _test_dataclass_defaults,
            "Dataclass defaults are correctly applied",
            "Test default values for dataclass fields",
            "Test parameter default values",
        )

        return suite.finish_suite()


if __name__ == "__main__":
    import traceback

    try:
        print("🧪 Running Common Parameters & Dataclass Definitions comprehensive test suite...")
        success = run_comprehensive_tests()
    except Exception:
        print("\n[ERROR] Unhandled exception during Common Parameters tests:", file=sys.stderr)
        traceback.print_exc()
        success = False

    sys.exit(0 if success else 1)


# End of common_params.py

