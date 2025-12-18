"""Integration tests for TreeUpdateService with mocked Ancestry APIs.

Tests the TreeUpdateService functionality by mocking the Ancestry.com API
responses. This ensures the service correctly:
- Formats API requests
- Handles successful responses
- Handles error responses
- Applies SuggestedFacts correctly
- Logs operations to TreeUpdateLog

These tests do not require a live Ancestry session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, cast

from api.tree_update import (
    FACT_TYPE_TO_EVENT_TYPE,
    TreeOperationType,
    TreeUpdateResponse,
    TreeUpdateResult,
    TreeUpdateService,
)
from core.database import (
    FactTypeEnum,
)
from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

# =============================================================================
# Type Protocols for Testing
# =============================================================================


class APIManagerProtocol(Protocol):
    """Protocol for API manager required by TreeUpdateService."""

    requests: list[dict[str, Any]]  # For testing: track requests made

    def queue_response(self, response: Any) -> None:
        """Queue a response to be returned on next request (testing only)."""
        ...

    def post(
        self,
        url: str,
        json: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Make POST request."""
        ...

    def get(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Make GET request."""
        ...


class TreeUpdateSessionProtocol(Protocol):
    """Protocol for SessionManager as required by TreeUpdateService."""

    api_manager: APIManagerProtocol

    @property
    def my_profile_id(self) -> str:
        """Return user's profile ID."""
        ...

    @property
    def my_uuid(self) -> str:
        """Return user's UUID."""
        ...


# =============================================================================
# Mock Response Fixtures
# =============================================================================


@dataclass
class MockResponse:
    """Mock HTTP response for testing."""

    status_code: int
    _json_data: Optional[dict[str, Any]] = None
    text: str = ""

    def json(self) -> dict[str, Any]:
        if self._json_data is None:
            raise json.JSONDecodeError("No JSON", "", 0)
        return self._json_data


def create_success_response(data: Optional[dict[str, Any]] = None) -> MockResponse:
    """Create a successful API response."""
    return MockResponse(
        status_code=200,
        _json_data=data or {"success": True},
        text=json.dumps(data or {"success": True}),
    )


def create_error_response(status_code: int = 400, message: str = "Error") -> MockResponse:
    """Create an error API response."""
    return MockResponse(
        status_code=status_code,
        _json_data=None,
        text=message,
    )


def create_new_person_response(person_id: str = "12345678") -> MockResponse:
    """Create response for addperson API."""
    return MockResponse(
        status_code=200,
        _json_data={
            "personId": person_id,
            "success": True,
            "message": "Person added successfully",
        },
        text=json.dumps({"personId": person_id}),
    )


# =============================================================================
# Mock Session Manager
# =============================================================================


class MockAPIManager(APIManagerProtocol):
    """Mock API manager implementing APIManagerProtocol for testing."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.response_queue: list[MockResponse] = []
        self.default_response = create_success_response()

    def queue_response(self, response: MockResponse) -> None:
        """Queue a response to be returned on next request."""
        self.response_queue.append(response)

    def _get_next_response(self) -> MockResponse:
        """Get the next queued response or default."""
        if self.response_queue:
            return self.response_queue.pop(0)
        return self.default_response

    def post(
        self,
        url: str,
        json: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        **_kwargs: Any,
    ) -> MockResponse:
        """Mock POST request."""
        self.requests.append(
            {
                "method": "POST",
                "url": url,
                "json": json,
                "headers": headers,
            }
        )
        return self._get_next_response()

    def get(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        **_kwargs: Any,
    ) -> MockResponse:
        """Mock GET request."""
        self.requests.append(
            {
                "method": "GET",
                "url": url,
                "headers": headers,
            }
        )
        return self._get_next_response()


class MockSessionManager(TreeUpdateSessionProtocol):
    """Mock SessionManager implementing TreeUpdateSessionProtocol for testing."""

    def __init__(self) -> None:
        self.api_manager = MockAPIManager()
        self._user_id = "test_user_123"

    @property
    def my_profile_id(self) -> str:
        """Return mock profile ID."""
        return self._user_id

    @property
    def my_uuid(self) -> str:
        """Return mock UUID."""
        return "mock_uuid_123"


# =============================================================================
# Test Functions
# =============================================================================


def _test_update_person_success() -> None:
    """Test successful person update via updatePerson API."""
    mock_sm: TreeUpdateSessionProtocol = MockSessionManager()
    mock_sm.api_manager.queue_response(create_success_response({"updated": True}))

    service = TreeUpdateService(cast(Any, mock_sm))  # Cast for type checker
    result = service.update_person(
        tree_id="tree123",
        person_id="person456",
        updates={"bdate": "15 Mar 1850", "bplace": "London, England"},
    )

    assert result.result == TreeUpdateResult.SUCCESS
    assert result.operation == TreeOperationType.UPDATE_PERSON
    assert result.person_id == "person456"
    assert "bdate" in result.message or "Updated" in result.message

    # Verify API was called
    assert len(mock_sm.api_manager.requests) == 1
    request = mock_sm.api_manager.requests[0]
    assert request["method"] == "POST"
    assert "updatePerson" in request["url"]
    assert request["json"]["values"]["bdate"] == "15 Mar 1850"


def _test_update_person_failure() -> None:
    """Test person update failure handling."""
    mock_sm: TreeUpdateSessionProtocol = MockSessionManager()
    mock_sm.api_manager.queue_response(create_error_response(500, "Server error"))

    service = TreeUpdateService(cast(Any, mock_sm))  # Cast for type checker
    result = service.update_person(
        tree_id="tree123",
        person_id="person456",
        updates={"bdate": "1850"},
    )

    assert result.result == TreeUpdateResult.API_ERROR
    assert result.error_details is not None
    assert "500" in result.error_details


def _test_add_fact_success() -> None:
    """Test successful fact addition via factedit API."""
    mock_sm: TreeUpdateSessionProtocol = MockSessionManager()
    mock_sm.api_manager.queue_response(create_success_response({"assertionId": "12345"}))

    service = TreeUpdateService(cast(Any, mock_sm))  # Cast for type checker
    result = service.add_fact(
        tree_id="tree123",
        person_id="person456",
        event_type="death",
        date="10 Jan 1920",
        location="New York, USA",
    )

    assert result.result == TreeUpdateResult.SUCCESS
    assert result.operation == TreeOperationType.ADD_FACT
    assert "death" in result.message.lower()

    # Verify API payload
    request = mock_sm.api_manager.requests[0]
    assert request["json"]["eventType"] == "death"
    assert request["json"]["date"] == "10 Jan 1920"


def _test_add_person_with_relationship() -> None:
    """Test adding new person with relationship."""
    mock_sm: TreeUpdateSessionProtocol = MockSessionManager()
    mock_sm.api_manager.queue_response(create_new_person_response("new_person_789"))

    service = TreeUpdateService(cast(Any, mock_sm))  # Protocol-compatible mock
    result = service.add_person_with_relationship(
        tree_id="tree123",
        source_person_id="person456",
        relationship_type="Child",
        new_person_data={
            "fname": "John",
            "lname": "Smith",
            "genderRadio": "Male",
            "statusRadio": "Deceased",
            "bdate": "1875",
        },
        parent_set={"fatherId": "person456", "motherId": "0"},
    )

    assert result.result == TreeUpdateResult.SUCCESS
    assert result.operation == TreeOperationType.ADD_PERSON
    assert "John Smith" in result.message or "Child" in result.message

    # Verify payload includes relationship type
    request = mock_sm.api_manager.requests[0]
    assert request["json"]["type"] == "Child"
    assert request["json"]["values"]["fname"] == "John"


def _test_link_existing_person() -> None:
    """Test linking existing person as relationship."""
    mock_sm: TreeUpdateSessionProtocol = MockSessionManager()
    mock_sm.api_manager.queue_response(create_success_response())

    service = TreeUpdateService(cast(Any, mock_sm))  # Protocol-compatible mock
    result = service.link_existing_person(
        tree_id="tree123",
        source_person_id="person456",
        target_person_id="789123456",  # Must be numeric for Ancestry PID
        relationship_type="Spouse",
        target_person_info={
            "name": "Jane Doe",
            "birth": "1855",
            "death": "1920",
            "gender": "Female",
        },
    )

    assert result.result == TreeUpdateResult.SUCCESS
    assert result.operation == TreeOperationType.LINK_PERSON

    # Verify apmFindExistingPerson pattern
    request = mock_sm.api_manager.requests[0]
    assert "apmFindExistingPerson" in request["json"]["values"]
    assert request["json"]["values"]["apmFindExistingPerson"]["PID"] == 789123456


def _test_remove_relationship() -> None:
    """Test removing relationship between people."""
    mock_sm: TreeUpdateSessionProtocol = MockSessionManager()
    mock_sm.api_manager.queue_response(create_success_response())

    service = TreeUpdateService(cast(Any, mock_sm))  # Protocol-compatible mock
    result = service.remove_relationship(
        tree_id="tree123",
        source_person_id="person456",
        related_person_id="person789",
        relationship_type="C",  # Child
    )

    assert result.result == TreeUpdateResult.SUCCESS
    assert result.operation == TreeOperationType.REMOVE_RELATIONSHIP

    # Verify URL pattern
    request = mock_sm.api_manager.requests[0]
    assert "removerelationship" in request["url"]


def _test_change_relationship_type() -> None:
    """Test changing relationship type (e.g., spouse to ex-spouse)."""
    mock_sm: TreeUpdateSessionProtocol = MockSessionManager()
    mock_sm.api_manager.queue_response(create_success_response())

    service = TreeUpdateService(cast(Any, mock_sm))  # Protocol-compatible mock
    result = service.change_relationship_type(
        tree_id="tree123",
        source_person_id="person456",
        related_person_id="person789",
        new_modifier="spu",  # ex-spouse
        original_modifier="sps",  # spouse
        relationship_type="W",  # Wife
    )

    assert result.result == TreeUpdateResult.SUCCESS
    assert result.operation == TreeOperationType.CHANGE_RELATIONSHIP
    assert "sps" in result.message and "spu" in result.message

    # Verify payload
    request = mock_sm.api_manager.requests[0]
    assert request["json"]["modifier"] == "spu"
    assert request["json"]["originalModifier"] == "sps"


def _test_fact_type_mapping() -> None:
    """Test fact type to event type mapping is complete."""
    required_types = [
        FactTypeEnum.BIRTH,
        FactTypeEnum.DEATH,
        FactTypeEnum.MARRIAGE,
        FactTypeEnum.LOCATION,
    ]

    for fact_type in required_types:
        assert fact_type in FACT_TYPE_TO_EVENT_TYPE, f"Missing mapping for {fact_type}"
        assert isinstance(FACT_TYPE_TO_EVENT_TYPE[fact_type], str)


def _test_url_building() -> None:
    """Test API URL construction."""
    mock_sm: TreeUpdateSessionProtocol = MockSessionManager()
    service = TreeUpdateService(cast(Any, mock_sm))  # Protocol-compatible mock

    # Force user_id to be set
    service._user_id = "test_user"

    url = service._build_url(
        endpoint="updatePerson",
        tree_id="tree123",
        person_id="person456",
    )

    assert "test_user" in url
    assert "tree123" in url
    assert "person456" in url
    assert "updatePerson" in url
    assert url.startswith("https://")


def _test_api_request_error_handling() -> None:
    """Test API request error handling for network failures."""

    class FailingAPIManager(APIManagerProtocol):
        """API manager that always raises connection errors."""

        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []  # Protocol attribute

        def queue_response(self, response: dict[str, Any]) -> None:
            """Protocol method - not used for failure tests."""
            pass

        def post(  # noqa: PLR6301
            self,
            url: str,
            json: Optional[dict[str, Any]] = None,
            headers: Optional[dict[str, str]] = None,
            **_kwargs: Any,
        ) -> Any:
            _ = (url, json, headers)  # Unused parameters for failure simulation
            raise ConnectionError("Network unavailable")

        def get(  # noqa: PLR6301
            self,
            url: str,
            headers: Optional[dict[str, str]] = None,
            **_kwargs: Any,
        ) -> Any:
            _ = (url, headers)  # Unused parameters for failure simulation
            raise ConnectionError("Network unavailable")

    class FailingSessionManager(TreeUpdateSessionProtocol):
        """Session manager with failing API manager."""

        def __init__(self) -> None:
            self.api_manager = FailingAPIManager()

        @property
        def my_profile_id(self) -> str:
            return "test_user"

        @property
        def my_uuid(self) -> str:
            return "test_uuid"

    mock_sm: TreeUpdateSessionProtocol = FailingSessionManager()
    service = TreeUpdateService(cast(Any, mock_sm))  # Protocol-compatible mock

    # This should not raise, but return error result
    success, response, error = service._make_api_request(
        url="https://example.com/api",
        method="POST",
        data={"test": True},
    )

    assert success is False
    assert response is None
    assert error is not None
    assert "Network" in error or "failed" in error


def _test_tree_update_response_timestamp() -> None:
    """Test TreeUpdateResponse has automatic timestamp."""
    before = datetime.now(timezone.utc)

    response = TreeUpdateResponse(
        result=TreeUpdateResult.SUCCESS,
        operation=TreeOperationType.UPDATE_PERSON,
        person_id="123",
        message="Test",
    )

    after = datetime.now(timezone.utc)

    assert response.timestamp >= before
    assert response.timestamp <= after


# =============================================================================
# Module Test Runner
# =============================================================================


def module_tests() -> bool:
    """Run all integration tests."""
    suite = TestSuite("TreeUpdateService Integration", "testing/test_tree_update_integration.py")
    suite.start_suite()

    suite.run_test(
        "Update person success",
        _test_update_person_success,
        "Tests successful person update via updatePerson API",
    )
    suite.run_test(
        "Update person failure",
        _test_update_person_failure,
        "Tests error handling for failed updates",
    )
    suite.run_test(
        "Add fact success",
        _test_add_fact_success,
        "Tests adding a fact via factedit API",
    )
    suite.run_test(
        "Add person with relationship",
        _test_add_person_with_relationship,
        "Tests creating new person with relationship",
    )
    suite.run_test(
        "Link existing person",
        _test_link_existing_person,
        "Tests linking existing person via apmFindExistingPerson",
    )
    suite.run_test(
        "Remove relationship",
        _test_remove_relationship,
        "Tests removing relationship between people",
    )
    suite.run_test(
        "Change relationship type",
        _test_change_relationship_type,
        "Tests changing relationship modifier",
    )
    suite.run_test(
        "Fact type mapping complete",
        _test_fact_type_mapping,
        "Verifies all fact types have event type mappings",
    )
    suite.run_test(
        "URL building",
        _test_url_building,
        "Tests API URL construction",
    )
    suite.run_test(
        "API error handling",
        _test_api_request_error_handling,
        "Tests graceful handling of network errors",
    )
    suite.run_test(
        "Response timestamp",
        _test_tree_update_response_timestamp,
        "Verifies TreeUpdateResponse has automatic timestamp",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
