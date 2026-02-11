"""
Tree Update Service for Ancestry.com Family Tree Modifications.

This module provides programmatic access to Ancestry's internal APIs
for updating family tree data. It is the core implementation for
Phase 8: Tree Update Automation.

Features:
- Apply APPROVED SuggestedFacts to the tree
- Add/update person facts (birth, death, etc.)
- Manage relationships (add, remove, change type)
- Audit logging for all modifications
- Post-update verification

Dependencies:
- SessionManager: For authenticated API requests
- RateLimiter: Respect Ancestry rate limits
- Database: SuggestedFact, TreeUpdateLog models

See Also:
- docs/specs/ancestry_tree_api.md: Complete API documentation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from core.database import (
    FactStatusEnum,
    FactTypeEnum,
    Person,
    SuggestedFact,
)
from testing.test_framework import TestSuite, create_standard_test_runner

if TYPE_CHECKING:
    from core.session_manager import SessionManager

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# API Base Path Pattern
API_BASE = "/family-tree/person"


class TreeUpdateResult(Enum):
    """Result status for tree update operations."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    ALREADY_APPLIED = "ALREADY_APPLIED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    API_ERROR = "API_ERROR"
    SESSION_ERROR = "SESSION_ERROR"


class TreeOperationType(Enum):
    """Types of tree modification operations."""

    UPDATE_PERSON = "UPDATE_PERSON"
    ADD_FACT = "ADD_FACT"
    ADD_PERSON = "ADD_PERSON"
    LINK_PERSON = "LINK_PERSON"
    REMOVE_PERSON = "REMOVE_PERSON"
    ADD_RELATIONSHIP = "ADD_RELATIONSHIP"
    REMOVE_RELATIONSHIP = "REMOVE_RELATIONSHIP"
    CHANGE_RELATIONSHIP = "CHANGE_RELATIONSHIP"
    ADD_WEB_LINK = "ADD_WEB_LINK"


# Mapping from FactTypeEnum to Ancestry eventType values
FACT_TYPE_TO_EVENT_TYPE: dict[FactTypeEnum, str] = {
    FactTypeEnum.BIRTH: "birth",
    FactTypeEnum.DEATH: "death",
    FactTypeEnum.MARRIAGE: "marriage",
    FactTypeEnum.LOCATION: "residence",
    FactTypeEnum.OTHER: "other",
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TreeUpdateRequest:
    """Request to update a tree entity."""

    operation: TreeOperationType
    tree_id: str
    person_id: str
    fact_type: FactTypeEnum | None = None
    suggested_fact_id: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeUpdateResponse:
    """Response from a tree update operation."""

    result: TreeUpdateResult
    operation: TreeOperationType
    person_id: str
    message: str
    api_response: dict[str, Any] | None = None
    error_details: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


# =============================================================================
# Tree Update Service
# =============================================================================


class TreeUpdateService:
    """
    Service for applying updates to Ancestry.com family trees.

    This service handles all tree modification operations including:
    - Updating person facts (birth, death, name, etc.)
    - Adding new facts
    - Managing relationships
    - Linking existing persons

    All operations are authenticated through SessionManager and
    respect rate limiting constraints.
    """

    def __init__(self, session_manager: SessionManager) -> None:
        """
        Initialize the TreeUpdateService.

        Args:
            session_manager: Active SessionManager with authenticated session
        """
        self.session_manager = session_manager
        self._user_id: str | None = None
        self._default_tree_id: str | None = None

    @property
    def user_id(self) -> str:
        """Get the authenticated user ID."""
        if self._user_id is None:
            # Extract from session cookies or config
            self._user_id = self._get_user_id_from_session()
        return self._user_id

    def _get_user_id_from_session(self) -> str:
        """Extract user ID from active session."""
        # Try to get from session manager profile ID
        try:
            if hasattr(self.session_manager, "my_profile_id") and self.session_manager.my_profile_id:
                return self.session_manager.my_profile_id
            if hasattr(self.session_manager, "my_uuid") and self.session_manager.my_uuid:
                return self.session_manager.my_uuid
        except Exception:
            pass

        # Fallback to config
        from config import config_schema

        return getattr(config_schema, "ancestry_user_id", "") or ""

    def _build_url(
        self,
        endpoint: str,
        tree_id: str,
        person_id: str,
        include_user: bool = True,
    ) -> str:
        """
        Build the full API URL for an endpoint.

        Args:
            endpoint: The endpoint path (e.g., "updatePerson")
            tree_id: The tree ID
            person_id: The person ID
            include_user: Whether to include user ID in path

        Returns:
            Full URL string
        """
        base = "https://www.ancestry.co.uk"

        if include_user:
            path = f"{API_BASE}/addedit/user/{self.user_id}/tree/{tree_id}/person/{person_id}/{endpoint}"
        else:
            path = f"{API_BASE}/tree/{tree_id}/person/{person_id}/{endpoint}"

        return f"{base}{path}"

    def _make_api_request(
        self,
        url: str,
        method: str = "POST",
        data: dict[str, Any] | None = None,
        content_type: str = "application/json",
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
        """
        Make an authenticated API request.

        Args:
            url: Full API URL
            method: HTTP method (GET, POST)
            data: Request body data
            content_type: Content-Type header

        Returns:
            Tuple of (success, response_data, error_message)
        """
        try:
            api_manager = self.session_manager.api_manager

            headers = {
                "Accept": "application/json",
                "Content-Type": content_type,
            }

            if method == "POST":
                response = api_manager.post(
                    url,
                    json=data,
                    headers=headers,
                )
            else:
                response = api_manager.get(url, headers=headers)

            if response.status_code in {200, 201}:
                try:
                    return True, response.json(), None
                except json.JSONDecodeError:
                    return True, {"raw": response.text}, None
            else:
                error_msg = f"API returned {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                return False, None, error_msg

        except Exception as e:
            error_msg = f"API request failed: {e}"
            logger.exception(error_msg)
            return False, None, error_msg

    # -------------------------------------------------------------------------
    # Core Update Operations
    # -------------------------------------------------------------------------

    def update_person(
        self,
        tree_id: str,
        person_id: str,
        updates: dict[str, str],
        gender: str = "Unknown",
    ) -> TreeUpdateResponse:
        """
        Update core person facts using the quick edit API.

        Supports: fname, mname, lname, sufname, bdate, bplace, ddate, dplace,
                  genderRadio, statusRadio

        Args:
            tree_id: Tree ID
            person_id: Person ID
            updates: Dictionary of field -> value updates
            gender: Current gender of the person

        Returns:
            TreeUpdateResponse with result details
        """
        url = self._build_url("updatePerson", tree_id, person_id)

        payload = {
            "person": {
                "personId": person_id,
                "treeId": tree_id,
                "userId": self.user_id,
                "gender": gender,
            },
            "values": updates,
        }

        logger.info(f"Updating person {person_id}: {list(updates.keys())}")

        success, response, error = self._make_api_request(url, "POST", payload)

        if success:
            return TreeUpdateResponse(
                result=TreeUpdateResult.SUCCESS,
                operation=TreeOperationType.UPDATE_PERSON,
                person_id=person_id,
                message=f"Updated fields: {', '.join(updates.keys())}",
                api_response=response,
            )
        return TreeUpdateResponse(
            result=TreeUpdateResult.API_ERROR,
            operation=TreeOperationType.UPDATE_PERSON,
            person_id=person_id,
            message="Failed to update person",
            error_details=error,
        )

    def add_fact(
        self,
        tree_id: str,
        person_id: str,
        event_type: str,
        date: str = "",
        location: str = "",
        description: str = "",
    ) -> TreeUpdateResponse:
        """
        Add a fact to a person using the factedit API.

        Supports any event type: birth, death, marriage, residence,
        occupation, military, immigration, etc.

        Args:
            tree_id: Tree ID
            person_id: Person ID
            event_type: Ancestry event type (e.g., "death", "residence")
            date: Date string (free text format)
            location: Location string
            description: Additional description

        Returns:
            TreeUpdateResponse with result details
        """
        url = self._build_url("factedit", tree_id, person_id)
        url = f"{url}/assertion/0/save"

        payload = {
            "assertionId": "0",  # 0 = new fact
            "eventType": event_type,
            "date": date,
            "location": location,
            "description": description,
            "sourceInfo": None,
        }

        logger.info(f"Adding {event_type} fact to person {person_id}")

        success, response, error = self._make_api_request(url, "POST", payload)

        if success:
            return TreeUpdateResponse(
                result=TreeUpdateResult.SUCCESS,
                operation=TreeOperationType.ADD_FACT,
                person_id=person_id,
                message=f"Added {event_type} fact",
                api_response=response,
            )
        return TreeUpdateResponse(
            result=TreeUpdateResult.API_ERROR,
            operation=TreeOperationType.ADD_FACT,
            person_id=person_id,
            message=f"Failed to add {event_type} fact",
            error_details=error,
        )

    def add_person_with_relationship(
        self,
        tree_id: str,
        source_person_id: str,
        relationship_type: str,
        new_person_data: dict[str, str],
        parent_set: dict[str, str] | None = None,
        source_gender: str = "Unknown",
    ) -> TreeUpdateResponse:
        """
        Add a new person with a relationship to an existing person.

        Args:
            tree_id: Tree ID
            source_person_id: The existing person to relate to
            relationship_type: Type (Spouse, Child, Father, Mother, Sibling)
            new_person_data: Dict with fname, lname, genderRadio, statusRadio, etc.
            parent_set: For Child type, dict with fatherId, motherId
            source_gender: Gender of the source person

        Returns:
            TreeUpdateResponse with new person ID if successful
        """
        url = self._build_url("addperson", tree_id, source_person_id)

        values: dict[str, Any] = {
            "fname": new_person_data.get("fname", ""),
            "lname": new_person_data.get("lname", ""),
            "sufname": new_person_data.get("sufname", ""),
            "genderRadio": new_person_data.get("genderRadio", "Person"),
            "statusRadio": new_person_data.get("statusRadio", "Living"),
            "bdate": new_person_data.get("bdate", ""),
            "bplace": new_person_data.get("bplace", ""),
            "isAlternateParent": False,
        }

        if relationship_type == "Spouse":
            values["spousalRelationship"] = "Spouse"
        elif relationship_type == "Child" and parent_set:
            values["parentSet"] = parent_set

        payload = {
            "addTarget": None,
            "person": {
                "personId": source_person_id,
                "treeId": tree_id,
                "userId": self.user_id,
                "gender": source_gender,
            },
            "type": relationship_type,
            "values": values,
        }

        logger.info(f"Adding {relationship_type} to person {source_person_id}")

        success, response, error = self._make_api_request(url, "POST", payload)

        if success:
            new_person_id = response.get("personId") if response else None
            return TreeUpdateResponse(
                result=TreeUpdateResult.SUCCESS,
                operation=TreeOperationType.ADD_PERSON,
                person_id=new_person_id or source_person_id,
                message=f"Added {relationship_type}: {new_person_data.get('fname', '')} {new_person_data.get('lname', '')}",
                api_response=response,
            )
        return TreeUpdateResponse(
            result=TreeUpdateResult.API_ERROR,
            operation=TreeOperationType.ADD_PERSON,
            person_id=source_person_id,
            message=f"Failed to add {relationship_type}",
            error_details=error,
        )

    def link_existing_person(
        self,
        tree_id: str,
        source_person_id: str,
        target_person_id: str,
        relationship_type: str,
        target_person_info: dict[str, str],
        parent_set: dict[str, Any] | None = None,
    ) -> TreeUpdateResponse:
        """
        Link an existing person in the tree as a relationship.

        This uses the apmFindExistingPerson pattern instead of creating
        a new person.

        Args:
            tree_id: Tree ID
            source_person_id: The person to link FROM
            target_person_id: The EXISTING person to link TO (PID)
            relationship_type: Type (Child, Spouse, etc.)
            target_person_info: Info about target (name, birth, death, gender)
            parent_set: For Child relationships, parent details

        Returns:
            TreeUpdateResponse with result details
        """
        url = self._build_url("addperson", tree_id, source_person_id)

        values: dict[str, Any] = {
            "apmFindExistingPerson": {
                "name": target_person_info.get("name", ""),
                "birth": target_person_info.get("birth", ""),
                "death": target_person_info.get("death", ""),
                "PID": int(target_person_id),
                "genderIconType": target_person_info.get("gender", "Person"),
            }
        }

        if parent_set:
            values["parentSet"] = parent_set

        payload = {
            "person": {
                "personId": source_person_id,
                "treeId": tree_id,
                "userId": self.user_id,
            },
            "type": relationship_type,
            "values": values,
        }

        logger.info(f"Linking existing person {target_person_id} as {relationship_type}")

        success, response, error = self._make_api_request(url, "POST", payload)

        if success:
            return TreeUpdateResponse(
                result=TreeUpdateResult.SUCCESS,
                operation=TreeOperationType.LINK_PERSON,
                person_id=target_person_id,
                message=f"Linked {target_person_info.get('name', '')} as {relationship_type}",
                api_response=response,
            )
        return TreeUpdateResponse(
            result=TreeUpdateResult.API_ERROR,
            operation=TreeOperationType.LINK_PERSON,
            person_id=target_person_id,
            message=f"Failed to link person as {relationship_type}",
            error_details=error,
        )

    def remove_relationship(
        self,
        tree_id: str,
        source_person_id: str,
        related_person_id: str,
        relationship_type: str,
        parent_type: str = "",
    ) -> TreeUpdateResponse:
        """
        Remove a relationship between two people (keeps both in tree).

        Args:
            tree_id: Tree ID
            source_person_id: The person from whose perspective to remove
            related_person_id: The related person ID
            relationship_type: Type code (C=Child, F=Father, M=Mother, H=Husband, W=Wife)
            parent_type: For child relationships, which parent (M/F)

        Returns:
            TreeUpdateResponse with result details
        """
        url = self._build_url("relationship", tree_id, source_person_id)
        url = f"{url}/{related_person_id}/removerelationship"

        payload = {
            "type": relationship_type,
            "parentType": parent_type,
        }

        logger.info(f"Removing relationship {relationship_type} between {source_person_id} and {related_person_id}")

        success, response, error = self._make_api_request(url, "POST", payload)

        if success:
            return TreeUpdateResponse(
                result=TreeUpdateResult.SUCCESS,
                operation=TreeOperationType.REMOVE_RELATIONSHIP,
                person_id=source_person_id,
                message=f"Removed relationship to {related_person_id}",
                api_response=response,
            )
        return TreeUpdateResponse(
            result=TreeUpdateResult.API_ERROR,
            operation=TreeOperationType.REMOVE_RELATIONSHIP,
            person_id=source_person_id,
            message="Failed to remove relationship",
            error_details=error,
        )

    def change_relationship_type(
        self,
        tree_id: str,
        source_person_id: str,
        related_person_id: str,
        new_modifier: str,
        original_modifier: str,
        relationship_type: str,
        parent_type: str = "",
    ) -> TreeUpdateResponse:
        """
        Change the type of an existing relationship.

        Modifier codes for spouses: sps (spouse), spu (ex-spouse), spp (partner)
        Modifier codes for parents: bio, adp (adoptive), fos (foster), stp (step)

        Args:
            tree_id: Tree ID
            source_person_id: The person from whose perspective
            related_person_id: The related person ID
            new_modifier: New relationship modifier (e.g., "spu" for ex-spouse)
            original_modifier: Current modifier (e.g., "sps" for spouse)
            relationship_type: Type code (H, W, F, M, etc.)
            parent_type: Parent type if applicable

        Returns:
            TreeUpdateResponse with result details
        """
        url = self._build_url("relationship", tree_id, source_person_id)
        url = f"{url}/{related_person_id}/changerelationship"

        payload = {
            "modifier": new_modifier,
            "originalModifier": original_modifier,
            "type": relationship_type,
            "parentType": parent_type,
            "pty": -1,
        }

        logger.info(f"Changing relationship type from {original_modifier} to {new_modifier}")

        success, response, error = self._make_api_request(url, "POST", payload)

        if success:
            return TreeUpdateResponse(
                result=TreeUpdateResult.SUCCESS,
                operation=TreeOperationType.CHANGE_RELATIONSHIP,
                person_id=source_person_id,
                message=f"Changed relationship: {original_modifier} → {new_modifier}",
                api_response=response,
            )
        return TreeUpdateResponse(
            result=TreeUpdateResult.API_ERROR,
            operation=TreeOperationType.CHANGE_RELATIONSHIP,
            person_id=source_person_id,
            message="Failed to change relationship type",
            error_details=error,
        )

    # -------------------------------------------------------------------------
    # SuggestedFact Integration
    # -------------------------------------------------------------------------

    def apply_suggested_fact(  # noqa: PLR0911
        self,
        db_session: Session,
        suggested_fact: SuggestedFact,
        tree_id: str,
    ) -> TreeUpdateResponse:
        """
        Apply an APPROVED SuggestedFact to the Ancestry tree.

        This is the main entry point for Phase 8 automation.

        Args:
            db_session: Database session
            suggested_fact: The SuggestedFact to apply
            tree_id: Target tree ID

        Returns:
            TreeUpdateResponse with result details
        """
        # Validate status
        if suggested_fact.status != FactStatusEnum.APPROVED:
            return TreeUpdateResponse(
                result=TreeUpdateResult.VALIDATION_ERROR,
                operation=TreeOperationType.UPDATE_PERSON,
                person_id=str(suggested_fact.people_id),
                message=f"SuggestedFact status is {suggested_fact.status.value}, expected APPROVED",
            )

        # Get person's Ancestry person_id
        person = db_session.query(Person).filter(Person.id == suggested_fact.people_id).first()
        if not person:
            return TreeUpdateResponse(
                result=TreeUpdateResult.VALIDATION_ERROR,
                operation=TreeOperationType.UPDATE_PERSON,
                person_id=str(suggested_fact.people_id),
                message="Person not found in database",
            )

        # Get the Ancestry person ID (profile_id or from tree)
        ancestry_person_id = person.profile_id
        if not ancestry_person_id:
            return TreeUpdateResponse(
                result=TreeUpdateResult.VALIDATION_ERROR,
                operation=TreeOperationType.UPDATE_PERSON,
                person_id=str(suggested_fact.people_id),
                message="Person has no Ancestry profile_id",
            )

        # Determine the update type based on fact_type
        fact_type = suggested_fact.fact_type
        new_value = suggested_fact.new_value

        # Route to appropriate API based on fact type
        if fact_type in {FactTypeEnum.BIRTH, FactTypeEnum.DEATH}:
            return self._apply_vital_fact(ancestry_person_id, tree_id, fact_type, new_value)
        if fact_type == FactTypeEnum.RELATIONSHIP:
            logger.info(f"Relationship facts require manual review: {new_value}")
            return TreeUpdateResponse(
                result=TreeUpdateResult.VALIDATION_ERROR,
                operation=TreeOperationType.ADD_RELATIONSHIP,
                person_id=ancestry_person_id,
                message="Relationship facts require manual configuration",
            )
        if fact_type in {FactTypeEnum.MARRIAGE, FactTypeEnum.LOCATION, FactTypeEnum.OTHER}:
            event_type = FACT_TYPE_TO_EVENT_TYPE.get(fact_type, "other")
            return self._apply_general_fact(ancestry_person_id, tree_id, event_type, new_value)
        return TreeUpdateResponse(
            result=TreeUpdateResult.VALIDATION_ERROR,
            operation=TreeOperationType.UPDATE_PERSON,
            person_id=ancestry_person_id,
            message=f"Unknown fact type: {fact_type}",
        )

    def _apply_vital_fact(
        self,
        person_id: str,
        tree_id: str,
        fact_type: FactTypeEnum,
        value: str,
    ) -> TreeUpdateResponse:
        """Apply birth or death fact using updatePerson API."""
        # Parse the value (expected format: "date|place" or just value)
        parts = value.split("|") if "|" in value else [value, ""]
        date_value = parts[0].strip() if parts else ""
        place_value = parts[1].strip() if len(parts) > 1 else ""

        if fact_type == FactTypeEnum.BIRTH:
            updates = {}
            if date_value:
                updates["bdate"] = date_value
            if place_value:
                updates["bplace"] = place_value
        else:  # DEATH
            updates = {}
            if date_value:
                updates["ddate"] = date_value
            if place_value:
                updates["dplace"] = place_value

        if not updates:
            return TreeUpdateResponse(
                result=TreeUpdateResult.VALIDATION_ERROR,
                operation=TreeOperationType.UPDATE_PERSON,
                person_id=person_id,
                message=f"No valid {fact_type.value} data to apply",
            )

        return self.update_person(tree_id, person_id, updates)

    def _apply_general_fact(
        self,
        person_id: str,
        tree_id: str,
        event_type: str,
        value: str,
    ) -> TreeUpdateResponse:
        """Apply general fact using factedit API."""
        # Parse the value (expected format: "date|location|description")
        parts = value.split("|") if "|" in value else [value]
        date_value = parts[0].strip() if parts else ""
        location_value = parts[1].strip() if len(parts) > 1 else ""
        description = parts[2].strip() if len(parts) > 2 else ""

        return self.add_fact(
            tree_id,
            person_id,
            event_type,
            date=date_value,
            location=location_value,
            description=description,
        )


# =============================================================================
# Batch Operations
# =============================================================================


def apply_approved_facts_batch(
    session_manager: SessionManager,
    db_session: Session,
    tree_id: str,
    limit: int = 10,
) -> list[TreeUpdateResponse]:
    """
    Apply all APPROVED SuggestedFacts in batch.

    Args:
        session_manager: Active SessionManager
        db_session: Database session
        tree_id: Target tree ID
        limit: Maximum number of facts to process

    Returns:
        List of TreeUpdateResponse for each fact
    """
    service = TreeUpdateService(session_manager)
    results: list[TreeUpdateResponse] = []

    # Query APPROVED facts that haven't been applied yet
    approved_facts = (
        db_session.query(SuggestedFact).filter(SuggestedFact.status == FactStatusEnum.APPROVED).limit(limit).all()
    )

    logger.info(f"Processing {len(approved_facts)} approved suggested facts")

    for fact in approved_facts:
        try:
            result = service.apply_suggested_fact(db_session, fact, tree_id)
            results.append(result)

            # Log the result
            if result.result == TreeUpdateResult.SUCCESS:
                logger.info(f"✅ Applied fact {fact.id}: {result.message}")
                # Mark as applied (we'll add APPLIED status later)
            else:
                logger.warning(f"⚠️ Failed to apply fact {fact.id}: {result.message}")

        except Exception as e:
            logger.exception(f"Error applying fact {fact.id}: {e}")
            results.append(
                TreeUpdateResponse(
                    result=TreeUpdateResult.FAILURE,
                    operation=TreeOperationType.UPDATE_PERSON,
                    person_id=str(fact.people_id),
                    message=f"Exception: {e}",
                )
            )

    return results


# =============================================================================
# Module Tests
# =============================================================================


def _test_tree_update_result_enum() -> None:
    """Test TreeUpdateResult enum has expected values."""
    assert TreeUpdateResult.SUCCESS.value == "SUCCESS"
    assert TreeUpdateResult.FAILURE.value == "FAILURE"
    assert TreeUpdateResult.API_ERROR.value == "API_ERROR"


def _test_tree_operation_type_enum() -> None:
    """Test TreeOperationType enum has expected values."""
    assert TreeOperationType.UPDATE_PERSON.value == "UPDATE_PERSON"
    assert TreeOperationType.ADD_FACT.value == "ADD_FACT"
    assert TreeOperationType.LINK_PERSON.value == "LINK_PERSON"


def _test_fact_type_to_event_type_mapping() -> None:
    """Test fact type to event type mapping."""
    assert FACT_TYPE_TO_EVENT_TYPE[FactTypeEnum.BIRTH] == "birth"
    assert FACT_TYPE_TO_EVENT_TYPE[FactTypeEnum.DEATH] == "death"
    assert FACT_TYPE_TO_EVENT_TYPE[FactTypeEnum.LOCATION] == "residence"
    assert FACT_TYPE_TO_EVENT_TYPE[FactTypeEnum.MARRIAGE] == "marriage"


def _test_tree_update_request_creation() -> None:
    """Test TreeUpdateRequest dataclass creation."""
    req = TreeUpdateRequest(
        operation=TreeOperationType.UPDATE_PERSON,
        tree_id="12345",
        person_id="67890",
        data={"bdate": "1900"},
    )
    assert req.tree_id == "12345"
    assert req.person_id == "67890"
    assert req.operation == TreeOperationType.UPDATE_PERSON


def _test_tree_update_response_creation() -> None:
    """Test TreeUpdateResponse dataclass creation."""
    resp = TreeUpdateResponse(
        result=TreeUpdateResult.SUCCESS,
        operation=TreeOperationType.UPDATE_PERSON,
        person_id="12345",
        message="Test message",
    )
    assert resp.result == TreeUpdateResult.SUCCESS
    assert resp.timestamp is not None
    assert resp.person_id == "12345"


def _test_update_person_builds_correct_payload() -> None:
    """Test update_person constructs the correct API payload."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_123"
    mock_sm.my_uuid = "UUID_123"
    service = TreeUpdateService(mock_sm)

    # Capture what _make_api_request receives
    with patch.object(service, "_make_api_request", return_value=(True, {"ok": True}, None)) as mock_req:
        result = service.update_person(
            tree_id="TREE_1",
            person_id="PERSON_2",
            updates={"bdate": "1900-01-01", "bplace": "London"},
            gender="Male",
        )

    assert result.result == TreeUpdateResult.SUCCESS, f"Expected SUCCESS, got {result.result}"
    assert result.operation == TreeOperationType.UPDATE_PERSON
    assert result.person_id == "PERSON_2"

    # Verify the API was called with correct payload structure
    mock_req.assert_called_once()
    call_args = mock_req.call_args
    url = call_args[0][0]
    payload = call_args[0][2]

    assert "updatePerson" in url, f"URL should contain 'updatePerson', got {url}"
    assert "TREE_1" in url, "URL should contain tree_id"
    assert "PERSON_2" in url, "URL should contain person_id"
    assert payload["person"]["personId"] == "PERSON_2"
    assert payload["person"]["treeId"] == "TREE_1"
    assert payload["person"]["gender"] == "Male"
    assert payload["values"]["bdate"] == "1900-01-01"
    assert payload["values"]["bplace"] == "London"


def _test_add_fact_builds_correct_payload() -> None:
    """Test add_fact constructs the correct API payload."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_123"
    service = TreeUpdateService(mock_sm)

    with patch.object(service, "_make_api_request", return_value=(True, {"ok": True}, None)) as mock_req:
        result = service.add_fact(
            tree_id="TREE_1",
            person_id="PERSON_2",
            event_type="death",
            date="1975-03-15",
            location="Manchester, England",
            description="Died peacefully",
        )

    assert result.result == TreeUpdateResult.SUCCESS
    assert result.operation == TreeOperationType.ADD_FACT
    assert "death" in result.message.lower()

    mock_req.assert_called_once()
    call_args = mock_req.call_args
    url = call_args[0][0]
    payload = call_args[0][2]

    assert "factedit" in url, f"URL should contain 'factedit', got {url}"
    assert payload["eventType"] == "death"
    assert payload["date"] == "1975-03-15"
    assert payload["location"] == "Manchester, England"
    assert payload["description"] == "Died peacefully"
    assert payload["assertionId"] == "0", "New fact should use assertionId=0"


def _test_update_person_handles_api_failure() -> None:
    """Test update_person returns API_ERROR on failure."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_123"
    service = TreeUpdateService(mock_sm)

    with patch.object(service, "_make_api_request", return_value=(False, None, "Server error 500")):
        result = service.update_person(
            tree_id="TREE_1",
            person_id="PERSON_2",
            updates={"bdate": "1900"},
        )

    assert result.result == TreeUpdateResult.API_ERROR, f"Expected API_ERROR, got {result.result}"
    assert result.error_details == "Server error 500"
    assert result.person_id == "PERSON_2"


def _test_apply_suggested_fact_rejects_non_approved() -> None:
    """Test apply_suggested_fact validates status is APPROVED."""
    from unittest.mock import MagicMock

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_123"
    service = TreeUpdateService(mock_sm)

    mock_db_session = MagicMock()
    mock_fact = MagicMock()
    mock_fact.status = FactStatusEnum.PENDING
    mock_fact.people_id = 42

    result = service.apply_suggested_fact(mock_db_session, mock_fact, "TREE_1")

    assert result.result == TreeUpdateResult.VALIDATION_ERROR
    assert "APPROVED" in result.message


def _test_build_url_includes_user() -> None:
    """Test _build_url constructs correct URLs with and without user ID."""
    from unittest.mock import MagicMock

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_42"
    service = TreeUpdateService(mock_sm)

    url_with_user = service._build_url("updatePerson", "TREE_1", "PERSON_2", include_user=True)
    assert "USER_42" in url_with_user, "URL should include user ID"
    assert "TREE_1" in url_with_user
    assert "PERSON_2" in url_with_user
    assert "updatePerson" in url_with_user

    url_without_user = service._build_url("factedit", "TREE_1", "PERSON_2", include_user=False)
    assert "USER_42" not in url_without_user, "URL should not include user ID"
    assert "TREE_1" in url_without_user


def _test_service_instantiation() -> None:
    """Test TreeUpdateService can be instantiated with a mock SessionManager."""
    from unittest.mock import MagicMock

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_99"
    mock_sm.my_uuid = "UUID_99"
    service = TreeUpdateService(mock_sm)

    assert service.session_manager is mock_sm, "session_manager should be stored"
    assert service._user_id is None, "_user_id should start as None (lazy)"
    assert service._default_tree_id is None, "_default_tree_id should start as None"
    # Verify lazy user_id resolution
    assert service.user_id == "USER_99", "user_id property should resolve from my_profile_id"


def _test_service_method_signatures() -> None:
    """Test TreeUpdateService has expected methods with correct signatures."""
    import inspect
    from unittest.mock import MagicMock

    mock_sm = MagicMock()
    service = TreeUpdateService(mock_sm)

    # Verify key public methods exist
    expected_methods = [
        "update_person",
        "add_fact",
        "add_person_with_relationship",
        "link_existing_person",
        "remove_relationship",
        "change_relationship_type",
        "apply_suggested_fact",
    ]
    for method_name in expected_methods:
        assert hasattr(service, method_name), f"Missing method: {method_name}"
        assert callable(getattr(service, method_name)), f"{method_name} should be callable"

    # Verify update_person signature
    sig = inspect.signature(service.update_person)
    params = list(sig.parameters.keys())
    assert "tree_id" in params, "update_person should accept tree_id"
    assert "person_id" in params, "update_person should accept person_id"
    assert "updates" in params, "update_person should accept updates"

    # Verify add_fact signature
    sig = inspect.signature(service.add_fact)
    params = list(sig.parameters.keys())
    assert "event_type" in params, "add_fact should accept event_type"
    assert "date" in params, "add_fact should accept date"
    assert "location" in params, "add_fact should accept location"

    # Verify add_person_with_relationship signature
    sig = inspect.signature(service.add_person_with_relationship)
    params = list(sig.parameters.keys())
    assert "relationship_type" in params, "add_person_with_relationship should accept relationship_type"
    assert "new_person_data" in params, "add_person_with_relationship should accept new_person_data"


def _test_add_person_with_relationship_spouse_payload() -> None:
    """Test add_person_with_relationship builds correct payload for Spouse."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_123"
    service = TreeUpdateService(mock_sm)

    new_person = {
        "fname": "Jane",
        "lname": "Doe",
        "genderRadio": "Female",
        "statusRadio": "Deceased",
        "bdate": "1850-06-15",
        "bplace": "Liverpool",
    }

    with patch.object(service, "_make_api_request", return_value=(True, {"personId": "NEW_99"}, None)) as mock_req:
        result = service.add_person_with_relationship(
            tree_id="TREE_1",
            source_person_id="PERSON_2",
            relationship_type="Spouse",
            new_person_data=new_person,
            source_gender="Male",
        )

    assert result.result == TreeUpdateResult.SUCCESS
    assert result.operation == TreeOperationType.ADD_PERSON
    assert result.person_id == "NEW_99", "Should use personId from API response"
    assert "Jane" in result.message

    mock_req.assert_called_once()
    call_args = mock_req.call_args
    url = call_args[0][0]
    payload = call_args[0][2]

    assert "addperson" in url, f"URL should contain 'addperson', got {url}"
    assert payload["type"] == "Spouse"
    assert payload["person"]["personId"] == "PERSON_2"
    assert payload["person"]["gender"] == "Male"
    assert payload["values"]["fname"] == "Jane"
    assert payload["values"]["lname"] == "Doe"
    assert payload["values"]["genderRadio"] == "Female"
    assert payload["values"]["statusRadio"] == "Deceased"
    assert payload["values"]["bdate"] == "1850-06-15"
    assert payload["values"]["spousalRelationship"] == "Spouse"


def _test_add_person_with_relationship_child_payload() -> None:
    """Test add_person_with_relationship builds correct payload for Child with parentSet."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_123"
    service = TreeUpdateService(mock_sm)

    new_person = {"fname": "Tommy", "lname": "Smith"}
    parent_set = {"fatherId": "P_FATHER", "motherId": "P_MOTHER"}

    with patch.object(service, "_make_api_request", return_value=(True, {"personId": "CHILD_1"}, None)) as mock_req:
        result = service.add_person_with_relationship(
            tree_id="TREE_1",
            source_person_id="PERSON_2",
            relationship_type="Child",
            new_person_data=new_person,
            parent_set=parent_set,
        )

    assert result.result == TreeUpdateResult.SUCCESS
    mock_req.assert_called_once()
    payload = mock_req.call_args[0][2]

    assert payload["type"] == "Child"
    assert payload["values"]["parentSet"] == parent_set
    assert payload["values"]["fname"] == "Tommy"
    # Spouse-specific fields should NOT be present
    assert "spousalRelationship" not in payload["values"]


def _test_add_person_with_relationship_api_failure() -> None:
    """Test add_person_with_relationship returns API_ERROR on failure."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_123"
    service = TreeUpdateService(mock_sm)

    with patch.object(service, "_make_api_request", return_value=(False, None, "Timeout")):
        result = service.add_person_with_relationship(
            tree_id="TREE_1",
            source_person_id="PERSON_2",
            relationship_type="Father",
            new_person_data={"fname": "John"},
        )

    assert result.result == TreeUpdateResult.API_ERROR
    assert result.person_id == "PERSON_2", "On failure, should return source_person_id"
    assert result.error_details == "Timeout"


def _test_link_existing_person_payload() -> None:
    """Test link_existing_person constructs correct API payload."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_123"
    service = TreeUpdateService(mock_sm)

    target_info = {
        "name": "Mary Jones",
        "birth": "1820",
        "death": "1890",
        "gender": "Female",
    }

    with patch.object(service, "_make_api_request", return_value=(True, {"ok": True}, None)) as mock_req:
        result = service.link_existing_person(
            tree_id="TREE_1",
            source_person_id="PERSON_2",
            target_person_id="12345",
            relationship_type="Mother",
            target_person_info=target_info,
        )

    assert result.result == TreeUpdateResult.SUCCESS
    assert result.operation == TreeOperationType.LINK_PERSON
    assert result.person_id == "12345"
    assert "Mary Jones" in result.message

    mock_req.assert_called_once()
    payload = mock_req.call_args[0][2]

    assert payload["type"] == "Mother"
    apm = payload["values"]["apmFindExistingPerson"]
    assert apm["name"] == "Mary Jones"
    assert apm["birth"] == "1820"
    assert apm["death"] == "1890"
    assert apm["PID"] == 12345, "PID should be converted to int"
    assert apm["genderIconType"] == "Female"


def _test_remove_relationship_payload() -> None:
    """Test remove_relationship constructs correct URL and payload."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_123"
    service = TreeUpdateService(mock_sm)

    with patch.object(service, "_make_api_request", return_value=(True, {"ok": True}, None)) as mock_req:
        result = service.remove_relationship(
            tree_id="TREE_1",
            source_person_id="PERSON_2",
            related_person_id="PERSON_5",
            relationship_type="C",
            parent_type="M",
        )

    assert result.result == TreeUpdateResult.SUCCESS
    assert result.operation == TreeOperationType.REMOVE_RELATIONSHIP
    assert "PERSON_5" in result.message

    mock_req.assert_called_once()
    url = mock_req.call_args[0][0]
    payload = mock_req.call_args[0][2]

    assert "relationship" in url
    assert "PERSON_5" in url
    assert "removerelationship" in url
    assert payload["type"] == "C"
    assert payload["parentType"] == "M"


def _test_change_relationship_type_payload() -> None:
    """Test change_relationship_type constructs correct URL and payload."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_123"
    service = TreeUpdateService(mock_sm)

    with patch.object(service, "_make_api_request", return_value=(True, {"ok": True}, None)) as mock_req:
        result = service.change_relationship_type(
            tree_id="TREE_1",
            source_person_id="PERSON_2",
            related_person_id="PERSON_7",
            new_modifier="spu",
            original_modifier="sps",
            relationship_type="W",
        )

    assert result.result == TreeUpdateResult.SUCCESS
    assert result.operation == TreeOperationType.CHANGE_RELATIONSHIP
    assert "sps" in result.message and "spu" in result.message

    mock_req.assert_called_once()
    url = mock_req.call_args[0][0]
    payload = mock_req.call_args[0][2]

    assert "changerelationship" in url
    assert "PERSON_7" in url
    assert payload["modifier"] == "spu"
    assert payload["originalModifier"] == "sps"
    assert payload["type"] == "W"
    assert payload["pty"] == -1


def _test_apply_vital_fact_routes_birth() -> None:
    """Test _apply_vital_fact correctly routes birth facts to update_person."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_123"
    service = TreeUpdateService(mock_sm)

    with patch.object(service, "update_person") as mock_update:
        mock_update.return_value = TreeUpdateResponse(
            result=TreeUpdateResult.SUCCESS,
            operation=TreeOperationType.UPDATE_PERSON,
            person_id="P1",
            message="ok",
        )
        service._apply_vital_fact("P1", "TREE_1", FactTypeEnum.BIRTH, "1850-03-20|London")

    mock_update.assert_called_once()
    call_args = mock_update.call_args
    assert call_args[0][0] == "TREE_1"
    assert call_args[0][1] == "P1"
    updates = call_args[0][2]
    assert updates["bdate"] == "1850-03-20"
    assert updates["bplace"] == "London"


def _test_apply_vital_fact_routes_death() -> None:
    """Test _apply_vital_fact correctly routes death facts with ddate/dplace keys."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_123"
    service = TreeUpdateService(mock_sm)

    with patch.object(service, "update_person") as mock_update:
        mock_update.return_value = TreeUpdateResponse(
            result=TreeUpdateResult.SUCCESS,
            operation=TreeOperationType.UPDATE_PERSON,
            person_id="P1",
            message="ok",
        )
        service._apply_vital_fact("P1", "TREE_1", FactTypeEnum.DEATH, "1920-11-05|Manchester")

    updates = mock_update.call_args[0][2]
    assert "ddate" in updates, "Death fact should use ddate key"
    assert "dplace" in updates, "Death fact should use dplace key"
    assert updates["ddate"] == "1920-11-05"
    assert updates["dplace"] == "Manchester"


def _test_apply_vital_fact_empty_value() -> None:
    """Test _apply_vital_fact returns VALIDATION_ERROR for empty value."""
    from unittest.mock import MagicMock

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_123"
    service = TreeUpdateService(mock_sm)

    result = service._apply_vital_fact("P1", "TREE_1", FactTypeEnum.BIRTH, "|")
    assert result.result == TreeUpdateResult.VALIDATION_ERROR
    assert "No valid" in result.message


def _test_apply_general_fact_parses_value() -> None:
    """Test _apply_general_fact parses pipe-separated value and calls add_fact."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.my_profile_id = "USER_123"
    service = TreeUpdateService(mock_sm)

    with patch.object(service, "add_fact") as mock_add:
        mock_add.return_value = TreeUpdateResponse(
            result=TreeUpdateResult.SUCCESS,
            operation=TreeOperationType.ADD_FACT,
            person_id="P1",
            message="ok",
        )
        service._apply_general_fact("P1", "TREE_1", "residence", "1900|London|Census record")

    mock_add.assert_called_once_with(
        "TREE_1", "P1", "residence",
        date="1900",
        location="London",
        description="Census record",
    )


def module_tests() -> bool:
    """Module-specific tests for TreeUpdateService."""
    suite = TestSuite("TreeUpdateService", "api/tree_update.py")
    suite.start_suite()

    suite.run_test(
        "TreeUpdateResult enum values",
        _test_tree_update_result_enum,
        "Verifies TreeUpdateResult enum has expected values",
    )
    suite.run_test(
        "TreeOperationType enum values",
        _test_tree_operation_type_enum,
        "Verifies TreeOperationType enum has expected values",
    )
    suite.run_test(
        "Fact type to event type mapping",
        _test_fact_type_to_event_type_mapping,
        "Verifies mapping from FactTypeEnum to Ancestry event types",
    )
    suite.run_test(
        "TreeUpdateRequest creation",
        _test_tree_update_request_creation,
        "Verifies TreeUpdateRequest dataclass instantiation",
    )
    suite.run_test(
        "TreeUpdateResponse creation",
        _test_tree_update_response_creation,
        "Verifies TreeUpdateResponse dataclass instantiation",
    )
    suite.run_test(
        "update_person builds correct payload",
        _test_update_person_builds_correct_payload,
        "Verifies update_person constructs correct API URL and payload",
    )
    suite.run_test(
        "add_fact builds correct payload",
        _test_add_fact_builds_correct_payload,
        "Verifies add_fact constructs correct API URL and payload",
    )
    suite.run_test(
        "update_person handles API failure",
        _test_update_person_handles_api_failure,
        "Verifies update_person returns API_ERROR on failure",
    )
    suite.run_test(
        "apply_suggested_fact rejects non-APPROVED",
        _test_apply_suggested_fact_rejects_non_approved,
        "Verifies apply_suggested_fact validates APPROVED status",
    )
    suite.run_test(
        "build_url constructs correct URLs",
        _test_build_url_includes_user,
        "Verifies _build_url includes/excludes user ID correctly",
    )
    suite.run_test(
        "Service instantiation with mock",
        _test_service_instantiation,
        "Verifies TreeUpdateService stores session_manager and resolves user_id",
    )
    suite.run_test(
        "Service method signatures",
        _test_service_method_signatures,
        "Verifies key methods exist with expected parameter names",
    )
    suite.run_test(
        "add_person_with_relationship Spouse payload",
        _test_add_person_with_relationship_spouse_payload,
        "Verifies Spouse payload structure including spousalRelationship",
    )
    suite.run_test(
        "add_person_with_relationship Child payload",
        _test_add_person_with_relationship_child_payload,
        "Verifies Child payload includes parentSet and omits spousal fields",
    )
    suite.run_test(
        "add_person_with_relationship API failure",
        _test_add_person_with_relationship_api_failure,
        "Verifies API_ERROR result and source_person_id on failure",
    )
    suite.run_test(
        "link_existing_person payload",
        _test_link_existing_person_payload,
        "Verifies apmFindExistingPerson structure with PID as int",
    )
    suite.run_test(
        "remove_relationship payload",
        _test_remove_relationship_payload,
        "Verifies URL contains removerelationship and payload has type/parentType",
    )
    suite.run_test(
        "change_relationship_type payload",
        _test_change_relationship_type_payload,
        "Verifies changerelationship URL and modifier fields in payload",
    )
    suite.run_test(
        "_apply_vital_fact routes birth",
        _test_apply_vital_fact_routes_birth,
        "Verifies birth value parsed to bdate/bplace and routed to update_person",
    )
    suite.run_test(
        "_apply_vital_fact routes death",
        _test_apply_vital_fact_routes_death,
        "Verifies death value parsed to ddate/dplace keys",
    )
    suite.run_test(
        "_apply_vital_fact empty value",
        _test_apply_vital_fact_empty_value,
        "Verifies VALIDATION_ERROR for empty pipe-separated value",
    )
    suite.run_test(
        "_apply_general_fact parses value",
        _test_apply_general_fact_parses_value,
        "Verifies pipe-separated value parsed and passed to add_fact",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
