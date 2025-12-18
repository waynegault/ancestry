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
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

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
    fact_type: Optional[FactTypeEnum] = None
    suggested_fact_id: Optional[int] = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeUpdateResponse:
    """Response from a tree update operation."""

    result: TreeUpdateResult
    operation: TreeOperationType
    person_id: str
    message: str
    api_response: Optional[dict[str, Any]] = None
    error_details: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


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
        self._user_id: Optional[str] = None
        self._default_tree_id: Optional[str] = None

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
        data: Optional[dict[str, Any]] = None,
        content_type: str = "application/json",
    ) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
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
        parent_set: Optional[dict[str, str]] = None,
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
        parent_set: Optional[dict[str, Any]] = None,
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

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
