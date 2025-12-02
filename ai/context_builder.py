#!/usr/bin/env python3
"""
ContextBuilder - Assembles Rich Context for AI Response Generation

Aggregates data from multiple sources (Database, GEDCOM, Conversation History)
to create a comprehensive context payload for the LLM to generate personalized
genealogical responses.

Sprint 1, Task 2: Implement Context Builder
"""

# === STANDARD LIBRARY IMPORTS ===
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class MatchContext:
    """
    Comprehensive context about a DNA match for AI response generation.

    This is the primary output of the ContextBuilder, designed to be
    serialized to JSON and included in AI prompts.
    """

    # Identity
    identity: dict[str, Any] = field(default_factory=dict)

    # Genetic information
    genetics: dict[str, Any] = field(default_factory=dict)

    # Genealogical context
    genealogy: dict[str, Any] = field(default_factory=dict)

    # Conversation history
    history: dict[str, Any] = field(default_factory=dict)

    # Extracted facts from conversations
    extracted_facts: dict[str, Any] = field(default_factory=dict)

    # Metadata
    context_generated_at: str = ""
    context_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "identity": self.identity,
            "genetics": self.genetics,
            "genealogy": self.genealogy,
            "history": self.history,
            "extracted_facts": self.extracted_facts,
            "context_generated_at": self.context_generated_at,
            "context_version": self.context_version,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_prompt_string(self) -> str:
        """
        Convert to a formatted string suitable for inclusion in an AI prompt.
        """
        lines = ["=== DNA MATCH CONTEXT ==="]

        # Identity section
        if self.identity:
            lines.append("\n## Identity")
            lines.append(f"Name: {self.identity.get('name', 'Unknown')}")
            if self.identity.get('managed_by'):
                lines.append(f"Managed by: {self.identity.get('managed_by')}")

        # Genetics section
        if self.genetics:
            lines.append("\n## Genetic Connection")
            lines.append(f"Shared DNA: {self.genetics.get('shared_cm', 'Unknown')} cM")
            lines.append(f"Segments: {self.genetics.get('segments', 'Unknown')}")
            lines.append(f"Predicted Relationship: {self.genetics.get('prediction', 'Unknown')}")

        # Genealogy section
        if self.genealogy:
            lines.append("\n## Genealogical Connection")
            if self.genealogy.get('known_common_ancestors'):
                lines.append("Common Ancestors:")
                for ancestor in self.genealogy['known_common_ancestors'][:3]:
                    name = ancestor.get('name', 'Unknown')
                    birth = ancestor.get('birth', '')
                    death = ancestor.get('death', '')
                    lines.append(f"  - {name} ({birth}-{death})")
            if self.genealogy.get('relationship_description'):
                lines.append(f"Relationship: {self.genealogy['relationship_description']}")

        # History section
        if self.history:
            lines.append("\n## Conversation History")
            lines.append(f"Last Interaction: {self.history.get('last_interaction_date', 'Never')}")
            if self.history.get('summary'):
                lines.append(f"Summary: {self.history['summary']}")
            if self.history.get('messages'):
                lines.append("Recent Messages:")
                for msg in self.history['messages'][-3:]:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:200]
                    lines.append(f"  [{role}]: {content}")

        # Facts section
        if self.extracted_facts:
            lines.append("\n## Extracted Facts")
            for key, values in self.extracted_facts.items():
                if values:
                    lines.append(f"{key}: {', '.join(str(v) for v in values[:5])}")

        lines.append("\n=== END CONTEXT ===")
        return "\n".join(lines)


class ContextBuilder:
    """
    Builds comprehensive context for AI response generation.

    Aggregates data from:
    - Database (Person, ConversationLog tables)
    - GEDCOM file (via TreeQueryService)
    - Extracted facts from previous conversations
    """

    def __init__(self, db_session: Session, tree_service: Optional[Any] = None):
        """
        Initialize the ContextBuilder.

        Args:
            db_session: SQLAlchemy database session
            tree_service: Optional TreeQueryService instance (lazy-loaded if not provided)
        """
        self._session = db_session
        self._tree_service = tree_service
        self._tree_service_initialized = tree_service is not None

    def _ensure_tree_service(self) -> Any:
        """Lazy initialization of TreeQueryService."""
        if not self._tree_service_initialized:
            self._tree_service_initialized = True
            try:
                from genealogy.tree_query_service import TreeQueryService

                self._tree_service = TreeQueryService()
            except Exception as e:
                logger.warning(f"Could not initialize TreeQueryService: {e}")
                self._tree_service = None
        return self._tree_service

    def build_context(self, match_uuid: str) -> MatchContext:
        """
        Build comprehensive context for a DNA match.

        Args:
            match_uuid: UUID of the DNA match (Person.uuid)

        Returns:
            MatchContext with all available information
        """
        context = MatchContext(
            context_generated_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            # Get Person from database
            from database import Person

            person = self._session.query(Person).filter(Person.uuid == match_uuid.upper()).first()

            if not person:
                logger.warning(f"Person not found for UUID: {match_uuid}")
                return context

            # Build each section
            context.identity = ContextBuilder._build_identity(person)
            context.genetics = ContextBuilder._build_genetics(person)
            context.history = self._build_history(person)
            context.extracted_facts = self._build_extracted_facts(person)
            context.genealogy = self._build_genealogy(person)

        except Exception as e:
            logger.error(f"Error building context for {match_uuid}: {e}", exc_info=True)

        return context

    @staticmethod
    def _build_identity(person: Any) -> dict[str, Any]:
        """Build identity section from Person record."""
        identity = {
            "name": person.display_name if hasattr(person, 'display_name') else person.username,
            "uuid": person.uuid,
            "profile_id": person.profile_id,
        }

        # Check if managed by administrator
        if person.administrator_profile_id:
            identity["managed_by"] = "Administrator (non-member DNA test)"
            identity["administrator_id"] = person.administrator_profile_id
        else:
            identity["managed_by"] = "Self"

        # Add notes if present
        if hasattr(person, 'note') and person.note:
            identity["user_notes"] = person.note

        return identity

    @staticmethod
    def _build_genetics(person: Any) -> dict[str, Any]:
        """Build genetics section from Person record."""
        genetics: dict[str, Any] = {}

        if hasattr(person, 'shared_cm') and person.shared_cm:
            genetics["shared_cm"] = person.shared_cm

        if hasattr(person, 'segments') and person.segments:
            genetics["segments"] = person.segments

        if hasattr(person, 'predicted_relationship') and person.predicted_relationship:
            genetics["prediction"] = person.predicted_relationship

        # Add relationship bucket for prompt context
        shared_cm = genetics.get("shared_cm", 0)
        if shared_cm >= 1500:
            genetics["relationship_bucket"] = "Close family (parent/child/sibling)"
        elif shared_cm >= 400:
            genetics["relationship_bucket"] = "Extended family (1st-2nd cousin)"
        elif shared_cm >= 90:
            genetics["relationship_bucket"] = "Distant relative (3rd-4th cousin)"
        else:
            genetics["relationship_bucket"] = "Remote relative (5th+ cousin)"

        return genetics

    def _build_history(self, person: Any) -> dict[str, Any]:
        """Build conversation history section."""
        history: dict[str, Any] = {
            "messages": [],
            "last_interaction_date": None,
            "summary": None,
        }

        try:
            from database import ConversationLog, MessageDirectionEnum

            # Get recent conversations for this person
            logs = (
                self._session.query(ConversationLog)
                .filter(ConversationLog.people_id == person.id)
                .order_by(ConversationLog.latest_timestamp.desc())
                .limit(10)
                .all()
            )

            if logs:
                history["last_interaction_date"] = logs[0].latest_timestamp.isoformat()

                # Build message list
                for log in reversed(logs):  # Oldest first
                    role = "user" if log.direction == MessageDirectionEnum.OUT else "match"
                    if log.latest_message_content:
                        history["messages"].append(
                            {
                                "role": role,
                                "content": log.latest_message_content,
                                "timestamp": log.latest_timestamp.isoformat(),
                                "sentiment": log.ai_sentiment if hasattr(log, 'ai_sentiment') else None,
                            }
                        )

                # Generate summary from last message sentiment
                if logs[0].ai_sentiment:
                    history["summary"] = f"Last message classified as: {logs[0].ai_sentiment}"

        except Exception as e:
            logger.warning(f"Error building history: {e}")

        return history

    def _build_extracted_facts(self, person: Any) -> dict[str, Any]:
        """Build extracted facts section from SuggestedFacts."""
        facts: dict[str, list[str]] = {
            "mentioned_surnames": [],
            "mentioned_locations": [],
            "mentioned_dates": [],
            "mentioned_relationships": [],
        }

        try:
            from database import FactStatusEnum, FactTypeEnum, SuggestedFact

            # Get approved facts for this person
            suggested_facts = (
                self._session.query(SuggestedFact)
                .filter(
                    SuggestedFact.people_id == person.id,
                    SuggestedFact.status.in_([FactStatusEnum.APPROVED, FactStatusEnum.PENDING]),
                )
                .all()
            )

            for fact in suggested_facts:
                if fact.fact_type == FactTypeEnum.LOCATION:
                    facts["mentioned_locations"].append(fact.new_value)
                elif fact.fact_type == FactTypeEnum.BIRTH:
                    facts["mentioned_dates"].append(f"Birth: {fact.new_value}")
                elif fact.fact_type == FactTypeEnum.DEATH:
                    facts["mentioned_dates"].append(f"Death: {fact.new_value}")
                elif fact.fact_type == FactTypeEnum.RELATIONSHIP:
                    facts["mentioned_relationships"].append(fact.new_value)

        except Exception as e:
            logger.debug(f"Error building extracted facts: {e}")

        return facts

    def _build_genealogy(self, person: Any) -> dict[str, Any]:
        """Build genealogy section using TreeQueryService."""
        genealogy: dict[str, Any] = {
            "known_common_ancestors": [],
            "surnames_in_common": [],
            "relationship_description": None,
            "in_tree": False,
        }

        # Check if person is in tree
        if hasattr(person, 'in_my_tree'):
            genealogy["in_tree"] = person.in_my_tree

        # Try to get relationship path from GEDCOM
        tree_service = self._ensure_tree_service()
        if tree_service and person.uuid:
            try:
                # Get relationship explanation
                rel_result = tree_service.explain_relationship(person.uuid)
                if rel_result.found:
                    genealogy["relationship_description"] = rel_result.relationship_description
                    genealogy["relationship_label"] = rel_result.relationship_label

                    if rel_result.common_ancestor:
                        genealogy["known_common_ancestors"].append(rel_result.common_ancestor)

                # Get common ancestors
                common = tree_service.get_common_ancestors(person.uuid)
                if common:
                    genealogy["known_common_ancestors"].extend(common[:5])

            except Exception as e:
                logger.debug(f"Error getting genealogy from TreeQueryService: {e}")

        return genealogy


# === TESTS ===


def module_tests() -> bool:
    """Run module tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("ContextBuilder", "ai/context_builder.py")
    suite.start_suite()

    # Test 1: MatchContext dataclass
    def test_match_context():
        context = MatchContext(
            identity={"name": "John Smith", "uuid": "ABC123"},
            genetics={"shared_cm": 150, "segments": 8},
        )
        assert context.identity["name"] == "John Smith"
        assert context.genetics["shared_cm"] == 150

        # Test serialization
        d = context.to_dict()
        assert "identity" in d
        assert "genetics" in d

        # Test JSON
        j = context.to_json()
        assert "John Smith" in j

    suite.run_test(
        "MatchContext dataclass",
        test_match_context,
        test_summary="Verify MatchContext dataclass and serialization",
    )

    # Test 2: MatchContext prompt string
    def test_match_context_prompt():
        context = MatchContext(
            identity={"name": "Jane Doe", "managed_by": "Self"},
            genetics={"shared_cm": 200, "segments": 10, "prediction": "3rd Cousin"},
            history={"last_interaction_date": "2025-01-15", "messages": []},
        )
        prompt = context.to_prompt_string()
        assert "Jane Doe" in prompt
        assert "200 cM" in prompt
        assert "3rd Cousin" in prompt

    suite.run_test(
        "MatchContext prompt string",
        test_match_context_prompt,
        test_summary="Verify MatchContext converts to prompt-friendly string",
    )

    # Test 3: ContextBuilder initialization
    def test_context_builder_init():
        # Test with None session (for unit testing)
        builder = ContextBuilder(db_session=None)  # type: ignore
        assert builder._session is None
        assert builder._tree_service_initialized is False

    suite.run_test(
        "ContextBuilder initialization",
        test_context_builder_init,
        test_summary="Verify ContextBuilder initializes correctly",
    )

    # Test 4: Genetics bucket calculation
    def test_genetics_bucket():
        from unittest.mock import MagicMock

        _ = ContextBuilder(db_session=None)  # type: ignore

        # Test close family
        person = MagicMock()
        person.shared_cm = 1600
        person.segments = 50
        person.predicted_relationship = "Parent/Child"

        genetics = ContextBuilder._build_genetics(person)
        assert genetics["relationship_bucket"] == "Close family (parent/child/sibling)"

        # Test distant
        person.shared_cm = 50
        genetics = ContextBuilder._build_genetics(person)
        assert genetics["relationship_bucket"] == "Remote relative (5th+ cousin)"

    suite.run_test(
        "Genetics relationship bucket",
        test_genetics_bucket,
        test_summary="Verify genetics bucket calculation based on shared cM",
    )

    return suite.finish_suite()


run_comprehensive_tests = module_tests

if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
