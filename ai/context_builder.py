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
from typing import Any, Optional, cast

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

    # Research insights (best-effort enrichment)
    research: dict[str, Any] = field(default_factory=dict)

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
            "research": self.research,
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
        lines.extend(self._format_identity())
        lines.extend(self._format_genetics())
        lines.extend(self._format_genealogy())
        lines.extend(self._format_history())
        lines.extend(self._format_facts())
        lines.extend(self._format_research())
        lines.append("\n=== END CONTEXT ===")
        return "\n".join(lines)

    def _format_identity(self) -> list[str]:
        lines: list[str] = []
        if self.identity:
            lines.append("\n## Identity")
            lines.append(f"Name: {self.identity.get('name', 'Unknown')}")
            if self.identity.get('managed_by'):
                lines.append(f"Managed by: {self.identity.get('managed_by')}")
        return lines

    def _format_genetics(self) -> list[str]:
        lines: list[str] = []
        if self.genetics:
            lines.append("\n## Genetic Connection")
            lines.append(f"Shared DNA: {self.genetics.get('shared_cm', 'Unknown')} cM")
            lines.append(f"Segments: {self.genetics.get('segments', 'Unknown')}")
            lines.append(f"Predicted Relationship: {self.genetics.get('prediction', 'Unknown')}")
        return lines

    def _format_genealogy(self) -> list[str]:
        lines: list[str] = []
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
        return lines

    def _format_history(self) -> list[str]:
        lines: list[str] = []
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
        return lines

    def _format_facts(self) -> list[str]:
        lines: list[str] = []
        if self.extracted_facts:
            lines.append("\n## Extracted Facts")
            for key, values in self.extracted_facts.items():
                if values:
                    lines.append(f"{key}: {', '.join(str(v) for v in values[:5])}")
        return lines

    def _format_research(self) -> list[str]:
        lines: list[str] = []
        if not self.research:
            return lines

        lines.append("\n## Research Insights")

        shared_regions = self.research.get("ethnicity_shared_regions")
        if shared_regions:
            lines.append(f"Ethnicity overlap: {', '.join(str(r) for r in shared_regions[:3])}")

        cluster = self.research.get("shared_match_cluster")
        cluster_count: Any = None
        if isinstance(cluster, dict):
            cluster_dict = cast(dict[str, Any], cluster)
            cluster_count = cluster_dict.get("shared_match_count")
        if isinstance(cluster_count, int) and cluster_count > 0:
            lines.append(f"Shared matches: {cluster_count}")

        return lines


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
            from core.database import Person

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
            context.research = self._build_research_insights(person)

        except Exception as e:
            logger.error(f"Error building context for {match_uuid}: {e}", exc_info=True)

        return context

    @staticmethod
    def _resolve_owner_profile_id() -> Optional[str]:
        """Best-effort owner profile ID resolution (session manager > config)."""
        try:
            from core.session_utils import get_session_manager

            session_manager = get_session_manager()
            if session_manager and getattr(session_manager, "my_profile_id", None):
                return session_manager.my_profile_id
        except Exception:
            pass

        try:
            from config import config_schema

            return getattr(config_schema, "testing_profile_id", None)
        except Exception:
            return None

    def _build_research_insights(self, person: Any) -> dict[str, Any]:
        """Build lightweight research insights used for draft suggestions."""
        research: dict[str, Any] = {}

        # Shared-match cluster insight (pure DB; no external calls)
        try:
            research["shared_match_cluster"] = self._build_shared_match_cluster(person)
        except Exception as exc:
            logger.debug(f"Shared match cluster enrichment skipped: {exc}")

        # Ethnicity commonality insight (best-effort; requires owner_profile_id)
        try:
            owner_profile_id = ContextBuilder._resolve_owner_profile_id()
            person_id = getattr(person, "id", None)
            if owner_profile_id and person_id:
                from genealogy.tree_stats_utils import calculate_ethnicity_commonality

                ethnicity = calculate_ethnicity_commonality(self._session, owner_profile_id, int(person_id))
                shared_regions = ethnicity.get("shared_regions", []) or []
                if shared_regions:
                    research.update(
                        {
                            "ethnicity_shared_regions": shared_regions,
                            "ethnicity_similarity": ethnicity.get("similarity_score"),
                            "ethnicity_top_region": ethnicity.get("top_shared_region"),
                        }
                    )
        except Exception as exc:
            logger.debug(f"Ethnicity enrichment skipped: {exc}")

        return research

    def _build_shared_match_cluster(self, person: Any) -> dict[str, Any]:
        """Summarize the shared-match network for this match (count + sample)."""
        cluster: dict[str, Any] = {
            "shared_match_count": 0,
            "example_shared_matches": [],
        }

        person_id = getattr(person, "id", None)
        if not person_id:
            return cluster

        from core.database import SharedMatch

        shared_matches = self._session.query(SharedMatch).filter(SharedMatch.person_id == int(person_id)).all()
        cluster["shared_match_count"] = len(shared_matches)

        # Provide a small sample (first N) to aid human review.
        examples: list[dict[str, Any]] = []
        for row in shared_matches[:5]:
            shared_person = getattr(row, "shared_match_person", None)
            examples.append(
                {
                    "uuid": getattr(shared_person, "uuid", None),
                    "name": getattr(shared_person, "display_name", None)
                    or getattr(shared_person, "username", None)
                    or "Unknown",
                    "shared_cm": getattr(row, "shared_cm", None),
                }
            )
        cluster["example_shared_matches"] = examples
        return cluster

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
            from core.database import ConversationLog, MessageDirectionEnum

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
            from core.database import FactStatusEnum, FactTypeEnum, SuggestedFact

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

        # Try to get relationship path from GEDCOM.
        # IMPORTANT: `Person.uuid` is the Ancestry DNA sample GUID, not a GEDCOM person id.
        # We must resolve a GEDCOM person id (best-effort) before asking TreeQueryService
        # for relationship paths.
        tree_service = self._ensure_tree_service()
        if tree_service:
            try:
                search_name: Optional[str] = None

                family_tree = getattr(person, "family_tree", None)
                if family_tree and getattr(family_tree, "person_name_in_tree", None):
                    search_name = family_tree.person_name_in_tree
                else:
                    search_name = getattr(person, "username", None)

                approx_birth_year = getattr(person, "birth_year", None)

                resolved_person_id: Optional[str] = None
                if search_name:
                    search_result = tree_service.find_person(
                        name=search_name,
                        approx_birth_year=approx_birth_year,
                    )
                    if getattr(search_result, "found", False) and getattr(search_result, "person_id", None):
                        resolved_person_id = search_result.person_id
                        if hasattr(search_result, "to_dict"):
                            genealogy["gedcom_person_match"] = search_result.to_dict()

                if resolved_person_id:
                    # Get relationship explanation
                    rel_result = tree_service.explain_relationship(resolved_person_id)
                    if rel_result.found:
                        genealogy["relationship_description"] = rel_result.relationship_description
                        genealogy["relationship_label"] = rel_result.relationship_label

                        if rel_result.common_ancestor:
                            genealogy["known_common_ancestors"].append(rel_result.common_ancestor)

                    # Get common ancestors
                    common = tree_service.get_common_ancestors(resolved_person_id)
                    if common:
                        genealogy["known_common_ancestors"].extend(common[:5])

            except Exception as e:
                logger.debug(f"Error getting genealogy from TreeQueryService: {e}")

        return genealogy


# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    return True


def _test_to_prompt_string() -> bool:
    """Test to_prompt_string formatting."""
    context = MatchContext(
        identity={"name": "John Doe", "managed_by": "Self"},
        genetics={"shared_cm": 100, "segments": 5, "prediction": "3rd Cousin"},
        history={"last_interaction_date": "2023-01-01"},
    )
    prompt = context.to_prompt_string()

    assert "=== DNA MATCH CONTEXT ===" in prompt
    assert "Name: John Doe" in prompt
    assert "Shared DNA: 100 cM" in prompt
    assert "Last Interaction: 2023-01-01" in prompt
    return True


def _test_build_genealogy_resolves_gedcom_person_id() -> bool:
    class _StubSearchResult:
        def __init__(self) -> None:
            self.found = True
            self.person_id = "I123"

        def to_dict(self) -> dict[str, Any]:
            return {"found": True, "person_id": self.person_id, "name": "John Doe", "confidence": "high"}

    class _StubRelResult:
        def __init__(self) -> None:
            self.found = True
            self.relationship_description = "3rd cousin"
            self.relationship_label = "3rd cousin"
            self.common_ancestor = {"name": "Common Ancestor"}

    class _StubTreeService:
        def __init__(self) -> None:
            self.find_person_calls: list[tuple[str, Optional[int]]] = []
            self.explain_relationship_calls: list[str] = []
            self.get_common_ancestors_calls: list[str] = []

        def find_person(
            self,
            name: str,
            approx_birth_year: Optional[int] = None,
            location: Optional[str] = None,
            max_results: int = 5,
        ) -> Any:
            self.find_person_calls.append((name, approx_birth_year))
            return _StubSearchResult()

        def explain_relationship(self, person_a_id: str, person_b_id: Optional[str] = None) -> Any:
            self.explain_relationship_calls.append(person_a_id)
            return _StubRelResult()

        def get_common_ancestors(self, person_id: str) -> list[dict[str, Any]]:
            self.get_common_ancestors_calls.append(person_id)
            return [{"name": "CA2"}]

    class _StubPerson:
        def __init__(self) -> None:
            self.uuid = "DNA-GUID-IGNORED"
            self.username = "John Doe"
            self.birth_year = 1900
            self.in_my_tree = False
            self.family_tree = None

    tree_service = _StubTreeService()
    ctx = ContextBuilder(db_session=cast(Session, None), tree_service=tree_service)
    genealogy = ctx._build_genealogy(_StubPerson())

    assert tree_service.find_person_calls == [("John Doe", 1900)]
    assert tree_service.explain_relationship_calls == ["I123"]
    assert tree_service.get_common_ancestors_calls == ["I123"]
    assert genealogy.get("relationship_description") == "3rd cousin"
    assert genealogy.get("relationship_label") == "3rd cousin"
    assert genealogy.get("known_common_ancestors"), "Expected common ancestors list to be populated"
    assert genealogy.get("gedcom_person_match", {}).get("person_id") == "I123"
    return True


def _test_build_genealogy_skips_when_no_name() -> bool:
    class _StubTreeService:
        def __init__(self) -> None:
            self.explain_relationship_calls: list[str] = []

        def find_person(
            self,
            name: str,
            approx_birth_year: Optional[int] = None,
            location: Optional[str] = None,
            max_results: int = 5,
        ) -> Any:  # pragma: no cover
            raise AssertionError("find_person should not be called")

        def explain_relationship(self, person_a_id: str, person_b_id: Optional[str] = None) -> Any:  # pragma: no cover
            self.explain_relationship_calls.append(person_a_id)
            raise AssertionError("explain_relationship should not be called")

        def get_common_ancestors(self, person_id: str) -> list[dict[str, Any]]:  # pragma: no cover
            raise AssertionError("get_common_ancestors should not be called")

    class _StubPerson:
        def __init__(self) -> None:
            self.uuid = "DNA-GUID-IGNORED"
            self.username = None
            self.birth_year = None
            self.in_my_tree = False
            self.family_tree = None

    tree_service = _StubTreeService()
    ctx = ContextBuilder(db_session=cast(Session, None), tree_service=tree_service)
    genealogy = ctx._build_genealogy(_StubPerson())
    assert genealogy["relationship_description"] is None
    assert genealogy["known_common_ancestors"] == []
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)
# Add the new test to the runner manually if the utility supports it,
# or just run it as part of a suite.
# Since create_standard_test_runner takes a single function or list, let's just make a simple suite wrapper
# or simpler: just stick to the pattern.
# Actually create_standard_test_runner usually takes a single entry point.
# Let's define a combined test function.


def _run_local_tests() -> bool:
    return (
        _test_module_integrity()
        and _test_to_prompt_string()
        and _test_build_genealogy_resolves_gedcom_person_id()
        and _test_build_genealogy_skips_when_no_name()
    )


run_comprehensive_tests = create_standard_test_runner(_run_local_tests)


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
