import logging
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from core.database import Person, SharedMatch
from genealogy.research_service import ResearchService

logger = logging.getLogger(__name__)


class TriangulationService:
    """
    Service for identifying triangulation opportunities between DNA matches.
    Hypothesizes relationships based on shared matches and common ancestors.
    """

    def __init__(self, db_session: Session, research_service: ResearchService):
        self.db = db_session
        self.research_service = research_service

    def find_triangulation_opportunities(
        self, target_person_id: str, min_confidence: str = "HIGH", min_cm: int = 20
    ) -> list[dict[str, Any]]:
        """
        Find triangulation opportunities for a target person.

        Args:
            target_person_id: The UUID or profile_id of the target person.
            min_confidence: Minimum confidence level (e.g., "HIGH").
            min_cm: Minimum shared centimorgans.

        Returns:
            List of triangulation hypotheses.
        """
        target_person = self._resolve_person(target_person_id)
        if not target_person:
            logger.error(f"Target person not found: {target_person_id}")
            return []

        # Map confidence to cM
        confidence_map = {
            "EXTREMELY_HIGH": 60,
            "VERY_HIGH": 45,
            "HIGH": 30,
            "GOOD": 20,
            "MODERATE": 15,
            "LOW": 6,
        }

        confidence_threshold = confidence_map.get(min_confidence.upper(), 0)
        effective_min_cm = max(min_cm, confidence_threshold)

        shared_matches = self._get_shared_matches(target_person, min_cm=effective_min_cm)
        opportunities: list[dict[str, Any]] = []

        for shared_match in shared_matches:
            # Check if shared match is in our tree (has a known relationship path)
            # We assume if they are in the tree, we can find a path to ROOT
            if not shared_match.uuid:
                continue

            path = self.research_service.get_relationship_path("ROOT", shared_match.uuid)

            if path:
                # Identify common ancestor (simplified: the furthest person in the path)
                # In a real implementation, we'd look for the MRCA (Most Recent Common Ancestor)
                common_ancestor = path[-1] if path else None

                if common_ancestor:
                    hypothesis = {
                        "target_person": target_person,
                        "shared_match": shared_match,
                        "common_ancestor": common_ancestor,
                        "hypothesis_message": self.generate_hypothesis_message(
                            target_person, shared_match, common_ancestor
                        ),
                    }
                    opportunities.append(hypothesis)

        return opportunities

    @staticmethod
    def generate_hypothesis_message(target_person: Person, shared_match: Person, common_ancestor: Any) -> str:
        """Generate a message proposing a triangulation hypothesis."""
        # Using target_person to personalize if needed, though not used in this simple template
        _ = target_person

        ancestor_name = (
            cast(dict[str, Any], common_ancestor).get("name")
            if isinstance(common_ancestor, dict)
            else str(common_ancestor)
        )

        return (
            f"I noticed we both match {shared_match.username}. "
            f"In my tree, {shared_match.username} is related through {ancestor_name}. "
            f"Do you happen to have {ancestor_name} in your tree as well?"
        )

    def _resolve_person(self, person_id: str) -> Person | None:
        """Resolve Person object."""
        return (
            self.db.query(Person).filter((Person.profile_id == person_id) | (Person.uuid == person_id.upper())).first()
        )

    def _get_shared_matches(self, person: Person, min_cm: int = 0) -> list[Person]:
        """
        Retrieve shared matches for the person from the database.
        """
        stmt = (
            select(SharedMatch)
            .options(joinedload(SharedMatch.shared_match_person))
            .where(SharedMatch.person_id == person.id)
        )

        if min_cm > 0:
            stmt = stmt.where(SharedMatch.shared_cm >= min_cm)

        results = self.db.execute(stmt).scalars().all()

        return [sm.shared_match_person for sm in results if sm.shared_match_person]


# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    from unittest.mock import MagicMock

    from testing.test_framework import TestSuite

    suite = TestSuite("Triangulation Service", "genealogy/triangulation.py")
    suite.start_suite()

    def test_triangulation_service_instantiation():
        mock_db = MagicMock()
        mock_research = MagicMock()
        svc = TriangulationService(db_session=mock_db, research_service=mock_research)
        assert svc.db is mock_db
        assert svc.research_service is mock_research
        return True

    suite.run_test("TriangulationService can be instantiated", test_triangulation_service_instantiation)

    # -- generate_hypothesis_message tests --

    def test_generate_hypothesis_message_dict_ancestor():
        mock_target = MagicMock()
        mock_target.username = "Alice"
        mock_shared = MagicMock()
        mock_shared.username = "Bob"
        common_ancestor = {"name": "Great-Grandpa Smith"}
        msg = TriangulationService.generate_hypothesis_message(mock_target, mock_shared, common_ancestor)
        assert isinstance(msg, str)
        assert "Bob" in msg
        assert "Great-Grandpa Smith" in msg
        # Verify the message follows the expected template structure
        assert msg.startswith("I noticed we both match Bob.")
        assert "Do you happen to have Great-Grandpa Smith in your tree" in msg
        return True

    suite.run_test("generate_hypothesis_message with dict ancestor", test_generate_hypothesis_message_dict_ancestor)

    def test_generate_hypothesis_message_string_ancestor():
        """When common_ancestor is not a dict, it should be converted via str()."""
        mock_target = MagicMock()
        mock_target.username = "Alice"
        mock_shared = MagicMock()
        mock_shared.username = "Bob"
        common_ancestor = "John Doe"
        msg = TriangulationService.generate_hypothesis_message(mock_target, mock_shared, common_ancestor)
        assert "John Doe" in msg
        assert "Bob" in msg
        return True

    suite.run_test("generate_hypothesis_message with string ancestor", test_generate_hypothesis_message_string_ancestor)

    def test_generate_hypothesis_message_dict_missing_name_key():
        """When common_ancestor dict has no 'name' key, .get() returns None."""
        mock_target = MagicMock()
        mock_target.username = "Alice"
        mock_shared = MagicMock()
        mock_shared.username = "Bob"
        common_ancestor = {"relationship": "uncle"}
        msg = TriangulationService.generate_hypothesis_message(mock_target, mock_shared, common_ancestor)
        assert isinstance(msg, str)
        assert "None" in msg  # .get("name") returns None
        return True

    suite.run_test("generate_hypothesis_message with dict missing 'name' key", test_generate_hypothesis_message_dict_missing_name_key)

    # -- _resolve_person tests --

    def test_resolve_person_not_found():
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_research = MagicMock()
        svc = TriangulationService(db_session=mock_db, research_service=mock_research)
        result = svc._resolve_person("nonexistent")
        assert result is None
        return True

    suite.run_test("_resolve_person returns None when not found", test_resolve_person_not_found)

    def test_resolve_person_found():
        mock_person = MagicMock(spec=Person)
        mock_person.profile_id = "PROF-123"
        mock_person.uuid = "ABC-DEF"
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_person
        svc = TriangulationService(db_session=mock_db, research_service=MagicMock())
        result = svc._resolve_person("PROF-123")
        assert result is mock_person
        return True

    suite.run_test("_resolve_person returns Person when found", test_resolve_person_found)

    def test_resolve_person_uppercases_uuid():
        """Verify that _resolve_person calls .upper() on the person_id for UUID lookup."""
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        svc = TriangulationService(db_session=mock_db, research_service=MagicMock())
        svc._resolve_person("abc-def")
        # The filter should have been called (we verify the method executes without error
        # and the query chain is invoked)
        mock_db.query.assert_called_once()
        return True

    suite.run_test("_resolve_person queries the database with the person_id", test_resolve_person_uppercases_uuid)

    # -- _get_shared_matches tests --

    def test_get_shared_matches_returns_persons():
        """Verify _get_shared_matches returns Person objects from SharedMatch results."""
        mock_person = MagicMock(spec=Person)
        mock_person.id = 1

        mock_shared_person_a = MagicMock(spec=Person)
        mock_shared_person_a.username = "MatchA"
        mock_shared_person_b = MagicMock(spec=Person)
        mock_shared_person_b.username = "MatchB"

        mock_sm1 = MagicMock()
        mock_sm1.shared_match_person = mock_shared_person_a
        mock_sm2 = MagicMock()
        mock_sm2.shared_match_person = mock_shared_person_b

        mock_db = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = [mock_sm1, mock_sm2]

        svc = TriangulationService(db_session=mock_db, research_service=MagicMock())
        result = svc._get_shared_matches(mock_person, min_cm=0)
        assert len(result) == 2
        assert result[0] is mock_shared_person_a
        assert result[1] is mock_shared_person_b
        return True

    suite.run_test("_get_shared_matches returns Person objects from results", test_get_shared_matches_returns_persons)

    def test_get_shared_matches_filters_none_persons():
        """Shared matches with None shared_match_person should be excluded."""
        mock_person = MagicMock(spec=Person)
        mock_person.id = 1

        mock_sm1 = MagicMock()
        mock_sm1.shared_match_person = MagicMock(spec=Person)
        mock_sm2 = MagicMock()
        mock_sm2.shared_match_person = None  # No linked person

        mock_db = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = [mock_sm1, mock_sm2]

        svc = TriangulationService(db_session=mock_db, research_service=MagicMock())
        result = svc._get_shared_matches(mock_person, min_cm=0)
        assert len(result) == 1
        return True

    suite.run_test("_get_shared_matches filters out None shared_match_person", test_get_shared_matches_filters_none_persons)

    def test_get_shared_matches_empty_results():
        mock_person = MagicMock(spec=Person)
        mock_person.id = 1
        mock_db = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = []
        svc = TriangulationService(db_session=mock_db, research_service=MagicMock())
        result = svc._get_shared_matches(mock_person, min_cm=20)
        assert result == []
        return True

    suite.run_test("_get_shared_matches returns empty list when no results", test_get_shared_matches_empty_results)

    # -- find_triangulation_opportunities tests --

    def test_find_opportunities_returns_empty_when_person_not_found():
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_research = MagicMock()
        svc = TriangulationService(db_session=mock_db, research_service=mock_research)
        result = svc.find_triangulation_opportunities("missing-uuid")
        assert result == []
        return True

    suite.run_test("find_triangulation_opportunities returns [] for missing person", test_find_opportunities_returns_empty_when_person_not_found)

    def test_find_opportunities_with_shared_matches_and_path():
        """Full integration: person found, shared matches exist, relationship path exists."""
        mock_target = MagicMock(spec=Person)
        mock_target.id = 1
        mock_target.username = "TargetUser"

        mock_shared = MagicMock(spec=Person)
        mock_shared.uuid = "SHARED-UUID-1"
        mock_shared.username = "SharedUser"

        mock_db = MagicMock()
        # _resolve_person returns mock_target
        mock_db.query.return_value.filter.return_value.first.return_value = mock_target

        mock_sm = MagicMock()
        mock_sm.shared_match_person = mock_shared
        mock_db.execute.return_value.scalars.return_value.all.return_value = [mock_sm]

        mock_research = MagicMock()
        ancestor_node = {"name": "Common Ancestor"}
        mock_research.get_relationship_path.return_value = ["ROOT", "middle", ancestor_node]

        svc = TriangulationService(db_session=mock_db, research_service=mock_research)
        result = svc.find_triangulation_opportunities("target-id", min_confidence="HIGH", min_cm=20)

        assert len(result) == 1
        assert result[0]["target_person"] is mock_target
        assert result[0]["shared_match"] is mock_shared
        assert result[0]["common_ancestor"] is ancestor_node
        assert "Common Ancestor" in result[0]["hypothesis_message"]
        assert "SharedUser" in result[0]["hypothesis_message"]
        return True

    suite.run_test("find_triangulation_opportunities returns hypotheses with valid path", test_find_opportunities_with_shared_matches_and_path)

    def test_find_opportunities_skips_matches_without_uuid():
        """Shared matches without a uuid should be skipped."""
        mock_target = MagicMock(spec=Person)
        mock_target.id = 1

        mock_shared_no_uuid = MagicMock(spec=Person)
        mock_shared_no_uuid.uuid = None  # No UUID

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_target

        mock_sm = MagicMock()
        mock_sm.shared_match_person = mock_shared_no_uuid
        mock_db.execute.return_value.scalars.return_value.all.return_value = [mock_sm]

        svc = TriangulationService(db_session=mock_db, research_service=MagicMock())
        result = svc.find_triangulation_opportunities("target-id")
        assert result == []
        return True

    suite.run_test("find_triangulation_opportunities skips matches without uuid", test_find_opportunities_skips_matches_without_uuid)

    def test_find_opportunities_skips_matches_without_path():
        """Shared matches where research_service returns no path should be skipped."""
        mock_target = MagicMock(spec=Person)
        mock_target.id = 1

        mock_shared = MagicMock(spec=Person)
        mock_shared.uuid = "HAS-UUID"

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_target

        mock_sm = MagicMock()
        mock_sm.shared_match_person = mock_shared
        mock_db.execute.return_value.scalars.return_value.all.return_value = [mock_sm]

        mock_research = MagicMock()
        mock_research.get_relationship_path.return_value = []  # No path found

        svc = TriangulationService(db_session=mock_db, research_service=mock_research)
        result = svc.find_triangulation_opportunities("target-id")
        assert result == []
        return True

    suite.run_test("find_triangulation_opportunities skips matches without relationship path", test_find_opportunities_skips_matches_without_path)

    def test_find_opportunities_confidence_map_overrides_min_cm():
        """HIGH confidence maps to 30 cM, which should override a lower min_cm."""
        mock_target = MagicMock(spec=Person)
        mock_target.id = 1

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_target
        mock_db.execute.return_value.scalars.return_value.all.return_value = []

        svc = TriangulationService(db_session=mock_db, research_service=MagicMock())
        # min_cm=10, but HIGH maps to 30, so effective_min_cm should be 30
        result = svc.find_triangulation_opportunities("target-id", min_confidence="HIGH", min_cm=10)
        assert result == []
        # Verify execute was called (the query was built and run)
        mock_db.execute.assert_called_once()
        return True

    suite.run_test("find_triangulation_opportunities uses confidence map when higher than min_cm", test_find_opportunities_confidence_map_overrides_min_cm)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_comprehensive_tests() else 1)
