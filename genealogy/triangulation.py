import logging
from typing import Any, Optional, cast

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from database import Person, SharedMatch
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

    def find_triangulation_opportunities(self, target_person_id: str) -> list[dict[str, Any]]:
        """
        Find triangulation opportunities for a target person.

        Args:
            target_person_id: The UUID or profile_id of the target person.

        Returns:
            List of triangulation hypotheses.
        """
        target_person = self._resolve_person(target_person_id)
        if not target_person:
            logger.error(f"Target person not found: {target_person_id}")
            return []

        shared_matches = self._get_shared_matches(target_person)
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

    def _resolve_person(self, person_id: str) -> Optional[Person]:
        """Resolve Person object."""
        return (
            self.db.query(Person).filter((Person.profile_id == person_id) | (Person.uuid == person_id.upper())).first()
        )

    def _get_shared_matches(self, person: Person) -> list[Person]:
        """
        Retrieve shared matches for the person from the database.
        """
        stmt = (
            select(SharedMatch)
            .options(joinedload(SharedMatch.shared_match_person))
            .where(SharedMatch.person_id == person.id)
        )
        results = self.db.execute(stmt).scalars().all()

        return [sm.shared_match_person for sm in results if sm.shared_match_person]
