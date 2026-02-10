#!/usr/bin/env python3
"""
TreeQueryService - Real-time Genealogical Query Service

Provides a clean interface for querying family tree data from GEDCOM files
and explaining relationships in natural language. Wraps the existing action10
logic into a service suitable for the Context Builder and Response Engine.

Sprint 1, Task 1: Core Intelligence & Retrieval
"""

# === STANDARD LIBRARY IMPORTS ===
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PersonSearchResult:
    """Result of searching for a person in the tree."""

    found: bool = False
    person_id: str | None = None
    name: str = ""
    first_name: str | None = None
    last_name: str | None = None
    birth_year: int | None = None
    birth_place: str | None = None
    death_year: int | None = None
    death_place: str | None = None
    gender: str | None = None
    match_score: int = 0
    confidence: str = "low"  # low, medium, high
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "found": self.found,
            "person_id": self.person_id,
            "name": self.name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "birth_year": self.birth_year,
            "birth_place": self.birth_place,
            "death_year": self.death_year,
            "death_place": self.death_place,
            "gender": self.gender,
            "match_score": self.match_score,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
        }


@dataclass
class RelationshipResult:
    """Result of calculating relationship between two people."""

    found: bool = False
    relationship_label: str = ""  # e.g., "3rd cousin twice removed"
    relationship_description: str = ""  # Natural language explanation
    path: list[dict[str, Any]] = field(default_factory=list)  # Full path with details
    common_ancestor: dict[str, Any] | None = None
    generations_apart: int = 0
    confidence: str = "low"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "found": self.found,
            "relationship_label": self.relationship_label,
            "relationship_description": self.relationship_description,
            "path": self.path,
            "common_ancestor": self.common_ancestor,
            "generations_apart": self.generations_apart,
            "confidence": self.confidence,
        }

    def get_surname_line(self) -> str:
        """
        Extract the primary surname line from the common ancestor.

        Returns:
            Surname line description like "via the Smith line" or empty string
        """
        if not self.common_ancestor:
            return ""
        ancestor_name = self.common_ancestor.get("name", "")
        if not ancestor_name:
            return ""
        # Extract surname (last word of name, typically)
        name_parts = ancestor_name.split()
        if len(name_parts) >= 2:
            surname = name_parts[-1]
            return f"via the {surname} line"
        return f"via {ancestor_name}"

    def to_prompt_string(self) -> str:
        """
        Format the relationship for inclusion in an AI prompt.

        Returns:
            Human-readable relationship explanation for AI prompts
        """
        if not self.found:
            return "No relationship found between these individuals."

        parts: list[str] = []

        # Main relationship label
        if self.relationship_label:
            parts.append(f"RELATIONSHIP: {self.relationship_label}")

        # Surname line if available
        surname_line = self.get_surname_line()
        if surname_line:
            parts.append(f"CONNECTION: {surname_line}")

        # Common ancestor if available
        if self.common_ancestor:
            ancestor_name = self.common_ancestor.get("name", "Unknown")
            ancestor_birth = self.common_ancestor.get("birth_year")
            if ancestor_birth:
                parts.append(f"COMMON ANCESTOR: {ancestor_name} (b. {ancestor_birth})")
            else:
                parts.append(f"COMMON ANCESTOR: {ancestor_name}")

        # Generations apart
        if self.generations_apart > 0:
            parts.append(f"GENERATIONS APART: {self.generations_apart}")

        # Full path description
        if self.relationship_description:
            parts.append(f"PATH: {self.relationship_description}")

        # Confidence level
        if self.confidence:
            parts.append(f"CONFIDENCE: {self.confidence}")

        return "\n".join(parts)


@dataclass
class FamilyMember:
    """A single family member with basic details."""

    person_id: str
    name: str
    relation: str  # parent, sibling, spouse, child
    birth_year: int | None = None
    death_year: int | None = None
    birth_place: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "person_id": self.person_id,
            "name": self.name,
            "relation": self.relation,
            "birth_year": self.birth_year,
            "death_year": self.death_year,
            "birth_place": self.birth_place,
        }


@dataclass
class FamilyMembersResult:
    """Result of getting family members for a person."""

    found: bool = False
    person_id: str = ""
    person_name: str = ""
    parents: list[FamilyMember] = field(default_factory=list)
    siblings: list[FamilyMember] = field(default_factory=list)
    spouses: list[FamilyMember] = field(default_factory=list)
    children: list[FamilyMember] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "found": self.found,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "parents": [p.to_dict() for p in self.parents],
            "siblings": [s.to_dict() for s in self.siblings],
            "spouses": [s.to_dict() for s in self.spouses],
            "children": [c.to_dict() for c in self.children],
        }

    def to_prompt_string(self) -> str:
        """Format family members for inclusion in AI prompts."""
        if not self.found:
            return "No family members found."

        lines: list[str] = [f"Family of {self.person_name}:"]

        if self.parents:
            lines.append("\nParents:")
            for p in self.parents:
                details = f" (b. {p.birth_year})" if p.birth_year else ""
                lines.append(f"  - {p.name}{details}")

        if self.siblings:
            lines.append("\nSiblings:")
            for s in self.siblings:
                details = f" (b. {s.birth_year})" if s.birth_year else ""
                lines.append(f"  - {s.name}{details}")

        if self.spouses:
            lines.append("\nSpouses:")
            for s in self.spouses:
                details = f" (b. {s.birth_year})" if s.birth_year else ""
                lines.append(f"  - {s.name}{details}")

        if self.children:
            lines.append("\nChildren:")
            for c in self.children:
                details = f" (b. {c.birth_year})" if c.birth_year else ""
                lines.append(f"  - {c.name}{details}")

        return "\n".join(lines)


class TreeQueryService:
    """
    Service for querying family tree data.

    Provides:
    - find_person: Search for a person by name with fuzzy matching
    - explain_relationship: Get relationship between two people
    - get_ancestors: Get ancestors up to N generations
    - get_descendants: Get descendants up to N generations
    """

    def __init__(self, gedcom_path: Path | None = None):
        """
        Initialize the TreeQueryService.

        Args:
            gedcom_path: Path to GEDCOM file. If None, uses config default.
        """
        self._gedcom_data: Any = None
        self._gedcom_path = gedcom_path
        self._reference_person_id: str | None = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of GEDCOM data."""
        if self._initialized:
            return self._gedcom_data is not None

        self._initialized = True

        try:
            from actions.action10 import load_gedcom_data
            from config import config_schema

            # Determine GEDCOM path
            if self._gedcom_path is None:
                if config_schema and config_schema.database.gedcom_file_path:
                    self._gedcom_path = Path(config_schema.database.gedcom_file_path)
                else:
                    logger.warning("No GEDCOM path configured")
                    return False

            if not self._gedcom_path.exists():
                logger.error(f"GEDCOM file not found: {self._gedcom_path}")
                return False

            # Load GEDCOM data
            logger.info(f"Loading GEDCOM: {self._gedcom_path.name}")
            self._gedcom_data = load_gedcom_data(self._gedcom_path)

            if self._gedcom_data:
                logger.info(f"GEDCOM loaded: {len(self._gedcom_data.indi_index)} individuals")

                # Get reference person ID from config
                if config_schema and config_schema.reference_person_id:
                    self._reference_person_id = config_schema.reference_person_id

                return True

        except Exception as e:
            logger.error(f"Failed to initialize TreeQueryService: {e}", exc_info=True)

        return False

    def find_person(
        self,
        name: str,
        approx_birth_year: int | None = None,
        location: str | None = None,
        max_results: int = 5,
    ) -> PersonSearchResult:
        """
        Search for a person in the family tree.

        Args:
            name: Person's name (first, last, or full)
            approx_birth_year: Approximate birth year (±5 years tolerance)
            location: Birth or residence location
            max_results: Maximum number of alternative matches to return

        Returns:
            PersonSearchResult with best match and alternatives
        """
        if not self._ensure_initialized():
            return PersonSearchResult(found=False, confidence="low")

        try:
            from actions.action10 import filter_and_score_individuals
            from config import config_schema

            # Parse name into components
            name_parts = name.strip().split()
            first_name = name_parts[0].lower() if name_parts else ""
            last_name = name_parts[-1].lower() if len(name_parts) > 1 else ""

            # Build search criteria
            search_criteria = {
                "first_name": first_name,
                "surname": last_name,
                "birth_year": approx_birth_year,
                "birth_place": location or "",
            }

            # Search using existing action10 logic
            matches = filter_and_score_individuals(
                self._gedcom_data,
                search_criteria,
                search_criteria,
                dict(config_schema.common_scoring_weights),
                {"year_match_range": 5.0},
            )

            if not matches:
                return PersonSearchResult(found=False, name=name, confidence="low")

            # Process best match
            best_match = matches[0]
            score = int(best_match.get("total_score", 0))

            # Determine confidence based on score
            if score >= 80:
                confidence = "high"
            elif score >= 50:
                confidence = "medium"
            else:
                confidence = "low"

            result = PersonSearchResult(
                found=True,
                person_id=best_match.get("id"),
                name=best_match.get("name", name),
                first_name=best_match.get("first_name"),
                last_name=best_match.get("last_name"),
                birth_year=best_match.get("birth_year"),
                birth_place=best_match.get("birth_place"),
                death_year=best_match.get("death_year"),
                death_place=best_match.get("death_place"),
                gender=best_match.get("gender"),
                match_score=score,
                confidence=confidence,
            )

            # Add alternatives
            if len(matches) > 1:
                result.alternatives = [
                    {
                        "person_id": m.get("id"),
                        "name": m.get("name"),
                        "birth_year": m.get("birth_year"),
                        "score": int(m.get("total_score", 0)),
                    }
                    for m in matches[1 : max_results + 1]
                ]

            return result

        except Exception as e:
            logger.error(f"Error searching for person '{name}': {e}", exc_info=True)
            return PersonSearchResult(found=False, name=name, confidence="low")

    def explain_relationship(
        self,
        person_a_id: str,
        person_b_id: str | None = None,
    ) -> RelationshipResult:
        """
        Explain the relationship between two people.

        Args:
            person_a_id: GEDCOM ID of first person (or the DNA match)
            person_b_id: GEDCOM ID of second person (defaults to reference person)

        Returns:
            RelationshipResult with path and natural language explanation
        """
        not_found = RelationshipResult(found=False)

        if not self._ensure_initialized():
            return not_found

        try:
            from research.relationship_utils import convert_gedcom_path_to_unified_format, fast_bidirectional_bfs

            # Use reference person if person_b not specified
            target_id = person_b_id if person_b_id else self._reference_person_id
            if not target_id:
                logger.warning("No reference person ID configured")
                return not_found

            # Normalize IDs
            from genealogy.gedcom import gedcom_utils

            norm_a = gedcom_utils.normalize_id(person_a_id)
            norm_b = gedcom_utils.normalize_id(target_id)

            # Early exit if IDs invalid or no path found
            if not norm_a or not norm_b:
                return not_found

            # Find path using BFS
            path_ids = fast_bidirectional_bfs(
                norm_a,
                norm_b,
                self._gedcom_data.id_to_parents,
                self._gedcom_data.id_to_children,
                max_depth=25,
                node_limit=150000,
                timeout_sec=30,
            )

            # Convert to unified format with details
            unified_path = (
                convert_gedcom_path_to_unified_format(
                    path_ids,
                    self._gedcom_data.reader,
                    self._gedcom_data.id_to_parents,
                    self._gedcom_data.id_to_children,
                    self._gedcom_data.indi_index,
                )
                if path_ids
                else None
            )

            if not unified_path:
                return not_found

            # Generate natural language description
            description = self._generate_relationship_description(unified_path)
            label = TreeQueryService._generate_relationship_label(unified_path)

            # Find common ancestor (middle of path for simple cases)
            common_ancestor = None
            if len(unified_path) > 2:
                mid_idx = len(unified_path) // 2
                common_ancestor = unified_path[mid_idx]

            return RelationshipResult(
                found=True,
                relationship_label=label,
                relationship_description=description,
                path=unified_path,
                common_ancestor=common_ancestor,
                generations_apart=len(unified_path) - 1,
                confidence="high" if len(unified_path) < 10 else "medium",
            )

        except Exception as e:
            logger.error(f"Error explaining relationship: {e}", exc_info=True)
            return not_found

    @staticmethod
    def _generate_relationship_label(path: list[dict[str, Any]]) -> str:
        """Generate a relationship label like '3rd cousin twice removed'."""
        if not path:
            return "Unknown"

        generations = len(path) - 1

        # Handle special cases with dictionary lookup
        if generations <= 2:
            special_labels = {0: "Self", 2: "Grandparent/Grandchild or Sibling"}
            if generations in special_labels:
                return special_labels[generations]
            # generations == 1: Direct parent/child
            relation_type = path[0].get("relation_to_next", "").lower()
            if "parent" in relation_type or "father" in relation_type or "mother" in relation_type:
                return "Parent"
            return "Child" if "child" in relation_type else "Close relative"

        # Extended family
        if generations <= 4:
            return f"{generations - 1}x Great-Grandparent/Grandchild or Cousin"
        cousin_degree = (generations - 2) // 2
        return f"{TreeQueryService._ordinal(cousin_degree)} Cousin (approx)"

    @staticmethod
    def _ordinal(n: int) -> str:
        """Convert number to ordinal (1st, 2nd, 3rd, etc.)."""
        suffix = "th" if 11 <= n % 100 <= 13 else ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
        return f"{n}{suffix}"

    @staticmethod
    def _generate_relationship_description(path: list[dict[str, Any]]) -> str:
        """Generate a natural language description of the relationship path."""
        if not path:
            return "No relationship path found."

        if len(path) == 1:
            return f"This is {path[0].get('name', 'the same person')}."

        # Build description from path
        parts: list[str] = []
        for i, node in enumerate(path):
            name = node.get("name", "Unknown")
            birth = node.get("birth_year")
            relation = node.get("relation_to_next", "")

            if i == 0:
                parts.append(f"Starting from {name}")
                if birth:
                    parts[-1] += f" (b. {birth})"
            else:
                if relation:
                    parts.append(f"→ {relation}: {name}")
                else:
                    parts.append(f"→ {name}")
                if birth:
                    parts[-1] += f" (b. {birth})"

        return " ".join(parts)

    def get_person_details(self, person_id: str) -> dict[str, Any] | None:
        """
        Get detailed information about a person.

        Args:
            person_id: GEDCOM ID of the person

        Returns:
            Dictionary with person details or None if not found
        """
        if not self._ensure_initialized():
            return None

        try:
            from genealogy.gedcom import gedcom_utils

            norm_id = gedcom_utils.normalize_id(person_id)

            if norm_id not in self._gedcom_data.indi_index:
                return None

            indi = self._gedcom_data.indi_index[norm_id]

            # Get basic info
            name = gedcom_utils.get_full_name(indi) if hasattr(gedcom_utils, 'get_full_name') else str(indi)

            # Use get_event_info from gedcom_utils
            birth_date, _, birth_place = gedcom_utils.get_event_info(indi, "BIRT")
            birth_info = {"year": birth_date.year if birth_date else None, "place": birth_place}

            death_date, _, death_place = gedcom_utils.get_event_info(indi, "DEAT")
            death_info = {"year": death_date.year if death_date else None, "place": death_place}

            # Get family
            parents = list(self._gedcom_data.id_to_parents.get(norm_id, set()))
            children = list(self._gedcom_data.id_to_children.get(norm_id, set()))

            return {
                "person_id": norm_id,
                "name": name,
                "birth_year": birth_info["year"],
                "birth_place": birth_info["place"],
                "death_year": death_info["year"],
                "death_place": death_info["place"],
                "parents": parents,
                "children": children,
            }

        except Exception as e:
            logger.error(f"Error getting person details: {e}", exc_info=True)
            return None

    def _collect_parents(self, norm_id: str, result: FamilyMembersResult) -> None:
        """Collect parent family members."""
        parent_ids = self._gedcom_data.id_to_parents.get(norm_id, set())
        for pid in parent_ids:
            member = self._get_family_member_details(pid, "parent")
            if member:
                result.parents.append(member)

    def _collect_siblings(self, norm_id: str, result: FamilyMembersResult) -> None:
        """Collect sibling family members."""
        parent_ids = self._gedcom_data.id_to_parents.get(norm_id, set())
        for parent_id in parent_ids:
            sibling_ids = self._gedcom_data.id_to_children.get(parent_id, set())
            for sid in sibling_ids:
                if sid != norm_id and not any(s.person_id == sid for s in result.siblings):
                    member = self._get_family_member_details(sid, "sibling")
                    if member:
                        result.siblings.append(member)

    def _collect_spouses_and_children(self, norm_id: str, result: FamilyMembersResult) -> None:
        """Collect spouse and children family members."""
        spouse_ids = self._get_spouse_ids(norm_id)
        for sid in spouse_ids:
            member = self._get_family_member_details(sid, "spouse")
            if member:
                result.spouses.append(member)
        children_ids = self._gedcom_data.id_to_children.get(norm_id, set())
        for cid in children_ids:
            member = self._get_family_member_details(cid, "child")
            if member:
                result.children.append(member)

    def get_family_members(self, person_id: str) -> FamilyMembersResult:
        """Get all family members for a person: parents, siblings, spouses, and children."""
        not_found = FamilyMembersResult(found=False)
        if not self._ensure_initialized():
            return not_found

        try:
            from genealogy.gedcom import gedcom_utils

            norm_id = gedcom_utils.normalize_id(person_id)
            if norm_id is None or norm_id not in self._gedcom_data.indi_index:
                return not_found

            indi = self._gedcom_data.indi_index[norm_id]
            person_name = gedcom_utils.get_full_name(indi) if hasattr(gedcom_utils, "get_full_name") else str(indi)
            result = FamilyMembersResult(found=True, person_id=norm_id, person_name=person_name)

            self._collect_parents(norm_id, result)
            self._collect_siblings(norm_id, result)
            self._collect_spouses_and_children(norm_id, result)
            return result

        except Exception as e:
            logger.error(f"Error getting family members: {e}", exc_info=True)
            return not_found

    def _get_family_member_details(self, person_id: str, relation: str) -> FamilyMember | None:
        """Get details for a single family member."""
        try:
            from genealogy.gedcom import gedcom_utils

            if person_id not in self._gedcom_data.indi_index:
                return None

            indi = self._gedcom_data.indi_index[person_id]
            name = gedcom_utils.get_full_name(indi) if hasattr(gedcom_utils, "get_full_name") else str(indi)

            birth_date, _, birth_place = gedcom_utils.get_event_info(indi, "BIRT")
            death_date, _, _ = gedcom_utils.get_event_info(indi, "DEAT")

            return FamilyMember(
                person_id=person_id,
                name=name,
                relation=relation,
                birth_year=birth_date.year if birth_date else None,
                death_year=death_date.year if death_date else None,
                birth_place=birth_place,
            )
        except Exception:
            return None

    def _get_spouse_ids(self, person_id: str) -> set[str]:
        """Get spouse IDs from FAMS records."""
        spouse_ids: set[str] = set()
        if not self._gedcom_data or person_id not in self._gedcom_data.indi_index:
            return spouse_ids

        try:
            indi = self._gedcom_data.indi_index[person_id]
            spouse_ids = self._extract_spouse_ids_from_individual(indi, person_id)
        except Exception:
            pass  # Silent failure for spouse lookup
        return spouse_ids

    def _extract_spouse_ids_from_individual(self, indi: Any, person_id: str) -> set[str]:
        """Extract spouse IDs from individual's FAMS records."""
        spouse_ids: set[str] = set()
        spouse_tags = {"HUSB", "WIFE"}

        for sub in indi.sub_records:
            if sub.tag != "FAMS" or not sub.value:
                continue
            fam_id = sub.value
            if fam_id not in self._gedcom_data.fam_index:
                continue
            fam = self._gedcom_data.fam_index[fam_id]
            for fam_sub in fam.sub_records:
                if fam_sub.tag in spouse_tags and fam_sub.value and fam_sub.value != person_id:
                    spouse_ids.add(fam_sub.value)

        return spouse_ids

    def get_common_ancestors(
        self,
        person_a_id: str,
        person_b_id: str | None = None,
        max_generations: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Find common ancestors between two people.

        Args:
            person_a_id: GEDCOM ID of first person
            person_b_id: GEDCOM ID of second person (defaults to reference person)
            max_generations: Maximum generations to search

        Returns:
            List of common ancestors with details
        """
        if not self._ensure_initialized():
            return []

        if person_b_id is None:
            person_b_id = self._reference_person_id

        if not person_b_id:
            return []

        try:
            from genealogy.gedcom import gedcom_utils

            norm_a = gedcom_utils.normalize_id(person_a_id)
            norm_b = gedcom_utils.normalize_id(person_b_id)

            if not norm_a or not norm_b:
                return []

            # Get ancestors of both
            ancestors_a = self._get_ancestors_set(norm_a, max_generations)
            ancestors_b = self._get_ancestors_set(norm_b, max_generations)

            # Find common ancestors
            common = ancestors_a.intersection(ancestors_b)

            # Get details for each
            results: list[dict[str, Any]] = []
            for ancestor_id in list(common)[:10]:  # Limit to 10
                details = self.get_person_details(ancestor_id)
                if details:
                    results.append(details)

            return results

        except Exception as e:
            logger.error(f"Error finding common ancestors: {e}", exc_info=True)
            return []

    def _get_ancestors_set(self, person_id: str, max_generations: int) -> set[str]:
        """Get all ancestors up to max_generations."""
        ancestors: set[str] = set()
        current_gen = {person_id}

        for _ in range(max_generations):
            next_gen: set[str] = set()
            for pid in current_gen:
                parents = self._gedcom_data.id_to_parents.get(pid, set())
                next_gen.update(parents)
                ancestors.update(parents)
            if not next_gen:
                break
            current_gen = next_gen

        return ancestors


# === TESTS ===


def module_tests() -> bool:
    """Run module tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("TreeQueryService", "genealogy/tree_query_service.py")
    suite.start_suite()

    # Test 1: PersonSearchResult dataclass
    def test_person_search_result():
        result = PersonSearchResult(
            found=True,
            person_id="I001",
            name="John Smith",
            match_score=85,
            confidence="high",
        )
        assert result.found is True
        assert result.person_id == "I001"
        assert result.to_dict()["name"] == "John Smith"

    suite.run_test(
        "PersonSearchResult dataclass",
        test_person_search_result,
        test_summary="Verify PersonSearchResult dataclass works correctly",
    )

    # Test 2: RelationshipResult dataclass
    def test_relationship_result():
        result = RelationshipResult(
            found=True,
            relationship_label="3rd cousin",
            generations_apart=6,
        )
        assert result.found is True
        assert result.relationship_label == "3rd cousin"
        d = result.to_dict()
        assert d["generations_apart"] == 6

    suite.run_test(
        "RelationshipResult dataclass",
        test_relationship_result,
        test_summary="Verify RelationshipResult dataclass works correctly",
    )

    # Test 3: TreeQueryService initialization
    def test_service_init():
        service = TreeQueryService()
        # Should not crash even without GEDCOM
        assert service._initialized is False

    suite.run_test(
        "TreeQueryService initialization",
        test_service_init,
        test_summary="Verify TreeQueryService initializes without error",
    )

    # Test 4: Ordinal generation
    def test_ordinal():
        assert TreeQueryService._ordinal(1) == "1st"
        assert TreeQueryService._ordinal(2) == "2nd"
        assert TreeQueryService._ordinal(3) == "3rd"
        assert TreeQueryService._ordinal(4) == "4th"
        assert TreeQueryService._ordinal(11) == "11th"
        assert TreeQueryService._ordinal(21) == "21st"

    suite.run_test(
        "Ordinal number generation",
        test_ordinal,
        test_summary="Verify ordinal number generation works correctly",
    )

    # Test 5: FamilyMember dataclass
    def test_family_member():
        member = FamilyMember(
            person_id="I002",
            name="Jane Doe",
            relation="parent",
            birth_year=1950,
            death_year=2020,
            birth_place="Boston, MA",
        )
        assert member.name == "Jane Doe"
        assert member.relation == "parent"
        d = member.to_dict()
        assert d["birth_year"] == 1950

    suite.run_test(
        "FamilyMember dataclass",
        test_family_member,
        test_summary="Verify FamilyMember dataclass works correctly",
    )

    # Test 6: FamilyMembersResult dataclass
    def test_family_members_result():
        result = FamilyMembersResult(
            found=True,
            person_id="I001",
            person_name="John Smith",
            parents=[FamilyMember(person_id="I002", name="Jane Doe", relation="parent", birth_year=1950)],
            children=[FamilyMember(person_id="I003", name="Jack Smith", relation="child", birth_year=1990)],
        )
        assert result.found is True
        assert len(result.parents) == 1
        assert len(result.children) == 1
        d = result.to_dict()
        assert d["person_name"] == "John Smith"
        assert len(d["parents"]) == 1

    suite.run_test(
        "FamilyMembersResult dataclass",
        test_family_members_result,
        test_summary="Verify FamilyMembersResult dataclass works correctly",
    )

    # Test 7: FamilyMembersResult.to_prompt_string
    def test_family_members_prompt_string():
        result = FamilyMembersResult(
            found=True,
            person_id="I001",
            person_name="John Smith",
            parents=[FamilyMember(person_id="I002", name="Jane Doe", relation="parent", birth_year=1950)],
            siblings=[FamilyMember(person_id="I004", name="James Smith", relation="sibling", birth_year=1972)],
        )
        prompt_str = result.to_prompt_string()
        assert "John Smith" in prompt_str
        assert "Parents:" in prompt_str
        assert "Jane Doe" in prompt_str
        assert "(b. 1950)" in prompt_str
        assert "Siblings:" in prompt_str
        assert "James Smith" in prompt_str

    suite.run_test(
        "FamilyMembersResult.to_prompt_string",
        test_family_members_prompt_string,
        test_summary="Verify to_prompt_string formats family members correctly",
    )

    # Test 8: RelationshipResult.get_surname_line
    def test_relationship_surname_line():
        result = RelationshipResult(
            found=True,
            relationship_label="3rd cousin",
            common_ancestor={"name": "John Smith", "birth_year": 1850},
        )
        surname_line = result.get_surname_line()
        assert surname_line == "via the Smith line"

        # Test with single name
        result_single = RelationshipResult(
            found=True,
            common_ancestor={"name": "Elizabeth"},
        )
        assert result_single.get_surname_line() == "via Elizabeth"

        # Test with no common ancestor
        result_none = RelationshipResult(found=False)
        assert not result_none.get_surname_line()

    suite.run_test(
        "RelationshipResult.get_surname_line",
        test_relationship_surname_line,
        test_summary="Verify surname line extraction from common ancestor",
    )

    # Test 9: RelationshipResult.to_prompt_string
    def test_relationship_prompt_string():
        result = RelationshipResult(
            found=True,
            relationship_label="3rd cousin twice removed",
            relationship_description="Starting from John (b. 1970) → parent: Mary (b. 1940)",
            common_ancestor={"name": "William Smith", "birth_year": 1850},
            generations_apart=5,
            confidence="high",
        )
        prompt_str = result.to_prompt_string()
        assert "RELATIONSHIP: 3rd cousin twice removed" in prompt_str
        assert "via the Smith line" in prompt_str
        assert "COMMON ANCESTOR: William Smith (b. 1850)" in prompt_str
        assert "GENERATIONS APART: 5" in prompt_str
        assert "CONFIDENCE: high" in prompt_str

        # Test not found case
        not_found = RelationshipResult(found=False)
        assert "No relationship found" in not_found.to_prompt_string()

    suite.run_test(
        "RelationshipResult.to_prompt_string",
        test_relationship_prompt_string,
        test_summary="Verify to_prompt_string formats relationship for AI prompts",
    )

    return suite.finish_suite()


run_comprehensive_tests = module_tests

if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
