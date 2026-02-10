"""Canonical person and DNA-match summary dataclasses.

This module defines the **single source of truth** for lightweight person
data that flows between subsystems.  Before this module existed, at least
eight near-identical dataclasses / TypedDicts were scattered across the
codebase (``PersonInfo``, ``PersonData``, ``CandidatePerson``,
``PersonSearchResult``, ``PersonLookupResult``, ``FamilyMember``,
``GedcomPerson``, and the ad-hoc ``_create_person_dict`` factory).

Migration strategy
==================
* **PersonSummary** replaces ``PersonInfo`` (TypedDict), ``CandidatePerson``
  (dataclass), ``GedcomPerson`` (dataclass, core fields), and the plain-dict
  "unified format" in ``relationship_formatting.py``.
* **MatchSummary** replaces ``DNAMatchInfo`` (TypedDict) and ``PersonData``
  (TypedDict).
* Domain-specific types that carry extra fields (``PersonSearchResult``,
  ``PersonLookupResult``, ``FamilyMember``) **keep their own classes** but
  inherit from or compose ``PersonSummary`` and expose a
  ``.to_person_summary()`` converter.

Usage::

    from core.person_summary import PersonSummary, MatchSummary

    person = PersonSummary(name="John Smith", birth_year=1850)
    match  = MatchSummary.from_person_data(person_data_dict)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PersonSummary — core identity of a person in the tree
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PersonSummary:
    """Canonical lightweight representation of a person.

    Covers the union of fields shared by the former ``PersonInfo``,
    ``CandidatePerson``, ``GedcomPerson``, ``FamilyMember``, and the
    ad-hoc unified-format dicts used in relationship path conversion.

    All fields are optional except ``name``.
    """

    # Identity
    name: str = ""
    person_id: str | None = None
    given_name: str | None = None
    surname: str | None = None

    # Vital dates
    birth_year: int | None = None
    birth_date: str | None = None
    birth_place: str | None = None
    death_year: int | None = None
    death_date: str | None = None
    death_place: str | None = None

    # Demographics
    gender: str | None = None  # "M", "F", or None
    is_living: bool | None = None

    # Relationship context (populated when part of a path)
    relationship: str | None = None  # e.g. "father", "2nd cousin"

    # Source / confidence (populated by search/lookup results)
    source: str | None = None  # "gedcom", "api", "search", …
    confidence: str | None = None  # "low", "medium", "high"
    match_score: int | None = None

    # ── Conversions ────────────────────────────────────────────────

    def to_dict(self, *, exclude_none: bool = False) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-friendly)."""
        d = asdict(self)
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d

    def to_unified_dict(self) -> dict[str, str | None]:
        """Return the 5-key "unified format" dict used by
        ``relationship_formatting`` converters.

        Keys: ``name``, ``birth_year``, ``death_year``, ``relationship``,
        ``gender``.  Years are cast to ``str`` to match the legacy contract.
        """
        return {
            "name": self.name,
            "birth_year": str(self.birth_year) if self.birth_year is not None else None,
            "death_year": str(self.death_year) if self.death_year is not None else None,
            "relationship": self.relationship,
            "gender": self.gender,
        }

    # ── Factories ─────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonSummary:
        """Create from a plain dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_unified_dict(cls, d: dict[str, str | None]) -> PersonSummary:
        """Create from the 5-key "unified format" dict.

        Parses ``birth_year`` / ``death_year`` strings back to ``int``.
        """

        def _safe_int(v: str | None) -> int | None:
            if v is None:
                return None
            try:
                return int(v)
            except (ValueError, TypeError):
                return None

        return cls(
            name=d.get("name") or "",
            birth_year=_safe_int(d.get("birth_year")),
            death_year=_safe_int(d.get("death_year")),
            relationship=d.get("relationship"),
            gender=d.get("gender"),
        )


# ---------------------------------------------------------------------------
# MatchSummary — a DNA match with shared-DNA metrics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MatchSummary:
    """Canonical lightweight representation of a DNA match.

    Replaces the former ``DNAMatchInfo`` TypedDict and ``PersonData``
    TypedDict.  Contains matched-person identity + shared-DNA metrics +
    profile / tree metadata.
    """

    # DNA identity
    uuid: str = ""
    name: str = ""

    # Shared DNA
    shared_cm: float = 0.0
    shared_segments: int = 0
    relationship_range: str | None = None

    # Ancestry profile metadata
    profile_id: str | None = None
    administrator_profile_id: str | None = None
    has_tree: bool = False
    tree_size: int | None = None

    # ── Conversions ────────────────────────────────────────────────

    def to_dict(self, *, exclude_none: bool = False) -> dict[str, Any]:
        """Serialize to a plain dict."""
        d = asdict(self)
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d

    def to_person_summary(self) -> PersonSummary:
        """Promote to a ``PersonSummary`` (loses DNA metrics)."""
        return PersonSummary(name=self.name)

    # ── Factories ─────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchSummary:
        """Create from a plain dict, ignoring unknown keys.

        Handles both ``tree_size`` and ``tree_person_count`` as aliases.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        cleaned = {k: v for k, v in data.items() if k in known}
        # Handle legacy alias
        if "tree_person_count" in data and "tree_size" not in cleaned:
            cleaned["tree_size"] = data["tree_person_count"]
        return cls(**cleaned)

    @classmethod
    def from_person_data(cls, data: dict[str, Any]) -> MatchSummary:
        """Create from the former ``PersonData`` TypedDict layout."""
        return cls(
            uuid=data.get("uuid", ""),
            name=data.get("name", ""),
            profile_id=data.get("profile_id"),
            administrator_profile_id=data.get("administrator_profile_id"),
            shared_cm=data.get("shared_cm", 0.0),
            shared_segments=data.get("shared_segments", 0),
            relationship_range=data.get("relationship_range"),
            has_tree=data.get("has_tree", False),
            tree_size=data.get("tree_person_count") or data.get("tree_size"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

from testing.test_framework import TestSuite, create_standard_test_runner


def _test_person_summary_creation() -> bool:
    """PersonSummary basic construction and defaults."""
    p = PersonSummary(name="John Smith", birth_year=1850, gender="M")
    assert p.name == "John Smith"
    assert p.birth_year == 1850
    assert p.gender == "M"
    assert p.death_year is None
    assert p.person_id is None
    assert p.is_living is None
    return True


def _test_person_summary_to_dict() -> bool:
    """to_dict() and exclude_none filtering."""
    p = PersonSummary(name="Jane", birth_year=1900, death_year=1980)
    d = p.to_dict()
    assert d["name"] == "Jane"
    assert d["birth_year"] == 1900
    assert d["death_year"] == 1980
    assert d["gender"] is None  # included when exclude_none=False

    d2 = p.to_dict(exclude_none=True)
    assert "gender" not in d2
    assert d2["name"] == "Jane"
    return True


def _test_person_summary_to_unified_dict() -> bool:
    """to_unified_dict() produces the 5-key format with string years."""
    p = PersonSummary(name="Ancestor", birth_year=1820, death_year=1890, gender="F", relationship="mother")
    u = p.to_unified_dict()
    assert u == {
        "name": "Ancestor",
        "birth_year": "1820",
        "death_year": "1890",
        "relationship": "mother",
        "gender": "F",
    }
    # None years → None strings
    p2 = PersonSummary(name="Unknown")
    u2 = p2.to_unified_dict()
    assert u2["birth_year"] is None
    assert u2["death_year"] is None
    return True


def _test_person_summary_from_dict() -> bool:
    """from_dict() ignores unknown keys."""
    d = {"name": "Test", "birth_year": 1900, "unknown_field": 42, "gender": "M"}
    p = PersonSummary.from_dict(d)
    assert p.name == "Test"
    assert p.birth_year == 1900
    assert p.gender == "M"
    assert not hasattr(p, "unknown_field")
    return True


def _test_person_summary_from_unified_dict() -> bool:
    """from_unified_dict() parses string years to int."""
    u = {"name": "Grandpa", "birth_year": "1850", "death_year": "1920", "relationship": "grandfather", "gender": "M"}
    p = PersonSummary.from_unified_dict(u)
    assert p.name == "Grandpa"
    assert p.birth_year == 1850
    assert p.death_year == 1920
    assert p.relationship == "grandfather"
    assert p.gender == "M"

    # None and invalid years
    u2 = {"name": "X", "birth_year": None, "death_year": "invalid", "relationship": None, "gender": None}
    p2 = PersonSummary.from_unified_dict(u2)
    assert p2.birth_year is None
    assert p2.death_year is None
    return True


def _test_match_summary_creation() -> bool:
    """MatchSummary basic construction."""
    m = MatchSummary(uuid="ABC-123", name="Cousin", shared_cm=250.5, shared_segments=12)
    assert m.uuid == "ABC-123"
    assert m.shared_cm == 250.5
    assert m.shared_segments == 12
    assert m.has_tree is False
    assert m.tree_size is None
    return True


def _test_match_summary_from_dict() -> bool:
    """from_dict() handles tree_person_count alias."""
    d = {"uuid": "U1", "name": "Match", "shared_cm": 100.0, "tree_person_count": 55, "extra": True}
    m = MatchSummary.from_dict(d)
    assert m.uuid == "U1"
    assert m.tree_size == 55
    assert m.shared_cm == 100.0
    return True


def _test_match_summary_from_person_data() -> bool:
    """from_person_data() mirrors the former PersonData TypedDict."""
    pd = {
        "uuid": "XYZ",
        "name": "Person Data Match",
        "profile_id": "prof_1",
        "administrator_profile_id": "admin_1",
        "shared_cm": 320.0,
        "shared_segments": 15,
        "relationship_range": "2nd cousin",
        "has_tree": True,
        "tree_person_count": 200,
    }
    m = MatchSummary.from_person_data(pd)
    assert m.uuid == "XYZ"
    assert m.name == "Person Data Match"
    assert m.profile_id == "prof_1"
    assert m.administrator_profile_id == "admin_1"
    assert m.shared_cm == 320.0
    assert m.shared_segments == 15
    assert m.relationship_range == "2nd cousin"
    assert m.has_tree is True
    assert m.tree_size == 200
    return True


def _test_match_summary_to_person_summary() -> bool:
    """to_person_summary() converts to PersonSummary retaining name."""
    m = MatchSummary(uuid="U", name="Test Name", shared_cm=50.0)
    p = m.to_person_summary()
    assert isinstance(p, PersonSummary)
    assert p.name == "Test Name"
    return True


def _test_roundtrip_unified_dict() -> bool:
    """PersonSummary → unified dict → PersonSummary roundtrip preserves data."""
    original = PersonSummary(name="Roundtrip", birth_year=1900, death_year=1975, relationship="uncle", gender="M")
    unified = original.to_unified_dict()
    restored = PersonSummary.from_unified_dict(unified)
    assert restored.name == original.name
    assert restored.birth_year == original.birth_year
    assert restored.death_year == original.death_year
    assert restored.relationship == original.relationship
    assert restored.gender == original.gender
    return True


def person_summary_module_tests() -> bool:
    """Run all PersonSummary module tests."""
    suite = TestSuite("PersonSummary & MatchSummary", "core/person_summary.py")

    suite.run_test("PersonSummary creation and defaults", _test_person_summary_creation,
                   "Construct PersonSummary and verify fields")
    suite.run_test("PersonSummary to_dict and exclude_none", _test_person_summary_to_dict,
                   "Serialize to dict with optional None filtering")
    suite.run_test("PersonSummary to_unified_dict format", _test_person_summary_to_unified_dict,
                   "5-key unified format with string years")
    suite.run_test("PersonSummary from_dict ignores unknowns", _test_person_summary_from_dict,
                   "Factory method drops unrecognized keys")
    suite.run_test("PersonSummary from_unified_dict parses years", _test_person_summary_from_unified_dict,
                   "Parse string years back to int, handle invalid")
    suite.run_test("MatchSummary creation and defaults", _test_match_summary_creation,
                   "Construct MatchSummary and verify DNA fields")
    suite.run_test("MatchSummary from_dict with alias", _test_match_summary_from_dict,
                   "from_dict handles tree_person_count alias")
    suite.run_test("MatchSummary from_person_data", _test_match_summary_from_person_data,
                   "Factory mirrors former PersonData TypedDict")
    suite.run_test("MatchSummary to_person_summary", _test_match_summary_to_person_summary,
                   "Convert to PersonSummary retaining name")
    suite.run_test("Roundtrip unified dict", _test_roundtrip_unified_dict,
                   "PersonSummary → unified dict → PersonSummary preserves data")

    return bool(suite.finish_suite())


run_comprehensive_tests = create_standard_test_runner(person_summary_module_tests)

if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
