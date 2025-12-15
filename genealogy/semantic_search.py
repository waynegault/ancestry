"""Tree-aware semantic search.

This module provides a lightweight, evidence-backed Q&A retrieval layer that can
be reused by inbound processing (Action 7), outbound draft generation (Action 8),
and operator tooling.

Design goals:
- Tree-first retrieval (GEDCOM via TreeQueryService)
- No invented facts (fail closed to clarification)
- JSON-serializable structured results

Note: This initial implementation is intentionally conservative and avoids
introducing new external dependencies or vector stores.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, cast

from genealogy.tree_query_service import PersonSearchResult, TreeQueryService

logger = logging.getLogger(__name__)


class SemanticSearchIntent(str, Enum):
    PERSON_LOOKUP = "PERSON_LOOKUP"
    RELATIONSHIP_EXPLANATION = "RELATIONSHIP_EXPLANATION"
    RECORD_SUGGESTION = "RECORD_SUGGESTION"
    GENERAL_GENEALOGY_QA = "GENERAL_GENEALOGY_QA"
    CLARIFICATION_NEEDED = "CLARIFICATION_NEEDED"


@dataclass(slots=True)
class SemanticPersonEntity:
    name: str
    approx_birth_year: Optional[int] = None
    location: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "approx_birth_year": self.approx_birth_year,
            "location": self.location,
        }


@dataclass(slots=True)
class EvidenceBlock:
    source_type: str
    source_id: Optional[str]
    summary: str
    confidence: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "summary": self.summary,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class CandidatePerson:
    person_id: Optional[str]
    name: str
    birth_year: Optional[int] = None
    birth_place: Optional[str] = None
    death_year: Optional[int] = None
    death_place: Optional[str] = None
    match_score: Optional[int] = None
    confidence: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "name": self.name,
            "birth_year": self.birth_year,
            "birth_place": self.birth_place,
            "death_year": self.death_year,
            "death_place": self.death_place,
            "match_score": self.match_score,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class SemanticSearchResult:
    intent: SemanticSearchIntent
    people: list[SemanticPersonEntity] = field(default_factory=list)
    candidates: dict[str, list[CandidatePerson]] = field(default_factory=dict)
    evidence: list[EvidenceBlock] = field(default_factory=list)
    answer_draft: str = ""
    confidence: int = 0
    missing_information: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "entities": {
                "people": [p.to_dict() for p in self.people],
            },
            "candidates": {key: [c.to_dict() for c in value] for key, value in self.candidates.items()},
            "evidence": [e.to_dict() for e in self.evidence],
            "answer_draft": self.answer_draft,
            "confidence": self.confidence,
            "missing_information": list(self.missing_information),
        }

    def to_prompt_string(self) -> str:
        """Format semantic search results for inclusion in AI prompts.

        Returns a human-readable summary suitable for the genealogical_reply prompt.
        """
        lines: list[str] = []

        # Intent and confidence
        lines.append(f"Search Intent: {self.intent.value}")
        lines.append(f"Confidence: {self.confidence}%")

        # Answer draft (main result)
        if self.answer_draft:
            lines.append(f"\nPreliminary Answer: {self.answer_draft}")

        # Candidates found
        if self.candidates:
            lines.append("\nCandidates Found in Tree:")
            for name, candidate_list in self.candidates.items():
                if candidate_list:
                    lines.append(f"  {name}:")
                    for c in candidate_list[:3]:  # Limit to top 3 per name
                        details = []
                        if c.birth_year:
                            details.append(f"b. {c.birth_year}")
                        if c.birth_place:
                            details.append(f"in {c.birth_place}")
                        if c.death_year:
                            details.append(f"d. {c.death_year}")
                        if c.match_score:
                            details.append(f"score: {c.match_score}")
                        detail_str = f" ({', '.join(details)})" if details else ""
                        lines.append(f"    - {c.name}{detail_str}")

        # Evidence blocks
        if self.evidence:
            lines.append("\nEvidence:")
            for e in self.evidence[:5]:  # Limit to top 5 evidence blocks
                conf_str = f" (confidence: {e.confidence}%)" if e.confidence else ""
                lines.append(f"  [{e.source_type}]{conf_str}: {e.summary}")

        # Missing information
        if self.missing_information:
            lines.append("\nInformation Needed for Better Match:")
            for info in self.missing_information:
                lines.append(f"  - {info}")

        return "\n".join(lines) if lines else "No semantic search results."


class SemanticSearchService:
    """Minimal semantic search service (tree-first, evidence-backed)."""

    _QUESTION_PREFIXES = (
        "do you",
        "can you",
        "could you",
        "who",
        "what",
        "when",
        "where",
        "how",
        "is ",
        "are ",
    )

    _AMBIGUOUS_SCORE_GAP = 10

    def should_run(self, message_text: str) -> bool:
        text = (message_text or "").strip()
        if not text:
            return False
        lower = text.lower()
        return ("?" in text) or lower.startswith(self._QUESTION_PREFIXES)

    def search(
        self,
        query: str,
        *,
        extracted_entities: Optional[dict[str, Any]] = None,
        tree_query_service: Optional[TreeQueryService] = None,
        max_candidates: int = 3,
    ) -> SemanticSearchResult:
        intent = self._infer_intent(query)
        people = self._extract_people(query, extracted_entities)

        if intent == SemanticSearchIntent.PERSON_LOOKUP and not people:
            return SemanticSearchResult(
                intent=SemanticSearchIntent.CLARIFICATION_NEEDED,
                answer_draft="I can check my tree, but I need the person's name (and ideally a birth year or location). Who should I look for?",
                confidence=0,
                missing_information=[
                    "Person name",
                    "Approximate birth year (optional)",
                    "Location (optional)",
                ],
            )

        if not people and intent == SemanticSearchIntent.RELATIONSHIP_EXPLANATION:
            return SemanticSearchResult(
                intent=SemanticSearchIntent.CLARIFICATION_NEEDED,
                answer_draft="I can explain how we might be related, but I need at least one person name from your line (or a key ancestor surname/location).",
                confidence=0,
                missing_information=[
                    "Any ancestor name from your side",
                    "Approximate dates/locations",
                ],
            )

        service = tree_query_service or TreeQueryService()
        result = SemanticSearchResult(intent=intent, people=people)

        for person_entity in people:
            candidates, evidence = self._lookup_person(service, person_entity, max_candidates=max_candidates)
            result.candidates[person_entity.name] = candidates
            result.evidence.extend(evidence)

        self._compose_answer(result)
        return result

    @staticmethod
    def persist_jsonl(
        *,
        payload: dict[str, Any],
        person_id: Optional[int],
        sender_id: str,
        conversation_id: str,
    ) -> None:
        """Append a semantic search record for later review.

        This is intentionally file-based (jsonl) to avoid schema churn.
        """
        try:
            record = {
                "ts": time.time(),
                "person_id": person_id,
                "sender_id": sender_id,
                "conversation_id": conversation_id,
                "semantic_search": payload,
            }
            out_path = Path("Logs") / "semantic_search_results.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.debug("Failed to persist semantic search jsonl: %s", exc)

    @staticmethod
    def _extract_people_from_entities(extracted_entities: dict[str, Any]) -> list[SemanticPersonEntity]:
        data_any = extracted_entities.get("extracted_data")
        if not isinstance(data_any, dict):
            return []

        mentioned = cast(dict[str, Any], data_any).get("mentioned_people")
        if not isinstance(mentioned, list):
            return []

        out: list[SemanticPersonEntity] = []
        for item_any in mentioned:
            if not isinstance(item_any, dict):
                continue
            item = cast(dict[str, Any], item_any)

            name_raw = item.get("name")
            name = str(name_raw).strip() if name_raw else ""
            if not name:
                continue

            by_raw = item.get("birth_year")
            by_text = str(by_raw).strip() if by_raw is not None else ""
            loc_raw = item.get("birth_place") or item.get("death_place")
            loc_text = str(loc_raw).strip() if loc_raw else None

            out.append(
                SemanticPersonEntity(
                    name=name,
                    approx_birth_year=int(by_text) if by_text.isdigit() else None,
                    location=loc_text,
                )
            )

        return out

    @staticmethod
    def _infer_intent(query: str) -> SemanticSearchIntent:
        q = (query or "").strip().lower()
        if not q:
            return SemanticSearchIntent.CLARIFICATION_NEEDED
        if "related" in q or "relationship" in q or "how are we" in q:
            return SemanticSearchIntent.RELATIONSHIP_EXPLANATION
        if "record" in q or "certificate" in q or "census" in q or "obituary" in q:
            return SemanticSearchIntent.RECORD_SUGGESTION
        if "in your tree" in q or "do you have" in q or "have you" in q:
            return SemanticSearchIntent.PERSON_LOOKUP
        if "?" in q:
            return SemanticSearchIntent.GENERAL_GENEALOGY_QA
        return SemanticSearchIntent.GENERAL_GENEALOGY_QA

    @staticmethod
    def _extract_people(query: str, extracted_entities: Optional[dict[str, Any]]) -> list[SemanticPersonEntity]:
        # Prefer upstream extracted entities (LLM-based) when available.
        if extracted_entities:
            extracted = SemanticSearchService._extract_people_from_entities(extracted_entities)
            if extracted:
                return extracted

        # Fallback heuristic: try to capture "First Last" style names.
        # We intentionally require 2+ capitalized tokens to avoid grabbing sentence starters.
        candidates = re.findall(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query or "")
        unique: list[str] = []
        for cand in candidates:
            cand_norm = cand.strip()
            if cand_norm and cand_norm not in unique:
                unique.append(cand_norm)

        years = re.findall(r"\b(18\d{2}|19\d{2}|20\d{2})\b", query or "")
        approx_year = int(years[0]) if years else None

        return [SemanticPersonEntity(name=n, approx_birth_year=approx_year) for n in unique[:3]]

    @staticmethod
    def _confidence_to_score(confidence: Optional[str]) -> int:
        if confidence == "high":
            return 80
        if confidence == "medium":
            return 60
        return 40

    @staticmethod
    def _parse_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _candidate_from_result(entity: SemanticPersonEntity, r: PersonSearchResult) -> CandidatePerson:
        return CandidatePerson(
            person_id=r.person_id,
            name=r.name or entity.name,
            birth_year=r.birth_year,
            birth_place=r.birth_place,
            death_year=r.death_year,
            death_place=r.death_place,
            match_score=r.match_score,
            confidence=r.confidence,
        )

    @staticmethod
    def _candidates_from_alternatives(alternatives: Any, *, max_count: int) -> list[CandidatePerson]:
        if not alternatives or max_count <= 0:
            return []
        if not isinstance(alternatives, list):
            return []

        out: list[CandidatePerson] = []
        for alt in alternatives[:max_count]:
            if not isinstance(alt, dict):
                continue

            alt_dict = cast(dict[str, Any], alt)

            out.append(
                CandidatePerson(
                    person_id=alt_dict.get("id"),
                    name=alt_dict.get("name") or "",
                    birth_year=alt_dict.get("birth_year"),
                    birth_place=alt_dict.get("birth_place"),
                    death_year=alt_dict.get("death_year"),
                    death_place=alt_dict.get("death_place"),
                    match_score=SemanticSearchService._parse_optional_int(alt_dict.get("total_score")),
                    confidence=None,
                )
            )

        return out

    @staticmethod
    def _evidence_for_top(
        entity: SemanticPersonEntity, top: CandidatePerson, *, confidence_score: int
    ) -> EvidenceBlock:
        summary_bits = [f"Tree match for '{entity.name}': {top.name}"]
        if top.birth_year:
            summary_bits.append(f"b. {top.birth_year}")
        if top.birth_place:
            summary_bits.append(str(top.birth_place))

        return EvidenceBlock(
            source_type="GEDCOM",
            source_id=top.person_id,
            summary="; ".join(summary_bits),
            confidence=confidence_score,
        )

    @staticmethod
    def _lookup_person(
        service: TreeQueryService,
        entity: SemanticPersonEntity,
        *,
        max_candidates: int,
    ) -> tuple[list[CandidatePerson], list[EvidenceBlock]]:
        res: PersonSearchResult = service.find_person(
            entity.name,
            approx_birth_year=entity.approx_birth_year,
            location=entity.location,
            max_results=max(5, max_candidates),
        )

        if not res.found:
            return (
                [],
                [
                    EvidenceBlock(
                        source_type="GEDCOM",
                        source_id=None,
                        summary=f"No strong tree match found for '{entity.name}'.",
                        confidence=0,
                    )
                ],
            )

        top_candidate = SemanticSearchService._candidate_from_result(entity, res)
        alt_candidates = SemanticSearchService._candidates_from_alternatives(
            res.alternatives,
            max_count=max(0, max_candidates - 1),
        )

        candidates = [top_candidate, *alt_candidates]
        confidence_score = SemanticSearchService._confidence_to_score(res.confidence)
        evidence = [SemanticSearchService._evidence_for_top(entity, top_candidate, confidence_score=confidence_score)]

        return candidates[:max_candidates], evidence

    @staticmethod
    def _compose_answer(result: SemanticSearchResult) -> None:
        # Conservative answer composition: never claim certainty.
        if result.intent == SemanticSearchIntent.PERSON_LOOKUP and result.people:
            SemanticSearchService._compose_person_lookup_answer(result)
            return

        if result.intent == SemanticSearchIntent.RELATIONSHIP_EXPLANATION:
            result.answer_draft = (
                "I can try to explain how we might be related, but I'll need a specific ancestor name (and ideally dates/places) "
                "from your line to compare against my tree."
            )
            result.confidence = 10
            result.missing_information = ["Any ancestor name", "Approximate dates/locations"]
            return

        result.answer_draft = (
            "I can help with that. If you share a person name (and ideally a birth year/location), "
            "I can check my tree and report what I find."
        )
        result.confidence = 10

    @staticmethod
    def _compose_person_lookup_answer(result: SemanticSearchResult) -> None:
        first = result.people[0]
        matches = result.candidates.get(first.name) or []
        if not matches:
            result.answer_draft = (
                f"I checked my tree and couldn't find a strong match for {first.name}. "
                "If you can share an approximate birth year, spouse, or a place (county/state), I can look again."
            )
            result.confidence = 20
            result.missing_information = [
                "Approximate birth year",
                "Spouse/parent names",
                "Location (county/state)",
            ]
            return

        if SemanticSearchService._is_ambiguous_candidates(matches):
            first_two = matches[:2]

            def _candidate_summary(candidate: CandidatePerson) -> str:
                details: list[str] = []
                if candidate.birth_year:
                    details.append(f"b. {candidate.birth_year}")
                if candidate.birth_place:
                    details.append(f"in {candidate.birth_place}")
                return f"{candidate.name} ({', '.join(details)})" if details else candidate.name

            result.intent = SemanticSearchIntent.CLARIFICATION_NEEDED
            result.answer_draft = (
                f"I found a couple possible matches for {first.name} in my tree: "
                f"1) {_candidate_summary(first_two[0])} "
                f"2) {_candidate_summary(first_two[1])}. "
                "Do you know an exact birth year/place, or a spouse/parent name, so I can confirm the right one?"
            )
            result.confidence = 35
            result.missing_information = [
                "Exact birth year",
                "Birth place (county/state)",
                "Spouse/parent names",
            ]
            return

        top = matches[0]
        line = f"I may have {first.name} in my tree as {top.name}"
        details: list[str] = []
        if top.birth_year:
            details.append(f"born {top.birth_year}")
        if top.birth_place:
            details.append(f"in {top.birth_place}")
        if details:
            line += " (" + ", ".join(details) + ")"
        line += ". Does that match what you know?"

        evidence_conf = next((e.confidence for e in result.evidence if e.source_id == top.person_id), None)
        result.confidence = int(max(20, min(80, evidence_conf if evidence_conf is not None else 50)))
        if result.confidence < 60:
            line += " If you can share an approximate birth year or place, I can confirm."

        result.answer_draft = line

    @staticmethod
    def _is_ambiguous_candidates(matches: list[CandidatePerson]) -> bool:
        if len(matches) < 2:
            return False

        top, second = matches[0], matches[1]

        if top.match_score is None or second.match_score is None:
            return True

        return abs(top.match_score - second.match_score) <= SemanticSearchService._AMBIGUOUS_SCORE_GAP


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def module_tests() -> bool:
    """Module-specific tests for SemanticSearchService."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Semantic Search Service", "genealogy/semantic_search.py")
    suite.start_suite()

    def test_should_run() -> None:
        svc = SemanticSearchService()
        assert svc.should_run("Who is John Doe?") is True
        assert svc.should_run("Hello there") is False

    suite.run_test(
        "should_run heuristic",
        test_should_run,
        test_summary="Detects question-like messages",
        functions_tested="SemanticSearchService.should_run",
        method_description="Check '?' and prefix heuristics",
    )

    def test_person_lookup_missing_name() -> None:
        svc = SemanticSearchService()
        res = svc.search("Do you have this person in your tree?")
        assert res.intent == SemanticSearchIntent.CLARIFICATION_NEEDED
        assert "Person name" in res.missing_information

    suite.run_test(
        "clarification when missing person name",
        test_person_lookup_missing_name,
        test_summary="Person lookup without a name fails closed",
        functions_tested="SemanticSearchService.search",
        method_description="Infer PERSON_LOOKUP but require a name",
    )

    def test_candidate_retrieval_and_scoring() -> None:
        class StubTreeQueryService:
            @staticmethod
            def find_person(
                name: str,
                *,
                approx_birth_year: Optional[int] = None,
                location: Optional[str] = None,
                max_results: int = 5,
            ) -> PersonSearchResult:
                _ = (name, approx_birth_year, location, max_results)
                return PersonSearchResult(
                    found=True,
                    person_id="P1",
                    name="John Doe",
                    birth_year=1900,
                    birth_place="Ohio",
                    confidence="medium",
                    match_score=88,
                    alternatives=[
                        {
                            "id": "P2",
                            "name": "John Doe (alt)",
                            "birth_year": 1901,
                            "birth_place": "Ohio",
                            "total_score": "77",
                        }
                    ],
                )

        svc = SemanticSearchService()
        res = svc.search(
            "Do you have John Doe in your tree? He was born 1900 in Ohio.",
            tree_query_service=cast(TreeQueryService, StubTreeQueryService()),
            max_candidates=2,
        )

        assert res.intent == SemanticSearchIntent.PERSON_LOOKUP
        assert res.people and res.people[0].name == "John Doe"
        assert "John Doe" in res.candidates
        assert len(res.candidates["John Doe"]) == 2
        assert res.candidates["John Doe"][1].match_score == 77
        assert "I may have John Doe in my tree" in res.answer_draft
        assert res.confidence == 60

    suite.run_test(
        "candidate retrieval + alt score parsing",
        test_candidate_retrieval_and_scoring,
        test_summary="Returns top candidates and parses alt scores safely",
        functions_tested="SemanticSearchService._lookup_person, SemanticSearchService.search",
        method_description="Stub tree query to validate ranking and score parsing",
    )

    def test_ambiguity_produces_clarification() -> None:
        class StubTreeQueryService:
            @staticmethod
            def find_person(
                name: str,
                *,
                approx_birth_year: Optional[int] = None,
                location: Optional[str] = None,
                max_results: int = 5,
            ) -> PersonSearchResult:
                _ = (name, approx_birth_year, location, max_results)
                return PersonSearchResult(
                    found=True,
                    person_id="P1",
                    name="Mary Smith",
                    birth_year=1900,
                    birth_place="Ohio",
                    confidence="medium",
                    match_score=80,
                    alternatives=[
                        {
                            "id": "P2",
                            "name": "Mary Smith (alt)",
                            "birth_year": 1901,
                            "birth_place": "Ohio",
                            "total_score": "72",
                        }
                    ],
                )

        svc = SemanticSearchService()
        res = svc.search(
            "Do you have Mary Smith in your tree?",
            tree_query_service=cast(TreeQueryService, StubTreeQueryService()),
            max_candidates=2,
        )

        assert res.intent == SemanticSearchIntent.CLARIFICATION_NEEDED
        assert res.missing_information
        assert "Spouse/parent names" in res.missing_information
        assert "possible matches" in res.answer_draft

    suite.run_test(
        "ambiguity prompts clarification",
        test_ambiguity_produces_clarification,
        test_summary="Close candidates trigger clarification questions",
        functions_tested="SemanticSearchService._compose_person_lookup_answer",
        method_description="When candidate scores are close, fail closed and ask for disambiguating details",
    )

    def test_to_prompt_string() -> None:
        """Test SemanticSearchResult.to_prompt_string() formatting."""
        result = SemanticSearchResult(
            intent=SemanticSearchIntent.PERSON_LOOKUP,
            people=[SemanticPersonEntity(name="John Doe", approx_birth_year=1900, location="Ohio")],
            candidates={
                "John Doe": [
                    CandidatePerson(
                        person_id="P1",
                        name="John Doe",
                        birth_year=1900,
                        birth_place="Ohio",
                        death_year=1970,
                        match_score=85,
                    )
                ]
            },
            evidence=[EvidenceBlock(source_type="GEDCOM", source_id="P1", summary="Born 1900 in Ohio", confidence=90)],
            answer_draft="I may have John Doe in my tree as John Doe (born 1900, in Ohio).",
            confidence=75,
            missing_information=[],
        )
        prompt_str = result.to_prompt_string()
        assert "Search Intent: PERSON_LOOKUP" in prompt_str
        assert "Confidence: 75%" in prompt_str
        assert "Preliminary Answer:" in prompt_str
        assert "I may have John Doe" in prompt_str
        assert "Candidates Found in Tree:" in prompt_str
        assert "John Doe" in prompt_str
        assert "b. 1900" in prompt_str
        assert "in Ohio" in prompt_str
        assert "Evidence:" in prompt_str
        assert "[GEDCOM]" in prompt_str
        assert "Born 1900 in Ohio" in prompt_str

    suite.run_test(
        "to_prompt_string formatting",
        test_to_prompt_string,
        test_summary="Formats SemanticSearchResult for AI prompts",
        functions_tested="SemanticSearchResult.to_prompt_string",
        method_description="Validate prompt string contains all key sections",
    )

    return suite.finish_suite()


from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
