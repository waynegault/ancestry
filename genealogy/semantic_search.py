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

        candidates: list[CandidatePerson] = []
        evidence: list[EvidenceBlock] = []

        if not res.found:
            evidence.append(
                EvidenceBlock(
                    source_type="GEDCOM",
                    source_id=None,
                    summary=f"No strong tree match found for '{entity.name}'.",
                    confidence=0,
                )
            )
            return candidates, evidence

        def add_candidate_from_result(r: PersonSearchResult) -> None:
            candidates.append(
                CandidatePerson(
                    person_id=r.person_id,
                    name=r.name or entity.name,
                    birth_year=r.birth_year,
                    birth_place=r.birth_place,
                    death_year=r.death_year,
                    death_place=r.death_place,
                    match_score=r.match_score,
                    confidence=r.confidence,
                )
            )

        add_candidate_from_result(res)

        for alt in (res.alternatives or [])[: max(0, max_candidates - 1)]:
            if not isinstance(alt, dict):
                continue

            raw_score = alt.get("total_score")
            alt_score: Optional[int]
            if raw_score is None:
                alt_score = None
            else:
                try:
                    alt_score = int(raw_score)
                except (TypeError, ValueError):
                    alt_score = None
            candidates.append(
                CandidatePerson(
                    person_id=alt.get("id"),
                    name=alt.get("name") or "",
                    birth_year=alt.get("birth_year"),
                    birth_place=alt.get("birth_place"),
                    death_year=alt.get("death_year"),
                    death_place=alt.get("death_place"),
                    match_score=alt_score,
                    confidence=None,
                )
            )

        # Evidence block for the top candidate.
        top = candidates[0]
        summary_bits = [f"Tree match for '{entity.name}': {top.name}"]
        if top.birth_year:
            summary_bits.append(f"b. {top.birth_year}")
        if top.birth_place:
            summary_bits.append(str(top.birth_place))
        evidence.append(
            EvidenceBlock(
                source_type="GEDCOM",
                source_id=top.person_id,
                summary="; ".join(summary_bits),
                confidence=80 if (res.confidence == "high") else 60 if (res.confidence == "medium") else 40,
            )
        )

        return candidates[:max_candidates], evidence

    @staticmethod
    def _compose_answer(result: SemanticSearchResult) -> None:
        # Conservative answer composition: never claim certainty.
        if result.intent == SemanticSearchIntent.PERSON_LOOKUP and result.people:
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

            result.answer_draft = line
            result.confidence = 70
            return

        if result.intent == SemanticSearchIntent.RELATIONSHIP_EXPLANATION:
            result.answer_draft = (
                "I can try to explain how we might be related, but I'll need a specific ancestor name (and ideally dates/places) "
                "from your line to compare against my tree."
            )
            result.confidence = 10
            result.missing_information = ["Any ancestor name", "Approximate dates/locations"]
            return

        # Default.
        result.answer_draft = "I can help with that. If you share a person name (and ideally a birth year/location), I can check my tree and report what I find."
        result.confidence = 10
