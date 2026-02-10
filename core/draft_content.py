"""Helpers for storing internal (review-only) metadata inside draft text.

Constraint: no schema migrations. We embed internal metadata as a delimited block
inside DraftReply.content, and strip it before sending outbound messages.

This module intentionally has no SQLAlchemy dependencies.
"""


import json
from dataclasses import dataclass
from typing import Any

_INTERNAL_BLOCK_BEGIN = "\n\n---\n[INTERNAL_DRAFT_METADATA]\n"
_INTERNAL_BLOCK_END = "\n[/INTERNAL_DRAFT_METADATA]\n"

_LEGACY_RESEARCH_SUGGESTIONS_MARKER = "\n\n---\nResearch Suggestions:\n"


@dataclass(frozen=True, slots=True)
class DraftInternalMetadata:
    ai_confidence: int | None = None
    ai_reasoning: str | None = None
    context_summary: str | None = None
    research_suggestions: str | None = None
    research_metadata: dict[str, Any] | None = None


def _sanitize_field(value: str | None, *, max_len: int) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if len(cleaned) > max_len:
        return cleaned[:max_len] + "…"
    return cleaned


def _sanitize_json(value: dict[str, Any] | None, *, max_len: int) -> str | None:
    if not value:
        return None
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        rendered = str(value)
    return _sanitize_field(rendered, max_len=max_len)


def build_internal_metadata_block(
    metadata: DraftInternalMetadata,
    *,
    max_reasoning_len: int = 2000,
    max_context_len: int = 4000,
    max_research_suggestions_len: int = 4000,
    max_research_metadata_len: int = 2000,
) -> str:
    """Return a human-readable internal metadata block.

    The block is delimited by stable markers so it can be stripped reliably.
    """

    reasoning = _sanitize_field(metadata.ai_reasoning, max_len=max_reasoning_len)
    context = _sanitize_field(metadata.context_summary, max_len=max_context_len)
    research_suggestions = _sanitize_field(
        metadata.research_suggestions,
        max_len=max_research_suggestions_len,
    )
    research_metadata = _sanitize_json(
        metadata.research_metadata,
        max_len=max_research_metadata_len,
    )

    if reasoning is None and context is None and research_suggestions is None and research_metadata is None:
        return ""

    lines: list[str] = []
    if metadata.ai_confidence is not None:
        lines.append(f"ai_confidence: {int(metadata.ai_confidence)}")

    if reasoning is not None:
        lines.append("ai_reasoning:")
        lines.append(reasoning)

    if context is not None:
        lines.append("context_summary:")
        lines.append(context)

    if research_suggestions is not None:
        lines.append("research_suggestions:")
        lines.append(research_suggestions)

    if research_metadata is not None:
        lines.append("research_metadata:")
        lines.append(research_metadata)

    payload = "\n".join(lines).strip()
    if not payload:
        return ""

    return f"{_INTERNAL_BLOCK_BEGIN}{payload}{_INTERNAL_BLOCK_END}"


def append_internal_metadata(message_text: str, metadata: DraftInternalMetadata) -> str:
    """Append internal metadata to a message (idempotent for the same markers)."""

    base = (message_text or "").rstrip()
    if not base:
        return base

    # Strip any existing internal metadata first to avoid stacking blocks.
    base = strip_internal_metadata(base).rstrip()

    block = build_internal_metadata_block(metadata)
    if not block:
        return base

    return base + block


def strip_internal_metadata(text: str) -> str:
    """Remove internal metadata block from text if present."""

    if not text:
        return text

    start = text.find(_INTERNAL_BLOCK_BEGIN)
    if start == -1:
        return text

    end = text.find(_INTERNAL_BLOCK_END, start + len(_INTERNAL_BLOCK_BEGIN))
    if end == -1:
        # If block start exists but end marker is missing, be conservative and
        # strip from start to end of text.
        return text[:start].rstrip()

    return (text[:start] + text[end + len(_INTERNAL_BLOCK_END) :]).rstrip()


def strip_legacy_research_suggestions(text: str) -> str:
    """Remove legacy Action 8 research suggestion appendix from message text.

    Historically, some drafts appended a review-only section:
        ---\nResearch Suggestions:\n...
    That content should never be sent outbound.
    """

    cleaned = (text or "").rstrip()
    if not cleaned:
        return cleaned

    idx = cleaned.find(_LEGACY_RESEARCH_SUGGESTIONS_MARKER)
    if idx == -1:
        return cleaned

    return cleaned[:idx].rstrip()


def strip_review_only_content(text: str) -> str:
    """Remove any review-only additions (internal metadata + legacy appendices)."""

    cleaned = strip_internal_metadata(text)
    cleaned = strip_legacy_research_suggestions(cleaned)
    return cleaned.rstrip()


def module_tests() -> bool:
    from testing.test_framework import TestSuite

    suite = TestSuite("Draft Content Helpers", "core/draft_content.py")
    suite.start_suite()

    def test_append_and_strip_round_trip() -> None:
        text = append_internal_metadata(
            "Hello",
            DraftInternalMetadata(ai_confidence=90, ai_reasoning="Reason", context_summary="Context"),
        )
        assert "Hello" in text
        assert strip_internal_metadata(text) == "Hello"

    suite.run_test(
        "Append + strip round-trip",
        test_append_and_strip_round_trip,
        test_summary="Internal metadata is appended and reliably stripped.",
        functions_tested="append_internal_metadata, strip_internal_metadata",
        method_description="Build a message with metadata, then strip it back to the clean outbound text.",
    )

    def test_strip_missing_end_marker_is_safe() -> None:
        broken = "Hello" + _INTERNAL_BLOCK_BEGIN + "ai_confidence: 10"  # missing end marker
        assert strip_internal_metadata(broken) == "Hello"

    suite.run_test(
        "Strip handles missing end marker",
        test_strip_missing_end_marker_is_safe,
        test_summary="If the internal block end marker is missing, strip removes from the start marker onward.",
        functions_tested="strip_internal_metadata",
        method_description="Provide malformed internal block and verify safe stripping.",
    )

    def test_strip_review_only_content_strips_legacy_appendix() -> None:
        draft = "Hello there" + _LEGACY_RESEARCH_SUGGESTIONS_MARKER + "Something internal"
        assert strip_review_only_content(draft) == "Hello there"

    suite.run_test(
        "Strip review-only content removes legacy appendix",
        test_strip_review_only_content_strips_legacy_appendix,
        test_summary="Legacy research-suggestions appendices are removed for outbound safety.",
        functions_tested="strip_review_only_content, strip_legacy_research_suggestions",
        method_description="Append the legacy research suggestion section and verify it is removed.",
    )

    return suite.finish_suite()


if __name__ == "__main__":
    raise SystemExit(0 if module_tests() else 1)
