#!/usr/bin/env python3
"""
AI Draft Moderator - Secondary AI Review Layer

Provides independent AI moderation of draft messages BEFORE human review.
Uses a different AI provider (or different prompt context) to catch:
- Factual errors or hallucinations
- Tone/appropriateness issues
- Context inversions (explaining their ancestors to them)
- Self-message attempts
- Misleading or confusing content

The moderator acts as a quality gate between AI generation and human review,
reducing the cognitive load on the human reviewer.

Architecture:
    [AI Generator] → [Draft] → [AI Moderator] → [Modified Draft] → [Human Review]

The moderator can:
- APPROVE: Draft is good, pass to human review as-is
- MODIFY: Fix issues and pass modified version to human review
- FLAG: Mark for priority human review with warnings
- REJECT: Block draft entirely (e.g., self-message, hostile content)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from core.session_manager import SessionManager

logger = logging.getLogger(__name__)


class ModerationAction(Enum):
    """Actions the moderator can take."""

    APPROVE = "APPROVE"  # Pass through unchanged
    MODIFY = "MODIFY"  # Fix issues, pass modified version
    FLAG = "FLAG"  # Pass with warnings for human attention
    REJECT = "REJECT"  # Block entirely


class ModerationCategory(Enum):
    """Categories of issues detected."""

    NONE = "NONE"
    FACTUAL_ERROR = "FACTUAL_ERROR"  # Incorrect or invented facts
    TONE_ISSUE = "TONE_ISSUE"  # Inappropriate tone
    CONTEXT_INVERSION = "CONTEXT_INVERSION"  # Explaining their info to them
    SELF_MESSAGE = "SELF_MESSAGE"  # Sending to self
    CONFUSION = "CONFUSION"  # Confusing or unclear content
    TOO_LONG = "TOO_LONG"  # Exceeds reasonable length
    MISSING_QUESTION = "MISSING_QUESTION"  # No follow-up question
    GENERIC = "GENERIC"  # Too generic, not personalized


@dataclass
class ModerationResult:
    """Result of AI moderation."""

    action: ModerationAction
    original_draft: str
    moderated_draft: str  # Same as original if APPROVE, modified if MODIFY
    issues_found: list[str] = field(default_factory=list)
    categories: list[ModerationCategory] = field(default_factory=list)
    confidence: int = 100  # Moderator's confidence in its assessment
    reasoning: str = ""
    flags_for_human: list[str] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return self.action == ModerationAction.MODIFY

    @property
    def needs_human_attention(self) -> bool:
        return self.action in {ModerationAction.FLAG, ModerationAction.REJECT}

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "was_modified": self.was_modified,
            "issues_found": self.issues_found,
            "categories": [c.value for c in self.categories],
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "flags_for_human": self.flags_for_human,
        }


# The moderation prompt - intentionally simple and focused
MODERATION_PROMPT = """You are a quality control reviewer for genealogy messages. Review this draft and fix any issues.

DRAFT TO REVIEW:
{draft_message}

CONTEXT:
- Sender: {sender_name} (the tree owner, writing in first person)
- Recipient: {recipient_name}
- Recipient's profile ID: {recipient_profile_id}
- Sender's profile ID: {sender_profile_id}

KNOWN FACTS FROM SENDER'S TREE:
{verified_facts}

CHECK FOR THESE ISSUES:

1. SELF-MESSAGE: Is sender_profile_id == recipient_profile_id? → REJECT

2. CONTEXT INVERSION: Is the draft explaining the RECIPIENT's own ancestors TO them?
   BAD: "Your grandfather John was born in 1920" (they know this!)
   GOOD: "My grandfather John was born in 1920"
   → If found, MODIFY to share OUR info instead

3. FACTUAL ERRORS: Does the draft claim facts NOT in KNOWN FACTS?
   → If found, MODIFY to remove or qualify with "I believe" / "possibly"

4. LENGTH: Is it over 300 words? → MODIFY to be concise (150-250 words ideal)

5. MISSING QUESTION: Does it end with a follow-up question?
   → If missing, MODIFY to add one

6. TOO GENERIC: Is it just pleasantries with no genealogical substance?
   → FLAG for human review

7. TONE: Is it warm and collaborative, not pushy or demanding?
   → If pushy, MODIFY tone

OUTPUT (JSON only):
{
  "action": "APPROVE" | "MODIFY" | "FLAG" | "REJECT",
  "moderated_draft": "The corrected message text (or original if APPROVE)",
  "issues_found": ["Issue 1", "Issue 2"],
  "reasoning": "Brief explanation",
  "flags_for_human": ["Things human should double-check"]
}"""


class DraftModerator:
    """
    AI-powered draft moderation service.

    Uses a secondary AI call to review and potentially modify drafts
    before they reach the human review queue.
    """

    def __init__(self, session_manager: SessionManager | None = None) -> None:
        """Initialize the moderator."""
        self._session_manager = session_manager
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def moderate(
        self,
        draft_message: str,
        sender_name: str,
        recipient_name: str,
        sender_profile_id: str,
        recipient_profile_id: str,
        verified_facts: str = "",
    ) -> ModerationResult:
        """
        Moderate a draft message using AI.

        Args:
            draft_message: The draft to review
            sender_name: Name of the sender (tree owner)
            recipient_name: Name of the recipient
            sender_profile_id: Profile ID of sender
            recipient_profile_id: Profile ID of recipient
            verified_facts: Known facts from sender's tree

        Returns:
            ModerationResult with action and potentially modified draft
        """
        # Quick self-message check (no AI needed)
        if sender_profile_id and recipient_profile_id and str(sender_profile_id) == str(recipient_profile_id):
            self._logger.warning("🚫 BLOCKED: Self-message detected by moderator")
            return ModerationResult(
                action=ModerationAction.REJECT,
                original_draft=draft_message,
                moderated_draft="",
                issues_found=["Self-message: sender and recipient are the same person"],
                categories=[ModerationCategory.SELF_MESSAGE],
                confidence=100,
                reasoning="Cannot send message to yourself",
            )

        # Build the moderation prompt
        prompt = MODERATION_PROMPT.format(
            draft_message=draft_message,
            sender_name=sender_name,
            recipient_name=recipient_name,
            sender_profile_id=sender_profile_id or "unknown",
            recipient_profile_id=recipient_profile_id or "unknown",
            verified_facts=verified_facts or "No verified facts provided",
        )

        # Call AI for moderation
        try:
            response = self._call_moderation_ai(prompt)
            return self._parse_moderation_response(response, draft_message)
        except Exception as e:
            self._logger.error(f"Moderation AI call failed: {e}")
            # On error, flag for human review
            return ModerationResult(
                action=ModerationAction.FLAG,
                original_draft=draft_message,
                moderated_draft=draft_message,
                issues_found=[f"Moderation failed: {e}"],
                categories=[],
                confidence=0,
                reasoning="AI moderation failed, requires human review",
                flags_for_human=["Moderation AI unavailable - manual review required"],
            )

    def _call_moderation_ai(self, prompt: str) -> str:
        """Call the AI provider for moderation."""
        from ai.ai_interface import call_ai_with_prompt

        # Use a simpler, faster model if available
        response = call_ai_with_prompt(
            prompt=prompt,
            session_manager=self._session_manager,
            max_tokens=1000,
            temperature=0.3,  # Lower temperature for more consistent moderation
        )

        return response or ""

    def _parse_moderation_response(self, response: str, original_draft: str) -> ModerationResult:
        """Parse the AI moderation response."""
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            action_str = data.get("action", "FLAG").upper()
            action = (
                ModerationAction[action_str] if action_str in ModerationAction.__members__ else ModerationAction.FLAG
            )

            moderated_draft = data.get("moderated_draft", original_draft)
            if not moderated_draft or not moderated_draft.strip():
                moderated_draft = original_draft

            issues = data.get("issues_found", [])
            if isinstance(issues, str):
                issues = [issues]

            return ModerationResult(
                action=action,
                original_draft=original_draft,
                moderated_draft=moderated_draft,
                issues_found=issues,
                categories=self._infer_categories(issues),
                confidence=data.get("confidence", 80),
                reasoning=data.get("reasoning", ""),
                flags_for_human=data.get("flags_for_human", []),
            )

        except Exception as e:
            self._logger.warning(f"Failed to parse moderation response: {e}")
            # Return FLAG action on parse failure
            return ModerationResult(
                action=ModerationAction.FLAG,
                original_draft=original_draft,
                moderated_draft=original_draft,
                issues_found=[f"Parse error: {e}"],
                categories=[],
                confidence=50,
                reasoning="Could not parse AI moderation response",
                flags_for_human=["Moderation response unclear - manual review recommended"],
            )

    # Mapping of keywords to categories for _infer_categories
    _CATEGORY_KEYWORDS: ClassVar[list[tuple[ModerationCategory, tuple[str, ...]]]] = [
        (ModerationCategory.SELF_MESSAGE, ("self-message", "same person")),
        (ModerationCategory.CONTEXT_INVERSION, ("context inversion", "their ancestor")),
        (ModerationCategory.FACTUAL_ERROR, ("factual", "incorrect", "invented")),
        (ModerationCategory.TONE_ISSUE, ("tone", "pushy")),
        (ModerationCategory.TOO_LONG, ("length", "too long")),
        (ModerationCategory.MISSING_QUESTION, ("question",)),
        (ModerationCategory.GENERIC, ("generic",)),
    ]

    @staticmethod
    def _infer_categories(issues: list[str]) -> list[ModerationCategory]:
        """Infer issue categories from issue descriptions."""
        issues_lower = " ".join(issues).lower()
        categories = [
            category
            for category, keywords in DraftModerator._CATEGORY_KEYWORDS
            if any(kw in issues_lower for kw in keywords)
        ]
        return categories if categories else [ModerationCategory.NONE]


def moderate_draft(
    draft_message: str,
    sender_name: str,
    recipient_name: str,
    sender_profile_id: str,
    recipient_profile_id: str,
    verified_facts: str = "",
    session_manager: SessionManager | None = None,
) -> ModerationResult:
    """
    Convenience function to moderate a draft message.

    Args:
        draft_message: The draft to review
        sender_name: Name of the sender (tree owner)
        recipient_name: Name of the recipient
        sender_profile_id: Profile ID of sender
        recipient_profile_id: Profile ID of recipient
        verified_facts: Known facts from sender's tree
        session_manager: Optional session manager

    Returns:
        ModerationResult with action and potentially modified draft
    """
    moderator = DraftModerator(session_manager)
    return moderator.moderate(
        draft_message=draft_message,
        sender_name=sender_name,
        recipient_name=recipient_name,
        sender_profile_id=sender_profile_id,
        recipient_profile_id=recipient_profile_id,
        verified_facts=verified_facts,
    )


# === MODULE TESTS ===


def _module_tests() -> bool:
    """Run module tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("AI Draft Moderator", "ai/draft_moderator.py")
    suite.start_suite()

    # Test 1: Self-message detection
    def test_self_message_detection() -> None:
        result = moderate_draft(
            draft_message="Hello, this is a test message.",
            sender_name="Wayne Gault",
            recipient_name="Wayne Gault",
            sender_profile_id="12345",
            recipient_profile_id="12345",  # Same as sender
        )
        assert result.action == ModerationAction.REJECT, f"Expected REJECT, got {result.action}"

    suite.run_test("Self-message detection blocks draft", test_self_message_detection)

    # Test 2: ModerationResult properties
    def test_moderation_result_properties() -> None:
        result = ModerationResult(
            action=ModerationAction.MODIFY,
            original_draft="Original",
            moderated_draft="Modified",
            issues_found=["Fixed something"],
        )
        assert result.was_modified, "Expected was_modified=True"
        assert not result.needs_human_attention, "Expected needs_human_attention=False"

    suite.run_test("ModerationResult properties work correctly", test_moderation_result_properties)

    # Test 3: FLAG action needs human attention
    def test_flag_needs_attention() -> None:
        result = ModerationResult(
            action=ModerationAction.FLAG,
            original_draft="Original",
            moderated_draft="Original",
        )
        assert result.needs_human_attention, "Expected needs_human_attention=True for FLAG"

    suite.run_test("FLAG action requires human attention", test_flag_needs_attention)

    # Test 4: Category inference
    def test_category_inference() -> None:
        categories = DraftModerator._infer_categories(["Context inversion detected", "Too long"])
        assert ModerationCategory.CONTEXT_INVERSION in categories, "Expected CONTEXT_INVERSION"
        assert ModerationCategory.TOO_LONG in categories, "Expected TOO_LONG"

    suite.run_test("Category inference from issue descriptions", test_category_inference)

    # Test 5: to_dict serialization
    def test_to_dict() -> None:
        result = ModerationResult(
            action=ModerationAction.APPROVE,
            original_draft="Test",
            moderated_draft="Test",
            categories=[ModerationCategory.NONE],
        )
        d = result.to_dict()
        assert d["action"] == "APPROVE", f"Expected APPROVE, got {d['action']}"
        assert "NONE" in d["categories"], f"Expected NONE in categories, got {d['categories']}"

    suite.run_test("to_dict serialization works", test_to_dict)

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Standard test runner entry point."""
    return _module_tests()


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
