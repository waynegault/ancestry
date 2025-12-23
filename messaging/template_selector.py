"""
Template Selector for Unified Message Send Orchestrator.

This module centralizes all message template selection logic that determines
which template variant to use for a given person and context. Extracted from
actions/action8_messaging.py as part of the unified messaging refactoring (Phase 2.2).

Responsibilities:
    - Template variant selection (Confident/Exploratory/Short)
    - A/B testing template assignment
    - Confidence-based template selection
    - Template selection tracking/logging

Usage:
    from messaging.template_selector import TemplateSelector

    selector = TemplateSelector(message_templates)
    result = selector.select_template("Out_Tree-Initial", person, context)
    print(f"Selected template: {result.template_key}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Optional

from testing.test_framework import TestSuite

if TYPE_CHECKING:
    from core.database import DnaMatch, FamilyTree, Person

logger = logging.getLogger(__name__)


# ============================================================
# Enums
# ============================================================


class TemplateVariant(Enum):
    """Variants of a message template."""

    BASE = auto()  # Original template
    SHORT = auto()  # Shorter version
    CONFIDENT = auto()  # High-confidence relationship
    EXPLORATORY = auto()  # Distant/uncertain relationship
    AB_TEST_A = auto()  # A/B test variant A
    AB_TEST_B = auto()  # A/B test variant B


class SelectionReason(Enum):
    """Reasons for template selection."""

    DEFAULT = auto()  # Default selection
    CONFIDENCE_HIGH = auto()  # High relationship confidence
    CONFIDENCE_LOW = auto()  # Low relationship confidence
    DISTANT_RELATIONSHIP = auto()  # Distant family relationship
    AB_TEST_ASSIGNMENT = auto()  # A/B test assignment
    SHORT_PREFERRED = auto()  # Short variant preferred
    FALLBACK = auto()  # Fallback to base template


# ============================================================
# Data Classes
# ============================================================


@dataclass
class TemplateSelectionResult:
    """Result of template selection."""

    template_key: str
    variant: TemplateVariant
    reason: SelectionReason
    confidence_score: int = 0
    is_ab_test: bool = False


@dataclass
class TemplateSelectionContext:
    """Context for template selection decisions."""

    family_tree: Optional[FamilyTree] = None
    dna_match: Optional[DnaMatch] = None
    person_id: Optional[int] = None
    enable_ab_testing: bool = False
    prefer_short: bool = False
    additional_context: dict[str, Any] = field(default_factory=dict)


# ============================================================
# TemplateSelector Class
# ============================================================


class TemplateSelector:
    """
    Centralized template selector for messaging.

    Consolidates template selection logic previously scattered across:
    - action8_messaging.py: select_template_by_confidence(), select_template_variant_ab_testing()
    - Various template selection helpers

    Args:
        templates: Dictionary of template_key -> template_content

    Example:
        selector = TemplateSelector(MESSAGE_TEMPLATES)
        result = selector.select_template("Out_Tree-Initial", person, context)
        template_content = templates[result.template_key]
    """

    # Distant relationship patterns
    DISTANT_RELATIONSHIP_PATTERNS = frozenset([
        "4th cousin",
        "5th cousin",
        "6th cousin",
        "distant cousin",
        "half",
        "removed",
    ])

    def __init__(self, templates: dict[str, str]) -> None:
        """Initialize with available templates."""
        self._templates = templates

    def select_template(
        self,
        base_template_key: str,
        person: Optional[Person] = None,
        context: Optional[TemplateSelectionContext] = None,
    ) -> TemplateSelectionResult:
        """
        Select the appropriate template variant for a person.

        Selection priority:
        1. A/B testing (if enabled)
        2. Distant relationship → Exploratory template
        3. Confidence-based (High → Confident, Low → Exploratory)
        4. Short variant (if preferred)
        5. Base template (fallback)

        Args:
            base_template_key: Base template key (e.g., "Out_Tree-Initial")
            person: Person to message (optional, for person-specific selection)
            context: Additional selection context

        Returns:
            TemplateSelectionResult with selected template key and metadata
        """
        ctx = context or TemplateSelectionContext()

        # Get person_id for A/B testing
        if person and ctx.person_id is None:
            ctx.person_id = getattr(person, "id", None)

        # 1. A/B testing takes priority if enabled
        if ctx.enable_ab_testing and ctx.person_id:
            return self._select_ab_test_variant(base_template_key, ctx.person_id)

        # 2. Check for distant relationship
        if ctx.family_tree:
            is_distant = self._is_distant_relationship(ctx.family_tree)
            if is_distant:
                return self._select_distant_variant(base_template_key)

        # 3. Confidence-based selection
        confidence_score = self._calculate_confidence_score(ctx)
        if confidence_score > 0:
            return self._select_by_confidence(base_template_key, confidence_score)

        # 4. Short variant if preferred
        if ctx.prefer_short:
            short_result = self._try_short_variant(base_template_key)
            if short_result:
                return short_result

        # 5. Fallback to base template
        return TemplateSelectionResult(
            template_key=base_template_key,
            variant=TemplateVariant.BASE,
            reason=SelectionReason.DEFAULT,
        )

    def select_initial_template(
        self,
        person: Person,
        context: Optional[TemplateSelectionContext] = None,
    ) -> TemplateSelectionResult:
        """Select initial outreach template."""
        ctx = context or TemplateSelectionContext()
        in_tree = getattr(person, "in_my_tree", False)
        base_key = "In_Tree-Initial" if in_tree else "Out_Tree-Initial"
        return self.select_template(base_key, person, ctx)

    def select_followup_template(
        self,
        person: Person,
        context: Optional[TemplateSelectionContext] = None,
    ) -> TemplateSelectionResult:
        """Select follow-up template."""
        ctx = context or TemplateSelectionContext()
        in_tree = getattr(person, "in_my_tree", False)
        base_key = "In_Tree-Follow_Up" if in_tree else "Out_Tree-Follow_Up"
        return self.select_template(base_key, person, ctx)

    def select_final_reminder_template(
        self,
        person: Person,
        context: Optional[TemplateSelectionContext] = None,
    ) -> TemplateSelectionResult:
        """Select final reminder template."""
        ctx = context or TemplateSelectionContext()
        in_tree = getattr(person, "in_my_tree", False)
        base_key = "In_Tree-Final_Reminder" if in_tree else "Out_Tree-Final_Reminder"
        return self.select_template(base_key, person, ctx)

    def _select_ab_test_variant(
        self,
        base_template_key: str,
        person_id: int,
    ) -> TemplateSelectionResult:
        """Select template variant for A/B testing."""
        # Use person ID for consistent assignment (50/50 split)
        use_variant_b = person_id % 2 == 0

        if use_variant_b:
            short_key = self._get_short_key(base_template_key)
            if short_key:
                return TemplateSelectionResult(
                    template_key=short_key,
                    variant=TemplateVariant.AB_TEST_B,
                    reason=SelectionReason.AB_TEST_ASSIGNMENT,
                    is_ab_test=True,
                )

        return TemplateSelectionResult(
            template_key=base_template_key,
            variant=TemplateVariant.AB_TEST_A,
            reason=SelectionReason.AB_TEST_ASSIGNMENT,
            is_ab_test=True,
        )

    def _select_distant_variant(
        self,
        base_template_key: str,
    ) -> TemplateSelectionResult:
        """Select template for distant relationships."""
        exploratory_key = f"{base_template_key}_Exploratory"
        if exploratory_key in self._templates:
            return TemplateSelectionResult(
                template_key=exploratory_key,
                variant=TemplateVariant.EXPLORATORY,
                reason=SelectionReason.DISTANT_RELATIONSHIP,
            )

        # Fallback to short variant
        short_key = self._get_short_key(base_template_key)
        if short_key:
            return TemplateSelectionResult(
                template_key=short_key,
                variant=TemplateVariant.SHORT,
                reason=SelectionReason.DISTANT_RELATIONSHIP,
            )

        return TemplateSelectionResult(
            template_key=base_template_key,
            variant=TemplateVariant.BASE,
            reason=SelectionReason.FALLBACK,
        )

    def _select_by_confidence(
        self,
        base_template_key: str,
        confidence_score: int,
    ) -> TemplateSelectionResult:
        """Select template based on confidence score."""
        if confidence_score >= 4:
            confident_key = f"{base_template_key}_Confident"
            if confident_key in self._templates:
                return TemplateSelectionResult(
                    template_key=confident_key,
                    variant=TemplateVariant.CONFIDENT,
                    reason=SelectionReason.CONFIDENCE_HIGH,
                    confidence_score=confidence_score,
                )

        if confidence_score <= 2:
            exploratory_key = f"{base_template_key}_Exploratory"
            if exploratory_key in self._templates:
                return TemplateSelectionResult(
                    template_key=exploratory_key,
                    variant=TemplateVariant.EXPLORATORY,
                    reason=SelectionReason.CONFIDENCE_LOW,
                    confidence_score=confidence_score,
                )

        # Fallback to short or base
        short_key = self._get_short_key(base_template_key)
        if short_key:
            return TemplateSelectionResult(
                template_key=short_key,
                variant=TemplateVariant.SHORT,
                reason=SelectionReason.SHORT_PREFERRED,
                confidence_score=confidence_score,
            )

        return TemplateSelectionResult(
            template_key=base_template_key,
            variant=TemplateVariant.BASE,
            reason=SelectionReason.FALLBACK,
            confidence_score=confidence_score,
        )

    def _try_short_variant(
        self,
        base_template_key: str,
    ) -> Optional[TemplateSelectionResult]:
        """Try to select short variant if it exists."""
        short_key = self._get_short_key(base_template_key)
        if short_key:
            return TemplateSelectionResult(
                template_key=short_key,
                variant=TemplateVariant.SHORT,
                reason=SelectionReason.SHORT_PREFERRED,
            )
        return None

    def _get_short_key(self, base_template_key: str) -> Optional[str]:
        """Get short template key if it exists."""
        short_key = f"{base_template_key}_Short"
        return short_key if short_key in self._templates else None

    def _is_distant_relationship(self, family_tree: FamilyTree) -> bool:
        """Check if family tree indicates a distant relationship."""
        actual_rel = getattr(family_tree, "actual_relationship", None)
        if not actual_rel or actual_rel == "N/A":
            return False

        actual_rel_lower = actual_rel.lower()
        return any(pattern in actual_rel_lower for pattern in self.DISTANT_RELATIONSHIP_PATTERNS)

    def _calculate_confidence_score(self, context: TemplateSelectionContext) -> int:  # noqa: PLR6301
        """Calculate confidence score from context."""
        score = 0

        # Family tree confidence
        if context.family_tree:
            actual_rel = getattr(context.family_tree, "actual_relationship", None)
            if actual_rel and actual_rel != "N/A" and actual_rel.strip():
                score += 2

            common_ancestor = getattr(context.family_tree, "common_ancestor", None)
            if common_ancestor and common_ancestor.strip():
                score += 1

            person_name = getattr(context.family_tree, "person_name_in_tree", None)
            if person_name and person_name.strip():
                score += 1

        # DNA match confidence
        if context.dna_match:
            predicted_rel = getattr(context.dna_match, "predicted_relationship", None)
            if predicted_rel and predicted_rel != "N/A" and predicted_rel.strip():
                score += 1

        return score


# ============================================================
# Module Tests
# ============================================================


def module_tests() -> bool:
    """Run module tests for template_selector.py."""
    from dataclasses import dataclass as dc

    suite = TestSuite("Template Selector", "messaging/template_selector.py")
    suite.start_suite()

    # Test templates
    test_templates = {
        "Out_Tree-Initial": "Initial message",
        "Out_Tree-Initial_Short": "Short initial",
        "Out_Tree-Initial_Confident": "Confident initial",
        "Out_Tree-Initial_Exploratory": "Exploratory initial",
        "In_Tree-Initial": "In tree initial",
    }

    # Mock Person class
    @dc
    class MockPerson:
        id: int = 1
        username: str = "test_user"
        in_my_tree: bool = False

    # Mock FamilyTree class
    @dc
    class MockFamilyTree:
        actual_relationship: str = "2nd cousin"
        common_ancestor: str = "John Smith"
        person_name_in_tree: str = "Jane Doe"

    # Test 1: TemplateVariant enum
    def test_variant_enum() -> None:
        assert len(TemplateVariant) == 6

    suite.run_test(
        "TemplateVariant enum has 6 values",
        test_variant_enum,
        expected_outcome="BASE, SHORT, CONFIDENT, EXPLORATORY, AB_TEST_A, AB_TEST_B",
    )

    # Test 2: SelectionReason enum
    def test_reason_enum() -> None:
        assert len(SelectionReason) == 7

    suite.run_test(
        "SelectionReason enum has 7 values",
        test_reason_enum,
        expected_outcome="DEFAULT, CONFIDENCE_HIGH/LOW, DISTANT_RELATIONSHIP, AB_TEST, SHORT_PREFERRED, FALLBACK",
    )

    # Test 3: Basic template selection
    selector = TemplateSelector(test_templates)
    result = selector.select_template("Out_Tree-Initial")

    def test_basic_selection() -> None:
        assert result.template_key == "Out_Tree-Initial"
        assert result.variant == TemplateVariant.BASE

    suite.run_test(
        "Basic template selection returns base variant",
        test_basic_selection,
        expected_outcome="template_key='Out_Tree-Initial', variant=BASE",
    )

    # Test 4: A/B testing - variant A (odd person_id)
    ctx_a = TemplateSelectionContext(enable_ab_testing=True, person_id=1)
    result_a = selector.select_template("Out_Tree-Initial", context=ctx_a)

    def test_ab_variant_a() -> None:
        assert result_a.variant == TemplateVariant.AB_TEST_A
        assert result_a.is_ab_test

    suite.run_test(
        "A/B testing returns variant A for odd person_id",
        test_ab_variant_a,
        expected_outcome="variant=AB_TEST_A, is_ab_test=True",
    )

    # Test 5: A/B testing - variant B (even person_id)
    ctx_b = TemplateSelectionContext(enable_ab_testing=True, person_id=2)
    result_b = selector.select_template("Out_Tree-Initial", context=ctx_b)

    def test_ab_variant_b() -> None:
        assert result_b.variant == TemplateVariant.AB_TEST_B
        assert result_b.template_key == "Out_Tree-Initial_Short"

    suite.run_test(
        "A/B testing returns variant B (short) for even person_id",
        test_ab_variant_b,
        expected_outcome="variant=AB_TEST_B, template='Out_Tree-Initial_Short'",
    )

    # Test 6: Confidence-based selection
    ctx_high = TemplateSelectionContext(
        family_tree=MockFamilyTree(),  # type: ignore[arg-type]
    )
    result_high = selector.select_template("Out_Tree-Initial", context=ctx_high)

    def test_confidence_selection() -> None:
        assert result_high.confidence_score >= 4
        assert result_high.template_key == "Out_Tree-Initial_Confident"

    suite.run_test(
        "High confidence selects Confident variant",
        test_confidence_selection,
        expected_outcome="confidence_score>=4, template='Out_Tree-Initial_Confident'",
    )

    # Test 7: Select initial template for out-tree person
    person = MockPerson(in_my_tree=False)
    initial_result = selector.select_initial_template(person)  # type: ignore[arg-type]

    def test_initial_template() -> None:
        assert "Out_Tree-Initial" in initial_result.template_key

    suite.run_test(
        "select_initial_template for out-tree person",
        test_initial_template,
        expected_outcome="template_key contains 'Out_Tree-Initial'",
    )

    # Test 8: Select initial template for in-tree person
    in_tree_person = MockPerson(in_my_tree=True)
    in_tree_result = selector.select_initial_template(in_tree_person)  # type: ignore[arg-type]

    def test_in_tree_initial() -> None:
        assert in_tree_result.template_key == "In_Tree-Initial"

    suite.run_test(
        "select_initial_template for in-tree person",
        test_in_tree_initial,
        expected_outcome="template_key='In_Tree-Initial'",
    )

    return suite.finish_suite()


# Standard test runner pattern
def run_comprehensive_tests() -> bool:
    """Standard entry point for test runner."""
    return module_tests()


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
