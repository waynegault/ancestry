#!/usr/bin/env python3

"""
Empathetic Response Guidelines Module (Phase 6.3)

Provides draft response templates for escalation-path cases that require
human editing and review before sending. These templates are designed to:
- Acknowledge sensitive situations with empathy
- Never attempt to resolve complex emotional/legal issues via automation
- Provide a starting point for human reviewers to personalize
- Flag clearly that these are drafts requiring human judgment

Templates cover:
- BEREAVEMENT: Messages mentioning recent death, grief, loss
- LEGAL_PRIVACY: GDPR requests, attorney mentions, privacy concerns
- DNA_RESULT_SHOCK: NPE discoveries, unexpected ethnicity, adoption revelations
- HIGH_CONFLICT: Family disputes, estrangement, historical grievances
- SELF_HARM: Crisis situations (template emphasizes professional resources)

CRITICAL: These templates are NEVER auto-sent. They are placed in the
review queue with HUMAN_REVIEW status for mandatory human editing.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# === ENUMS ===


class EscalationCategory(Enum):
    """Categories requiring empathetic, human-reviewed responses."""

    BEREAVEMENT = "BEREAVEMENT"
    LEGAL_PRIVACY = "LEGAL_PRIVACY"
    DNA_RESULT_SHOCK = "DNA_RESULT_SHOCK"
    HIGH_CONFLICT = "HIGH_CONFLICT"
    SELF_HARM = "SELF_HARM"
    GENERAL_SENSITIVE = "GENERAL_SENSITIVE"


# === DATA CLASSES ===


@dataclass
class EmpatheticTemplate:
    """Template for an empathetic response draft."""

    category: EscalationCategory
    template_id: str
    opening: str
    body: str
    closing: str
    human_guidance: str  # Instructions for human reviewer
    placeholders: list[str] = field(default_factory=list)  # Variables to personalize


@dataclass
class EscalationDraft:
    """A draft response generated for human review."""

    category: EscalationCategory
    template_used: str
    draft_text: str
    guidance_notes: str
    requires_review: bool = True
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


# === TEMPLATE LIBRARY ===


# Bereavement templates - for messages mentioning recent death/loss
BEREAVEMENT_TEMPLATES: list[EmpatheticTemplate] = [
    EmpatheticTemplate(
        category=EscalationCategory.BEREAVEMENT,
        template_id="bereavement_recent_loss",
        opening="I'm deeply sorry to hear about your loss.",
        body=(
            "Thank you for sharing this difficult news with me. "
            "Losing a loved one is never easy, and I want you to know that "
            "your family's history and memories matter deeply. "
            "There's no rush on any genealogy research during this time - "
            "please take all the time you need."
        ),
        closing=(
            "If and when you're ready to continue our conversation, "
            "I'll be here. My sincere condolences to you and your family."
        ),
        human_guidance=(
            "REVIEW NEEDED: This person has experienced a recent loss. "
            "Consider: (1) Adjusting the tone based on their specific situation, "
            "(2) Removing any research requests that might feel intrusive, "
            "(3) Whether to pause automated follow-ups for this contact."
        ),
        placeholders=["[FAMILY_MEMBER_NAME]", "[RELATIONSHIP]"],
    ),
    EmpatheticTemplate(
        category=EscalationCategory.BEREAVEMENT,
        template_id="bereavement_condolence",
        opening="My heart goes out to you during this difficult time.",
        body=(
            "I understand that genealogy conversations may not be a priority right now, "
            "and that's completely okay. Family history is about honoring those who "
            "came before us, and sometimes that means taking time to grieve and remember."
        ),
        closing=(
            "Please know that I'm here whenever you feel ready. "
            "Wishing you peace and comfort."
        ),
        human_guidance=(
            "REVIEW NEEDED: Adjust timing of any future outreach. "
            "Consider adding contact to a 'paused' list for 30-90 days."
        ),
        placeholders=[],
    ),
]


# Legal/Privacy templates - for GDPR, attorney mentions, privacy concerns
LEGAL_PRIVACY_TEMPLATES: list[EmpatheticTemplate] = [
    EmpatheticTemplate(
        category=EscalationCategory.LEGAL_PRIVACY,
        template_id="privacy_data_request",
        opening="I understand your privacy concerns and take them seriously.",
        body=(
            "Thank you for reaching out about your data and privacy. "
            "I want to assure you that I handle all genealogical information "
            "with care and respect for personal boundaries. "
            "If you have specific concerns or requests regarding your data, "
            "I'm happy to address them."
        ),
        closing=(
            "Please let me know how you'd like me to proceed, and I'll "
            "respect your wishes completely."
        ),
        human_guidance=(
            "LEGAL REVIEW NEEDED: This contact has raised privacy/data concerns. "
            "Actions: (1) DO NOT send without legal review if GDPR/CCPA is mentioned, "
            "(2) Document all correspondence, (3) Consider adding to opt-out list, "
            "(4) If attorney is mentioned, STOP all communication pending review."
        ),
        placeholders=["[SPECIFIC_CONCERN]"],
    ),
    EmpatheticTemplate(
        category=EscalationCategory.LEGAL_PRIVACY,
        template_id="legal_cease_contact",
        opening="I completely respect your request.",
        body=(
            "I apologize for any inconvenience my outreach may have caused. "
            "I have noted your preference and will ensure you are not contacted again "
            "through this channel."
        ),
        closing="Thank you for letting me know, and I wish you well.",
        human_guidance=(
            "IMMEDIATE ACTION: Add contact to permanent opt-out list. "
            "Document the request with timestamp. DO NOT send any further messages "
            "after this acknowledgment. This is the ONLY response to send."
        ),
        placeholders=[],
    ),
]


# DNA Result Shock templates - NPE, unexpected results, adoption discoveries
DNA_RESULT_SHOCK_TEMPLATES: list[EmpatheticTemplate] = [
    EmpatheticTemplate(
        category=EscalationCategory.DNA_RESULT_SHOCK,
        template_id="dna_unexpected_results",
        opening="I can only imagine how surprising and perhaps overwhelming this discovery must be.",
        body=(
            "DNA results can sometimes reveal unexpected information about our families. "
            "These discoveries often bring up complex emotions, and there's no 'right' way "
            "to feel about them. Many people have walked this path and found support "
            "in communities like DNA Detectives, NPE Friends Fellowship, and similar groups."
        ),
        closing=(
            "I'm here if you'd like to talk about this more, but I also understand "
            "if you need time to process. Whatever you decide, I support you."
        ),
        human_guidance=(
            "SENSITIVE SITUATION: NPE/unexpected parentage discovery. "
            "Guidelines: (1) Never push for more DNA testing, (2) Offer resources but "
            "don't be pushy, (3) Respect if they want to end the conversation, "
            "(4) Consider whether sharing additional match details would be helpful or harmful."
        ),
        placeholders=["[SUPPORT_GROUP_LINKS]"],
    ),
    EmpatheticTemplate(
        category=EscalationCategory.DNA_RESULT_SHOCK,
        template_id="dna_adoption_discovery",
        opening="Thank you for sharing something so personal with me.",
        body=(
            "Discovering adoption or different biological origins through DNA can be "
            "a life-changing revelation. Your feelings are valid, whatever they may be. "
            "Many adoptees and those with similar discoveries have found it helpful to "
            "connect with others who understand this unique experience."
        ),
        closing=(
            "I'm honored that you've trusted me with this. I'm here to help with "
            "your genealogical journey in whatever way feels right to you."
        ),
        human_guidance=(
            "HIGHLY SENSITIVE: Adoption discovery. "
            "Actions: (1) Proceed with extreme care, (2) Do not make assumptions about "
            "their feelings, (3) Offer adoption-specific resources if appropriate, "
            "(4) Be prepared for complex family dynamics."
        ),
        placeholders=[],
    ),
]


# High Conflict templates - family disputes, estrangement
HIGH_CONFLICT_TEMPLATES: list[EmpatheticTemplate] = [
    EmpatheticTemplate(
        category=EscalationCategory.HIGH_CONFLICT,
        template_id="family_estrangement",
        opening="I appreciate you sharing this context with me.",
        body=(
            "Family relationships can be complicated, and I understand that genealogy "
            "can sometimes intersect with difficult family dynamics. I want to be "
            "respectful of your situation and boundaries."
        ),
        closing=(
            "Please let me know if there are specific sensitivities I should be aware of "
            "as we continue our research. I'm here to help in whatever way works best for you."
        ),
        human_guidance=(
            "FAMILY CONFLICT DETECTED: Proceed carefully. "
            "Guidelines: (1) Do not share information between estranged parties, "
            "(2) Be neutral and avoid taking sides, (3) Focus on historical facts "
            "rather than recent events, (4) Consider if continuing is appropriate."
        ),
        placeholders=["[SPECIFIC_CONCERN]"],
    ),
]


# Self-Harm templates - crisis situations
SELF_HARM_TEMPLATES: list[EmpatheticTemplate] = [
    EmpatheticTemplate(
        category=EscalationCategory.SELF_HARM,
        template_id="crisis_resources",
        opening="I'm concerned about you and want you to know that you matter.",
        body=(
            "What you're going through sounds incredibly difficult. "
            "While I'm not able to provide the support you deserve, there are people "
            "who can help:\n\n"
            "• National Suicide Prevention Lifeline: 988 (call or text)\n"
            "• Crisis Text Line: Text HOME to 741741\n"
            "• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/\n\n"
            "Please consider reaching out to one of these resources."
        ),
        closing="You are not alone, and help is available.",
        human_guidance=(
            "⚠️ CRISIS ALERT - IMMEDIATE HUMAN REVIEW REQUIRED ⚠️\n"
            "This message may indicate a person in crisis. "
            "Actions: (1) Review message immediately, (2) Ensure crisis resources are included, "
            "(3) Do NOT send any message that could be perceived as dismissive, "
            "(4) Consider whether genealogy communication is appropriate at all, "
            "(5) Document the situation carefully."
        ),
        placeholders=[],
    ),
]


# General sensitive template - catch-all
GENERAL_SENSITIVE_TEMPLATES: list[EmpatheticTemplate] = [
    EmpatheticTemplate(
        category=EscalationCategory.GENERAL_SENSITIVE,
        template_id="sensitive_general",
        opening="Thank you for reaching out and sharing this with me.",
        body=(
            "I want to make sure I respond thoughtfully to your message. "
            "Your feelings and concerns are important, and I want to be helpful "
            "while being respectful of your situation."
        ),
        closing=(
            "Please let me know how you'd like to proceed, and I'll do my best "
            "to be supportive."
        ),
        human_guidance=(
            "SENSITIVE MESSAGE DETECTED: Review carefully before sending. "
            "Personalize the response based on the specific situation."
        ),
        placeholders=["[SPECIFIC_RESPONSE]"],
    ),
]


# === TEMPLATE REGISTRY ===


ALL_TEMPLATES: dict[EscalationCategory, list[EmpatheticTemplate]] = {
    EscalationCategory.BEREAVEMENT: BEREAVEMENT_TEMPLATES,
    EscalationCategory.LEGAL_PRIVACY: LEGAL_PRIVACY_TEMPLATES,
    EscalationCategory.DNA_RESULT_SHOCK: DNA_RESULT_SHOCK_TEMPLATES,
    EscalationCategory.HIGH_CONFLICT: HIGH_CONFLICT_TEMPLATES,
    EscalationCategory.SELF_HARM: SELF_HARM_TEMPLATES,
    EscalationCategory.GENERAL_SENSITIVE: GENERAL_SENSITIVE_TEMPLATES,
}


# === CATEGORY DETECTION PATTERNS ===


CATEGORY_KEYWORDS: dict[EscalationCategory, list[str]] = {
    EscalationCategory.BEREAVEMENT: [
        "passed away",
        "died",
        "death",
        "funeral",
        "memorial",
        "grieving",
        "loss of",
        "lost my",
        "mourning",
        "condolences",
        "rest in peace",
        "rip",
    ],
    EscalationCategory.DNA_RESULT_SHOCK: [
        "npe",
        "not parent expected",
        "different father",
        "different mother",
        "biological parent",
        "adopted",
        "adoption",
        "birth parent",
        "birth father",
        "birth mother",
        "shocked by results",
        "unexpected ethnicity",
        "half sibling",
        "secret",
        "affair",
    ],
    EscalationCategory.HIGH_CONFLICT: [
        "estranged",
        "don't speak",
        "cut off",
        "family feud",
        "inheritance dispute",
        "will contest",
        "lawsuit",
        "bitter",
        "betrayal",
        "toxic",
        "abuse",
        "abusive",
    ],
}


# === FUNCTIONS ===


def detect_escalation_category(message_text: str) -> Optional[EscalationCategory]:
    """
    Detect the escalation category from message content.

    Args:
        message_text: The message text to analyze

    Returns:
        Detected EscalationCategory or None
    """
    message_lower = message_text.lower()

    # Check each category's keywords
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in message_lower:
                logger.debug(f"Detected escalation category {category.value} from keyword '{keyword}'")
                return category

    return None


def get_template_for_category(
    category: EscalationCategory, template_id: Optional[str] = None
) -> Optional[EmpatheticTemplate]:
    """
    Get a template for the given escalation category.

    Args:
        category: The escalation category
        template_id: Optional specific template ID

    Returns:
        EmpatheticTemplate or None if not found
    """
    templates = ALL_TEMPLATES.get(category, [])
    if not templates:
        return None

    if template_id:
        for template in templates:
            if template.template_id == template_id:
                return template

    # Return first template if no specific ID requested
    return templates[0]


def format_template(template: EmpatheticTemplate, variables: Optional[dict[str, str]] = None) -> str:
    """
    Format a template with personalization variables.

    Args:
        template: The template to format
        variables: Optional dict of placeholder -> value mappings

    Returns:
        Formatted template text
    """
    full_text = f"{template.opening}\n\n{template.body}\n\n{template.closing}"

    if variables:
        for placeholder, value in variables.items():
            full_text = full_text.replace(placeholder, value)

    return full_text


def generate_empathetic_draft(
    category: EscalationCategory,
    match_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    template_id: Optional[str] = None,
) -> Optional[EscalationDraft]:
    """
    Generate an empathetic draft response for human review.

    IMPORTANT: This draft is NEVER auto-sent. It is placed in the
    review queue with HUMAN_REVIEW status for mandatory editing.

    Args:
        category: The escalation category
        match_name: Optional name for personalization
        context: Optional additional context
        template_id: Optional specific template to use

    Returns:
        EscalationDraft for human review, or None if no template available
    """
    template = get_template_for_category(category, template_id)
    if not template:
        logger.warning(f"No template found for category {category.value}")
        return None

    # Build variables for personalization
    variables: dict[str, str] = {}
    if match_name:
        variables["[MATCH_NAME]"] = match_name

    if context:
        for key, value in context.items():
            placeholder = f"[{key.upper()}]"
            variables[placeholder] = str(value)

    # Format the template
    draft_text = format_template(template, variables)

    # Create the draft
    draft = EscalationDraft(
        category=category,
        template_used=template.template_id,
        draft_text=draft_text,
        guidance_notes=template.human_guidance,
        requires_review=True,
        metadata={
            "match_name": match_name,
            "context": context,
            "placeholders_remaining": [p for p in template.placeholders if p in draft_text],
        },
    )

    logger.info(
        f"Generated empathetic draft for category {category.value}, "
        f"template={template.template_id}, requires_review=True"
    )

    return draft


def map_safety_category_to_escalation(
    safety_category_value: str,
) -> Optional[EscalationCategory]:
    """
    Map a CriticalAlertCategory value to an EscalationCategory.

    Args:
        safety_category_value: The value from CriticalAlertCategory enum

    Returns:
        Corresponding EscalationCategory or None
    """
    mapping = {
        "SELF_HARM": EscalationCategory.SELF_HARM,
        "THREATS_HOSTILITY": EscalationCategory.HIGH_CONFLICT,
        "LEGAL_PRIVACY": EscalationCategory.LEGAL_PRIVACY,
        "HIGH_VALUE_DISCOVERY": None,  # Not an escalation
    }
    return mapping.get(safety_category_value)


def get_all_template_ids() -> list[str]:
    """Get all available template IDs."""
    template_ids: list[str] = []
    for templates in ALL_TEMPLATES.values():
        for template in templates:
            template_ids.append(template.template_id)
    return template_ids


# === TESTS ===


def module_tests() -> bool:
    """Run module tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Empathetic Response Guidelines", "messaging/empathetic_responses.py")

    # Test: EscalationCategory enum values
    def test_escalation_category_enum() -> None:
        assert EscalationCategory.BEREAVEMENT.value == "BEREAVEMENT"
        assert EscalationCategory.LEGAL_PRIVACY.value == "LEGAL_PRIVACY"
        assert EscalationCategory.DNA_RESULT_SHOCK.value == "DNA_RESULT_SHOCK"
        assert EscalationCategory.SELF_HARM.value == "SELF_HARM"

    suite.run_test(
        "EscalationCategory enum values",
        test_escalation_category_enum,
        test_summary="Verify escalation category enum has expected values",
        functions_tested="EscalationCategory enum",
        method_description="Check all category values",
    )

    # Test: EmpatheticTemplate dataclass
    def test_empathetic_template_dataclass() -> None:
        template = EmpatheticTemplate(
            category=EscalationCategory.BEREAVEMENT,
            template_id="test_template",
            opening="Opening",
            body="Body text",
            closing="Closing",
            human_guidance="Review needed",
            placeholders=["[NAME]"],
        )
        assert template.category == EscalationCategory.BEREAVEMENT
        assert template.template_id == "test_template"
        assert "[NAME]" in template.placeholders

    suite.run_test(
        "EmpatheticTemplate dataclass",
        test_empathetic_template_dataclass,
        test_summary="Verify EmpatheticTemplate stores values correctly",
        functions_tested="EmpatheticTemplate dataclass",
        method_description="Create template and verify fields",
    )

    # Test: EscalationDraft dataclass
    def test_escalation_draft_dataclass() -> None:
        draft = EscalationDraft(
            category=EscalationCategory.LEGAL_PRIVACY,
            template_used="privacy_data_request",
            draft_text="Draft text here",
            guidance_notes="Review carefully",
        )
        assert draft.category == EscalationCategory.LEGAL_PRIVACY
        assert draft.requires_review is True
        assert draft.generated_at is not None

    suite.run_test(
        "EscalationDraft dataclass",
        test_escalation_draft_dataclass,
        test_summary="Verify EscalationDraft stores values correctly",
        functions_tested="EscalationDraft dataclass",
        method_description="Create draft and verify fields",
    )

    # Test: ALL_TEMPLATES has all categories
    def test_all_templates_coverage() -> None:
        for category in EscalationCategory:
            assert category in ALL_TEMPLATES, f"Missing templates for {category.value}"
            assert len(ALL_TEMPLATES[category]) > 0, f"No templates for {category.value}"

    suite.run_test(
        "ALL_TEMPLATES has all categories",
        test_all_templates_coverage,
        test_summary="Verify all escalation categories have templates",
        functions_tested="ALL_TEMPLATES registry",
        method_description="Check each category has at least one template",
    )

    # Test: detect_escalation_category
    def test_detect_escalation_category() -> None:
        # Bereavement
        result = detect_escalation_category("My father passed away last week")
        assert result == EscalationCategory.BEREAVEMENT

        # DNA shock
        result = detect_escalation_category("I found out I was adopted through DNA")
        assert result == EscalationCategory.DNA_RESULT_SHOCK

        # High conflict
        result = detect_escalation_category("I've been estranged from my family for years")
        assert result == EscalationCategory.HIGH_CONFLICT

        # No match
        result = detect_escalation_category("Hello, I'm interested in genealogy")
        assert result is None

    suite.run_test(
        "detect_escalation_category function",
        test_detect_escalation_category,
        test_summary="Verify category detection from keywords",
        functions_tested="detect_escalation_category",
        method_description="Test various message types for category detection",
    )

    # Test: get_template_for_category
    def test_get_template_for_category() -> None:
        # Get default template
        template = get_template_for_category(EscalationCategory.BEREAVEMENT)
        assert template is not None
        assert template.category == EscalationCategory.BEREAVEMENT

        # Get specific template
        template = get_template_for_category(
            EscalationCategory.LEGAL_PRIVACY, "legal_cease_contact"
        )
        assert template is not None
        assert template.template_id == "legal_cease_contact"

    suite.run_test(
        "get_template_for_category function",
        test_get_template_for_category,
        test_summary="Verify template retrieval by category",
        functions_tested="get_template_for_category",
        method_description="Test default and specific template retrieval",
    )

    # Test: format_template
    def test_format_template() -> None:
        template = EmpatheticTemplate(
            category=EscalationCategory.GENERAL_SENSITIVE,
            template_id="test",
            opening="Hello [NAME]",
            body="Message about [TOPIC]",
            closing="Best wishes",
            human_guidance="Review",
        )
        result = format_template(template, {"[NAME]": "John", "[TOPIC]": "research"})
        assert "Hello John" in result
        assert "Message about research" in result
        assert "Best wishes" in result

    suite.run_test(
        "format_template function",
        test_format_template,
        test_summary="Verify template formatting with variables",
        functions_tested="format_template",
        method_description="Test placeholder substitution",
    )

    # Test: generate_empathetic_draft
    def test_generate_empathetic_draft() -> None:
        draft = generate_empathetic_draft(
            category=EscalationCategory.BEREAVEMENT,
            match_name="Jane Doe",
        )
        assert draft is not None
        assert draft.category == EscalationCategory.BEREAVEMENT
        assert draft.requires_review is True
        assert "loss" in draft.draft_text.lower() or "sorry" in draft.draft_text.lower()
        assert len(draft.guidance_notes) > 0

    suite.run_test(
        "generate_empathetic_draft function",
        test_generate_empathetic_draft,
        test_summary="Verify draft generation for bereavement",
        functions_tested="generate_empathetic_draft",
        method_description="Generate draft and verify structure",
    )

    # Test: generate_empathetic_draft for self-harm includes resources
    def test_self_harm_draft_includes_resources() -> None:
        draft = generate_empathetic_draft(category=EscalationCategory.SELF_HARM)
        assert draft is not None
        assert "988" in draft.draft_text  # Suicide prevention line
        assert "CRISIS" in draft.guidance_notes.upper()

    suite.run_test(
        "Self-harm draft includes crisis resources",
        test_self_harm_draft_includes_resources,
        test_summary="Verify self-harm templates include helpline info",
        functions_tested="generate_empathetic_draft",
        method_description="Check crisis resources in self-harm draft",
    )

    # Test: map_safety_category_to_escalation
    def test_map_safety_category() -> None:
        assert map_safety_category_to_escalation("SELF_HARM") == EscalationCategory.SELF_HARM
        assert map_safety_category_to_escalation("LEGAL_PRIVACY") == EscalationCategory.LEGAL_PRIVACY
        assert map_safety_category_to_escalation("HIGH_VALUE_DISCOVERY") is None

    suite.run_test(
        "map_safety_category_to_escalation function",
        test_map_safety_category,
        test_summary="Verify mapping from SafetyGuard categories",
        functions_tested="map_safety_category_to_escalation",
        method_description="Test category mapping",
    )

    # Test: get_all_template_ids
    def test_get_all_template_ids() -> None:
        ids = get_all_template_ids()
        assert len(ids) > 0
        assert "bereavement_recent_loss" in ids
        assert "crisis_resources" in ids
        # Verify uniqueness
        assert len(ids) == len(set(ids))

    suite.run_test(
        "get_all_template_ids function",
        test_get_all_template_ids,
        test_summary="Verify template ID retrieval",
        functions_tested="get_all_template_ids",
        method_description="Get all IDs and verify known templates exist",
    )

    # Test: EscalationDraft always requires_review
    def test_draft_always_requires_review() -> None:
        for category in EscalationCategory:
            draft = generate_empathetic_draft(category=category)
            if draft:
                assert draft.requires_review is True, f"Draft for {category.value} should require review"

    suite.run_test(
        "All drafts require human review",
        test_draft_always_requires_review,
        test_summary="Verify no draft can be auto-sent",
        functions_tested="generate_empathetic_draft",
        method_description="Check requires_review flag for all categories",
    )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests for this module."""
    from testing.test_framework import create_standard_test_runner

    runner = create_standard_test_runner(module_tests)
    return runner()


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
