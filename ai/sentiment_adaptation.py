"""
Sentiment Adaptation Module.

Adapts message tone and style based on conversation history,
recipient characteristics, and engagement patterns. Uses sentiment
analysis to optimize outreach effectiveness.

Features:
- Conversation sentiment tracking
- Adaptive tone selection
- Engagement pattern analysis
- Message style recommendations
"""

import logging
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


class Sentiment(Enum):
    """Sentiment classification levels."""

    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class MessageTone(Enum):
    """Message tone options."""

    FORMAL = "formal"
    FRIENDLY = "friendly"
    ENTHUSIASTIC = "enthusiastic"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    RESERVED = "reserved"


class EngagementLevel(Enum):
    """Engagement level classification."""

    HIGH = "high"  # Quick responses, detailed info
    MEDIUM = "medium"  # Moderate response time
    LOW = "low"  # Slow/minimal responses
    UNRESPONSIVE = "unresponsive"  # No responses


@dataclass
class SentimentScore:
    """Sentiment analysis result for a message or conversation."""

    sentiment: Sentiment
    confidence: float  # 0.0 to 1.0
    positive_signals: list[str]
    negative_signals: list[str]
    raw_score: float  # -1.0 to 1.0


@dataclass
class ConversationProfile:
    """Profile of a conversation's characteristics."""

    person_id: str
    person_name: str
    message_count: int
    avg_response_time_hours: float | None
    overall_sentiment: Sentiment
    engagement_level: EngagementLevel
    preferred_tone: MessageTone
    topics_of_interest: list[str]
    communication_style: str
    last_interaction_date: str | None


@dataclass
class ToneRecommendation:
    """Recommendation for message tone and style."""

    recommended_tone: MessageTone
    confidence: float
    reasoning: list[str]
    avoid_topics: list[str]
    suggested_openings: list[str]
    suggested_closings: list[str]
    personalization_hints: list[str]


class SentimentAdapter:
    """
    Adapts message tone based on conversation sentiment and patterns.

    Analyzes conversation history to recommend optimal communication
    styles for better engagement.
    """

    # Keyword patterns for sentiment detection
    POSITIVE_PATTERNS: ClassVar[list[str]] = [
        r"\bthank(?:s|you)\b",
        r"\bgreat\b",
        r"\bexcellent\b",
        r"\bhelpful\b",
        r"\bexcit(?:ed|ing)\b",
        r"\bwonderful\b",
        r"\bamazing\b",
        r"\blove\b",
        r"\bappreciate\b",
        r"\binterest(?:ed|ing)\b",
        r"\bfascin(?:ated|ating)\b",
        r"!+",  # Exclamation marks
        r":[\)D]",  # Smileys
    ]

    NEGATIVE_PATTERNS: ClassVar[list[str]] = [
        r"\bstop\b",
        r"\bunsubscribe\b",
        r"\bremove\b",
        r"\bdon't\s+contact\b",
        r"\bnot\s+interested\b",
        r"\bbusy\b",
        r"\bsorry\b",
        r"\bcan't\s+help\b",
        r"\bno\s+time\b",
        r"\bleave\s+me\s+alone\b",
    ]

    # Tone mapping based on sentiment
    SENTIMENT_TO_TONE: ClassVar[dict[Sentiment, MessageTone]] = {
        Sentiment.VERY_POSITIVE: MessageTone.ENTHUSIASTIC,
        Sentiment.POSITIVE: MessageTone.FRIENDLY,
        Sentiment.NEUTRAL: MessageTone.PROFESSIONAL,
        Sentiment.NEGATIVE: MessageTone.FORMAL,
        Sentiment.VERY_NEGATIVE: MessageTone.RESERVED,
    }

    def __init__(
        self,
        db_session: Any | None = None,
    ):
        """
        Initialize the sentiment adapter.

        Args:
            db_session: Database session for conversation history
        """
        self.db_session = db_session

    def analyze_message(self, message: str) -> SentimentScore:
        """
        Analyze sentiment of a single message.

        Args:
            message: The message text to analyze

        Returns:
            SentimentScore with analysis results
        """
        message_lower = message.lower()

        # Count positive and negative signals
        positive_signals: list[str] = []
        negative_signals: list[str] = []

        for pattern in self.POSITIVE_PATTERNS:
            matches = re.findall(pattern, message_lower, re.IGNORECASE)
            positive_signals.extend(matches)

        for pattern in self.NEGATIVE_PATTERNS:
            matches = re.findall(pattern, message_lower, re.IGNORECASE)
            negative_signals.extend(matches)

        # Calculate raw score
        pos_count = len(positive_signals)
        neg_count = len(negative_signals)
        total = pos_count + neg_count

        if total == 0:
            raw_score = 0.0
            confidence = 0.3  # Low confidence for neutral
        else:
            raw_score = (pos_count - neg_count) / total
            confidence = min(1.0, total * 0.2)  # More signals = more confidence

        # Determine sentiment level
        sentiment = self._score_to_sentiment(raw_score)

        return SentimentScore(
            sentiment=sentiment,
            confidence=confidence,
            positive_signals=positive_signals,
            negative_signals=negative_signals,
            raw_score=raw_score,
        )

    def analyze_conversation(
        self,
        messages: list[dict[str, Any]],
    ) -> ConversationProfile:
        """
        Analyze a complete conversation to build a profile.

        Args:
            messages: List of message dictionaries with 'text', 'sender', 'timestamp'

        Returns:
            ConversationProfile with analysis
        """
        if not messages:
            return self._empty_profile()

        # Analyze each message
        sentiments: list[SentimentScore] = []
        response_times: list[float] = []
        topics: list[str] = []

        for i, msg in enumerate(messages):
            text = msg.get("text", "")
            sentiments.append(self.analyze_message(text))

            # Extract topics (simple keyword extraction)
            topics.extend(self._extract_topics(text))

            # Calculate response time if previous message exists
            if i > 0 and msg.get("timestamp") and messages[i - 1].get("timestamp"):
                try:
                    # Simplified - would need proper timestamp parsing
                    time_diff = float(msg.get("response_hours", 0))
                    if time_diff > 0:
                        response_times.append(time_diff)
                except (ValueError, TypeError):
                    pass

        # Calculate aggregate sentiment
        avg_score = sum(s.raw_score for s in sentiments) / len(sentiments)
        overall_sentiment = self._score_to_sentiment(avg_score)

        # Calculate engagement level
        avg_response = sum(response_times) / len(response_times) if response_times else None
        engagement = self._calculate_engagement(avg_response, len(messages))

        # Determine preferred tone
        preferred_tone = self.SENTIMENT_TO_TONE.get(overall_sentiment, MessageTone.PROFESSIONAL)

        # Unique topics
        unique_topics = list(dict.fromkeys(topics))[:10]

        person_id = messages[0].get("person_id", "")
        person_name = messages[0].get("person_name", "Unknown")

        return ConversationProfile(
            person_id=person_id,
            person_name=person_name,
            message_count=len(messages),
            avg_response_time_hours=avg_response,
            overall_sentiment=overall_sentiment,
            engagement_level=engagement,
            preferred_tone=preferred_tone,
            topics_of_interest=unique_topics,
            communication_style=self._infer_style(sentiments),
            last_interaction_date=messages[-1].get("timestamp"),
        )

    def recommend_tone(
        self,
        profile: ConversationProfile,
    ) -> ToneRecommendation:
        """
        Generate tone recommendations based on conversation profile.

        Args:
            profile: ConversationProfile from analyze_conversation

        Returns:
            ToneRecommendation with suggestions
        """
        reasoning: list[str] = []
        avoid_topics: list[str] = []

        # Base tone on sentiment
        base_tone = self.SENTIMENT_TO_TONE.get(profile.overall_sentiment, MessageTone.PROFESSIONAL)

        # Adjust based on engagement
        if profile.engagement_level == EngagementLevel.HIGH:
            reasoning.append("High engagement - can be more conversational")
            if base_tone == MessageTone.FORMAL:
                base_tone = MessageTone.PROFESSIONAL
        elif profile.engagement_level == EngagementLevel.LOW:
            reasoning.append("Low engagement - keep messages concise")
            if base_tone == MessageTone.ENTHUSIASTIC:
                base_tone = MessageTone.FRIENDLY
        elif profile.engagement_level == EngagementLevel.UNRESPONSIVE:
            reasoning.append("Unresponsive - consider if re-engagement is appropriate")
            base_tone = MessageTone.RESERVED

        # Generate suggestions based on tone
        openings, closings = self._generate_suggestions(base_tone, profile)

        # Personalization hints
        hints: list[str] = []
        if profile.topics_of_interest:
            hints.append(f"Reference shared interests: {', '.join(profile.topics_of_interest[:3])}")
        if profile.person_name != "Unknown":
            hints.append(f"Use their name: {profile.person_name}")

        # Calculate confidence
        confidence = 0.7
        if profile.message_count >= 5:
            confidence += 0.2
        if profile.avg_response_time_hours is not None:
            confidence += 0.1

        return ToneRecommendation(
            recommended_tone=base_tone,
            confidence=min(1.0, confidence),
            reasoning=reasoning,
            avoid_topics=avoid_topics,
            suggested_openings=openings,
            suggested_closings=closings,
            personalization_hints=hints,
        )

    def adapt_message(
        self,
        original_message: str,
        recommendation: ToneRecommendation,
    ) -> str:
        """
        Adapt a message based on tone recommendation.

        This is a simple implementation - production would use AI.

        Args:
            original_message: The message to adapt
            recommendation: ToneRecommendation to apply

        Returns:
            Adapted message
        """
        # Simple adaptations based on tone
        adapted = original_message

        if recommendation.recommended_tone == MessageTone.FORMAL:
            adapted = self._make_formal(adapted)
        elif recommendation.recommended_tone == MessageTone.CASUAL:
            adapted = self._make_casual(adapted)
        elif recommendation.recommended_tone == MessageTone.ENTHUSIASTIC:
            adapted = self._make_enthusiastic(adapted)

        return adapted

    @staticmethod
    def _score_to_sentiment(score: float) -> Sentiment:
        """Convert raw score to sentiment level."""
        if score >= 0.6:
            return Sentiment.VERY_POSITIVE
        if score >= 0.2:
            return Sentiment.POSITIVE
        if score >= -0.2:
            return Sentiment.NEUTRAL
        if score >= -0.6:
            return Sentiment.NEGATIVE
        return Sentiment.VERY_NEGATIVE

    @staticmethod
    def _calculate_engagement(
        avg_response_hours: float | None,
        message_count: int,
    ) -> EngagementLevel:
        """Calculate engagement level from metrics."""
        if message_count <= 1:
            return EngagementLevel.UNRESPONSIVE

        if avg_response_hours is None:
            if message_count >= 5:
                return EngagementLevel.MEDIUM
            return EngagementLevel.LOW

        if avg_response_hours < 24:
            return EngagementLevel.HIGH
        if avg_response_hours < 72:
            return EngagementLevel.MEDIUM
        return EngagementLevel.LOW

    @staticmethod
    def _extract_topics(text: str) -> list[str]:
        """Extract potential topics from message text."""
        # Simple keyword extraction
        genealogy_keywords = [
            "ancestor",
            "family",
            "tree",
            "DNA",
            "match",
            "cousin",
            "grandparent",
            "great",
            "genealogy",
            "research",
            "records",
            "census",
            "birth",
            "death",
            "marriage",
        ]

        topics: list[str] = []
        text_lower = text.lower()
        for keyword in genealogy_keywords:
            if keyword in text_lower:
                topics.append(keyword)

        return topics

    @staticmethod
    def _infer_style(sentiments: list[SentimentScore]) -> str:
        """Infer communication style from sentiment patterns."""
        if not sentiments:
            return "unknown"

        avg_confidence = sum(s.confidence for s in sentiments) / len(sentiments)
        avg_score = sum(s.raw_score for s in sentiments) / len(sentiments)

        if avg_score > 0.3 and avg_confidence > 0.5:
            return "expressive"
        if avg_score < -0.3:
            return "reserved"
        if avg_confidence < 0.3:
            return "brief"
        return "moderate"

    @staticmethod
    def _empty_profile() -> ConversationProfile:
        """Create an empty conversation profile."""
        return ConversationProfile(
            person_id="",
            person_name="Unknown",
            message_count=0,
            avg_response_time_hours=None,
            overall_sentiment=Sentiment.NEUTRAL,
            engagement_level=EngagementLevel.UNRESPONSIVE,
            preferred_tone=MessageTone.PROFESSIONAL,
            topics_of_interest=[],
            communication_style="unknown",
            last_interaction_date=None,
        )

    @staticmethod
    def _generate_suggestions(
        tone: MessageTone,
        profile: ConversationProfile,
    ) -> tuple[list[str], list[str]]:
        """Generate opening and closing suggestions based on tone."""
        name = profile.person_name if profile.person_name != "Unknown" else "there"

        openings = {
            MessageTone.FORMAL: [
                f"Dear {name},",
                "Good day,",
                "I hope this message finds you well.",
            ],
            MessageTone.FRIENDLY: [
                f"Hi {name}!",
                f"Hello {name},",
                "Great to connect with you!",
            ],
            MessageTone.ENTHUSIASTIC: [
                f"Hi {name}!",
                "So excited to reach out!",
                "What an amazing discovery!",
            ],
            MessageTone.PROFESSIONAL: [
                f"Hello {name},",
                "Greetings,",
                "I hope you're doing well.",
            ],
            MessageTone.CASUAL: [
                f"Hey {name}!",
                "Hi!",
                "Quick note -",
            ],
            MessageTone.RESERVED: [
                "Hello,",
                f"Hi {name},",
                "I wanted to briefly reach out.",
            ],
        }

        closings = {
            MessageTone.FORMAL: [
                "Respectfully,",
                "Sincerely,",
                "Best regards,",
            ],
            MessageTone.FRIENDLY: [
                "Looking forward to hearing from you!",
                "Thanks so much!",
                "Best,",
            ],
            MessageTone.ENTHUSIASTIC: [
                "Can't wait to learn more!",
                "So excited to connect!",
                "Talk soon!",
            ],
            MessageTone.PROFESSIONAL: [
                "Best regards,",
                "Thank you,",
                "Looking forward to your response,",
            ],
            MessageTone.CASUAL: [
                "Cheers!",
                "Thanks!",
                "Take care!",
            ],
            MessageTone.RESERVED: [
                "Thank you for your time,",
                "Regards,",
                "Best,",
            ],
        }

        return openings.get(tone, openings[MessageTone.PROFESSIONAL]), closings.get(
            tone, closings[MessageTone.PROFESSIONAL]
        )

    @staticmethod
    def _make_formal(message: str) -> str:
        """Make message more formal."""
        replacements = {
            "hey": "hello",
            "hi!": "Hello,",
            "thanks!": "Thank you.",
            "awesome": "excellent",
            "cool": "great",
            "!": ".",
        }
        result = message
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result

    @staticmethod
    def _make_casual(message: str) -> str:
        """Make message more casual."""
        replacements = {
            "Hello,": "Hi!",
            "Sincerely,": "Cheers!",
            "Thank you.": "Thanks!",
            "excellent": "awesome",
        }
        result = message
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result

    @staticmethod
    def _make_enthusiastic(message: str) -> str:
        """Make message more enthusiastic."""
        # Add exclamation marks to key sentences
        result = message
        if not result.endswith("!"):
            result = result.rstrip(".") + "!"
        return result


# --- Test Suite ---


def _test_sentiment_enum() -> None:
    """Test sentiment enumeration."""
    assert Sentiment.POSITIVE.value == "positive"
    assert Sentiment.NEGATIVE.value == "negative"
    assert len(Sentiment) == 5


def _test_message_tone_enum() -> None:
    """Test message tone enumeration."""
    assert MessageTone.FORMAL.value == "formal"
    assert MessageTone.CASUAL.value == "casual"
    assert len(MessageTone) == 6


def _test_positive_sentiment_detection() -> None:
    """Test detection of positive sentiment."""
    adapter = SentimentAdapter()

    message = "Thank you so much! This is wonderful news, I'm excited to learn more!"
    score = adapter.analyze_message(message)

    assert score.sentiment in {Sentiment.POSITIVE, Sentiment.VERY_POSITIVE}
    assert len(score.positive_signals) >= 2
    assert score.raw_score > 0


def _test_negative_sentiment_detection() -> None:
    """Test detection of negative sentiment."""
    adapter = SentimentAdapter()

    message = "Please stop contacting me. I'm not interested and don't have time for this."
    score = adapter.analyze_message(message)

    assert score.sentiment in {Sentiment.NEGATIVE, Sentiment.VERY_NEGATIVE}
    assert len(score.negative_signals) >= 2
    assert score.raw_score < 0


def _test_neutral_sentiment() -> None:
    """Test detection of neutral sentiment."""
    adapter = SentimentAdapter()

    message = "I received your message about our family connection."
    score = adapter.analyze_message(message)

    assert score.sentiment == Sentiment.NEUTRAL
    assert abs(score.raw_score) <= 0.2


def _test_conversation_analysis() -> None:
    """Test conversation profile generation."""
    adapter = SentimentAdapter()

    messages = [
        {"text": "Hi! I noticed we're DNA matches.", "person_id": "p1", "person_name": "John"},
        {"text": "Thanks for reaching out! I'm interested in our connection.", "response_hours": 12},
        {"text": "Great! I have some information about our shared ancestor.", "response_hours": 6},
    ]

    profile = adapter.analyze_conversation(messages)

    assert profile.person_id == "p1"
    assert profile.person_name == "John"
    assert profile.message_count == 3
    assert profile.overall_sentiment in {Sentiment.POSITIVE, Sentiment.VERY_POSITIVE}


def _test_tone_recommendation() -> None:
    """Test tone recommendation generation."""
    adapter = SentimentAdapter()

    profile = ConversationProfile(
        person_id="p1",
        person_name="Jane",
        message_count=5,
        avg_response_time_hours=12.0,
        overall_sentiment=Sentiment.POSITIVE,
        engagement_level=EngagementLevel.HIGH,
        preferred_tone=MessageTone.FRIENDLY,
        topics_of_interest=["DNA", "family"],
        communication_style="expressive",
        last_interaction_date="2025-01-01",
    )

    recommendation = adapter.recommend_tone(profile)

    assert recommendation.recommended_tone in {MessageTone.FRIENDLY, MessageTone.PROFESSIONAL}
    assert recommendation.confidence > 0.5
    assert len(recommendation.suggested_openings) > 0


def _test_message_adaptation() -> None:
    """Test message tone adaptation."""
    adapter = SentimentAdapter()

    original = "hey! this is awesome news!"

    formal_rec = ToneRecommendation(
        recommended_tone=MessageTone.FORMAL,
        confidence=0.8,
        reasoning=[],
        avoid_topics=[],
        suggested_openings=[],
        suggested_closings=[],
        personalization_hints=[],
    )

    adapted = adapter.adapt_message(original, formal_rec)

    assert "hey" not in adapted.lower() or "hello" in adapted.lower()


def _test_engagement_calculation() -> None:
    """Test engagement level calculation."""
    # High engagement
    level = SentimentAdapter._calculate_engagement(12.0, 10)
    assert level == EngagementLevel.HIGH

    # Medium engagement
    level = SentimentAdapter._calculate_engagement(48.0, 5)
    assert level == EngagementLevel.MEDIUM

    # Low engagement
    level = SentimentAdapter._calculate_engagement(100.0, 3)
    assert level == EngagementLevel.LOW

    # Unresponsive
    level = SentimentAdapter._calculate_engagement(None, 1)
    assert level == EngagementLevel.UNRESPONSIVE


def module_tests() -> bool:
    """Run all module tests."""
    suite = TestSuite("Sentiment Adaptation", "sentiment_adaptation.py")
    suite.start_suite()

    suite.run_test("Sentiment enumeration", _test_sentiment_enum)
    suite.run_test("Message tone enumeration", _test_message_tone_enum)
    suite.run_test("Positive sentiment detection", _test_positive_sentiment_detection)
    suite.run_test("Negative sentiment detection", _test_negative_sentiment_detection)
    suite.run_test("Neutral sentiment detection", _test_neutral_sentiment)
    suite.run_test("Conversation analysis", _test_conversation_analysis)
    suite.run_test("Tone recommendation", _test_tone_recommendation)
    suite.run_test("Message adaptation", _test_message_adaptation)
    suite.run_test("Engagement level calculation", _test_engagement_calculation)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
