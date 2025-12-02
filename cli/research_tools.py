"""CLI commands for Innovation Features.

Provides command-line access to:
- Triangulation Intelligence
- Predictive Gap Detection
- Sentiment Analysis

These tools help genealogists analyze DNA matches, identify research gaps,
and optimize communication with matches.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


class ResearchToolsCLI:
    """CLI interface for Innovation Features."""

    def __init__(self) -> None:
        """Initialize the research tools CLI."""
        self._triangulation = None
        self._gap_detector = None
        self._sentiment_adapter = None
        self._conflict_detector = None

    def _ensure_triangulation(self) -> Any:
        """Lazy-load TriangulationIntelligence."""
        if self._triangulation is None:
            try:
                from research.triangulation_intelligence import TriangulationIntelligence

                self._triangulation = TriangulationIntelligence()
            except ImportError as e:
                logger.warning(f"TriangulationIntelligence not available: {e}")
                return None
        return self._triangulation

    def _ensure_conflict_detector(self) -> Any:
        """Lazy-load ConflictDetector."""
        if self._conflict_detector is None:
            try:
                from research.conflict_detector import ConflictDetector

                self._conflict_detector = ConflictDetector()
            except ImportError as e:
                logger.warning(f"ConflictDetector not available: {e}")
                return None
        return self._conflict_detector

    def _ensure_gap_detector(self) -> Any:
        """Lazy-load PredictiveGapDetector."""
        if self._gap_detector is None:
            try:
                from research.predictive_gaps import PredictiveGapDetector

                self._gap_detector = PredictiveGapDetector()
            except ImportError as e:
                logger.warning(f"PredictiveGapDetector not available: {e}")
                return None
        return self._gap_detector

    def _ensure_sentiment_adapter(self) -> Any:
        """Lazy-load SentimentAdapter."""
        if self._sentiment_adapter is None:
            try:
                from ai.sentiment_adaptation import SentimentAdapter

                self._sentiment_adapter = SentimentAdapter()
            except ImportError as e:
                logger.warning(f"SentimentAdapter not available: {e}")
                return None
        return self._sentiment_adapter

    # ------------------------------------------------------------------
    # Triangulation Intelligence
    # ------------------------------------------------------------------

    def analyze_match_triangulation(
        self,
        match_uuid: str,
        tree_data: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Analyze a DNA match for triangulation opportunities.

        Args:
            match_uuid: UUID of the DNA match
            tree_data: Optional tree data for context

        Returns:
            Dictionary with hypothesis and evidence, or None if unavailable
        """
        triangulation = self._ensure_triangulation()
        if not triangulation:
            print("❌ TriangulationIntelligence not available")
            return None

        try:
            hypothesis = triangulation.analyze_match(match_uuid, tree_data or {})
            print(f"\n📊 Triangulation Analysis for {match_uuid}")
            print("=" * 50)

            if hypothesis:
                print(f"Hypothesis: {hypothesis.proposed_relationship}")
                print(f"Confidence: {hypothesis.confidence_level.value} ({hypothesis.total_score:.2f})")
                print(f"Evidence pieces: {len(hypothesis.supporting_evidence)}")
                if hypothesis.supporting_evidence:
                    print("\nSupporting Evidence:")
                    for evidence in hypothesis.supporting_evidence[:5]:
                        print(f"  • {evidence.evidence_type}: {evidence.description} (weight: {evidence.weight:.2f})")
            else:
                print("No triangulation hypothesis generated")

            return {
                "match_uuid": match_uuid,
                "hypothesis": hypothesis.__dict__ if hypothesis else None,
            }

        except Exception as e:
            logger.error(f"Triangulation analysis failed: {e}")
            print(f"❌ Analysis failed: {e}")
            return None

    def find_match_clusters(
        self,
        match_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Find clusters of related matches.

        Args:
            match_list: List of match dictionaries

        Returns:
            List of cluster dictionaries
        """
        triangulation = self._ensure_triangulation()
        if not triangulation:
            print("❌ TriangulationIntelligence not available")
            return []

        try:
            clusters = triangulation.find_clusters(match_list)
            print(f"\n🔗 Match Clusters Found: {len(clusters)}")
            print("=" * 50)

            for i, cluster in enumerate(clusters, 1):
                print(f"\nCluster {i}: {cluster.common_ancestor or 'Unknown ancestor'}")
                print(f"  Members: {len(cluster.members)}")
                print(f"  Shared DNA: {cluster.avg_shared_cm:.1f} cM avg")
                if cluster.members:
                    print(f"  Matches: {', '.join(cluster.members[:5])}")
                    if len(cluster.members) > 5:
                        print(f"    ... and {len(cluster.members) - 5} more")

            return [c.__dict__ for c in clusters]

        except Exception as e:
            logger.error(f"Cluster detection failed: {e}")
            print(f"❌ Cluster detection failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Predictive Gap Detection
    # ------------------------------------------------------------------

    def analyze_person_gaps(
        self,
        person_data: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """
        Analyze a person's record for research gaps.

        Args:
            person_data: Dictionary with person information

        Returns:
            Gap analysis report dictionary, or None if unavailable
        """
        detector = self._ensure_gap_detector()
        if not detector:
            print("❌ PredictiveGapDetector not available")
            return None

        try:
            report = detector.analyze_person(person_data)
            name = person_data.get("name", "Unknown")
            self._print_gap_report(name, report)
            return {
                "name": name,
                "completeness_score": report.completeness_score,
                "gaps": [g.__dict__ for g in report.gaps],
            }

        except Exception as e:
            logger.error(f"Gap analysis failed: {e}")
            print(f"❌ Gap analysis failed: {e}")
            return None

    @staticmethod
    def _print_gap_report(name: str, report: Any) -> None:
        """Print formatted gap analysis report."""
        print(f"\n🔍 Gap Analysis for {name}")
        print("=" * 50)
        print(f"Completeness Score: {report.completeness_score:.0f}/100")
        print(f"Gaps Found: {len(report.gaps)}")

        if not report.gaps:
            return

        print("\nResearch Gaps (by priority):")
        priority_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        for priority in priority_order:
            gaps_at_priority = [g for g in report.gaps if g.priority.name == priority]
            if gaps_at_priority:
                print(f"\n  {priority}:")
                for gap in gaps_at_priority[:3]:
                    print(f"    • [{gap.gap_type.value}] {gap.description}")
                    if gap.suggestions:
                        print(f"      Suggested: {gap.suggestions[0]}")

    def identify_brick_walls(
        self,
        tree_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Identify brick walls in a family tree.

        Args:
            tree_data: Dictionary with tree information

        Returns:
            List of brick wall candidate dictionaries
        """
        detector = self._ensure_gap_detector()
        if not detector:
            print("❌ PredictiveGapDetector not available")
            return []

        try:
            brick_walls = detector.identify_brick_walls(tree_data)
            print(f"\n🧱 Brick Walls Identified: {len(brick_walls)}")
            print("=" * 50)

            for i, wall in enumerate(brick_walls, 1):
                print(f"\n{i}. {wall.person_name}")
                print(f"   Line: {wall.line_name}")
                print(f"   Blocking factors: {', '.join(wall.blocking_factors[:3])}")
                if wall.suggested_approaches:
                    print("   Suggested approaches:")
                    for approach in wall.suggested_approaches[:2]:
                        print(f"     • {approach}")

            return [w.__dict__ for w in brick_walls]

        except Exception as e:
            logger.error(f"Brick wall identification failed: {e}")
            print(f"❌ Brick wall identification failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Sentiment Analysis
    # ------------------------------------------------------------------

    def analyze_message_sentiment(
        self,
        message: str,
    ) -> Optional[dict[str, Any]]:
        """
        Analyze sentiment of a message.

        Args:
            message: Message text to analyze

        Returns:
            Sentiment score dictionary, or None if unavailable
        """
        adapter = self._ensure_sentiment_adapter()
        if not adapter:
            print("❌ SentimentAdapter not available")
            return None

        try:
            score = adapter.analyze_message(message)

            print("\n💬 Sentiment Analysis")
            print("=" * 50)
            print(f"Message: {message[:100]}{'...' if len(message) > 100 else ''}")
            print(f"\nSentiment: {score.sentiment.value}")
            print(f"Confidence: {score.confidence:.1%}")
            print(f"Raw Score: {score.raw_score:+.2f}")

            if score.positive_signals:
                print(f"\nPositive signals: {', '.join(score.positive_signals[:5])}")
            if score.negative_signals:
                print(f"Negative signals: {', '.join(score.negative_signals[:5])}")

            return {
                "sentiment": score.sentiment.value,
                "confidence": score.confidence,
                "raw_score": score.raw_score,
                "positive_signals": score.positive_signals,
                "negative_signals": score.negative_signals,
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            print(f"❌ Sentiment analysis failed: {e}")
            return None

    def recommend_message_tone(
        self,
        messages: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """
        Recommend message tone based on conversation history.

        Args:
            messages: List of message dictionaries with 'text' keys

        Returns:
            Tone recommendation dictionary, or None if unavailable
        """
        adapter = self._ensure_sentiment_adapter()
        if not adapter:
            print("❌ SentimentAdapter not available")
            return None

        try:
            profile = adapter.analyze_conversation(messages)
            recommendation = adapter.recommend_tone(profile)

            print("\n🎯 Tone Recommendation")
            print("=" * 50)
            print(f"Overall Sentiment: {profile.overall_sentiment.value}")
            print(f"Engagement Level: {profile.engagement_level.value}")
            print(f"Messages Analyzed: {profile.message_count}")

            print(f"\nRecommended Tone: {recommendation.recommended_tone.value}")
            print(f"Confidence: {recommendation.confidence:.1%}")

            if recommendation.reasoning:
                print("\nReasoning:")
                for reason in recommendation.reasoning[:3]:
                    print(f"  • {reason}")

            if recommendation.suggested_openings:
                print("\nSuggested openings:")
                for opening in recommendation.suggested_openings[:2]:
                    print(f"  • \"{opening}\"")

            return {
                "recommended_tone": recommendation.recommended_tone.value,
                "confidence": recommendation.confidence,
                "reasoning": recommendation.reasoning,
                "suggested_openings": recommendation.suggested_openings,
            }

        except Exception as e:
            logger.error(f"Tone recommendation failed: {e}")
            print(f"❌ Tone recommendation failed: {e}")
            return None

    def adapt_message(
        self,
        message: str,
        target_tone: str,
    ) -> Optional[str]:
        """
        Adapt a message to a target tone.

        Args:
            message: Original message text
            target_tone: Target tone (formal, friendly, enthusiastic, etc.)

        Returns:
            Adapted message text, or None if unavailable
        """
        adapter = self._ensure_sentiment_adapter()
        if not adapter:
            print("❌ SentimentAdapter not available")
            return None

        try:
            from ai.sentiment_adaptation import MessageTone, ToneRecommendation

            # Create a recommendation with the target tone
            tone_map = {
                "formal": MessageTone.FORMAL,
                "friendly": MessageTone.FRIENDLY,
                "enthusiastic": MessageTone.ENTHUSIASTIC,
                "professional": MessageTone.PROFESSIONAL,
                "casual": MessageTone.CASUAL,
                "reserved": MessageTone.RESERVED,
            }

            tone = tone_map.get(target_tone.lower(), MessageTone.PROFESSIONAL)
            recommendation = ToneRecommendation(
                recommended_tone=tone,
                confidence=1.0,
                reasoning=[f"User requested {target_tone} tone"],
                avoid_topics=[],
                suggested_openings=[],
                suggested_closings=[],
                personalization_hints=[],
            )

            adapted = adapter.adapt_message(message, recommendation)

            print("\n✏️ Message Adaptation")
            print("=" * 50)
            print(f"Original: {message}")
            print(f"\nTarget tone: {target_tone}")
            print(f"Adapted: {adapted}")

            return adapted

        except Exception as e:
            logger.error(f"Message adaptation failed: {e}")
            print(f"❌ Message adaptation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Conflict Resolution
    # ------------------------------------------------------------------

    def resolve_conflicts_interactive(self) -> None:
        """Interactive conflict resolution workflow."""
        detector = self._ensure_conflict_detector()
        if not detector:
            print("❌ ConflictDetector not available")
            return

        try:
            from core.session_utils import get_session_manager
            from database import ConflictStatusEnum, Person

            session_manager = get_session_manager()
            if not session_manager:
                print("❌ SessionManager not available")
                return

            # Use db_manager to get the session
            session_obj = session_manager.db_manager.get_session()
            if not session_obj:
                print("❌ Could not obtain database session")
                return

            with session_obj as session:
                conflicts = detector.get_open_conflicts(session, limit=20)

                if not conflicts:
                    print("\n✅ No open conflicts found.")
                    return

                print(f"\n🔍 Found {len(conflicts)} open conflicts.")
                print("=" * 60)

                for i, conflict in enumerate(conflicts, 1):
                    person = session.query(Person).filter(Person.id == conflict.people_id).first()
                    person_name = person.display_name if person else f"Person #{conflict.people_id}"

                    print(f"\nConflict {i}/{len(conflicts)}")
                    print(f"Person:   {person_name}")
                    print(f"Field:    {conflict.field_name}")
                    print(f"Existing: {conflict.existing_value}")
                    print(f"New:      {conflict.new_value}")
                    print(f"Source:   {conflict.source}")
                    print("-" * 30)

                    print("Actions:")
                    print("  [a] Accept New (Update Database)")
                    print("  [k] Keep Existing (Reject New)")
                    print("  [i] Ignore (Skip for now)")
                    print("  [q] Quit")

                    choice = input("Select action: ").strip().lower()

                    if choice == "q":
                        break

                    if choice == "a":
                        detector.resolve_conflict(
                            session,
                            conflict.id,
                            ConflictStatusEnum.RESOLVED,
                            apply_new_value=True,
                            resolved_by="user_cli",
                        )
                        print("✅ Updated database with new value.")
                    elif choice == "k":
                        detector.resolve_conflict(
                            session,
                            conflict.id,
                            ConflictStatusEnum.RESOLVED,
                            apply_new_value=False,
                            resolution_notes="User rejected new value",
                            resolved_by="user_cli",
                        )
                        print("✅ Kept existing value.")
                    elif choice == "i":
                        print("Skipped.")
                    else:
                        print("Invalid choice, skipping.")

                    session.commit()

        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            print(f"❌ Error: {e}")


# ------------------------------------------------------------------
# Interactive Menu Handlers
# ------------------------------------------------------------------


def _handle_triangulation(cli: ResearchToolsCLI) -> None:
    """Handle triangulation analysis menu option."""
    match_uuid = input("Enter match UUID: ").strip()
    if match_uuid:
        cli.analyze_match_triangulation(match_uuid)
    else:
        print("UUID required")


def _handle_gap_detection(cli: ResearchToolsCLI) -> None:
    """Handle gap detection menu option."""
    name = input("Enter person name: ").strip()
    birth_year = input("Enter birth year (optional): ").strip()
    death_year = input("Enter death year (optional): ").strip()
    birth_place = input("Enter birth place (optional): ").strip()

    person_data: dict[str, Any] = {"name": name or "Unknown"}
    if birth_year:
        person_data["birth_year"] = int(birth_year)
    if death_year:
        person_data["death_year"] = int(death_year)
    if birth_place:
        person_data["birth_place"] = birth_place

    cli.analyze_person_gaps(person_data)


def _handle_sentiment_analysis(cli: ResearchToolsCLI) -> None:
    """Handle sentiment analysis menu option."""
    message = input("Enter message to analyze: ").strip()
    if message:
        cli.analyze_message_sentiment(message)
    else:
        print("Message required")


def _handle_tone_recommendation(cli: ResearchToolsCLI) -> None:
    """Handle tone recommendation menu option."""
    print("Enter messages (one per line, empty line to finish):")
    messages: list[dict[str, str]] = []
    while True:
        line = input().strip()
        if not line:
            break
        messages.append({"text": line})

    if messages:
        cli.recommend_message_tone(messages)
    else:
        print("At least one message required")


def _handle_adapt_message(cli: ResearchToolsCLI) -> None:
    """Handle message adaptation menu option."""
    message = input("Enter message: ").strip()
    print("Available tones: formal, friendly, enthusiastic, professional, casual, reserved")
    tone = input("Enter target tone: ").strip()
    if message and tone:
        cli.adapt_message(message, tone)
    else:
        print("Message and tone required")


def _handle_conflict_resolution(cli: ResearchToolsCLI) -> None:
    """Handle conflict resolution menu option."""
    cli.resolve_conflicts_interactive()


def _print_menu() -> None:
    """Print the interactive menu."""
    print("\n" + "=" * 60)
    print("🔬 RESEARCH TOOLS - Innovation Features")
    print("=" * 60)
    print("\n1. Triangulation Analysis")
    print("   └─ Analyze a DNA match for triangulation opportunities")
    print("\n2. Gap Detection")
    print("   └─ Identify research gaps in a person's record")
    print("\n3. Sentiment Analysis")
    print("   └─ Analyze message sentiment")
    print("\n4. Tone Recommendation")
    print("   └─ Get message tone recommendation")
    print("\n5. Adapt Message")
    print("   └─ Convert message to target tone")
    print("\n6. Conflict Resolution")
    print("   └─ Interactively resolve data conflicts")
    print("\n0. Exit")
    print("\n" + "-" * 60)


def run_interactive_menu() -> None:
    """Run the interactive research tools menu."""
    cli = ResearchToolsCLI()

    # Map choices to handler functions
    handlers: dict[str, Callable[[ResearchToolsCLI], None]] = {
        "1": _handle_triangulation,
        "2": _handle_gap_detection,
        "3": _handle_sentiment_analysis,
        "4": _handle_tone_recommendation,
        "5": _handle_adapt_message,
        "6": _handle_conflict_resolution,
    }

    while True:
        _print_menu()
        choice = input("Select option: ").strip()

        if choice == "0":
            print("\nExiting research tools...")
            break

        handler = handlers.get(choice)
        if handler:
            handler(cli)
        else:
            print("Invalid option")

        input("\nPress Enter to continue...")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def _test_cli_initialization() -> None:
    """Test ResearchToolsCLI initialization."""
    cli = ResearchToolsCLI()
    assert cli._triangulation is None
    assert cli._gap_detector is None
    assert cli._sentiment_adapter is None


def _test_ensure_triangulation() -> None:
    """Test lazy loading of TriangulationIntelligence."""
    cli = ResearchToolsCLI()
    result = cli._ensure_triangulation()
    # May be None if module not available, but shouldn't raise
    assert result is None or hasattr(result, "analyze_match")


def _test_ensure_gap_detector() -> None:
    """Test lazy loading of PredictiveGapDetector."""
    cli = ResearchToolsCLI()
    result = cli._ensure_gap_detector()
    assert result is None or hasattr(result, "analyze_person")


def _test_ensure_sentiment_adapter() -> None:
    """Test lazy loading of SentimentAdapter."""
    cli = ResearchToolsCLI()
    result = cli._ensure_sentiment_adapter()
    assert result is None or hasattr(result, "analyze_message")


def _test_analyze_sentiment() -> None:
    """Test sentiment analysis CLI method."""
    cli = ResearchToolsCLI()
    result = cli.analyze_message_sentiment("Thank you so much! This is wonderful!")
    # If sentiment adapter available, should return dict
    if result is not None:
        assert "sentiment" in result
        assert "confidence" in result


def _test_gap_analysis() -> None:
    """Test gap analysis CLI method."""
    cli = ResearchToolsCLI()
    result = cli.analyze_person_gaps({"name": "John Smith", "birth_year": 1900})
    if result is not None:
        assert "completeness_score" in result
        assert "gaps" in result


def module_tests() -> bool:
    """Run all module tests."""
    suite = TestSuite("Research Tools CLI", "cli/research_tools.py")

    suite.run_test("CLI initialization", _test_cli_initialization)
    suite.run_test("Lazy load triangulation", _test_ensure_triangulation)
    suite.run_test("Lazy load gap detector", _test_ensure_gap_detector)
    suite.run_test("Lazy load sentiment adapter", _test_ensure_sentiment_adapter)
    suite.run_test("Sentiment analysis", _test_analyze_sentiment)
    suite.run_test("Gap analysis", _test_gap_analysis)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    # When run as module (python -m cli.research_tools), run tests by default
    # Use --menu or --interactive for interactive mode
    if "--menu" in sys.argv or "--interactive" in sys.argv:
        run_interactive_menu()
    else:
        # Default to running tests when invoked as module
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
