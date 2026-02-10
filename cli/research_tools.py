"""CLI commands for Innovation Features.

Provides command-line access to:
- Triangulation Intelligence
- Predictive Gap Detection
- Sentiment Analysis

These tools help genealogists analyze DNA matches, identify research gaps,
and optimize communication with matches.
"""


import importlib
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text
from tqdm import tqdm

from config import config_schema
from core.database import DnaMatch, Person
from core.database_manager import DatabaseManager
from genealogy.dna.dna_ethnicity_utils import load_ethnicity_metadata
from genealogy.gedcom.gedcom_utils import GedcomData, get_full_name
from research.relationship_diagram import generate_relationship_diagram
from research.relationship_utils import (
    convert_gedcom_path_to_unified_format,
    fast_bidirectional_bfs,
)
from research.research_prioritization import IntelligentResearchPrioritizer

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


@dataclass
class SimpleReport:
    """Simple report structure for gap analysis results."""

    gaps: list[Any]
    completeness_score: float


def _resolve_gedcom_path() -> Path:
    """Resolve GEDCOM path strictly from .env (GEDCOM_FILE_PATH).

    Returns:
        Path to the GEDCOM file if configured.

    Raises:
        FileNotFoundError: If GEDCOM_FILE_PATH is unset or empty.
    """

    db_cfg = getattr(config_schema, "database", None)
    gedcom_path = getattr(db_cfg, "gedcom_file_path", None) if db_cfg else None

    if not gedcom_path:
        raise FileNotFoundError("GEDCOM_FILE_PATH is not set in .env")

    return Path(gedcom_path)


class ResearchToolsCLI:
    """CLI interface for Innovation Features."""

    def __init__(self) -> None:
        """Initialize the research tools CLI."""
        self._triangulation = None
        self._gap_detector = None
        self._sentiment_adapter = None
        self._conflict_detector = None

    def _ensure_component(self, attr_name: str, module_path: str, class_name: str) -> Any:
        """Generic lazy-loader for optional components."""
        instance = getattr(self, attr_name)
        if instance is None:
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                instance = cls()
                setattr(self, attr_name, instance)
            except ImportError as e:
                logger.warning(f"{class_name} not available: {e}")
                return None
        return instance

    def _ensure_triangulation(self) -> Any:
        """Lazy-load TriangulationIntelligence."""
        return self._ensure_component("_triangulation", "research.triangulation_intelligence", "TriangulationIntelligence")

    def _ensure_conflict_detector(self) -> Any:
        """Lazy-load ConflictDetector."""
        return self._ensure_component("_conflict_detector", "research.conflict_detector", "ConflictDetector")

    def _ensure_gap_detector(self) -> Any:
        """Lazy-load PredictiveGapDetector."""
        return self._ensure_component("_gap_detector", "research.predictive_gaps", "PredictiveGapDetector")

    def _ensure_sentiment_adapter(self) -> Any:
        """Lazy-load SentimentAdapter."""
        return self._ensure_component("_sentiment_adapter", "ai.sentiment_adaptation", "SentimentAdapter")

    # ------------------------------------------------------------------
    # Triangulation Intelligence
    # ------------------------------------------------------------------

    def analyze_match_triangulation(
        self,
        match_uuid: str,
        tree_data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
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

        match_data = dict(tree_data or {})
        target_uuid = (
            match_data.get("target_uuid")
            or match_data.get("target_person_uuid")
            or match_data.get("owner_uuid")
            or match_data.get("root_uuid")
            or "ROOT"
        )

        try:
            hypothesis = triangulation.analyze_match(target_uuid, match_uuid, match_data)
            print(f"\n📊 Triangulation Analysis for {match_uuid}")
            print("=" * 50)

            if hypothesis:
                print(f"Hypothesis: {hypothesis.proposed_relationship}")
                print(f"Confidence: {hypothesis.confidence_level.value} ({hypothesis.confidence_score:.2f})")
                evidence_list = hypothesis.evidence or []
                print(f"Evidence pieces: {len(evidence_list)}")
                if evidence_list:
                    print("\nSupporting Evidence:")
                    for evidence in evidence_list[:5]:
                        etype = getattr(evidence, "evidence_type", "unknown")
                        desc = getattr(evidence, "description", "n/a")
                        weight = float(getattr(evidence, "weight", 0.0) or 0.0)
                        print(f"  • {etype}: {desc} (weight: {weight:.2f})")
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
    ) -> dict[str, Any] | None:
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
            # detector.analyze_person returns list[ResearchGap]
            gaps = detector.analyze_person(person_data)

            # Calculate simple completeness score
            score = 100.0
            for gap in gaps:
                # Access priority name safely
                p_name = gap.priority.name if hasattr(gap.priority, "name") else str(gap.priority)
                if p_name == "CRITICAL":
                    score -= 20
                elif p_name == "HIGH":
                    score -= 10
                elif p_name == "MEDIUM":
                    score -= 5
                else:
                    score -= 2

            completeness_score = max(0.0, score)

            report = SimpleReport(gaps=gaps, completeness_score=completeness_score)

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
                    # Handle both attribute names for compatibility
                    suggestions = getattr(gap, "suggested_actions", getattr(gap, "suggestions", []))
                    if suggestions:
                        print(f"      Suggested: {suggestions[0]}")

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
    ) -> dict[str, Any] | None:
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
    ) -> dict[str, Any] | None:
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
    ) -> str | None:
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

                self._process_conflicts(session, conflicts, detector)

        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            print(f"❌ Error: {e}")

    @staticmethod
    def _process_conflicts(session: Any, conflicts: list[Any], detector: Any) -> None:
        """Process a list of conflicts interactively."""
        from core.database import ConflictStatusEnum, Person

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


# ------------------------------------------------------------------
# Interactive Menu Handlers
# ------------------------------------------------------------------


def sanitize_input(value: str) -> str | None:
    """
    Sanitize user input for safe processing.
    """
    if not value:
        return None
    sanitized = value.strip()
    return sanitized if sanitized else None


def _handle_triangulation(cli: ResearchToolsCLI) -> None:
    """Handle triangulation analysis menu option."""
    match_uuid = sanitize_input(input("Enter match UUID: "))
    if match_uuid:
        cli.analyze_match_triangulation(match_uuid)
    else:
        print("UUID required")


def _merge_tree_details(person_data: dict[str, Any], details: dict[str, Any]) -> None:
    """Merge details from tree into person_data."""
    # We prioritize tree data over manual search inputs
    if details.get("birth_year"):
        person_data["birth_year"] = details["birth_year"]
        person_data["birth_date"] = f"1 Jan {details['birth_year']}"

    if details.get("birth_place"):
        person_data["birth_place"] = details["birth_place"]

    if details.get("death_year"):
        person_data["death_year"] = details["death_year"]
        person_data["death_date"] = f"1 Jan {details['death_year']}"

    if details.get("death_place"):
        d_place = details["death_place"]
        # Filter out placeholder values like "N/A"
        if d_place and d_place.upper() != "N/A":
            person_data["death_place"] = d_place

    # Map parents for gap detector
    parents = details.get("parents", [])
    if parents:
        # Assign to father/mother slots to satisfy gap detector
        # We don't know gender here easily, but existence is what matters for "missing parents"
        if len(parents) > 0:
            person_data["father_id"] = parents[0]
        if len(parents) > 1:
            person_data["mother_id"] = parents[1]

    # Pass through ID
    person_data["id"] = details.get("person_id")


def _enrich_person_data_from_tree(
    full_name: str,
    birth_year_int: int | None,
    person_data: dict[str, Any],
) -> None:
    """Try to find person in tree to enrich data."""
    try:
        from genealogy.tree_query_service import TreeQueryService

        service = TreeQueryService()

        print(f"🔎 Searching tree for '{full_name}'...")

        search_result = service.find_person(full_name, approx_birth_year=birth_year_int)

        if search_result.found and search_result.person_id:
            print(f"   ✓ Linked to tree: {search_result.name} (ID: {search_result.person_id})")
            details = service.get_person_details(search_result.person_id)

            if details:
                _merge_tree_details(person_data, details)

                # Note: Sources are not currently returned by get_person_details
                # so "undocumented source" gap may still appear.
        else:
            print(f"   ❌ Could not find '{full_name}' in tree. Using manual input only.")
            logger.warning(f"Gap Analysis: Could not find '{full_name}' in tree. Criteria: birth_year={birth_year_int}")
            if birth_year_int:
                print(f"   Checked approx birth year: {birth_year_int}")
            else:
                print("   No birth year provided for filtering.")
    except Exception as e:
        # Log but continue with manual data
        logger.error(f"Tree lookup failed: {e}")


def _handle_gap_detection(cli: ResearchToolsCLI) -> None:
    """Handle gap detection menu option."""
    first_name = sanitize_input(input("Enter first name: "))
    last_name = sanitize_input(input("Enter last name: "))
    birth_year_str = sanitize_input(input("Enter birth year (optional): "))
    death_year_str = sanitize_input(input("Enter death year (optional): "))
    birth_place = sanitize_input(input("Enter birth place (optional): "))

    full_name = f"{first_name or ''} {last_name or ''}".strip()
    person_data: dict[str, Any] = {"name": full_name or "Unknown"}

    if birth_year_str and birth_year_str.isdigit():
        year = int(birth_year_str)
        person_data["birth_year"] = year
        # Synthesize birth_date to satisfy gap detector
        person_data["birth_date"] = f"1 Jan {year}"

    if death_year_str and death_year_str.isdigit():
        year = int(death_year_str)
        person_data["death_year"] = year
        # Synthesize death_date to satisfy gap detector
        person_data["death_date"] = f"1 Jan {year}"

    if birth_place:
        person_data["birth_place"] = birth_place

    birth_year_int = int(birth_year_str) if birth_year_str and birth_year_str.isdigit() else None
    _enrich_person_data_from_tree(full_name, birth_year_int, person_data)

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


def _select_ethnicity_region(regions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Display available regions and get user selection."""
    print("\nAvailable Regions:")
    for i, region in enumerate(regions, 1):
        print(f"{i}. {region.get('name', 'Unknown')} ({region.get('percentage', 0)}%)")

    choice = input("\nSelect region number: ").strip()
    if not choice.isdigit():
        print("Invalid selection")
        return None

    idx = int(choice) - 1
    if idx < 0 or idx >= len(regions):
        print("Invalid selection")
        return None

    return regions[idx]


def _process_ethnicity_matches(cli: ResearchToolsCLI, matches: list[Any], count: int) -> None:
    """Process the list of matches for ethnicity analysis."""
    print(f"\nRunning analysis on {count} matches...")
    progress = tqdm(matches, total=count, desc="Analyzing matches", unit="match")

    for i, person in enumerate(progress, 1):
        uuid = person.uuid
        name = person.username
        if not uuid:
            progress.write(f"[{i}/{count}] Skipping {name} (No UUID)")
            continue

        progress.write(f"\n[{i}/{count}] Analyzing {name} ({uuid})...")
        cli.analyze_match_triangulation(uuid)

    progress.close()


def _handle_ethnicity_analysis(cli: ResearchToolsCLI) -> None:
    """Handle ethnicity analysis menu option."""
    print("\n--- Ethnicity Analysis ---")

    # Load metadata
    try:
        metadata = load_ethnicity_metadata()
        regions = metadata.get("tree_owner_regions", [])
    except Exception as e:
        print(f"Error loading ethnicity metadata: {e}")
        return

    if not regions:
        print("No ethnicity regions found. Please run Action 6 first to fetch ethnicity data.")
        return

    selected_region = _select_ethnicity_region(regions)
    if not selected_region:
        return

    region_name = selected_region.get("name")
    column_name = selected_region.get("column_name")

    print(f"\nSelected: {region_name}")

    # Query DB
    db_manager = DatabaseManager()
    session = db_manager.get_session()
    if not session:
        print("❌ Could not obtain database session")
        return

    try:
        # Count matches
        # We need to use raw SQL or text() because the column is dynamic
        query = session.query(Person).join(DnaMatch).filter(text(f"dna_match.{column_name} > 0"))
        count = query.count()

        print(f"Found {count} matches sharing {region_name}.")

        if count == 0:
            return

        confirm = input(f"Run analysis on {count} matches? (y/N): ").strip().lower()
        if confirm != "y":
            return

        # Run analysis
        matches = query.all()
        _process_ethnicity_matches(cli, matches, count)

    except Exception as e:
        print(f"Error during analysis: {e}")
        logger.error(f"Ethnicity analysis error: {e}", exc_info=True)
    finally:
        session.close()


def _search_and_select_by_name(gedcom_data: GedcomData, search_term: str) -> tuple[str | None, str]:
    """Search for a person by name and allow user selection."""
    found: list[tuple[str, str]] = []
    for pid, indi in gedcom_data.indi_index.items():
        full_name = get_full_name(indi)
        if search_term.lower() in full_name.lower():
            found.append((pid, full_name))

    if not found:
        print(f"❌ Person '{search_term}' not found.")
        return None, "Unknown"

    if len(found) == 1:
        return found[0]

    print(f"Found {len(found)} matches:")
    for i, (pid, name) in enumerate(found[:10]):
        print(f"{i + 1}. {name} ({pid})")

    try:
        sel = input("Select person (number): ").strip()
        idx = int(sel) - 1
        if 0 <= idx < len(found):
            return found[idx]
    except ValueError:
        pass

    print("Invalid selection.")
    return None, "Unknown"


def _select_person_from_gedcom(
    gedcom_data: GedcomData, prompt: str, allow_empty: bool = False, default_id: str | None = None
) -> tuple[str | None, str]:
    """Select a person from GEDCOM data by ID or name."""
    input_val = input(prompt).strip()

    if not input_val:
        if allow_empty and default_id:
            indi = gedcom_data.indi_index.get(default_id)
            return (default_id, get_full_name(indi)) if indi else (default_id, "Tree Owner")
        return None, "Unknown"

    # Simple search by ID first
    if input_val in gedcom_data.indi_index:
        indi = gedcom_data.indi_index[input_val]
        return input_val, get_full_name(indi)

    return _search_and_select_by_name(gedcom_data, input_val)


def _handle_relationship_diagram(_cli: ResearchToolsCLI) -> None:
    """Handle relationship diagram generation."""
    print("\n" + "=" * 60)
    print("📊 RELATIONSHIP DIAGRAM GENERATOR")
    print("=" * 60)

    # Check if GEDCOM path is configured (no defaults)
    try:
        gedcom_path = _resolve_gedcom_path()
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        print("Set GEDCOM_FILE_PATH in your .env (e.g., GEDCOM_FILE_PATH=Data/Gault Family.ged).")
        return

    if not Path(gedcom_path).exists():
        print(f"❌ GEDCOM file not found at: {gedcom_path}")
        print("Set GEDCOM_FILE_PATH in your .env to the correct location (e.g., Data/Gault Family.ged).")
        return

    print(f"Loading GEDCOM data from: {gedcom_path}...")
    try:
        gedcom_data = GedcomData(Path(gedcom_path))
    except Exception as e:
        print(f"❌ Failed to load GEDCOM data: {e}")
        return

    # Get Person 1 (Start)
    print("\n--- Person 1 (Start) ---")
    p1_id, p1_name = _select_person_from_gedcom(
        gedcom_data,
        "Enter name or ID (leave empty for tree owner): ",
        allow_empty=True,
        default_id=config_schema.reference_person_id,
    )

    if not p1_id:
        if not config_schema.reference_person_id:
            print("❌ No reference person ID configured.")
        return

    # Get Person 2 (End)
    print("\n--- Person 2 (End) ---")
    p2_id, p2_name = _select_person_from_gedcom(gedcom_data, "Enter name or ID: ")

    if not p2_id:
        print("❌ Person 2 is required.")
        return

    print(f"\nCalculating path between {p1_name} and {p2_name}...")

    path_ids = fast_bidirectional_bfs(p1_id, p2_id, gedcom_data.id_to_parents, gedcom_data.id_to_children)

    if not path_ids:
        print("❌ No relationship path found.")
        return

    # Convert to unified format
    unified_path = convert_gedcom_path_to_unified_format(
        path_ids, gedcom_data.reader, gedcom_data.id_to_parents, gedcom_data.id_to_children, gedcom_data.indi_index
    )

    sanitized_path: list[dict[str, str]] = [
        {
            "name": (step.get("name") or "Unknown"),
            "relationship": (step.get("relationship") or ""),
        }
        for step in unified_path
    ]

    # Generate diagram
    print("\nSelect Diagram Style:")
    print("1. Vertical (Standard)")
    print("2. Horizontal (Wide)")
    print("3. Compact (Minimal)")
    style_choice = input("Select style (1-3) [1]: ").strip()

    style_map = {"1": "vertical", "2": "horizontal", "3": "compact"}
    style = style_map.get(style_choice, "vertical")

    diagram = generate_relationship_diagram(p1_name, p2_name, sanitized_path, style)

    print("\n" + diagram)


def _sample_gedcom_individuals(gedcom_data: GedcomData, limit: int = 100) -> list[dict[str, str]]:
    """Sample individuals from GEDCOM data for analysis."""
    individuals_data: list[dict[str, str]] = []

    for i, (pid, indi) in enumerate(gedcom_data.indi_index.items()):
        if i >= limit:
            break

        name = get_full_name(indi)
        surname = ""
        if indi and hasattr(indi, 'name') and hasattr(indi.name, 'surname'):
            surname = indi.name.surname

        individuals_data.append({"id": pid, "name": name, "surname": surname})

    return individuals_data


def _handle_research_prioritization(_cli: ResearchToolsCLI) -> None:
    """Handle research prioritization analysis."""
    print("\n" + "=" * 60)
    print("📋 RESEARCH PRIORITIZATION")
    print("=" * 60)

    # Check if GEDCOM path is configured (no defaults)
    try:
        gedcom_path = _resolve_gedcom_path()
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        print("Set GEDCOM_FILE_PATH in your .env (e.g., GEDCOM_FILE_PATH=Data/Gault Family.ged).")
        return

    if not Path(gedcom_path).exists():
        print(f"❌ GEDCOM file not found at: {gedcom_path}")
        print("Set GEDCOM_FILE_PATH in your .env to the correct location (e.g., Data/Gault Family.ged).")
        return

    print(f"Loading GEDCOM data from: {gedcom_path}...")
    try:
        gedcom_data = GedcomData(Path(gedcom_path))
    except Exception as e:
        print(f"❌ Failed to load GEDCOM data: {e}")
        return

    print("Analyzing tree structure (sample of 100 individuals)...")

    individuals_data = _sample_gedcom_individuals(gedcom_data)

    gedcom_analysis = {"statistics": {"generation_depth": 5}, "individuals": individuals_data, "gaps": []}

    dna_crossref_analysis = {}

    prioritizer = IntelligentResearchPrioritizer()
    plan = prioritizer.prioritize_research_tasks(gedcom_analysis, dna_crossref_analysis)

    print(f"\nIdentified {len(plan['prioritized_tasks'])} priority tasks.")

    if plan['prioritized_tasks']:
        print("\nTop 5 Priorities:")
        for i, task in enumerate(plan['prioritized_tasks'][:5]):
            print(f"{i + 1}. [{task['urgency'].upper()}] {task['description']}")
            print(f"   Score: {task['priority_score']:.1f}")
            if task['target_people']:
                print(f"   Target: {', '.join(task['target_people'])}")
            print("")


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
    print("\n7. Ethnicity Analysis")
    print("   └─ Run analysis on matches sharing a specific ethnicity region")
    print("\n8. Relationship Diagram")
    print("   └─ Generate ASCII relationship diagram between two people")
    print("\n9. Research Prioritization")
    print("   └─ Generate prioritized research tasks based on tree analysis")
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
        "7": _handle_ethnicity_analysis,
        "8": _handle_relationship_diagram,
        "9": _handle_research_prioritization,
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
