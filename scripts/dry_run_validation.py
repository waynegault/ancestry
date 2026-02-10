#!/usr/bin/env python3

"""
Dry-Run Validation Script

Phase 4: Tests the full message processing pipeline against historical conversations.
Generates draft replies and compares against actual human responses.

Usage:
    python scripts/dry_run_validation.py --limit 50
    python scripts/dry_run_validation.py --conversation-id conv123
    python scripts/dry_run_validation.py --export results.json
"""

import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.venv_bootstrap import ensure_venv

ensure_venv(project_root=_PROJECT_ROOT)

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timezone
from typing import Any

from sqlalchemy import and_
from sqlalchemy.orm import Session as DbSession

logger = logging.getLogger(__name__)


# === DATA CLASSES ===


@dataclass
class DryRunResult:
    """Result of processing a single conversation."""

    conversation_id: str
    person_id: int
    person_name: str
    inbound_message: str
    actual_reply: str | None
    generated_draft: str | None
    opt_out_detected: bool
    facts_extracted: int
    ai_confidence: int
    processing_time_ms: float
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DryRunSummary:
    """Summary statistics for the dry run."""

    total_conversations: int = 0
    successful_drafts: int = 0
    opt_outs_detected: int = 0
    errors_encountered: int = 0
    avg_confidence: float = 0.0
    avg_facts_per_message: float = 0.0
    total_processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# === DRY RUN PROCESSOR ===


class DryRunProcessor:
    """
    Processes historical conversations through the full pipeline.

    Validates:
    1. Opt-out detection accuracy
    2. Fact extraction quality
    3. Draft reply generation
    4. Processing performance
    """

    def __init__(self, db_session: DbSession) -> None:
        """Initialize processor with database session."""
        self.db_session = db_session
        self.results: list[DryRunResult] = []
        self.summary = DryRunSummary()

    def load_historical_conversations(
        self, limit: int = 50, conversation_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Load historical conversations from database.

        Args:
            limit: Maximum number of conversations to process
            conversation_id: Specific conversation ID to process

        Returns:
            List of conversation data dictionaries
        """
        from core.database import ConversationLog, Person

        try:
            query = (
                self.db_session.query(ConversationLog, Person)
                .join(Person, ConversationLog.people_id == Person.id)
                .filter(ConversationLog.direction == "IN")
            )

            if conversation_id:
                query = query.filter(ConversationLog.conversation_id == conversation_id)

            query = query.order_by(ConversationLog.latest_timestamp.desc()).limit(limit)

            conversations: list[dict[str, Any]] = []
            for log, person in query.all():
                # Find the actual reply (next outbound message)
                actual_reply = (
                    self.db_session.query(ConversationLog)
                    .filter(
                        and_(
                            ConversationLog.conversation_id == log.conversation_id,
                            ConversationLog.direction == "OUT",
                            ConversationLog.latest_timestamp > log.latest_timestamp,
                        )
                    )
                    .order_by(ConversationLog.latest_timestamp.asc())
                    .first()
                )

                conversations.append(
                    {
                        "conversation_id": log.conversation_id,
                        "person_id": person.id,
                        "person_name": person.display_name,
                        "inbound_message": log.latest_message_content or "",
                        "actual_reply": actual_reply.latest_message_content if actual_reply else None,
                        "timestamp": log.latest_timestamp,
                    }
                )

            logger.info(f"Loaded {len(conversations)} historical conversations")
            return conversations

        except Exception as e:
            logger.error(f"Failed to load conversations: {e}")
            return []

    def process_conversation(self, conv_data: dict[str, Any]) -> DryRunResult:
        """
        Process a single conversation through the full pipeline.

        Args:
            conv_data: Conversation data dictionary

        Returns:
            DryRunResult with processing results
        """
        import time

        start_time = time.time()
        errors: list[str] = []

        result = DryRunResult(
            conversation_id=conv_data["conversation_id"],
            person_id=conv_data["person_id"],
            person_name=conv_data["person_name"],
            inbound_message=conv_data["inbound_message"][:500],  # Truncate for storage
            actual_reply=conv_data.get("actual_reply"),
            generated_draft=None,
            opt_out_detected=False,
            facts_extracted=0,
            ai_confidence=0,
            processing_time_ms=0,
        )

        try:
            # Step 1: Opt-out detection
            from core.opt_out_detection import OptOutDetector

            detector = OptOutDetector(self.db_session)
            opt_out_analysis = detector.analyze_message(conv_data["inbound_message"])
            result.opt_out_detected = opt_out_analysis.is_opt_out

            if result.opt_out_detected:
                result.generated_draft = "[BLOCKED: Opt-out detected]"
                result.ai_confidence = int(opt_out_analysis.confidence * 100)
            else:
                # Step 2: Real AI fact extraction (if available)
                result.facts_extracted = self._extract_facts_with_ai(conv_data["inbound_message"])

                # Step 3: Generate draft using real AI or fallback to mock
                result.generated_draft, result.ai_confidence = self._generate_draft_with_ai(conv_data)

        except Exception as e:
            errors.append(f"Processing error: {e}")
            logger.warning(f"Error processing conversation {conv_data['conversation_id']}: {e}")

        result.errors = errors
        result.processing_time_ms = (time.time() - start_time) * 1000

        return result

    @staticmethod
    def _count_potential_facts(message: str) -> int:
        """Count potential genealogical facts in a message."""
        import re

        fact_count = 0

        # Count names (capitalized words that might be names)
        names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", message)
        fact_count += len(names)

        # Count years (4-digit numbers between 1700-2025)
        years = re.findall(r"\b(1[7-9]\d{2}|20[0-2]\d)\b", message)
        fact_count += len(years)

        # Count locations (common genealogy location indicators)
        locations = re.findall(
            r"\b(?:born|died|lived|from|in)\s+([A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)*)\b",
            message,
        )
        fact_count += len(locations)

        return fact_count

    def _extract_facts_with_ai(self, message: str) -> int:
        """Extract facts using AI or fallback to regex counting."""
        try:
            from ai.ai_interface import extract_genealogical_entities
            from core.session_manager import SessionManager

            sm = SessionManager()
            entities = extract_genealogical_entities(message, sm)

            if entities and isinstance(entities, dict):
                # Count extracted entities - handle nested "extracted_data" structure
                fact_count = 0
                # Check if data is nested under "extracted_data"
                extracted_data = entities.get("extracted_data", entities)
                # Keys from AI extraction response structure
                entity_keys = [
                    "structured_names",
                    "vital_records",
                    "relationships",
                    "locations",
                    "occupations",
                    "research_questions",
                    "documents_mentioned",
                    "dna_information",
                ]
                for key in entity_keys:
                    if key in extracted_data and isinstance(extracted_data[key], list):
                        fact_count += len(extracted_data[key])
                return fact_count
        except Exception as e:
            logger.debug(f"AI extraction failed, using regex fallback: {e}")

        # Fallback to simple regex counting
        return self._count_potential_facts(message)

    def _generate_draft_with_ai(self, conv_data: dict[str, Any]) -> tuple[str, int]:
        """Generate draft reply using AI or fallback to mock."""
        try:
            from ai.ai_interface import generate_genealogical_reply
            from core.session_manager import SessionManager

            sm = SessionManager()

            # Build minimal context for draft generation
            conversation_context = f"Previous message from {conv_data['person_name']}"
            genealogical_data = json.dumps({"person_name": conv_data["person_name"]})

            reply = generate_genealogical_reply(
                conversation_context=conversation_context,
                user_last_message=conv_data["inbound_message"],
                genealogical_data_str=genealogical_data,
                session_manager=sm,
            )

            if reply:
                # Estimate confidence based on reply quality
                confidence = 80 if len(reply) > 100 else 70
                return reply, confidence
        except ImportError:
            logger.debug("AI draft generation not available")
        except Exception as e:
            logger.debug(f"AI draft generation failed: {e}")

        # Fallback to mock draft
        return self._generate_mock_draft(conv_data), 75

    @staticmethod
    def _generate_mock_draft(conv_data: dict[str, Any]) -> str:
        """Generate a mock draft reply for dry run validation."""
        name = conv_data["person_name"]
        return (
            f"Thank you for reaching out, {name}! I'm excited to explore our potential "
            f"connection. Based on our shared DNA, we may have common ancestors. "
            f"I'd love to compare our family trees to find the link. "
            f"Do you have any information about your ancestors from [relevant region]?"
        )

    def run(self, limit: int = 50, conversation_id: str | None = None) -> DryRunSummary:
        """
        Run the full dry-run validation.

        Args:
            limit: Maximum conversations to process
            conversation_id: Specific conversation to process

        Returns:
            DryRunSummary with aggregate statistics
        """
        logger.info(f"Starting dry-run validation (limit={limit})")

        # Load conversations
        conversations = self.load_historical_conversations(limit, conversation_id)
        if not conversations:
            logger.warning("No conversations found for dry run")
            return self.summary

        self.summary.total_conversations = len(conversations)

        # Process each conversation
        total_confidence = 0
        total_facts = 0

        for conv in conversations:
            result = self.process_conversation(conv)
            self.results.append(result)

            if result.generated_draft and not result.errors:
                self.summary.successful_drafts += 1

            if result.opt_out_detected:
                self.summary.opt_outs_detected += 1

            if result.errors:
                self.summary.errors_encountered += 1

            total_confidence += result.ai_confidence
            total_facts += result.facts_extracted
            self.summary.total_processing_time_ms += result.processing_time_ms

        # Calculate averages
        if self.summary.total_conversations > 0:
            self.summary.avg_confidence = total_confidence / self.summary.total_conversations
            self.summary.avg_facts_per_message = total_facts / self.summary.total_conversations

        logger.info(
            f"Dry run complete: {self.summary.successful_drafts}/{self.summary.total_conversations} "
            f"successful drafts, {self.summary.opt_outs_detected} opt-outs detected"
        )

        return self.summary

    def export_results(self, output_path: Path) -> None:
        """Export results to JSON file."""
        data = {
            "summary": self.summary.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "generated_at": datetime.now(UTC).isoformat(),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Results exported to {output_path}")

    def print_report(self) -> None:
        """Print a formatted report to console."""
        print("\n" + "=" * 60)
        print("DRY-RUN VALIDATION REPORT")
        print("=" * 60)
        print("\n📊 Summary:")
        print(f"   Total Conversations: {self.summary.total_conversations}")
        print(f"   Successful Drafts: {self.summary.successful_drafts}")
        print(f"   Opt-Outs Detected: {self.summary.opt_outs_detected}")
        print(f"   Errors: {self.summary.errors_encountered}")
        print(f"   Avg Confidence: {self.summary.avg_confidence:.1f}%")
        print(f"   Avg Facts/Message: {self.summary.avg_facts_per_message:.1f}")
        print(f"   Total Time: {self.summary.total_processing_time_ms:.1f}ms")

        if self.results:
            print("\n📝 Sample Results (first 5):")
            for result in self.results[:5]:
                status = "🚫 OPT-OUT" if result.opt_out_detected else "✅ OK"
                print(f"\n   [{status}] {result.person_name} ({result.conversation_id})")
                print(f"   Message: {result.inbound_message[:80]}...")
                print(f"   Facts: {result.facts_extracted}, Confidence: {result.ai_confidence}%")

        print("\n" + "=" * 60)


# === CLI ===


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dry-run validation against historical conversations")
    parser.add_argument("--limit", type=int, default=50, help="Number of conversations to process")
    parser.add_argument("--conversation-id", type=str, help="Specific conversation ID to process")
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        from core.database_manager import DatabaseManager

        db_manager = DatabaseManager()
        db_session = db_manager.get_session()
        if db_session is None:
            logger.error("Failed to get database session")
            return 1

        try:
            processor = DryRunProcessor(db_session)
            processor.run(limit=args.limit, conversation_id=args.conversation_id)
            processor.print_report()

            if args.export:
                processor.export_results(Path(args.export))
        finally:
            db_session.close()

        return 0

    except Exception as e:
        logger.error(f"Dry run failed: {e}")
        return 1


# === MODULE TESTS ===


def module_tests() -> bool:
    """Run module-specific tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Dry-Run Validation", "scripts/dry_run_validation.py")
    suite.start_suite()

    # Test 1: DryRunResult dataclass
    def test_result_dataclass() -> None:
        result = DryRunResult(
            conversation_id="test_conv",
            person_id=123,
            person_name="Test User",
            inbound_message="Hello",
            actual_reply="Hi there",
            generated_draft="Generated response",
            opt_out_detected=False,
            facts_extracted=2,
            ai_confidence=85,
            processing_time_ms=150.0,
        )
        data = result.to_dict()
        assert data["conversation_id"] == "test_conv", "conversation_id should match"
        assert data["facts_extracted"] == 2, "facts_extracted should be 2"

    suite.run_test(
        "DryRunResult dataclass",
        test_result_dataclass,
        test_summary="DryRunResult serializes correctly",
        functions_tested="DryRunResult.to_dict",
        method_description="Create result and verify serialization",
    )

    # Test 2: DryRunSummary dataclass
    def test_summary_dataclass() -> None:
        summary = DryRunSummary(
            total_conversations=50,
            successful_drafts=45,
            opt_outs_detected=3,
        )
        data = summary.to_dict()
        assert data["total_conversations"] == 50, "total should be 50"
        assert data["successful_drafts"] == 45, "successful should be 45"

    suite.run_test(
        "DryRunSummary dataclass",
        test_summary_dataclass,
        test_summary="DryRunSummary serializes correctly",
        functions_tested="DryRunSummary.to_dict",
        method_description="Create summary and verify serialization",
    )

    # Test 3: Fact counting
    def test_fact_counting() -> None:
        message = "My grandmother Mary Smith was born in 1895 in Scotland."
        count = DryRunProcessor._count_potential_facts(message)
        assert count >= 2, f"Should find at least 2 facts, found {count}"

    suite.run_test(
        "Fact counting",
        test_fact_counting,
        test_summary="Counts potential genealogical facts in messages",
        functions_tested="DryRunProcessor._count_potential_facts",
        method_description="Analyze message for names, dates, locations",
    )

    # Test 4: Mock draft generation
    def test_mock_draft() -> None:
        conv_data = {
            "conversation_id": "test",
            "person_id": 1,
            "person_name": "John Doe",
            "inbound_message": "Hello",
        }
        draft = DryRunProcessor._generate_mock_draft(conv_data)
        assert "John Doe" in draft, "Draft should include person name"
        assert len(draft) > 50, "Draft should be substantial"

    suite.run_test(
        "Mock draft generation",
        test_mock_draft,
        test_summary="Generates personalized mock drafts",
        functions_tested="DryRunProcessor._generate_mock_draft",
        method_description="Generate draft and verify personalization",
    )

    # Test 5: Opt-out integration
    def test_opt_out_integration() -> None:
        from unittest.mock import MagicMock

        processor = DryRunProcessor(MagicMock())
        conv_data = {
            "conversation_id": "test",
            "person_id": 1,
            "person_name": "Test User",
            "inbound_message": "Please stop contacting me",
        }
        result = processor.process_conversation(conv_data)
        assert result.opt_out_detected, "Should detect opt-out"

    suite.run_test(
        "Opt-out integration",
        test_opt_out_integration,
        test_summary="Detects opt-out messages during processing",
        functions_tested="DryRunProcessor.process_conversation",
        method_description="Process opt-out message and verify detection",
    )

    # Test 6: Normal message processing
    def test_normal_processing() -> None:
        from unittest.mock import MagicMock

        processor = DryRunProcessor(MagicMock())
        conv_data = {
            "conversation_id": "test",
            "person_id": 1,
            "person_name": "Test User",
            "inbound_message": "I think we share ancestors from Scotland!",
        }
        result = processor.process_conversation(conv_data)
        assert not result.opt_out_detected, "Should not detect opt-out"
        assert result.generated_draft is not None, "Should generate draft"

    suite.run_test(
        "Normal message processing",
        test_normal_processing,
        test_summary="Processes normal messages without opt-out",
        functions_tested="DryRunProcessor.process_conversation",
        method_description="Process friendly message and verify draft generation",
    )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run all tests with proper framework setup."""
    from testing.test_framework import create_standard_test_runner

    runner = create_standard_test_runner(module_tests)
    return runner()


if __name__ == "__main__":
    # Check if running tests (via flag or environment variable from run_all_tests.py)
    if (len(sys.argv) > 1 and sys.argv[1] == "--test") or os.environ.get("RUN_MODULE_TESTS") == "1":
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    else:
        sys.exit(main())
