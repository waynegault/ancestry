#!/usr/bin/env python3

"""
Action 12: DNA Match Triangulation Analysis

Identifies triangulation opportunities between DNA matches by analyzing shared matches
and common ancestors. Generates hypotheses for how matches are related.

Features:
- Target person selection (UUID or Profile ID)
- Shared match analysis
- Common ancestor identification
- Hypothesis generation
"""

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from config import config_schema
from core.logging_utils import log_action_banner
from core.session_manager import SessionManager
from genealogy.research_service import ResearchService
from genealogy.triangulation import TriangulationService
from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


def _export_results(opportunities: list[dict[str, Any]], format: str) -> None:
    """Export triangulation results to CSV or HTML."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"triangulation_results_{timestamp}.{format}"

    try:
        if format == "csv":
            with Path(filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Shared Match Name", "Shared Match UUID", "Common Ancestor", "Hypothesis", "Confidence"]
                )
                for opp in opportunities:
                    sm = opp["shared_match"]
                    ancestor = opp["common_ancestor"]
                    writer.writerow(
                        [
                            sm.name,
                            sm.uuid,
                            ancestor.get("name", "Unknown"),
                            opp["hypothesis_message"],
                            opp.get("confidence_score", 0.0),
                        ]
                    )
        elif format == "html":
            with Path(filename).open("w", encoding="utf-8") as f:
                f.write("<html><body><h1>Triangulation Results</h1><table border='1'>")
                f.write(
                    "<tr><th>Shared Match</th><th>UUID</th><th>Common Ancestor</th><th>Hypothesis</th><th>Confidence</th></tr>"
                )
                for opp in opportunities:
                    sm = opp["shared_match"]
                    ancestor = opp["common_ancestor"]
                    f.write(f"<tr><td>{sm.name}</td><td>{sm.uuid}</td><td>{ancestor.get('name', 'Unknown')}</td>")
                    f.write(f"<td>{opp['hypothesis_message']}</td><td>{opp.get('confidence_score', 0.0)}</td></tr>")
                f.write("</table></body></html>")

        print(f"Results exported to {filename}")
    except Exception as e:
        logger.error(f"Failed to export results: {e}")


def run_triangulation_analysis(session_manager: SessionManager) -> bool:
    """
    Main entry point for Action 12.
    """
    log_action_banner("Action 12: DNA Match Triangulation", "start")

    if not session_manager.ensure_db_ready():
        logger.error("Database not ready. Aborting.")
        return False

    session = session_manager.db_manager.get_session()
    if not session:
        logger.error("Failed to get database session.")
        return False

    try:
        # 1. Get Target Person
        print("\n--- Triangulation Analysis ---")
        target_input = input("Enter Target Person UUID or Profile ID (or 'q' to quit): ").strip()
        if target_input.lower() == "q":
            return True

        # 2. Initialize Services
        gedcom_path = str(config_schema.database.gedcom_file_path) if config_schema.database.gedcom_file_path else None
        research_service = ResearchService(gedcom_path)
        triangulation_service = TriangulationService(session, research_service)

        # 3. Run Analysis
        print(f"\nAnalyzing matches for: {target_input}...")
        opportunities = triangulation_service.find_triangulation_opportunities(target_input)

        # 4. Display Results
        if not opportunities:
            print("No triangulation opportunities found.")
            print("(Note: Ensure shared matches are populated and linked to tree data)")
        else:
            print(f"\nFound {len(opportunities)} opportunities:")
            for i, opp in enumerate(opportunities, 1):
                shared_match = opp["shared_match"]
                ancestor = opp["common_ancestor"]
                print(f"\n{i}. Shared Match: {shared_match.name} (UUID: {shared_match.uuid})")
                print(f"   Common Ancestor: {ancestor.get('name')}")
                print(f"   Hypothesis: {opp['hypothesis_message']}")

            # 5. Export
            export_choice = input("\nExport results? (csv/html/n): ").strip().lower()
            if export_choice in {"csv", "html"}:
                _export_results(opportunities, export_choice)

        return True

    except Exception as e:
        logger.error(f"Error during triangulation analysis: {e}", exc_info=True)
        return False
    finally:
        session_manager.db_manager.return_session(session)


# --- Test Suite ---


def _test_triangulation_service_integration() -> None:
    """Test basic integration of TriangulationService."""
    from unittest.mock import MagicMock

    # Mock dependencies
    mock_session = MagicMock()
    mock_research_service = MagicMock()

    service = TriangulationService(mock_session, mock_research_service)
    assert service is not None
    assert service.db == mock_session
    assert service.research_service == mock_research_service


def module_tests() -> bool:
    """Run module-specific tests."""
    suite = TestSuite("Action 12 - Triangulation", __file__)
    suite.start_suite()

    suite.run_test(
        "TriangulationService Integration",
        _test_triangulation_service_integration,
        "Verify service initialization",
        "TriangulationService.__init__",
        "Initialize service with mock dependencies",
        "Service initialized correctly",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    print("🧬 Running Action 12 Tests...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
