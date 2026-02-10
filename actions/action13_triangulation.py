#!/usr/bin/env python3

"""
Action 13: DNA Match Triangulation Analysis

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

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import config_schema
from core.database import Person
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

        msg = f"Results exported to {filename}"
        print(msg)
        logger.info(msg)
    except Exception as e:
        logger.error(f"Failed to export results: {e}")


def _search_target_by_name(session: Session) -> tuple[str | None, str | None]:
    """Search for a DNA match in the database by name."""
    print("\n--- Search DNA Matches by Name ---")
    name_query = input("Enter name (partial or full): ").strip()
    if not name_query:
        return None, None

    # Search Person table for matches with a UUID (indicating they are DNA matches)
    stmt = select(Person).where(Person.username.ilike(f"%{name_query}%")).where(Person.uuid.is_not(None)).limit(20)
    results = session.execute(stmt).scalars().all()

    if not results:
        print("No matches found in database.")
        return None, None

    print(f"\nFound {len(results)} matches:")
    for i, person in enumerate(results, 1):
        print(f"{i}. {person.username} (UUID: {person.uuid})")

    while True:
        choice = input("\nSelect a match (number) or 'n' for none: ").strip().lower()
        if choice == "n":
            return None, None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                selected = results[idx]
                return selected.uuid, selected.username
        print("Invalid choice.")


def _render_triangulation_menu(target_name: str | None, results_count: int) -> str:
    """Render the triangulation action menu."""
    print("\n" + "=" * 40)
    print("🧬 ACTION 13: TRIANGULATION ANALYSIS")
    print("=" * 40)

    target_display = target_name if target_name else "None selected"
    print(f"🎯 Target Person: {target_display}")

    if results_count > 0:
        print(f"📊 Analysis Results: {results_count} opportunities found")
    else:
        print("📊 Analysis Results: None")

    print("-" * 40)
    print("1. 🔍 Search for Target Person (by Name)")
    print("2. 🆔 Enter Target ID Manually")

    if target_name:
        print("3. 🚀 Run Triangulation Analysis")

    if results_count > 0:
        print("4. 👁️  View Results")
        print("5. 💾 Export Results")

    print("x. 🔙 Return to Main Menu")
    print("-" * 40)

    return input("Select option: ").strip().lower()


def _handle_search_target(session: Session) -> tuple[str | None, str | None]:
    """Handle searching for a target person by name."""
    uid, name = _search_target_by_name(session)
    if uid:
        return uid, name
    return None, None


def _handle_enter_id() -> tuple[str | None, str | None]:
    """Handle manually entering a target ID."""
    uid = input("\nEnter Target Person UUID or Profile ID: ").strip()
    if uid:
        return uid, uid
    return None, None


def _handle_run_analysis(
    triangulation_service: TriangulationService,
    selected_target_id: str | None,
    selected_target_name: str | None,
) -> list[dict[str, Any]]:
    """Handle running the triangulation analysis."""
    if not selected_target_id:
        print("\n⚠️  Please select a target person first.")
        return []

    print(f"\n🚀 Analyzing matches for: {selected_target_name}...")
    print("   This may take a moment...")

    try:
        new_opportunities = triangulation_service.find_triangulation_opportunities(selected_target_id)
        if not new_opportunities:
            print("\n❌ No triangulation opportunities found.")
            print("   (Ensure shared matches are populated and linked to tree data)")
        else:
            print(f"\n✅ Found {len(new_opportunities)} triangulation opportunities!")
        return new_opportunities
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\n❌ Analysis failed: {e}")
        return []


def _handle_view_results(opportunities: list[dict[str, Any]]) -> None:
    """Handle viewing the analysis results."""
    if not opportunities:
        print("\n⚠️  No results to view. Run analysis first.")
        return

    print(f"\n--- Analysis Results ({len(opportunities)}) ---")
    for i, opp in enumerate(opportunities, 1):
        shared_match = opp["shared_match"]
        ancestor = opp["common_ancestor"]
        print(f"\n{i}. Shared Match: {shared_match.name} (UUID: {shared_match.uuid})")
        print(f"   Common Ancestor: {ancestor.get('name', 'Unknown')}")
        print(f"   Hypothesis: {opp['hypothesis_message']}")
        print(f"   Confidence: {opp.get('confidence_score', 0.0):.2f}")
    input("\nPress Enter to continue...")


def _handle_export_results(opportunities: list[dict[str, Any]]) -> None:
    """Handle exporting the analysis results."""
    if not opportunities:
        print("\n⚠️  No results to export.")
        return

    fmt = input("Export format (csv/html): ").strip().lower()
    if fmt in {"csv", "html"}:
        _export_results(opportunities, fmt)
    else:
        print("Invalid format.")


def _handle_menu_choice(
    choice: str,
    session: Session,
    triangulation_service: TriangulationService,
    selected_target_id: str | None,
    selected_target_name: str | None,
    opportunities: list[dict[str, Any]],
) -> tuple[str | None, str | None, list[dict[str, Any]], bool]:
    """Handle the user's menu choice. Returns updated state and exit flag."""
    if choice == "1":
        uid, name = _handle_search_target(session)
        if uid:
            return uid, name, [], False

    elif choice == "2":
        uid, name = _handle_enter_id()
        if uid:
            return uid, name, [], False

    elif choice == "3":
        opportunities = _handle_run_analysis(triangulation_service, selected_target_id, selected_target_name)
        return selected_target_id, selected_target_name, opportunities, False

    elif choice == "4":
        _handle_view_results(opportunities)

    elif choice == "5":
        _handle_export_results(opportunities)

    elif choice == "x":
        return selected_target_id, selected_target_name, opportunities, True

    else:
        print("\nInvalid option. Please try again.")

    return selected_target_id, selected_target_name, opportunities, False


def run_triangulation_analysis(session_manager: SessionManager) -> bool:
    """
    Main entry point for Action 13 with interactive menu.
    """
    log_action_banner("Action 13: DNA Match Triangulation", "start")

    if not session_manager.ensure_db_ready():
        logger.error("Database not ready. Aborting.")
        return False

    session = session_manager.db_manager.get_session()
    if not session:
        logger.error("Failed to get database session.")
        return False

    try:
        # Initialize Services
        gedcom_path = str(config_schema.database.gedcom_file_path) if config_schema.database.gedcom_file_path else None
        research_service = ResearchService(gedcom_path)
        triangulation_service = TriangulationService(session, research_service)

        # State
        selected_target_id: str | None = None
        selected_target_name: str | None = None
        opportunities: list[dict[str, Any]] = []

        while True:
            choice = _render_triangulation_menu(selected_target_name, len(opportunities))
            selected_target_id, selected_target_name, opportunities, should_exit = _handle_menu_choice(
                choice, session, triangulation_service, selected_target_id, selected_target_name, opportunities
            )
            if should_exit:
                return True

    except Exception as e:
        logger.error(f"Error during triangulation analysis: {e}", exc_info=True)
        return False
    finally:
        session_manager.db_manager.return_session(session)


# --- Test Suite ---


def _test_triangulation_service_integration() -> None:
    """Test TriangulationService initializes with correct dependencies."""
    from unittest.mock import MagicMock

    mock_session = MagicMock()
    mock_research_service = MagicMock()

    service = TriangulationService(mock_session, mock_research_service)
    assert service is not None
    assert service.db == mock_session
    assert service.research_service == mock_research_service


def _test_find_triangulation_groups_no_matches() -> None:
    """Test find_triangulation_opportunities returns empty for unknown person."""
    from unittest.mock import MagicMock, patch

    mock_session = MagicMock()
    mock_research_service = MagicMock()
    # Query returns empty list
    mock_session.query.return_value.filter.return_value.all.return_value = []

    service = TriangulationService(mock_session, mock_research_service)
    # Use a non-existent UUID to ensure empty results
    result = service.find_triangulation_opportunities("NONEXISTENT-UUID")
    assert isinstance(result, list), "Should return a list"
    assert len(result) == 0, "Should return empty list for unknown person"


def _test_menu_rendering() -> None:
    """Test menu rendering function produces expected output."""
    output = _render_triangulation_menu(None, 0)
    assert output is not None, "Menu should return a choice"


def module_tests() -> bool:
    """Run module-specific tests."""
    suite = TestSuite("Action 13 - Triangulation", __file__)
    suite.start_suite()

    suite.run_test(
        "TriangulationService initialization",
        _test_triangulation_service_integration,
        "Verify service initialization with dependencies",
    )

    suite.run_test(
        "find_triangulation_groups empty result",
        _test_find_triangulation_groups_no_matches,
        "Verify empty result for unknown person UUID",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    print("🧬 Running Action 13 Tests...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
