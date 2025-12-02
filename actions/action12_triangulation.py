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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import config_schema
from core.logging_utils import log_action_banner
from core.session_manager import SessionManager
from genealogy.research_service import ResearchService
from genealogy.triangulation import TriangulationService
from research.search_criteria_utils import get_unified_search_criteria
from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


def _create_date_objects(criteria: dict[str, Any]) -> dict[str, Any]:
    """Create date objects from year criteria."""
    birth_date_obj_crit: Optional[datetime] = None
    if criteria["birth_year"]:
        try:
            birth_date_obj_crit = datetime(criteria["birth_year"], 1, 1, tzinfo=timezone.utc)
        except ValueError:
            logger.warning(f"Cannot create date object for birth year {criteria['birth_year']}.")
            criteria["birth_year"] = None

    death_date_obj_crit: Optional[datetime] = None
    if criteria["death_year"]:
        try:
            death_date_obj_crit = datetime(criteria["death_year"], 1, 1, tzinfo=timezone.utc)
        except ValueError:
            logger.warning(f"Cannot create date object for death year {criteria['death_year']}.")
            criteria["death_year"] = None

    criteria["birth_date_obj"] = birth_date_obj_crit
    criteria["death_date_obj"] = death_date_obj_crit
    return criteria


def _build_filter_criteria(scoring_criteria: dict[str, Any]) -> dict[str, Any]:
    """Build filter criteria from scoring criteria (case-insensitive for strings)."""
    fn = scoring_criteria.get("first_name")
    sn = scoring_criteria.get("surname")
    bp = scoring_criteria.get("birth_place")
    dp = scoring_criteria.get("death_place")
    return {
        "first_name": fn.lower() if isinstance(fn, str) else fn,
        "surname": sn.lower() if isinstance(sn, str) else sn,
        "birth_year": scoring_criteria.get("birth_year"),
        "birth_place": bp.lower() if isinstance(bp, str) else bp,
        "death_place": dp.lower() if isinstance(dp, str) else dp,
    }


def _get_scoring_config() -> tuple[dict[str, Any], dict[str, Any]]:
    """Get scoring weights and date flexibility from config."""
    date_flexibility_value = config_schema.date_flexibility if config_schema else 2
    date_flex = {"year_match_range": int(date_flexibility_value)}

    scoring_weights = (
        dict(config_schema.common_scoring_weights)
        if config_schema
        else {
            "name_match": 50,
            "birth_year_match": 30,
            "birth_place_match": 20,
            "death_year_match": 25,
            "death_place_match": 15,
        }
    )
    return date_flex, scoring_weights


def _select_match_from_results(matches: list[dict[str, Any]]) -> Optional[str]:
    """Display matches and let user select one."""
    if not matches:
        print("No matches found.")
        return None

    print(f"\nFound {len(matches)} matches:")
    for i, match in enumerate(matches[:5], 1):  # Show top 5
        print(f"{i}. {match['full_name_disp']} (Score: {match['total_score']:.1f})")
        print(f"   ID: {match['display_id']}")
        print(f"   Born: {match['birth_date']} {match['birth_place'] or ''}")
        print(f"   Died: {match['death_date']} {match['death_place'] or ''}")

    while True:
        choice = input("\nSelect a match (number) or 'n' for none: ").strip().lower()
        if choice == "n":
            return None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(matches[:5]):
                return matches[idx]["id"]  # Return the normalized ID (UUID)
        print("Invalid choice.")


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
        # 1. Initialize Services
        gedcom_path = str(config_schema.database.gedcom_file_path) if config_schema.database.gedcom_file_path else None
        research_service = ResearchService(gedcom_path)
        triangulation_service = TriangulationService(session, research_service)

        # 2. Get Target Person
        print("\n--- Triangulation Analysis ---")
        target_input = None

        while not target_input:
            mode = input("Search by name (s) or enter ID (i)? [s/i/q]: ").strip().lower()
            if mode == "q":
                return True
            elif mode == "i":
                target_input = input("Enter Target Person UUID or Profile ID: ").strip()
            elif mode == "s":
                # Use unified search criteria collection
                basic_criteria = get_unified_search_criteria()
                if not basic_criteria:
                    continue

                # Prepare criteria for scoring
                scoring_criteria = _create_date_objects(basic_criteria)
                filter_criteria = _build_filter_criteria(scoring_criteria)
                date_flex, scoring_weights = _get_scoring_config()

                # Perform search
                matches = research_service.search_people(
                    filter_criteria=filter_criteria,
                    scoring_criteria=scoring_criteria,
                    scoring_weights=scoring_weights,
                    date_flex=date_flex,
                )

                target_input = _select_match_from_results(matches)
            else:
                print("Invalid choice.")

        if not target_input:
            return True

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
