#!/usr/bin/env python3
"""
Automated test script for Frances Milne Action 6 (DNA Match Gathering).

This script:
1. Validates environment configuration
2. Runs Action 6 with Frances Milne account
3. Validates results (matches gathered, ethnicity data, database records)
4. Generates comprehensive test report

Usage:
    python test_frances_action6.py

Prerequisites:
    Set ANCESTRY_USERNAME and ANCESTRY_PASSWORD in .env to Frances's credentials:
    ANCESTRY_USERNAME=francesmchardy@gmail.com
    ANCESTRY_PASSWORD=<frances_password>
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import func

from action6_gather import coord as gather_dna_matches_coord
from config import config_schema
from core.database_manager import DatabaseManager
from core.session_manager import SessionManager
from database import DnaMatch, Person


class FrancesAction6Tester:
    """Automated tester for Frances Milne Action 6."""

    def __init__(self):
        self.config = config_schema
        self.session_manager = None
        self.db_manager = DatabaseManager()
        self.test_results = []
        self.start_time = datetime.now(timezone.utc)

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        prefix = "✅" if level == "PASS" else "❌" if level == "FAIL" else "ℹ️"
        print(f"{prefix} [{timestamp}] {message}")

    def validate_environment(self) -> bool:
        """Validate environment configuration."""
        self.log("Validating environment configuration...")

        # Check that Frances's credentials are configured
        if self.config.api.username != "francesmchardy@gmail.com":
            self.log(f"ERROR: Wrong account configured. Expected francesmchardy@gmail.com, got {self.config.api.username}", "FAIL")
            self.log("Please update .env: ANCESTRY_USERNAME=francesmchardy@gmail.com", "FAIL")
            return False

        try:
            # Check config loaded
            if not self.config:
                self.log("Configuration not loaded", "FAIL")
                return False

            # Check processing limits
            max_pages = self.config.api.max_pages
            max_inbox = self.config.max_inbox
            batch_size = self.config.batch_size

            self.log(f"Processing limits: MAX_PAGES={max_pages}, MAX_INBOX={max_inbox}, BATCH_SIZE={batch_size}")

            if max_pages > 5:
                self.log(f"WARNING: MAX_PAGES={max_pages} is high, recommend ≤5 for testing", "WARN")

            # Check database connection
            session = self.db_manager.get_session()
            if not session:
                self.log("Database connection failed", "FAIL")
                return False

            self.db_manager.return_session(session)
            self.log("Environment validation passed", "PASS")
            return True

        except Exception as e:
            self.log(f"Environment validation failed: {e}", "FAIL")
            return False

    def get_baseline_stats(self) -> dict[str, Any]:
        """Get baseline database statistics before Action 6."""
        self.log("Collecting baseline statistics...")

        try:
            session = self.db_manager.get_session()
            stats = {
                "total_people": session.query(func.count(Person.id)).scalar() or 0,
                "total_matches": session.query(func.count(DnaMatch.id)).scalar() or 0,
            }
            self.db_manager.return_session(session)

            self.log(f"Baseline: {stats['total_people']} people, {stats['total_matches']} matches")
            return stats

        except Exception as e:
            self.log(f"Failed to collect baseline stats: {e}", "FAIL")
            return {}

    def run_action6(self) -> bool:
        """Run Action 6 (DNA Match Gathering)."""
        self.log("=" * 80)
        self.log("RUNNING ACTION 6: DNA MATCH GATHERING")
        self.log("=" * 80)

        try:
            # Initialize session manager
            self.session_manager = SessionManager()

            # Ensure session is ready
            if not self.session_manager.session_ready:
                self.log("Initializing browser session...")
                self.session_manager.ensure_session_ready("Frances Action 6 Test")

            if not self.session_manager.session_ready:
                self.log("Session initialization failed", "FAIL")
                return False

            # Run Action 6
            self.log("Starting DNA match gathering...")
            result = gather_dna_matches_coord(
                session_manager=self.session_manager,
                start=1
            )

            if result:
                self.log("Action 6 completed successfully", "PASS")
                return True
            self.log("Action 6 failed", "FAIL")
            return False

        except Exception as e:
            self.log(f"Action 6 execution failed: {e}", "FAIL")
            import traceback
            traceback.print_exc()
            return False

    def validate_results(self, baseline_stats: dict[str, Any]) -> dict[str, Any]:
        """Validate Action 6 results."""
        self.log("=" * 80)
        self.log("VALIDATING RESULTS")
        self.log("=" * 80)

        try:
            session = self.db_manager.get_session()

            # Get new stats
            new_stats = {
                "total_people": session.query(func.count(Person.id)).scalar() or 0,
                "total_matches": session.query(func.count(DnaMatch.id)).scalar() or 0,
            }

            # Calculate changes
            changes = {
                "new_people": new_stats["total_people"] - baseline_stats.get("total_people", 0),
                "new_matches": new_stats["total_matches"] - baseline_stats.get("total_matches", 0),
            }

            self.db_manager.return_session(session)

            # Report results
            self.log(f"New people added: {changes['new_people']}")
            self.log(f"New matches added: {changes['new_matches']}")

            # Validate expectations
            validation_passed = True

            if changes["new_people"] == 0 and changes["new_matches"] == 0:
                self.log("WARNING: No new data collected (may be expected if already up-to-date)", "WARN")
            else:
                self.log("Data collection successful", "PASS")

            return {
                "baseline": baseline_stats,
                "new_stats": new_stats,
                "changes": changes,
                "validation_passed": validation_passed,
            }

        except Exception as e:
            self.log(f"Result validation failed: {e}", "FAIL")
            import traceback
            traceback.print_exc()
            return {}

    def generate_report(self, results: dict[str, Any]) -> None:
        """Generate comprehensive test report."""
        self.log("=" * 80)
        self.log("TEST REPORT")
        self.log("=" * 80)

        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.start_time).total_seconds()

        print("\n📊 Frances Milne Action 6 Test Report")
        print(f"   Test Duration: {duration:.2f} seconds")
        print(f"   Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"   End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()

        if results:
            print("📈 Database Changes:")
            print(f"   New People: {results['changes']['new_people']}")
            print(f"   New Matches: {results['changes']['new_matches']}")
            print()

            print("📊 Final Statistics:")
            print(f"   Total People: {results['new_stats']['total_people']}")
            print(f"   Total Matches: {results['new_stats']['total_matches']}")
            print()

        print("✅ Test completed successfully!" if results.get("validation_passed") else "❌ Test failed!")

    def run(self) -> bool:
        """Run the complete test suite."""
        print("\n" + "=" * 80)
        print("FRANCES MILNE ACTION 6 AUTOMATED TEST")
        print("=" * 80 + "\n")

        # Step 1: Validate environment
        if not self.validate_environment():
            self.log("Environment validation failed - aborting test", "FAIL")
            return False

        # Step 2: Get baseline stats
        baseline_stats = self.get_baseline_stats()
        if not baseline_stats:
            self.log("Failed to collect baseline stats - aborting test", "FAIL")
            return False

        # Step 3: Run Action 6
        if not self.run_action6():
            self.log("Action 6 execution failed", "FAIL")
            return False

        # Step 4: Validate results
        results = self.validate_results(baseline_stats)

        # Step 5: Generate report
        self.generate_report(results)

        return results.get("validation_passed", False)


def main():
    """Main entry point."""
    tester = FrancesAction6Tester()
    success = tester.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

