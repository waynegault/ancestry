import sys
from pathlib import Path

# Add project root to Python path to allow running script directly
sys.path.append(str(Path(__file__).parent.parent))

from unittest.mock import MagicMock

from core.database import Person
from genealogy.triangulation import TriangulationService
from testing.test_framework import TestSuite, suppress_logging
from testing.test_utilities import create_standard_test_runner


def triangulation_tests() -> bool:
    """Run tests for the TriangulationService module."""
    with suppress_logging():
        suite = TestSuite("Triangulation Service", __name__)
        suite.start_suite()

        def test_filtering_logic():
            # Mock DB session
            mock_db = MagicMock()
            mock_research_service = MagicMock()

            service = TriangulationService(mock_db, mock_research_service)

            # Mock target person
            target_person = Person(id=1, uuid="TARGET_UUID", profile_id="TARGET_PROFILE")

            # Mock _resolve_person
            service._resolve_person = MagicMock(return_value=target_person)

            # Mock _get_shared_matches
            service._get_shared_matches = MagicMock(return_value=[])

            # Test with default params (HIGH -> 30)
            service.find_triangulation_opportunities("TARGET_UUID")
            service._get_shared_matches.assert_called_with(target_person, min_cm=30)

            # Test with explicit min_cm
            service.find_triangulation_opportunities("TARGET_UUID", min_cm=50)
            service._get_shared_matches.assert_called_with(target_person, min_cm=50)

            # Test with min_confidence
            service.find_triangulation_opportunities("TARGET_UUID", min_confidence="EXTREMELY_HIGH")
            service._get_shared_matches.assert_called_with(target_person, min_cm=60)

            # Test with both (max wins)
            service.find_triangulation_opportunities("TARGET_UUID", min_confidence="GOOD", min_cm=25)
            # GOOD is 20, min_cm is 25 -> 25
            service._get_shared_matches.assert_called_with(target_person, min_cm=25)

            return True

        suite.run_test(
            "Test filtering logic parameters",
            test_filtering_logic,
            test_summary="Verify filtering by min_cm and min_confidence",
            expected_outcome="Correct parameters passed to _get_shared_matches",
        )

        return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(triangulation_tests)


if __name__ == "__main__":
    triangulation_tests()
