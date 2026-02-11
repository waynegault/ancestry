import sys
from typing import Any
from unittest.mock import MagicMock, patch

from genealogy.research_service import ResearchService
from testing.test_framework import TestSuite, suppress_logging


def research_service_tests() -> bool:
    """Run tests for the ResearchService module."""
    with suppress_logging():
        suite = TestSuite("Research Service", __name__)
        suite.start_suite()

        def test_load_gedcom_success():
            with (
                patch("genealogy.research_service.load_gedcom_with_aggressive_caching") as mock_load,
                patch("pathlib.Path.exists", return_value=True),
            ):
                mock_data = MagicMock()
                mock_data.indi_index = {"I1": "data"}
                mock_load.return_value = mock_data

                service = ResearchService("test.ged")

                assert service.gedcom_data is not None
                assert len(service.gedcom_data.indi_index) == 1
                mock_load.assert_called_once()

        def test_load_gedcom_failure():
            with (
                patch("genealogy.research_service.load_gedcom_with_aggressive_caching") as mock_load,
                patch("pathlib.Path.exists", return_value=False),
            ):
                service = ResearchService("nonexistent.ged")
                assert service.gedcom_data is None
                mock_load.assert_not_called()

        def test_search_people_no_data():
            service = ResearchService()
            results = service.search_people({}, {}, {}, {})
            assert results == []

        def test_search_people_with_matches():
            with (
                patch("genealogy.research_service.load_gedcom_with_aggressive_caching") as mock_load,
                patch("pathlib.Path.exists", return_value=True),
                patch("genealogy.research_service.calculate_match_score") as mock_score,
            ):
                # Setup mock data
                mock_data = MagicMock()
                mock_data.processed_data_cache = {
                    "I1": {
                        "first_name": "John",
                        "surname": "Doe",
                        "birth_year": 1900,
                        "full_name_disp": "John Doe",
                    },
                    "I2": {
                        "first_name": "Jane",
                        "surname": "Smith",
                        "birth_year": 1905,
                        "full_name_disp": "Jane Smith",
                    },
                }
                mock_load.return_value = mock_data

                # Setup mock score
                # Return a high score for John, low for Jane
                def score_side_effect(**kwargs: Any) -> tuple[float, dict[str, float], list[str]]:
                    candidate = kwargs.get("candidate_processed_data", {})
                    if candidate.get("first_name") == "John":
                        return (100.0, {"first_name": 100.0}, ["Match"])
                    return (0.0, {}, [])

                mock_score.side_effect = score_side_effect

                service = ResearchService("test.ged")

                # Test search
                filter_criteria = {"first_name": "John"}
                scoring_criteria = {"first_name": "John"}
                scoring_weights = {"first_name": 100}
                date_flex = {"year_match_range": 2}

                results = service.search_people(
                    filter_criteria,
                    scoring_criteria,
                    scoring_weights,
                    date_flex,
                )

                assert len(results) == 1
                assert results[0]["full_name_disp"] == "John Doe"
                assert results[0]["total_score"] == 100.0

        suite.run_test("Load GEDCOM Success", test_load_gedcom_success, "Should load data when file exists")
        suite.run_test("Load GEDCOM Failure", test_load_gedcom_failure, "Should handle missing file gracefully")
        suite.run_test("Search No Data", test_search_people_no_data, "Should return empty list if no data loaded")
        suite.run_test("Search With Matches", test_search_people_with_matches, "Should find matching individuals")

        return suite.finish_suite()


# Standard test runner for test discovery
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(research_service_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
