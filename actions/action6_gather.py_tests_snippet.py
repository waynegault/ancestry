# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================


def _test_extract_match_info() -> None:
    """Test extraction of match info from dictionary."""
    match_data = {
        "uuid": "test-uuid-123",
        "username": "Test User",
        "predicted_relationship": "Parent/Child",
        "in_my_tree": True,
    }
    info = _extract_match_info(match_data)
    assert info.uuid == "test-uuid-123"
    assert info.username == "Test User"
    assert info.predicted_relationship == "Parent/Child"
    assert info.in_my_tree is True
    assert "UUID=test-uuid-123" in info.log_ref_short


def _test_determine_tree_operation() -> None:
    """Test logic for determining tree operations."""
    # Mock logger
    logger_mock = logging.getLogger("test_logger")

    # Case 1: Match in tree, no existing record, has data -> create
    op = _determine_tree_operation(
        match_in_my_tree=True,
        existing_family_tree=None,
        prefetched_tree_data={"some": "data"},
        their_cfpid_final="cfpid-123",
        facts_link="http://link",
        view_in_tree_link="http://view",
        log_ref_short="test",
        logger_instance=logger_mock,
    )
    assert op == "create"

    # Case 2: Match in tree, no existing record, no data -> none
    op = _determine_tree_operation(
        match_in_my_tree=True,
        existing_family_tree=None,
        prefetched_tree_data=None,
        their_cfpid_final=None,
        facts_link=None,
        view_in_tree_link=None,
        log_ref_short="test",
        logger_instance=logger_mock,
    )
    assert op == "none"


def _test_check_tree_update_needed() -> None:
    """Test logic for checking if tree update is needed."""
    # Mock logger
    logger_mock = logging.getLogger("test_logger")

    # Mock existing tree
    existing_tree = FamilyTree()
    existing_tree.cfpid = "old-cfpid"
    existing_tree.person_name_in_tree = "Old Name"

    # Case 1: Data changed -> True
    needed = _check_tree_update_needed(
        existing_family_tree=existing_tree,
        prefetched_tree_data={"their_firstname": "New Name"},
        their_cfpid_final="old-cfpid",
        facts_link=None,
        view_in_tree_link=None,
        log_ref_short="test",
        logger_instance=logger_mock,
    )
    assert needed is True

    # Case 2: Data same -> False
    needed = _check_tree_update_needed(
        existing_family_tree=existing_tree,
        prefetched_tree_data={"their_firstname": "Old Name"},
        their_cfpid_final="old-cfpid",
        facts_link=None,
        view_in_tree_link=None,
        log_ref_short="test",
        logger_instance=logger_mock,
    )
    assert needed is False


# ==============================================
# MAIN TEST SUITE RUNNER
# ==============================================


def action6_gather_module_tests() -> bool:
    """Comprehensive test suite for action6_gather.py."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Action 6 Gather Logic", "action6_gather.py")
    suite.start_suite()

    suite.run_test(
        "Extract Match Info",
        _test_extract_match_info,
        "Correctly extracts match info from dictionary",
        "Test _extract_match_info helper",
        "Verify UUID, username, and flags are parsed correctly",
    )

    suite.run_test(
        "Determine Tree Operation",
        _test_determine_tree_operation,
        "Correctly determines create/update/none operation",
        "Test _determine_tree_operation logic",
        "Verify logic for creating vs updating tree records",
    )

    suite.run_test(
        "Check Tree Update Needed",
        _test_check_tree_update_needed,
        "Correctly identifies when update is needed",
        "Test _check_tree_update_needed logic",
        "Verify field comparison logic",
    )

    return suite.finish_suite()


# Use centralized test runner utility from test_utilities
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(action6_gather_module_tests)

if __name__ == "__main__":
    import sys

    # Ensure project root is in path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
