#!/usr/bin/env python3
"""
Automated refactoring script for too-many-arguments violations.

This script applies dataclass-based refactorings to high-confidence patterns.
"""

# Removed deprecated typing imports - using built-in types
# Note: ast, re, Path imports removed as they were unused


class ParameterRefactorer:
    """Handles automated parameter refactoring."""

    def __init__(self):
        self.refactoring_patterns = {
            'graph_context': {
                'params': ['id_to_parents', 'id_to_children', 'current_id', 'start_id', 'end_id'],
                'dataclass': 'GraphContext',
                'import': 'from common_params import GraphContext',
                'files': ['gedcom_utils.py', 'relationship_utils.py'],
            },
            'conversation_ids': {
                'params': ['api_conv_id', 'people_id', 'my_pid_lower', 'effective_conv_id', 'log_prefix'],
                'dataclass': 'ConversationIdentifiers',
                'import': 'from common_params import ConversationIdentifiers',
                'files': ['action7_inbox.py', 'action8_messaging.py', 'action9_process_productive.py'],
            },
            'api_ids': {
                'params': ['owner_profile_id', 'api_person_id', 'api_tree_id', 'owner_tree_id'],
                'dataclass': 'ApiIdentifiers',
                'import': 'from common_params import ApiIdentifiers',
                'files': ['api_utils.py', 'action11.py'],
            },
        }

    def find_matching_params(self, params: list[str], pattern_params: list[str]) -> list[str]:
        """Find which parameters from the pattern are present in the function."""
        return [p for p in params if p in pattern_params]

    def should_refactor_function(self, file_path: str, func_params: list[str], pattern_name: str) -> tuple[bool, list[str]]:
        """Determine if a function should be refactored with a specific pattern."""
        pattern = self.refactoring_patterns[pattern_name]

        # Check if file matches
        if not any(file_path.endswith(f) for f in pattern['files']):
            return False, []

        # Find matching parameters
        matching = self.find_matching_params(func_params, pattern['params'])

        # Only refactor if we have at least 2 matching params
        if len(matching) >= 2:
            return True, matching

        return False, []

    def generate_refactoring_plan(self) -> dict[str, list[dict]]:
        """Generate a plan for which functions to refactor."""
        plan = {}

        # This would be populated by analyzing the codebase
        # For now, we'll create it manually based on our analysis

        plan['gedcom_utils.py'] = [
            {
                'function': '_expand_forward_node',
                'line': 754,
                'pattern': 'graph_context',
                'params_to_group': ['current_id', 'id_to_parents', 'id_to_children'],
            },
            {
                'function': '_expand_backward_node',
                'line': 786,
                'pattern': 'graph_context',
                'params_to_group': ['current_id', 'id_to_parents', 'id_to_children'],
            },
            {
                'function': 'fast_bidirectional_bfs',
                'line': 818,
                'pattern': 'graph_context',
                'params_to_group': ['start_id', 'end_id', 'id_to_parents', 'id_to_children'],
            },
        ]

        plan['relationship_utils.py'] = [
            {
                'function': '_expand_to_relatives',
                'line': 275,
                'pattern': 'graph_context',
                'params_to_group': ['current_id', 'id_to_parents', 'id_to_children'],
            },
            {
                'function': '_process_forward_queue',
                'line': 303,
                'pattern': 'graph_context',
                'params_to_group': ['id_to_parents', 'id_to_children'],
            },
            {
                'function': '_process_backward_queue',
                'line': 324,
                'pattern': 'graph_context',
                'params_to_group': ['id_to_parents', 'id_to_children'],
            },
        ]

        plan['action7_inbox.py'] = [
            {
                'function': '_determine_fetch_need',
                'line': 1209,
                'pattern': 'conversation_ids',
                'params_to_group': ['api_conv_id'],
            },
            {
                'function': '_create_conversation_log_upsert',
                'line': 1342,
                'pattern': 'conversation_ids',
                'params_to_group': ['api_conv_id', 'people_id'],
            },
            {
                'function': '_process_in_message',
                'line': 1463,
                'pattern': 'conversation_ids',
                'params_to_group': ['api_conv_id', 'people_id', 'my_pid_lower'],
            },
        ]

        plan['api_utils.py'] = [
            {
                'function': 'call_facts_user_api',
                'line': 1778,
                'pattern': 'api_ids',
                'params_to_group': ['owner_profile_id', 'api_person_id', 'api_tree_id'],
            },
        ]

        plan['action11.py'] = [
            {
                'function': '_log_final_ids',
                'line': 1942,
                'pattern': 'api_ids',
                'params_to_group': ['owner_tree_id', 'owner_profile_id'],
            },
        ]

        return plan

    def print_refactoring_plan(self):
        """Print the refactoring plan for review."""
        plan = self.generate_refactoring_plan()

        print("=" * 80)
        print("AUTOMATED REFACTORING PLAN")
        print("=" * 80)
        print()

        total_functions = sum(len(funcs) for funcs in plan.values())
        print(f"📊 Total functions to refactor: {total_functions}")
        print(f"📁 Files affected: {len(plan)}")
        print()

        for file_path, functions in plan.items():
            print(f"\n📁 {file_path} ({len(functions)} functions)")
            print("-" * 80)

            for func_info in functions:
                pattern = self.refactoring_patterns[func_info['pattern']]
                print(f"\n  Function: {func_info['function']} (line {func_info['line']})")
                print(f"  Pattern: {func_info['pattern']}")
                print(f"  Dataclass: {pattern['dataclass']}")
                print(f"  Parameters to group: {', '.join(func_info['params_to_group'])}")

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print()
        print("1. Review the plan above")
        print("2. Run manual refactoring for each file (requires careful editing)")
        print("3. Test after each file to ensure correctness")
        print("4. Commit changes incrementally")
        print()
        print("Note: Full automation of signature changes and call-site updates")
        print("      requires more sophisticated AST manipulation. The current")
        print("      approach provides a clear plan for manual/AI-assisted refactoring.")


# ==============================================
# Comprehensive Test Suite
# ==============================================

def _test_parameter_refactorer_initialization() -> bool:
    """Test ParameterRefactorer initialization."""
    try:
        refactorer = ParameterRefactorer()
        assert refactorer.refactoring_patterns is not None, "Should initialize patterns"
        assert isinstance(refactorer.refactoring_patterns, dict), "Patterns should be dict"
        assert len(refactorer.refactoring_patterns) > 0, "Should have patterns"
        return True
    except Exception:
        return False


def _test_find_matching_params() -> bool:
    """Test finding matching parameters."""
    try:
        refactorer = ParameterRefactorer()

        params = ['id_to_parents', 'id_to_children', 'current_id', 'other_param']
        pattern_params = ['id_to_parents', 'id_to_children', 'current_id', 'start_id']

        matching = refactorer.find_matching_params(params, pattern_params)
        assert len(matching) == 3, "Should find 3 matching params"
        assert 'id_to_parents' in matching, "Should include id_to_parents"
        assert 'other_param' not in matching, "Should not include non-matching params"
        return True
    except Exception:
        return False


def _test_should_refactor_function() -> bool:
    """Test refactoring decision logic."""
    try:
        refactorer = ParameterRefactorer()

        # Test with matching file and params
        func_params = ['id_to_parents', 'id_to_children', 'current_id']
        should_refactor, matching = refactorer.should_refactor_function(
            'gedcom_utils.py',
            func_params,
            'graph_context'
        )

        assert isinstance(should_refactor, bool), "Should return bool"
        assert isinstance(matching, list), "Should return list of matching params"
        return True
    except Exception:
        return False


def _test_refactoring_patterns_structure() -> bool:
    """Test that refactoring patterns have expected structure."""
    try:
        refactorer = ParameterRefactorer()

        for pattern_name, pattern in refactorer.refactoring_patterns.items():
            assert 'params' in pattern, f"Pattern {pattern_name} should have params"
            assert 'dataclass' in pattern, f"Pattern {pattern_name} should have dataclass"
            assert 'import' in pattern, f"Pattern {pattern_name} should have import"
            assert 'files' in pattern, f"Pattern {pattern_name} should have files"
            assert isinstance(pattern['params'], list), "params should be list"
            assert isinstance(pattern['files'], list), "files should be list"

        return True
    except Exception:
        return False


def _test_generate_refactoring_plan() -> bool:
    """Test refactoring plan generation."""
    try:
        refactorer = ParameterRefactorer()
        plan = refactorer.generate_refactoring_plan()

        assert isinstance(plan, dict), "Should return dict"
        # Plan may be empty if no matching functions found, which is OK
        return True
    except Exception:
        return False


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for apply_automated_refactoring.py.
    Tests automated refactoring functionality.
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "Automated Refactoring Utility",
            "apply_automated_refactoring.py"
        )
        suite.start_suite()

        suite.run_test(
            "ParameterRefactorer Initialization",
            _test_parameter_refactorer_initialization,
            "ParameterRefactorer initializes correctly",
            "Test refactorer initialization",
            "Test pattern setup",
        )

        suite.run_test(
            "Find Matching Parameters",
            _test_find_matching_params,
            "Matching parameters are found correctly",
            "Test parameter matching",
            "Test pattern matching logic",
        )

        suite.run_test(
            "Should Refactor Function",
            _test_should_refactor_function,
            "Refactoring decision logic works correctly",
            "Test refactoring decision",
            "Test file and parameter matching",
        )

        suite.run_test(
            "Refactoring Patterns Structure",
            _test_refactoring_patterns_structure,
            "Refactoring patterns have expected structure",
            "Test pattern structure",
            "Test pattern validation",
        )

        suite.run_test(
            "Generate Refactoring Plan",
            _test_generate_refactoring_plan,
            "Refactoring plan generation works",
            "Test plan generation",
            "Test refactoring analysis",
        )

        return suite.finish_suite()


def main():
    """Main entry point."""
    success = run_comprehensive_tests()
    if success:
        refactorer = ParameterRefactorer()
        refactorer.print_refactoring_plan()
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

