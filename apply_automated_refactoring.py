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


def main():
    """Main entry point."""
    refactorer = ParameterRefactorer()
    refactorer.print_refactoring_plan()


if __name__ == "__main__":
    main()

