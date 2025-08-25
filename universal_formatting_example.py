# Universal Test Formatting Example

"""
Example showing how any test module can use the universal formatting functions
from test_framework.py to create consistent, beautiful test output.
"""

from test_framework import (
    Colors,
    Icons,
    TestSuite,
    clean_test_output,
    format_score_breakdown_table,
    format_search_criteria,
    format_test_result,
    format_test_section_header,
)


def example_test_with_beautiful_formatting() -> bool:
    """Example test showing universal formatting usage"""

    # Create test suite
    suite = TestSuite("Example Universal Formatting", "example.py")
    suite.start_suite()

    # Beautiful section header
    print(format_test_section_header("Database Connection Test", "🔗"))

    # Clean search criteria display
    search_criteria = {
        "database": "ancestry_db",
        "table": "individuals",
        "max_results": 1000,
        "timeout_seconds": 30
    }
    print(format_search_criteria(search_criteria))

    # Simulate some test work with clean output (no debug noise)
    print(f"\n{Colors.CYAN}🔄 Executing database query...{Colors.RESET}")

    with clean_test_output():
        # This would suppress debug logging during the actual test
        # result = some_database_function()
        pass

    # Display results in a beautiful table
    performance_scores = {
        "connection": 25,
        "query_speed": 20,
        "result_accuracy": 15,
        "timeout_handling": 10,
        "error_handling": 5
    }

    print(format_score_breakdown_table(performance_scores, 75))

    # Format test results consistently
    print(f"\n{format_test_result('Database Connection', True, 2.45)}")
    print(f"{format_test_result('Query Performance', True, 0.89)}")
    print(f"{format_test_result('Error Handling', False, 1.23)}")

    # Beautiful final status
    print(f"\n{Colors.BOLD}{Colors.GREEN}{Icons.PASS} All core tests completed!{Colors.RESET}")

    suite.finish_suite()

def example_api_test_formatting() -> None:
    """Example showing API test formatting"""

    print(format_test_section_header("API Response Validation", "🌐"))

    # Test parameters
    api_params = {
        "endpoint": "/api/v1/search",
        "method": "POST",
        "timeout": 10,
        "retry_count": 3
    }
    print(format_search_criteria(api_params))

    # Simulate API response scoring
    api_scores = {
        "status_code": 25,    # 200 OK
        "response_time": 20,  # < 500ms
        "data_format": 15,    # Valid JSON
        "field_count": 10,    # All expected fields
        "data_quality": 0     # Some issues found
    }

    print(format_score_breakdown_table(api_scores, 70))

    print(f"\n{Colors.YELLOW}{Icons.WARNING} API Quality Score: 70/100 - Review data quality{Colors.RESET}")

def example_file_processing_test() -> None:
    """Example showing file processing test formatting"""

    print(format_test_section_header("GEDCOM File Processing", "📁"))

    file_criteria = {
        "input_file": "family_tree.ged",
        "expected_individuals": 15000,
        "expected_families": 4500,
        "encoding": "UTF-8"
    }
    print(format_search_criteria(file_criteria))

    # Processing scores
    processing_scores = {
        "parse_success": 25,
        "individual_count": 20,
        "family_count": 20,
        "relationship_links": 15,
        "data_validation": 10,
        "performance": 5
    }

    print(format_score_breakdown_table(processing_scores, 95))

    print(f"\n{Colors.GREEN}{Icons.PASS} File processing completed successfully!{Colors.RESET}")

if __name__ == "__main__":
    print("=" * 60)
    print("🎨 UNIVERSAL TEST FORMATTING EXAMPLES")
    print("=" * 60)
    print("These examples show how ANY test module can use the universal")
    print("formatting functions to create beautiful, consistent output.\n")

    example_test_with_beautiful_formatting()
    print("\n" + "─" * 60 + "\n")

    example_api_test_formatting()
    print("\n" + "─" * 60 + "\n")

    example_file_processing_test()

    print(f"\n{Colors.BOLD}{Colors.CYAN}✨ All examples use the same universal formatting functions!{Colors.RESET}")
    print("Any module can import these from test_framework.py for consistent styling.")
