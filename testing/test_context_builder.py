#!/usr/bin/env python3
"""
Tests for ContextBuilder module.
"""

import sys
from unittest.mock import MagicMock

from ai.context_builder import ContextBuilder, MatchContext
from testing.test_framework import TestSuite


def module_tests() -> bool:
    """Run all tests for ContextBuilder."""
    suite = TestSuite("ContextBuilder", "ai/context_builder.py")
    suite.start_suite()

    # Test 1: MatchContext dataclass
    def test_match_context():
        context = MatchContext(
            identity={"name": "John Smith"},
            genetics={"shared_cm": 150},
        )
        assert context.identity["name"] == "John Smith"
        assert context.genetics["shared_cm"] == 150

        # Test serialization
        d = context.to_dict()
        assert "identity" in d
        assert "genetics" in d

        # Test JSON
        j = context.to_json()
        assert "John Smith" in j

    suite.run_test(
        "MatchContext dataclass",
        test_match_context,
        test_summary="Verify MatchContext dataclass and serialization",
    )

    # Test 2: MatchContext prompt string
    def test_match_context_prompt():
        context = MatchContext(
            identity={"name": "Jane Doe", "managed_by": "Self"},
            genetics={"shared_cm": 200, "segments": 10, "prediction": "3rd Cousin"},
            history={"last_interaction_date": "2025-01-15", "messages": []},
        )
        prompt = context.to_prompt_string()
        assert "Jane Doe" in prompt
        assert "200 cM" in prompt
        assert "3rd Cousin" in prompt

    suite.run_test(
        "MatchContext prompt string",
        test_match_context_prompt,
        test_summary="Verify MatchContext converts to prompt-friendly string",
    )

    # Test 3: ContextBuilder initialization
    def test_context_builder_init():
        # Test with Mock session
        mock_session = MagicMock()
        builder = ContextBuilder(db_session=mock_session)
        assert builder._session is mock_session
        assert builder._tree_service_initialized is False

    suite.run_test(
        "ContextBuilder initialization",
        test_context_builder_init,
        test_summary="Verify ContextBuilder initializes correctly",
    )

    # Test 4: Genetics bucket calculation
    def test_genetics_bucket():
        # Test close family
        person = MagicMock()
        person.shared_cm = 1600
        person.segments = 50
        person.predicted_relationship = "Parent/Child"

        genetics = ContextBuilder._build_genetics(person)
        assert genetics["relationship_bucket"] == "Close family (parent/child/sibling)"

        # Test distant
        person.shared_cm = 50
        genetics = ContextBuilder._build_genetics(person)
        assert genetics["relationship_bucket"] == "Remote relative (5th+ cousin)"

    suite.run_test(
        "Genetics relationship bucket",
        test_genetics_bucket,
        test_summary="Verify genetics bucket calculation based on shared cM",
    )

    return suite.finish_suite()


run_comprehensive_tests = module_tests


if __name__ == "__main__":
    success = module_tests()
    sys.exit(0 if success else 1)
