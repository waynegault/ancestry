#!/usr/bin/env python3
"""
Additional Action 8 hardening tests integrated from test_action8_hardening.py
This file is temporary and will be merged into action8_messaging.py Standalone Test Block.
"""
from unittest.mock import Mock

from action8_messaging import (
    MESSAGE_TEMPLATES,
    MESSAGE_TYPES_ACTION8,
    ErrorCategorizer,
    ProactiveApiManager,
    _validate_system_health,
    select_template_by_confidence,
)


def run_extra_action8_hardening_tests() -> bool:
    ok = True

    # System health validation (hardening)
    try:
        ok &= (_validate_system_health(None) is False)  # type: ignore[arg-type]
        mock_session = Mock()
        mock_session.should_halt_operations.return_value = False
        mock_session.validate_system_health.return_value = True
        mock_session.session_health_monitor = {'death_cascade_count': 0}
        for k in set(MESSAGE_TYPES_ACTION8.keys()):
            MESSAGE_TEMPLATES.setdefault(k, "test template")
        ok &= (_validate_system_health(mock_session) is True)
        mock_session.should_halt_operations.return_value = True
        mock_session.validate_system_health.return_value = False
        mock_session.session_health_monitor = {'death_cascade_count': 5}
        ok &= (_validate_system_health(mock_session) is False)
    except Exception:
        ok = False

    # Confidence scoring hardening
    try:
        family = Mock()
        family.actual_relationship = "6th cousin"
        family.relationship_path = "Some path"
        dna = Mock()
        dna.predicted_relationship = "Distant cousin"
        key = select_template_by_confidence("In_Tree-Initial", family, dna)
        ok &= isinstance(key, str) and key.startswith("In_Tree-Initial")
    except Exception:
        ok = False

    # Halt signal integration
    try:
        mock_session = Mock()
        mock_session.should_halt_operations.return_value = True
        mock_session.session_health_monitor = {'death_cascade_count': 3}
        mock_session.validate_system_health.return_value = False
        ok &= (_validate_system_health(mock_session) is False)
    except Exception:
        ok = False

    # Proactive API manager minimal
    try:
        class MockSessionManager:
            def __init__(self):
                self.session_health_monitor = {'death_cascade_count': 0}
                self.should_halt_operations = lambda: False
                self._my_profile_id = "test_profile_123"
            def is_sess_valid(self):
                return True
            @property
            def my_profile_id(self):
                return self._my_profile_id
        api = ProactiveApiManager(MockSessionManager())
        delay = api.calculate_delay()
        ok &= isinstance(delay, (int, float)) and delay >= 0
        ok &= api.validate_api_response(("delivered OK", "conv_123"), "send_message_test") is True
    except Exception:
        ok = False

    # Error categorization minimal
    try:
        categorizer = ErrorCategorizer()
        category, error_type = categorizer.categorize_status("skipped (interval)")
        ok &= category == 'skipped' and 'interval' in error_type
    except Exception:
        ok = False

    return ok

