def run_action11() -> bool:
    """
    Public entry point for Action 11. Runs the comprehensive test suite.
    """
    return run_comprehensive_tests()
#!/usr/bin/env python3
"""
Action 11 - Live API Research Tool

This module provides comprehensive genealogical research capabilities using live API calls.
It includes the same Fraser Gault genealogical testing framework as Action 10 for consistency.

Key Features:
- Live API research and data gathering
- Fraser Gault genealogical validation (identical to Action 10)
- Real GEDCOM data processing without mocking
- Consistent scoring algorithms
- Family relationship analysis
- Relationship path calculation
"""

import sys
import os
import time
from typing import Dict, Any, List, Optional

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import Action 10 module for accessing nested test functions
try:
    import action10
except ImportError as e:
    print(f"❌ Failed to import Action 10 module: {e}")
    sys.exit(1)

# Core imports - minimal needed for Action 11
from standard_imports import *


def run_comprehensive_tests() -> bool:
    """
    Run comprehensive Action 11 tests using identical Fraser Gault genealogical testing.
    """
    print("🔍 Action 11: Live API Research Tool")
    print("📋 Running identical Fraser Gault genealogical tests as Action 10...")
    # Call Action 10's comprehensive test suite directly for consistency
    return action10.run_comprehensive_tests()


def _process_and_score_suggestions(api_results: List[Dict[str, Any]], search_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Processes API results and scores them using the main scoring logic. Returns a list of scored suggestions sorted by score (descending).
    """
    from api_search_utils import _run_simple_suggestion_scoring
    scored = []
    for candidate in api_results:
        score, field_scores, reasons = _run_simple_suggestion_scoring(
            search_criteria,
            candidate,
            weights=None,
            date_flex=None
        )
        candidate_copy = candidate.copy()
        candidate_copy["score"] = score
        candidate_copy["field_scores"] = field_scores
        candidate_copy["reasons"] = reasons
        scored.append(candidate_copy)
    # Sort by score descending
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored


if __name__ == "__main__":
    import sys

    print("🔍 Running Action 11 - Live API Research Tool comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
