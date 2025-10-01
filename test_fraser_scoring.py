#!/usr/bin/env python3
"""
Test script to determine the correct expected score for Fraser Gault
using both Action 10 and Action 11 scoring mechanisms.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get test person data from .env
test_first_name = os.getenv("TEST_PERSON_FIRST_NAME", "Fraser")
test_last_name = os.getenv("TEST_PERSON_LAST_NAME", "Gault")
test_birth_year = int(os.getenv("TEST_PERSON_BIRTH_YEAR", "1941"))
test_gender = os.getenv("TEST_PERSON_GENDER", "m")
test_birth_place = os.getenv("TEST_PERSON_BIRTH_PLACE", "Banff")

print("=" * 80)
print("FRASER GAULT SCORING TEST")
print("=" * 80)
print(f"\nTest Person Data from .env:")
print(f"  Name: {test_first_name} {test_last_name}")
print(f"  Birth Year: {test_birth_year}")
print(f"  Gender: {test_gender}")
print(f"  Birth Place: {test_birth_place}")

# Test Action 10 scoring
print("\n" + "=" * 80)
print("ACTION 10 SCORING (GEDCOM-based)")
print("=" * 80)

try:
    from action10 import filter_and_score_individuals, get_cached_gedcom
    from config import config_schema
    
    gedcom_data = get_cached_gedcom()
    if not gedcom_data:
        print("❌ GEDCOM data not available")
    else:
        search_criteria = {
            "first_name": test_first_name.lower(),
            "surname": test_last_name.lower(),
            "birth_year": test_birth_year,
            "gender": test_gender.lower(),
            "birth_place": test_birth_place
        }
        
        results = filter_and_score_individuals(
            gedcom_data, 
            search_criteria, 
            search_criteria,
            dict(config_schema.common_scoring_weights),
            {"year_match_range": 5.0}
        )
        
        if results:
            top_result = results[0]
            score = top_result.get('total_score', 0)
            field_scores = top_result.get('field_scores', {})
            
            print(f"\n✅ Action 10 Score: {score}")
            print(f"\nField Scores:")
            for field, value in sorted(field_scores.items()):
                if value > 0:
                    print(f"  {field}: {value}")
            print(f"\nTotal: {sum(field_scores.values())}")
        else:
            print("❌ No results found")
            
except Exception as e:
    print(f"❌ Action 10 test failed: {e}")
    import traceback
    traceback.print_exc()

# Test Action 11 scoring
print("\n" + "=" * 80)
print("ACTION 11 SCORING (API-based)")
print("=" * 80)

try:
    from gedcom_utils import calculate_match_score
    from config import config_schema
    
    # Prepare search criteria
    search_criteria = {
        "first_name": test_first_name.lower(),
        "surname": test_last_name.lower(),
        "birth_year": test_birth_year,
        "gender": test_gender.lower(),
        "birth_place": test_birth_place.lower()
    }
    
    # Prepare candidate data (simulating Fraser Gault from GEDCOM)
    candidate_data = {
        "first_name": "fraser",
        "surname": "gault",
        "birth_year": 1941,
        "gender_norm": "m",
        "birth_place_disp": "Banff, Banffshire, Scotland",
        "death_place_disp": "",
        "norm_id": "TEST"
    }
    
    scoring_weights = dict(config_schema.common_scoring_weights)
    date_flex = {"year_match_range": 5}
    
    score, field_scores, reasons = calculate_match_score(
        search_criteria,
        candidate_data,
        scoring_weights,
        date_flexibility=date_flex
    )
    
    print(f"\n✅ Action 11 Score: {score}")
    print(f"\nField Scores:")
    for field, value in sorted(field_scores.items()):
        if value > 0:
            print(f"  {field}: {value}")
    print(f"\nTotal: {sum(field_scores.values())}")
    print(f"\nReasons:")
    for reason in reasons:
        print(f"  - {reason}")
        
except Exception as e:
    print(f"❌ Action 11 test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("SCORING WEIGHTS FROM CONFIG")
print("=" * 80)

try:
    from config import config_schema
    weights = dict(config_schema.common_scoring_weights)
    print("\nScoring Weights:")
    for key, value in sorted(weights.items()):
        print(f"  {key}: {value}")
except Exception as e:
    print(f"❌ Failed to load config: {e}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("\nBased on the scores above, update .env with:")
print("TEST_PERSON_EXPECTED_SCORE=<score from above>")
print("\nBoth Action 10 and Action 11 should produce the same score.")
print("=" * 80)

