#!/usr/bin/env python3
"""
Test Phase 5 Integration in Dry-Run Mode

This script tests that Phase 5 features (source citations, relationship diagrams,
research suggestions) are properly integrated and configured correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

print("\n" + "=" * 80)
print("PHASE 5 INTEGRATION TEST - CONFIGURATION & MODULE AVAILABILITY")
print("=" * 80)

# Test 1: Check environment variables
print("\n📋 Test 1: Environment Configuration")
print("─" * 80)
enable_sources = os.getenv('PHASE5_ENABLE_SOURCE_CITATIONS', 'not set')
enable_diagrams = os.getenv('PHASE5_ENABLE_RELATIONSHIP_DIAGRAMS', 'not set')
enable_suggestions = os.getenv('PHASE5_ENABLE_RESEARCH_SUGGESTIONS', 'not set')

print(f"  PHASE5_ENABLE_SOURCE_CITATIONS: {enable_sources}")
print(f"  PHASE5_ENABLE_RELATIONSHIP_DIAGRAMS: {enable_diagrams}")
print(f"  PHASE5_ENABLE_RESEARCH_SUGGESTIONS: {enable_suggestions}")

if enable_sources == 'true' and enable_diagrams == 'true':
    print("  ✅ Phase 5 configuration is correct")
else:
    print("  ❌ Phase 5 configuration is incorrect")
    sys.exit(1)

# Test 2: Check Phase 5 integration modules exist
print("\n📦 Test 2: Phase 5 Integration Modules")
print("─" * 80)

try:
    from action8_phase5_integration import enhance_message_format_data_phase5
    print("  ✅ action8_phase5_integration module loaded")
except ImportError as e:
    print(f"  ❌ Failed to load action8_phase5_integration: {e}")
    sys.exit(1)

try:
    from action9_phase5_integration import (
        calculate_task_priority_from_relationship,
        create_enhanced_research_task,
        generate_ai_response_prompt,
    )
    print("  ✅ action9_phase5_integration module loaded")
except ImportError as e:
    print(f"  ❌ Failed to load action9_phase5_integration: {e}")
    sys.exit(1)

# Test 3: Check Phase 5 core modules exist
print("\n🔧 Test 3: Phase 5 Core Modules")
print("─" * 80)

modules_to_test = [
    ('gedcom_utils', ['get_person_sources', 'format_source_citations']),
    ('research_suggestions', ['generate_research_suggestions']),
    ('relationship_diagram', ['generate_relationship_diagram', 'format_relationship_for_message']),
    ('record_sharing', ['format_record_reference', 'create_record_sharing_message']),
    ('research_guidance_prompts', ['create_research_guidance_prompt', 'create_conversation_response_prompt']),
]

all_modules_ok = True
for module_name, functions in modules_to_test:
    try:
        module = __import__(module_name)
        missing_functions = [f for f in functions if not hasattr(module, f)]
        if missing_functions:
            print(f"  ❌ {module_name}: Missing functions {missing_functions}")
            all_modules_ok = False
        else:
            print(f"  ✅ {module_name}: All functions available")
    except ImportError as e:
        print(f"  ❌ {module_name}: Failed to import - {e}")
        all_modules_ok = False

if not all_modules_ok:
    sys.exit(1)

# Test 4: Check Action 8 integration
print("\n🔌 Test 4: Action 8 Integration")
print("─" * 80)

try:
    with open('action8_messaging.py') as f:
        content = f.read()
        if 'PHASE5_INTEGRATION_AVAILABLE' in content:
            print("  ✅ Action 8 has Phase 5 integration flag")
        else:
            print("  ❌ Action 8 missing Phase 5 integration flag")
            all_modules_ok = False

        if 'enhance_message_format_data_phase5' in content:
            print("  ✅ Action 8 calls Phase 5 enhancement function")
        else:
            print("  ❌ Action 8 not calling Phase 5 enhancement function")
            all_modules_ok = False

        if 'PHASE5_ENABLE_SOURCE_CITATIONS' in content:
            print("  ✅ Action 8 reads Phase 5 configuration")
        else:
            print("  ❌ Action 8 not reading Phase 5 configuration")
            all_modules_ok = False
except Exception as e:
    print(f"  ❌ Failed to check Action 8: {e}")
    all_modules_ok = False

# Test 5: Check Action 9 integration
print("\n🔌 Test 5: Action 9 Integration")
print("─" * 80)

try:
    with open('action9_process_productive.py') as f:
        content = f.read()
        if 'PHASE5_INTEGRATION_AVAILABLE' in content:
            print("  ✅ Action 9 has Phase 5 integration flag")
        else:
            print("  ❌ Action 9 missing Phase 5 integration flag")
            all_modules_ok = False

        if 'from action9_phase5_integration import' in content:
            print("  ✅ Action 9 imports Phase 5 functions")
        else:
            print("  ❌ Action 9 not importing Phase 5 functions")
            all_modules_ok = False
except Exception as e:
    print(f"  ❌ Failed to check Action 9: {e}")
    all_modules_ok = False

print(f"\n{'=' * 80}")
if all_modules_ok:
    print("✅ ALL PHASE 5 INTEGRATION TESTS PASSED")
    print("=" * 80)
    print("\nPhase 5 features are properly integrated and ready for use!")
    print("Next step: Run Action 8 in dry_run mode to test with real data.")
    sys.exit(0)
else:
    print("❌ SOME PHASE 5 INTEGRATION TESTS FAILED")
    print("=" * 80)
    sys.exit(1)

