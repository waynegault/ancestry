#!/usr/bin/env python3
"""
Fast Test Runner - Run only unit tests for rapid feedback.

This script runs a subset of fast tests (<5s total) for quick validation
during development. Use the full test suite (run_all_tests.py) for comprehensive
validation before commits.

Usage:
    python testing/run_tests_fast.py           # Run fast unit tests
    python testing/run_tests_fast.py --all     # Run all tests (same as run_all_tests.py)
    python testing/run_tests_fast.py --list    # List available test modules
"""

import subprocess
import sys
from pathlib import Path

# Ensure project root is in sys.path for imports to work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _ensure_venv() -> None:
    """Ensure running in venv, auto-restart if needed."""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        return

    venv_python = _PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'
    if not venv_python.exists():
        venv_python = _PROJECT_ROOT / '.venv' / 'bin' / 'python'
        if not venv_python.exists():
            print("⚠️  WARNING: Not running in virtual environment")
            return

    import os as _os

    print(f"🔄 Re-running with venv Python: {venv_python}")
    _os.chdir(_PROJECT_ROOT)
    _os.execv(str(venv_python), [str(venv_python), __file__] + sys.argv[1:])


_ensure_venv()

import argparse
import importlib
import time

# Fast test modules - comprehensive set that covers all testable modules
# These mirror the full test suite in run_all_tests.py for complete coverage
FAST_TEST_MODULES = [
    # Testing infrastructure
    "testing.test_framework",
    "testing.test_utilities",
    "testing.code_quality_checker",
    "testing.dead_code_scan",
    "testing.import_audit",
    "testing.check_type_ignores",
    "testing.protocol_mocks",
    # Configuration
    "config.config_schema",
    "config.config_manager",
    "config.validator",
    # Core infrastructure
    "core.database",
    "core.database_manager",
    "core.error_handling",
    "core.common_params",
    "core.action_registry",
    "core.action_runner",
    "core.app_mode_policy",
    "core.cache_backend",
    "core.cache_registry",
    "core.caching_bootstrap",
    "core.cancellation",
    "core.circuit_breaker",
    "core.config_validation",
    "core.correlation",
    "core.dependency_injection",
    "core.feature_flags",
    "core.health_check",
    "core.lifecycle",
    "core.logging_config",
    "core.logging_utils",
    "core.maintenance_actions",
    "core.metrics_collector",
    "core.metrics_integration",
    "core.opt_out_detection",
    "core.progress_indicators",
    "core.protocols",
    "core.rate_limiter",
    "core.registry_utils",
    "core.schema_migrator",
    "core.session_cache",
    "core.session_guards",
    "core.session_manager",
    "core.session_mixins",
    "core.session_utils",
    "core.session_validator",
    "core.system_cache",
    "core.type_definitions",
    "core.unified_cache_manager",
    "core.utils",
    "core.validation_factory",
    "core.workflow_actions",
    "core.approval_queue",
    "core.analytics_helpers",
    "core.api_manager",
    "core.browser_manager",
    "core.draft_content",
    "core.selenium_utils",
    # Core cache subpackage
    "core.cache.adapters",
    "core.cache.interface",
    # Caching
    "caching.cache",
    "caching.cache_manager",
    "caching.cache_retention",
    # API
    "api.api_constants",
    "api.api_search_core",
    "api.api_search_utils",
    "api.api_utils",
    # Browser
    "browser.chromedriver",
    "browser.css_selectors",
    "browser.diagnose_chrome",
    "browser.selenium_utils",
    # AI
    "ai.prompts",
    "ai.ai_prompt_utils",
    "ai.ai_interface",
    "ai.ab_testing",
    "ai.context_builder",
    "ai.prompt_telemetry",
    "ai.quality_regression_gate",
    "ai.sentiment_adaptation",
    # AI providers
    "ai.providers.base",
    "ai.providers.deepseek",
    "ai.providers.gemini",
    "ai.providers.local_llm",
    "ai.providers.moonshot",
    # CLI
    "cli.maintenance",
    "cli.research_tools",
    # Genealogy
    "genealogy.genealogical_normalization",
    "genealogy.genealogy_presenter",
    "genealogy.relationship_calculations",
    "genealogy.research_service",
    "genealogy.semantic_search",
    "genealogy.fact_validator",
    "genealogy.tree_stats_utils",
    "genealogy.triangulation",
    "genealogy.universal_scoring",
    "genealogy.test_research_service",
    "genealogy.test_triangulation",
    # Genealogy - DNA
    "genealogy.dna.dna_ethnicity_utils",
    "genealogy.dna.dna_gedcom_crossref",
    "genealogy.dna.dna_utils",
    # Genealogy - GEDCOM
    "genealogy.gedcom.gedcom_cache",
    "genealogy.gedcom.gedcom_intelligence",
    "genealogy.gedcom.gedcom_search_utils",
    "genealogy.gedcom.gedcom_utils",
    # Integrations
    "integrations.ms_graph_utils",
    # Messaging
    "messaging.inbound",
    "messaging.message_personalization",
    "messaging.message_types",
    "messaging.safety",
    "messaging.workflow_helpers",
    "messaging.test_inbound",
    # Observability
    "observability.analytics",
    "observability.apm",
    "observability.conversation_analytics",
    "observability.metrics_exporter",
    "observability.metrics_registry",
    "observability.utils",
    # Performance
    "performance.connection_resilience",
    "performance.grafana_checker",
    "performance.health_monitor",
    "performance.memory_utils",
    "performance.performance_cache",
    "performance.performance_monitor",
    "performance.performance_orchestrator",
    "performance.performance_profiling",
    # Research
    "research.conflict_detector",
    "research.person_lookup_utils",
    "research.predictive_gaps",
    "research.record_sharing",
    "research.relationship_diagram",
    "research.relationship_utils",
    "research.research_guidance_prompts",
    "research.research_prioritization",
    "research.research_suggestions",
    "research.search_criteria_utils",
    "research.triangulation_intelligence",
    # Scripts
    "scripts.deploy_dashboards",
    "scripts.dry_run_validation",
    "scripts.maintain_code_graph",
    # UI
    "ui.menu",
    "ui.review_server",
    "ui.terminal_test_agent",
    # Actions
    "actions.action_review",
    "actions.action6_gather",
    "actions.action7_inbox",
    "actions.action8_messaging",
    "actions.action9_process_productive",
    "actions.action10",
    "actions.action11_send_approved_drafts",
    "actions.action12_shared_matches",
    "actions.action13_triangulation",
    "actions.action14_research_tools",
    # Actions - Gather submodule
    "actions.gather.api_implementations",
    "actions.gather.checkpoint",
    "actions.gather.fetch",
    "actions.gather.metrics",
    "actions.gather.orchestrator",
    "actions.gather.persistence",
    "actions.gather.prefetch",
]

# Integration test modules (requires browser/live API) - optional slow tests
INTEGRATION_TEST_MODULES = [
    "testing.test_integration_e2e",
    "testing.test_integration_workflow",
    "testing.test_prometheus_smoke",
    "testing.test_triangulation_service",
    "testing.verify_opt_out",
]


def run_module_tests(module_name: str) -> tuple[bool, float, int]:
    """
    Run tests for a single module.

    Returns:
        Tuple of (passed: bool, duration: float, test_count: int)
    """
    try:
        start_time = time.time()
        module = importlib.import_module(module_name)

        # Try different test entry points
        test_runner = getattr(module, "run_comprehensive_tests", None)
        if test_runner is None:
            test_runner = getattr(module, "module_tests", None)
        if test_runner is None:
            # No tests found
            return True, 0.0, 0

        result = test_runner()
        duration = time.time() - start_time

        # Estimate test count from module
        test_count = 1  # Default
        if hasattr(module, "TEST_COUNT"):
            test_count = module.TEST_COUNT

        return bool(result), duration, test_count

    except Exception as e:
        print(f"  ❌ Error running {module_name}: {e}")
        return False, 0.0, 0


def list_modules() -> None:
    """List available test modules."""
    print("\n📋 Fast Test Modules (run by default):")
    for module in FAST_TEST_MODULES:
        print(f"  • {module}")

    print("\n📋 Integration Test Modules (run with --all):")
    for module in INTEGRATION_TEST_MODULES:
        print(f"  • {module}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fast test runner for rapid development feedback")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests including integration tests",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test modules",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    args = parser.parse_args()

    if args.list:
        list_modules()
        return 0

    # Determine which modules to run
    modules = FAST_TEST_MODULES.copy()
    if args.all:
        modules.extend(INTEGRATION_TEST_MODULES)

    print("=" * 60)
    print("🚀 FAST TEST RUNNER")
    print("=" * 60)
    print(f"Running {len(modules)} test modules...")
    print()

    total_start = time.time()
    passed = 0
    failed = 0
    total_tests = 0
    failed_modules: list[str] = []

    for module_name in modules:
        if args.verbose:
            print(f"  Testing {module_name}...", end=" ", flush=True)

        success, duration, test_count = run_module_tests(module_name)
        total_tests += test_count

        if success:
            passed += 1
            if args.verbose:
                print(f"✅ ({duration:.2f}s)")
        else:
            failed += 1
            failed_modules.append(module_name)
            if args.verbose:
                print(f"❌ ({duration:.2f}s)")

    total_duration = time.time() - total_start

    # Summary
    print()
    print("=" * 60)
    print("📊 FAST TEST SUMMARY")
    print("=" * 60)
    print(f"⏰ Duration: {total_duration:.2f}s")
    print(f"🧪 Modules: {passed + failed} ({passed} passed, {failed} failed)")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    if failed_modules:
        print()
        print("Failed modules:")
        for module in failed_modules:
            print(f"  • {module}")

    if failed == 0:
        print()
        print("🎉 ALL FAST TESTS PASSED!")
        if not args.all:
            print("💡 Run with --all for full test coverage")
    else:
        print()
        print("⚠️  Some tests failed. Run full suite for details:")
        print("   python run_all_tests.py")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
