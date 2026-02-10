"""AI package with provider adapters and prompt utilities."""

# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that key submodules are importable."
    expected = ["ai_interface", "ai_prompt_utils", "prompt_telemetry",
                 "context_builder", "prompts", "sentiment_adaptation"]
    missing: list[str] = []
    pkg = __package__ or __name__
    for name in expected:
        try:
            import importlib
            importlib.import_module(f"{pkg}.{name}")
        except ImportError as exc:
            missing.append(f"{name}: {exc}")
    if missing:
        print(f"  FAIL  {pkg}: {', '.join(missing)}")
        return False
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_comprehensive_tests() else 1)
