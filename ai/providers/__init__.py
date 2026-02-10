"""AI provider adapters for external services."""

# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that provider submodules are discoverable."
    import pathlib

    pkg_dir = pathlib.Path(__file__).parent
    py_files = [f.stem for f in pkg_dir.glob("*.py") if f.name != "__init__.py"]
    missing: list[str] = []
    pkg = __package__ or __name__
    for name in py_files:
        try:
            import importlib
            importlib.import_module(f"{pkg}.{name}")
        except ImportError as exc:
            missing.append(f"{name}: {exc}")
    if missing:
        print(f"  FAIL  {pkg}: {', '.join(missing)}")
        return False
    if not py_files:
        print(f"  WARN  {__name__}: no provider modules found")
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_comprehensive_tests() else 1)
