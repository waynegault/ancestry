"""Shared virtual environment bootstrap utility.

This module provides a single canonical _ensure_venv() function used by all
entry-point scripts. Import and call at module level before any project imports.

Usage at the top of any entry-point script::

    import sys
    from pathlib import Path
    # ... other stdlib imports only ...

    # Must be called before project imports
    from core.venv_bootstrap import ensure_venv
    ensure_venv()
"""

import sys
from pathlib import Path


def ensure_venv(*, project_root: Path | None = None, strict: bool = False) -> None:
    """Ensure the process is running inside the project's .venv.

    If not in a virtual environment, attempts to locate `.venv/` and re-executes
    the current script under that Python interpreter.

    Args:
        project_root: Directory containing `.venv/`. Defaults to cwd.
        strict: If True and no venv is found, exit with error. If False, warn
                and continue (allows running outside venv during development).
    """
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if in_venv:
        return

    root = project_root or Path.cwd()
    venv_candidates = [
        root / ".venv" / "Scripts" / "python.exe",  # Windows
        root / ".venv" / "bin" / "python",  # macOS/Linux
    ]
    venv_python = next((p for p in venv_candidates if p.exists()), None)

    if venv_python is None:
        if strict:
            print("❌ No virtual environment found at .venv. Please create it first.")
            sys.exit(1)
        else:
            print("⚠️  WARNING: Not running in virtual environment and .venv not found")
            print("   Some operations may fail due to missing dependencies")
            return

    import subprocess

    print(f"🔄 Re-running with venv Python: {venv_python}")
    result = subprocess.run(
        [str(venv_python), sys.argv[0], *sys.argv[1:]],
        cwd=str(root),
        check=False,
    )
    sys.exit(result.returncode)
