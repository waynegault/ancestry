#!/usr/bin/env python3

"""
LM Studio Process Manager

Manages LM Studio lifecycle: detection, auto-start, and API readiness validation.
Follows project patterns with comprehensive error handling and logging.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

# === THIRD-PARTY IMPORTS ===
import psutil
import requests


class LMStudioError(Exception):
    """Base exception for LM Studio management errors."""
    pass


class LMStudioStartupError(LMStudioError):
    """Raised when LM Studio fails to start or become ready."""
    pass


class LMStudioManager:
    """
    Manages LM Studio process lifecycle and API validation.

    Features:
    - Process detection with robust name matching
    - Auto-start with configurable path
    - API health checking with timeout
    - Detailed logging and error reporting
    """

    def __init__(
        self,
        executable_path: str = r"C:\Program Files\LM Studio\LM Studio.exe",
        api_url: str = "http://localhost:1234/v1/models",
        startup_timeout: int = 60,
        auto_start: bool = True
    ):
        """
        Initialize LM Studio manager.

        Args:
            executable_path: Full path to LM Studio executable
            api_url: API endpoint for health checks
            startup_timeout: Max seconds to wait for API readiness
            auto_start: Whether to auto-start if not running
        """
        self.executable_path = Path(executable_path)
        self.api_url = api_url
        self.startup_timeout = startup_timeout
        self.auto_start = auto_start

    def is_process_running(self) -> bool:
        """
        Check if LM Studio process is running.

        Returns:
            True if process found, False otherwise
        """
        for proc in psutil.process_iter(['name']):
            try:
                proc_name = proc.info['name']
                if proc_name:
                    proc_name_lower = proc_name.lower()
                    # Match various LM Studio process names
                    if any(name in proc_name_lower for name in ['lm studio', 'lmstudio', 'lm-studio']):
                        logger.debug(f"Found LM Studio process: {proc_name}")
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return False

    def is_api_ready(self, timeout: int = 2) -> tuple[bool, Optional[str]]:
        """
        Check if LM Studio API is responding.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Tuple of (is_ready, error_message)
        """
        try:
            response = requests.get(self.api_url, timeout=timeout)
            if response.status_code == 200:
                return True, None
            return False, f"API returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "API connection refused (server not ready)"
        except requests.exceptions.Timeout:
            return False, f"API timeout after {timeout}s"
        except Exception as e:
            return False, f"API check failed: {type(e).__name__}: {e}"

    def start_process(self) -> bool:
        """
        Start LM Studio process.

        Returns:
            True if started successfully, False if already running

        Raises:
            LMStudioStartupError: If executable not found or fails to start
        """
        if self.is_process_running():
            logger.info("✅ LM Studio already running")
            return False

        # Validate executable exists
        if not self.executable_path.exists():
            raise LMStudioStartupError(
                f"LM Studio executable not found: {self.executable_path}\n"
                f"   Please update LM_STUDIO_PATH in .env or config"
            )

        logger.info(f"🚀 Starting LM Studio from: {self.executable_path}")

        try:
            # Start detached process (won't block, won't terminate with parent)
            subprocess.Popen(
                [str(self.executable_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == 'nt' else 0
            )
            logger.info("✅ LM Studio process started")
            return True

        except Exception as e:
            raise LMStudioStartupError(
                f"Failed to start LM Studio: {type(e).__name__}: {e}"
            ) from e

    def wait_for_api_ready(self) -> bool:
        """
        Wait for LM Studio API to become ready.

        Returns:
            True if API became ready within timeout, False otherwise
        """
        logger.info(f"⏳ Waiting for LM Studio API (timeout: {self.startup_timeout}s)...")

        start_time = time.time()
        last_error: Optional[str] = None
        check_interval = 2  # seconds between checks

        while time.time() - start_time < self.startup_timeout:
            is_ready, error_msg = self.is_api_ready(timeout=2)

            if is_ready:
                elapsed = time.time() - start_time
                logger.info(f"✅ LM Studio API ready (took {elapsed:.1f}s)")
                return True

            # Log error only if it changed (reduce noise)
            if error_msg != last_error:
                logger.debug(f"   API not ready: {error_msg}")
                last_error = error_msg

            time.sleep(check_interval)

        elapsed = time.time() - start_time
        logger.warning(
            f"⚠️ Timeout waiting for LM Studio API after {elapsed:.1f}s\n"
            f"   Last error: {last_error}"
        )
        return False

    def ensure_ready(self) -> tuple[bool, Optional[str]]:
        """
        Ensure LM Studio is running and API is ready.

        This is the main entry point for ensuring LM Studio availability.

        Returns:
            Tuple of (success, error_message)
            - success: True if LM Studio is ready, False otherwise
            - error_message: Description of the problem if not ready
        """
        # Check if already running and ready
        if self.is_process_running():
            is_ready, error_msg = self.is_api_ready(timeout=5)
            if is_ready:
                logger.info("✅ LM Studio and API OK")
                return True, None
            logger.debug(f"LM Studio running but API not ready: {error_msg}")

        # Should we auto-start?
        if not self.auto_start:
            return False, (
                "LM Studio is not running and auto-start is disabled\n"
                "   Please start LM Studio manually or set LM_STUDIO_AUTO_START=true"
            )

        # Try to start
        try:
            self.start_process()

            # Wait for API readiness (even if already running, API might still be starting)
            if self.wait_for_api_ready():
                return True, None

            return False, (
                f"LM Studio started but API did not become ready within {self.startup_timeout}s\n"
                "   Check that no firewall is blocking localhost:1234\n"
                "   You may need to manually load a model in LM Studio"
            )

        except LMStudioStartupError as e:
            return False, str(e)
        except Exception as e:
            logger.exception("Unexpected error ensuring LM Studio ready")
            return False, f"Unexpected error: {type(e).__name__}: {e}"


def create_manager_from_config(config_schema: Any) -> LMStudioManager:
    """
    Create LMStudioManager from application config.

    Args:
        config_schema: Application config schema with api settings

    Returns:
        Configured LMStudioManager instance
    """
    api_config = config_schema.api

    return LMStudioManager(
        executable_path=api_config.lm_studio_path,
        api_url=f"{api_config.local_llm_base_url}/models",
        startup_timeout=api_config.lm_studio_startup_timeout,
        auto_start=api_config.lm_studio_auto_start
    )


# === MODULE TESTING ===

def module_tests() -> bool:
    """Test LM Studio manager functionality."""
    from test_framework import TestSuite

    suite = TestSuite("LM Studio Manager", "lm_studio_manager.py")
    suite.start_suite()

    # Test 1: Manager initialization
    def test_manager_init():
        manager = LMStudioManager(auto_start=False)
        assert manager is not None
        assert manager.executable_path is not None
        assert manager.api_url is not None

    suite.run_test(
        test_name="Manager Initialization",
        test_func=test_manager_init,
        test_summary="Verify LMStudioManager can be instantiated with defaults",
        expected_outcome="Manager object created with all required attributes"
    )

    # Test 2: Process detection doesn't crash
    def test_process_detection():
        manager = LMStudioManager(auto_start=False)
        result = manager.is_process_running()
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"

    suite.run_test(
        test_name="Process Detection",
        test_func=test_process_detection,
        test_summary="Verify process detection returns boolean without crashing",
        expected_outcome="Boolean result indicating process state"
    )

    # Test 3: API check doesn't crash (even if server not running)
    def test_api_check():
        manager = LMStudioManager(auto_start=False)
        is_ready, error_msg = manager.is_api_ready(timeout=1)
        assert isinstance(is_ready, bool), f"Expected bool, got {type(is_ready)}"
        assert error_msg is None or isinstance(error_msg, str), f"Expected str or None, got {type(error_msg)}"

    suite.run_test(
        test_name="API Health Check",
        test_func=test_api_check,
        test_summary="Verify API check returns proper tuple without crashing",
        expected_outcome="Tuple of (bool, Optional[str]) with status and optional error"
    )

    # Test 4: Config integration
    def test_config_integration():
        from config import config_schema
        api_config = config_schema.api

        # Check new config fields exist
        assert hasattr(api_config, 'lm_studio_path'), "Missing lm_studio_path field"
        assert hasattr(api_config, 'lm_studio_auto_start'), "Missing lm_studio_auto_start field"
        assert hasattr(api_config, 'lm_studio_startup_timeout'), "Missing lm_studio_startup_timeout field"

        # Verify defaults are sensible
        assert isinstance(api_config.lm_studio_path, str), "lm_studio_path should be string"
        assert isinstance(api_config.lm_studio_auto_start, bool), "lm_studio_auto_start should be bool"
        assert isinstance(api_config.lm_studio_startup_timeout, int), "lm_studio_startup_timeout should be int"
        assert api_config.lm_studio_startup_timeout > 0, "Timeout should be positive"

    suite.run_test(
        test_name="Config Schema Integration",
        test_func=test_config_integration,
        test_summary="Verify config schema has LM Studio fields with correct types",
        expected_outcome="All LM Studio config fields present with valid default values"
    )

    return suite.finish_suite()


if __name__ == "__main__":
    import sys

    # If run directly, execute tests
    success = module_tests()
    sys.exit(0 if success else 1)
