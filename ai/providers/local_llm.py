
"""Local LLM (LM Studio) provider adapter."""

import sys
from pathlib import Path

# Support standalone execution
if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

import logging
from typing import Any

from ai.providers.base import (
    OpenAICompatibleProvider,
    ProviderConfigurationError,
    ProviderRequest,
    ProviderResponse,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


class LocalLLMProvider(OpenAICompatibleProvider):
    """Adapter for LM Studio's OpenAI-compatible API."""

    _api_key_attr = "local_llm_api_key"
    _model_attr = "local_llm_model"
    _base_url_attr = "local_llm_base_url"
    _supports_response_format = False  # LM Studio doesn't support response_format

    def __init__(self, config: Any) -> None:
        super().__init__(name="local_llm", config=config)

    # Keep backward-compatible alias
    def _get_api_settings(self) -> tuple[str, str, str]:
        return self._get_api_credentials()

    @staticmethod
    def _validate_model_loaded(client: Any, requested_name: str) -> tuple[str | None, str | None]:
        try:
            models = client.models.list()
            available_models = [model.id for model in models.data]

            if not available_models:
                return None, "Local LLM: No models loaded. Please load a model in LM Studio."

            if requested_name in available_models:
                return requested_name, None

            for available_model in available_models:
                if available_model.endswith(requested_name) or available_model.endswith(f"/{requested_name}"):
                    logger.debug("Local LLM: Matched '%s' to '%s'", requested_name, available_model)
                    return available_model, None

            return None, f"Local LLM: Model '{requested_name}' not loaded. Available models: {available_models}"
        except Exception as exc:  # pragma: no cover - network/process errors
            error_str = str(exc).lower()
            if "connection" in error_str or "refused" in error_str or "timeout" in error_str:
                return None, "Local LLM: Connection error. Please ensure LM Studio is running."
            return None, f"Local LLM: Failed to check loaded models: {exc}"

    def _pre_call(self, client: Any, model_name: str) -> str:
        """Validate model is loaded in LM Studio before calling."""
        actual_model_name, error_msg = self._validate_model_loaded(client, model_name)
        if error_msg:
            raise ProviderUnavailableError(error_msg)
        assert actual_model_name is not None  # For type checkers; guarded above
        return actual_model_name


# =============================================================================
# Module Tests
# =============================================================================


def _test_local_llm_provider_initialization() -> None:
    """Test LocalLLMProvider initialization."""
    from types import SimpleNamespace

    mock_config = SimpleNamespace(
        api=SimpleNamespace(
            local_llm_api_key="lm-studio",
            local_llm_model="local-model",
            local_llm_base_url="http://localhost:1234/v1",
        )
    )
    provider = LocalLLMProvider(mock_config)
    assert provider.name == "local_llm"
    assert provider._config == mock_config


def _test_local_llm_is_available() -> None:
    """Test is_available method."""
    from types import SimpleNamespace

    mock_config = SimpleNamespace(api=SimpleNamespace())
    provider = LocalLLMProvider(mock_config)
    result = provider.is_available()
    assert isinstance(result, bool)


def _test_local_llm_get_api_settings() -> None:
    """Test _get_api_settings method."""
    from types import SimpleNamespace

    valid_config = SimpleNamespace(
        api=SimpleNamespace(
            local_llm_api_key="lm-studio",
            local_llm_model="local-model",
            local_llm_base_url="http://localhost:1234/v1",
        )
    )
    provider = LocalLLMProvider(valid_config)
    api_key, model_name, base_url = provider._get_api_settings()
    assert api_key == "lm-studio"
    assert model_name == "local-model"
    assert base_url == "http://localhost:1234/v1"

    # Test with missing config
    missing_config = SimpleNamespace(api=None)
    provider_missing = LocalLLMProvider(missing_config)
    try:
        provider_missing._get_api_settings()
        raise AssertionError("Should have raised ProviderConfigurationError")
    except ProviderConfigurationError:
        pass


def _test_local_llm_build_request() -> None:
    """Test _build_request method."""
    from types import SimpleNamespace

    mock_config = SimpleNamespace(
        api=SimpleNamespace(
            local_llm_api_key="lm-studio",
            local_llm_model="local-model",
            local_llm_base_url="http://localhost:1234/v1",
        )
    )
    provider = LocalLLMProvider(mock_config)
    request = ProviderRequest(
        system_prompt="You are a local assistant.",
        user_content="Test prompt",
        max_tokens=256,
        temperature=0.8,
    )
    payload = provider._build_request(request, "local-model")

    assert payload["model"] == "local-model"
    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 0.8
    assert payload["stream"] is False
    assert len(payload["messages"]) == 2
    # Local LLM should NOT include response_format
    assert "response_format" not in payload


def local_llm_provider_module_tests() -> bool:
    """Run module tests for Local LLM provider."""
    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite("Local LLM Provider", "ai/providers/local_llm.py")
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Provider initialization",
            _test_local_llm_provider_initialization,
            "Should initialize with config",
            "LocalLLMProvider.__init__",
            "Test provider name and config storage",
        )
        suite.run_test(
            "is_available method",
            _test_local_llm_is_available,
            "Should return bool based on SDK availability",
            "LocalLLMProvider.is_available",
            "Test SDK detection",
        )
        suite.run_test(
            "Get API settings",
            _test_local_llm_get_api_settings,
            "Should extract settings from config",
            "LocalLLMProvider._get_api_settings",
            "Test settings extraction",
        )
        suite.run_test(
            "Build request",
            _test_local_llm_build_request,
            "Should build valid request payload",
            "LocalLLMProvider._build_request",
            "Test request building",
        )

    return suite.finish_suite()


from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(local_llm_provider_module_tests)

if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
