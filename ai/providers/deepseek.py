
"""DeepSeek provider adapter built on the OpenAI-compatible SDK."""

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
)

logger = logging.getLogger(__name__)


class DeepSeekProvider(OpenAICompatibleProvider):
    """Adapter that wraps DeepSeek's OpenAI-compatible API."""

    _api_key_attr = "deepseek_api_key"
    _model_attr = "deepseek_ai_model"
    _base_url_attr = "deepseek_ai_base_url"

    def __init__(self, config: Any) -> None:
        super().__init__(name="deepseek", config=config)


# =============================================================================
# Module Tests
# =============================================================================


def _test_deepseek_provider_initialization() -> None:
    """Test DeepSeekProvider initialization."""
    from types import SimpleNamespace

    mock_config = SimpleNamespace(
        api=SimpleNamespace(
            deepseek_api_key="test_key",
            deepseek_ai_model="deepseek-chat",
            deepseek_ai_base_url="https://api.deepseek.com/v1",
        )
    )
    provider = DeepSeekProvider(mock_config)
    assert provider.name == "deepseek"
    assert provider._config == mock_config


def _test_deepseek_is_available() -> None:
    """Test is_available method."""
    from types import SimpleNamespace

    mock_config = SimpleNamespace(api=SimpleNamespace())
    provider = DeepSeekProvider(mock_config)
    result = provider.is_available()
    # Returns True if OpenAI SDK is installed, False otherwise
    assert isinstance(result, bool)


def _test_deepseek_get_api_credentials() -> None:
    """Test _get_api_credentials method."""
    from types import SimpleNamespace

    valid_config = SimpleNamespace(
        api=SimpleNamespace(
            deepseek_api_key="test_api_key",
            deepseek_ai_model="deepseek-chat",
            deepseek_ai_base_url="https://api.deepseek.com/v1",
        )
    )
    provider = DeepSeekProvider(valid_config)
    api_key, model_name, base_url = provider._get_api_credentials()
    assert api_key == "test_api_key"
    assert model_name == "deepseek-chat"
    assert base_url == "https://api.deepseek.com/v1"

    # Test with missing config
    missing_config = SimpleNamespace(api=None)
    provider_missing = DeepSeekProvider(missing_config)
    try:
        provider_missing._get_api_credentials()
        raise AssertionError("Should have raised ProviderConfigurationError")
    except ProviderConfigurationError:
        pass


def _test_deepseek_build_request() -> None:
    """Test _build_request method."""
    from types import SimpleNamespace

    mock_config = SimpleNamespace(
        api=SimpleNamespace(
            deepseek_api_key="k", deepseek_ai_model="m", deepseek_ai_base_url="u"
        )
    )
    provider = DeepSeekProvider(mock_config)
    request = ProviderRequest(
        system_prompt="You are helpful.",
        user_content="Hello",
        max_tokens=500,
        temperature=0.7,
        response_format_type=None,
    )
    payload = provider._build_request(request, "deepseek-chat")

    assert payload["model"] == "deepseek-chat"
    assert payload["max_tokens"] == 500
    assert payload["temperature"] == 0.7
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert "response_format" not in payload

    # Test with json_object format
    request_json = ProviderRequest(
        system_prompt="Return JSON",
        user_content="Test",
        max_tokens=100,
        temperature=0.5,
        response_format_type="json_object",
    )
    payload_json = provider._build_request(request_json, "deepseek-chat")
    assert payload_json["response_format"] == {"type": "json_object"}


def deepseek_provider_module_tests() -> bool:
    """Run module tests for DeepSeek provider."""
    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite("DeepSeek Provider", "ai/providers/deepseek.py")
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Provider initialization",
            _test_deepseek_provider_initialization,
            "Should initialize with config",
            "DeepSeekProvider.__init__",
            "Test provider name and config storage",
        )
        suite.run_test(
            "is_available method",
            _test_deepseek_is_available,
            "Should return bool based on SDK availability",
            "DeepSeekProvider.is_available",
            "Test OpenAI SDK detection",
        )
        suite.run_test(
            "Get API credentials",
            _test_deepseek_get_api_credentials,
            "Should extract credentials from config",
            "DeepSeekProvider._get_api_credentials",
            "Test credential extraction",
        )
        suite.run_test(
            "Build request",
            _test_deepseek_build_request,
            "Should build valid request payload",
            "DeepSeekProvider._build_request",
            "Test request building with and without json format",
        )

    return suite.finish_suite()


from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(deepseek_provider_module_tests)

if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
