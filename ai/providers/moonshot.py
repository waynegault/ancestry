
"""Moonshot (Kimi) provider adapter using the OpenAI-compatible endpoint."""

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


class MoonshotProvider(OpenAICompatibleProvider):
    """Adapter that wraps Moonshot's OpenAI-compatible API."""

    _api_key_attr = "moonshot_api_key"
    _model_attr = "moonshot_ai_model"
    _base_url_attr = "moonshot_ai_base_url"

    def __init__(self, config: Any) -> None:
        super().__init__(name="moonshot", config=config)

    @staticmethod
    def _post_extract(response: Any, _content: str | None) -> None:
        """Log Moonshot's reasoning trace if present."""
        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            message = getattr(choices[0], "message", None)
            reasoning_content = getattr(message, "reasoning_content", None)
            if reasoning_content:
                logger.debug("Moonshot reasoning preview: %s", str(reasoning_content)[:200])


# =============================================================================
# Module Tests
# =============================================================================


def _test_moonshot_provider_initialization() -> None:
    """Test MoonshotProvider initialization."""
    from types import SimpleNamespace

    mock_config = SimpleNamespace(
        api=SimpleNamespace(
            moonshot_api_key="test_key",
            moonshot_ai_model="moonshot-v1-8k",
            moonshot_ai_base_url="https://api.moonshot.cn/v1",
        )
    )
    provider = MoonshotProvider(mock_config)
    assert provider.name == "moonshot"
    assert provider._config == mock_config


def _test_moonshot_is_available() -> None:
    """Test is_available method."""
    from types import SimpleNamespace

    mock_config = SimpleNamespace(api=SimpleNamespace())
    provider = MoonshotProvider(mock_config)
    result = provider.is_available()
    assert isinstance(result, bool)


def _test_moonshot_get_api_credentials() -> None:
    """Test _get_api_credentials method."""
    from types import SimpleNamespace

    valid_config = SimpleNamespace(
        api=SimpleNamespace(
            moonshot_api_key="test_api_key",
            moonshot_ai_model="moonshot-v1-8k",
            moonshot_ai_base_url="https://api.moonshot.cn/v1",
        )
    )
    provider = MoonshotProvider(valid_config)
    api_key, model_name, base_url = provider._get_api_credentials()
    assert api_key == "test_api_key"
    assert model_name == "moonshot-v1-8k"
    assert base_url == "https://api.moonshot.cn/v1"

    # Test with missing config
    missing_config = SimpleNamespace(api=None)
    provider_missing = MoonshotProvider(missing_config)
    try:
        provider_missing._get_api_credentials()
        raise AssertionError("Should have raised ProviderConfigurationError")
    except ProviderConfigurationError:
        pass


def _test_moonshot_build_request() -> None:
    """Test _build_request method."""
    from types import SimpleNamespace

    mock_config = SimpleNamespace(
        api=SimpleNamespace(
            moonshot_api_key="k", moonshot_ai_model="m", moonshot_ai_base_url="u"
        )
    )
    provider = MoonshotProvider(mock_config)
    request = ProviderRequest(
        system_prompt="You are Kimi.",
        user_content="Hello",
        max_tokens=1000,
        temperature=0.3,
        response_format_type=None,
    )
    payload = provider._build_request(request, "moonshot-v1-8k")

    assert payload["model"] == "moonshot-v1-8k"
    assert payload["max_tokens"] == 1000
    assert payload["temperature"] == 0.3
    assert payload["stream"] is False
    assert len(payload["messages"]) == 2
    assert "response_format" not in payload

    # Test with json_object format
    request_json = ProviderRequest(
        system_prompt="Return JSON",
        user_content="Test",
        max_tokens=100,
        temperature=0.5,
        response_format_type="json_object",
    )
    payload_json = provider._build_request(request_json, "moonshot-v1-8k")
    assert payload_json["response_format"] == {"type": "json_object"}


def moonshot_provider_module_tests() -> bool:
    """Run module tests for Moonshot provider."""
    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite("Moonshot Provider", "ai/providers/moonshot.py")
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Provider initialization",
            _test_moonshot_provider_initialization,
            "Should initialize with config",
            "MoonshotProvider.__init__",
            "Test provider name and config storage",
        )
        suite.run_test(
            "is_available method",
            _test_moonshot_is_available,
            "Should return bool based on SDK availability",
            "MoonshotProvider.is_available",
            "Test OpenAI SDK detection",
        )
        suite.run_test(
            "Get API credentials",
            _test_moonshot_get_api_credentials,
            "Should extract credentials from config",
            "MoonshotProvider._get_api_credentials",
            "Test credential extraction",
        )
        suite.run_test(
            "Build request",
            _test_moonshot_build_request,
            "Should build valid request payload",
            "MoonshotProvider._build_request",
            "Test request building with and without json format",
        )

    return suite.finish_suite()


from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(moonshot_provider_module_tests)

if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
