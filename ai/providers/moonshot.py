from __future__ import annotations

"""Moonshot (Kimi) provider adapter using the OpenAI-compatible endpoint."""

import logging
from typing import Any

from .base import (
    BaseProvider,
    ProviderConfigurationError,
    ProviderRequest,
    ProviderResponse,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled by is_available
    OpenAI = None


class MoonshotProvider(BaseProvider):
    """Adapter that wraps Moonshot's OpenAI-compatible API."""

    def __init__(self, config: Any) -> None:
        super().__init__(name="moonshot")
        self._config = config

    def is_available(self) -> bool:
        return bool(OpenAI) and bool(self._config)

    def _get_api_credentials(self) -> tuple[str, str, str]:
        api_config = getattr(self._config, "api", None)
        if api_config is None:
            raise ProviderConfigurationError("API configuration missing for Moonshot")

        api_key = getattr(api_config, "moonshot_api_key", None)
        model_name = getattr(api_config, "moonshot_ai_model", None)
        base_url = getattr(api_config, "moonshot_ai_base_url", None)

        if not all([api_key, model_name, base_url]):
            raise ProviderConfigurationError("Moonshot configuration incomplete (api key/model/base url)")
        return str(api_key), str(model_name), str(base_url)

    @staticmethod
    def _build_request(request: ProviderRequest, model_name: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_content},
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }
        if request.response_format_type == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.ensure_available()
        api_key, model_name, base_url = self._get_api_credentials()

        if OpenAI is None:  # Defensive guard despite ensure_available()
            raise ProviderUnavailableError("OpenAI SDK not available for Moonshot")

        client = OpenAI(api_key=api_key, base_url=base_url)
        payload = self._build_request(request, model_name)
        response: Any = client.chat.completions.create(**payload)

        content: str | None = None
        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            if message and getattr(message, "content", None):
                content = str(message.content).strip()

            # Moonshot exposes a reasoning trace; logging a short preview aids debugging.
            reasoning_content = getattr(message, "reasoning_content", None)
            if reasoning_content:
                logger.debug("Moonshot reasoning preview: %s", str(reasoning_content)[:200])

        if content is None:
            logger.error("Moonshot returned an empty or invalid response structure.")

        return ProviderResponse(content=content, raw_response=response)


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
    """Test _build_request static method."""
    request = ProviderRequest(
        system_prompt="You are Kimi.",
        user_content="Hello",
        max_tokens=1000,
        temperature=0.3,
        response_format_type=None,
    )
    payload = MoonshotProvider._build_request(request, "moonshot-v1-8k")

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
    payload_json = MoonshotProvider._build_request(request_json, "moonshot-v1-8k")
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
