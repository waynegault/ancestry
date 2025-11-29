from __future__ import annotations

"""DeepSeek provider adapter built on the OpenAI-compatible SDK."""

import logging
from collections.abc import Sequence
from typing import Any

from .base import (
    BaseProvider,
    ProviderConfigurationError,
    ProviderRequest,
    ProviderResponse,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency import
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled via is_available
    OpenAI = None


class DeepSeekProvider(BaseProvider):
    """Adapter that wraps DeepSeek's OpenAI-compatible API."""

    def __init__(self, config: Any) -> None:
        super().__init__(name="deepseek")
        self._config = config

    def is_available(self) -> bool:
        return bool(OpenAI) and bool(self._config)

    def _get_api_credentials(self) -> tuple[str, str, str]:
        api_config = getattr(self._config, "api", None)
        if api_config is None:
            raise ProviderConfigurationError("API configuration missing for DeepSeek")

        api_key = getattr(api_config, "deepseek_api_key", None)
        model_name = getattr(api_config, "deepseek_ai_model", None)
        base_url = getattr(api_config, "deepseek_ai_base_url", None)

        if not all([api_key, model_name, base_url]):
            raise ProviderConfigurationError("DeepSeek configuration incomplete (api key/model/base url)")
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

        if OpenAI is None:  # Defensive - ensure_available should guard this
            raise ProviderUnavailableError("OpenAI SDK not available for DeepSeek")

        client = OpenAI(api_key=api_key, base_url=base_url)
        payload = self._build_request(request, model_name)
        response: Any = client.chat.completions.create(**payload)

        content: str | None = None
        choices_obj = getattr(response, "choices", None)
        choices: Sequence[Any] = choices_obj if isinstance(choices_obj, Sequence) else []
        first_choice = next(iter(choices), None)
        if first_choice is not None:
            message = getattr(first_choice, "message", None)
            msg_content = getattr(message, "content", None)
            if isinstance(msg_content, str):
                content = msg_content.strip()

        if content is None:
            logger.error("DeepSeek returned an empty or invalid response structure.")

        return ProviderResponse(content=content, raw_response=response)


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
    """Test _build_request static method."""
    request = ProviderRequest(
        system_prompt="You are helpful.",
        user_content="Hello",
        max_tokens=500,
        temperature=0.7,
        response_format_type=None,
    )
    payload = DeepSeekProvider._build_request(request, "deepseek-chat")

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
    payload_json = DeepSeekProvider._build_request(request_json, "deepseek-chat")
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
