
"""Common types for AI provider adapters."""

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProviderRequest:
    """Canonical request payload for provider adapters."""

    system_prompt: str
    user_content: str
    max_tokens: int
    temperature: float
    response_format_type: str | None = None


@dataclass(slots=True)
class ProviderResponse:
    """Standardized provider response container."""

    content: str | None
    raw_response: Any | None = None


class ProviderError(RuntimeError):
    """Base error for provider failures."""


class ProviderConfigurationError(ProviderError):
    """Raised when a provider is missing required configuration."""


class ProviderUnavailableError(ProviderError):
    """Raised when an adapter dependency (SDK/API) is missing."""


@runtime_checkable
class ProviderAdapter(Protocol):
    """Protocol implemented by provider adapters."""

    name: str

    def is_available(self) -> bool:
        """Return True when dependencies are installed and configured."""
        ...

    def call(self, request: ProviderRequest) -> ProviderResponse:  # pragma: no cover - protocol definition
        """Execute an AI request and return the standardized response."""
        ...


class BaseProvider:
    """Convenience base class implementing shared helpers."""

    def __init__(self, name: str) -> None:
        self.name = name

    def ensure_available(self) -> None:
        if not self.is_available():  # pragma: no cover - thin helper
            raise ProviderUnavailableError(f"Provider '{self.name}' is not available")

    def is_available(self) -> bool:  # pragma: no cover - override expected
        _ = self.name  # Explicit reference keeps linters satisfied
        return False


# ---------------------------------------------------------------------------
# OpenAI-compatible provider base (DeepSeek, Moonshot, Local LLM, …)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - optional dependency import
    from openai import OpenAI as _OpenAI
except ImportError:  # pragma: no cover - handled via is_available
    _OpenAI = None


class OpenAICompatibleProvider(BaseProvider):
    """Base class for providers that expose an OpenAI-compatible chat API.

    Subclasses only need to set three class attributes that name the config
    keys for api_key, model, and base_url.  Override hooks are provided for
    provider-specific behaviour (e.g. model validation, reasoning traces).
    """

    # --- subclass must override these three ---
    _api_key_attr: str = ""
    _model_attr: str = ""
    _base_url_attr: str = ""
    # If True the ``response_format`` field is included in the API payload
    _supports_response_format: bool = True

    def __init__(self, name: str, config: Any) -> None:
        super().__init__(name=name)
        self._config = config

    def is_available(self) -> bool:
        return bool(_OpenAI) and bool(self._config)

    # -- credential helpers --------------------------------------------------

    def _get_api_credentials(self) -> tuple[str, str, str]:
        """Extract (api_key, model, base_url) from the shared config object."""
        api_config = getattr(self._config, "api", None)
        if api_config is None:
            raise ProviderConfigurationError(f"API configuration missing for {self.name}")

        api_key = getattr(api_config, self._api_key_attr, None)
        model_name = getattr(api_config, self._model_attr, None)
        base_url = getattr(api_config, self._base_url_attr, None)

        if not all([api_key, model_name, base_url]):
            raise ProviderConfigurationError(
                f"{self.name} configuration incomplete (api key/model/base url)"
            )
        return str(api_key), str(model_name), str(base_url)

    # -- request / response --------------------------------------------------

    def _build_request(self, request: ProviderRequest, model_name: str) -> dict[str, Any]:
        """Build the chat completions payload."""
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
        if self._supports_response_format and request.response_format_type == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def _extract_content(response: Any) -> str | None:
        """Pull the text from an OpenAI-style response object."""
        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            if message and getattr(message, "content", None):
                return str(message.content).strip()
        return None

    # -- hooks (override in subclasses) --------------------------------------

    @staticmethod
    def _pre_call(_client: Any, model_name: str) -> str:
        """Called before the API request.  Return the (possibly adjusted) model name."""
        return model_name

    def _post_extract(self, response: Any, content: str | None) -> None:
        """Called after content extraction for provider-specific logging."""

    # -- main entry point ----------------------------------------------------

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.ensure_available()
        api_key, model_name, base_url = self._get_api_credentials()

        if _OpenAI is None:  # Defensive — ensure_available should guard this
            raise ProviderUnavailableError(f"OpenAI SDK not available for {self.name}")

        client = _OpenAI(api_key=api_key, base_url=base_url)
        model_name = self._pre_call(client, model_name)

        payload = self._build_request(request, model_name)
        try:
            response: Any = client.chat.completions.create(**payload)
        except Exception as exc:
            raise ProviderUnavailableError(f"{self.name} API call failed: {exc}") from exc

        content = self._extract_content(response)
        self._post_extract(response, content)

        if content is None:
            logger.error("%s returned an empty or invalid response structure.", self.name)

        return ProviderResponse(content=content, raw_response=response)


# =============================================================================
# Module Tests
# =============================================================================


def _test_provider_request_creation() -> None:
    """Test ProviderRequest dataclass creation."""
    request = ProviderRequest(
        system_prompt="You are a helpful assistant.",
        user_content="Hello, how are you?",
        max_tokens=100,
        temperature=0.7,
        response_format_type="json_object",
    )
    assert request.system_prompt == "You are a helpful assistant."
    assert request.user_content == "Hello, how are you?"
    assert request.max_tokens == 100
    assert request.temperature == 0.7
    assert request.response_format_type == "json_object"


def _test_provider_response_creation() -> None:
    """Test ProviderResponse dataclass creation."""
    response = ProviderResponse(content="Test response", raw_response={"test": True})
    assert response.content == "Test response"
    assert response.raw_response == {"test": True}

    # Test with None content
    empty_response = ProviderResponse(content=None)
    assert empty_response.content is None
    assert empty_response.raw_response is None


def _test_provider_errors() -> None:
    """Test provider error classes."""
    # Test ProviderError
    error = ProviderError("Test error")
    assert str(error) == "Test error"
    assert isinstance(error, RuntimeError)

    # Test ProviderConfigurationError
    config_error = ProviderConfigurationError("Config missing")
    assert isinstance(config_error, ProviderError)

    # Test ProviderUnavailableError
    unavail_error = ProviderUnavailableError("SDK not installed")
    assert isinstance(unavail_error, ProviderError)


def _test_base_provider_initialization() -> None:
    """Test BaseProvider initialization."""
    provider = BaseProvider(name="test_provider")
    assert provider.name == "test_provider"
    assert provider.is_available() is False


def _test_base_provider_ensure_available() -> None:
    """Test BaseProvider.ensure_available raises when not available."""
    provider = BaseProvider(name="test_provider")
    try:
        provider.ensure_available()
        raise AssertionError("Should have raised ProviderUnavailableError")
    except ProviderUnavailableError as e:
        assert "test_provider" in str(e)


def _test_provider_adapter_protocol() -> None:
    """Test that ProviderAdapter protocol can be checked at runtime."""

    # Create a class that implements the protocol
    # Note: Methods must be instance methods (not static) to match the Protocol definition
    class TestAdapter:
        name = "test"

        def is_available(self) -> bool:  # noqa: PLR6301
            return True

        def call(self, request: ProviderRequest) -> ProviderResponse:  # noqa: ARG002, PLR6301
            return ProviderResponse(content="test")

    adapter = TestAdapter()
    # Runtime protocol check
    assert isinstance(adapter, ProviderAdapter)


def base_provider_module_tests() -> bool:
    """Run module tests for base provider types."""
    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite("AI Provider Base Types", "ai/providers/base.py")
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "ProviderRequest creation",
            _test_provider_request_creation,
            "Should create valid ProviderRequest instances",
            "ProviderRequest dataclass",
            "Test dataclass fields and slot behavior",
        )
        suite.run_test(
            "ProviderResponse creation",
            _test_provider_response_creation,
            "Should create valid ProviderResponse instances",
            "ProviderResponse dataclass",
            "Test response content and raw_response fields",
        )
        suite.run_test(
            "Provider error classes",
            _test_provider_errors,
            "Should have proper error hierarchy",
            "ProviderError, ProviderConfigurationError, ProviderUnavailableError",
            "Test inheritance and message handling",
        )
        suite.run_test(
            "BaseProvider initialization",
            _test_base_provider_initialization,
            "Should initialize with name and default is_available=False",
            "BaseProvider.__init__, is_available",
            "Test base class behavior",
        )
        suite.run_test(
            "BaseProvider.ensure_available",
            _test_base_provider_ensure_available,
            "Should raise ProviderUnavailableError when not available",
            "BaseProvider.ensure_available",
            "Test availability guard",
        )
        suite.run_test(
            "ProviderAdapter protocol",
            _test_provider_adapter_protocol,
            "Should be checkable at runtime",
            "ProviderAdapter protocol",
            "Test runtime_checkable decorator",
        )

    return suite.finish_suite()


from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(base_provider_module_tests)

if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
