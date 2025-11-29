from __future__ import annotations

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
