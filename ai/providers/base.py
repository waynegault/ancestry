from __future__ import annotations

"""Common types for AI provider adapters."""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from standard_imports import setup_module

logger = setup_module(globals(), __name__)


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
