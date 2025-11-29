from __future__ import annotations

"""Gemini provider adapter powered by google-genai."""

import importlib
import logging
from collections.abc import Iterable
from typing import Any

from ai.providers.base import (
    BaseProvider,
    ProviderConfigurationError,
    ProviderRequest,
    ProviderResponse,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Adapter responsible for executing Gemini API calls."""

    def __init__(self, config: Any) -> None:
        super().__init__(name="gemini")
        self._config = config
        self._genai = self._safe_import("google.genai")
        self._genai_types = self._safe_import("google.genai.types")

    @staticmethod
    def _safe_import(module_name: str) -> Any | None:  # pragma: no cover - import shim
        try:
            return importlib.import_module(module_name)
        except Exception:
            return None

    def is_available(self) -> bool:
        return bool(self._genai and hasattr(self._genai, "Client"))

    def _get_api_credentials(self) -> tuple[str, str]:
        api_config = getattr(self._config, "api", None)
        if api_config is None:
            raise ProviderConfigurationError("API configuration missing for Gemini")

        api_key = getattr(api_config, "google_api_key", None)
        model_name = getattr(api_config, "google_ai_model", None)
        if not api_key or not model_name:
            raise ProviderConfigurationError("Gemini configuration incomplete (api key/model)")
        return str(api_key), str(model_name)

    def _list_models(self, api_key: str) -> list[str]:
        if not self._genai:
            return []
        try:
            client = self._genai.Client(api_key=api_key)
            models: list[str] = []
            model_iter: Iterable[Any] = client.models.list()
            for model in model_iter:
                model_name = getattr(model, "name", "")
                normalized_name = model_name.replace("models/", "")
                if normalized_name:
                    models.append(normalized_name)
            return models
        except Exception as exc:
            logger.debug("Failed to list Gemini models: %s", exc)
            return []

    def _validate_model(self, api_key: str, model_name: str) -> bool:
        available = self._list_models(api_key)
        if not available:
            return True  # If listing failed we allow the request to proceed

        normalized = model_name.replace("models/", "")
        if normalized in available:
            logger.debug("✅ Model '%s' validated successfully", normalized)
            return True

        logger.error("❌ Model '%s' not found or unsupported", model_name)
        preview = ", ".join(available[:5])
        if preview:
            logger.error("📋 Available models: %s%s", preview, "" if len(available) <= 5 else " ...")
        return False

    def _initialize_client(self, api_key: str, model_name: str) -> Any | None:
        if not self._genai or not hasattr(self._genai, "Client"):
            raise ProviderUnavailableError("Gemini SDK missing Client class")

        if not self._validate_model(api_key, model_name):
            return None

        try:
            client = self._genai.Client(api_key=api_key)
            logger.debug("✅ Gemini client initialized successfully for model '%s'", model_name)
            return client
        except Exception as exc:
            logger.error("Failed initializing Gemini client: %s", exc)
            return None

    def _build_generation_config(self, request: ProviderRequest) -> Any | None:
        if not self._genai_types:
            return {
                "candidateCount": 1,
                "maxOutputTokens": request.max_tokens,
                "temperature": request.temperature,
            }

        config_cls = getattr(self._genai_types, "GenerateContentConfig", None)
        if config_cls is None:
            return {
                "candidateCount": 1,
                "maxOutputTokens": request.max_tokens,
                "temperature": request.temperature,
            }

        try:
            return config_cls(
                candidateCount=1,
                maxOutputTokens=request.max_tokens,
                temperature=request.temperature,
            )
        except Exception:
            return None

    def _generate_content(self, client: Any, model_name: str, full_prompt: str, generation_config: Any | None) -> Any:
        _ = self  # Explicit reference for analyzer clarity
        if not client or not hasattr(client, "models"):
            return None

        try:
            return client.models.generate_content(model=model_name, contents=full_prompt, config=generation_config)
        except Exception as exc:
            logger.error("Gemini generation failed: %s", exc)
            return None

    @staticmethod
    def _extract_response_text(response: Any | None) -> str | None:
        if response is not None and getattr(response, "text", None):
            return getattr(response, "text", "").strip()

        block_reason = "Unknown"
        try:
            if response is not None and hasattr(response, "prompt_feedback"):
                pf = getattr(response, "prompt_feedback", None)
                if pf and hasattr(pf, "block_reason"):
                    br = getattr(pf, "block_reason", None)
                    if hasattr(br, "name"):
                        block_reason = getattr(br, "name", "Unknown")
                    elif br is not None:
                        block_reason = str(br)
        except Exception:
            pass
        logger.error("Gemini returned an empty or blocked response. Reason: %s", block_reason)
        return None

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.ensure_available()
        api_key, model_name = self._get_api_credentials()
        client = self._initialize_client(api_key, model_name)
        if client is None:
            return ProviderResponse(content=None, raw_response=None)

        full_prompt = f"{request.system_prompt}\n\n---\n\nUser Query/Content:\n{request.user_content}"
        generation_config = self._build_generation_config(request)
        response = self._generate_content(client, model_name, full_prompt, generation_config)
        text = self._extract_response_text(response)
        return ProviderResponse(content=text, raw_response=response)


# =============================================================================
# Module Tests
# =============================================================================


def _test_gemini_provider_initialization() -> None:
    """Test GeminiProvider initialization."""
    from types import SimpleNamespace

    # Test with mock config
    mock_config = SimpleNamespace(
        api=SimpleNamespace(
            google_api_key="test_key",
            google_ai_model="gemini-1.5-flash",
        )
    )
    provider = GeminiProvider(mock_config)
    assert provider.name == "gemini"
    assert provider._config == mock_config


def _test_gemini_is_available() -> None:
    """Test is_available method."""
    from types import SimpleNamespace

    mock_config = SimpleNamespace(api=SimpleNamespace(google_api_key="test"))
    provider = GeminiProvider(mock_config)
    # Will return False if google.genai is not installed, True if installed
    result = provider.is_available()
    assert isinstance(result, bool)


def _test_gemini_get_api_credentials() -> None:
    """Test _get_api_credentials method."""
    from types import SimpleNamespace

    # Test with valid config
    valid_config = SimpleNamespace(
        api=SimpleNamespace(
            google_api_key="test_api_key",
            google_ai_model="gemini-1.5-flash",
        )
    )
    provider = GeminiProvider(valid_config)
    api_key, model_name = provider._get_api_credentials()
    assert api_key == "test_api_key"
    assert model_name == "gemini-1.5-flash"

    # Test with missing config
    missing_config = SimpleNamespace(api=None)
    provider_missing = GeminiProvider(missing_config)
    try:
        provider_missing._get_api_credentials()
        raise AssertionError("Should have raised ProviderConfigurationError")
    except ProviderConfigurationError:
        pass  # Expected


def _test_gemini_build_generation_config() -> None:
    """Test _build_generation_config method."""
    from types import SimpleNamespace

    mock_config = SimpleNamespace(api=SimpleNamespace())
    provider = GeminiProvider(mock_config)

    request = ProviderRequest(
        system_prompt="Test",
        user_content="Hello",
        max_tokens=500,
        temperature=0.5,
    )
    config = provider._build_generation_config(request)
    # Should return dict or config object depending on SDK availability
    assert config is not None


def _test_gemini_extract_response_text() -> None:
    """Test _extract_response_text static method."""
    # Test with None response
    result = GeminiProvider._extract_response_text(None)
    assert result is None

    # Test with mock response that has text
    from types import SimpleNamespace

    mock_response = SimpleNamespace(text="  Test response  ")
    result = GeminiProvider._extract_response_text(mock_response)
    assert result == "Test response"

    # Test with empty text
    empty_response = SimpleNamespace(text="")
    result = GeminiProvider._extract_response_text(empty_response)
    assert result is None


def gemini_provider_module_tests() -> bool:
    """Run module tests for Gemini provider."""
    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite("Gemini Provider", "ai/providers/gemini.py")
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Provider initialization",
            _test_gemini_provider_initialization,
            "Should initialize with config",
            "GeminiProvider.__init__",
            "Test provider name and config storage",
        )
        suite.run_test(
            "is_available method",
            _test_gemini_is_available,
            "Should return bool based on SDK availability",
            "GeminiProvider.is_available",
            "Test SDK detection",
        )
        suite.run_test(
            "Get API credentials",
            _test_gemini_get_api_credentials,
            "Should extract credentials from config",
            "GeminiProvider._get_api_credentials",
            "Test credential extraction and validation",
        )
        suite.run_test(
            "Build generation config",
            _test_gemini_build_generation_config,
            "Should create generation config",
            "GeminiProvider._build_generation_config",
            "Test config building",
        )
        suite.run_test(
            "Extract response text",
            _test_gemini_extract_response_text,
            "Should extract text from response",
            "GeminiProvider._extract_response_text",
            "Test response parsing",
        )

    return suite.finish_suite()


from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(gemini_provider_module_tests)

if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
