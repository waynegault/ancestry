from __future__ import annotations

"""Gemini provider adapter powered by google-genai."""

import importlib
from collections.abc import Iterable
from typing import Any

from ai.providers.base import (
    BaseProvider,
    ProviderConfigurationError,
    ProviderRequest,
    ProviderResponse,
    ProviderUnavailableError,
)
from standard_imports import setup_module

logger = setup_module(globals(), __name__)


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
