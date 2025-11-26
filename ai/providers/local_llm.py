from __future__ import annotations

"""Local LLM (LM Studio) provider adapter."""

from typing import Any

from standard_imports import setup_module

from .base import (
    BaseProvider,
    ProviderConfigurationError,
    ProviderRequest,
    ProviderResponse,
    ProviderUnavailableError,
)

logger = setup_module(globals(), __name__)

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled by is_available
    OpenAI = None


class LocalLLMProvider(BaseProvider):
    """Adapter for LM Studio's OpenAI-compatible API."""

    def __init__(self, config: Any) -> None:
        super().__init__(name="local_llm")
        self._config = config

    def is_available(self) -> bool:
        return bool(OpenAI) and bool(self._config)

    def _get_api_settings(self) -> tuple[str, str, str]:
        api_config = getattr(self._config, "api", None)
        if api_config is None:
            raise ProviderConfigurationError("API configuration missing for Local LLM")

        api_key = getattr(api_config, "local_llm_api_key", None)
        model_name = getattr(api_config, "local_llm_model", None)
        base_url = getattr(api_config, "local_llm_base_url", None)

        if not all([api_key, model_name, base_url]):
            raise ProviderConfigurationError("Local LLM configuration incomplete (api key/model/base url)")
        return str(api_key), str(model_name), str(base_url)

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

    @staticmethod
    def _build_request(request: ProviderRequest, model_name: str) -> dict[str, Any]:
        return {
            "model": model_name,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_content},
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.ensure_available()
        api_key, configured_model, base_url = self._get_api_settings()

        if OpenAI is None:  # Defensive guard despite ensure_available()
            raise ProviderUnavailableError("OpenAI SDK not available for Local LLM")

        client = OpenAI(api_key=api_key, base_url=base_url)
        actual_model_name, error_msg = self._validate_model_loaded(client, configured_model)
        if error_msg:
            raise ProviderUnavailableError(error_msg)
        assert actual_model_name is not None  # For type checkers; guarded above

        payload = self._build_request(request, actual_model_name)
        try:
            response = client.chat.completions.create(**payload)
        except Exception as exc:  # pragma: no cover - network/process errors
            raise ProviderUnavailableError(f"Local LLM API call failed: {exc}") from exc

        content: str | None = None
        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            if message and getattr(message, "content", None):
                content = str(message.content).strip()

        if content is None:
            logger.error("Local LLM returned an empty or invalid response structure.")

        return ProviderResponse(content=content, raw_response=response)
