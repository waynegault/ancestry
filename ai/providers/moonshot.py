from __future__ import annotations

"""Moonshot (Kimi) provider adapter using the OpenAI-compatible endpoint."""

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
        if getattr(response, "choices", None):
            first_choice = response.choices[0]
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
