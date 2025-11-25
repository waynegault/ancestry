from __future__ import annotations

"""DeepSeek provider adapter built on the OpenAI-compatible SDK."""

from collections.abc import Sequence
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
