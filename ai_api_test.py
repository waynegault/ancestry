#!/usr/bin/env python3
"""Interactive AI provider connectivity tester.

Loads credentials from the project's .env file, asks which provider to test,
validates the configured endpoint, and attempts a simple completion request.
Currently supports Moonshot, DeepSeek, Google Gemini, Local LLM, Inception Mercury, and Grok (xAI).
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import subprocess
import sys
import textwrap
import time

# Suppress httpx INFO logging (HTTP Request messages)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
from collections.abc import Iterable
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Callable, cast

from test_framework import TestSuite, create_standard_test_runner

# Optional dependency imports with proper typing
# These packages may not have type stubs, so we declare types before import
OpenAI: type[Any] | None
genai: Any
genai_types: Any
load_dotenv: Callable[..., Any] | None
XAIClient: type[Any] | None
xai_system_message: Callable[..., Any] | None
xai_user_message: Callable[..., Any] | None

try:
    from openai import OpenAI as _OpenAI

    OpenAI = _OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

genai_import_error: str | None = None
try:
    import google.genai as _genai
    import google.genai.types as _genai_types

    genai = _genai
    genai_types = _genai_types
except ImportError as import_error:  # pragma: no cover - optional dependency
    genai = None
    genai_types = None
    genai_import_error = str(import_error)

try:
    from dotenv import load_dotenv as _load_dotenv

    load_dotenv = _load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    import xai_sdk as _xai_sdk
    import xai_sdk.chat as _xai_chat

    XAIClient = _xai_sdk.Client
    xai_system_message = _xai_chat.system
    xai_user_message = _xai_chat.user
except ImportError:  # pragma: no cover - optional dependency
    XAIClient = None
    xai_system_message = None
    xai_user_message = None

DEFAULT_PROMPT = (
    "I'm interested in genealogy. Could you succinctly tell me the maximum number of great-great-great grandparents I may be descended from?"
)

# Cloud API providers
CLOUD_PROVIDERS = ("moonshot", "deepseek", "gemini", "inception", "grok", "tetrate")

# Local LLM models available in LM Studio (model_id -> display_name)
# Updated from: curl -s http://localhost:1234/v1/models
LOCAL_LLM_MODELS: dict[str, str] = {
    "qwen3-4b-instruct-2507": "Qwen3 4B Instruct",
    "deepseek-r1-distill-qwen-7b": "DeepSeek R1 Distill Qwen 7B",
    "phi-3-mini-128k-instruct-imatrix-smashed": "Phi-3 Mini 128K (smashed)",
    "phi-3-mini-4k-instruct": "Phi-3 Mini 4K",
    "mistral-7b-instruct-v0.2": "Mistral 7B Instruct v0.2",
}

# All providers: cloud + local
PROVIDERS = CLOUD_PROVIDERS + tuple(f"local:{model_id}" for model_id in LOCAL_LLM_MODELS)

PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "moonshot": "Moonshot (Kimi)",
    "deepseek": "DeepSeek",
    "gemini": "Google Gemini",
    "inception": "Inception Mercury",
    "grok": "Grok (xAI)",
    "tetrate": "Tetrate (TARS)",
    # Local models are added dynamically from LOCAL_LLM_MODELS
    **{f"local:{model_id}": name for model_id, name in LOCAL_LLM_MODELS.items()},
}

PROVIDER_ENV_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "moonshot": ("MOONSHOT_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "gemini": ("GOOGLE_API_KEY",),
    "inception": ("INCEPTION_API_KEY",),
    "grok": ("XAI_API_KEY",),
    "tetrate": ("TARS_API_KEY",),
    # Local models don't require env keys
    **{f"local:{model_id}": () for model_id in LOCAL_LLM_MODELS},
}


def _get_local_llm_base_url() -> str:
    return (os.getenv("LOCAL_LLM_BASE_URL") or "http://localhost:1234/v1").rstrip("/")


_PROVIDER_BASE_URL_FACTORIES: dict[str, Callable[[], str]] = {
    "moonshot": lambda: (os.getenv("MOONSHOT_AI_BASE_URL") or "https://api.moonshot.ai/v1").rstrip("/"),
    "deepseek": lambda: (os.getenv("DEEPSEEK_AI_BASE_URL") or "https://api.deepseek.com").rstrip("/"),
    "gemini": lambda: os.getenv("GOOGLE_AI_BASE_URL") or "https://generativelanguage.googleapis.com",
    "inception": lambda: (os.getenv("INCEPTION_AI_BASE_URL") or "https://api.inceptionlabs.ai/v1").rstrip("/"),
    "grok": lambda: os.getenv("XAI_API_HOST") or "api.x.ai",
    "tetrate": lambda: (os.getenv("TETRATE_AI_BASE_URL") or "https://api.router.tetrate.ai/v1").rstrip("/"),
    # All local models use the same LM Studio endpoint
    **{f"local:{model_id}": _get_local_llm_base_url for model_id in LOCAL_LLM_MODELS},
}

ListModelsFn = Callable[[], Iterable[Any]]


# Correct answer for the default genealogy prompt
CORRECT_ANSWER = 32


@dataclass
class TestResult:
    provider: str
    api_status: bool
    endpoint_status: bool
    messages: list[str] = field(default_factory=list)
    full_output: str | None = None
    finish_reason: str | None = None  # Track why generation stopped
    model_name: str | None = None  # Model used for the test
    load_time: float | None = None  # Time to load model (local LLM only)
    inference_time: float | None = None  # Pure inference time (excludes load)


def _load_env_file(env_path: Path, override: bool = False) -> None:
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            cleaned = value.strip().strip('"').strip("'")
            if override:
                os.environ[key.strip()] = cleaned
            else:
                os.environ.setdefault(key.strip(), cleaned)


def _find_env_path(env_file: str) -> Path | None:
    """Locate an env file relative to common project locations."""
    if not env_file:
        return None

    requested_path = Path(env_file)
    candidates: list[Path] = []

    if requested_path.is_absolute():
        candidates.append(requested_path)
    else:
        candidates.append(Path.cwd() / env_file)
        script_path = Path(__file__).resolve()
        candidates.append(script_path.parent / env_file)
        for ancestor in script_path.parents:
            candidates.append(ancestor / env_file)

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate
    return None


def _ensure_env_loaded(env_file: str, parser: argparse.ArgumentParser) -> None:
    if not env_file:
        return
    env_path = _find_env_path(env_file)
    if env_path is None:
        if env_file != ".env":
            parser.error(f"Specified env file '{env_file}' not found.")
        return
    if load_dotenv:
        load_dotenv(env_path, override=False)
    else:
        _load_env_file(env_path, override=False)


def _normalize_base_url(
    raw_base_url: str,
    *,
    required_suffix: str | None = None,
    drop_suffixes: tuple[str, ...] = (),
) -> tuple[str, bool]:
    trimmed = raw_base_url.strip()
    if not trimmed:
        return trimmed, False

    normalized = trimmed.rstrip("/")
    changed = normalized != trimmed

    for suffix in drop_suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            normalized = normalized.rstrip("/")
            changed = True

    if required_suffix:
        suffix = required_suffix
        if not suffix.startswith("/"):
            suffix = f"/{suffix}"
        if not normalized.endswith(suffix):
            normalized = f"{normalized.rstrip('/')}{suffix}"
            changed = True

    return normalized, changed


def _provider_missing_env(provider: str) -> list[str]:
    """Return list of required environment variables that are unset for provider."""
    required = PROVIDER_ENV_REQUIREMENTS.get(provider, ())
    return [env_var for env_var in required if not os.getenv(env_var)]


def _print_result(result: TestResult) -> None:
    print(f"\n=== {result.provider.upper()} Test Summary ===")
    print(f"Endpoint check : {'PASS' if result.endpoint_status else 'FAIL'}")
    print(f"API call : {'PASS' if result.api_status else 'FAIL'}")
    # Removed the Details section to simplify output


def _is_response_truncated(text: str, finish_reason: str | None) -> bool:
    """Check if response appears truncated."""
    if finish_reason == "length":
        return True
    return bool(text and not text.endswith((".", "!", "?", ")", '"', "'")))


def _build_messages_preview(content: str) -> str:
    preview = textwrap.shorten(content.replace("\n", " "), width=140, placeholder="...")
    return f"Completion preview: {preview}" if preview else "Completion preview unavailable."


_QUOTA_KEYWORDS = ("quota", "credit", "limit")


def _infer_status_code(exc: Exception, error_str: str) -> str | int | None:
    """Return HTTP status code from exception metadata or string content."""

    for attr in ("status_code", "status"):
        status_code = getattr(exc, attr, None)
        if status_code:
            return status_code

    match = re.search(r"Error code:\s*(\d+)", error_str)
    return match.group(1) if match else None


def _collect_error_details(error_str: str) -> tuple[list[str], bool]:
    """Extract notable fragments from the provider error string."""

    detail_patterns: tuple[tuple[str, str, Callable[[str], str]], ...] = (
        ("message", r"'message':\s*'([^']+)'", lambda value: value),
        ("code", r"'code':\s*'([^']+)'", lambda value: f"code: {value}"),
        ("request", r"request id:\s*([\w-]+)", lambda value: f"request id: {value}"),
    )

    details: list[str] = []
    seen: set[str] = set()
    quota_detected = False

    def add_detail(text: str) -> None:
        nonlocal quota_detected
        normalized = text.lower()
        if any(keyword in normalized for keyword in _QUOTA_KEYWORDS):
            quota_detected = True
        if normalized not in seen:
            details.append(text)
            seen.add(normalized)

    for _, pattern, formatter in detail_patterns:
        match = re.search(pattern, error_str, re.IGNORECASE)
        if match:
            add_detail(formatter(match.group(1)))

    return details, quota_detected


def _format_provider_error(provider: str, exc: Exception) -> str:
    """Return a friendly error message extracted from an exception."""

    provider_name = provider.capitalize()
    error_str = str(exc)
    status_code = _infer_status_code(exc, error_str)
    details, quota_detected = _collect_error_details(error_str)

    if quota_detected:
        return f"{provider_name} request failed: Insufficient credits or quota on the provider account."

    detail_text = "; ".join(details) if details else error_str
    if status_code:
        return f"{provider_name} request failed (HTTP {status_code}): {detail_text}"
    return f"{provider_name} request failed: {detail_text}"


def _normalize_grok_entry(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry.strip()
    for attr in ("text", "content"):
        value = getattr(entry, attr, None)
        if isinstance(value, str):
            return value.strip()
    return None


def _collapse_grok_parts(items: Iterable[Any]) -> str | None:
    parts = [text for text in (_normalize_grok_entry(item) for item in items) if text]
    return "\n".join(parts) if parts else None


def _normalize_grok_payload(payload: Any) -> str | None:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        return _collapse_grok_parts(payload)
    return None


def _normalize_grok_message(message: Any) -> str | None:
    if message is None:
        return None
    message_content = getattr(message, "content", None)
    return _normalize_grok_payload(message_content)


def _extract_grok_content(response: Any | None) -> str | None:
    """Extract text content from Grok SDK responses."""

    if response is None:
        return None

    content = getattr(response, "content", None)
    normalized = _normalize_grok_payload(content)
    if normalized:
        return normalized

    message = getattr(response, "message", None)
    normalized = _normalize_grok_message(message)
    if normalized:
        return normalized

    fallback = str(content if content is not None else response).strip()
    return fallback or None


def _test_moonshot(prompt: str, max_tokens: int) -> TestResult:
    messages: list[str] = []
    if OpenAI is None:
        messages.append("OpenAI library not available. Install the openai package.")
        return TestResult("moonshot", False, False, messages)

    api_key = os.getenv("MOONSHOT_API_KEY")
    base_url = os.getenv("MOONSHOT_AI_BASE_URL", "https://api.moonshot.ai/v1")
    model_name = os.getenv("MOONSHOT_AI_MODEL", "kimi-k2-thinking")

    if not api_key:
        messages.append("MOONSHOT_API_KEY not configured.")
        return TestResult("moonshot", False, False, messages)

    normalized_base_url, changed = _normalize_base_url(
        base_url,
        required_suffix="/v1",
        drop_suffixes=("/chat/completions",),
    )
    endpoint_ok = normalized_base_url.endswith("/v1")
    if changed:
        messages.append(f"Normalized base URL from '{base_url}' to '{normalized_base_url}'.")
    if not endpoint_ok:
        messages.append("Base URL is missing the required /v1 suffix.")

    client = OpenAI(api_key=api_key, base_url=normalized_base_url)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are Kimi from Moonshot."},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            stream=False,
            temperature=0.7,
            max_tokens=max_tokens,
        )
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            full_output = completion.choices[0].message.content.strip()
            finish_reason = getattr(completion.choices[0], "finish_reason", None)
            messages.append(_build_messages_preview(full_output))
            return TestResult(
                "moonshot", True, endpoint_ok, messages, full_output=full_output, finish_reason=finish_reason
            )
        messages.append("Moonshot returned an empty response.")
        return TestResult("moonshot", False, endpoint_ok, messages)
    except Exception as exc:
        messages.append(_format_provider_error("moonshot", exc))
        return TestResult("moonshot", False, endpoint_ok, messages)


def _test_deepseek(prompt: str, max_tokens: int) -> TestResult:
    messages: list[str] = []
    if OpenAI is None:
        messages.append("OpenAI library not available. Install the openai package.")
        return TestResult("deepseek", False, False, messages)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_AI_BASE_URL", "https://api.deepseek.com")
    model_name = os.getenv("DEEPSEEK_AI_MODEL", "deepseek-chat")

    if not api_key:
        messages.append("DEEPSEEK_API_KEY not configured.")
        return TestResult("deepseek", False, False, messages)

    # DeepSeek uses base URL without /v1 suffix - use as-is
    normalized_base_url = base_url.rstrip("/")
    endpoint_ok = True
    messages.append(f"Using DeepSeek endpoint: {normalized_base_url}")

    client = OpenAI(api_key=api_key, base_url=normalized_base_url)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are DeepSeek, an AI assistant."},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            stream=False,
            temperature=0.7,
            max_tokens=max_tokens,
        )
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            full_output = completion.choices[0].message.content.strip()
            finish_reason = getattr(completion.choices[0], "finish_reason", None)
            messages.append(_build_messages_preview(full_output))
            return TestResult(
                "deepseek", True, endpoint_ok, messages, full_output=full_output, finish_reason=finish_reason
            )
        messages.append("Received empty response payload from DeepSeek.")
        return TestResult("deepseek", False, endpoint_ok, messages)
    except Exception as exc:
        messages.append(_format_provider_error("deepseek", exc))
        return TestResult("deepseek", False, endpoint_ok, messages)


def _test_tetrate(prompt: str, max_tokens: int) -> TestResult:
    """Test Tetrate (TARS) via its OpenAI-compatible router endpoint."""
    messages: list[str] = []
    if OpenAI is None:
        messages.append("OpenAI library not available. Install the openai package.")
        return TestResult("tetrate", False, False, messages)

    api_key = os.getenv("TARS_API_KEY")
    base_url = os.getenv("TETRATE_AI_BASE_URL", "https://api.router.tetrate.ai/v1")
    # Default to the xAI Grok code model route if not configured
    model_name = os.getenv("TETRATE_AI_MODEL", "xai/grok-code-fast-1")

    if not api_key:
        messages.append("TARS_API_KEY not configured.")
        return TestResult("tetrate", False, False, messages)

    # Ensure we are using the /v1 router endpoint, drop any /chat/completions suffix
    normalized_base_url, changed = _normalize_base_url(
        base_url,
        required_suffix="/v1",
        drop_suffixes=("/chat/completions",),
    )
    endpoint_ok = normalized_base_url.endswith("/v1")
    if changed:
        messages.append(f"Normalized base URL from '{base_url}' to '{normalized_base_url}'.")
    if not endpoint_ok:
        messages.append("Base URL is missing the required /v1 suffix.")

    client = OpenAI(api_key=api_key, base_url=normalized_base_url)
    try:
        # Use a minimal OpenAI-compatible payload: just model, messages,
        # and optionally max_tokens. Tetrate's router expects standard
        # OpenAI Chat Completions parameters.
        create_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
        if max_tokens > 0:
            create_kwargs["max_tokens"] = max_tokens

        completion = client.chat.completions.create(**create_kwargs)
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            full_output = completion.choices[0].message.content.strip()
            finish_reason = getattr(completion.choices[0], "finish_reason", None)
            messages.append(_build_messages_preview(full_output))
            return TestResult(
                "tetrate", True, endpoint_ok, messages, full_output=full_output, finish_reason=finish_reason
            )
        messages.append("Tetrate returned an empty response.")
        return TestResult("tetrate", False, endpoint_ok, messages)
    except Exception as exc:
        messages.append(_format_provider_error("tetrate", exc))
        return TestResult("tetrate", False, endpoint_ok, messages)


def _model_supports_generate_content(model: Any) -> bool:
    methods = getattr(model, "supported_generation_methods", [])
    return not methods or "generateContent" in methods


def _gather_gemini_models(client: Any, *, limit: int = 10) -> tuple[list[str], bool]:
    names: list[str] = []
    has_more = False
    for model in client.models.list():
        if not _model_supports_generate_content(model):
            continue
        name = getattr(model, "name", "").replace("models/", "")
        if not name:
            continue
        names.append(name)
        if len(names) >= limit:
            has_more = True
            break
    return names, has_more


def _append_gemini_model_listing(
    names: list[str],
    configured: str,
    has_more: bool,
    messages: list[str],
) -> None:
    if not names:
        messages.append("   ⚠️  No models found")
        messages.append("")
        return

    found_configured = False
    for name in names:
        marker = " ← CONFIGURED" if name == configured else ""
        messages.append(f"   • {name}{marker}")
        if marker:
            found_configured = True
    if has_more:
        messages.append("   ... (additional models available)")
    if not found_configured:
        messages.append(f"   ⚠️  Configured model '{configured}' not found in available models")
    messages.append("")


def _describe_gemini_models(client: Any, model_name: str, messages: list[str]) -> None:
    if client is None:
        return
    messages.append("\n📋 Available Gemini models with generateContent support:")
    try:
        names, has_more = _gather_gemini_models(client)
    except Exception as exc:  # pragma: no cover - diagnostic helper
        messages.append(f"   ⚠️  Could not list models: {exc}")
        messages.append("")
        return
    _append_gemini_model_listing(names, model_name, has_more, messages)


def _suggestion_suffix(error_msg: str, suggestions: list[str]) -> str:
    if not suggestions:
        if "Try:" in error_msg:
            return error_msg
        return error_msg + " Try updating GOOGLE_AI_MODEL in .env to a supported model."

    suffix = f" Try: {', '.join(suggestions)}"
    if suffix.strip() in error_msg:
        return error_msg
    return error_msg + suffix


def _suggest_gemini_model_alternatives(client: Any, error_msg: str) -> str:
    if client is None:
        return error_msg
    try:
        names, _ = _gather_gemini_models(client, limit=3)
    except Exception:  # pragma: no cover - suggestion helper
        return error_msg
    return _suggestion_suffix(error_msg, names)


def _parse_gemini_response(response: Any, messages: list[str]) -> tuple[bool, str | None, str | None]:
    candidates = getattr(response, "candidates", [])
    if not candidates:
        messages.append("Gemini returned no candidates.")
        return False, None, None

    candidate = candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)
    safety_ratings = getattr(candidate, "safety_ratings", [])

    # In new SDK, finish_reason might be 'STOP' or similar string/enum
    # We'll just log it if it looks abnormal
    if finish_reason and str(finish_reason) not in {"STOP", "1", "FinishReason.STOP"}:
        messages.append(f"⚠️  Generation stopped with finish_reason: {finish_reason}")
        if safety_ratings:
            messages.append(f"Safety ratings: {safety_ratings}")

    text_parts: list[str] = []
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", [])
    for part in parts:
        value = getattr(part, "text", "")
        if value:
            text_parts.append(value)

    combined = " ".join(text_parts).strip()
    if combined:
        messages.append(_build_messages_preview(combined))
        finish_reason_text = str(finish_reason) if finish_reason is not None else None
        return True, combined, finish_reason_text

    messages.append("Gemini response contained no text content.")
    # prompt_feedback is not always present in new SDK response objects in the same way
    if hasattr(response, "prompt_feedback"):
        messages.append(f"Prompt feedback: {response.prompt_feedback}")
    return False, None, None


def _test_gemini(prompt: str, max_tokens: int) -> TestResult:
    messages: list[str] = []
    if genai is None:
        messages.append(f"google-genai package not available ({genai_import_error or 'Install google-genai.'}).")
        return TestResult("gemini", False, False, messages)

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GOOGLE_AI_MODEL", "gemini-2.5-flash")
    base_url = os.getenv("GOOGLE_AI_BASE_URL", "")

    if not api_key:
        messages.append("GOOGLE_API_KEY not configured.")
        return TestResult("gemini", False, False, messages)

    endpoint_ok = True
    if base_url:
        messages.append(f"Using custom Gemini endpoint: {base_url}")
    else:
        messages.append("Using default Google Generative Language endpoint.")

    client: Any = None
    try:
        # Initialize the client
        # Note: The new SDK uses 'http_options' for base_url if needed, or just api_key
        if base_url:
            client = genai.Client(api_key=api_key, http_options=cast(Any, {'api_endpoint': base_url}))
        else:
            client = genai.Client(api_key=api_key)

        _describe_gemini_models(client, model_name, messages)

        # Create config using proper type
        config = None
        if genai_types:
            config = genai_types.GenerateContentConfig(
                candidate_count=1,
                max_output_tokens=max_tokens,
                temperature=0.7,
            )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        success, full_output, finish_reason_str = _parse_gemini_response(response, messages)

    except Exception as exc:
        error_msg = _format_provider_error("gemini", exc)
        lowered = error_msg.lower()
        if "is not found" in lowered or "not supported" in lowered:
            error_msg = _suggest_gemini_model_alternatives(client, error_msg)
        messages.append(error_msg)
        return TestResult("gemini", False, endpoint_ok, messages)

    return TestResult(
        "gemini",
        success,
        endpoint_ok,
        messages,
        full_output=full_output,
        finish_reason=finish_reason_str,
    )


def _local_llm_endpoint_active(target_base_url: str) -> bool:
    try:
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(target_base_url)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((parsed.hostname or "localhost", parsed.port or 1234))
        sock.close()
        return result == 0
    except Exception:
        return False


def _maybe_start_lm_studio(base_url: str, messages: list[str]) -> None:
    auto_start = os.getenv("LM_STUDIO_AUTO_START", "false").lower() == "true"
    if not auto_start:
        return

    lm_path = os.getenv("LM_STUDIO_PATH")
    if not lm_path or not Path(lm_path).exists():
        messages.append("LM_STUDIO_AUTO_START is true but LM_STUDIO_PATH is not configured or does not exist.")
        return

    if _local_llm_endpoint_active(base_url):
        messages.append("LM Studio is already running.")
        return

    try:
        subprocess.Popen([lm_path], shell=True)
        startup_timeout = int(os.getenv("LM_STUDIO_STARTUP_TIMEOUT", "60"))
        wait_time = min(startup_timeout, 10)
        messages.append(f"Starting LM Studio (waiting {wait_time}s for initialization)...")
        time.sleep(wait_time)
    except Exception as exc:
        messages.append(f"Failed to start LM Studio: {exc}")


def _normalize_local_llm_endpoint(base_url: str, messages: list[str]) -> tuple[str, bool]:
    normalized_base_url, changed = _normalize_base_url(base_url, required_suffix="/v1")
    endpoint_ok = normalized_base_url.endswith("/v1")
    if changed:
        messages.append(f"Normalized base URL from '{base_url}' to '{normalized_base_url}'.")
    if not endpoint_ok:
        messages.append("Base URL is missing the required /v1 suffix.")
    return normalized_base_url, endpoint_ok


def _check_model_loaded(base_url: str, model_id: str) -> tuple[bool, list[str]]:
    """Check if the requested model is available in LM Studio.

    Returns (is_available, list of available model ids).
    """
    import requests

    try:
        resp = requests.get(f"{base_url}/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        available_models = [m.get("id", "") for m in data.get("data", [])]

        # Check exact match or partial match (model_id might have org prefix)
        for available in available_models:
            if model_id == available or model_id.endswith(f"/{available}"):
                return True, available_models

        return False, available_models
    except requests.exceptions.ConnectionError:
        return False, []
    except Exception:
        return False, []


def _warm_up_model(client: Any, model_id: str) -> tuple[bool, float]:
    """Send a minimal request to ensure model is loaded. Returns (success, load_time)."""
    import time

    start = time.time()
    try:
        # Minimal request - just ask for 1 token to trigger model load
        client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1,
            temperature=0,
        )
        return True, time.time() - start
    except Exception:
        return False, time.time() - start


def _create_local_llm_tester(model_id: str) -> Callable[[str, int], TestResult]:
    """Factory function to create a tester for a specific local LLM model."""

    def _test_local_model(prompt: str, max_tokens: int) -> TestResult:
        provider_name = f"local:{model_id}"
        messages: list[str] = []
        if OpenAI is None:
            messages.append("OpenAI library not available. Install the openai package.")
            return TestResult(provider_name, False, False, messages)

        api_key = os.getenv("LOCAL_LLM_API_KEY") or ""
        base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")

        if not base_url:
            messages.append("LOCAL_LLM_BASE_URL not configured.")
            return TestResult(provider_name, False, False, messages)

        _maybe_start_lm_studio(base_url, messages)
        normalized_base_url, endpoint_ok = _normalize_local_llm_endpoint(base_url, messages)

        # Check if model is available in LM Studio
        is_available, available_models = _check_model_loaded(normalized_base_url, model_id)
        if not is_available:
            if not available_models:
                messages.append("⚠️  LM Studio not running or not accessible")
            else:
                messages.append(f"⚠️  Model '{model_id}' not found in LM Studio.")
                messages.append(f"   Available models: {', '.join(available_models)}")
            return TestResult(provider_name, False, endpoint_ok, messages, model_name=model_id)

        client = OpenAI(api_key=api_key or "lm-studio", base_url=normalized_base_url)

        # Warm up: trigger model load (if needed) before timing
        messages.append(f"Loading model: {model_id}...")
        warm_ok, load_time = _warm_up_model(client, model_id)
        if not warm_ok:
            messages.append(f"⚠️  Failed to load model '{model_id}'")
            return TestResult(provider_name, False, endpoint_ok, messages, model_name=model_id)
        messages.append(f"✓ Model ready (load time: {load_time:.1f}s)")

        # Now time only the inference (with 20s timeout, no retries)
        inference_start = time.time()
        try:
            # Create client with timeout for inference and NO retries
            client_with_timeout = OpenAI(
                api_key=api_key or "lm-studio",
                base_url=normalized_base_url,
                timeout=20.0,  # 20 second timeout for inference
                max_retries=0,  # No retries - fail fast on timeout
            )
            # Use only user message - some models (e.g., Mistral) don't support system role
            completion = client_with_timeout.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=0.7,
                max_tokens=max_tokens,
            )
            inference_time = time.time() - inference_start
            if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                full_output = completion.choices[0].message.content.strip()
                finish_reason = getattr(completion.choices[0], "finish_reason", None)
                messages.append(_build_messages_preview(full_output))
                return TestResult(
                    provider_name,
                    True,
                    endpoint_ok,
                    messages,
                    full_output=full_output,
                    finish_reason=finish_reason,
                    model_name=model_id,
                    load_time=load_time,
                    inference_time=inference_time,
                )
            messages.append("Local LLM returned an empty response.")
            return TestResult(provider_name, False, endpoint_ok, messages, model_name=model_id, load_time=load_time)
        except Exception as exc:
            inference_time = time.time() - inference_start
            error_msg = _format_provider_error(provider_name, exc)
            if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
                error_msg = f"⏱️  TIMEOUT after {inference_time:.1f}s - model too slow for practical use"
            elif "no models loaded" in str(exc).lower():
                error_msg += f" Please ensure '{model_id}' is loaded in LM Studio."
            messages.append(error_msg)
            return TestResult(provider_name, False, endpoint_ok, messages, model_name=model_id, load_time=load_time)

    return _test_local_model


def _test_inception(prompt: str, max_tokens: int) -> TestResult:
    """Test Inception Mercury API (OpenAI-compatible endpoint)."""
    messages: list[str] = []
    if OpenAI is None:
        messages.append("OpenAI library not available. Install the openai package.")
        return TestResult("inception", False, False, messages)

    api_key = os.getenv("INCEPTION_API_KEY")
    base_url = os.getenv("INCEPTION_AI_BASE_URL", "https://api.inceptionlabs.ai/v1")
    model_name = os.getenv("INCEPTION_AI_MODEL", "mercury")

    if not api_key:
        messages.append("INCEPTION_API_KEY not configured.")
        return TestResult("inception", False, False, messages)

    normalized_base_url, changed = _normalize_base_url(
        base_url,
        required_suffix="/v1",
        drop_suffixes=("/chat/completions",),
    )
    endpoint_ok = normalized_base_url.endswith("/v1")
    if changed:
        messages.append(f"Normalized base URL from '{base_url}' to '{normalized_base_url}'.")
    if not endpoint_ok:
        messages.append("Base URL is missing the required /v1 suffix.")

    client = OpenAI(api_key=api_key, base_url=normalized_base_url)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            stream=False,
            temperature=0.7,
            max_tokens=max_tokens,
        )
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            full_output = completion.choices[0].message.content.strip()
            finish_reason = getattr(completion.choices[0], "finish_reason", None)
            messages.append(_build_messages_preview(full_output))
            return TestResult(
                "inception", True, endpoint_ok, messages, full_output=full_output, finish_reason=finish_reason
            )
        messages.append("Inception Mercury returned an empty response.")
        return TestResult("inception", False, endpoint_ok, messages)
    except Exception as exc:
        messages.append(_format_provider_error("inception", exc))
        return TestResult("inception", False, endpoint_ok, messages)


def _test_grok(prompt: str, max_tokens: int) -> TestResult:
    messages: list[str] = []
    if XAIClient is None or xai_system_message is None or xai_user_message is None:
        messages.append("xai-sdk library not available. Install the xai-sdk package.")
        return TestResult("grok", False, False, messages)

    api_key = os.getenv("XAI_API_KEY")
    model_name = os.getenv("XAI_MODEL", "grok-4-fast-non-reasoning")
    api_host = os.getenv("XAI_API_HOST", "api.x.ai") or "api.x.ai"

    if not api_key:
        messages.append("XAI_API_KEY not configured.")
        return TestResult("grok", False, False, messages)

    if not model_name:
        messages.append("XAI_MODEL not configured.")
        return TestResult("grok", False, False, messages)

    messages.append(f"Using Grok host: {api_host}")
    endpoint_ok = True

    try:
        client = XAIClient(api_key=api_key, api_host=api_host, timeout=60)
        chat_session = client.chat.create(
            model=model_name,
            max_tokens=max_tokens or None,
            temperature=0.7,
        )
        chat_session.append(xai_system_message("You are Grok, a helpful genealogy assistant."))
        chat_session.append(xai_user_message(prompt))
        response = chat_session.sample()
        content = _extract_grok_content(response)
        finish_reason = getattr(response, "finish_reason", None) or getattr(response, "stop_reason", None)
        if content:
            messages.append(_build_messages_preview(content))
            return TestResult("grok", True, endpoint_ok, messages, full_output=content, finish_reason=finish_reason)
        messages.append("Grok returned an empty response payload.")
        return TestResult("grok", False, endpoint_ok, messages)
    except Exception as exc:
        messages.append(_format_provider_error("grok", exc))
        return TestResult("grok", False, endpoint_ok, messages)


PROVIDER_TESTERS: dict[str, Callable[[str, int], TestResult]] = {
    "moonshot": _test_moonshot,
    "deepseek": _test_deepseek,
    "gemini": _test_gemini,
    "inception": _test_inception,
    "grok": _test_grok,
    "tetrate": _test_tetrate,
    # Local LLM models - each gets its own tester
    **{f"local:{model_id}": _create_local_llm_tester(model_id) for model_id in LOCAL_LLM_MODELS},
}


def _provider_base_url(provider: str) -> str:
    factory = _PROVIDER_BASE_URL_FACTORIES.get(provider)
    if not factory:
        return ""
    return factory()


def _prompt_for_provider() -> str | None:
    print("Available providers:\n")
    provider_list = list(PROVIDERS)
    for idx, provider in enumerate(provider_list, start=1):
        display_name = PROVIDER_DISPLAY_NAMES.get(provider, provider.capitalize())
        base_url = _provider_base_url(provider)
        missing_env = _provider_missing_env(provider)
        status_note = "" if not missing_env else f" - missing: {', '.join(missing_env)}"
        print(f"  {idx}. {display_name} [{base_url}]{status_note}")

    while True:
        choice = input("\nEnter the number of the provider you want to test (or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            return None
        if not choice.isdigit():
            print("Please enter a number from the list above or 'q' to quit.")
            continue
        index = int(choice)
        if 1 <= index <= len(provider_list):
            return provider_list[index - 1]
        print("Invalid selection. Choose a valid number.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive AI provider connectivity tester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--provider", choices=PROVIDERS, help="Provider to test. If omitted, you will be prompted.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send to the provider.")
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=2048, help="Maximum tokens/output length.")
    parser.add_argument("--env-file", dest="env_file", default=".env", help="Path to .env file to load before running.")
    parser.add_argument(
        "--show-truncation-warning", action="store_true", help="Show warning if response appears truncated."
    )
    return parser


def _require_provider_tester(provider: str, parser: argparse.ArgumentParser) -> Callable[[str, int], TestResult]:
    tester = PROVIDER_TESTERS.get(provider)
    if not tester:
        parser.error(f"Unsupported provider: {provider}")
    return tester


def _ensure_provider_env_ready(provider: str, *, interactive: bool) -> bool:
    missing_env = _provider_missing_env(provider)
    if not missing_env:
        return True

    missing_list = ", ".join(missing_env)
    if interactive:
        print(f"\n⚠️  Provider '{provider}' is missing required environment variables: {missing_list}.")
        print("    Update your .env (or pass --env-file) and choose again.\n")
    else:
        print(f"Provider '{provider}' is missing required environment variables: {missing_list}.")
        print("Update your .env (or pass --env-file) and try again.")
    return False


def _execute_provider_test(
    tester: Callable[[str, int], TestResult], prompt: str, max_tokens: int
) -> tuple[TestResult, float]:
    start_time = time.time()
    result = tester(prompt, max_tokens)
    duration = time.time() - start_time
    return result, duration


def _check_answer_correctness(response: str) -> tuple[bool, str]:
    """Check if response contains the correct answer (32 great-great-great grandparents)."""
    import re

    response_lower = response.lower()

    # Check for correct answer (32)
    if re.search(r"\b32\b", response) or "thirty-two" in response_lower or "thirty two" in response_lower:
        return True, "✅ CORRECT - The answer is 32 great-great-great-grandparents."

    # Check for common wrong answers using word boundaries to avoid partial matches
    wrong_answers = [
        (r"\b16\b|sixteen", "16 (off by one generation)"),
        (r"\b64\b|sixty[- ]?four", "64 (one generation too many)"),
        (r"\b128\b|one hundred (and )?twenty[- ]?eight", "128 (two generations too many)"),
        (r"\b8\b|eight", "8 (way off - confused generations)"),
        (r"\b4\b|four", "4 (way off - confused generations)"),
        (r"\b2\b|two", "2 (way off - confused generations)"),
    ]
    for pattern, desc in wrong_answers:
        if re.search(pattern, response_lower):
            return False, f"❌ INCORRECT - Response says {desc}. Correct answer: 32."

    return False, "❓ UNCLEAR - Could not find a clear numeric answer. Correct answer: 32."


def _render_test_output(args: argparse.Namespace, result: TestResult, duration: float) -> None:
    _print_result(result)

    prompt: str = getattr(args, "prompt", DEFAULT_PROMPT)
    max_tokens: int = getattr(args, "max_tokens", 2048)

    if result.api_status and result.full_output:
        # Show model name for local_llm
        if result.model_name:
            print(f"\nModel: {result.model_name}")
        print("\nPrompt:")
        print(f'"{prompt}"')
        print("\nResponse:")
        print(result.full_output)

        # For local LLMs, show load time and inference time separately
        if result.load_time is not None and result.inference_time is not None:
            print(f"\nLoad time: {result.load_time:.2f}s")
            print(f"Inference time: {result.inference_time:.2f}s")
            print(f"Total time: {duration:.2f}s")
        else:
            print(f"\nResponse time: {duration:.2f}s")

        # Check correctness if using the default genealogy prompt
        if prompt == DEFAULT_PROMPT:
            _, verdict = _check_answer_correctness(result.full_output)
            print(f"\nConclusion: {verdict}")

        if _is_response_truncated(result.full_output, result.finish_reason):
            print(f"⚠️  WARNING: Response appears truncated (finish_reason: {result.finish_reason or 'unknown'})")
            print(f"   Consider increasing --max-tokens (current: {max_tokens})")
        print()
        return

    if result.api_status or not result.messages:
        return

    print("\nError details:")
    for msg in result.messages:
        print(f"  {msg}")


def _run_single_provider(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    provider: str = getattr(args, "provider", "")
    prompt: str = getattr(args, "prompt", DEFAULT_PROMPT)
    max_tokens: int = getattr(args, "max_tokens", 2048)

    tester = _require_provider_tester(provider, parser)
    if not _ensure_provider_env_ready(provider, interactive=False):
        return 1

    print(f"\nTesting provider: {provider}")
    result, duration = _execute_provider_test(tester, prompt, max_tokens)
    _render_test_output(args, result, duration)
    return 0


def _interactive_provider_loop(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    prompt: str = getattr(args, "prompt", DEFAULT_PROMPT)
    max_tokens: int = getattr(args, "max_tokens", 2048)

    while True:
        provider = _prompt_for_provider()
        if provider is None:
            break

        tester = _require_provider_tester(provider, parser)
        if not _ensure_provider_env_ready(provider, interactive=True):
            continue

        print(f"\nTesting provider: {provider}")
        result, duration = _execute_provider_test(tester, prompt, max_tokens)
        _render_test_output(args, result, duration)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    _ensure_env_loaded(".env", parser)
    args = parser.parse_args(argv)

    env_file: str = getattr(args, "env_file", ".env")
    provider: str | None = getattr(args, "provider", None)

    if env_file and env_file != ".env":
        _ensure_env_loaded(env_file, parser)

    if not provider and _should_prefer_local_llm():
        print("⚙️ Defaulting to local_llm provider for automated test execution.")
        provider = "local_llm"
        setattr(args, "provider", provider)

    if provider:
        return _run_single_provider(args, parser)

    _interactive_provider_loop(args, parser)
    return 0


@contextmanager
def _temporary_env(overrides: dict[str, str | None]):
    original = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _temporary_cwd(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


@contextmanager
def _patched_provider_tester(provider: str, tester: Callable[[str, int], TestResult]):
    original = PROVIDER_TESTERS.get(provider)
    PROVIDER_TESTERS[provider] = tester
    try:
        yield tester
    finally:
        if original is None:
            PROVIDER_TESTERS.pop(provider, None)
        else:
            PROVIDER_TESTERS[provider] = original


def _test_normalize_base_url_applies_suffix() -> bool:
    normalized, changed = _normalize_base_url(
        "https://api.example.com/chat/completions",
        required_suffix="/v1",
        drop_suffixes=("/chat/completions",),
    )
    assert normalized == "https://api.example.com/v1"
    assert changed is True
    return True


def _test_provider_missing_env_detects_requirements() -> bool:
    with _temporary_env({"MOONSHOT_API_KEY": None}):
        assert _provider_missing_env("moonshot") == ["MOONSHOT_API_KEY"]
    with _temporary_env({"MOONSHOT_API_KEY": "token"}):
        assert _provider_missing_env("moonshot") == []
    return True


def _test_is_response_truncated_and_preview_helpers() -> bool:
    assert _is_response_truncated("Incomplete thought", None) is True
    assert _is_response_truncated("Complete sentence.", None) is False
    assert _is_response_truncated("Whatever", "length") is True
    preview = _build_messages_preview("word " * 80)
    assert preview.startswith("Completion preview")
    assert preview.endswith("...")
    return True


def _test_format_provider_error_extracts_status_and_details() -> bool:
    class FakeError(Exception):
        def __init__(self) -> None:
            super().__init__("{'message': 'quota exceeded', 'code': 'quota'}")
            self.status_code = 429

    message = _format_provider_error("moonshot", FakeError())
    assert "Insufficient credits" in message
    assert "Moonshot" in message
    return True


def _test_extract_grok_content_handles_part_lists() -> bool:
    response = SimpleNamespace(content=[SimpleNamespace(text="Hello"), SimpleNamespace(text="World")])
    assert _extract_grok_content(response) == "Hello\nWorld"
    fallback = SimpleNamespace(content=None, message=SimpleNamespace(content=[SimpleNamespace(text="Hi")]))
    assert _extract_grok_content(fallback) == "Hi"
    return True


def _test_load_env_file_respects_override_flag() -> bool:
    with TemporaryDirectory() as tmp_dir:
        env_path = Path(tmp_dir) / ".env"
        env_path.write_text("FOO=bar\n", encoding="utf-8")
        with _temporary_env({"FOO": "existing"}):
            _load_env_file(env_path, override=False)
            assert os.environ["FOO"] == "existing"
        with _temporary_env({"FOO": "existing"}):
            _load_env_file(env_path, override=True)
            assert os.environ["FOO"] == "bar"
    return True


def _test_find_env_path_locates_relative_files() -> bool:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        env_path = root / ".env.local"
        env_path.write_text("KEY=VALUE", encoding="utf-8")
        with _temporary_cwd(root):
            found = _find_env_path(".env.local")
    assert found == env_path
    return True


def _test_main_requires_env_for_cli_provider() -> bool:
    with TemporaryDirectory() as tmp_dir:
        temp_root = Path(tmp_dir)
        (temp_root / ".env").write_text("", encoding="utf-8")
        with _temporary_cwd(temp_root), _temporary_env({"MOONSHOT_API_KEY": None}):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["--provider", "moonshot", "--prompt", "Short prompt"])
    output = buffer.getvalue()
    assert exit_code == 1
    assert "missing required environment variables" in output
    return True


def _test_main_cli_executes_stubbed_provider_once() -> bool:
    fake_result = TestResult(
        provider="moonshot",
        api_status=True,
        endpoint_status=True,
        full_output="Complete response.",
        finish_reason="stop",
    )
    calls: list[tuple[str, int]] = []

    def fake_tester(prompt: str, max_tokens: int) -> TestResult:
        calls.append((prompt, max_tokens))
        return fake_result

    with TemporaryDirectory() as tmp_dir:
        temp_root = Path(tmp_dir)
        (temp_root / ".env").write_text("", encoding="utf-8")
        with (
            _temporary_cwd(temp_root),
            _temporary_env({"MOONSHOT_API_KEY": "token"}),
            _patched_provider_tester("moonshot", fake_tester),
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(
                    [
                        "--provider",
                        "moonshot",
                        "--prompt",
                        "CLI prompt.",
                        "--max-tokens",
                        "321",
                    ]
                )
    output = buffer.getvalue()
    assert exit_code == 0
    assert calls == [("CLI prompt.", 321)]
    assert "Testing provider: moonshot" in output
    assert '"CLI prompt."' in output
    assert "Response time:" in output
    return True


def module_tests() -> bool:
    suite = TestSuite("ai_api_test", "ai_api_test.py")
    suite.run_test(
        "Normalize base URL",
        _test_normalize_base_url_applies_suffix,
        "Ensures suffix handling trims chat endpoints and appends /v1.",
    )
    suite.run_test(
        "Provider env detection",
        _test_provider_missing_env_detects_requirements,
        "Ensures missing env vars are surfaced per provider requirements.",
    )
    suite.run_test(
        "Response truncation helpers",
        _test_is_response_truncated_and_preview_helpers,
        "Ensures truncation and preview helpers flag incomplete outputs.",
    )
    suite.run_test(
        "Provider error formatting",
        _test_format_provider_error_extracts_status_and_details,
        "Ensures diagnostic errors include codes and quota messaging.",
    )
    suite.run_test(
        "Grok content extraction",
        _test_extract_grok_content_handles_part_lists,
        "Ensures Grok response normalization handles list payloads.",
    )
    suite.run_test(
        "Env utilities",
        _test_load_env_file_respects_override_flag,
        "Ensures env loader honors override flag semantics.",
    )
    suite.run_test(
        "Env path discovery",
        _test_find_env_path_locates_relative_files,
        "Ensures .env discovery resolves relative paths via cwd.",
    )
    suite.run_test(
        "CLI missing env guard",
        _test_main_requires_env_for_cli_provider,
        "Ensures CLI provider mode surfaces missing env vars before testing.",
    )
    suite.run_test(
        "CLI provider execution",
        _test_main_cli_executes_stubbed_provider_once,
        "Ensures CLI provider args invoke the tester once and print responses.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


def _should_run_module_tests() -> bool:
    return os.environ.get("RUN_MODULE_TESTS") == "1"


def _should_prefer_local_llm() -> bool:
    if _should_run_module_tests():
        return True
    skip_live = os.environ.get("SKIP_LIVE_API_TESTS", "").lower()
    return skip_live in {"1", "true", "yes"}


if __name__ == "__main__":
    if _should_run_module_tests():
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    sys.exit(main())
