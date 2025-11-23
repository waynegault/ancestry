#!/usr/bin/env python3
"""Interactive AI provider connectivity tester.

Loads credentials from the project's .env file, asks which provider to test,
validates the configured endpoint, and attempts a simple completion request.
Currently supports Moonshot, DeepSeek, Google Gemini, Local LLM, Inception Mercury, and Grok (xAI).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

try:
    from openai import OpenAI as _OpenAI
except ImportError:  # pragma: no cover - optional dependency
    _OpenAI = None

OpenAI: Any | None = _OpenAI

genai_import_error: str | None = None
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as import_error:  # pragma: no cover - optional dependency
    genai = None
    genai_types = None
    genai_import_error = str(import_error)

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    _load_dotenv = None

load_dotenv: Callable[..., Any] | None = _load_dotenv

try:
    from xai_sdk import Client as _XAIClient
    from xai_sdk.chat import system as _xai_system_message, user as _xai_user_message
except ImportError:  # pragma: no cover - optional dependency
    _XAIClient = None
    _xai_system_message = None
    _xai_user_message = None

XAIClient: Callable[..., Any] | None = cast(Callable[..., Any] | None, _XAIClient)
xai_system_message: Callable[..., Any] | None = _xai_system_message
xai_user_message: Callable[..., Any] | None = _xai_user_message

DEFAULT_PROMPT = "I'm interested in geneology. Could you succinctly tell me how many individual great-great-great grandparents did I have?"
PROVIDERS = ("moonshot", "deepseek", "gemini", "local_llm", "inception", "grok", "tetrate")

PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "moonshot": "Moonshot (Kimi)",
    "deepseek": "DeepSeek",
    "gemini": "Google Gemini",
    "local_llm": "Local LLM (LM Studio)",
    "inception": "Inception Mercury",
    "grok": "Grok (xAI)",
    "tetrate": "Tetrate (TARS)",
}

PROVIDER_ENV_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "moonshot": ("MOONSHOT_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "gemini": ("GOOGLE_API_KEY",),
    "local_llm": (),
    "inception": ("INCEPTION_API_KEY",),
    "grok": ("XAI_API_KEY",),
    "tetrate": ("TARS_API_KEY",),
}

ListModelsFn = Callable[[], Iterable[Any]]


@dataclass
class TestResult:
    provider: str
    api_status: bool
    endpoint_status: bool
    messages: list[str] = field(default_factory=list)
    full_output: str | None = None
    finish_reason: str | None = None  # Track why generation stopped


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


def _format_provider_error(provider: str, exc: Exception) -> str:
    """Return a friendly error message extracted from an exception."""
    provider_name = provider.capitalize()
    status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    error_str = str(exc)

    if status_code is None:
        status_match = re.search(r"Error code:\s*(\d+)", error_str)
        if status_match:
            status_code = status_match.group(1)

    message_match = re.search(r"'message':\s*'([^']+)'", error_str)
    code_match = re.search(r"'code':\s*'([^']+)'", error_str)
    req_match = re.search(r"request id:\s*([\w-]+)", error_str)

    details: list[str] = []
    seen: set[str] = set()

    quota_keywords = {"quota", "credit", "limit"}
    quota_detected = False

    def add_detail(text: str) -> None:
        nonlocal quota_detected
        normalized = text.lower()
        if any(keyword in normalized for keyword in quota_keywords):
            quota_detected = True
        if normalized not in seen:
            details.append(text)
            seen.add(normalized)

    if message_match:
        add_detail(message_match.group(1))
    if code_match:
        add_detail(f"code: {code_match.group(1)}")
    if req_match and not any("request id" in d.lower() for d in details):
        add_detail(f"request id: {req_match.group(1)}")

    if quota_detected:
        return f"{provider_name} request failed: Insufficient credits or quota on the provider account."

    detail_text = "; ".join(details) if details else error_str
    if status_code:
        return f"{provider_name} request failed (HTTP {status_code}): {detail_text}"
    return f"{provider_name} request failed: {detail_text}"


def _extract_grok_content(response: Any | None) -> str | None:
    """Extract text content from Grok SDK responses."""
    if response is None:
        return None

    def _normalize(entry: Any) -> str | None:
        if isinstance(entry, str):
            return entry.strip()
        for attr in ("text", "content"):
            value = getattr(entry, attr, None)
            if isinstance(value, str):
                return value.strip()
        return None

    normalized: str | None = None
    content = getattr(response, "content", None)
    if isinstance(content, str):
        normalized = content.strip()
    elif isinstance(content, list):
        parts = [part for part in (_normalize(item) for item in content) if part]
        if parts:
            normalized = "\n".join(parts)

    if normalized is None:
        message = getattr(response, "message", None)
        if message is not None:
            message_content = getattr(message, "content", None)
            if isinstance(message_content, str):
                normalized = message_content.strip()
            elif isinstance(message_content, list):
                parts = [part for part in (_normalize(item) for item in message_content) if part]
                if parts:
                    normalized = "\n".join(parts)

    if normalized is None:
        normalized = str(content if content is not None else response).strip()

    return normalized


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


def _describe_gemini_models(client: Any, model_name: str, messages: list[str]) -> None:
    if client is None:
        return
    try:
        messages.append("\n📋 Available Gemini models with generateContent support:")
        model_count = 0
        found_configured = False
        # client.models.list() returns an iterator of Model objects
        for model in client.models.list():
            # Check supported generation methods if available, otherwise assume it's a model
            methods = getattr(model, "supported_generation_methods", [])
            if methods and "generateContent" not in methods:
                continue

            # model.name is usually "models/gemini-1.5-flash"
            name = getattr(model, "name", "").replace("models/", "")
            if not name:
                continue

            marker = " ← CONFIGURED" if name == model_name else ""
            messages.append(f"   • {name}{marker}")
            model_count += 1
            if name == model_name:
                found_configured = True
            if model_count >= 10:  # Increased limit slightly
                messages.append("   ... (additional models available)")
                break
        if model_count == 0:
            messages.append("   ⚠️  No models found")
        elif not found_configured:
            messages.append(f"   ⚠️  Configured model '{model_name}' not found in available models")
        messages.append("")
    except Exception as exc:  # pragma: no cover - diagnostic helper
        messages.append(f"   ⚠️  Could not list models: {exc}")
        messages.append("")


def _suggest_gemini_model_alternatives(client: Any, error_msg: str) -> str:
    if client is None:
        return error_msg
    try:
        available_models: list[str] = []
        for model in client.models.list():
            methods = getattr(model, "supported_generation_methods", [])
            if methods and "generateContent" not in methods:
                continue
            name = getattr(model, "name", "").replace("models/", "")
            if name:
                available_models.append(name)
            if len(available_models) >= 3:
                break
        if available_models:
            suggestion = f" Try: {', '.join(available_models)}"
            if suggestion.strip() not in error_msg:
                error_msg += suggestion
    except Exception:  # pragma: no cover - suggestion helper
        return error_msg
    if "Try:" not in error_msg:
        error_msg += " Try updating GOOGLE_AI_MODEL in .env to a supported model."
    return error_msg


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
    model_name = os.getenv("GOOGLE_AI_MODEL", "gemini-1.5-flash")
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

        # Create config
        config = None
        if genai_types:
            config = {
                "candidateCount": 1,
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
            }

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=cast(Any, config),
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


def _test_local_llm(prompt: str, max_tokens: int) -> TestResult:
    messages: list[str] = []
    if OpenAI is None:
        messages.append("OpenAI library not available. Install the openai package.")
        return TestResult("local_llm", False, False, messages)

    api_key = os.getenv("LOCAL_LLM_API_KEY") or ""
    base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
    model_name = os.getenv("LOCAL_LLM_MODEL", "qwen3-4b-2507")

    if not base_url:
        messages.append("LOCAL_LLM_BASE_URL not configured.")
        return TestResult("local_llm", False, False, messages)

    def _maybe_start_lm_studio(target_base_url: str) -> None:
        auto_start = os.getenv("LM_STUDIO_AUTO_START", "false").lower() == "true"
        if not auto_start:
            return

        lm_path = os.getenv("LM_STUDIO_PATH")
        if not lm_path or not Path(lm_path).exists():
            messages.append("LM_STUDIO_AUTO_START is true but LM_STUDIO_PATH is not configured or does not exist.")
            return

        def _endpoint_active() -> bool:
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

        if _endpoint_active():
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

    _maybe_start_lm_studio(base_url)

    normalized_base_url, changed = _normalize_base_url(base_url, required_suffix="/v1")
    endpoint_ok = normalized_base_url.endswith("/v1")
    if changed:
        messages.append(f"Normalized base URL from '{base_url}' to '{normalized_base_url}'.")
    if not endpoint_ok:
        messages.append("Base URL is missing the required /v1 suffix.")

    client = OpenAI(api_key=api_key or "lm-studio", base_url=normalized_base_url)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are the local LM Studio model."},
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
                "local_llm", True, endpoint_ok, messages, full_output=full_output, finish_reason=finish_reason
            )
        messages.append("Local LLM returned an empty response.")
        return TestResult("local_llm", False, endpoint_ok, messages)
    except Exception as exc:
        error_msg = _format_provider_error("local_llm", exc)
        if "no models loaded" in str(exc).lower():
            error_msg += " Please start LM Studio, load a model (e.g., qwen3-4b-2507), and ensure it's running on the configured endpoint."
        messages.append(error_msg)
        return TestResult("local_llm", False, endpoint_ok, messages)


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
    "local_llm": _test_local_llm,
    "inception": _test_inception,
    "grok": _test_grok,
    "tetrate": _test_tetrate,
}


def _provider_base_url(provider: str) -> str:
    base_url: str
    if provider == "moonshot":
        base_url = (os.getenv("MOONSHOT_AI_BASE_URL") or "https://api.moonshot.ai/v1").rstrip("/")
    elif provider == "deepseek":
        base_url = (os.getenv("DEEPSEEK_AI_BASE_URL") or "https://api.deepseek.com").rstrip("/")
    elif provider == "gemini":
        base_url = os.getenv("GOOGLE_AI_BASE_URL") or "(default Google endpoint)"
    elif provider == "local_llm":
        base_url = (os.getenv("LOCAL_LLM_BASE_URL") or "http://localhost:1234/v1").rstrip("/")
    elif provider == "inception":
        base_url = (os.getenv("INCEPTION_AI_BASE_URL") or "https://api.inceptionlabs.ai/v1").rstrip("/")
    elif provider == "grok":
        base_url = os.getenv("XAI_API_HOST") or "api.x.ai"
    elif provider == "tetrate":
        base_url = (os.getenv("TETRATE_AI_BASE_URL") or "https://api.router.tetrate.ai/v1").rstrip("/")
    else:
        base_url = ""
    return base_url


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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    _ensure_env_loaded(".env", parser)
    args = parser.parse_args(argv)

    if args.env_file and args.env_file != ".env":
        _ensure_env_loaded(args.env_file, parser)

    # If provider specified via CLI arg, test once and exit
    if args.provider:
        tester = PROVIDER_TESTERS.get(args.provider)
        if not tester:
            parser.error(f"Unsupported provider: {args.provider}")

        missing_env = _provider_missing_env(args.provider)
        if missing_env:
            missing_list = ", ".join(missing_env)
            print(f"Provider '{args.provider}' is missing required environment variables: {missing_list}.")
            print("Update your .env (or pass --env-file) and try again.")
            return 1

        print(f"\nTesting provider: {args.provider}")

        # Start timer when making the API call
        start_time = time.time()
        result = tester(args.prompt, args.max_tokens)
        duration = time.time() - start_time

        _print_result(result)

        # Always show prompt and response if available
        if result.api_status and result.full_output:
            print("\nPrompt:")
            print(f'"{args.prompt}"')
            print("\nResponse:")
            print(result.full_output)
            print(f"\nResponse time: {duration:.2f}s")

            # Check for truncation
            if _is_response_truncated(result.full_output, result.finish_reason):
                print(f"⚠️  WARNING: Response appears truncated (finish_reason: {result.finish_reason or 'unknown'})")
                print(f"   Consider increasing --max-tokens (current: {args.max_tokens})")
            print()
        elif not result.api_status:
            # Show error details if API call failed
            if result.messages:
                print("\nError details:")
                for msg in result.messages:
                    print(f"  {msg}")

        return 0

    # Interactive mode - loop through providers
    while True:
        provider = _prompt_for_provider()
        if provider is None:
            break

        tester = PROVIDER_TESTERS.get(provider)
        if not tester:
            parser.error(f"Unsupported provider: {provider}")

        missing_env = _provider_missing_env(provider)
        if missing_env:
            missing_list = ", ".join(missing_env)
            print(f"\n⚠️  Provider '{provider}' is missing required environment variables: {missing_list}.")
            print("    Update your .env (or pass --env-file) and choose again.\n")
            continue

        print(f"\nTesting provider: {provider}")

        # Start timer when making the API call
        start_time = time.time()
        result = tester(args.prompt, args.max_tokens)
        duration = time.time() - start_time

        _print_result(result)

        # Always show prompt and response if available
        if result.api_status and result.full_output:
            print("\nPrompt:")
            print(f'"{args.prompt}"')
            print("\nResponse:")
            print(result.full_output)
            print(f"\nResponse time: {duration:.2f}s")

            # Check for truncation
            if _is_response_truncated(result.full_output, result.finish_reason):
                print(f"⚠️  WARNING: Response appears truncated (finish_reason: {result.finish_reason or 'unknown'})")
                print(f"   Consider increasing --max-tokens (current: {args.max_tokens})")
            print()
        elif not result.api_status:
            # Show error details if API call failed
            if result.messages:
                print("\nError details:")
                for msg in result.messages:
                    print(f"  {msg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
