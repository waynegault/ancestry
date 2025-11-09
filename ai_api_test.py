#!/usr/bin/env python3
"""Interactive AI provider connectivity tester.

Loads credentials from the project's .env file, asks which provider to test,
validates the configured endpoint, and attempts a simple completion request.
Currently supports Comet, OpenRouter, DeepSeek, and Google Gemini.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

try:
    import google.generativeai as genai
    GENAI_IMPORT_ERROR: str | None = None
except Exception as import_error:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]
    GENAI_IMPORT_ERROR = str(import_error)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]

DEFAULT_PROMPT = "Please confirm you can see this message."
PROVIDERS = ("comet", "openrouter", "moonshot", "deepseek", "gemini", "local_llm")

PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "comet": "Comet API",
    "openrouter": "OpenRouter (Moonshot Kimi)",
    "moonshot": "Moonshot (Kimi)",
    "deepseek": "DeepSeek",
    "gemini": "Google Gemini",
    "local_llm": "Local LLM (LM Studio)",
}


@dataclass
class TestResult:
    provider: str
    api_status: bool
    endpoint_status: bool
    messages: list[str] = field(default_factory=list)
    full_output: Optional[str] = None


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


def _ensure_env_loaded(env_file: str, parser: argparse.ArgumentParser) -> None:
    if not env_file:
        return
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = Path(__file__).resolve().parent / env_path
    if not env_path.exists():
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


def _print_result(result: TestResult) -> None:
    print(f"\n=== {result.provider.upper()} Test Summary ===")
    print(f"Endpoint check : {'PASS' if result.endpoint_status else 'FAIL'}")
    print(f"API call       : {'PASS' if result.api_status else 'FAIL'}")
    if result.messages:
        print("Details:")
        for msg in result.messages:
            print(f"  - {msg}")


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


def _test_comet(prompt: str, max_tokens: int) -> TestResult:
    messages: list[str] = []
    if OpenAI is None:
        messages.append("OpenAI library not available. Install the openai package.")
        return TestResult("comet", False, False, messages)

    api_key = os.getenv("COMET_API_KEY")
    base_url = os.getenv("COMET_AI_BASE_URL", "https://api.cometapi.com/v1")
    model_name = os.getenv("COMET_AI_MODEL", "gpt-4o-mini")

    if not api_key:
        messages.append("COMET_API_KEY not configured.")
        return TestResult("comet", False, False, messages)

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
                {"role": "system", "content": "You are Comet, an AI assistant."},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            stream=False,
            temperature=0.7,
            max_tokens=max_tokens,
        )
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            full_output = completion.choices[0].message.content.strip()
            messages.append(_build_messages_preview(full_output))
            return TestResult("comet", True, endpoint_ok, messages, full_output=full_output)
        messages.append("Received empty response payload from Comet.")
        return TestResult("comet", False, endpoint_ok, messages)
    except Exception as exc:  # pylint: disable=broad-except
        messages.append(_format_provider_error("comet", exc))
        return TestResult("comet", False, endpoint_ok, messages)


def _default_openrouter_headers() -> dict[str, str] | None:
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_TITLE")
    headers: dict[str, str] = {}
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers or None


def _test_openrouter(prompt: str, max_tokens: int) -> TestResult:
    messages: list[str] = []
    if OpenAI is None:
        messages.append("OpenAI library not available. Install the openai package.")
        return TestResult("openrouter", False, False, messages)

    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model_name = os.getenv("OPENROUTER_MODEL", "moonshotai/kimi-k2:free")

    if not api_key:
        messages.append("OPENROUTER_API_KEY not configured.")
        return TestResult("openrouter", False, False, messages)

    normalized_base_url, changed = _normalize_base_url(
        base_url,
        required_suffix="/api/v1",
        drop_suffixes=("/chat/completions",),
    )
    endpoint_ok = normalized_base_url.endswith("/api/v1")
    if changed:
        messages.append(f"Normalized base URL from '{base_url}' to '{normalized_base_url}'.")
    if not endpoint_ok:
        messages.append("Base URL is missing the required /api/v1 suffix.")

    headers = _default_openrouter_headers()
    if headers:
        messages.append("Applying OpenRouter headers for referer/title identification.")

    client = OpenAI(api_key=api_key, base_url=normalized_base_url, default_headers=headers)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are Kimi, accessible via OpenRouter."},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            stream=False,
            temperature=0.7,
            max_tokens=max_tokens,
        )
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            full_output = completion.choices[0].message.content.strip()
            messages.append(_build_messages_preview(full_output))
            return TestResult("openrouter", True, endpoint_ok, messages, full_output=full_output)
        messages.append("Received empty response payload from OpenRouter.")
        return TestResult("openrouter", False, endpoint_ok, messages)
    except Exception as exc:  # pylint: disable=broad-except
        messages.append(_format_provider_error("openrouter", exc))
        return TestResult("openrouter", False, endpoint_ok, messages)


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
            messages.append(_build_messages_preview(full_output))
            return TestResult("moonshot", True, endpoint_ok, messages, full_output=full_output)
        messages.append("Moonshot returned an empty response.")
        return TestResult("moonshot", False, endpoint_ok, messages)
    except Exception as exc:  # pylint: disable=broad-except
        messages.append(_format_provider_error("moonshot", exc))
        return TestResult("moonshot", False, endpoint_ok, messages)


def _test_deepseek(prompt: str, max_tokens: int) -> TestResult:
    messages: list[str] = []
    if OpenAI is None:
        messages.append("OpenAI library not available. Install the openai package.")
        return TestResult("deepseek", False, False, messages)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_AI_BASE_URL", "https://api.deepseek.com/v1")
    model_name = os.getenv("DEEPSEEK_AI_MODEL", "deepseek-chat")

    if not api_key:
        messages.append("DEEPSEEK_API_KEY not configured.")
        return TestResult("deepseek", False, False, messages)

    normalized_base_url, changed = _normalize_base_url(base_url, required_suffix="/v1")
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
                {"role": "system", "content": "You are DeepSeek, an AI assistant."},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            stream=False,
            temperature=0.7,
            max_tokens=max_tokens,
        )
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            full_output = completion.choices[0].message.content.strip()
            messages.append(_build_messages_preview(full_output))
            return TestResult("deepseek", True, endpoint_ok, messages, full_output=full_output)
        messages.append("Received empty response payload from DeepSeek.")
        return TestResult("deepseek", False, endpoint_ok, messages)
    except Exception as exc:  # pylint: disable=broad-except
        messages.append(_format_provider_error("deepseek", exc))
        return TestResult("deepseek", False, endpoint_ok, messages)


def _test_gemini(prompt: str, max_tokens: int) -> TestResult:
    messages: list[str] = []
    if genai is None:
        detail = GENAI_IMPORT_ERROR or "Install google-generativeai."
        messages.append(f"google-generativeai package not available ({detail}).")
        return TestResult("gemini", False, False, messages)

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GOOGLE_AI_MODEL", "gemini-1.5-pro")
    base_url = os.getenv("GOOGLE_AI_BASE_URL", "")

    if not api_key:
        messages.append("GOOGLE_API_KEY not configured.")
        return TestResult("gemini", False, False, messages)

    endpoint_ok = True
    if base_url:
        messages.append(f"Using custom Gemini endpoint: {base_url}")
    else:
        messages.append("Using default Google Generative Language endpoint.")

    try:
        configure_fn: Callable[..., Any] | None = getattr(genai, "configure", None)
        generation_config_cls: Any = getattr(genai, "GenerationConfig", None)
        generative_model_cls: Any = getattr(genai, "GenerativeModel", None)

        if not callable(configure_fn) or generation_config_cls is None or generative_model_cls is None:
            messages.append("google-generativeai module is missing required interfaces (configure/GenerativeModel).")
            return TestResult("gemini", False, endpoint_ok, messages)

        configure_fn(api_key=api_key)
        model = generative_model_cls(model_name)
        response = model.generate_content(
            prompt,
            generation_config=generation_config_cls(
                candidate_count=1,
                max_output_tokens=max_tokens,
                temperature=0.7,
            ),
        )
        if not response.candidates:
            messages.append("Gemini returned no candidates.")
            return TestResult("gemini", False, endpoint_ok, messages)
        text_parts = []
        for part in response.candidates[0].content.parts:
            value = getattr(part, "text", "")
            if value:
                text_parts.append(value)
        combined = " ".join(text_parts).strip()
        if combined:
            messages.append(_build_messages_preview(combined))
            return TestResult("gemini", True, endpoint_ok, messages, full_output=combined)
        messages.append("Gemini response contained no text content.")
        return TestResult("gemini", False, endpoint_ok, messages)
    except Exception as exc:  # pylint: disable=broad-except
        messages.append(_format_provider_error("gemini", exc))
        return TestResult("gemini", False, endpoint_ok, messages)


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
            messages.append(_build_messages_preview(full_output))
            return TestResult("local_llm", True, endpoint_ok, messages, full_output=full_output)
        messages.append("Local LLM returned an empty response.")
        return TestResult("local_llm", False, endpoint_ok, messages)
    except Exception as exc:  # pylint: disable=broad-except
        messages.append(_format_provider_error("local_llm", exc))
        return TestResult("local_llm", False, endpoint_ok, messages)


PROVIDER_TESTERS: dict[str, Callable[[str, int], TestResult]] = {
    "comet": _test_comet,
    "openrouter": _test_openrouter,
    "moonshot": _test_moonshot,
    "deepseek": _test_deepseek,
    "gemini": _test_gemini,
    "local_llm": _test_local_llm,
}


def _provider_base_url(provider: str) -> str:
    if provider == "comet":
        return os.getenv("COMET_AI_BASE_URL") or "https://api.cometapi.com/v1"
    if provider == "openrouter":
        return os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    if provider == "moonshot":
        return os.getenv("MOONSHOT_AI_BASE_URL") or "https://api.moonshot.ai/v1"
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_AI_BASE_URL") or "https://api.deepseek.com/v1"
    if provider == "gemini":
        return os.getenv("GOOGLE_AI_BASE_URL") or "(default Google endpoint)"
    if provider == "local_llm":
        return os.getenv("LOCAL_LLM_BASE_URL") or "http://localhost:1234/v1"
    return ""


def _prompt_for_provider() -> str:
    print("Available providers:")
    provider_list = list(PROVIDERS)
    for idx, provider in enumerate(provider_list, start=1):
        display_name = PROVIDER_DISPLAY_NAMES.get(provider, provider.capitalize())
        base_url = _provider_base_url(provider)
        print(f"  {idx}. {display_name} [{base_url}]")

    while True:
        choice = input("Enter the number of the provider you want to test: ").strip()
        if not choice.isdigit():
            print("Please enter a number from the list above.")
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
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=256, help="Maximum tokens/output length.")
    parser.add_argument("--env-file", dest="env_file", default=".env", help="Path to .env file to load before running.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    _ensure_env_loaded(".env", parser)
    args = parser.parse_args(argv)

    if args.env_file and args.env_file != ".env":
        _ensure_env_loaded(args.env_file, parser)

    provider = args.provider or _prompt_for_provider()
    tester = PROVIDER_TESTERS.get(provider)
    if not tester:
        parser.error(f"Unsupported provider: {provider}")

    print(f"\nTesting provider: {provider}")
    result = tester(args.prompt, args.max_tokens)
    _print_result(result)

    if result.api_status and result.full_output:
        follow_up = input("The model responded successfully. View the full output? [y/N]: ").strip().lower()
        if follow_up in ("y", "yes"):
            print("\n=== Full Response ===")
            print(result.full_output)
            print("=== End Full Response ===")

    return 0 if result.api_status else 1


if __name__ == "__main__":
    sys.exit(main())
