#!/usr/bin/env python3
"""Interactive AI provider connectivity tester.

Loads credentials from the project's .env file, asks which provider to test,
validates the configured endpoint, and attempts a simple completion request.
Currently supports Moonshot, DeepSeek, Google Gemini, and Local LLM.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

try:
    import google.generativeai as genai  # type: ignore[import-untyped]
    GENAI_IMPORT_ERROR: str | None = None
except Exception as import_error:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]
    GENAI_IMPORT_ERROR = str(import_error)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]

DEFAULT_PROMPT = "I'm interested in geneology. How many great-great-great grandparents did I have?"
PROVIDERS = ("moonshot", "deepseek", "gemini", "local_llm")

PROVIDER_DISPLAY_NAMES: dict[str, str] = {
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
    full_output: str | None = None


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
    print(f"API call : {'PASS' if result.api_status else 'FAIL'}")
    # Removed the Details section to simplify output


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
    model_name = os.getenv("GOOGLE_AI_MODEL", "gemini-1.5-flash-latest")
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

        # Configure API first (required for list_models)
        configure_fn(api_key=api_key)

        # List available models with generateContent support
        list_models_fn = getattr(genai, "list_models", None)
        if callable(list_models_fn):
            try:
                messages.append("\n📋 Available Gemini models with generateContent support:")
                model_count = 0
                found_configured = False
                for model in list_models_fn():  # type: ignore[misc]
                    methods = getattr(model, "supported_generation_methods", [])
                    if "generateContent" in methods:
                        name = getattr(model, "name", "").replace("models/", "")
                        if name:
                            marker = " ← CONFIGURED" if name == model_name else ""
                            messages.append(f"   • {name}{marker}")
                            model_count += 1
                            if name == model_name:
                                found_configured = True
                            # Limit output to first 5 models
                            if model_count >= 5:
                                messages.append("   ... (additional models available)")
                                break
                if model_count == 0:
                    messages.append("   ⚠️  No models found with generateContent support")
                elif not found_configured:
                    messages.append(f"   ⚠️  Configured model '{model_name}' not found in available models")
                messages.append("")  # Blank line for readability
            except Exception as e:
                messages.append(f"   ⚠️  Could not list models: {e}")
                messages.append("")  # Blank line for readability

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
        error_msg = _format_provider_error("gemini", exc)
        if "is not found" in error_msg.lower() or "not supported" in error_msg.lower():
            # Suggest available models from the list
            try:
                list_models_fn = getattr(genai, "list_models", None)
                if callable(list_models_fn):
                    available_models = []
                    for model in list_models_fn():  # type: ignore[misc]
                        methods = getattr(model, "supported_generation_methods", [])
                        if "generateContent" in methods:
                            name = getattr(model, "name", "").replace("models/", "")
                            if name:
                                available_models.append(name)
                            if len(available_models) >= 3:
                                break
                    if available_models:
                        error_msg += f" Try: {', '.join(available_models)}"
            except Exception:
                pass
            if "Try:" not in error_msg:
                error_msg += " Try updating GOOGLE_AI_MODEL in .env to a supported model."
        messages.append(error_msg)
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

    # Auto-start LM Studio if configured
    auto_start = os.getenv("LM_STUDIO_AUTO_START", "false").lower() == "true"
    if auto_start:
        lm_path = os.getenv("LM_STUDIO_PATH")
        if lm_path and Path(lm_path).exists():
            # Check if LM Studio is already running by testing the endpoint first
            try:
                import socket
                from urllib.parse import urlparse
                parsed = urlparse(base_url)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((parsed.hostname or "localhost", parsed.port or 1234))
                sock.close()
                already_running = (result == 0)
            except Exception:
                already_running = False

            if not already_running:
                try:
                    subprocess.Popen([lm_path], shell=True)
                    startup_timeout = int(os.getenv("LM_STUDIO_STARTUP_TIMEOUT", "60"))
                    messages.append(f"Starting LM Studio (waiting {min(startup_timeout, 10)}s for initialization)...")
                    time.sleep(min(startup_timeout, 10))
                except Exception as e:
                    messages.append(f"Failed to start LM Studio: {e}")
            else:
                messages.append("LM Studio is already running.")
        else:
            messages.append("LM_STUDIO_AUTO_START is true but LM_STUDIO_PATH is not configured or does not exist.")

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
        error_msg = _format_provider_error("local_llm", exc)
        if "no models loaded" in str(exc).lower():
            error_msg += " Please start LM Studio, load a model (e.g., qwen3-4b-2507), and ensure it's running on the configured endpoint."
        messages.append(error_msg)
        return TestResult("local_llm", False, endpoint_ok, messages)


PROVIDER_TESTERS: dict[str, Callable[[str, int], TestResult]] = {
    "moonshot": _test_moonshot,
    "deepseek": _test_deepseek,
    "gemini": _test_gemini,
    "local_llm": _test_local_llm,
}


def _provider_base_url(provider: str) -> str:
    if provider == "moonshot":
        return (os.getenv("MOONSHOT_AI_BASE_URL") or "https://api.moonshot.ai/v1").rstrip("/")
    if provider == "deepseek":
        return (os.getenv("DEEPSEEK_AI_BASE_URL") or "https://api.deepseek.com").rstrip("/")
    if provider == "gemini":
        return os.getenv("GOOGLE_AI_BASE_URL") or "(default Google endpoint)"
    if provider == "local_llm":
        return (os.getenv("LOCAL_LLM_BASE_URL") or "http://localhost:1234/v1").rstrip("/")
    return ""


def _prompt_for_provider() -> str | None:
    print("Available providers:")
    provider_list = list(PROVIDERS)
    for idx, provider in enumerate(provider_list, start=1):
        display_name = PROVIDER_DISPLAY_NAMES.get(provider, provider.capitalize())
        base_url = _provider_base_url(provider)
        print(f"  {idx}. {display_name} [{base_url}]")

    while True:
        choice = input("Enter the number of the provider you want to test (or 'q' to quit): ").strip()
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
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=256, help="Maximum tokens/output length.")
    parser.add_argument("--env-file", dest="env_file", default=".env", help="Path to .env file to load before running.")
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

        print(f"\nTesting provider: {args.provider}")

        # Start timer when making the API call
        start_time = time.time()
        result = tester(args.prompt, args.max_tokens)
        duration = time.time() - start_time

        _print_result(result)

        # Always show prompt and response if available
        if result.api_status and result.full_output:
            print(f"\nPrompt:")
            print(f'"{args.prompt}"')
            print(f"\nResponse:")
            print(result.full_output)
            print(f"\nResponse time: {duration:.2f}s")
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

        print(f"\nTesting provider: {provider}")

        # Start timer when making the API call
        start_time = time.time()
        result = tester(args.prompt, args.max_tokens)
        duration = time.time() - start_time

        _print_result(result)

        # Always show prompt and response if available
        if result.api_status and result.full_output:
            print(f"\nPrompt:")
            print(f'"{args.prompt}"')
            print(f"\nResponse:")
            print(result.full_output)
            print(f"\nResponse time: {duration:.2f}s")
        elif not result.api_status:
            # Show error details if API call failed
            if result.messages:
                print("\nError details:")
                for msg in result.messages:
                    print(f"  {msg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
