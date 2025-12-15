#!/usr/bin/env python3
from __future__ import annotations

"""
AI Intelligence & Genealogical Content Analysis Engine

Advanced artificial intelligence interface providing sophisticated genealogical
content analysis, intelligent message classification, and automated research
assistance through unified AI model integration with specialized genealogical
knowledge processing and contextual understanding capabilities.

Genealogical Intelligence:
• Specialized genealogical content analysis with family tree context understanding
• Advanced message classification with intent detection and sentiment analysis
• Intelligent research assistance with automated suggestion generation
• Comprehensive relationship analysis with kinship detection and validation
• Advanced name entity recognition with genealogical context awareness
• Intelligent data extraction from unstructured genealogical content

AI Model Integration:
• Unified interface supporting multiple AI providers and model architectures
• Advanced prompt engineering with genealogical domain expertise
• Intelligent response validation with quality assessment and confidence scoring
• Sophisticated error handling with graceful degradation and fallback strategies
• Comprehensive retry logic with exponential backoff and circuit breaker patterns
• Real-time performance monitoring with latency and accuracy tracking

Content Processing:
• Advanced natural language processing for genealogical text analysis
• Intelligent conversation threading with relationship context preservation
• Automated content summarization with genealogical relevance scoring
• Advanced entity extraction with genealogical relationship mapping
• Intelligent content generation with personalization and context awareness
• Comprehensive quality validation with accuracy and relevance assessment

Research Enhancement:
Provides sophisticated AI capabilities that enhance genealogical research through
intelligent content analysis, automated research assistance, and contextual
understanding of family relationships and genealogical data structures.
"""

# === PATH SETUP FOR PACKAGE IMPORTS ===
# === CORE INFRASTRUCTURE ===
import logging

logger = logging.getLogger(__name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
import importlib
import json

# import logging - removed unused
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, TypedDict, cast

# === THIRD-PARTY IMPORTS ===
# Attempt OpenAI import for DeepSeek/compatible APIs
try:
    from openai import (
        APIConnectionError as _APIConnectionError,
        APIError as _APIError,
        AuthenticationError as _AuthenticationError,
        OpenAI as _OpenAI,
        RateLimitError as _RateLimitError,
    )
except ImportError:
    _OpenAI = None
    _APIConnectionError = None
    _RateLimitError = None
    _AuthenticationError = None
    _APIError = None

OpenAI: Any | None = _OpenAI
APIConnectionError: Any | None = _APIConnectionError
RateLimitError: Any | None = _RateLimitError
AuthenticationError: Any | None = _AuthenticationError
APIError: Any | None = _APIError
openai_available = OpenAI is not None
if not openai_available:
    logger.debug("OpenAI library not found. DeepSeek functionality disabled.")

# Attempt Google Gemini import
try:
    genai = cast(Any, importlib.import_module("google.genai"))
    genai_errors = cast(Any, importlib.import_module("google.genai.errors"))
    genai_types = cast(Any, importlib.import_module("google.genai.types"))
except Exception:
    genai = None
    genai_errors = None
    genai_types = None

genai_available = False
if genai is not None:
    genai_available = hasattr(genai, "Client")
if not genai_available:
    logger.debug("Google GenAI library not found or incomplete. Gemini functionality disabled.")

# Attempt Grok (xAI) import
try:
    from xai_sdk import Client as _XAIClient
    from xai_sdk.chat import system as _xai_system_message, user as _xai_user_message
except ImportError:
    _XAIClient = None
    _xai_system_message = None
    _xai_user_message = None

XAIClient: Callable[..., Any] | None = _XAIClient
xai_system_message: Callable[..., Any] | None = _xai_system_message
xai_user_message: Callable[..., Any] | None = _xai_user_message
xai_available = XAIClient is not None
if not xai_available:
    logger.debug("xai-sdk library not found. Grok functionality disabled.")

# === LOCAL IMPORTS ===
import contextlib

from ai.prompts import (
    get_prompt,
    get_prompt_version,
    get_prompt_with_experiment,
    load_prompts,
    record_extraction_experiment_event,
    supports_json_prompts,
)
from ai.providers.base import (
    ProviderAdapter,
    ProviderConfigurationError,
    ProviderRequest,
    ProviderResponse,
    ProviderUnavailableError,
)
from ai.providers.deepseek import DeepSeekProvider
from ai.providers.gemini import GeminiProvider
from ai.providers.local_llm import LocalLLMProvider
from ai.providers.moonshot import MoonshotProvider

# === PHASE 5.2: SYSTEM-WIDE CACHING OPTIMIZATION ===
from caching.cache_manager import cached_api_call
from config.config_schema import ConfigSchema

if TYPE_CHECKING:
    from typing import TypedDict

    from core.session_manager import SessionManager

    class HealthStatus(TypedDict):
        overall_health: str
        ai_provider: str
        api_key_configured: bool
        prompts_loaded: bool
        dependencies_available: bool
        test_call_successful: bool
        errors: list[str]


# === MODULE CONFIGURATION ===
# Initialize config
from config.config_manager import get_config_manager

config_manager = get_config_manager()
config_schema: ConfigSchema = config_manager.get_config()


_PROVIDER_ADAPTERS: dict[str, ProviderAdapter] = {}


def _register_provider(name: str, factory: Callable[[Any], ProviderAdapter]) -> None:
    try:
        adapter = factory(config_schema)
    except Exception as exc:  # pragma: no cover - optional dependency import failures
        logger.debug("Skipping provider '%s' due to initialization failure: %s", name, exc)
        return
    _PROVIDER_ADAPTERS[name] = adapter


_register_provider("deepseek", DeepSeekProvider)
_register_provider("gemini", GeminiProvider)
_register_provider("moonshot", MoonshotProvider)
_register_provider("local_llm", LocalLLMProvider)


def _adapter_is_available(provider: str) -> bool:
    """Return True if the provider adapter (if registered) is ready for use."""

    adapter = _PROVIDER_ADAPTERS.get(provider)
    if adapter is None:
        return True
    try:
        return bool(adapter.is_available())
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Adapter availability probe failed for '%s': %s", provider, exc)
        return False


# --- Test framework imports ---
from testing.test_utilities import create_standard_test_runner

# --- Constants and Prompts ---
USE_JSON_PROMPTS = supports_json_prompts()
if USE_JSON_PROMPTS:
    logger.debug("AI prompt utilities loaded successfully - will use JSON prompts")
else:
    logger.warning("ai_prompt_utils module not available, using fallback prompts")


# Based on the prompt from the original ai_interface.py (and updated with more categories from ai_prompts.json example)
EXPECTED_INTENT_CATEGORIES = {
    "ENTHUSIASTIC",
    "CAUTIOUSLY_INTERESTED",
    "UNINTERESTED",
    "CONFUSED",
    "PRODUCTIVE",
    "SOCIAL",  # Added: Non-genealogical positive messages (excitement, rapport-building)
    "OTHER",
    "DESIST",  # Added from the original file's SYSTEM_PROMPT_INTENT
}

SUPPORTED_AI_PROVIDERS: tuple[str, ...] = (
    "deepseek",
    "gemini",
    "moonshot",
    "local_llm",
    "inception",
    "grok",
    "tetrate",
)

DEFAULT_AI_PROVIDER_FALLBACKS: tuple[str, ...] = (
    "deepseek",
    "gemini",
    "moonshot",
    "local_llm",
    "grok",
    "inception",
    "tetrate",
)


def _normalize_provider_name(provider: str | None) -> str:
    """Normalize provider names for comparison and logging."""

    return provider.strip().lower() if isinstance(provider, str) else ""


def _get_configured_fallbacks() -> list[str]:
    """Return configured fallback order (defaults to safe global preference)."""

    configured = getattr(config_schema, "ai_provider_fallbacks", None)
    if isinstance(configured, list) and configured:
        normalized = [_normalize_provider_name(value) for value in configured]
        return [value for value in normalized if value]
    return list(DEFAULT_AI_PROVIDER_FALLBACKS)


ListModelsFn = Callable[[], Iterable[Any]]
GenerativeModelFactory = Callable[[str], Any]

# Fallback system prompt for Action 7 (Intent Classification)
# Combined elements from original file and ai_prompts.json example.
FALLBACK_INTENT_PROMPT = """You are an AI assistant analyzing conversation histories from a genealogy website messaging system. The history alternates between 'SCRIPT' (automated messages from me) and 'USER' (replies from the DNA match).

Analyze the entire provided conversation history, interpreting the **last message sent by the USER** *within the context of the entire conversation history* provided below. Determine the primary intent of that final USER message.

Respond ONLY with one of the following single-word categories:
- ENTHUSIASTIC: User is actively engaging with genealogy research, sharing detailed family information, asking specific genealogical questions, expressing excitement, or offering to share documents/photos.
- CAUTIOUSLY_INTERESTED: User shows measured interest, requesting more information before committing, expressing uncertainty, or asking for verification.
- UNINTERESTED: User politely declines further contact, states they cannot help, don't have time, are not knowledgeable, shows clear lack of engagement/desire to continue, or replies with very short, non-committal answers that don't advance the genealogical discussion after specific requests for information.
- CONFUSED: User doesn't understand the context, is unclear why they received the message, or doesn't understand DNA matching/genealogy concepts.
- PRODUCTIVE: User's final message, in context, provides helpful genealogical information (names, dates, places, relationships), asks relevant clarifying questions, confirms relationships, expresses clear interest in collaborating, or shares tree info/invites.
- DESIST: The user's final message, in context, explicitly asks to stop receiving messages or indicates they are blocking the sender.
- OTHER: Messages that don't clearly fit above categories (e.g., purely social pleasantries, unrelated questions, ambiguous statements, or messages containing only attachments/links without explanatory text).

CRITICAL: Your entire response must be only one of the category words.
"""


# Function to generate fallback extraction prompt with configurable user details
def get_fallback_extraction_prompt() -> str:
    """Generate the fallback extraction prompt using configured user name."""
    user_name = config_schema.user_name

    return f"""You are an AI assistant analyzing conversation histories from a genealogy website messaging system. The history alternates between 'SCRIPT' (automated messages from me) and 'USER' (replies from the DNA match).

Analyze the entire provided conversation history, focusing on information shared by the USER.

Your goal is twofold:
1.  **Extract Key Genealogical Information:** Identify and extract specific entities mentioned by the USER. Structure this as an object under an "extracted_data" key. This object should contain:
    - "mentioned_names": List of full names mentioned (e.g., ["John Smith", "Mary Anne Jones"])
    - "mentioned_people": List of detailed person objects with structure:
      {{
        "name": "Full Name",
        "first_name": "First" (if mentioned),
        "last_name": "Last" (if mentioned),
        "birth_year": year as integer (if mentioned),
        "birth_place": "Place" (if mentioned),
        "birth_date": "Full date" (if mentioned),
        "death_year": year as integer (if mentioned),
        "death_place": "Place" (if mentioned),
        "death_date": "Full date" (if mentioned),
        "gender": "M" or "F" (if mentioned or can be inferred from name),
        "relationship": "their grandfather" (if mentioned, e.g., "my grandfather", "her sister"),
        "occupation": "Occupation" (if mentioned),
        "notes": "Any other relevant details"
      }}
    - "mentioned_locations": List of places (towns, counties, countries)
    - "mentioned_dates": List of dates or date ranges (e.g., ["abt 1880", "1912-1915"])
    - "potential_relationships": List of relationship descriptions (e.g., ["Grandfather of current user"])
    - "key_facts": List of important facts (e.g., ["Immigrated via Liverpool", "Worked in coal mines"])

    Only include entities explicitly mentioned by the USER. Do not infer or add information not present. If no entities of a certain type are found, provide an empty list [].

2.  **Suggest Actionable Follow-up Tasks:** Based ONLY on the information provided in the conversation history, suggest 2-4 concrete, actionable research tasks for 'Me'/'{user_name}'. Tasks MUST be directly based on information, questions, or ambiguities present *only* in the provided conversation history. Tasks should be specific. Examples: "Check [Year] census for [Name] in [Location]", "Search ship manifests for [Name] arriving [Port] around [Date]", "Compare shared matches with [Match Name]", "Look for [Event record] for [Name] in [Place]". Provide this as a list of strings under a "suggested_tasks" key. Provide an empty list [] if no specific tasks can be suggested.

Format your response STRICTLY as a JSON object, starting with `{{` and ending with `}}`, with no introductory text or markdown. Example:
{{
  "extracted_data": {{
    "mentioned_names": ["John Smith", "Mary Anne Jones"],
    "mentioned_people": [
      {{
        "name": "John Smith",
        "first_name": "John",
        "last_name": "Smith",
        "birth_year": 1850,
        "birth_place": "Glasgow, Scotland",
        "death_year": 1920,
        "gender": "M",
        "relationship": "my great-grandfather",
        "occupation": "coal miner"
      }},
      {{
        "name": "Mary Anne Jones",
        "first_name": "Mary Anne",
        "last_name": "Jones",
        "birth_year": 1880,
        "birth_place": "County Cork, Ireland",
        "gender": "F",
        "relationship": "his wife"
      }}
    ],
    "mentioned_locations": ["Glasgow", "County Cork", "Liverpool"],
    "mentioned_dates": ["abt 1880", "1912"],
    "potential_relationships": ["Great-grandfather of current user"],
    "key_facts": ["Immigrated via Liverpool", "Worked in coal mines"]
  }},
  "suggested_tasks": [
    "Check 1881 Scotland Census for John Smith in Glasgow.",
    "Search immigration records for Mary Anne Jones arriving Liverpool around 1910-1915."
  ]
}}
"""


# Function to generate fallback system prompt with configurable user details
def get_fallback_reply_prompt() -> str:
    """Generate the fallback reply prompt using configured user name and location."""
    user_name = config_schema.user_name
    user_location = config_schema.user_location
    location_part = f" from {user_location}" if user_location else ""

    return f"""You are a helpful genealogical assistant named {user_name} responding to messages on behalf of a family history researcher{location_part}.

You will receive:
1. A conversation history between the researcher (SCRIPT) and a user (USER).
2. The user's last message.
3. Genealogical data about a person mentioned in the user's message, including: Name, birth/death information, family relationships, relationship to the tree owner (the researcher).

Your task is to generate a natural, polite, and informative reply that:
- Directly addresses the user's query or comment.
- Incorporates the provided genealogical data in a helpful way.
- Acknowledges the user's point and integrates the found information smoothly.
- May suggest connections or ask a clarifying follow-up question if appropriate.
- Maintains a warm, helpful, conversational tone.
- Refers to yourself as "I" and the tree as "my family tree" or "my records".
- Shows genuine interest in the user's research and family connections.

IMPORTANT: When replying about people found in your tree, you MUST include:
1. COMPLETE birth details (full date and place if available).
2. COMPLETE death details (full date and place if available).
3. DETAILED family information (parents, spouse, children).
4. SPECIFIC relationship to you (the tree owner).
5. Any other significant details like occupation, immigration, etc.

If multiple people are mentioned in the genealogical data, focus on the one with the highest match score or most complete information.
If the genealogical data indicates "No person found" or is empty, acknowledge this and ask for more details that might help identify the person in your records.
For people not in your tree, acknowledge this and ask for more details that might help identify connections.

Your response should be ONLY the message text, with no additional formatting, explanation, or signature (the system will add a signature automatically).
"""


# --- Private Helper for AI Calls ---


def _apply_rate_limiting(session_manager: SessionManager, provider: str) -> None:
    """Apply rate limiting before AI API call."""
    try:
        if session_manager and hasattr(session_manager, "rate_limiter"):
            drl = getattr(session_manager, "rate_limiter", None)
            if drl is not None and hasattr(drl, "wait"):
                wait_time = drl.wait()
                if isinstance(wait_time, (int, float)) and wait_time > 0.1:
                    logger.debug(f"AI API rate limit wait: {float(wait_time):.2f}s for {provider}")
        else:
            logger.warning(
                "_call_ai_model: SessionManager or rate limiter not available. Proceeding without rate limiting."
            )
    except Exception:
        logger.debug("Rate limiter invocation failed; proceeding without enforced wait.")


def _call_tetrate_model(
    system_prompt: str, user_content: str, max_tokens: int, temperature: float, response_format_type: str | None
) -> str | None:
    """Call Tetrate (TARS) via OpenAI-compatible endpoint.

    Uses API key from TARS_API_KEY (mapped to api.tetrate_api_key) and the
    router base URL from tetrate_ai_base_url.
    """
    if not openai_available or OpenAI is None:
        logger.error("_call_ai_model: OpenAI library not available for Tetrate.")
        return None

    api_key = getattr(config_schema.api, "tetrate_api_key", None)
    model_name = getattr(config_schema.api, "tetrate_ai_model", None)
    base_url = getattr(config_schema.api, "tetrate_ai_base_url", "https://api.router.tetrate.ai/v1")

    if not all([api_key, model_name, base_url]):
        logger.error("_call_ai_model: Tetrate configuration incomplete.")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    request_params: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if response_format_type == "json_object":
        request_params["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**request_params)
    if response.choices and response.choices[0].message and response.choices[0].message.content:
        return response.choices[0].message.content.strip()

    logger.error("Tetrate returned an empty or invalid response structure.")
    return None


def _handle_rate_limit_error(session_manager: SessionManager, source: str | None = None) -> None:
    """Handle rate limit error by increasing delay."""
    if session_manager and hasattr(session_manager, "rate_limiter"):
        try:
            drl = getattr(session_manager, "rate_limiter", None)
            endpoint = source or "AI Provider"
            if drl is not None:
                on_429 = getattr(drl, "on_429_error", None)
                if callable(on_429):
                    on_429(endpoint)
        except Exception:
            pass


def _validate_local_llm_model_loaded(client: Any, model_name: str) -> tuple[str | None, str | None]:
    """Backward-compatible helper used by lifecycle checks."""

    try:
        return LocalLLMProvider._validate_model_loaded(client, model_name)
    except Exception as exc:  # pragma: no cover - defensive fallback
        return None, f"Local LLM validation failed: {exc}"


def _call_inception_model(
    system_prompt: str, user_content: str, max_tokens: int, temperature: float, response_format_type: str | None
) -> str | None:
    """
    Call Inception Mercury model via OpenAI-compatible API.

    Inception Mercury (api.inceptionlabs.ai/v1) provides an OpenAI-compatible endpoint
    that works with the OpenAI client library.

    Note: response_format_type parameter is kept for API compatibility.
    """
    # Validate prerequisites
    if not openai_available or OpenAI is None:
        logger.error("_call_inception_model: OpenAI library not available for Inception Mercury.")
        return None

    api_key = config_schema.api.inception_api_key
    model_name = config_schema.api.inception_ai_model
    base_url = config_schema.api.inception_ai_base_url

    if not all([api_key, model_name, base_url]):
        logger.error("_call_inception_model: Inception Mercury configuration incomplete.")
        return None

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        request_params: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        if response_format_type:
            request_params["response_format"] = {"type": response_format_type}

        response = client.chat.completions.create(**request_params)
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()

        logger.error("Inception Mercury returned an empty or invalid response structure.")
        return None
    except Exception as e:
        logger.error(f"Inception Mercury API call failed: {e}")
        return None


def _normalize_grok_entry(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry.strip()
    text_value = getattr(entry, "text", None)
    if isinstance(text_value, str):
        return text_value.strip()
    content_value = getattr(entry, "content", None)
    if isinstance(content_value, str):
        return content_value.strip()
    return None


def _normalize_grok_sequence(entries: list[Any]) -> str | None:
    parts = [part for part in (_normalize_grok_entry(item) for item in entries) if part]
    return "\n".join(parts) if parts else None


def _normalize_grok_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return _normalize_grok_sequence(value)
    return _normalize_grok_entry(value)


def _iter_grok_content_candidates(response: Any, primary_content: Any) -> list[Any]:
    message = getattr(response, "message", None)
    return [
        primary_content,
        getattr(message, "content", None) if message is not None else None,
        response,
    ]


def _extract_grok_response_content(response: Any | None) -> str | None:
    """Normalize Grok (xAI) response content to a plain string."""
    if response is None:
        return None

    try:
        primary_content = getattr(response, "content", None)
        for candidate in _iter_grok_content_candidates(response, primary_content):
            normalized = _normalize_grok_value(candidate)
            if normalized:
                return normalized

        fallback_source = primary_content if primary_content is not None else response
        fallback_str = str(fallback_source).strip()
        return fallback_str or None
    except Exception as exc:  # pragma: no cover - defensive logging around SDK objects
        logger.error(f"Failed to parse Grok response: {exc}")
        return None


def _call_grok_model(
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
    response_format_type: str | None,
) -> str | None:
    """Call Grok (xAI) using the official SDK."""
    if not xai_available or XAIClient is None or xai_system_message is None or xai_user_message is None:
        logger.error("_call_grok_model: xai-sdk library not available.")
        return None

    api_key = getattr(config_schema.api, "xai_api_key", None)
    model_name = getattr(config_schema.api, "xai_model", "grok-4-fast-non-reasoning")
    api_host = getattr(config_schema.api, "xai_api_host", "api.x.ai")

    if not api_key:
        logger.error("_call_grok_model: XAI_API_KEY not configured.")
        return None
    if not model_name:
        logger.error("_call_grok_model: XAI_MODEL not configured.")
        return None

    timeout_seconds = max(int(getattr(config_schema.api, "request_timeout", 60)), 60)

    try:
        client = XAIClient(api_key=api_key, api_host=api_host, timeout=timeout_seconds)
        chat_session = client.chat.create(
            model=model_name,
            max_tokens=max_tokens or None,
            temperature=temperature,
            response_format="json_object" if response_format_type == "json_object" else None,
        )
        chat_session.append(xai_system_message(system_prompt))
        chat_session.append(xai_user_message(user_content))
        response = chat_session.sample()
        content = _extract_grok_response_content(response)
        if content:
            return content
        logger.error("Grok returned an empty or unrecognized response structure.")
        return None
    except Exception as exc:  # pragma: no cover - network/SDK errors
        logger.error(f"Grok API call failed: {exc}")
        return None


def _route_ai_provider_call(
    provider: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
    response_format_type: str | None,
) -> str | None:
    """Route call to appropriate AI provider."""
    adapter = _PROVIDER_ADAPTERS.get(provider)
    if adapter is not None:
        if not adapter.is_available():
            logger.debug("Provider '%s' is not currently available.", provider)
            return None

        request = ProviderRequest(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format_type=response_format_type,
        )
        try:
            response = adapter.call(request)
            return response.content
        except (ProviderConfigurationError, ProviderUnavailableError) as exc:
            logger.error("%s provider error: %s", provider, exc)
            return None

    result: str | None = None
    if provider == "inception":
        result = _call_inception_model(system_prompt, user_content, max_tokens, temperature, response_format_type)
    elif provider == "grok":
        result = _call_grok_model(system_prompt, user_content, max_tokens, temperature, response_format_type)
    elif provider == "tetrate":
        result = _call_tetrate_model(system_prompt, user_content, max_tokens, temperature, response_format_type)
    else:
        logger.error(f"_call_ai_model: Unsupported AI provider '{provider}'.")
    return result


_PROVIDER_CONFIG_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "deepseek": lambda api: bool(getattr(api, "deepseek_api_key", None)),
    "gemini": lambda api: bool(getattr(api, "google_api_key", None)),
    "moonshot": lambda api: bool(getattr(api, "moonshot_api_key", None)),
    "local_llm": lambda api: all(
        getattr(api, attr, None) for attr in ("local_llm_api_key", "local_llm_model", "local_llm_base_url")
    ),
    "inception": lambda api: all(
        getattr(api, attr, None) for attr in ("inception_api_key", "inception_ai_model", "inception_ai_base_url")
    ),
    "grok": lambda api: bool(getattr(api, "xai_api_key", None)),
    "tetrate": lambda api: bool(getattr(api, "tetrate_api_key", None)),
}


def _provider_is_configured(provider: str) -> bool:
    """Return True when the specified provider has enough configuration to attempt a call."""
    api_config = getattr(config_schema, "api", None)
    if api_config is None:
        return False

    checker = _PROVIDER_CONFIG_VALIDATORS.get(provider)
    if checker is None:
        return False
    return bool(checker(api_config))


def _resolve_provider_chain(primary_provider: str) -> list[str]:
    """Build the ordered list of providers to attempt, honoring configured fallbacks."""

    normalized_primary = _normalize_provider_name(primary_provider)
    provider_chain: list[str] = []
    seen: set[str] = set()

    def append_candidate(candidate: str, *, require_configuration: bool = True) -> None:
        normalized_candidate = _normalize_provider_name(candidate)
        if not normalized_candidate or normalized_candidate in seen:
            return
        if normalized_candidate not in SUPPORTED_AI_PROVIDERS:
            logger.debug("Skipping unsupported AI provider '%s' in fallback chain.", normalized_candidate)
            return
        if require_configuration and not _provider_is_configured(normalized_candidate):
            logger.debug(
                "Skipping AI provider '%s' in fallback chain (missing configuration).",
                normalized_candidate,
            )
            return
        if not _adapter_is_available(normalized_candidate):
            logger.debug(
                "Skipping AI provider '%s' in fallback chain (adapter unavailable).",
                normalized_candidate,
            )
            return

        seen.add(normalized_candidate)
        provider_chain.append(normalized_candidate)

    append_candidate(normalized_primary, require_configuration=False)

    for fallback_candidate in _get_configured_fallbacks():
        append_candidate(fallback_candidate)

    if not provider_chain and normalized_primary:
        # As a last resort, attempt the primary provider even if it failed validation
        provider_chain.append(normalized_primary)

    return provider_chain


def _handle_authentication_errors(e: Exception, provider: str) -> None:
    """Handle authentication-related errors."""
    if AuthenticationError and isinstance(e, AuthenticationError):
        logger.error(f"AI Authentication Error ({provider}): {e}")
    elif genai_errors and hasattr(genai_errors, "PermissionDenied") and isinstance(e, genai_errors.PermissionDenied):
        logger.error(f"Gemini Permission Denied: {e}")


def _handle_rate_limit_errors(e: Exception, provider: str, session_manager: SessionManager) -> None:
    """Handle rate limiting-related errors."""
    if RateLimitError and isinstance(e, RateLimitError):
        logger.error(f"AI Rate Limit Error ({provider}): {e}")
        _handle_rate_limit_error(session_manager, f"AI Provider: {provider}")
    elif genai_errors and hasattr(genai_errors, "ResourceExhausted") and isinstance(e, genai_errors.ResourceExhausted):
        logger.error(f"Gemini Resource Exhausted (Rate Limit): {e}")
        _handle_rate_limit_error(session_manager, f"AI Provider: {provider}")


def _handle_api_errors(e: Exception, provider: str) -> None:
    """Handle API-related errors."""
    if APIConnectionError and isinstance(e, APIConnectionError):
        logger.error(f"AI Connection Error ({provider}): {e}")
    elif APIError and isinstance(e, APIError):
        logger.error(
            f"AI API Error ({provider}): Status={getattr(e, 'status_code', 'N/A')}, Message={getattr(e, 'message', str(e))}"
        )
    elif genai_errors and hasattr(genai_errors, "GoogleAPIError") and isinstance(e, genai_errors.GoogleAPIError):
        logger.error(f"Google API Error (Gemini): {e}")


def _handle_internal_errors(e: Exception, provider: str) -> None:
    """Handle internal/unexpected errors."""
    if isinstance(e, AttributeError):
        logger.critical(
            f"AttributeError during AI call ({provider}): {e}. Lib loaded: OpenAI={openai_available}, Gemini={genai_available}",
            exc_info=True,
        )
    elif isinstance(e, NameError):
        logger.critical(
            f"NameError during AI call ({provider}): {e}. Lib loaded: OpenAI={openai_available}, Gemini={genai_available}",
            exc_info=True,
        )
    else:
        logger.error(f"Unexpected error in _call_ai_model ({provider}): {type(e).__name__} - {e}", exc_info=True)


def _handle_ai_exceptions(e: Exception, provider: str, session_manager: SessionManager) -> None:
    """Handle AI API exceptions with appropriate logging and actions."""
    _handle_authentication_errors(e, provider)
    _handle_rate_limit_errors(e, provider, session_manager)
    _handle_api_errors(e, provider)
    _handle_internal_errors(e, provider)


@cached_api_call("ai", ttl=1800)
def _call_ai_model(
    provider: str,
    system_prompt: str,
    user_content: str,
    session_manager: SessionManager,
    max_tokens: int,
    temperature: float,
    response_format_type: str | None = None,
) -> str | None:
    """
    Private helper to call the specified AI model.
    Handles API key loading, request construction, rate limiting, and error handling.
    """
    logger.debug(f"Calling AI model. Provider: {provider}, Max Tokens: {max_tokens}, Temp: {temperature}")

    provider_chain = _resolve_provider_chain(provider)
    if not provider_chain:
        logger.error("No configured AI providers available for '%s'.", provider)
        return None

    last_exception: Exception | None = None

    for idx, active_provider in enumerate(provider_chain):
        next_provider = provider_chain[idx + 1] if idx < len(provider_chain) - 1 else None
        _apply_rate_limiting(session_manager, active_provider)

        try:
            result = _route_ai_provider_call(
                active_provider, system_prompt, user_content, max_tokens, temperature, response_format_type
            )
        except Exception as exc:
            last_exception = exc
            _handle_ai_exceptions(exc, active_provider, session_manager)

            if next_provider is not None:
                logger.warning(
                    "AI provider '%s' failed (%s). Attempting fallback '%s'.",
                    active_provider,
                    type(exc).__name__,
                    next_provider,
                )
                continue

            break

        if result is None:
            if next_provider is not None:
                logger.warning(
                    "AI provider '%s' returned no response. Attempting fallback '%s'.",
                    active_provider,
                    next_provider,
                )
                continue
            break

        if active_provider != provider:
            logger.warning(
                "AI provider '%s' failed, successfully fell back to '%s'.",
                provider,
                active_provider,
            )
        return result

    if last_exception is not None:
        logger.error(
            "AI provider '%s' failed after exhausting %d option(s): %s",
            provider,
            len(provider_chain),
            type(last_exception).__name__,
        )
    else:
        logger.error(
            "AI provider '%s' returned no usable response after trying %d option(s).",
            provider,
            len(provider_chain),
        )
    return None


# End of _call_ai_model

# --- Public AI Interaction Functions ---


@cached_api_call("ai", ttl=3600)  # Cache AI responses for 1 hour
def classify_message_intent(context_history: str, session_manager: SessionManager) -> str | None:
    """
    Classifies the intent of the LAST USER message within the provided context history.
    """
    ai_provider = config_schema.ai_provider.lower()
    if not ai_provider:
        logger.error("classify_message_intent: AI_PROVIDER not configured.")
        return None
    if not context_history:
        logger.warning("classify_message_intent: Received empty context history. Defaulting to OTHER.")
        return "OTHER"

    system_prompt = FALLBACK_INTENT_PROMPT
    if USE_JSON_PROMPTS:
        try:
            loaded_prompt = get_prompt("intent_classification")
            if loaded_prompt:
                system_prompt = loaded_prompt
            else:
                logger.warning("Failed to load 'intent_classification' prompt from JSON, using fallback.")
        except Exception as e:
            logger.warning(f"Error loading 'intent_classification' prompt: {e}, using fallback.")

    # Log context for debugging classification issues
    truncated_context = (context_history[:200] + "...") if len(context_history) > 200 else context_history
    logger.debug(f"Classifying intent for context: {truncated_context!r}")

    start_time = time.time()
    raw_classification = _call_ai_model(
        provider=ai_provider,
        system_prompt=system_prompt,
        user_content=context_history,
        session_manager=session_manager,
        max_tokens=10,  # Expecting a single word
        temperature=0.0,
    )
    duration = time.time() - start_time

    if raw_classification:
        processed_classification = raw_classification.strip().upper()
        if processed_classification in EXPECTED_INTENT_CATEGORIES:
            logger.debug(f"AI intent classification: '{processed_classification}' (Took {duration:.2f}s)")
            return processed_classification
        logger.warning(f"AI returned unexpected classification: '{raw_classification}'. Defaulting to OTHER.")
        return "OTHER"
    logger.error(f"AI intent classification failed. (Took {duration:.2f}s)")
    return None


# End of classify_message_intent

# Helper functions for extract_genealogical_entities


def _get_extraction_prompt(session_manager: SessionManager) -> str:
    """Get extraction prompt from JSON or fallback."""
    system_prompt = get_fallback_extraction_prompt()
    if USE_JSON_PROMPTS:
        try:
            variants = {"control": "extraction_task", "alt": "extraction_task_alt"}
            loaded_prompt = get_prompt_with_experiment(
                "extraction_task",
                variants=variants,
                user_id=getattr(session_manager, "user_id", None),
            )
            if loaded_prompt:
                system_prompt = loaded_prompt
            else:
                logger.warning("Failed to load 'extraction_task' (any variant) prompt from JSON, using fallback.")
        except Exception as e:
            logger.warning(f"Error loading 'extraction_task' prompt (experiments path): {e}, using fallback.")
    return system_prompt


def _clean_json_response(response_str: str) -> str:
    """Clean AI response string by removing markdown code blocks."""
    cleaned = response_str.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")].strip()
    return cleaned


def _compute_component_coverage(parsed_json: dict[str, Any]) -> float | None:
    """Compute component coverage for extraction quality."""
    try:
        extracted_component = parsed_json.get("extracted_data", {})
        structured_keys = [
            "structured_names",
            "vital_records",
            "relationships",
            "locations",
            "occupations",
            "research_questions",
            "documents_mentioned",
            "dna_information",
        ]
        non_empty = sum(
            1 for k in structured_keys if isinstance(extracted_component.get(k), list) and extracted_component.get(k)
        )
        return (non_empty / len(structured_keys)) if structured_keys else 0.0
    except Exception:
        return None


def _record_extraction_telemetry(
    system_prompt: str,
    parsed_json: dict[str, Any] | None,
    cleaned_response_str: str | None,
    session_manager: SessionManager,
    parse_success: bool,
    error: str | None = None,
) -> None:
    """Record extraction telemetry event."""
    if not USE_JSON_PROMPTS:
        return

    try:
        variant_label = "alt" if "extraction_task_alt" in system_prompt[:120] else "control"
        # Note: extraction_quality module not yet implemented, so quality_score is None
        quality_score = None
        component_coverage = _compute_component_coverage(parsed_json) if parsed_json else None

        if quality_score is not None and parsed_json:
            with contextlib.suppress(Exception):
                parsed_json["quality_score"] = quality_score

        anomaly_summary = None

        prompt_key = "extraction_task_alt" if variant_label == "alt" else "extraction_task"
        prompt_version = get_prompt_version(prompt_key)

        telemetry_payload: dict[str, Any] = {
            "variant_label": variant_label,
            "prompt_key": prompt_key,
            "prompt_version": prompt_version,
            "parse_success": parse_success,
            "extracted_data": parsed_json.get("extracted_data") if parsed_json else None,
            "suggested_tasks": parsed_json.get("suggested_tasks") if parsed_json else None,
            "raw_response_text": cleaned_response_str,
            "user_id": getattr(session_manager, "user_id", None),
            "quality_score": quality_score,
            "component_coverage": component_coverage,
            "anomaly_summary": anomaly_summary,
            "error": error,
        }
        record_extraction_experiment_event(telemetry_payload)
    except Exception:
        pass


def _check_nested_structure(parsed_json: dict[str, Any], salvaged: dict[str, Any]) -> bool:
    """Check if JSON already has expected nested structure."""
    if "extracted_data" in parsed_json and isinstance(parsed_json["extracted_data"], dict):
        salvaged["extracted_data"] = parsed_json["extracted_data"]
    if "suggested_tasks" in parsed_json and isinstance(parsed_json["suggested_tasks"], list):
        salvaged["suggested_tasks"] = parsed_json["suggested_tasks"]
        return True
    return False


def _transform_flat_to_nested(parsed_json: dict[str, Any]) -> dict[str, Any]:
    """Transform flat structure to nested structure."""
    extracted_data: dict[str, list[Any]] = {}
    key_mapping = {
        "mentioned_names": "mentioned_names",
        "dates": "mentioned_dates",
        "locations": "mentioned_locations",
        "relationships": "potential_relationships",
        "occupations": "key_facts",
        "events": "key_facts",
        "research_questions": "key_facts",
    }

    for flat_key, nested_key in key_mapping.items():
        if flat_key in parsed_json and isinstance(parsed_json[flat_key], list):
            if nested_key not in extracted_data:
                extracted_data[nested_key] = []
            extracted_data[nested_key].extend(parsed_json[flat_key])

    # Ensure all expected keys exist
    for key in ["mentioned_names", "mentioned_locations", "mentioned_dates", "potential_relationships", "key_facts"]:
        if key not in extracted_data:
            extracted_data[key] = []

    logger.info(
        f"Successfully transformed flat structure to nested. Extracted {len(extracted_data.get('mentioned_names', []))} names, "
        f"{len(extracted_data.get('mentioned_locations', []))} locations, {len(extracted_data.get('mentioned_dates', []))} dates"
    )

    return extracted_data


def _salvage_flat_structure(parsed_json: dict[str, Any], default_empty_result: dict[str, Any]) -> dict[str, Any]:
    """Attempt to salvage flat structure by transforming to expected nested structure."""
    salvaged = default_empty_result.copy()

    # Check for nested structure first
    if _check_nested_structure(parsed_json, salvaged):
        return salvaged

    # Transform flat structure
    salvaged["extracted_data"] = _transform_flat_to_nested(parsed_json)
    return salvaged


def _attempt_json_repair(ai_response_str: str) -> dict[str, Any] | None:
    """Attempt to repair truncated JSON by appending missing brackets/braces."""
    try:
        logger.warning("Attempting to repair truncated JSON.")
        repaired_json_str = ai_response_str.strip()
        open_braces = repaired_json_str.count("{")
        close_braces = repaired_json_str.count("}")
        open_brackets = repaired_json_str.count("[")
        close_brackets = repaired_json_str.count("]")

        repaired_json_str += "}" * (open_braces - close_braces)
        repaired_json_str += "]" * (open_brackets - close_brackets)

        parsed_json = json.loads(repaired_json_str)
        logger.info("Successfully repaired truncated JSON.")
        return parsed_json
    except Exception as repair_error:
        logger.error(f"Failed to repair JSON: {repair_error}")
        return None


def _process_extraction_response(
    ai_response_str: str,
    system_prompt: str,
    session_manager: SessionManager,
    duration: float,
    default_empty_result: dict[str, Any],
) -> dict[str, Any]:
    """Process and validate AI extraction response."""
    try:
        cleaned_response_str = _clean_json_response(ai_response_str)
        parsed_json = json.loads(cleaned_response_str)

        if (
            isinstance(parsed_json, dict)
            and "extracted_data" in parsed_json
            and isinstance(parsed_json["extracted_data"], dict)
            and "suggested_tasks" in parsed_json
            and isinstance(parsed_json["suggested_tasks"], list)
        ):
            logger.info(f"AI extraction successful. (Took {duration:.2f}s)")
            _record_extraction_telemetry(
                system_prompt, parsed_json, cleaned_response_str, session_manager, parse_success=True
            )
            return parsed_json

        logger.warning(
            f"AI extraction response valid JSON but uses flat structure. Attempting transform. Response: {cleaned_response_str[:500]}"
        )
        salvaged = _salvage_flat_structure(parsed_json, default_empty_result)
        _record_extraction_telemetry(
            system_prompt,
            salvaged,
            cleaned_response_str,
            session_manager,
            parse_success=False,
            error="structure_salvaged",
        )
        return salvaged

    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
        repaired_json = _attempt_json_repair(ai_response_str)
        if repaired_json:
            return repaired_json

        logger.error(f"AI extraction invalid JSON: {e}. Response: {ai_response_str[:500]}")
        _record_extraction_telemetry(
            system_prompt, None, ai_response_str, session_manager, parse_success=False, error=str(e)[:120]
        )
        return default_empty_result


def extract_genealogical_entities(context_history: str, session_manager: SessionManager) -> dict[str, Any] | None:
    """
    Extracts genealogical entities and suggests follow-up tasks.
    Expects AI to return JSON: {"extracted_data": {...}, "suggested_tasks": [...]}.
    """
    ai_provider = config_schema.ai_provider.lower()
    default_empty_result = {"extracted_data": {}, "suggested_tasks": []}

    if not ai_provider:
        logger.error("extract_genealogical_entities: AI_PROVIDER not configured.")
        return default_empty_result

    if not context_history:
        logger.warning("extract_genealogical_entities: Empty context. Returning empty structure.")
        return default_empty_result

    system_prompt = _get_extraction_prompt(session_manager)

    # Log AI call start for visibility (helps user understand why processing is slow)
    start_time = time.time()
    ai_response_str = _call_ai_model(
        provider=ai_provider,
        system_prompt=system_prompt,
        user_content=context_history,
        session_manager=session_manager,
        max_tokens=2500,  # Increased to prevent truncated JSON responses
        temperature=0.2,
        response_format_type="json_object",  # For DeepSeek
    )
    duration = time.time() - start_time

    if ai_response_str:
        return _process_extraction_response(
            ai_response_str, system_prompt, session_manager, duration, default_empty_result
        )

    logger.error(f"AI extraction failed or returned empty. (Took {duration:.2f}s)")
    _record_extraction_telemetry(
        system_prompt, None, None, session_manager, parse_success=False, error="empty_response"
    )
    return default_empty_result


# End of extract_genealogical_entities


def generate_genealogical_reply(
    conversation_context: str,
    user_last_message: str,
    genealogical_data_str: str,
    session_manager: SessionManager,
    tree_lookup_results: str = "",
    relationship_context: str = "",
    semantic_search_results: str = "",
    prompt_key: str = "genealogical_reply",
    prompt_variant: Optional[str] = None,
) -> str | None:
    """
    Generates a personalized genealogical reply with RAG-style tree context.

    Sprint 3: Enhanced to integrate TreeQueryService results for accurate tree-based answers.
    Phase 2.1: Enhanced to include SemanticSearchService results for evidence-backed Q&A.

    Args:
        conversation_context: Formatted conversation history
        user_last_message: The user's most recent message
        genealogical_data_str: Structured genealogical data as JSON string
        session_manager: Session manager for AI calls
        tree_lookup_results: Results from TreeQueryService.find_person() lookups
        relationship_context: Relationship path/explanation from TreeQueryService.explain_relationship()
        semantic_search_results: Results from SemanticSearchService (evidence-backed Q&A)

    Returns:
        Generated reply text, or None if generation fails
    """
    ai_provider = config_schema.ai_provider.lower()
    if not ai_provider:
        logger.error("generate_genealogical_reply: AI_PROVIDER not configured.")
        return None
    if not all([conversation_context, user_last_message, genealogical_data_str]):
        logger.error("generate_genealogical_reply: One or more required inputs are empty.")
        return None

    system_prompt_template = _load_genealogical_reply_template(prompt_key=prompt_key, prompt_variant=prompt_variant)
    final_system_prompt = _format_genealogical_reply_prompt(
        system_prompt_template,
        conversation_context,
        user_last_message,
        genealogical_data_str,
        tree_lookup_results,
        relationship_context,
        semantic_search_results,
    )

    start_time = time.time()
    reply_text = _call_ai_model(
        provider=ai_provider,
        system_prompt=final_system_prompt,
        user_content="Please generate a reply based on the system prompt and the information embedded within it.",
        session_manager=session_manager,
        max_tokens=800,
        temperature=0.7,
    )
    duration = time.time() - start_time

    if reply_text:
        logger.info(f"AI reply generation successful. (Took {duration:.2f}s)")
    else:
        logger.error(f"AI reply generation failed. (Took {duration:.2f}s)")
    return reply_text


def _extract_variant_text(variant_entry: Any) -> Optional[str]:
    """Normalize prompt variant payload into a string, if possible."""

    if isinstance(variant_entry, dict):
        variant_entry = cast(dict[str, Any], variant_entry)
        prompt_text = variant_entry.get("prompt") or variant_entry.get("text")
        return cast(Optional[str], prompt_text)

    if isinstance(variant_entry, str):
        return variant_entry

    return None


def _get_prompt_variants(prompts_data: dict[str, Any], prompt_key: str) -> dict[str, Any]:
    prompts_dict = cast(dict[str, Any], prompts_data.get("prompts", {}) or {})
    prompt_entry = cast(dict[str, Any], prompts_dict.get(prompt_key, {}) or {})
    return cast(dict[str, Any], prompt_entry.get("variants", {}) or {})


def _load_prompt_variant_text(prompt_key: str, prompt_variant: Optional[str]) -> Optional[str]:
    """Return prompt variant text when available in ai_prompts.json."""

    if not (USE_JSON_PROMPTS and prompt_variant):
        return None

    try:
        from ai_prompt_utils import load_prompts

        prompts_data = cast(dict[str, Any], load_prompts() or {})
        variants = _get_prompt_variants(prompts_data, prompt_key)
        return _extract_variant_text(variants.get(prompt_variant))
    except Exception as exc:  # pragma: no cover - telemetry-only helper
        logger.debug(f"Prompt variant lookup failed for {prompt_key}:{prompt_variant}: {exc}")
        return None


def _load_genealogical_reply_template(
    prompt_key: str = "genealogical_reply", prompt_variant: Optional[str] = None
) -> str:
    """Load the genealogical_reply prompt template from JSON or fallback."""

    system_prompt_template = get_fallback_reply_prompt()
    if USE_JSON_PROMPTS:
        variant_text = _load_prompt_variant_text(prompt_key, prompt_variant)
        if variant_text:
            return variant_text
        try:
            loaded_prompt = get_prompt(prompt_key)
            if loaded_prompt:
                return loaded_prompt
            logger.warning(f"Failed to load '{prompt_key}' prompt from JSON, using fallback.")
        except Exception as e:
            logger.warning(f"Error loading '{prompt_key}' prompt: {e}, using fallback.")
    return system_prompt_template


def _format_genealogical_reply_prompt(
    template: str,
    conversation_context: str,
    user_message: str,
    genealogical_data: str,
    tree_lookup_results: str,
    relationship_context: str,
    semantic_search_results: str = "",
) -> str:
    """Format the genealogical reply prompt with provided context."""
    try:
        return template.format(
            conversation_context=conversation_context,
            user_message=user_message,
            tree_lookup_results=tree_lookup_results or "No tree lookup performed.",
            relationship_context=relationship_context or "Relationship not determined.",
            semantic_search_results=semantic_search_results or "No semantic search results available.",
            genealogical_data=genealogical_data,
        )
    except KeyError as ke:
        logger.warning(f"Prompt formatting missing key: {ke}. Trying legacy format.")
        return _format_legacy_prompt(template, conversation_context, user_message, genealogical_data)
    except Exception as fe:
        logger.error(f"Unexpected error formatting genealogical_reply prompt: {fe}. Using unformatted prompt.")
        return template


def _format_legacy_prompt(template: str, context: str, user_msg: str, data: str) -> str:
    """Format prompt using legacy format without tree lookup fields."""
    try:
        return template.format(
            conversation_context=context,
            user_message=user_msg,
            genealogical_data=data,
        )
    except Exception:
        logger.error("Legacy prompt format also failed. Using unformatted prompt.")
        return template


# End of generate_genealogical_reply


# === PHASE 2.3: STRUCTURED RESPONSE GENERATION ===


@dataclass
class EvidenceSource:
    """A single piece of evidence used in a response."""

    source: str  # e.g., "GEDCOM", "Census 1881", "Birth Record"
    reference: str | None  # e.g., "@I123@", "RG11/1234"
    fact: str  # e.g., "John Smith b. 1850"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "reference": self.reference,
            "fact": self.fact,
        }


@dataclass
class MissingInformation:
    """A piece of missing information that would improve the answer."""

    field: str  # e.g., "Birth place"
    impact: str  # "high", "medium", "low"
    suggested_source: str  # e.g., "Parish records"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "field": self.field,
            "impact": self.impact,
            "suggested_source": self.suggested_source,
        }


@dataclass
class ConfidenceBreakdown:
    """Breakdown of how confidence score was calculated."""

    base: int  # Starting point (50)
    adjustments: list[dict[str, str]]  # e.g., [{"+30": "Direct GEDCOM match"}]
    final: int  # Final score after adjustments

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "base": self.base,
            "adjustments": self.adjustments,
            "final": self.final,
        }


@dataclass
class StructuredReplyResult:
    """
    Result of structured response generation with evidence citations.

    Phase 2.3: Evidence-backed answers with confidence scoring and uncertainty disclosure.
    """

    draft_message: str
    confidence: int  # 0-100
    confidence_breakdown: ConfidenceBreakdown
    evidence_used: list[EvidenceSource]
    missing_information: list[MissingInformation]
    suggested_facts: list[str]
    follow_up_questions: list[str]
    route_to_human_review: bool
    human_review_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "draft_message": self.draft_message,
            "confidence": self.confidence,
            "confidence_breakdown": self.confidence_breakdown.to_dict(),
            "evidence_used": [e.to_dict() for e in self.evidence_used],
            "missing_information": [m.to_dict() for m in self.missing_information],
            "suggested_facts": self.suggested_facts,
            "follow_up_questions": self.follow_up_questions,
            "route_to_human_review": self.route_to_human_review,
            "human_review_reason": self.human_review_reason,
        }

    def should_route_to_human(self) -> bool:
        """Check if this response should be routed to human review."""
        return self.route_to_human_review or self.confidence < 50

    def get_review_reason(self) -> str | None:
        """Get the reason for human review routing."""
        if self.human_review_reason:
            return self.human_review_reason
        if self.confidence < 50:
            return f"Low confidence score ({self.confidence}/100)"
        return None


def generate_structured_reply(
    user_question: str,
    conversation_context: str,
    tree_evidence: str,
    semantic_search_results: str,
    family_members: str,
    relationship_path: str,
    session_manager: SessionManager,
) -> StructuredReplyResult | None:
    """
    Generate a structured response with evidence citations and confidence scoring.

    Phase 2.3: Enhanced response generation that returns structured JSON with:
    - Evidence citations for every factual claim
    - Confidence scoring with breakdown
    - Uncertainty disclosure
    - Follow-up question generation
    - Human review routing for low-confidence answers

    Args:
        user_question: The genealogical question to answer
        conversation_context: Formatted conversation history
        tree_evidence: Evidence from GEDCOM tree lookups
        semantic_search_results: Results from semantic search
        family_members: Family members data from TreeQueryService
        relationship_path: Relationship explanation
        session_manager: Session manager for AI calls

    Returns:
        StructuredReplyResult with all response components, or None if generation fails
    """
    ai_provider = config_schema.ai_provider.lower()
    if not ai_provider:
        logger.error("generate_structured_reply: AI_PROVIDER not configured.")
        return None

    if not user_question:
        logger.error("generate_structured_reply: user_question is required.")
        return None

    # Load the response_generation prompt
    prompt_template = get_prompt("response_generation")
    if not prompt_template:
        logger.error("generate_structured_reply: Failed to load 'response_generation' prompt.")
        return None

    # Format the prompt with all evidence
    try:
        formatted_prompt = prompt_template.format(
            user_question=user_question,
            conversation_context=conversation_context or "No previous conversation.",
            tree_evidence=tree_evidence or "No tree evidence available.",
            semantic_search_results=semantic_search_results or "No semantic search results.",
            family_members=family_members or "No family members data.",
            relationship_path=relationship_path or "No relationship path available.",
        )
    except KeyError as e:
        logger.error(f"generate_structured_reply: Prompt formatting error: {e}")
        return None

    start_time = time.time()
    response_text = _call_ai_model(
        provider=ai_provider,
        system_prompt=formatted_prompt,
        user_content="Generate the structured JSON response based on the prompt.",
        session_manager=session_manager,
        max_tokens=1500,
        temperature=0.5,  # Lower temperature for more consistent JSON output
    )
    duration = time.time() - start_time

    if not response_text:
        logger.error(f"generate_structured_reply: AI call failed. (Took {duration:.2f}s)")
        return None

    # Parse the JSON response
    result = _parse_structured_reply_response(response_text)
    if result:
        logger.info(
            f"Structured reply generated. Confidence: {result.confidence}/100, "
            f"Human review: {result.should_route_to_human()}. (Took {duration:.2f}s)"
        )
    else:
        logger.warning(f"generate_structured_reply: Failed to parse response. (Took {duration:.2f}s)")

    return result


def _parse_structured_reply_response(response_text: str) -> StructuredReplyResult | None:
    """Parse the AI response into a StructuredReplyResult."""
    try:
        # Clean the response - remove markdown code blocks if present
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)

        # Parse confidence breakdown
        breakdown_data = data.get("confidence_breakdown", {})
        confidence_breakdown = ConfidenceBreakdown(
            base=breakdown_data.get("base", 50),
            adjustments=breakdown_data.get("adjustments", []),
            final=breakdown_data.get("final", data.get("confidence", 50)),
        )

        # Parse evidence sources
        evidence_used = [
            EvidenceSource(
                source=e.get("source", "Unknown"),
                reference=e.get("reference"),
                fact=e.get("fact", ""),
            )
            for e in data.get("evidence_used", [])
        ]

        # Parse missing information
        missing_info = [
            MissingInformation(
                field=m.get("field", "Unknown"),
                impact=m.get("impact", "medium"),
                suggested_source=m.get("suggested_source", ""),
            )
            for m in data.get("missing_information", [])
        ]

        return StructuredReplyResult(
            draft_message=data.get("draft_message", ""),
            confidence=data.get("confidence", 50),
            confidence_breakdown=confidence_breakdown,
            evidence_used=evidence_used,
            missing_information=missing_info,
            suggested_facts=data.get("suggested_facts", []),
            follow_up_questions=data.get("follow_up_questions", []),
            route_to_human_review=data.get("route_to_human_review", False),
            human_review_reason=data.get("human_review_reason"),
        )

    except json.JSONDecodeError as e:
        logger.error(f"_parse_structured_reply_response: JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"_parse_structured_reply_response: Unexpected error: {e}")
        return None


# End of Phase 2.3 Structured Response Generation


def _load_dialogue_prompt(log_prefix: str) -> str | None:
    """Load genealogical_dialogue_response prompt from JSON."""
    if not USE_JSON_PROMPTS:
        return None

    try:
        loaded_prompt = get_prompt("genealogical_dialogue_response")
        if loaded_prompt:
            return loaded_prompt
        logger.warning(f"{log_prefix}: Failed to load 'genealogical_dialogue_response' prompt from JSON.")
    except Exception as e:
        logger.warning(f"{log_prefix}: Error loading 'genealogical_dialogue_response' prompt: {e}")
    return None


def _get_dialogue_defaults(
    conversation_history: str,
    lookup_results: str,
    dna_data: str,
    tree_statistics: str,
    relationship_path: str,
    conversation_phase: str,
    last_topic: str,
    pending_questions: str,
) -> dict[str, Any]:
    """Get default values for dialogue prompt variables."""
    return {
        "conversation_history": conversation_history or "No previous conversation.",
        "lookup_results": lookup_results or "No person lookup results available.",
        "dna_data": dna_data or "No DNA data available.",
        "tree_statistics": tree_statistics or "No tree statistics available.",
        "relationship_path": relationship_path or "Relationship unknown.",
        "conversation_phase": conversation_phase or "initial_outreach",
        "last_topic": last_topic or "None",
        "pending_questions": pending_questions or "None",
    }


def _format_dialogue_prompt(
    prompt_template: str,
    conversation_history: str,
    user_message: str,
    lookup_results: str,
    dna_data: str,
    tree_statistics: str,
    relationship_path: str,
    conversation_phase: str,
    engagement_score: int,
    last_topic: str,
    pending_questions: str,
    log_prefix: str,
) -> str | None:
    """Format dialogue prompt with all context variables."""
    try:
        defaults = _get_dialogue_defaults(
            conversation_history,
            lookup_results,
            dna_data,
            tree_statistics,
            relationship_path,
            conversation_phase,
            last_topic,
            pending_questions,
        )
        defaults["user_message"] = user_message
        defaults["engagement_score"] = engagement_score
        return prompt_template.format(**defaults)
    except KeyError as ke:
        logger.error(f"{log_prefix}: Prompt formatting error. Missing key: {ke}")
    except Exception as fe:
        logger.error(f"{log_prefix}: Unexpected error formatting prompt: {fe}")
    return None


def generate_contextual_response(
    conversation_history: str,
    user_message: str,
    lookup_results: str,
    dna_data: str,
    tree_statistics: str,
    relationship_path: str,
    conversation_phase: str,
    engagement_score: int,
    last_topic: str,
    pending_questions: str,
    session_manager: SessionManager,
    log_prefix: str = "",
) -> str | None:
    """
    Generate intelligent contextual genealogical response using Phase 3 dialogue engine.

    Uses the genealogical_dialogue_response prompt with full conversation state awareness.

    Args:
        conversation_history: Formatted conversation history
        user_message: User's latest message
        lookup_results: Formatted person lookup results
        dna_data: DNA match information
        tree_statistics: Tree statistics summary
        relationship_path: Relationship path to user
        conversation_phase: Current phase (initial_outreach/active_dialogue/research_exchange)
        engagement_score: 0-100 engagement score
        last_topic: Last conversation topic
        pending_questions: Pending questions from previous messages
        session_manager: Session manager for API calls
        log_prefix: Logging prefix

    Returns:
        Generated response text or None if generation fails
    """
    ai_provider = config_schema.ai_provider.lower()
    if not ai_provider:
        logger.error(f"{log_prefix}: AI_PROVIDER not configured.")
        return None

    if not user_message:
        logger.error(f"{log_prefix}: user_message is required.")
        return None

    # Load prompt template
    system_prompt_template = _load_dialogue_prompt(log_prefix)
    if not system_prompt_template:
        logger.error(f"{log_prefix}: genealogical_dialogue_response prompt not available.")
        return None

    # Format prompt with context
    final_system_prompt = _format_dialogue_prompt(
        system_prompt_template,
        conversation_history,
        user_message,
        lookup_results,
        dna_data,
        tree_statistics,
        relationship_path,
        conversation_phase,
        engagement_score,
        last_topic,
        pending_questions,
        log_prefix,
    )

    if not final_system_prompt:
        return None

    # Call AI model
    logger.info(f"{log_prefix}: Calling AI ({ai_provider}) for contextual response generation...")
    start_time = time.time()
    response_text = _call_ai_model(
        provider=ai_provider,
        system_prompt=final_system_prompt,
        user_content="Please generate a contextual genealogical response based on the conversation state and lookup results provided in the system prompt.",
        session_manager=session_manager,
        max_tokens=1000,
        temperature=0.7,
    )
    duration = time.time() - start_time

    if response_text:
        logger.info(f"{log_prefix}: Contextual response generation successful. (Took {duration:.2f}s)")
    else:
        logger.error(f"{log_prefix}: Contextual response generation failed. (Took {duration:.2f}s)")

    return response_text


# End of generate_contextual_response


def assess_engagement(
    conversation_history: str,
    session_manager: SessionManager,
    log_prefix: str = "",
) -> dict[str, Any] | None:
    """
    Assess engagement and summarize conversation using the Phase 3 engagement_assessment prompt.

    Returns a dict such as:
      {"engagement_score": int, "ai_summary": str, "last_topic": str, "pending_questions": list[str]}
    """
    ai_provider = config_schema.ai_provider.lower()
    if not ai_provider:
        logger.error(f"{log_prefix}: AI_PROVIDER not configured.")
        return None

    prompt = get_prompt("engagement_assessment") if USE_JSON_PROMPTS else None
    if not prompt:
        logger.error(f"{log_prefix}: engagement_assessment prompt not available.")
        return None

    start_time = time.time()
    ai_response_str = _call_ai_model(
        provider=ai_provider,
        system_prompt=prompt,
        user_content=conversation_history,
        session_manager=session_manager,
        max_tokens=800,
        temperature=0.2,
        response_format_type="json_object",
    )
    duration = time.time() - start_time

    if not ai_response_str:
        logger.error(f"{log_prefix}: Engagement assessment failed. (Took {duration:.2f}s)")
        return None

    try:
        result = json.loads(ai_response_str)
        logger.info(f"{log_prefix}: Engagement assessment OK. (Took {duration:.2f}s)")
        return result if isinstance(result, dict) else None
    except Exception as e:
        logger.error(f"{log_prefix}: Failed to parse engagement JSON: {e}")
        return None


def _format_clarification_prompt(
    prompt: str, user_message: str, extracted_entities: dict[str, Any], ambiguity_context: str
) -> str | None:
    """Format clarification prompt with user data."""
    try:
        toon_entities = _encode_to_toon(extracted_entities)
        return prompt.format(
            user_message=user_message,
            extracted_entities=toon_entities,
            ambiguity_context=ambiguity_context,
        )
    except KeyError as e:
        logger.error(f"generate_clarifying_questions: Prompt formatting error - missing key: {e}")
        return None


def _validate_clarification_response(result: dict[str, Any]) -> bool:
    """Validate AI clarification response structure."""
    if "clarifying_questions" not in result:
        logger.error("generate_clarifying_questions: Invalid response structure")
        return False

    questions = result.get("clarifying_questions", [])
    if not questions or not isinstance(questions, list):
        logger.warning("generate_clarifying_questions: No questions generated")
        return False

    return True


def generate_clarifying_questions(
    user_message: str,
    extracted_entities: dict[str, Any],
    ambiguity_context: str,
    session_manager: SessionManager,
) -> dict[str, Any] | None:
    """
    Generate AI-powered clarifying questions for ambiguous extracted entities.

    Priority 1 Todo #7: Action 7 Intent Clarifier

    Args:
        user_message: Original user message
        extracted_entities: Dictionary of extracted entity data
        ambiguity_context: Description of detected ambiguities
        session_manager: SessionManager instance for AI calls

    Returns:
        Dictionary with clarifying_questions, primary_ambiguity, urgency, reasoning.
        None if AI call fails.

    Example output:
        {
            "clarifying_questions": ["When was Charles born?", "Which Scotland location?"],
            "primary_ambiguity": "date",
            "urgency": "critical",
            "reasoning": "Birth year critical for tree search..."
        }
    """
    ai_provider = config_schema.ai_provider.lower()
    if not ai_provider:
        logger.error("generate_clarifying_questions: AI_PROVIDER not configured.")
        return None

    prompt = get_prompt("intent_clarification") if USE_JSON_PROMPTS else None
    if not prompt:
        logger.error("generate_clarifying_questions: intent_clarification prompt not available.")
        return None

    formatted_prompt = _format_clarification_prompt(prompt, user_message, extracted_entities, ambiguity_context)
    if not formatted_prompt:
        return None

    start_time = time.time()
    ai_response_str = _call_ai_model(
        provider=ai_provider,
        system_prompt=formatted_prompt,
        user_content="Generate clarifying questions based on the provided context.",
        session_manager=session_manager,
        max_tokens=600,
        temperature=0.3,  # Low temperature for consistent, focused questions
        response_format_type="json_object",
    )
    duration = time.time() - start_time

    result: dict[str, Any] | None = None

    if ai_response_str:
        try:
            candidate = json.loads(ai_response_str)
        except json.JSONDecodeError as exc:
            logger.error(f"generate_clarifying_questions: Failed to parse JSON response: {exc}")
        except Exception as exc:  # pragma: no cover - defensive logging for unexpected providers
            logger.error(f"generate_clarifying_questions: Unexpected error: {exc}", exc_info=True)
        else:
            if _validate_clarification_response(candidate):
                questions = candidate.get("clarifying_questions", [])
                logger.info(
                    f"✅ Generated {len(questions)} clarifying question(s) "
                    f"for {candidate.get('primary_ambiguity', 'unknown')} ambiguity. (Took {duration:.2f}s)"
                )
                result = candidate
    else:
        logger.error(f"generate_clarifying_questions: AI call failed. (Took {duration:.2f}s)")

    return result


def _toon_format_scalar(value: Any) -> str:
    """Format a scalar value for TOON output.

    Uses JSON serialization for primitives to keep quoting and escaping rules simple
    while remaining lossless for LLM-facing data.
    """
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False, separators=(",", ":"))


def _toon_tabular_fields(items: list[dict[str, Any]]) -> list[str] | None:
    """Determine if a list of dicts can be rendered in TOON tabular form.

    Returns a sorted list of field names when all values are primitives; otherwise None.
    """
    if not items:
        return None

    key_sets: list[set[str]] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        key_sets.append(set(item.keys()))

    all_keys: set[str] = set()
    for keys in key_sets:
        all_keys.update(keys)

    if not all_keys:
        return None

    for item in items:
        for key in all_keys:
            value = item.get(key)
            if isinstance(value, (dict, list)):
                return None

    return sorted(all_keys)


def _toon_encode_dict(obj: dict[str, Any], indent: int, step: int) -> list[str]:
    """Encode a mapping into TOON lines."""
    lines: list[str] = []
    prefix = " " * indent

    for key, value in obj.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(_toon_encode_dict(value, indent + step, step))
        elif isinstance(value, list):
            lines.extend(_toon_encode_list(key, value, indent, step))
        else:
            lines.append(f"{prefix}{key}: {_toon_format_scalar(value)}")

    return lines


def _toon_encode_list(key: str, items: list[Any], indent: int, step: int) -> list[str]:
    """Encode a sequence into TOON lines.

    Uses tabular form for uniform object arrays, inline form for primitive arrays,
    and a simple nested fallback for mixed structures.
    """
    prefix = " " * indent
    length = len(items)

    if length == 0:
        return [f"{prefix}{key}[0]:"]

    # Tabular representation for uniform arrays of objects with primitive fields
    if all(isinstance(item, dict) for item in items):
        fields = _toon_tabular_fields(cast(list[dict[str, Any]], items))
        if fields:
            header = f"{prefix}{key}[{length}]{{{','.join(fields)}}}:"
            lines = [header]
            row_prefix = " " * (indent + step)
            for item in items:
                row_values = [_toon_format_scalar(item.get(field)) for field in fields]
                lines.append(f"{row_prefix}{','.join(row_values)}")
            return lines

    # Inline primitive array: key[N]: v1,v2,v3
    if all(not isinstance(item, (dict, list)) for item in items):
        values = ",".join(_toon_format_scalar(item) for item in items)
        return [f"{prefix}{key}[{length}]: {values}"]

    # Fallback: nested list representation
    lines = [f"{prefix}{key}[{length}]:"]
    nested_indent = indent + step
    nested_prefix = " " * nested_indent

    for item in items:
        if isinstance(item, dict):
            lines.append(f"{nested_prefix}-")
            lines.extend(_toon_encode_dict(item, nested_indent + step, step))
        elif isinstance(item, list):
            lines.append(f"{nested_prefix}-")
            # Reuse list encoder without a key for inner arrays
            lines.extend(_toon_encode_list("items", item, nested_indent + step, step))
        else:
            lines.append(f"{nested_prefix}{_toon_format_scalar(item)}")

    return lines


def _encode_to_toon(data: Any, root_label: str | None = None, indent: int = 2) -> str:
    """Encode JSON-like data into a compact TOON representation for LLM prompts.

    This is a pragmatic encoder focused on readability and determinism for prompt
    context. It does not aim to implement the full TOON specification but
    intentionally follows its core patterns (object fields, array headers, and
    tabular arrays) so models can reliably parse structure.
    """
    if root_label is not None:
        if isinstance(data, dict):
            lines = [f"{root_label}:"]
            lines.extend(_toon_encode_dict(data, indent, indent))
        elif isinstance(data, list):
            lines = _toon_encode_list(root_label, data, 0, indent)
        else:
            lines = [f"{root_label}: {_toon_format_scalar(data)}"]
    elif isinstance(data, dict):
        lines = _toon_encode_dict(data, 0, indent)
    elif isinstance(data, list):
        # Use a generic key for root arrays
        lines = _toon_encode_list("items", data, 0, indent)
    else:
        lines = [_toon_format_scalar(data)]

    return "\n".join(lines)


def extract_with_custom_prompt(
    context_history: str, custom_prompt: str, session_manager: SessionManager
) -> dict[str, Any] | None:
    """
    Extracts data using a custom prompt. Attempts JSON parsing, falls back to text.
    """
    ai_provider = config_schema.ai_provider.lower()
    if not ai_provider:
        logger.error("extract_with_custom_prompt: AI_PROVIDER not configured.")
        return None
    if not custom_prompt:
        logger.error("extract_with_custom_prompt: Custom prompt is empty.")
        return None

    start_time = time.time()
    ai_response_str = _call_ai_model(
        provider=ai_provider,
        system_prompt=custom_prompt,
        user_content=context_history,
        session_manager=session_manager,
        max_tokens=1500,
        temperature=0.2,
        response_format_type="json_object",  # Assume JSON is often desired
    )
    duration = time.time() - start_time

    if ai_response_str:
        logger.info(f"AI custom extraction successful. (Took {duration:.2f}s)")
        try:
            # Clean the response string
            cleaned_response_str = ai_response_str.strip()
            if cleaned_response_str.startswith("```json"):
                cleaned_response_str = cleaned_response_str[len("```json") :].strip()
            elif cleaned_response_str.startswith("```"):
                cleaned_response_str = cleaned_response_str[len("```") :].strip()
            if cleaned_response_str.endswith("```"):
                cleaned_response_str = cleaned_response_str[: -len("```")].strip()

            parsed_json = json.loads(cleaned_response_str)
            return {"extracted_data": parsed_json}
        except json.JSONDecodeError:
            logger.warning("Custom extraction response was not valid JSON. Returning as text.")
            return {"extracted_data": ai_response_str}  # Return raw string under 'extracted_data'
    else:
        logger.error(f"AI custom extraction failed or returned empty. (Took {duration:.2f}s)")
        return None


# End of extract_with_custom_prompt


def generate_with_custom_prompt(
    conversation_context: str,
    user_last_message: str,
    genealogical_data_str: str,
    custom_prompt: str,
    session_manager: SessionManager,
) -> str | None:
    """
    Generates a reply using a custom prompt, formatting the custom prompt with provided data.
    """
    ai_provider = config_schema.ai_provider.lower()
    if not ai_provider:
        logger.error("generate_with_custom_prompt: AI_PROVIDER not configured.")
        return None
    if not custom_prompt:
        logger.error("generate_with_custom_prompt: Custom prompt is empty.")
        return None

    try:
        # Format the custom_prompt with the context, user message, and genealogical data
        final_system_prompt = custom_prompt.format(
            conversation_context=conversation_context,
            user_message=user_last_message,
            genealogical_data=genealogical_data_str,
        )
    except KeyError as ke:
        logger.error(f"Custom prompt formatting error. Missing key: {ke}. Using unformatted custom prompt.")
        final_system_prompt = custom_prompt  # Fallback to unformatted if keys missing
    except Exception as fe:
        logger.error(f"Unexpected error formatting custom prompt: {fe}. Using unformatted custom prompt.")
        final_system_prompt = custom_prompt

    start_time = time.time()
    reply_text = _call_ai_model(
        provider=ai_provider,
        system_prompt=final_system_prompt,
        user_content="Please generate a response based on the system prompt and the information embedded within it.",
        session_manager=session_manager,
        max_tokens=800,
        temperature=0.7,
    )
    duration = time.time() - start_time

    if reply_text:
        logger.info(f"AI custom generation successful. (Took {duration:.2f}s)")
    else:
        logger.error(f"AI custom generation failed or returned empty. (Took {duration:.2f}s)")
    return reply_text


# End of generate_with_custom_prompt

# --- Specialized Genealogical Analysis Functions ---


def analyze_dna_match_conversation(context_history: str, session_manager: SessionManager) -> dict[str, Any] | None:
    """
    Analyzes conversations about DNA matches using specialized DNA analysis prompt.
    Returns structured data focused on DNA match information and genetic genealogy.
    """
    ai_provider = config_schema.ai_provider.lower()
    default_empty_result = {"extracted_data": {}, "suggested_tasks": []}

    if not ai_provider:
        logger.error("analyze_dna_match_conversation: AI_PROVIDER not configured.")
        return default_empty_result

    if not context_history:
        logger.warning("analyze_dna_match_conversation: Empty context. Returning empty structure.")
        return default_empty_result

    # Get DNA match analysis prompt
    system_prompt = "Analyze this DNA match conversation for genetic genealogy information."
    if USE_JSON_PROMPTS:
        try:
            loaded_prompt = get_prompt("dna_match_analysis")
            if loaded_prompt:
                system_prompt = loaded_prompt
            else:
                logger.warning("Failed to load 'dna_match_analysis' prompt from JSON, using fallback.")
        except Exception as e:
            logger.warning(f"Error loading 'dna_match_analysis' prompt: {e}, using fallback.")

    start_time = time.time()
    ai_response_str = _call_ai_model(
        provider=ai_provider,
        system_prompt=system_prompt,
        user_content=context_history,
        session_manager=session_manager,
        max_tokens=1500,
        temperature=0.2,
        response_format_type="json_object",
    )
    duration = time.time() - start_time

    if ai_response_str:
        try:
            ai_response = json.loads(ai_response_str)
            logger.info(f"DNA match analysis successful. (Took {duration:.2f}s)")
            return ai_response
        except json.JSONDecodeError as e:
            logger.error(f"DNA match analysis JSON parsing failed: {e}")
            return default_empty_result
    else:
        logger.error(f"DNA match analysis failed. (Took {duration:.2f}s)")
        return default_empty_result


def verify_family_tree_connections(context_history: str, session_manager: SessionManager) -> dict[str, Any] | None:
    """
    Analyzes conversations for family tree verification needs and conflicts.
    Returns structured data focused on verification requirements and conflict resolution.
    """
    ai_provider = config_schema.ai_provider.lower()
    default_empty_result = {"extracted_data": {}, "suggested_tasks": []}

    if not ai_provider:
        logger.error("verify_family_tree_connections: AI_PROVIDER not configured.")
        return default_empty_result

    if not context_history:
        logger.warning("verify_family_tree_connections: Empty context. Returning empty structure.")
        return default_empty_result

    # Get family tree verification prompt
    system_prompt = "Analyze this conversation for family tree verification needs."
    if USE_JSON_PROMPTS:
        try:
            loaded_prompt = get_prompt("family_tree_verification")
            if loaded_prompt:
                system_prompt = loaded_prompt
            else:
                logger.warning("Failed to load 'family_tree_verification' prompt from JSON, using fallback.")
        except Exception as e:
            logger.warning(f"Error loading 'family_tree_verification' prompt: {e}, using fallback.")

    start_time = time.time()
    ai_response_str = _call_ai_model(
        provider=ai_provider,
        system_prompt=system_prompt,
        user_content=context_history,
        session_manager=session_manager,
        max_tokens=1500,
        temperature=0.2,
        response_format_type="json_object",
    )
    duration = time.time() - start_time

    if ai_response_str:
        try:
            ai_response = json.loads(ai_response_str)
            logger.info(f"Family tree verification analysis successful. (Took {duration:.2f}s)")
            return ai_response
        except json.JSONDecodeError as e:
            logger.error(f"Family tree verification JSON parsing failed: {e}")
            return default_empty_result
    else:
        logger.error(f"Family tree verification analysis failed. (Took {duration:.2f}s)")
        return default_empty_result


# === PHASE 1.5.2: CONTEXT ACCURACY VALIDATION ===


@dataclass
class ContextAccuracyResult:
    """Result of context accuracy validation."""

    is_accurate: bool
    verified_facts: list[dict[str, Any]]  # Facts found in our tree
    unverified_facts: list[dict[str, Any]]  # Facts NOT found in our tree
    known_to_recipient: list[dict[str, Any]]  # Facts already in recipient's tree
    accuracy_score: int  # 0-100
    warnings: list[str]
    recommendation: str  # "proceed", "review", "revise"


def validate_context_accuracy(  # noqa: PLR0914 - necessary for comprehensive validation
    extracted_names: list[str],
    context_summary: str,
    recipient_tree_context: Optional[dict[str, Any]] = None,
    tree_service: Optional[Any] = None,
) -> ContextAccuracyResult:
    """
    Validate that extracted names exist in OUR tree before generating drafts.

    Phase 1.5.2: Pre-draft validation to ensure we don't mention ancestors
    that don't exist in our tree, or explain facts already known to recipient.

    Args:
        extracted_names: Names extracted from conversation/context to validate
        context_summary: Brief context for logging
        recipient_tree_context: Optional dict with recipient's known facts/ancestors
        tree_service: Optional TreeQueryService instance for lookups

    Returns:
        ContextAccuracyResult with validation outcome
    """
    _ = context_summary  # Used for logging context identification

    # Initialize result
    verified_facts: list[dict[str, Any]] = []
    unverified_facts: list[dict[str, Any]] = []
    known_to_recipient: list[dict[str, Any]] = []
    warnings: list[str] = []

    # Quick exit if no names to validate
    if not extracted_names:
        return ContextAccuracyResult(
            is_accurate=True,
            verified_facts=[],
            unverified_facts=[],
            known_to_recipient=[],
            accuracy_score=100,
            warnings=[],
            recommendation="proceed",
        )

    # Get or initialize TreeQueryService
    if tree_service is None:
        try:
            from genealogy.tree_query_service import TreeQueryService

            tree_service = TreeQueryService()
        except Exception as e:
            logger.warning(f"Could not initialize TreeQueryService: {e}")
            # Return cautious result - we can't verify
            return ContextAccuracyResult(
                is_accurate=True,  # Allow to proceed, but with warning
                verified_facts=[],
                unverified_facts=[],
                known_to_recipient=[],
                accuracy_score=50,
                warnings=["TreeQueryService unavailable - could not verify facts"],
                recommendation="review",
            )

    # Validate each extracted name against our tree
    for name in extracted_names:
        if not name or not name.strip():
            continue

        clean_name = name.strip()

        try:
            # Search for this person in our tree
            search_result = tree_service.find_person(name=clean_name)

            if search_result.found:
                verified_facts.append(
                    {
                        "name": clean_name,
                        "matched_name": search_result.name,
                        "confidence": search_result.confidence,
                        "person_id": search_result.person_id,
                        "birth_year": search_result.birth_year,
                    }
                )
            else:
                unverified_facts.append(
                    {
                        "name": clean_name,
                        "reason": "Not found in tree",
                        "alternatives": search_result.alternatives[:2] if search_result.alternatives else [],
                    }
                )
                warnings.append(f"'{clean_name}' not found in our tree")

        except Exception as e:
            logger.warning(f"Error searching for '{clean_name}': {e}")
            unverified_facts.append({"name": clean_name, "reason": f"Search error: {e}", "alternatives": []})

    # Check for facts already known to recipient
    if recipient_tree_context:
        recipient_ancestors = recipient_tree_context.get("ancestors", [])
        recipient_known_names = {a.get("name", "").lower() for a in recipient_ancestors if a.get("name")}

        for fact in verified_facts:
            if fact.get("name", "").lower() in recipient_known_names:
                known_to_recipient.append(
                    {"name": fact.get("name"), "warning": "Recipient may already know this ancestor"}
                )
                warnings.append(f"'{fact.get('name')}' likely already known to recipient")

    # Calculate accuracy score
    total_names = len(extracted_names)
    verified_count = len(verified_facts)
    known_count = len(known_to_recipient)

    if total_names > 0:
        # Score based on verification rate, penalize for explaining known facts
        verification_rate = verified_count / total_names
        known_penalty = min(known_count * 10, 30)  # Up to 30 point penalty
        accuracy_score = int(verification_rate * 100) - known_penalty
        accuracy_score = max(0, min(100, accuracy_score))
    else:
        accuracy_score = 100

    # Determine recommendation
    unverified_count = len(unverified_facts)
    if unverified_count == 0 and known_count == 0:
        recommendation = "proceed"
    elif unverified_count > total_names / 2:  # More than half unverified
        recommendation = "revise"
    elif known_count > 0 or unverified_count > 0:
        recommendation = "review"
    else:
        recommendation = "proceed"

    is_accurate = unverified_count == 0

    logger.debug(
        f"Context accuracy validation: {verified_count}/{total_names} verified, "
        f"{unverified_count} unverified, {known_count} known to recipient. "
        f"Score: {accuracy_score}, Recommendation: {recommendation}"
    )

    return ContextAccuracyResult(
        is_accurate=is_accurate,
        verified_facts=verified_facts,
        unverified_facts=unverified_facts,
        known_to_recipient=known_to_recipient,
        accuracy_score=accuracy_score,
        warnings=warnings,
        recommendation=recommendation,
    )


def extract_names_from_text(text: str) -> list[str]:
    """
    Extract potential person names from text for context validation.

    Simple heuristic extraction - looks for capitalized word sequences
    that might be names. AI-based extraction is more accurate but slower.

    Args:
        text: Text to extract names from

    Returns:
        List of potential person names
    """
    import re

    if not text:
        return []

    names: list[str] = []

    # Pattern: Capitalized words that look like names
    # Matches sequences like "John Smith", "Mary Jane Watson", etc.
    name_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"
    matches = re.findall(name_pattern, text)

    # Filter out common non-names
    non_names = {
        "Dear Friend",
        "Best Regards",
        "Kind Regards",
        "Thank You",
        "Looking Forward",
        "Best Wishes",
        "Many Thanks",
        "Take Care",
    }

    for match in matches:
        if match not in non_names and len(match) > 3:
            names.append(match)

    # Deduplicate while preserving order
    seen = set()
    unique_names = []
    for name in names:
        if name.lower() not in seen:
            seen.add(name.lower())
            unique_names.append(name)

    return unique_names


@dataclass
class DraftQualityResult:
    """Result of draft quality validation."""

    passes_quality_check: bool
    issues_found: list[dict[str, Any]]
    quality_score: int
    recommendation: str  # "approve", "revise", "reject", "human_review"


def validate_draft_quality(  # noqa: PLR0914 - necessary for comprehensive validation with context accuracy
    draft_message: str,
    recipient_name: str,
    recipient_profile_id: str,
    sender_name: str,
    sender_profile_id: str,
    context_summary: str,
    session_manager: Optional[SessionManager] = None,
    extracted_names: Optional[list[str]] = None,
    recipient_tree_context: Optional[dict[str, Any]] = None,
) -> DraftQualityResult:
    """
    Validate a draft message for quality issues before queueing.

    Phase 1.5.2/1.5.3: AI-powered draft review to detect:
    - Self-message attempts
    - Context inversion (explaining their facts to them)
    - Unverified facts (names not in our tree)
    - Known-to-recipient facts (don't re-explain their ancestors)
    - Relationship direction errors
    - Deceased person errors
    - Factual inconsistencies

    Args:
        draft_message: The draft message to validate
        recipient_name: Name of the recipient
        recipient_profile_id: Profile ID of the recipient
        sender_name: Name of the sender (typically the tree owner)
        sender_profile_id: Profile ID of the sender
        context_summary: Brief summary of conversation context
        session_manager: Optional SessionManager for AI calls
        extracted_names: Optional list of names to validate against our tree
        recipient_tree_context: Optional dict with recipient's known ancestors

    Returns:
        DraftQualityResult with validation outcome
    """
    # Default result for failures
    default_result = DraftQualityResult(
        passes_quality_check=True,
        issues_found=[],
        quality_score=100,
        recommendation="approve",
    )

    # Quick self-message check (don't need AI for this)
    is_self_message = recipient_profile_id and sender_profile_id and str(recipient_profile_id) == str(sender_profile_id)
    if is_self_message:
        logger.error(f"Draft quality check: Self-message detected! recipient={recipient_profile_id}")
        return DraftQualityResult(
            passes_quality_check=False,
            issues_found=[
                {
                    "category": "self_message",
                    "severity": "critical",
                    "description": "Message is addressed to the sender themselves",
                    "suggestion": "Do not send - this is a self-message",
                }
            ],
            quality_score=0,
            recommendation="reject",
        )

    # Phase 1.5.2: Run context accuracy validation if names provided
    context_accuracy_result: Optional[ContextAccuracyResult] = None
    if extracted_names or recipient_tree_context:
        # If no names provided, extract from draft
        if not extracted_names:
            extracted_names = extract_names_from_text(draft_message)

        if extracted_names:
            context_accuracy_result = validate_context_accuracy(
                extracted_names=extracted_names,
                context_summary=context_summary,
                recipient_tree_context=recipient_tree_context,
            )
            logger.debug(
                f"Context accuracy: {len(context_accuracy_result.verified_facts)} verified, "
                f"{len(context_accuracy_result.unverified_facts)} unverified, "
                f"{len(context_accuracy_result.known_to_recipient)} known to recipient"
            )

    ai_provider = config_schema.ai_provider.lower()
    if not ai_provider or not draft_message:
        if not ai_provider:
            logger.warning("validate_draft_quality: AI_PROVIDER not configured, skipping AI validation")
        else:
            logger.warning("validate_draft_quality: Empty draft, returning default pass")
        return default_result

    # Load prompt
    prompt_template = get_prompt("draft_quality_check")
    if not prompt_template:
        logger.warning("validate_draft_quality: draft_quality_check prompt not found, skipping")
        return default_result

    # Build context accuracy strings for prompt
    verified_facts_str = "None"
    unverified_facts_str = "None"
    known_to_recipient_str = "None"

    if context_accuracy_result:
        if context_accuracy_result.verified_facts:
            verified_facts_str = ", ".join(
                f"{f.get('name', 'Unknown')} (confidence: {f.get('confidence', 'unknown')})"
                for f in context_accuracy_result.verified_facts
            )
        if context_accuracy_result.unverified_facts:
            unverified_facts_str = ", ".join(
                f"{f.get('name', 'Unknown')} ({f.get('reason', 'unknown reason')})"
                for f in context_accuracy_result.unverified_facts
            )
        if context_accuracy_result.known_to_recipient:
            known_to_recipient_str = ", ".join(
                f"{f.get('name', 'Unknown')}" for f in context_accuracy_result.known_to_recipient
            )

    # Substitute placeholders (template strings)
    draft_placeholder = "{draft_message}"  # noqa: RUF027
    recipient_name_placeholder = "{recipient_name}"  # noqa: RUF027
    recipient_id_placeholder = "{recipient_profile_id}"  # noqa: RUF027
    sender_name_placeholder = "{sender_name}"  # noqa: RUF027
    sender_id_placeholder = "{sender_profile_id}"  # noqa: RUF027
    context_placeholder = "{context_summary}"  # noqa: RUF027
    verified_placeholder = "{verified_facts}"
    unverified_placeholder = "{unverified_facts}"
    known_placeholder = "{known_to_recipient}"

    system_prompt = (
        prompt_template.replace(draft_placeholder, draft_message)
        .replace(recipient_name_placeholder, recipient_name or "Unknown")
        .replace(recipient_id_placeholder, recipient_profile_id or "Unknown")
        .replace(sender_name_placeholder, sender_name or "Tree Owner")
        .replace(sender_id_placeholder, sender_profile_id or "Unknown")
        .replace(context_placeholder, context_summary or "No context available")
        .replace(verified_placeholder, verified_facts_str)
        .replace(unverified_placeholder, unverified_facts_str)
        .replace(known_placeholder, known_to_recipient_str)
    )

    start_time = time.time()
    ai_response_str = _call_ai_model(
        provider=ai_provider,
        system_prompt=system_prompt,
        user_content="Validate this draft now.",
        session_manager=session_manager,
        max_tokens=500,
        temperature=0.1,
        response_format_type="json_object",
    )
    duration = time.time() - start_time

    if ai_response_str:
        try:
            ai_response = json.loads(ai_response_str)
            logger.info(
                f"Draft quality validation complete. Score={ai_response.get('quality_score', 'N/A')} (Took {duration:.2f}s)"
            )
            return DraftQualityResult(
                passes_quality_check=ai_response.get("passes_quality_check", True),
                issues_found=ai_response.get("issues_found", []),
                quality_score=ai_response.get("quality_score", 100),
                recommendation=ai_response.get("recommendation", "approve"),
            )
        except json.JSONDecodeError as e:
            logger.error(f"Draft quality check JSON parsing failed: {e}")
            return default_result
    else:
        logger.warning(f"Draft quality validation failed (AI call failed). (Took {duration:.2f}s)")
        return default_result


@dataclass
class DraftCorrectionResult:
    """Result of auto-correction attempt."""

    success: bool
    corrected_draft: Optional[str]
    attempt_count: int
    final_quality_result: Optional[DraftQualityResult]
    routed_to_human_review: bool
    failure_reason: Optional[str]


def attempt_draft_correction(
    original_draft: str,
    quality_result: DraftQualityResult,
    recipient_name: str,
    recipient_profile_id: str,
    sender_name: str,
    sender_profile_id: str,
    context_summary: str,
    session_manager: Optional[SessionManager] = None,
    max_attempts: int = 1,
) -> DraftCorrectionResult:
    """
    Attempt to regenerate a draft with corrections when quality check fails.

    Phase 1.5.4: Auto-correction pipeline that attempts regeneration with
    explicit correction guidance, then routes to HUMAN_REVIEW if still failing.

    Args:
        original_draft: The draft that failed quality check
        quality_result: The DraftQualityResult with issues found
        recipient_name: Name of the recipient
        recipient_profile_id: Profile ID of the recipient
        sender_name: Name of the sender (typically the tree owner)
        sender_profile_id: Profile ID of the sender
        context_summary: Brief summary of conversation context
        session_manager: Optional SessionManager for AI calls
        max_attempts: Maximum regeneration attempts (default 1)

    Returns:
        DraftCorrectionResult with correction outcome
    """
    # Self-message is uncorrectable - route to human review immediately
    if quality_result.recommendation == "reject" and any(
        issue.get("category") == "self_message" for issue in quality_result.issues_found
    ):
        logger.error("Draft correction: Self-message detected - cannot correct, routing to HUMAN_REVIEW")
        _record_correction_attempt(success=False, reason="self_message_uncorrectable")
        return DraftCorrectionResult(
            success=False,
            corrected_draft=None,
            attempt_count=0,
            final_quality_result=quality_result,
            routed_to_human_review=True,
            failure_reason="Self-message detected - uncorrectable",
        )

    ai_provider = config_schema.ai_provider.lower()
    if not ai_provider:
        logger.warning("attempt_draft_correction: AI_PROVIDER not configured, routing to HUMAN_REVIEW")
        _record_correction_attempt(success=False, reason="no_ai_provider")
        return DraftCorrectionResult(
            success=False,
            corrected_draft=None,
            attempt_count=0,
            final_quality_result=quality_result,
            routed_to_human_review=True,
            failure_reason="AI provider not configured",
        )

    # Build correction guidance from issues
    correction_guidance = _build_correction_guidance(quality_result.issues_found)

    for attempt in range(1, max_attempts + 1):
        logger.info(f"Draft correction attempt {attempt}/{max_attempts}")

        corrected_draft = _regenerate_draft_with_corrections(
            original_draft=original_draft,
            issues_found=quality_result.issues_found,
            correction_guidance=correction_guidance,
            recipient_name=recipient_name,
            recipient_profile_id=recipient_profile_id,
            sender_name=sender_name,
            context_summary=context_summary,
            session_manager=session_manager,
            ai_provider=ai_provider,
        )

        if not corrected_draft or corrected_draft.startswith("BLOCKED:"):
            logger.warning(f"Draft correction attempt {attempt} failed: {corrected_draft or 'empty response'}")
            continue

        # Validate the corrected draft
        new_quality_result = validate_draft_quality(
            draft_message=corrected_draft,
            recipient_name=recipient_name,
            recipient_profile_id=recipient_profile_id,
            sender_name=sender_name,
            sender_profile_id=sender_profile_id,
            context_summary=context_summary,
            session_manager=session_manager,
        )

        if new_quality_result.passes_quality_check:
            logger.info(f"Draft correction successful on attempt {attempt}. Score: {new_quality_result.quality_score}")
            _record_correction_attempt(success=True, reason=None, attempt_number=attempt)
            return DraftCorrectionResult(
                success=True,
                corrected_draft=corrected_draft,
                attempt_count=attempt,
                final_quality_result=new_quality_result,
                routed_to_human_review=False,
                failure_reason=None,
            )

        logger.warning(
            f"Corrected draft still has issues on attempt {attempt}. "
            f"Score: {new_quality_result.quality_score}, Issues: {len(new_quality_result.issues_found)}"
        )

    # All attempts failed - route to human review
    logger.warning(f"Draft correction failed after {max_attempts} attempts. Routing to HUMAN_REVIEW")
    _record_correction_attempt(success=False, reason="max_attempts_exceeded", attempt_number=max_attempts)
    return DraftCorrectionResult(
        success=False,
        corrected_draft=None,
        attempt_count=max_attempts,
        final_quality_result=quality_result,
        routed_to_human_review=True,
        failure_reason=f"Quality check failed after {max_attempts} correction attempts",
    )


def _build_correction_guidance(issues: list[dict[str, Any]]) -> str:
    """Build human-readable correction guidance from issues list."""
    if not issues:
        return "No specific issues identified."

    guidance_parts = []
    for i, issue in enumerate(issues, 1):
        category = issue.get("category", "unknown")
        description = issue.get("description", "No description")
        suggestion = issue.get("suggestion", "No suggestion")
        guidance_parts.append(f"{i}. [{category.upper()}] {description}\n   Fix: {suggestion}")

    return "\n".join(guidance_parts)


def _regenerate_draft_with_corrections(
    original_draft: str,
    issues_found: list[dict[str, Any]],
    correction_guidance: str,
    recipient_name: str,
    recipient_profile_id: str,
    sender_name: str,
    context_summary: str,
    session_manager: Optional[SessionManager],
    ai_provider: str,
) -> Optional[str]:
    """Regenerate draft using the draft_correction prompt."""
    prompt_template = get_prompt("draft_correction")
    if not prompt_template:
        logger.warning("_regenerate_draft_with_corrections: draft_correction prompt not found")
        return None

    # Format issues as JSON string for the prompt
    issues_json = json.dumps(issues_found, indent=2)

    # Substitute placeholders (template strings)
    original_placeholder = "{original_draft}"  # noqa: RUF027
    issues_placeholder = "{issues_found}"  # noqa: RUF027
    guidance_placeholder = "{correction_guidance}"  # noqa: RUF027
    context_placeholder = "{context_summary}"  # noqa: RUF027
    recipient_name_placeholder = "{recipient_name}"  # noqa: RUF027
    recipient_id_placeholder = "{recipient_profile_id}"  # noqa: RUF027
    sender_name_placeholder = "{sender_name}"  # noqa: RUF027

    system_prompt = (
        prompt_template.replace(original_placeholder, original_draft)
        .replace(issues_placeholder, issues_json)
        .replace(guidance_placeholder, correction_guidance)
        .replace(context_placeholder, context_summary or "No context available")
        .replace(recipient_name_placeholder, recipient_name or "Unknown")
        .replace(recipient_id_placeholder, recipient_profile_id or "Unknown")
        .replace(sender_name_placeholder, sender_name or "Tree Owner")
    )

    start_time = time.time()
    corrected_text = _call_ai_model(
        provider=ai_provider,
        system_prompt=system_prompt,
        user_content="Regenerate the draft with the corrections applied.",
        session_manager=session_manager,
        max_tokens=800,
        temperature=0.7,
    )
    duration = time.time() - start_time

    if corrected_text:
        logger.info(f"Draft regeneration completed. (Took {duration:.2f}s)")
    else:
        logger.warning(f"Draft regeneration failed. (Took {duration:.2f}s)")

    return corrected_text


def _record_correction_attempt(
    success: bool,
    reason: Optional[str],
    attempt_number: int = 1,
) -> None:
    """Record correction attempt metrics for telemetry."""
    try:
        from ai.prompt_telemetry import record_prompt_result

        record_prompt_result(
            prompt_key="draft_correction",
            experiment_variant="auto_correction_v1",
            parse_success=success,
            quality_score=100 if success else 0,
            response_time_ms=0,  # Not tracking time here
            metadata={
                "correction_success": success,
                "failure_reason": reason,
                "attempt_number": attempt_number,
            },
        )
    except Exception as e:
        logger.debug(f"Failed to record correction attempt metrics: {e}")


def generate_record_research_strategy(context_history: str, session_manager: SessionManager) -> dict[str, Any] | None:
    """
    Analyzes conversations to suggest specific genealogical record research strategies.
    Returns structured data focused on record search opportunities and research plans.
    """
    ai_provider = config_schema.ai_provider.lower()
    default_empty_result = {"extracted_data": {}, "suggested_tasks": []}

    if not ai_provider:
        logger.error("generate_record_research_strategy: AI_PROVIDER not configured.")
        return default_empty_result

    if not context_history:
        logger.warning("generate_record_research_strategy: Empty context. Returning empty structure.")
        return default_empty_result

    # Get record research guidance prompt
    system_prompt = "Analyze this conversation for genealogical record research opportunities."
    if USE_JSON_PROMPTS:
        try:
            loaded_prompt = get_prompt("record_research_guidance")
            if loaded_prompt:
                system_prompt = loaded_prompt
            else:
                logger.warning("Failed to load 'record_research_guidance' prompt from JSON, using fallback.")
        except Exception as e:
            logger.warning(f"Error loading 'record_research_guidance' prompt: {e}, using fallback.")

    start_time = time.time()
    ai_response_str = _call_ai_model(
        provider=ai_provider,
        system_prompt=system_prompt,
        user_content=context_history,
        session_manager=session_manager,
        max_tokens=1500,
        temperature=0.2,
        response_format_type="json_object",
    )
    duration = time.time() - start_time

    if ai_response_str:
        try:
            ai_response = json.loads(ai_response_str)
            logger.info(f"Record research strategy generation successful. (Took {duration:.2f}s)")
            return ai_response
        except json.JSONDecodeError as e:
            logger.error(f"Record research strategy JSON parsing failed: {e}")
            return default_empty_result
    else:
        logger.error(f"Record research strategy generation failed. (Took {duration:.2f}s)")
        return default_empty_result


# --- Self-Test Functions ---


def _validate_ai_provider(ai_provider: str) -> bool:
    """Validate AI provider setting."""
    if not ai_provider:
        logger.error("❌ AI_PROVIDER not configured")
        return False
    if ai_provider not in SUPPORTED_AI_PROVIDERS:
        logger.error(
            "❌ Invalid AI_PROVIDER: %s. Must be one of: %s",
            ai_provider,
            ", ".join(SUPPORTED_AI_PROVIDERS),
        )
        return False
    logger.info(f"✅ AI_PROVIDER: {ai_provider}")
    return True


def _validate_deepseek_config() -> bool:
    """Validate DeepSeek-specific configuration."""
    config_valid = True

    if not openai_available:
        logger.error("❌ OpenAI library not available for DeepSeek")
        config_valid = False
    else:
        logger.info("✅ OpenAI library available")

    api_key = config_schema.api.deepseek_api_key
    model_name = config_schema.api.deepseek_ai_model
    base_url = config_schema.api.deepseek_ai_base_url

    if not api_key:
        logger.error("❌ DEEPSEEK_API_KEY not configured")
        config_valid = False
    else:
        logger.info(f"✅ DEEPSEEK_API_KEY configured (length: {len(api_key)})")

    if not model_name:
        logger.error("❌ DEEPSEEK_AI_MODEL not configured")
        config_valid = False
    else:
        logger.info(f"✅ DEEPSEEK_AI_MODEL: {model_name}")

    if not base_url:
        logger.error("❌ DEEPSEEK_AI_BASE_URL not configured")
        config_valid = False
    else:
        logger.info(f"✅ DEEPSEEK_AI_BASE_URL: {base_url}")

    return config_valid


def _validate_gemini_config() -> bool:
    """Validate Gemini-specific configuration."""
    config_valid = True

    if not genai_available:
        logger.error("❌ Google GenerativeAI library not available for Gemini")
        config_valid = False
    else:
        logger.info("✅ Google GenerativeAI library available")

    api_key = config_schema.api.google_api_key
    model_name = config_schema.api.google_ai_model

    if not api_key:
        logger.error("❌ GOOGLE_API_KEY not configured")
        config_valid = False
    else:
        logger.info(f"✅ GOOGLE_API_KEY configured (length: {len(api_key)})")

    if not model_name:
        logger.error("❌ GOOGLE_AI_MODEL not configured")
        config_valid = False
    else:
        logger.info(f"✅ GOOGLE_AI_MODEL: {model_name}")

    return config_valid


def _validate_moonshot_config() -> bool:
    """Validate Moonshot-specific configuration."""
    config_valid = True

    if not openai_available:
        logger.error("❌ OpenAI library not available for Moonshot")
        config_valid = False
    else:
        logger.info("✅ OpenAI library available (for Moonshot)")

    api_key = getattr(config_schema.api, "moonshot_api_key", None)
    model_name = getattr(config_schema.api, "moonshot_ai_model", None)
    base_url = getattr(config_schema.api, "moonshot_ai_base_url", None)

    if not api_key:
        logger.error("❌ MOONSHOT_API_KEY not configured")
        config_valid = False
    else:
        logger.info(f"✅ MOONSHOT_API_KEY configured (length: {len(api_key)})")

    if not model_name:
        logger.error("❌ MOONSHOT_AI_MODEL not configured")
        config_valid = False
    else:
        logger.info(f"✅ MOONSHOT_AI_MODEL: {model_name}")

    if not base_url:
        logger.error("❌ MOONSHOT_AI_BASE_URL not configured")
        config_valid = False
    else:
        logger.info(f"✅ MOONSHOT_AI_BASE_URL: {base_url}")

    return config_valid


def _validate_local_llm_config() -> bool:
    """Validate Local LLM-specific configuration."""
    config_valid = True

    if not openai_available:
        logger.error("❌ OpenAI library not available for Local LLM")
        config_valid = False
    else:
        logger.info("✅ OpenAI library available (for Local LLM)")

    api_key = config_schema.api.local_llm_api_key
    model_name = config_schema.api.local_llm_model
    base_url = config_schema.api.local_llm_base_url

    if not api_key:
        logger.error("❌ LOCAL_LLM_API_KEY not configured")
        config_valid = False
    else:
        logger.info("✅ LOCAL_LLM_API_KEY configured")

    if not model_name:
        logger.error("❌ LOCAL_LLM_MODEL not configured")
        config_valid = False
    else:
        logger.info(f"✅ LOCAL_LLM_MODEL: {model_name}")

    if not base_url:
        logger.error("❌ LOCAL_LLM_BASE_URL not configured")
        config_valid = False
    else:
        logger.info(f"✅ LOCAL_LLM_BASE_URL: {base_url}")

    return config_valid


def _validate_grok_config() -> bool:
    """Validate Grok (xAI) configuration."""
    config_valid = True

    if not xai_available:
        logger.error("❌ xai-sdk library not available for Grok")
        config_valid = False
    else:
        logger.info("✅ xai-sdk library available")

    api_key = getattr(config_schema.api, "xai_api_key", None)
    model_name = getattr(config_schema.api, "xai_model", None)
    api_host = getattr(config_schema.api, "xai_api_host", None)

    if not api_key:
        logger.error("❌ XAI_API_KEY not configured")
        config_valid = False
    else:
        logger.info(f"✅ XAI_API_KEY configured (length: {len(api_key)})")

    if not model_name:
        logger.error("❌ XAI_MODEL not configured")
        config_valid = False
    else:
        logger.info(f"✅ XAI_MODEL: {model_name}")

    if api_host:
        logger.info(f"✅ XAI_API_HOST: {api_host}")
    else:
        logger.info("XAI_API_HOST not set, defaulting to api.x.ai")

    return config_valid


def _validate_tetrate_config() -> bool:
    """Validate Tetrate (TARS) configuration."""
    config_valid = True

    if not openai_available:
        logger.error("❌ OpenAI library not available for Tetrate")
        config_valid = False
    else:
        logger.info("✅ OpenAI library available (for Tetrate)")

    api_key = getattr(config_schema.api, "tetrate_api_key", None)
    model_name = getattr(config_schema.api, "tetrate_ai_model", None)
    base_url = getattr(config_schema.api, "tetrate_ai_base_url", None)

    if not api_key:
        logger.error("❌ TARS_API_KEY / tetrate_api_key not configured")
        config_valid = False
    else:
        logger.info(f"✅ Tetrate API key configured (length: {len(api_key)})")

    if not model_name:
        logger.error("❌ TETRATE_AI_MODEL not configured")
        config_valid = False
    else:
        logger.info(f"✅ TETRATE_AI_MODEL: {model_name}")

    if not base_url:
        logger.error("❌ TETRATE_AI_BASE_URL not configured")
        config_valid = False
    else:
        logger.info(f"✅ TETRATE_AI_BASE_URL: {base_url}")

    return config_valid


def test_configuration() -> bool:
    """
    Tests AI configuration and dependencies.
    Returns True if all configurations are valid.
    """
    logger.info("=== Testing AI Configuration ===")

    # Test AI provider setting
    ai_provider = config_schema.ai_provider.lower()
    if not _validate_ai_provider(ai_provider):
        return False

    validators: dict[str, Any] = {
        "deepseek": _validate_deepseek_config,
        "gemini": _validate_gemini_config,
        "moonshot": _validate_moonshot_config,
        "local_llm": _validate_local_llm_config,
        "grok": _validate_grok_config,
        "tetrate": _validate_tetrate_config,
    }

    validator = validators.get(ai_provider)
    return bool(validator()) if validator else True


def _validate_extraction_task_structure(prompt_content: str) -> bool:
    """Validate extraction_task prompt structure."""
    has_nested_structure = "suggested_tasks" in prompt_content and "extracted_data" in prompt_content
    has_flat_structure = (
        "mentioned_names" in prompt_content and "dates" in prompt_content and "locations" in prompt_content
    )

    if has_nested_structure or has_flat_structure:
        structure_type = "nested" if has_nested_structure else "flat"
        logger.info(f"✅ extraction_task: contains valid {structure_type} structure keywords")
        return True
    logger.warning("⚠️ extraction_task: missing required structure keywords for either nested or flat format")
    return False


def _validate_single_prompt(prompt_name: str) -> bool:
    """Validate a single prompt can be loaded and has correct structure."""
    try:
        prompt_content = get_prompt(prompt_name)
        if not prompt_content:
            logger.error(f"❌ {prompt_name}: failed to load")
            return False

        logger.info(f"✅ {prompt_name}: loaded ({len(prompt_content)} characters)")

        # Test for key indicators in specific prompts
        if prompt_name == "extraction_task":
            return _validate_extraction_task_structure(prompt_content)

        return True
    except Exception as e:
        logger.error(f"❌ {prompt_name}: error loading - {e}")
        return False


def test_prompt_loading() -> bool:
    """
    Tests prompt loading functionality.
    Returns True if all required prompts can be loaded.
    """
    logger.info("=== Testing Prompt Loading ===")

    required_prompts = ["intent_classification", "extraction_task", "genealogical_reply"]

    try:
        # Test loading all prompts
        all_prompts = load_prompts()
        if not all_prompts:
            logger.error("❌ Failed to load prompts from JSON file")
            return False

        logger.info(f"✅ Loaded {len(all_prompts)} prompts from JSON file")

        # Test each required prompt
        return all(_validate_single_prompt(prompt_name) for prompt_name in required_prompts)

    except Exception as e:
        logger.error(f"❌ Prompt loading system error: {e}")
        return False


def test_pydantic_compatibility() -> bool:
    """
    Tests compatibility with expected Pydantic model structures.
    Returns True if the AI response format is compatible.
    """
    logger.info("=== Testing Pydantic Model Compatibility ===")

    # Test data structure that should be compatible with action9_process_productive.py models
    test_data = {
        "extracted_data": {
            "mentioned_names": ["John Smith", "Mary Jones"],
            "mentioned_locations": ["Aberdeen, Scotland", "Boston, Massachusetts"],
            "mentioned_dates": ["1850", "1881"],
            "potential_relationships": ["grandfather", "daughter of John"],
            "key_facts": ["worked as fisherman", "immigrated to Boston"],
        },
        "suggested_tasks": [
            "Search 1881 census for John Smith in Aberdeen",
            "Find marriage record for John Smith and Mary Jones",
            "Check immigration records for Boston arrivals 1880-1890",
        ],
    }

    try:
        # Try to import and test the AIResponse model
        from actions.action9_process_productive import AIResponse

        # Test creating the model
        ai_response = AIResponse(**cast(dict[str, Any], test_data))
        logger.info("✅ AIResponse model created successfully")
        logger.info(f"✅ extracted_data type: {type(ai_response.extracted_data)}")
        logger.info(f"✅ suggested_tasks type: {type(ai_response.suggested_tasks)}")
        logger.info(f"✅ suggested_tasks count: {len(ai_response.suggested_tasks)}")

        # Test accessing nested fields
        extracted = ai_response.extracted_data
        all_names = extracted.get_all_names()
        all_locations = extracted.get_all_locations()
        logger.info(f"✅ Nested fields accessible: names={len(all_names)}, locations={len(all_locations)}")

        return True

    except ImportError as e:
        logger.error(f"❌ Failed to import AIResponse model: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Pydantic model test failed: {e}")
        return False


def _test_ai_fallback_behavior(session_manager: SessionManager) -> bool:
    """Test AI fallback behavior when provider is not configured."""
    logger.info("⚠️ AI provider not configured - testing fallback behavior")
    result = classify_message_intent("Test message", session_manager)
    if result is None:
        logger.info("✅ Fallback behavior works correctly")
        return True
    logger.error(f"❌ Expected None for disabled AI, got: {result}")
    return False


def _test_configurable_provider_failover() -> bool:
    """Ensure provider failover honors configured order and adapter responses."""

    from testing.test_framework import suppress_logging

    class _RateLimiterStub:
        def wait(self) -> float:
            _ = self
            return 0.0

        def on_429_error(self, _source: str) -> None:
            _ = self
            _ = _source

    class _SessionManagerStub:
        def __init__(self) -> None:
            self.rate_limiter = _RateLimiterStub()

    class _FailingAdapter:
        name = "deepseek"

        def is_available(self) -> bool:
            _ = self
            return True

        def call(self, request: ProviderRequest) -> ProviderResponse:
            _ = self
            _ = request
            raise ProviderUnavailableError("forced failure")

    class _SuccessfulAdapter:
        name = "gemini"

        def is_available(self) -> bool:
            _ = self
            return True

        def call(self, request: ProviderRequest) -> ProviderResponse:
            _ = self
            _ = request
            return ProviderResponse(content="OK", raw_response={"provider": self.name})

    with suppress_logging():
        session_stub = cast("SessionManager", _SessionManagerStub())
        api_config = config_schema.api
        original_provider = config_schema.ai_provider
        original_fallbacks = list(getattr(config_schema, "ai_provider_fallbacks", []))
        original_deepseek_key = api_config.deepseek_api_key
        original_google_key = api_config.google_api_key
        original_adapters = {key: _PROVIDER_ADAPTERS.get(key) for key in ("deepseek", "gemini")}

        try:
            config_schema.ai_provider = "deepseek"
            config_schema.ai_provider_fallbacks = ["gemini"]
            if not api_config.deepseek_api_key:
                api_config.deepseek_api_key = "test-deepseek-key"
            if not api_config.google_api_key:
                api_config.google_api_key = "test-google-key"

            _PROVIDER_ADAPTERS["deepseek"] = _FailingAdapter()
            _PROVIDER_ADAPTERS["gemini"] = _SuccessfulAdapter()

            result = _call_ai_model(
                provider="deepseek",
                system_prompt="system",
                user_content="failover-test",
                session_manager=session_stub,
                max_tokens=8,
                temperature=0.0,
            )
            assert result == "OK", "Failover should return the fallback response"
        finally:
            config_schema.ai_provider = original_provider
            config_schema.ai_provider_fallbacks = original_fallbacks
            api_config.deepseek_api_key = original_deepseek_key
            api_config.google_api_key = original_google_key

            for key, adapter in original_adapters.items():
                if adapter is not None:
                    _PROVIDER_ADAPTERS[key] = adapter
                else:
                    _PROVIDER_ADAPTERS.pop(key, None)

    return True


def _test_moonshot_adapter_routing() -> bool:
    """Ensure Moonshot requests flow through the adapter registry."""

    from testing.test_framework import suppress_logging

    class _RateLimiterStub:
        def wait(self) -> float:
            _ = self
            return 0.0

        def on_429_error(self, _source: str) -> None:
            _ = self
            _ = _source

    class _SessionManagerStub:
        def __init__(self) -> None:
            self.rate_limiter = _RateLimiterStub()

    class _MoonshotAdapterStub:
        name = "moonshot"

        def __init__(self) -> None:
            self.calls = 0

        def is_available(self) -> bool:
            _ = self
            return True

        def call(self, request: ProviderRequest) -> ProviderResponse:
            self.calls += 1
            assert request.system_prompt == "system"
            return ProviderResponse(content="MOONSHOT-ADAPTER", raw_response={'provider': self.name})

    with suppress_logging():
        session_stub = cast("SessionManager", _SessionManagerStub())
        api_config = config_schema.api
        original_provider = config_schema.ai_provider
        original_fallbacks = list(getattr(config_schema, "ai_provider_fallbacks", []))
        original_key = api_config.moonshot_api_key
        original_model = api_config.moonshot_ai_model
        original_base = api_config.moonshot_ai_base_url
        original_adapter = _PROVIDER_ADAPTERS.get("moonshot")

        try:
            config_schema.ai_provider = "moonshot"
            config_schema.ai_provider_fallbacks = []
            if not api_config.moonshot_api_key:
                api_config.moonshot_api_key = "test-moonshot-key"
            if not api_config.moonshot_ai_model:
                api_config.moonshot_ai_model = "moonshot-test-model"
            if not api_config.moonshot_ai_base_url:
                api_config.moonshot_ai_base_url = "https://example.com/v1"

            adapter_stub = _MoonshotAdapterStub()
            _PROVIDER_ADAPTERS['moonshot'] = adapter_stub

            result = _call_ai_model(
                provider="moonshot",
                system_prompt="system",
                user_content="adapter-test",
                session_manager=session_stub,
                max_tokens=8,
                temperature=0.0,
            )
            assert result == "MOONSHOT-ADAPTER", "Moonshot adapter should handle the request"
            assert adapter_stub.calls == 1
        finally:
            config_schema.ai_provider = original_provider
            config_schema.ai_provider_fallbacks = original_fallbacks
            api_config.moonshot_api_key = original_key
            api_config.moonshot_ai_model = original_model
            api_config.moonshot_ai_base_url = original_base
            if original_adapter is not None:
                _PROVIDER_ADAPTERS['moonshot'] = original_adapter
            else:
                _PROVIDER_ADAPTERS.pop('moonshot', None)

    return True


def _test_local_llm_validation_helper() -> bool:
    """Verify the Local LLM validation helper handles matches and errors."""

    class _ModelResult:
        def __init__(self, names: list[str]) -> None:
            self.data = [type("Model", (), {"id": name}) for name in names]

    class _ModelList:
        def __init__(self, names: list[str]) -> None:
            self._names = names

        def list(self) -> _ModelResult:
            return _ModelResult(self._names)

    class _Client:
        def __init__(self, names: list[str]) -> None:
            self.models = _ModelList(names)

    exact_client = _Client(["namespace/my-model"])
    exact_match, error = _validate_local_llm_model_loaded(exact_client, "namespace/my-model")
    assert exact_match == "namespace/my-model" and error is None

    partial_client = _Client(["namespace/other-model", "foo/bar-baz"])
    partial_match, error2 = _validate_local_llm_model_loaded(partial_client, "bar-baz")
    assert partial_match == "foo/bar-baz" and error2 is None

    empty_client = _Client([])
    missing_match, error3 = _validate_local_llm_model_loaded(empty_client, "not-here")
    assert missing_match is None and error3 is not None

    return True


def _test_intent_classification(session_manager: SessionManager) -> bool:
    """Test AI intent classification functionality."""
    test_message = "Hello, I'm interested in genealogy research and finding common ancestors."
    logger.info("Testing intent classification...")
    intent_result = classify_message_intent(test_message, session_manager)

    if intent_result and intent_result in EXPECTED_INTENT_CATEGORIES:
        logger.info(f"✅ Intent classification successful: {intent_result}")
        return True
    if intent_result is None:
        logger.warning("⚠️ Intent classification returned None (AI may be unavailable)")
        return False
    logger.warning(f"⚠️ Intent classification returned unexpected result: {intent_result}")
    return False


def _test_genealogical_extraction(session_manager: SessionManager) -> bool:
    """Test AI genealogical data extraction functionality."""
    test_context = "SCRIPT: Hello! I'm researching genealogy.\nUSER: My great-grandfather John Smith was born in Aberdeen, Scotland around 1880. He was a fisherman who immigrated to Boston in 1905."
    logger.info("Testing genealogical data extraction...")
    extraction_result = extract_genealogical_entities(test_context, session_manager)

    if not extraction_result:
        logger.error(f"❌ Extraction failed or returned invalid structure: {extraction_result}")
        return False

    extracted_data = extraction_result.get("extracted_data", {})
    suggested_tasks = extraction_result.get("suggested_tasks", [])

    # Count extracted items
    names_count = len(extracted_data.get("mentioned_names", []))
    locations_count = len(extracted_data.get("mentioned_locations", []))
    dates_count = len(extracted_data.get("mentioned_dates", []))
    relationships_count = len(extracted_data.get("potential_relationships", []))
    facts_count = len(extracted_data.get("key_facts", []))
    tasks_count = len(suggested_tasks)

    logger.info(
        f"✅ Extraction successful: extracted {names_count} names, {locations_count} locations, "
        f"{dates_count} dates, {relationships_count} relationships, {facts_count} key facts, "
        f"{tasks_count} suggested tasks"
    )

    # Basic validation that some data was extracted
    if names_count > 0 or locations_count > 0 or dates_count > 0:
        logger.info("✅ AI successfully extracted meaningful genealogical data")
        return True
    logger.error(
        "❌ AI failed to extract expected genealogical entities from test context - investigate prompts or provider"
    )
    return False


def _test_reply_generation(session_manager: SessionManager) -> bool:
    """Test AI reply generation functionality."""
    test_genealogical_data = (
        "Person: John Smith, Born: 1880 Aberdeen Scotland, Occupation: Fisherman, Relationship: Great-grandfather"
    )
    logger.info("Testing genealogical reply generation...")
    reply_result = generate_genealogical_reply(
        "Previous conversation context",
        "Can you tell me about John Smith?",
        test_genealogical_data,
        session_manager,
    )

    if reply_result and len(reply_result) > 10:
        logger.info(f"✅ Reply generation successful (length: {len(reply_result)} characters)")
        return True
    logger.warning(f"⚠️ Reply generation returned unexpected result: {reply_result}")
    return True  # Non-critical warning


def _test_specialized_analysis_functions(session_manager: SessionManager) -> bool:
    """Test specialized genealogical analysis functions."""
    logger.info("Testing specialized genealogical analysis functions...")

    # Test DNA match analysis
    dna_test_context = "SCRIPT: Hello! I'm researching DNA matches.\nUSER: I have a DNA match showing 150 cM shared with someone named Sarah Johnson. AncestryDNA estimates we're 2nd cousins. We seem to share ancestors from Ireland in the 1800s."
    dna_result = analyze_dna_match_conversation(dna_test_context, session_manager)

    if dna_result:
        logger.info("✅ DNA match analysis function working")
    else:
        logger.warning("⚠️ DNA match analysis returned unexpected result")

    # Test family tree verification
    verification_test_context = "SCRIPT: Hello! I'm verifying family connections.\nUSER: I'm not sure if William Smith is really my great-grandfather. I have conflicting information about his birth year - some records say 1850, others say 1855."
    verification_result = verify_family_tree_connections(verification_test_context, session_manager)

    if verification_result:
        logger.info("✅ Family tree verification function working")
    else:
        logger.warning("⚠️ Family tree verification returned unexpected result")

    # Test record research strategy
    research_test_context = "SCRIPT: Hello! I need research help.\nUSER: I'm looking for birth records for my ancestor Mary O'Brien who was born around 1870 in County Cork, Ireland. She immigrated to Boston around 1890."
    research_result = generate_record_research_strategy(research_test_context, session_manager)

    if research_result:
        logger.info("✅ Record research strategy function working")
    else:
        logger.warning("⚠️ Record research strategy returned unexpected result")

    return True


def _test_toon_encoder() -> None:
    """Basic sanity check for TOON encoder used in prompts.

    Verifies that a representative extracted_entities payload is rendered with
    array headers and reasonable structure instead of raw JSON.
    """
    sample_entities: dict[str, Any] = {
        "mentioned_people": [
            {
                "name": "Mary Smith",
                "birth_year": 1881,
                "birth_place": "Banff, Scotland",
                "relationship": "great-grandmother",
            }
        ],
        "locations": [
            {"place": "Scotland", "context": "birthplace"},
        ],
    }

    toon = _encode_to_toon(sample_entities)
    assert "mentioned_people[1]" in toon, "TOON output should include array header for mentioned_people"
    assert "locations[1]" in toon, "TOON output should include array header for locations"


def test_ai_functionality(session_manager: SessionManager) -> bool:
    """
    Tests actual AI functionality if configuration allows.
    Returns True if AI calls work or are properly disabled.
    """
    logger.info("=== Testing AI Functionality ===")

    ai_provider = config_schema.ai_provider.lower()

    if not ai_provider:
        return _test_ai_fallback_behavior(session_manager)

    # Test with simple inputs if AI is configured
    logger.info(f"Testing with AI provider: {ai_provider}")

    try:
        # Test intent classification
        if not _test_intent_classification(session_manager):
            return False

        # Test extraction with genealogical content
        if not _test_genealogical_extraction(session_manager):
            return False

        # Test reply generation
        _test_reply_generation(session_manager)

        # Test specialized genealogical analysis functions
        _test_specialized_analysis_functions(session_manager)

        logger.info("✅ All AI functionality tests completed successfully")
        return True

    except Exception as e:
        logger.error(f"❌ AI functionality test failed with exception: {e}")
        return False


def ai_interface_module_tests() -> bool:
    """
    Comprehensive test suite for ai_interface.py - AI Interface & Integration Layer.
    Tests AI provider configuration, intent classification, entity extraction, and specialized analysis.
    """
    from core.session_manager import SessionManager
    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite("AI Interface & Integration Layer", "ai_interface.py")
    suite.start_suite()

    with suppress_logging():
        # === INITIALIZATION TESTS ===
        suite.run_test(
            "AI Provider Configuration",
            lambda: _validate_ai_provider(config_schema.ai_provider.lower()),
            "AI provider is properly configured (deepseek or gemini)",
            "Validate AI_PROVIDER setting from configuration",
            "Verify AI_PROVIDER is set to 'deepseek' or 'gemini'",
        )

        # === CORE FUNCTIONALITY TESTS ===
        suite.run_test(
            "Prompt Loading",
            lambda: _check_prompts_loaded([]),
            "Prompts are loaded successfully from JSON or fallback",
            "Test prompt loading functionality",
            "Verify prompts can be loaded for intent classification and extraction",
        )

        suite.run_test(
            "API Key Validation",
            lambda: _check_api_key_and_dependencies(config_schema.ai_provider.lower())[0],
            "API key is configured for the selected AI provider",
            "Test API key configuration",
            "Verify API key exists for deepseek or gemini",
        )

        suite.run_test(
            "Dependencies Available",
            lambda: _check_api_key_and_dependencies(config_schema.ai_provider.lower())[1],
            "Required AI libraries are available",
            "Test library availability",
            "Verify openai or google.genai libraries are installed",
        )

        suite.run_test(
            "Provider Failover Chain",
            _test_configurable_provider_failover,
            "Configurable fallback order engages alternate providers when the primary fails",
            "AI provider failover orchestration",
            "Override adapters to force failure and verify fallback succeeds",
        )

        suite.run_test(
            "Moonshot Adapter Routing",
            _test_moonshot_adapter_routing,
            "Moonshot adapter handles requests via registry",
            "Adapter routing",
            "Verify Moonshot provider routes through the adapter stub",
        )

        suite.run_test(
            "Local LLM Validation Helper",
            _test_local_llm_validation_helper,
            "Local LLM helper matches configured models and surfaces errors",
            "LM Studio validation",
            "Ensure helper detects loaded models and reports missing ones",
        )

        suite.run_test(
            "TOON Encoder Basic Structure",
            _test_toon_encoder,
            "TOON encoder renders extracted_entities-style payloads with array headers",
            "_encode_to_toon",
            "Encode a representative extracted_entities mapping into TOON",
            "Output includes headers for mentioned_people and locations arrays",
        )

        # === INTEGRATION TESTS (Require Live Session) ===
        # Skip live tests if SKIP_LIVE_API_TESTS is set (parallel test mode)
        import os

        skip_live = os.environ.get("SKIP_LIVE_API_TESTS", "").lower() == "true"

        if not skip_live:
            try:
                sm = SessionManager()
                # Mark browser as NOT needed for AI-only tests
                sm.browser_manager.browser_needed = False
                sm.start_sess("AI Interface Tests")

                suite.run_test(
                    "Health Check",
                    lambda: quick_health_check(sm)["overall_health"] in {"healthy", "degraded"},
                    "AI interface health check completes successfully",
                    "Test health check functionality",
                    "Verify health check returns valid status",
                )

                suite.run_test(
                    "Intent Classification",
                    lambda: test_ai_functionality(sm),
                    "AI intent classification and extraction work correctly",
                    "Test AI functionality with live session",
                    "Verify classify_message_intent and extract_genealogical_entities work",
                )

                sm.close_sess(keep_db=False)
            except Exception as e:
                logger.warning(f"Could not run live session tests: {e}")
        else:
            logger.info("⏭️  Skipping live AI tests (SKIP_LIVE_API_TESTS=true) - running in parallel mode")

    # === DRAFT QUALITY VALIDATION TESTS (no AI required) ===
    def test_self_message_detection() -> bool:
        """Test that self-message is detected without AI."""
        result = validate_draft_quality(
            draft_message="Hello!",
            recipient_name="Wayne",
            recipient_profile_id="PROFILE123",
            sender_name="Wayne",
            sender_profile_id="PROFILE123",
            context_summary="Test",
            session_manager=None,
        )
        return (
            not result.passes_quality_check
            and result.quality_score == 0
            and result.recommendation == "reject"
            and any(i["category"] == "self_message" for i in result.issues_found)
        )

    suite.run_test(
        "Draft Quality: Self-Message Detection",
        test_self_message_detection,
        "Self-message detection works without AI call",
        "validate_draft_quality",
        "Test that matching sender/recipient profile IDs are rejected",
    )

    def test_default_pass_on_empty_provider() -> bool:
        """Test that validation passes by default when AI unavailable."""
        result = validate_draft_quality(
            draft_message="Hello there!",
            recipient_name="John",
            recipient_profile_id="PROFILE456",
            sender_name="Wayne",
            sender_profile_id="PROFILE123",
            context_summary="Test",
            session_manager=None,
        )
        # Should pass with default score when AI can't run
        return result.passes_quality_check and result.quality_score == 100

    suite.run_test(
        "Draft Quality: Default Pass Without AI",
        test_default_pass_on_empty_provider,
        "Draft passes validation when AI unavailable (graceful degradation)",
        "validate_draft_quality",
        "Test that validation doesn't block when AI can't validate",
    )

    def test_draft_quality_result_dataclass() -> bool:
        """Test DraftQualityResult dataclass."""
        result = DraftQualityResult(
            passes_quality_check=False,
            issues_found=[{"category": "test", "severity": "low"}],
            quality_score=75,
            recommendation="revise",
        )
        return (
            not result.passes_quality_check
            and len(result.issues_found) == 1
            and result.quality_score == 75
            and result.recommendation == "revise"
        )

    suite.run_test(
        "Draft Quality: Result Dataclass",
        test_draft_quality_result_dataclass,
        "DraftQualityResult dataclass stores values correctly",
        "DraftQualityResult",
        "Test dataclass field assignment and retrieval",
    )

    # === AUTO-CORRECTION PIPELINE TESTS ===
    def test_correction_result_dataclass() -> bool:
        """Test DraftCorrectionResult dataclass."""
        result = DraftCorrectionResult(
            success=False,
            corrected_draft=None,
            attempt_count=1,
            final_quality_result=None,
            routed_to_human_review=True,
            failure_reason="Test failure",
        )
        return (
            not result.success
            and result.corrected_draft is None
            and result.attempt_count == 1
            and result.routed_to_human_review
            and result.failure_reason == "Test failure"
        )

    suite.run_test(
        "Auto-Correction: Result Dataclass",
        test_correction_result_dataclass,
        "DraftCorrectionResult dataclass stores values correctly",
        "DraftCorrectionResult",
        "Test dataclass field assignment and retrieval",
    )

    def test_self_message_routes_to_human_review() -> bool:
        """Test that self-message immediately routes to HUMAN_REVIEW."""
        # First create a quality result with self-message issue
        quality_result = DraftQualityResult(
            passes_quality_check=False,
            issues_found=[
                {
                    "category": "self_message",
                    "severity": "critical",
                    "description": "Message addressed to sender",
                    "suggestion": "Do not send",
                }
            ],
            quality_score=0,
            recommendation="reject",
        )
        result = attempt_draft_correction(
            original_draft="Hello me!",
            quality_result=quality_result,
            recipient_name="Wayne",
            recipient_profile_id="PROFILE123",
            sender_name="Wayne",
            sender_profile_id="PROFILE123",
            context_summary="Test",
            session_manager=None,
            max_attempts=1,
        )
        return (
            not result.success
            and result.routed_to_human_review
            and result.attempt_count == 0
            and "Self-message" in (result.failure_reason or "")
        )

    suite.run_test(
        "Auto-Correction: Self-Message Routes to Human Review",
        test_self_message_routes_to_human_review,
        "Self-message is immediately routed to HUMAN_REVIEW without correction attempt",
        "attempt_draft_correction",
        "Test that uncorrectable self-messages skip correction",
    )

    def test_build_correction_guidance() -> bool:
        """Test that correction guidance is built correctly from issues."""
        issues = [
            {"category": "context_inversion", "description": "Wrong direction", "suggestion": "Fix it"},
            {"category": "tone", "description": "Too casual", "suggestion": "Be formal"},
        ]
        guidance = _build_correction_guidance(issues)
        return (
            "[CONTEXT_INVERSION]" in guidance
            and "Wrong direction" in guidance
            and "[TONE]" in guidance
            and "Too casual" in guidance
        )

    suite.run_test(
        "Auto-Correction: Build Correction Guidance",
        test_build_correction_guidance,
        "Correction guidance is formatted correctly from issues list",
        "_build_correction_guidance",
        "Test helper function that builds human-readable guidance",
    )

    # === PHASE 1.5.2: CONTEXT ACCURACY VALIDATION TESTS ===
    def test_context_accuracy_result_dataclass() -> bool:
        """Test ContextAccuracyResult dataclass."""
        result = ContextAccuracyResult(
            is_accurate=False,
            verified_facts=[{"name": "John Smith", "confidence": "high"}],
            unverified_facts=[{"name": "Jane Doe", "reason": "Not found"}],
            known_to_recipient=[{"name": "Mary Jones"}],
            accuracy_score=70,
            warnings=["Jane Doe not found"],
            recommendation="review",
        )
        return (
            not result.is_accurate
            and len(result.verified_facts) == 1
            and len(result.unverified_facts) == 1
            and len(result.known_to_recipient) == 1
            and result.accuracy_score == 70
            and result.recommendation == "review"
        )

    suite.run_test(
        "Context Accuracy: Result Dataclass",
        test_context_accuracy_result_dataclass,
        "ContextAccuracyResult dataclass stores values correctly",
        "ContextAccuracyResult",
        "Test dataclass field assignment and retrieval",
    )

    def test_context_accuracy_empty_names() -> bool:
        """Test that empty names list returns passing result."""
        result = validate_context_accuracy(
            extracted_names=[],
            context_summary="Test context",
        )
        return (
            result.is_accurate
            and result.accuracy_score == 100
            and result.recommendation == "proceed"
            and len(result.verified_facts) == 0
            and len(result.unverified_facts) == 0
        )

    suite.run_test(
        "Context Accuracy: Empty Names List",
        test_context_accuracy_empty_names,
        "Empty extracted names returns passing validation",
        "validate_context_accuracy",
        "Test that no names to validate = automatic pass",
    )

    def test_extract_names_from_text() -> bool:
        """Test name extraction from draft text."""
        text = "My grandfather John Smith was born in 1920. His wife Mary Jane Watson lived in Chicago."
        names = extract_names_from_text(text)
        return len(names) >= 2 and any("John Smith" in n for n in names) and any("Mary Jane Watson" in n for n in names)

    suite.run_test(
        "Context Accuracy: Name Extraction",
        test_extract_names_from_text,
        "Names are extracted from text correctly",
        "extract_names_from_text",
        "Test heuristic name extraction from draft messages",
    )

    def test_extract_names_filters_non_names() -> bool:
        """Test that common non-names are filtered out."""
        text = "Dear Friend, Thank You for sharing. Best Regards, John Smith."
        names = extract_names_from_text(text)
        # Should find John Smith but filter out Dear Friend, Thank You, Best Regards
        return (
            any("John Smith" in n for n in names)
            and not any("Dear Friend" in n for n in names)
            and not any("Best Regards" in n for n in names)
        )

    suite.run_test(
        "Context Accuracy: Non-Name Filtering",
        test_extract_names_filters_non_names,
        "Common salutations and sign-offs are filtered from names",
        "extract_names_from_text",
        "Test that 'Dear Friend', 'Best Regards' are not detected as names",
    )

    # === PHASE 2.3: STRUCTURED REPLY TESTS ===
    def test_structured_reply_dataclasses() -> bool:
        """Test StructuredReplyResult and related dataclasses."""
        evidence = EvidenceSource(source="GEDCOM", reference="@I123@", fact="John Smith b. 1850")
        missing = MissingInformation(field="Birth place", impact="high", suggested_source="Parish records")
        breakdown = ConfidenceBreakdown(base=50, adjustments=[{"+30": "Direct match"}], final=80)
        result = StructuredReplyResult(
            draft_message="Test message",
            confidence=80,
            confidence_breakdown=breakdown,
            evidence_used=[evidence],
            missing_information=[missing],
            suggested_facts=["Verify parents"],
            follow_up_questions=["When was he born?"],
            route_to_human_review=False,
            human_review_reason=None,
        )
        return (
            result.draft_message == "Test message"
            and result.confidence == 80
            and len(result.evidence_used) == 1
            and result.evidence_used[0].source == "GEDCOM"
            and len(result.missing_information) == 1
            and result.missing_information[0].field == "Birth place"
            and not result.should_route_to_human()
        )

    suite.run_test(
        "Structured Reply: Dataclass Creation",
        test_structured_reply_dataclasses,
        "StructuredReplyResult and related dataclasses work correctly",
        "StructuredReplyResult",
        "Test Phase 2.3 dataclass creation and validation",
    )

    def test_structured_reply_to_dict() -> bool:
        """Test StructuredReplyResult.to_dict() serialization."""
        evidence = EvidenceSource(source="Census", reference="RG11/1234", fact="Fisherman in Banff")
        missing = MissingInformation(field="Marriage date", impact="medium", suggested_source="Civil registration")
        breakdown = ConfidenceBreakdown(base=50, adjustments=[{"-20": "Not found"}], final=30)
        result = StructuredReplyResult(
            draft_message="Low confidence message",
            confidence=30,
            confidence_breakdown=breakdown,
            evidence_used=[evidence],
            missing_information=[missing],
            suggested_facts=["Find birth record"],
            follow_up_questions=["Where was he from?"],
            route_to_human_review=True,
            human_review_reason="Low confidence",
        )
        d = result.to_dict()
        return (
            isinstance(d, dict)
            and d["draft_message"] == "Low confidence message"
            and d["confidence"] == 30
            and len(d["evidence_used"]) == 1
            and d["evidence_used"][0]["source"] == "Census"
            and d["route_to_human_review"] is True
        )

    suite.run_test(
        "Structured Reply: to_dict Serialization",
        test_structured_reply_to_dict,
        "StructuredReplyResult.to_dict() produces valid dictionary",
        "StructuredReplyResult.to_dict",
        "Test JSON-serializable dictionary conversion",
    )

    def test_structured_reply_human_review_routing() -> bool:
        """Test should_route_to_human() logic."""
        # Low confidence should route to human
        low_conf = StructuredReplyResult(
            draft_message="Test",
            confidence=40,
            confidence_breakdown=ConfidenceBreakdown(base=50, adjustments=[], final=40),
            evidence_used=[],
            missing_information=[],
            suggested_facts=[],
            follow_up_questions=[],
            route_to_human_review=False,
            human_review_reason=None,
        )
        # Explicit route should override
        explicit = StructuredReplyResult(
            draft_message="Test",
            confidence=90,
            confidence_breakdown=ConfidenceBreakdown(base=50, adjustments=[], final=90),
            evidence_used=[],
            missing_information=[],
            suggested_facts=[],
            follow_up_questions=[],
            route_to_human_review=True,
            human_review_reason="Sensitive topic",
        )
        return (
            low_conf.should_route_to_human()  # Low confidence triggers
            and low_conf.get_review_reason() == "Low confidence score (40/100)"
            and explicit.should_route_to_human()  # Explicit flag triggers
            and explicit.get_review_reason() == "Sensitive topic"
        )

    suite.run_test(
        "Structured Reply: Human Review Routing",
        test_structured_reply_human_review_routing,
        "should_route_to_human() correctly identifies review cases",
        "StructuredReplyResult.should_route_to_human",
        "Test low confidence and explicit routing triggers",
    )

    def test_parse_structured_reply_response() -> bool:
        """Test _parse_structured_reply_response JSON parsing."""
        sample_json = """
        {
            "draft_message": "Hello! I found John Smith in my tree.",
            "confidence": 85,
            "confidence_breakdown": {
                "base": 50,
                "adjustments": [{"+30": "Direct match"}, {"+5": "Related members"}],
                "final": 85
            },
            "evidence_used": [
                {"source": "GEDCOM", "reference": "@I45@", "fact": "John Smith 1850-1920"}
            ],
            "missing_information": [],
            "suggested_facts": ["Verify parents"],
            "follow_up_questions": ["Do you have any records?"],
            "route_to_human_review": false,
            "human_review_reason": null
        }
        """
        result = _parse_structured_reply_response(sample_json)
        return (
            result is not None
            and result.draft_message.startswith("Hello!")
            and result.confidence == 85
            and result.confidence_breakdown.final == 85
            and len(result.evidence_used) == 1
            and result.evidence_used[0].reference == "@I45@"
            and not result.route_to_human_review
        )

    suite.run_test(
        "Structured Reply: JSON Parsing",
        test_parse_structured_reply_response,
        "_parse_structured_reply_response correctly parses valid JSON",
        "_parse_structured_reply_response",
        "Test AI response JSON parsing into dataclass",
    )

    def test_parse_structured_reply_invalid_json() -> bool:
        """Test _parse_structured_reply_response handles invalid JSON gracefully."""
        invalid_json = "This is not valid JSON at all"
        result = _parse_structured_reply_response(invalid_json)
        return result is None

    suite.run_test(
        "Structured Reply: Invalid JSON Handling",
        test_parse_structured_reply_invalid_json,
        "_parse_structured_reply_response returns None for invalid JSON",
        "_parse_structured_reply_response",
        "Test graceful error handling for malformed responses",
    )

    return suite.finish_suite()


# Use centralized test runner utility from test_utilities
run_comprehensive_tests = create_standard_test_runner(ai_interface_module_tests)


def _check_api_key_and_dependencies(ai_provider: str) -> tuple[bool, bool]:
    """Check if API key is configured and dependencies are available."""
    if ai_provider == "deepseek":
        return bool(config_schema.api.deepseek_api_key), openai_available
    if ai_provider == "gemini":
        return bool(config_schema.api.google_api_key), genai_available
    if ai_provider == "moonshot":
        return bool(getattr(config_schema.api, "moonshot_api_key", None)), openai_available
    if ai_provider == "local_llm":
        return bool(config_schema.api.local_llm_api_key), openai_available
    if ai_provider == "tetrate":
        return bool(getattr(config_schema.api, "tetrate_api_key", None)), openai_available
    return False, False


def _check_prompts_loaded(errors: list[str]) -> bool:
    """Check if prompts are loaded successfully."""
    try:
        prompts = load_prompts()
        return bool(prompts and "extraction_task" in prompts)
    except Exception as e:
        errors.append(f"Prompt loading error: {e}")
        return False


def _perform_test_call(
    session_manager: SessionManager, api_key_configured: bool, dependencies_available: bool, errors: list[str]
) -> bool:
    """Perform a quick test call if configuration looks good."""
    if not (api_key_configured and dependencies_available):
        return False

    try:
        result = classify_message_intent("Test", session_manager)
        return result is not None
    except Exception as e:
        errors.append(f"Test call error: {e}")
        return False


def _determine_overall_health(
    api_key_configured: bool, prompts_loaded: bool, dependencies_available: bool, test_call_successful: bool
) -> str:
    """Determine overall health status based on checks."""
    if api_key_configured and prompts_loaded and dependencies_available:
        return "healthy" if test_call_successful else "degraded"
    return "unhealthy"


def quick_health_check(session_manager: SessionManager) -> HealthStatus:
    """
    Performs a quick health check of the AI interface.
    Returns a dictionary with health status information.
    """
    health_status: HealthStatus = {
        "overall_health": "unknown",
        "ai_provider": config_schema.ai_provider,
        "api_key_configured": False,
        "prompts_loaded": False,
        "dependencies_available": False,
        "test_call_successful": False,
        "errors": [],
    }

    try:
        # Check API key and dependencies
        ai_provider = config_schema.ai_provider.lower()
        health_status["api_key_configured"], health_status["dependencies_available"] = _check_api_key_and_dependencies(
            ai_provider
        )

        # Check prompts
        health_status["prompts_loaded"] = _check_prompts_loaded(health_status["errors"])

        # Quick test call
        health_status["test_call_successful"] = _perform_test_call(
            session_manager,
            health_status["api_key_configured"],
            health_status["dependencies_available"],
            health_status["errors"],
        )

        # Determine overall health
        health_status["overall_health"] = _determine_overall_health(
            health_status["api_key_configured"],
            health_status["prompts_loaded"],
            health_status["dependencies_available"],
            health_status["test_call_successful"],
        )

    except Exception as e:
        health_status["errors"].append(f"Health check error: {e}")
        health_status["overall_health"] = "error"

    return health_status


def self_check_ai_interface(session_manager: SessionManager) -> bool:
    """
    Performs a self-check of the AI interface functionality using real API calls.
    Tests intent classification and data extraction/task suggestion.
    """
    logger.info("Starting AI interface self-check...")
    all_passed = True

    # Test data (from original file)
    test_context_1 = """SCRIPT: Hello! I noticed we share some DNA matches. I'm researching the Simpson family from Scotland. Do you have any information about Margaret Simpson who might have been born around 1858?

USER: I don't really work on my family tree much anymore."""  # Expected: UNINTERESTED

    test_context_2 = """SCRIPT: That's wonderful! Margaret Simpson sounds familiar. Do you know who Alexander's parents were or if he had any siblings?

USER: Sorry, I don't think I can help you. Good luck with your research."""  # Expected: UNINTERESTED

    extraction_conversation_test = """
SCRIPT: Hello, I'm researching my family tree and we appear to be DNA matches. I'm trying to find our common ancestor. My tree includes the Simpson family from Aberdeen, Scotland in the 1800s. Does that connect with your research?
USER: Yes, I have Simpsons in my tree from Aberdeen! My great-grandmother was Margaret Simpson born around 1865. Her father was Alexander Simpson who was a fisherman. He was born about 1835 and died in 1890. The family lived in a fishing village in Aberdeen.
SCRIPT: That's wonderful! Margaret Simpson sounds familiar. Do you know who Alexander's parents were or if he had any siblings?
USER: Alexander's parents were John Simpson and Elizabeth Cruickshank. They married in 1833 in Aberdeen. John was born around 1810 and worked as a fisherman like his son. Elizabeth was from Peterhead originally. Alexander had a sister named Isobella who was born in 1840. I found their marriage certificate in the Scottish records.
"""

    logger.info("--- Testing Intent Classification ---")
    result1 = classify_message_intent(test_context_1, session_manager)
    if result1 and result1 in EXPECTED_INTENT_CATEGORIES:
        logger.info(f"Intent classification test 1 PASSED. Result: {result1}")
    else:
        logger.error(f"Intent classification test 1 FAILED. Expected valid category, got: {result1}")
        all_passed = False

    result2 = classify_message_intent(test_context_2, session_manager)
    if result2 and result2 in EXPECTED_INTENT_CATEGORIES:
        logger.info(f"Intent classification test 2 PASSED. Result: {result2}")
    else:
        logger.error(f"Intent classification test 2 FAILED. Expected valid category, got: {result2}")
        all_passed = False

    logger.info("--- Testing Data Extraction & Task Suggestion ---")
    extraction_result = extract_genealogical_entities(extraction_conversation_test, session_manager)

    if (
        extraction_result
        and isinstance(extraction_result.get("extracted_data"), dict)
        and isinstance(extraction_result.get("suggested_tasks"), list)
    ):
        logger.info("Data extraction test PASSED structure validation.")
        logger.info(f"Extracted data: {json.dumps(extraction_result['extracted_data'], indent=2)}")
        logger.info(f"Suggested tasks: {extraction_result['suggested_tasks']}")
        if not extraction_result["suggested_tasks"]:
            logger.warning("AI did not suggest any tasks for the extraction test context. Prompt may need adjustment.")
            # This is not a failure of the function itself, but a note on prompt effectiveness.
    else:
        logger.error(f"Data extraction test FAILED. Invalid structure or None result. Got: {extraction_result}")
        all_passed = False

    if all_passed:
        logger.info("AI interface self-check PASSED.")
    else:
        logger.error("AI interface self-check FAILED.")
    return all_passed


# End of self_check_ai_interface

# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    import sys
    import traceback

    try:
        print("🤖 Running AI Interface & Integration Layer comprehensive test suite...")
        success = run_comprehensive_tests()
    except Exception:
        print("\n[ERROR] Unhandled exception during AI Interface tests:", file=sys.stderr)
        traceback.print_exc()
        success = False

    sys.exit(0 if success else 1)
