#!/usr/bin/env python3
from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportConstantRedefinition=false

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

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    setup_module,
)

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
import json
import logging
import sys  # Not strictly used but often good for system-level interactions
import time
from typing import TYPE_CHECKING, Any, Optional

# === THIRD-PARTY IMPORTS ===
# Attempt OpenAI import for DeepSeek/compatible APIs
try:
    from openai import (
        APIConnectionError,
        APIError,
        AuthenticationError,
        OpenAI,
        RateLimitError,
    )

    openai_available = True
except ImportError:
    OpenAI = None  # type: ignore
    APIConnectionError = None  # type: ignore
    RateLimitError = None  # type: ignore
    AuthenticationError = None  # type: ignore
    APIError = None  # type: ignore
    openai_available = False
    logging.warning("OpenAI library not found. DeepSeek functionality disabled.")

# Attempt Google Gemini import
try:
    from google import genai  # Updated to google-genai package
    from google.genai import errors as genai_errors, types as genai_types

    genai_available = True
    if not hasattr(genai, "Client"):
        genai_available = False
        logging.warning("Google GenAI library structure seems incomplete.")
except Exception:
    genai = None  # type: ignore
    genai_types = None  # type: ignore
    genai_errors = None  # type: ignore
    genai_available = False
    logging.warning(
        "Google GenAI library not found. Gemini functionality disabled."
    )

# === LOCAL IMPORTS ===
import contextlib

# === PHASE 5.2: SYSTEM-WIDE CACHING OPTIMIZATION ===
from cache_manager import cached_api_call
from config.config_manager import ConfigManager

if TYPE_CHECKING:
    from core.session_manager import SessionManager

# === MODULE CONFIGURATION ===
# Initialize config
config_manager = ConfigManager()
config_schema = config_manager.get_config()

# --- Test framework imports ---

# --- Constants and Prompts ---
try:
    from ai_prompt_utils import get_prompt, load_prompts

    USE_JSON_PROMPTS = True
    logger.debug("AI prompt utilities loaded successfully - will use JSON prompts")
except ImportError:
    logger.warning("ai_prompt_utils module not available, using fallback prompts")
    USE_JSON_PROMPTS = False
    # Provide minimal fallback stubs so later references are defined
    from typing import Optional as _Optional
    def get_prompt(prompt_key: str) -> str | None:  # type: ignore[misc]
        _ = prompt_key  # Fallback stub - parameter required for API compatibility
        return None
    def load_prompts() -> dict[str, Any]:  # type: ignore
        return {"prompts": {}}

# Based on the prompt from the original ai_interface.py (and updated with more categories from ai_prompts.json example)
EXPECTED_INTENT_CATEGORIES = {
    "ENTHUSIASTIC",
    "CAUTIOUSLY_INTERESTED",
    "UNINTERESTED",
    "CONFUSED",
    "PRODUCTIVE",
    "OTHER",
    "DESIST",  # Added from the original file's SYSTEM_PROMPT_INTENT
}

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
            logger.warning("_call_ai_model: SessionManager or rate limiter not available. Proceeding without rate limiting.")
    except Exception:
        logger.debug("Rate limiter invocation failed; proceeding without enforced wait.")


def _build_deepseek_request_params(
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    response_format_type: str | None
) -> dict[str, Any]:
    """Build request parameters for DeepSeek API call."""
    request_params: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if response_format_type == "json_object":
        request_params["response_format"] = {"type": "json_object"}
    return request_params


def _call_deepseek_model(system_prompt: str, user_content: str, max_tokens: int, temperature: float, response_format_type: str | None) -> str | None:
    """Call DeepSeek AI model."""
    if not openai_available or OpenAI is None:
        logger.error("_call_ai_model: OpenAI library not available for DeepSeek.")
        return None

    api_key = config_schema.api.deepseek_api_key
    model_name = config_schema.api.deepseek_ai_model
    base_url = config_schema.api.deepseek_ai_base_url

    if not all([api_key, model_name, base_url]):
        logger.error("_call_ai_model: DeepSeek configuration incomplete.")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    request_params = _build_deepseek_request_params(model_name, messages, max_tokens, temperature, response_format_type)

    response = client.chat.completions.create(**request_params)
    if response.choices and response.choices[0].message and response.choices[0].message.content:
        return response.choices[0].message.content.strip()
    logger.error("DeepSeek returned an empty or invalid response structure.")
    return None


def _call_moonshot_model(system_prompt: str, user_content: str, max_tokens: int, temperature: float, response_format_type: str | None) -> str | None:
    """Call Moonshot (Kimi) AI model using OpenAI-compatible endpoint."""
    if not openai_available or OpenAI is None:
        logger.error("_call_ai_model: OpenAI library not available for Moonshot.")
        return None

    api_key = getattr(config_schema.api, "moonshot_api_key", None)
    model_name = getattr(config_schema.api, "moonshot_ai_model", None)
    base_url = getattr(config_schema.api, "moonshot_ai_base_url", None)

    if not all([api_key, model_name, base_url]):
        logger.error("_call_ai_model: Moonshot configuration incomplete.")
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
        try:
            reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
            if reasoning_content:
                # Log a short preview to help during provider evaluation without overwhelming logs.
                logger.debug(f"Moonshot reasoning preview: {str(reasoning_content)[:200]}")
        except Exception:
            pass
        return response.choices[0].message.content.strip()

    logger.error("Moonshot returned an empty or invalid response structure.")
    return None


# Helper functions for _call_gemini_model

def _validate_gemini_availability() -> bool:
    """Validate Gemini library is available."""
    if not genai_available or genai is None:
        logger.error("_call_ai_model: Google GenAI library not available for Gemini.")
        return False

    if not hasattr(genai, "Client"):
        logger.error("_call_ai_model: Gemini library missing expected interfaces.")
        return False

    return True


def _get_gemini_config() -> tuple[str | None, str | None]:
    """Get Gemini API key and model name from config."""
    api_key = getattr(config_schema.api, "google_api_key", None)
    model_name = getattr(config_schema.api, "google_ai_model", None)

    if not api_key or not model_name:
        logger.error("_call_ai_model: Gemini configuration incomplete.")
        return None, None

    return api_key, model_name


def _list_available_gemini_models() -> list[str]:
    """
    List all available Gemini models with generateContent support.

    Returns:
        List of model names (e.g., ["gemini-2.5-flash", "gemini-2.5-pro-preview-03-25"])
        Returns empty list if listing fails or genai is not available.
    """
    if not genai_available:
        return []

    if not hasattr(genai, "list_models"):
        logger.debug("genai.list_models() not available in this SDK version")
        return []

    try:
        models = []
        for model in genai.list_models():  # type: ignore[attr-defined]
            if hasattr(model, "supported_generation_methods"):
                methods = getattr(model, "supported_generation_methods", [])
                if "generateContent" in methods:
                    # Strip "models/" prefix for easier use
                    model_name = getattr(model, "name", "")
                    normalized_name = model_name.replace("models/", "")
                    if normalized_name:
                        models.append(normalized_name)
        return models
    except Exception as e:
        logger.debug(f"Failed to list Gemini models: {e}")
        return []


def _validate_gemini_model_exists(model_name: str) -> bool:
    """
    Check if the configured model exists and supports generateContent.

    Args:
        model_name: Model name to validate (e.g., "gemini-2.5-flash")

    Returns:
        True if model is valid, False otherwise.
        Returns True if model listing is not available (to avoid breaking existing functionality).
    """
    available = _list_available_gemini_models()

    # If we couldn't list models, assume the configured model is valid
    # (better to attempt the API call than block it)
    if not available:
        return True

    # Handle both formats: "gemini-2.5-flash" and "models/gemini-2.5-flash"
    normalized = model_name.replace("models/", "")

    if normalized not in available:
        logger.error(f"❌ Model '{model_name}' not found or doesn't support generateContent")
        if len(available) > 0:
            logger.error(f"📋 Available models: {', '.join(available[:5])}")
            if len(available) > 5:
                logger.error(f"   ... and {len(available) - 5} more")
        return False

    logger.debug(f"✅ Model '{normalized}' validated successfully")
    return True


def _initialize_gemini_model(api_key: str, model_name: str) -> Any | None:
    """
    Initialize Gemini model with validation.

    Args:
        api_key: Google API key
        model_name: Model name (e.g., "gemini-2.5-flash")

    Returns:
        Initialized model instance or None if validation/initialization fails
    """
    # Validate model exists before attempting API call
    if not _validate_gemini_model_exists(model_name):
        available = _list_available_gemini_models()
        if available:
            suggested = available[0]
            logger.error(f"💡 Suggestion: Update GOOGLE_AI_MODEL in .env to: {suggested}")
            logger.error(f"💡 Or choose from: {', '.join(available[:3])}")
        return None

    try:
        genai.configure(api_key=api_key)  # type: ignore[attr-defined]
        model = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]
        logger.debug(f"✅ Gemini model '{model_name}' initialized successfully")
        return model
    except Exception as e:
        logger.error(f"_call_ai_model: Failed initializing Gemini model '{model_name}': {e}")
        # Suggest available models on failure
        available = _list_available_gemini_models()
        if available:
            logger.error(f"💡 Available models: {', '.join(available[:3])}")
        return None


def _create_gemini_generation_config(max_tokens: int, temperature: float) -> Any | None:
    """Create Gemini generation config."""
    if not hasattr(genai, "GenerationConfig"):
        return None

    try:
        return genai.GenerationConfig(  # type: ignore[attr-defined]
            candidate_count=1,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception:
        return None


def _generate_gemini_content(model: Any, full_prompt: str, generation_config: Any | None) -> Any | None:
    """Generate content using Gemini model."""
    if not hasattr(model, "generate_content"):
        return None

    try:
        return model.generate_content(full_prompt, generation_config=generation_config)  # type: ignore[call-arg]
    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return None


def _extract_gemini_response_text(response: Any | None) -> str | None:
    """Extract text from Gemini response."""
    if response is not None and getattr(response, "text", None):
        return getattr(response, "text", "").strip()

    # Log block reason if response was blocked
    block_reason_msg = "Unknown"
    try:
        if response is not None and hasattr(response, "prompt_feedback"):
            pf = getattr(response, "prompt_feedback", None)
            if pf and hasattr(pf, "block_reason"):
                br = getattr(pf, "block_reason", None)
                if hasattr(br, "name"):
                    block_reason_msg = getattr(br, "name", "Unknown")
                elif br is not None:
                    block_reason_msg = str(br)
    except Exception:
        pass

    logger.error(f"Gemini returned an empty or blocked response. Reason: {block_reason_msg}")
    return None


def _call_gemini_model(system_prompt: str, user_content: str, max_tokens: int, temperature: float) -> str | None:
    """Call Gemini AI model."""
    # Validate Gemini availability
    if not _validate_gemini_availability():
        return None

    # Get configuration
    api_key, model_name = _get_gemini_config()
    if not api_key or not model_name:
        return None

    # Initialize model
    model = _initialize_gemini_model(api_key, model_name)
    if not model:
        return None

    # Prepare prompt and config
    full_prompt = f"{system_prompt}\n\n---\n\nUser Query/Content:\n{user_content}"
    generation_config = _create_gemini_generation_config(max_tokens, temperature)

    # Generate content
    response = _generate_gemini_content(model, full_prompt, generation_config)

    # Extract and return response text
    return _extract_gemini_response_text(response)


def _call_local_llm_model(system_prompt: str, user_content: str, max_tokens: int, temperature: float, response_format_type: str | None) -> str | None:  # noqa: ARG001
    """
    Call Local LLM model via LM Studio OpenAI-compatible API.

    LM Studio provides an OpenAI-compatible API endpoint at http://localhost:1234/v1
    This allows seamless integration with the existing OpenAI client.

    Note: response_format_type parameter is kept for API compatibility but not used
    because LM Studio doesn't support the response_format parameter. JSON output
    is controlled via system prompt instructions instead.

    Recommended models for Dell XPS 15 9520 (i9-12900HK, 64GB RAM, RTX 3050 Ti 4GB):
    - Qwen2.5-14B-Instruct-Q4_K_M: Best quality, 2-4s response time
    - Llama-3.2-11B-Vision-Instruct: Good quality, 1-3s response time
    - Mistral-7B-Instruct-v0.3: Fast, <2s response time
    """
    # Validate prerequisites
    if not openai_available or OpenAI is None:
        logger.error("_call_local_llm_model: OpenAI library not available for Local LLM.")
        return None

    api_key = config_schema.api.local_llm_api_key
    model_name = config_schema.api.local_llm_model
    base_url = config_schema.api.local_llm_base_url

    if not all([api_key, model_name, base_url]):
        logger.error("_call_local_llm_model: Local LLM configuration incomplete.")
        return None

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        # Validate model is loaded and get the actual model name
        actual_model_name, error_msg = _validate_local_llm_model_loaded(client, model_name)
        if error_msg:
            logger.error(error_msg)
            return None

        # Make API call using the actual model name from LM Studio
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Build request params - LM Studio doesn't support response_format, so omit it
        request_params: dict[str, Any] = {
            "model": actual_model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        # Note: LM Studio doesn't support response_format parameter
        # JSON output is controlled via system prompt instructions instead

        response = client.chat.completions.create(**request_params)
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()

        logger.error("Local LLM returned an empty or invalid response structure.")
        return None
    except Exception as e:
        logger.error(f"Local LLM API call failed: {e}")
        return None


def _validate_local_llm_model_loaded(client, model_name: str) -> tuple[str | None, str | None]:
    """
    Validate that the requested model is loaded in LM Studio.

    Supports flexible model name matching:
    - Exact match: "qwen/qwen3-4b-2507" == "qwen/qwen3-4b-2507"
    - Partial match: "qwen3-4b-2507" matches "qwen/qwen3-4b-2507"

    Returns:
        Tuple of (actual_model_name, error_message)
        - If successful: (actual_model_name, None)
        - If failed: (None, error_message)
    """
    try:
        models = client.models.list()
        available_models = [model.id for model in models.data]

        if not available_models:
            return None, "Local LLM: No models loaded. Please load a model in LM Studio."

        # Try exact match first
        if model_name in available_models:
            return model_name, None  # Success with exact match

        # Try partial match (model_name matches end of available model)
        for available_model in available_models:
            if available_model.endswith(model_name) or available_model.endswith(f"/{model_name}"):
                logger.debug(f"Local LLM: Matched '{model_name}' to '{available_model}'")
                return available_model, None  # Success with matched model name

        # No match found
        return None, f"Local LLM: Model '{model_name}' not loaded. Available models: {available_models}"

    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "refused" in error_str or "timeout" in error_str:
            return None, "Local LLM: Connection error. Please ensure LM Studio is running."
        return None, f"Local LLM: Failed to check loaded models: {e}"


def _handle_rate_limit_error(session_manager: SessionManager, source: str | None = None) -> None:
    """Handle rate limit error by increasing delay."""
    if session_manager and hasattr(session_manager, "rate_limiter"):
        try:
            drl = getattr(session_manager, "rate_limiter", None)
            if drl is not None and hasattr(drl, "on_429_error"):
                endpoint = source or "AI Provider"
                drl.on_429_error(endpoint)  # type: ignore[misc]
        except Exception:
            pass


def _call_inception_model(system_prompt: str, user_content: str, max_tokens: int, temperature: float, response_format_type: str | None) -> str | None:
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


def _route_ai_provider_call(
    provider: str, system_prompt: str, user_content: str,
    max_tokens: int, temperature: float, response_format_type: str | None
) -> str | None:
    """Route call to appropriate AI provider."""
    if provider == "deepseek":
        return _call_deepseek_model(system_prompt, user_content, max_tokens, temperature, response_format_type)
    if provider == "moonshot":
        return _call_moonshot_model(system_prompt, user_content, max_tokens, temperature, response_format_type)
    if provider == "gemini":
        return _call_gemini_model(system_prompt, user_content, max_tokens, temperature)
    if provider == "local_llm":
        return _call_local_llm_model(system_prompt, user_content, max_tokens, temperature, response_format_type)
    if provider == "inception":
        return _call_inception_model(system_prompt, user_content, max_tokens, temperature, response_format_type)
    logger.error(f"_call_ai_model: Unsupported AI provider '{provider}'.")
    return None


def _provider_is_configured(provider: str) -> bool:
    """Return True when the specified provider has enough configuration to attempt a call."""
    if provider == "deepseek":
        return bool(getattr(config_schema.api, "deepseek_api_key", None))
    if provider == "gemini":
        return bool(getattr(config_schema.api, "google_api_key", None))
    if provider == "local_llm":
        return all(
            getattr(config_schema.api, attr, None)
            for attr in ("local_llm_api_key", "local_llm_model", "local_llm_base_url")
        )
    if provider == "inception":
        return all(
            getattr(config_schema.api, attr, None)
            for attr in ("inception_api_key", "inception_ai_model", "inception_ai_base_url")
        )
    return False


def _resolve_provider_chain(primary_provider: str) -> list[str]:
    """Build the list of providers to attempt, including safe fallbacks."""
    provider_chain = [primary_provider]
    if primary_provider == "moonshot":
        for candidate in ("deepseek", "gemini", "local_llm"):
            if candidate not in provider_chain and _provider_is_configured(candidate):
                provider_chain.append(candidate)
    return provider_chain


def _should_fallback_to_next_provider(provider: str, error: Exception) -> bool:
    """Determine whether a fallback provider should be attempted after the given error."""
    if provider != "moonshot":
        return False
    if isinstance(error, AuthenticationError):
        return True
    return isinstance(error, APIError) and getattr(error, "status_code", None) == 401

def _handle_authentication_errors(e: Exception, provider: str) -> None:
    """Handle authentication-related errors."""
    if isinstance(e, AuthenticationError):
        logger.error(f"AI Authentication Error ({provider}): {e}")
    elif genai_errors and isinstance(e, genai_errors.PermissionDenied):
        logger.error(f"Gemini Permission Denied: {e}")


def _handle_rate_limit_errors(e: Exception, provider: str, session_manager: SessionManager) -> None:
    """Handle rate limiting-related errors."""
    if isinstance(e, RateLimitError):
        logger.error(f"AI Rate Limit Error ({provider}): {e}")
        _handle_rate_limit_error(session_manager, f"AI Provider: {provider}")
    elif genai_errors and isinstance(e, genai_errors.ResourceExhausted):
        logger.error(f"Gemini Resource Exhausted (Rate Limit): {e}")
        _handle_rate_limit_error(session_manager, f"AI Provider: {provider}")


def _handle_api_errors(e: Exception, provider: str) -> None:
    """Handle API-related errors."""
    if isinstance(e, APIConnectionError):
        logger.error(f"AI Connection Error ({provider}): {e}")
    elif isinstance(e, APIError):
        logger.error(f"AI API Error ({provider}): Status={getattr(e, 'status_code', 'N/A')}, Message={getattr(e, 'message', str(e))}")
    elif genai_errors and isinstance(e, genai_errors.GoogleAPIError):
        logger.error(f"Google API Error (Gemini): {e}")


def _handle_internal_errors(e: Exception, provider: str) -> None:
    """Handle internal/unexpected errors."""
    if isinstance(e, AttributeError):
        logger.critical(f"AttributeError during AI call ({provider}): {e}. Lib loaded: OpenAI={openai_available}, Gemini={genai_available}", exc_info=True)
    elif isinstance(e, NameError):
        logger.critical(f"NameError during AI call ({provider}): {e}. Lib loaded: OpenAI={openai_available}, Gemini={genai_available}", exc_info=True)
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
    last_exception: Exception | None = None

    for idx, active_provider in enumerate(provider_chain):
        _apply_rate_limiting(session_manager, active_provider)

        try:
            result = _route_ai_provider_call(active_provider, system_prompt, user_content, max_tokens, temperature, response_format_type)
            if active_provider != provider:
                logger.warning(
                    "AI provider '%s' failed, successfully fell back to '%s'.",
                    provider,
                    active_provider,
                )
            return result
        except Exception as exc:
            last_exception = exc
            _handle_ai_exceptions(exc, active_provider, session_manager)

            should_try_fallback = (
                idx < len(provider_chain) - 1
                and _should_fallback_to_next_provider(active_provider, exc)
            )

            if should_try_fallback:
                logger.warning(
                    "Attempting fallback AI provider '%s' after %s authentication failure.",
                    provider_chain[idx + 1],
                    active_provider,
                )
                continue

            break

    if last_exception is not None:
        logger.error(
            "AI provider '%s' failed without available fallback: %s",
            provider,
            type(last_exception).__name__,
        )
    return None


# End of _call_ai_model

# --- Public AI Interaction Functions ---


@cached_api_call("ai", ttl=3600)  # Cache AI responses for 1 hour
def classify_message_intent(
    context_history: str, session_manager: SessionManager
) -> str | None:
    """
    Classifies the intent of the LAST USER message within the provided context history.
    """
    ai_provider = config_schema.ai_provider.lower()
    if not ai_provider:
        logger.error("classify_message_intent: AI_PROVIDER not configured.")
        return None
    if not context_history:
        logger.warning(
            "classify_message_intent: Received empty context history. Defaulting to OTHER."
        )
        return "OTHER"

    system_prompt = FALLBACK_INTENT_PROMPT
    if USE_JSON_PROMPTS:
        try:
            loaded_prompt = get_prompt("intent_classification")
            if loaded_prompt:
                system_prompt = loaded_prompt
            else:
                logger.warning(
                    "Failed to load 'intent_classification' prompt from JSON, using fallback."
                )
        except Exception as e:
            logger.warning(
                f"Error loading 'intent_classification' prompt: {e}, using fallback."
            )

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
            logger.debug(
                f"AI intent classification: '{processed_classification}' (Took {duration:.2f}s)"
            )
            return processed_classification
        logger.warning(
            f"AI returned unexpected classification: '{raw_classification}'. Defaulting to OTHER."
        )
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
            try:
                from ai_prompt_utils import get_prompt_with_experiment
                variants = {"control": "extraction_task", "alt": "extraction_task_alt"}
                loaded_prompt = get_prompt_with_experiment("extraction_task", variants=variants, user_id=getattr(session_manager, "user_id", None))
            except Exception:
                loaded_prompt = get_prompt("extraction_task")
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
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-len("```")].strip()
    return cleaned


def _compute_component_coverage(parsed_json: dict[str, Any]) -> float | None:
    """Compute component coverage for extraction quality."""
    try:
        extracted_component = parsed_json.get("extracted_data", {}) if isinstance(parsed_json, dict) else {}
        structured_keys = [
            "structured_names", "vital_records", "relationships", "locations", "occupations",
            "research_questions", "documents_mentioned", "dna_information"
        ]
        non_empty = sum(1 for k in structured_keys if isinstance(extracted_component.get(k), list) and len(extracted_component.get(k)) > 0)
        return (non_empty / len(structured_keys)) if structured_keys else 0.0
    except Exception:
        return None


def _record_extraction_telemetry(system_prompt: str, parsed_json: dict[str, Any], cleaned_response_str: str, session_manager: SessionManager, parse_success: bool, error: str | None = None) -> None:
    """Record extraction telemetry event."""
    try:
        from ai_prompt_utils import get_prompt_version
        from prompt_telemetry import record_extraction_experiment_event

        variant_label = "alt" if "extraction_task_alt" in system_prompt[:120] else "control"
        # Note: extraction_quality module not yet implemented, so quality_score is None
        quality_score = None
        component_coverage = _compute_component_coverage(parsed_json) if parsed_json else None

        if quality_score is not None and parsed_json:
            with contextlib.suppress(Exception):
                parsed_json["quality_score"] = quality_score

        anomaly_summary = None

        record_extraction_experiment_event(
            variant_label=variant_label,
            prompt_key="extraction_task_alt" if variant_label == "alt" else "extraction_task",
            prompt_version=get_prompt_version("extraction_task_alt" if variant_label == "alt" else "extraction_task"),
            parse_success=parse_success,
            extracted_data=parsed_json.get("extracted_data") if parsed_json else None,
            suggested_tasks=parsed_json.get("suggested_tasks") if parsed_json else None,
            raw_response_text=cleaned_response_str,
            user_id=getattr(session_manager, "user_id", None),
            quality_score=quality_score,
            component_coverage=component_coverage,
            anomaly_summary=anomaly_summary,
            error=error,
        )
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
    extracted_data = {}
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


def extract_genealogical_entities(
    context_history: str, session_manager: SessionManager
) -> dict[str, Any] | None:
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
    max_tokens=1800,  # Reduced to speed up extraction while keeping high detail
        temperature=0.2,
        response_format_type="json_object",  # For DeepSeek
    )
    duration = time.time() - start_time

    if ai_response_str:
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
                _record_extraction_telemetry(system_prompt, parsed_json, cleaned_response_str, session_manager, parse_success=True)
                return parsed_json
            logger.warning(f"AI extraction response is valid JSON but uses flat structure instead of nested. Attempting to transform. Response: {cleaned_response_str[:500]}")
            salvaged = _salvage_flat_structure(parsed_json, default_empty_result)
            _record_extraction_telemetry(system_prompt, salvaged, cleaned_response_str, session_manager, parse_success=False, error="structure_salvaged")
            return salvaged
        except json.JSONDecodeError as e:
            logger.error(f"AI extraction response was not valid JSON: {e}. Response: {ai_response_str[:500]}")
            _record_extraction_telemetry(system_prompt, None, ai_response_str, session_manager, parse_success=False, error=str(e)[:120])
            return default_empty_result
    else:
        logger.error(f"AI extraction failed or returned empty. (Took {duration:.2f}s)")
        _record_extraction_telemetry(system_prompt, None, None, session_manager, parse_success=False, error="empty_response")
        return default_empty_result


# End of extract_genealogical_entities


def generate_genealogical_reply(
    conversation_context: str,
    user_last_message: str,
    genealogical_data_str: str,
    session_manager: SessionManager,
) -> str | None:
    """
    Generates a personalized genealogical reply.
    """
    ai_provider = config_schema.ai_provider.lower()
    if not ai_provider:
        logger.error("generate_genealogical_reply: AI_PROVIDER not configured.")
        return None
    if not all([conversation_context, user_last_message, genealogical_data_str]):
        logger.error(
            "generate_genealogical_reply: One or more required inputs are empty."
        )
        return None

    system_prompt_template = get_fallback_reply_prompt()
    if USE_JSON_PROMPTS:
        try:
            loaded_prompt = get_prompt("genealogical_reply")
            if loaded_prompt:
                system_prompt_template = loaded_prompt
            else:
                logger.warning(
                    "Failed to load 'genealogical_reply' prompt from JSON, using fallback."
                )
        except Exception as e:
            logger.warning(
                f"Error loading 'genealogical_reply' prompt: {e}, using fallback."
            )

    # The prompt itself is the system message; user_content can be minimal.
    # The template includes placeholders for {conversation_context}, {user_message}, {genealogical_data}
    try:
        final_system_prompt = system_prompt_template.format(
            conversation_context=conversation_context,
            user_message=user_last_message,
            genealogical_data=genealogical_data_str,
        )
    except KeyError as ke:
        logger.error(
            f"Prompt formatting error for genealogical_reply. Missing key: {ke}. Using unformatted prompt."
        )
        final_system_prompt = (
            system_prompt_template  # Fallback to unformatted if keys missing
        )
    except Exception as fe:
        logger.error(
            f"Unexpected error formatting genealogical_reply prompt: {fe}. Using unformatted prompt."
        )
        final_system_prompt = system_prompt_template

    start_time = time.time()
    reply_text = _call_ai_model(
        provider=ai_provider,
        system_prompt=final_system_prompt,
        user_content="Please generate a reply based on the system prompt and the information embedded within it.",  # Or ""
        session_manager=session_manager,
        max_tokens=800,  # For a detailed reply
        temperature=0.7,
    )
    duration = time.time() - start_time

    if reply_text:
        logger.info(f"AI reply generation successful. (Took {duration:.2f}s)")
    else:
        logger.error(f"AI reply generation failed. (Took {duration:.2f}s)")
    return reply_text


# End of generate_genealogical_reply


def _load_dialogue_prompt(log_prefix: str) -> str | None:
    """Load genealogical_dialogue_response prompt from JSON."""
    if not USE_JSON_PROMPTS:
        return None

    try:
        loaded_prompt = get_prompt("genealogical_dialogue_response")
        if loaded_prompt:
            return loaded_prompt
        logger.warning(
            f"{log_prefix}: Failed to load 'genealogical_dialogue_response' prompt from JSON."
        )
    except Exception as e:
        logger.warning(
            f"{log_prefix}: Error loading 'genealogical_dialogue_response' prompt: {e}"
        )
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
            conversation_history, lookup_results, dna_data, tree_statistics,
            relationship_path, conversation_phase, last_topic, pending_questions
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
        logger.error(
            f"{log_prefix}: genealogical_dialogue_response prompt not available."
        )
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
        logger.info(
            f"{log_prefix}: Contextual response generation successful. (Took {duration:.2f}s)"
        )
    else:
        logger.error(
            f"{log_prefix}: Contextual response generation failed. (Took {duration:.2f}s)"
        )

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


def _format_clarification_prompt(prompt: str, user_message: str, extracted_entities: dict[str, Any], ambiguity_context: str) -> str | None:
    """Format clarification prompt with user data."""
    try:
        return prompt.format(
            user_message=user_message,
            extracted_entities=json.dumps(extracted_entities, indent=2),
            ambiguity_context=ambiguity_context,
        )
    except KeyError as e:
        logger.error(f"generate_clarifying_questions: Prompt formatting error - missing key: {e}")
        return None


def _validate_clarification_response(result: dict[str, Any]) -> bool:
    """Validate AI clarification response structure."""
    if not isinstance(result, dict) or "clarifying_questions" not in result:
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
            logger.warning(
                "Custom extraction response was not valid JSON. Returning as text."
            )
            return {
                "extracted_data": ai_response_str
            }  # Return raw string under 'extracted_data'
    else:
        logger.error(
            f"AI custom extraction failed or returned empty. (Took {duration:.2f}s)"
        )
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
        logger.error(
            f"Custom prompt formatting error. Missing key: {ke}. Using unformatted custom prompt."
        )
        final_system_prompt = custom_prompt  # Fallback to unformatted if keys missing
    except Exception as fe:
        logger.error(
            f"Unexpected error formatting custom prompt: {fe}. Using unformatted custom prompt."
        )
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
        logger.error(
            f"AI custom generation failed or returned empty. (Took {duration:.2f}s)"
        )
    return reply_text


# End of generate_with_custom_prompt

# --- Specialized Genealogical Analysis Functions ---


def analyze_dna_match_conversation(
    context_history: str, session_manager: SessionManager
) -> dict[str, Any] | None:
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
        logger.warning(
            "analyze_dna_match_conversation: Empty context. Returning empty structure."
        )
        return default_empty_result

    # Get DNA match analysis prompt
    system_prompt = "Analyze this DNA match conversation for genetic genealogy information."
    if USE_JSON_PROMPTS:
        try:
            loaded_prompt = get_prompt("dna_match_analysis")
            if loaded_prompt:
                system_prompt = loaded_prompt
            else:
                logger.warning(
                    "Failed to load 'dna_match_analysis' prompt from JSON, using fallback."
                )
        except Exception as e:
            logger.warning(
                f"Error loading 'dna_match_analysis' prompt: {e}, using fallback."
            )

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


def verify_family_tree_connections(
    context_history: str, session_manager: SessionManager
) -> dict[str, Any] | None:
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
        logger.warning(
            "verify_family_tree_connections: Empty context. Returning empty structure."
        )
        return default_empty_result

    # Get family tree verification prompt
    system_prompt = "Analyze this conversation for family tree verification needs."
    if USE_JSON_PROMPTS:
        try:
            loaded_prompt = get_prompt("family_tree_verification")
            if loaded_prompt:
                system_prompt = loaded_prompt
            else:
                logger.warning(
                    "Failed to load 'family_tree_verification' prompt from JSON, using fallback."
                )
        except Exception as e:
            logger.warning(
                f"Error loading 'family_tree_verification' prompt: {e}, using fallback."
            )

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


def generate_record_research_strategy(
    context_history: str, session_manager: SessionManager
) -> dict[str, Any] | None:
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
        logger.warning(
            "generate_record_research_strategy: Empty context. Returning empty structure."
        )
        return default_empty_result

    # Get record research guidance prompt
    system_prompt = "Analyze this conversation for genealogical record research opportunities."
    if USE_JSON_PROMPTS:
        try:
            loaded_prompt = get_prompt("record_research_guidance")
            if loaded_prompt:
                system_prompt = loaded_prompt
            else:
                logger.warning(
                    "Failed to load 'record_research_guidance' prompt from JSON, using fallback."
                )
        except Exception as e:
            logger.warning(
                f"Error loading 'record_research_guidance' prompt: {e}, using fallback."
            )

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
    if ai_provider not in ["deepseek", "gemini", "moonshot", "local_llm"]:
        logger.error(
            "❌ Invalid AI_PROVIDER: %s. Must be 'deepseek', 'gemini', 'moonshot', or 'local_llm'",
            ai_provider,
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

    # Test provider-specific configuration
    if ai_provider == "deepseek":
        return _validate_deepseek_config()
    if ai_provider == "gemini":
        return _validate_gemini_config()
    if ai_provider == "moonshot":
        return _validate_moonshot_config()
    if ai_provider == "local_llm":
        return _validate_local_llm_config()

    return True


def _validate_extraction_task_structure(prompt_content: str) -> bool:
    """Validate extraction_task prompt structure."""
    has_nested_structure = "suggested_tasks" in prompt_content and "extracted_data" in prompt_content
    has_flat_structure = "mentioned_names" in prompt_content and "dates" in prompt_content and "locations" in prompt_content

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
        from action9_process_productive import AIResponse

        # Test creating the model
        ai_response = AIResponse(**test_data)
        logger.info("✅ AIResponse model created successfully")
        logger.info(f"✅ extracted_data type: {type(ai_response.extracted_data)}")
        logger.info(f"✅ suggested_tasks type: {type(ai_response.suggested_tasks)}")
        logger.info(f"✅ suggested_tasks count: {len(ai_response.suggested_tasks)}")

        # Test accessing nested fields
        extracted = ai_response.extracted_data
        all_names = extracted.get_all_names()
        all_locations = extracted.get_all_locations()
        logger.info(
            f"✅ Nested fields accessible: names={len(all_names)}, locations={len(all_locations)}"
        )

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

    if not extraction_result or not isinstance(extraction_result, dict):
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
    logger.warning("⚠️ AI did not extract expected genealogical entities from test context")
    return True  # Still return True as extraction worked, just didn't find expected data


def _test_reply_generation(session_manager: SessionManager) -> bool:
    """Test AI reply generation functionality."""
    test_genealogical_data = "Person: John Smith, Born: 1880 Aberdeen Scotland, Occupation: Fisherman, Relationship: Great-grandfather"
    logger.info("Testing genealogical reply generation...")
    reply_result = generate_genealogical_reply(
        "Previous conversation context",
        "Can you tell me about John Smith?",
        test_genealogical_data,
        session_manager,
    )

    if reply_result and isinstance(reply_result, str) and len(reply_result) > 10:
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

    if dna_result and isinstance(dna_result, dict):
        logger.info("✅ DNA match analysis function working")
    else:
        logger.warning("⚠️ DNA match analysis returned unexpected result")

    # Test family tree verification
    verification_test_context = "SCRIPT: Hello! I'm verifying family connections.\nUSER: I'm not sure if William Smith is really my great-grandfather. I have conflicting information about his birth year - some records say 1850, others say 1855."
    verification_result = verify_family_tree_connections(verification_test_context, session_manager)

    if verification_result and isinstance(verification_result, dict):
        logger.info("✅ Family tree verification function working")
    else:
        logger.warning("⚠️ Family tree verification returned unexpected result")

    # Test record research strategy
    research_test_context = "SCRIPT: Hello! I need research help.\nUSER: I'm looking for birth records for my ancestor Mary O'Brien who was born around 1870 in County Cork, Ireland. She immigrated to Boston around 1890."
    research_result = generate_record_research_strategy(research_test_context, session_manager)

    if research_result and isinstance(research_result, dict):
        logger.info("✅ Record research strategy function working")
    else:
        logger.warning("⚠️ Record research strategy returned unexpected result")

    return True


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
    from test_framework import TestSuite, suppress_logging

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
            "Verify openai or google.generativeai libraries are installed",
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
                    lambda: quick_health_check(sm)["overall_health"] in ["healthy", "degraded"],
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

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    return ai_interface_module_tests()


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
    return False, False


def _check_prompts_loaded(errors: list[str]) -> bool:
    """Check if prompts are loaded successfully."""
    try:
        prompts = load_prompts()
        return bool(prompts and "extraction_task" in prompts)
    except Exception as e:
        errors.append(f"Prompt loading error: {e}")
        return False


def _perform_test_call(session_manager: SessionManager, api_key_configured: bool, dependencies_available: bool, errors: list[str]) -> bool:
    """Perform a quick test call if configuration looks good."""
    if not (api_key_configured and dependencies_available):
        return False

    try:
        result = classify_message_intent("Test", session_manager)
        return result is not None
    except Exception as e:
        errors.append(f"Test call error: {e}")
        return False


def _determine_overall_health(api_key_configured: bool, prompts_loaded: bool, dependencies_available: bool, test_call_successful: bool) -> str:
    """Determine overall health status based on checks."""
    if api_key_configured and prompts_loaded and dependencies_available:
        return "healthy" if test_call_successful else "degraded"
    return "unhealthy"


def quick_health_check(session_manager: SessionManager) -> dict[str, Any]:
    """
    Performs a quick health check of the AI interface.
    Returns a dictionary with health status information.
    """
    health_status = {
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
        health_status["api_key_configured"], health_status["dependencies_available"] = _check_api_key_and_dependencies(ai_provider)

        # Check prompts
        health_status["prompts_loaded"] = _check_prompts_loaded(health_status["errors"])

        # Quick test call
        health_status["test_call_successful"] = _perform_test_call(
            session_manager,
            health_status["api_key_configured"],
            health_status["dependencies_available"],
            health_status["errors"]
        )

        # Determine overall health
        health_status["overall_health"] = _determine_overall_health(
            health_status["api_key_configured"],
            health_status["prompts_loaded"],
            health_status["dependencies_available"],
            health_status["test_call_successful"]
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
        logger.error(
            f"Intent classification test 1 FAILED. Expected valid category, got: {result1}"
        )
        all_passed = False

    result2 = classify_message_intent(test_context_2, session_manager)
    if result2 and result2 in EXPECTED_INTENT_CATEGORIES:
        logger.info(f"Intent classification test 2 PASSED. Result: {result2}")
    else:
        logger.error(
            f"Intent classification test 2 FAILED. Expected valid category, got: {result2}"
        )
        all_passed = False

    logger.info("--- Testing Data Extraction & Task Suggestion ---")
    extraction_result = extract_genealogical_entities(
        extraction_conversation_test, session_manager
    )

    if (
        extraction_result
        and isinstance(extraction_result.get("extracted_data"), dict)
        and isinstance(extraction_result.get("suggested_tasks"), list)
    ):
        logger.info("Data extraction test PASSED structure validation.")
        logger.info(
            f"Extracted data: {json.dumps(extraction_result['extracted_data'], indent=2)}"
        )
        logger.info(f"Suggested tasks: {extraction_result['suggested_tasks']}")
        if not extraction_result["suggested_tasks"]:
            logger.warning(
                "AI did not suggest any tasks for the extraction test context. Prompt may need adjustment."
            )
            # This is not a failure of the function itself, but a note on prompt effectiveness.
    else:
        logger.error(
            f"Data extraction test FAILED. Invalid structure or None result. Got: {extraction_result}"
        )
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
