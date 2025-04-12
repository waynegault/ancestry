# ai_interface.py
# V0.2: Added json import for extract_and_suggest_tasks

"""
Provides an interface for interacting with external AI models for message classification.
Reads configuration from config_instance and uses the appropriate client library.
Supports DeepSeek (via OpenAI library) and Google Gemini (using GenerativeModel).
Includes diagnostic checks for Gemini library attributes.
"""

import sys
import logging
import time
from typing import Optional, Dict, List, Any
import json  # <<< ADD THIS IMPORT

# --- Third-party Imports ---
# Attempt OpenAI import for DeepSeek
try:
    from openai import (
        OpenAI,
        APIConnectionError,
        RateLimitError,
        AuthenticationError,
        APIError,
    )
except ImportError:
    OpenAI = None
    APIConnectionError = None
    RateLimitError = None
    AuthenticationError = None
    APIError = None
# Attempt Google Gemini import (using GenerativeModel style)
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions

    genai_available = True
    # Use logging directly here since logger might not be fully set up yet
    # No FATAL logging here, just flag availability
    if not hasattr(genai, "configure"):
        genai_available = False
    if not hasattr(genai, "GenerativeModel"):
        genai_available = False
    if not google_exceptions:
        genai_available = False
except ImportError:
    genai = None
    google_exceptions = None
    genai_available = False

# --- Local Application Imports ---
from config import config_instance
from utils import SessionManager

# Initialize logging
logger = logging.getLogger("logger")

# --- System Prompt & Categories (Context-Aware) ---
SYSTEM_PROMPT = """You are an AI assistant analyzing conversation histories from a genealogy website messaging system. The history alternates between 'SCRIPT' (automated messages from me) and 'USER' (replies from the DNA match).

Analyze the entire provided conversation history, paying close attention to the **last message sent by the USER**. Based on the full context, determine the primary intent **of that final USER message**.

Respond ONLY with one of the following single-word categories:

DESIST: The user's final message, in the context of the conversation, explicitly asks to stop receiving messages or indicates they are blocking the sender.
UNINTERESTED: The user's final message, in context, politely declines further contact, states they cannot help, don't have time, are not knowledgeable, or shows clear lack of engagement/desire to continue.
PRODUCTIVE: The user's final message, in context, provides helpful information, asks relevant questions, confirms relationships, expresses clear interest in collaborating, or shares tree info.
OTHER: The user's final message, in context, does not clearly fall into the DESIST, UNINTERESTED, or PRODUCTIVE categories.

Output only the single category word."""
EXPECTED_CATEGORIES = {"DESIST", "UNINTERESTED", "PRODUCTIVE", "OTHER"}

# --- Extraction Prompt ---
EXTRACTION_TASK_SYSTEM_PROMPT = """You are an AI assistant analyzing conversation histories from a genealogy website messaging system. The history alternates between 'SCRIPT' (automated messages from me) and 'USER' (replies from the DNA match).

Analyze the entire provided conversation history, focusing on information shared by the USER, especially in their latest messages.

Your goal is twofold:
1.  **Extract Key Genealogical Information:** Identify and extract specific entities mentioned by the USER. Focus on names (full names preferred), specific dates or date ranges (e.g., "abt 1880", "1912-1915"), specific locations (towns, counties, countries), and explicitly stated family relationships (e.g., "my grandfather", "her sister"). Also capture any key facts stated by the user (e.g., "immigrated via Liverpool", "worked in coal mines").
2.  **Suggest Actionable Follow-up Tasks:** Based ONLY on the information provided in the conversation history, suggest 2-4 concrete, actionable research tasks for the recipient of this analysis (the person running the script, referred to as 'Me' or 'Wayne'). Tasks should be specific and help verify or extend the information provided by the USER. Examples: "Check [Year] census for [Name] in [Location]", "Search ship manifests for [Name] arriving [Port] around [Date]", "Compare shared matches with [Match Name]", "Look for [Event record] for [Name] in [Place]". Avoid generic tasks like "Research more".

Format your response STRICTLY as a JSON object containing two keys:
- "extracted_data": An object containing lists of strings for the extracted entities: "mentioned_names", "mentioned_locations", "mentioned_dates", "potential_relationships", "key_facts". If no entities of a certain type are found, provide an empty list [].
- "suggested_tasks": A list of strings, where each string is an actionable task. Provide an empty list [] if no specific tasks can be suggested.

Example JSON Output:
{
  "extracted_data": {
    "mentioned_names": ["John Smith", "Mary Anne Jones"],
    "mentioned_locations": ["Glasgow", "County Cork", "Liverpool"],
    "mentioned_dates": ["abt 1880", "1912"],
    "potential_relationships": ["Grandfather"],
    "key_facts": ["Immigrated via Liverpool", "Worked in coal mines"]
  },
  "suggested_tasks": [
    "Check 1881 Scotland Census for John Smith in Glasgow.",
    "Search immigration records for Mary Anne Jones arriving Liverpool around 1910-1915.",
    "Compare shared matches between Wayne and Mary Anne Jones.",
    "Review County Cork birth records for Jones around 1880."
  ]
}

Ensure the output is ONLY the JSON object, with no introductory text or explanation.
"""


def classify_message_intent(
    context_history: str, session_manager: SessionManager  # Expects formatted history
) -> Optional[str]:
    """
    Classifies the intent of the LAST USER message within the provided context history
    using the configured AI provider.

    Args:
        context_history: The formatted conversation history string.
        session_manager: The SessionManager instance (used for rate limiting).

    Returns:
        A classification string ("DESIST", "UNINTERESTED", "PRODUCTIVE", "OTHER")
        or None if classification fails or an error occurs.
    """
    ai_provider = config_instance.AI_PROVIDER
    if not session_manager or not hasattr(session_manager, "dynamic_rate_limiter"):
        logger.error("Invalid SM/rate limiter.")
        return None
    if not ai_provider:
        logger.error("AI_PROVIDER not configured.")
        return None

    # Rate Limiting
    wait_time = session_manager.dynamic_rate_limiter.wait()
    if wait_time > 0.1:
        logger.debug(f"AI API rate limit wait: {wait_time:.2f}s")

    classification_result: Optional[str] = None
    start_time = time.time()

    try:
        if ai_provider == "deepseek":
            # --- DeepSeek Logic ---
            if OpenAI is None:
                logger.error("OpenAI library not installed.")
                return None
            api_key = config_instance.DEEPSEEK_API_KEY
            model = config_instance.DEEPSEEK_AI_MODEL
            base_url = config_instance.DEEPSEEK_AI_BASE_URL
            if not all([api_key, model, base_url]):
                logger.error("DeepSeek config missing.")
                return None
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.debug(f"Attempting DeepSeek classification (Model: {model})...")
            openai_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context_history},
                ],
                stream=False,
                max_tokens=20,
                temperature=0.1,
            )
            if openai_response.choices and openai_response.choices[0].message:
                raw_classification = openai_response.choices[0].message.content.strip()
                if raw_classification in EXPECTED_CATEGORIES:
                    classification_result = raw_classification
                else:
                    logger.warning(
                        f"DeepSeek unexpected classification: '{raw_classification}'"
                    )
                    classification_result = "OTHER"
            else:
                logger.error("Invalid response structure from DeepSeek API.")

        elif ai_provider == "gemini":
            # --- Google Gemini Logic ---
            if not genai_available:
                logger.error("google.generativeai library/attributes unavailable.")
                return None
            api_key = config_instance.GOOGLE_API_KEY
            model_name = config_instance.GOOGLE_AI_MODEL
            if not api_key or not model_name:
                logger.error("Gemini config missing.")
                return None
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            logger.debug(f"Attempting Gemini classification (Model: {model_name})...")
            generation_config = {
                "candidate_count": 1,
                "max_output_tokens": 20,
                "temperature": 0.1,
            }
            # Combine system prompt and context history for Gemini model prompt
            prompt_content = (
                f"{SYSTEM_PROMPT}\n\nConversation History:\n{context_history}"
            )
            response = model.generate_content(
                prompt_content, generation_config=generation_config
            )

            # *** Corrected Syntax for block reason handling ***
            if not response.candidates:
                block_reason = "Unknown"
                try:
                    # Check prompt_feedback exists before accessing block_reason
                    if (
                        hasattr(response, "prompt_feedback")
                        and response.prompt_feedback
                    ):
                        block_reason = response.prompt_feedback.block_reason.name
                except Exception:
                    pass  # Ignore errors trying to get block reason details
                logger.warning(f"Gemini response blocked/empty. Reason: {block_reason}")
                classification_result = "OTHER (Blocked/Empty)"
            else:
                raw_classification = response.text.strip()
                if raw_classification in EXPECTED_CATEGORIES:
                    classification_result = raw_classification
                else:
                    logger.warning(
                        f"Gemini unexpected classification: '{raw_classification}'"
                    )
                    classification_result = "OTHER"
            # *** End Syntax Correction ***
        else:
            logger.error(f"Unsupported AI_PROVIDER: {ai_provider}")

    # --- Consolidated Error Handling ---
    except AuthenticationError as e:
        logger.error(f"AI Auth Error ({ai_provider}): {e}")
    except RateLimitError as e:
        logger.error(f"AI Rate Limit ({ai_provider}): {e}")
        session_manager.dynamic_rate_limiter.increase_delay()
    except APIConnectionError as e:
        logger.error(f"AI Connection Error ({ai_provider}): {e}")
    except APIError as e:
        logger.error(f"AI API Error ({ai_provider}): {e.status_code} - {e.message}")
    except google_exceptions.PermissionDenied as e:
        logger.error(f"AI Permission Denied (Gemini): {e}")
    except google_exceptions.ResourceExhausted as e:
        logger.error(f"AI Rate Limit (Gemini): {e}")
        session_manager.dynamic_rate_limiter.increase_delay()
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Google API Error (Gemini): {e}")
    except AttributeError as ae:
        logger.critical(f"AttributeError AI ({ai_provider}): {ae}", exc_info=True)
    except Exception as e:
        logger.error(
            f"Unexpected AI classification error ({ai_provider}): {type(e).__name__} - {e}",
            exc_info=True,
        )

    duration = time.time() - start_time
    if not classification_result:
        logger.error(f"AI classification failed ({ai_provider}). Took {duration:.2f}s")
    # else: logger.debug(f"AI classification: '{classification_result}' (Took {duration:.2f}s)") # Verbose

    return classification_result


# End of classify_message_intent


def extract_and_suggest_tasks(
    context_history: str, session_manager: SessionManager
) -> Optional[Dict[str, Any]]:
    """
    Calls the configured AI model to extract genealogical entities and suggest
    follow-up tasks based on the provided conversation history.

    Args:
        context_history: The formatted conversation history string.
        session_manager: The SessionManager instance (used for rate limiting).

    Returns:
        A dictionary containing 'extracted_data' and 'suggested_tasks' if successful
        and the response is valid JSON, otherwise None.
    """
    ai_provider = config_instance.AI_PROVIDER
    if not session_manager or not hasattr(session_manager, "dynamic_rate_limiter"):
        logger.error("Invalid SM/rate limiter for extraction.")
        return None
    if not ai_provider:
        logger.error("AI_PROVIDER not configured for extraction.")
        return None
    if not context_history:
        logger.warning("Cannot call AI for extraction: Empty context history provided.")
        return None  # Return None if context is empty

    # Rate Limiting
    wait_time = session_manager.dynamic_rate_limiter.wait()
    if wait_time > 0.1:
        logger.debug(f"AI Extraction API rate limit wait: {wait_time:.2f}s")

    extraction_result: Optional[Dict[str, Any]] = None
    start_time = time.time()
    max_tokens_extraction = 500  # Allow more tokens for JSON output

    try:
        if ai_provider == "deepseek":
            # --- DeepSeek Logic ---
            if OpenAI is None:
                logger.error("OpenAI library not installed.")
                return None
            api_key = config_instance.DEEPSEEK_API_KEY
            model = config_instance.DEEPSEEK_AI_MODEL
            base_url = config_instance.DEEPSEEK_AI_BASE_URL
            if not all([api_key, model, base_url]):
                logger.error("DeepSeek config missing.")
                return None

            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.debug(
                f"Attempting DeepSeek extraction/suggestion (Model: {model})..."
            )
            openai_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACTION_TASK_SYSTEM_PROMPT},
                    {"role": "user", "content": context_history},
                ],
                stream=False,
                max_tokens=max_tokens_extraction,
                temperature=0.2,  # Slightly higher temp might help creativity for tasks?
                response_format={"type": "json_object"},  # Request JSON mode
            )
            if openai_response.choices and openai_response.choices[0].message:
                raw_response_content = openai_response.choices[0].message.content
                try:
                    # <<< --- Use json.loads (requires import json) --- >>>
                    parsed_json = json.loads(raw_response_content)
                    # Basic validation of structure
                    if (
                        isinstance(parsed_json, dict)
                        and "extracted_data" in parsed_json
                        and "suggested_tasks" in parsed_json
                        and isinstance(parsed_json["extracted_data"], dict)
                        and isinstance(parsed_json["suggested_tasks"], list)
                    ):
                        extraction_result = parsed_json
                    else:
                        logger.warning(
                            f"DeepSeek response is valid JSON but missing expected structure: {raw_response_content}"
                        )
                except json.JSONDecodeError as e:
                    logger.error(
                        f"DeepSeek response was not valid JSON: {e}\nContent: {raw_response_content}"
                    )
            else:
                logger.error(
                    "Invalid response structure from DeepSeek API (Extraction)."
                )

        elif ai_provider == "gemini":
            # --- Google Gemini Logic ---
            if not genai_available:
                logger.error("google.generativeai library/attributes unavailable.")
                return None
            api_key = config_instance.GOOGLE_API_KEY
            model_name = config_instance.GOOGLE_AI_MODEL
            if not api_key or not model_name:
                logger.error("Gemini config missing.")
                return None

            # Configure GenAI (consider doing this once globally if possible)
            try:
                genai.configure(api_key=api_key)
            except Exception as config_err:
                logger.error(f"Failed to configure Gemini API: {config_err}")
                return None

            model = genai.GenerativeModel(model_name)
            logger.debug(
                f"Attempting Gemini extraction/suggestion (Model: {model_name})..."
            )
            generation_config = {
                "candidate_count": 1,
                "max_output_tokens": max_tokens_extraction,
                "temperature": 0.2,
                "response_mime_type": "application/json",  # Request JSON output
            }
            # Combine system prompt and context history
            prompt_content = f"{EXTRACTION_TASK_SYSTEM_PROMPT}\n\nConversation History:\n{context_history}"

            try:
                response = model.generate_content(
                    prompt_content, generation_config=generation_config
                )

                if not response.candidates:
                    block_reason = "Unknown"
                    try:
                        if (
                            hasattr(response, "prompt_feedback")
                            and response.prompt_feedback
                        ):
                            block_reason = response.prompt_feedback.block_reason.name
                    except Exception:
                        pass
                    logger.warning(
                        f"Gemini extraction response blocked/empty. Reason: {block_reason}"
                    )
                else:
                    raw_response_content = (
                        response.text
                    )  # Gemini API should return JSON string in .text
                    try:
                        # Gemini sometimes wraps the JSON in ```json ... ``` markdown, attempt removal
                        if raw_response_content.startswith("```json"):
                            raw_response_content = (
                                raw_response_content.strip("```json").strip("`").strip()
                            )

                        # <<< --- Use json.loads (requires import json) --- >>>
                        parsed_json = json.loads(raw_response_content)
                        # Basic validation of structure
                        if (
                            isinstance(parsed_json, dict)
                            and "extracted_data" in parsed_json
                            and "suggested_tasks" in parsed_json
                            and isinstance(parsed_json["extracted_data"], dict)
                            and isinstance(parsed_json["suggested_tasks"], list)
                        ):
                            extraction_result = parsed_json
                        else:
                            logger.warning(
                                f"Gemini response is valid JSON but missing expected structure: {raw_response_content}"
                            )
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Gemini response was not valid JSON: {e}\nContent: {raw_response_content}"
                        )
            except google_exceptions.GoogleAPIError as google_api_err:
                logger.error(f"Gemini API Error during generation: {google_api_err}")
                # Map specific errors if needed (e.g., PermissionDenied, ResourceExhausted)
                if isinstance(google_api_err, google_exceptions.ResourceExhausted):
                    session_manager.dynamic_rate_limiter.increase_delay()
                # Fall through to general error handling
                raise google_api_err  # Re-raise to be caught by outer handler

        else:
            logger.error(f"Unsupported AI_PROVIDER for extraction: {ai_provider}")

    # --- Consolidated Error Handling (Same as classify_message_intent) ---
    except AuthenticationError as e:
        logger.error(f"AI Auth Error ({ai_provider}): {e}")
    except RateLimitError as e:
        logger.error(f"AI Rate Limit ({ai_provider}): {e}")
        session_manager.dynamic_rate_limiter.increase_delay()
    except APIConnectionError as e:
        logger.error(f"AI Connection Error ({ai_provider}): {e}")
    except APIError as e:
        logger.error(f"AI API Error ({ai_provider}): {e.status_code} - {e.message}")
    except google_exceptions.PermissionDenied as e:
        logger.error(f"AI Permission Denied (Gemini): {e}")
    except google_exceptions.ResourceExhausted as e:
        logger.error(f"AI Rate Limit (Gemini): {e}")
        session_manager.dynamic_rate_limiter.increase_delay()
    except google_exceptions.GoogleAPIError as e:
        logger.error(
            f"Google API Error (Gemini): {e}"
        )  # Already caught above, but keep for outer handler
    except AttributeError as ae:
        logger.critical(f"AttributeError AI ({ai_provider}): {ae}", exc_info=True)
    # <<< --- ADD NameError Handling --- >>>
    except NameError as ne:
        logger.critical(
            f"NameError in AI function ({ai_provider}): {ne}. Missing import?",
            exc_info=True,
        )
    # <<< --- END ADD --- >>>
    except Exception as e:
        logger.error(
            f"Unexpected AI extraction error ({ai_provider}): {type(e).__name__} - {e}",
            exc_info=True,
        )
    # --- End Error Handling ---

    duration = time.time() - start_time
    if not extraction_result:
        logger.error(
            f"AI extraction/suggestion failed ({ai_provider}). Took {duration:.2f}s"
        )
    else:
        logger.debug(f"AI extraction/suggestion completed. Took {duration:.2f}s")

    return extraction_result


# End of extract_and_suggest_tasks

# End of ai_interface.py
