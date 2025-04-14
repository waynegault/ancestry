# ai_interface.py

"""
ai_interface.py - AI Model Interaction Layer

Provides functions to interact with external AI models (configured via config.py)
for tasks like message intent classification and genealogical data extraction.
Supports DeepSeek (OpenAI compatible) and Google Gemini Pro. Includes error
handling and rate limiting integration with SessionManager.
"""

# --- Standard library imports ---
import json  # For parsing JSON responses
import logging
import sys  # Potentially needed for exit on critical failures
import time
from typing import Any, Dict, List, Optional

# --- Third-party Imports ---
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
    OpenAI = None
    APIConnectionError = RateLimitError = AuthenticationError = APIError = (
        None  # Set error types to None
    )
    openai_available = False
    logging.warning("OpenAI library not found. DeepSeek functionality disabled.")

# Attempt Google Gemini import
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions

    genai_available = True
    # Perform quick checks for necessary attributes/classes
    if not hasattr(genai, "configure") or not hasattr(genai, "GenerativeModel"):
        genai_available = False
        logging.warning("Google GenerativeAI library structure seems incomplete.")
    if not google_exceptions:
        genai_available = False
        logging.warning("Google API Core exceptions not found.")
except ImportError:
    genai = None
    google_exceptions = None
    genai_available = False
    logging.warning(
        "Google GenerativeAI library not found. Gemini functionality disabled."
    )

# --- Local Application Imports ---
from config import config_instance  # Use configured instance
from utils import SessionManager  # For rate limiting access
from logging_config import logger  # Use configured logger

# --- Constants and Prompts ---

# System prompt for Action 7 (Intent Classification)
# Focuses on the intent of the *last USER message* within the conversation context.
SYSTEM_PROMPT_INTENT = """You are an AI assistant analyzing conversation histories from a genealogy website messaging system. The history alternates between 'SCRIPT' (automated messages from me) and 'USER' (replies from the DNA match).

Analyze the entire provided conversation history, interpreting the **last message sent by the USER** *within the context of the entire conversation history* provided below. Determine the primary intent of that final USER message.

Respond ONLY with one of the following single-word categories:

DESIST: The user's final message, in context, explicitly asks to stop receiving messages or indicates they are blocking the sender.
UNINTERESTED: The user's final message, in context, politely declines further contact, states they cannot help, don't have time, are not knowledgeable, shows clear lack of engagement/desire to continue, or replies with very short, non-committal answers that don't advance the genealogical discussion after specific requests for information.
PRODUCTIVE: The user's final message, in context, provides helpful genealogical information (names, dates, places, relationships), asks relevant clarifying questions, confirms relationships, expresses clear interest in collaborating, or shares tree info/invites.
OTHER: The user's final message, in context, does not clearly fall into the DESIST, UNINTERESTED, or PRODUCTIVE categories. Examples include purely social pleasantries, unrelated questions, ambiguous statements, or messages containing only attachments/links without explanatory text.

CRITICAL: Your entire response must be only one of the four category words (DESIST, UNINTERESTED, PRODUCTIVE, OTHER)."""
EXPECTED_INTENT_CATEGORIES = {"DESIST", "UNINTERESTED", "PRODUCTIVE", "OTHER"}

# System prompt for Action 9 (Data Extraction & Task Suggestion)
# Focuses on extracting specific genealogical entities and suggesting *actionable* research tasks.
EXTRACTION_TASK_SYSTEM_PROMPT = """You are an AI assistant analyzing conversation histories from a genealogy website messaging system. The history alternates between 'SCRIPT' (automated messages from me) and 'USER' (replies from the DNA match).

Analyze the entire provided conversation history, focusing on information shared by the USER, especially in their latest messages.

Your goal is twofold:
1.  **Extract Key Genealogical Information:** Identify and extract specific entities mentioned by the USER. Focus ONLY on names (full names preferred), specific dates or date ranges (e.g., "abt 1880", "1912-1915"), specific locations (towns, counties, countries), and explicitly stated family relationships (e.g., "my grandfather", "her sister"). Also capture any key facts stated by the user (e.g., "immigrated via Liverpool", "worked in coal mines"). Only include entities explicitly mentioned by the USER. Do not infer or add information not present.
2.  **Suggest Actionable Follow-up Tasks:** Based ONLY on the information provided in the conversation history, suggest 2-4 concrete, actionable research tasks for the recipient of this analysis ('Me'/'Wayne'). Tasks MUST be directly based on information, questions, or ambiguities present *only* in the provided conversation history. Tasks should be specific and help verify or extend the information provided by the USER. Examples: "Check [Year] census for [Name] in [Location]", "Search ship manifests for [Name] arriving [Port] around [Date]", "Compare shared matches with [Match Name]", "Look for [Event record] for [Name] in [Place]". Avoid generic tasks like "Research more".

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

CRITICAL: Ensure the output is ONLY the JSON object, starting with `{` and ending with `}`, with no introductory text, explanation, or markdown formatting.
"""

# --- Core AI Interaction Functions ---


def classify_message_intent(
    context_history: str, session_manager: SessionManager
) -> Optional[str]:
    """
    Classifies the intent of the LAST USER message within the provided context history
    using the configured AI provider (DeepSeek or Gemini).

    Args:
        context_history: The formatted conversation history string.
        session_manager: The active SessionManager instance (for rate limiting).

    Returns:
        A classification string ("DESIST", "UNINTERESTED", "PRODUCTIVE", "OTHER")
        or None if classification fails or an error occurs.
    """
    # Step 1: Validate inputs and configuration
    ai_provider = config_instance.AI_PROVIDER
    if not session_manager or not hasattr(session_manager, "dynamic_rate_limiter"):
        logger.error(
            "classify_message_intent: Invalid SessionManager or missing rate limiter."
        )
        return None
    if not ai_provider:
        logger.error("classify_message_intent: AI_PROVIDER not configured.")
        return None
    if not context_history:
        logger.warning(
            "classify_message_intent: Received empty context history. Cannot classify."
        )
        return "OTHER"  # Return OTHER for empty context

    # Step 2: Apply Rate Limiting
    wait_time = session_manager.dynamic_rate_limiter.wait()
    # Optional: Log if wait time was significant
    # if wait_time > 0.1: logger.debug(f"AI Intent API rate limit wait: {wait_time:.2f}s")

    # Step 3: Initialize result and timer
    classification_result: Optional[str] = None
    start_time = time.time()

    # Step 4: Call the appropriate AI provider
    try:
        if ai_provider == "deepseek":
            # --- DeepSeek/OpenAI Compatible API Call ---
            if not openai_available or OpenAI is None:  # Check both flag and object
                logger.error("classify_message_intent: OpenAI library not available.")
                return None
            # Load config
            api_key = config_instance.DEEPSEEK_API_KEY
            model = config_instance.DEEPSEEK_AI_MODEL
            base_url = config_instance.DEEPSEEK_AI_BASE_URL
            if not all([api_key, model, base_url]):
                logger.error(
                    "classify_message_intent: DeepSeek configuration incomplete (API Key, Model, Base URL)."
                )
                return None
            # Create client and make request
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.debug(f"Calling DeepSeek Intent Classification (Model: {model})...")
            openai_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_INTENT},
                    {"role": "user", "content": context_history},
                ],
                stream=False,  # Request non-streaming response
                max_tokens=20,  # Expecting single word response
                temperature=0.1,  # Low temperature for deterministic classification
            )
            # Process response
            if openai_response.choices and openai_response.choices[0].message:
                raw_classification = (
                    openai_response.choices[0].message.content.strip().upper()
                )  # Ensure uppercase
                if raw_classification in EXPECTED_INTENT_CATEGORIES:
                    classification_result = raw_classification
                else:
                    logger.warning(
                        f"DeepSeek returned unexpected classification: '{raw_classification}'. Defaulting to OTHER."
                    )
                    classification_result = "OTHER"
            else:
                logger.error(
                    "Invalid response structure received from DeepSeek API (Intent)."
                )

        elif ai_provider == "gemini":
            # --- Google Gemini API Call ---
            if not genai_available:
                logger.error(
                    "classify_message_intent: Google GenerativeAI library not available or incomplete."
                )
                return None
            # Load config
            api_key = config_instance.GOOGLE_API_KEY
            model_name = config_instance.GOOGLE_AI_MODEL
            if not api_key or not model_name:
                logger.error(
                    "classify_message_intent: Gemini configuration incomplete (API Key, Model Name)."
                )
                return None
            # Configure API and model
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
            except Exception as gemini_setup_err:
                logger.error(
                    f"Failed to configure/initialize Gemini model: {gemini_setup_err}"
                )
                return None
            # Prepare request
            logger.debug(
                f"Calling Gemini Intent Classification (Model: {model_name})..."
            )
            generation_config = {
                "candidate_count": 1,
                "max_output_tokens": 20,
                "temperature": 0.1,
            }
            # Combine system prompt and history for Gemini
            prompt_content = (
                f"{SYSTEM_PROMPT_INTENT}\n\nConversation History:\n{context_history}"
            )
            # Make API call
            response = model.generate_content(
                prompt_content, generation_config=generation_config
            )
            # Process response
            if not response.candidates:
                block_reason = "Unknown"
                try:
                    if (
                        hasattr(response, "prompt_feedback")
                        and response.prompt_feedback
                    ):
                        block_reason = response.prompt_feedback.block_reason.name
                except Exception:
                    pass  # Ignore errors getting block reason
                logger.warning(
                    f"Gemini intent response blocked or empty. Reason: {block_reason}. Defaulting to OTHER."
                )
                classification_result = "OTHER"  # Treat blocked/empty as OTHER
            else:
                # Extract text, strip whitespace, ensure uppercase
                raw_classification = response.text.strip().upper()
                if raw_classification in EXPECTED_INTENT_CATEGORIES:
                    classification_result = raw_classification
                else:
                    logger.warning(
                        f"Gemini returned unexpected classification: '{raw_classification}'. Defaulting to OTHER."
                    )
                    classification_result = "OTHER"
        else:
            # Handle unsupported provider
            logger.error(
                f"classify_message_intent: Unsupported AI_PROVIDER configured: {ai_provider}"
            )

    # Step 5: Handle Specific API/Library Errors
    except AuthenticationError as e:
        logger.error(f"AI Authentication Error ({ai_provider}): {e}")
    except RateLimitError as e:
        logger.error(f"AI Rate Limit Error ({ai_provider}): {e}")
        session_manager.dynamic_rate_limiter.increase_delay()
    except APIConnectionError as e:
        logger.error(f"AI Connection Error ({ai_provider}): {e}")
    except APIError as e:
        logger.error(
            f"AI API Error ({ai_provider}): Status={e.status_code}, Message={e.message}"
        )
    # Gemini specific exceptions
    except google_exceptions.PermissionDenied as e:
        logger.error(f"AI Permission Denied (Gemini): {e}")
    except google_exceptions.ResourceExhausted as e:
        logger.error(f"AI Rate Limit Error (Gemini): {e}")
        session_manager.dynamic_rate_limiter.increase_delay()
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Google API Error (Gemini): {e}")
    # General Python errors
    except AttributeError as ae:
        logger.critical(
            f"AttributeError during AI call ({ai_provider}): {ae}. Check library installation/imports.",
            exc_info=True,
        )
    except NameError as ne:
        logger.critical(
            f"NameError during AI call ({ai_provider}): {ne}. Check library installation/imports.",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"Unexpected error during AI intent classification ({ai_provider}): {type(e).__name__} - {e}",
            exc_info=True,
        )

    # Step 6: Log duration and result
    duration = time.time() - start_time
    if classification_result:
        logger.debug(
            f"AI intent classification result: '{classification_result}' (Took {duration:.2f}s)"
        )
    else:
        logger.error(
            f"AI intent classification failed for {ai_provider}. (Took {duration:.2f}s)"
        )

    # Step 7: Return the classification or None
    return classification_result


# End of classify_message_intent


def extract_and_suggest_tasks(
    context_history: str, session_manager: SessionManager
) -> Optional[Dict[str, Any]]:
    """
    Calls the configured AI model to extract genealogical entities and suggest
    follow-up tasks based on the provided conversation history. Expects JSON output.

    Args:
        context_history: The formatted conversation history string.
        session_manager: The active SessionManager instance (for rate limiting).

    Returns:
        A dictionary containing 'extracted_data' and 'suggested_tasks' if successful
        and the response is valid JSON matching the expected structure, otherwise None.
    """
    # Step 1: Validate inputs and configuration
    ai_provider = config_instance.AI_PROVIDER
    if not session_manager or not hasattr(session_manager, "dynamic_rate_limiter"):
        logger.error(
            "extract_and_suggest_tasks: Invalid SessionManager or missing rate limiter."
        )
        return None
    if not ai_provider:
        logger.error("extract_and_suggest_tasks: AI_PROVIDER not configured.")
        return None
    if not context_history:
        logger.warning(
            "extract_and_suggest_tasks: Received empty context history. Cannot extract."
        )
        # Return structure with empty lists for consistency downstream
        return {
            "extracted_data": {
                "mentioned_names": [],
                "mentioned_locations": [],
                "mentioned_dates": [],
                "potential_relationships": [],
                "key_facts": [],
            },
            "suggested_tasks": [],
        }

    # Step 2: Apply Rate Limiting
    wait_time = session_manager.dynamic_rate_limiter.wait()
    # Optional: Log if wait time was significant
    # if wait_time > 0.1: logger.debug(f"AI Extraction API rate limit wait: {wait_time:.2f}s")

    # Step 3: Initialize result and timer
    extraction_result: Optional[Dict[str, Any]] = None
    start_time = time.time()
    max_tokens_extraction = 700  # Allow more tokens for potentially detailed JSON

    # Step 4: Call the appropriate AI provider
    try:
        if ai_provider == "deepseek":
            # --- DeepSeek/OpenAI Compatible API Call ---
            if not openai_available or OpenAI is None:
                logger.error("extract_and_suggest_tasks: OpenAI library not available.")
                return None
            # Load config
            api_key = config_instance.DEEPSEEK_API_KEY
            model = config_instance.DEEPSEEK_AI_MODEL
            base_url = config_instance.DEEPSEEK_AI_BASE_URL
            if not all([api_key, model, base_url]):
                logger.error(
                    "extract_and_suggest_tasks: DeepSeek configuration incomplete."
                )
                return None
            # Create client and make request (requesting JSON object)
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.debug(f"Calling DeepSeek Extraction/Suggestion (Model: {model})...")
            openai_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACTION_TASK_SYSTEM_PROMPT},
                    {"role": "user", "content": context_history},
                ],
                stream=False,
                max_tokens=max_tokens_extraction,
                temperature=0.2,  # Allow some creativity for tasks
                response_format={"type": "json_object"},  # Explicitly request JSON
            )
            # Process response
            if openai_response.choices and openai_response.choices[0].message:
                raw_response_content = openai_response.choices[0].message.content
                # Attempt to parse the JSON response
                try:
                    parsed_json = json.loads(raw_response_content)
                    # Validate the structure of the parsed JSON
                    if (
                        isinstance(parsed_json, dict)
                        and "extracted_data" in parsed_json
                        and isinstance(parsed_json["extracted_data"], dict)
                        and "suggested_tasks" in parsed_json
                        and isinstance(parsed_json["suggested_tasks"], list)
                        and
                        # Validate sub-structure of extracted_data
                        all(
                            key in parsed_json["extracted_data"]
                            for key in [
                                "mentioned_names",
                                "mentioned_locations",
                                "mentioned_dates",
                                "potential_relationships",
                                "key_facts",
                            ]
                        )
                        and all(
                            isinstance(parsed_json["extracted_data"][key], list)
                            for key in [
                                "mentioned_names",
                                "mentioned_locations",
                                "mentioned_dates",
                                "potential_relationships",
                                "key_facts",
                            ]
                        )
                    ):
                        extraction_result = parsed_json  # Valid structure
                    else:
                        logger.warning(
                            f"DeepSeek extraction response is valid JSON but missing expected structure: {raw_response_content}"
                        )
                except json.JSONDecodeError as e:
                    logger.error(
                        f"DeepSeek extraction response was not valid JSON: {e}\nContent: {raw_response_content}"
                    )
            else:
                logger.error(
                    "Invalid response structure received from DeepSeek API (Extraction)."
                )

        elif ai_provider == "gemini":
            # --- Google Gemini API Call ---
            if not genai_available:
                logger.error(
                    "extract_and_suggest_tasks: Google GenerativeAI library not available or incomplete."
                )
                return None
            # Load config
            api_key = config_instance.GOOGLE_API_KEY
            model_name = config_instance.GOOGLE_AI_MODEL
            if not api_key or not model_name:
                logger.error(
                    "extract_and_suggest_tasks: Gemini configuration incomplete."
                )
                return None
            # Configure API and model
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
            except Exception as gemini_setup_err:
                logger.error(
                    f"Failed to configure/initialize Gemini model for extraction: {gemini_setup_err}"
                )
                return None
            # Prepare request (requesting JSON output)
            logger.debug(
                f"Calling Gemini Extraction/Suggestion (Model: {model_name})..."
            )
            generation_config = {
                "candidate_count": 1,
                "max_output_tokens": max_tokens_extraction,
                "temperature": 0.2,
                "response_mime_type": "application/json",
            }
            prompt_content = f"{EXTRACTION_TASK_SYSTEM_PROMPT}\n\nConversation History:\n{context_history}"
            # Make API call
            response = model.generate_content(
                prompt_content, generation_config=generation_config
            )
            # Process response
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
                    f"Gemini extraction response blocked or empty. Reason: {block_reason}."
                )
            else:
                raw_response_content = response.text
                # Attempt to parse the JSON response
                try:
                    # Clean potential markdown formatting (```json ... ```)
                    if raw_response_content.strip().startswith("```json"):
                        cleaned_content = (
                            raw_response_content.strip()
                            .strip("```json")
                            .strip("`")
                            .strip()
                        )
                    else:
                        cleaned_content = raw_response_content
                    parsed_json = json.loads(cleaned_content)
                    # Validate structure
                    if (
                        isinstance(parsed_json, dict)
                        and "extracted_data" in parsed_json
                        and isinstance(parsed_json["extracted_data"], dict)
                        and "suggested_tasks" in parsed_json
                        and isinstance(parsed_json["suggested_tasks"], list)
                        and all(
                            key in parsed_json["extracted_data"]
                            for key in [
                                "mentioned_names",
                                "mentioned_locations",
                                "mentioned_dates",
                                "potential_relationships",
                                "key_facts",
                            ]
                        )
                        and all(
                            isinstance(parsed_json["extracted_data"][key], list)
                            for key in [
                                "mentioned_names",
                                "mentioned_locations",
                                "mentioned_dates",
                                "potential_relationships",
                                "key_facts",
                            ]
                        )
                    ):
                        extraction_result = parsed_json  # Valid structure
                    else:
                        logger.warning(
                            f"Gemini extraction response is valid JSON but missing expected structure: {cleaned_content}"
                        )
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Gemini extraction response was not valid JSON: {e}\nContent: {raw_response_content}"
                    )  # Log original raw content
        else:
            # Handle unsupported provider
            logger.error(
                f"extract_and_suggest_tasks: Unsupported AI_PROVIDER configured: {ai_provider}"
            )

    # Step 5: Handle Specific API/Library Errors (same pattern as intent classification)
    except AuthenticationError as e:
        logger.error(f"AI Authentication Error ({ai_provider}): {e}")
    except RateLimitError as e:
        logger.error(f"AI Rate Limit Error ({ai_provider}): {e}")
        session_manager.dynamic_rate_limiter.increase_delay()
    except APIConnectionError as e:
        logger.error(f"AI Connection Error ({ai_provider}): {e}")
    except APIError as e:
        logger.error(
            f"AI API Error ({ai_provider}): Status={e.status_code}, Message={e.message}"
        )
    except google_exceptions.PermissionDenied as e:
        logger.error(f"AI Permission Denied (Gemini): {e}")
    except google_exceptions.ResourceExhausted as e:
        logger.error(f"AI Rate Limit Error (Gemini): {e}")
        session_manager.dynamic_rate_limiter.increase_delay()
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Google API Error (Gemini): {e}")
    except AttributeError as ae:
        logger.critical(
            f"AttributeError during AI call ({ai_provider}): {ae}. Check library installation/imports.",
            exc_info=True,
        )
    except NameError as ne:
        logger.critical(
            f"NameError during AI call ({ai_provider}): {ne}. Check library installation/imports.",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"Unexpected error during AI extraction ({ai_provider}): {type(e).__name__} - {e}",
            exc_info=True,
        )

    # Step 6: Log duration and result
    duration = time.time() - start_time
    if extraction_result:
        logger.debug(f"AI extraction/suggestion successful. (Took {duration:.2f}s)")
    else:
        logger.error(
            f"AI extraction/suggestion failed for {ai_provider}. (Took {duration:.2f}s)"
        )

    # Step 7: Return the parsed JSON dictionary or None
    return extraction_result


# End of extract_and_suggest_tasks

# End of ai_interface.py
