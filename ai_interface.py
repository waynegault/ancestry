# ai_interface.py

"""
ai_interface.py - AI Model Interaction Layer

Provides functions to interact with external AI models (configured via config.py)
for tasks like message intent classification and genealogical data extraction.
Supports DeepSeek (OpenAI compatible) and Google Gemini Pro. Includes error
handling and rate limiting integration with SessionManager.
"""

# --- Standard library imports ---
import json
import logging
import sys
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
# Try to import the AI prompt utilities
try:
    from ai_prompt_utils import get_prompt

    USE_JSON_PROMPTS = True
except ImportError:
    logger.warning("ai_prompt_utils module not available, using hardcoded prompts")
    USE_JSON_PROMPTS = False

# System prompt for generating genealogical replies (fallback if JSON prompts not available)
GENERATE_GENEALOGICAL_REPLY_PROMPT = """You are a helpful genealogical assistant named Wayne responding to messages on behalf of a family history researcher from Aberdeen, Scotland.

You will receive:
1. A conversation history between the researcher (SCRIPT) and a user (USER)
2. The user's last message
3. Genealogical data about a person mentioned in the user's message, including:
   - Name, birth/death information
   - Family relationships
   - Relationship to the tree owner (the researcher)

Your task is to generate a natural, polite, and informative reply that:
- Directly addresses the user's query or comment
- Incorporates the provided genealogical data in a helpful way
- Acknowledges the user's point and integrates the found information smoothly
- May suggest connections or ask a clarifying follow-up question if appropriate
- Maintains a warm, helpful, conversational tone
- Refers to yourself as "I" and the tree as "my family tree" or "my records"
- Shows genuine interest in the user's research and family connections

IMPORTANT: When replying about people found in your tree, you MUST include:
1. COMPLETE birth details (full date and place if available)
2. COMPLETE death details (full date and place if available)
3. DETAILED family information (parents, spouse, children)
4. SPECIFIC relationship to you (the tree owner)
5. Any other significant details like occupation, immigration, etc.

If multiple people are mentioned in the genealogical data, focus on the one with the highest match score or most complete information.

If the genealogical data indicates "No person found" or is empty, acknowledge this and ask for more details that might help identify the person in your records.

For people not in your tree, acknowledge this and ask for more details that might help identify connections.

Your response should be ONLY the message text, with no additional formatting, explanation, or signature (the system will add a signature automatically).
"""

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
EXPECTED_INTENT_CATEGORIES = {
    "ENTHUSIASTIC",
    "CAUTIOUSLY_INTERESTED",
    "UNINTERESTED",
    "CONFUSED",
    "PRODUCTIVE",
    "OTHER",
}

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

    # Step 2: Apply Rate Limiting (skip if session_manager is None)
    if session_manager is not None:
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
            # Get the intent classification prompt from the JSON file if available
            intent_prompt = SYSTEM_PROMPT_INTENT
            if USE_JSON_PROMPTS:
                try:
                    json_prompt = get_prompt("intent_classification")
                    if json_prompt:
                        intent_prompt = json_prompt
                        logger.debug(
                            "Using intent classification prompt from ai_prompts.json"
                        )
                except Exception as e:
                    logger.error(
                        f"Error getting intent classification prompt from JSON: {e}"
                    )

            openai_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": intent_prompt},
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
            # Get the intent classification prompt from the JSON file if available
            intent_prompt = SYSTEM_PROMPT_INTENT
            if USE_JSON_PROMPTS:
                try:
                    json_prompt = get_prompt("intent_classification")
                    if json_prompt:
                        intent_prompt = json_prompt
                        logger.debug(
                            "Using intent classification prompt from ai_prompts.json"
                        )
                except Exception as e:
                    logger.error(
                        f"Error getting intent classification prompt from JSON: {e}"
                    )

            # Combine system prompt and history for Gemini
            prompt_content = (
                f"{intent_prompt}\n\nConversation History:\n{context_history}"
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
        if session_manager is not None:
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
        if session_manager is not None:
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

    # Step 2: Apply Rate Limiting (skip if session_manager is None)
    if session_manager is not None:
        session_manager.dynamic_rate_limiter.wait()
        # Optional: Log if wait time was significant
        # if wait_time > 0.1: logger.debug(f"AI Extraction API rate limit wait: {wait_time:.2f}s")

    # Step 3: Initialize result and timer
    extraction_result: Optional[Dict[str, Any]] = None
    start_time = time.time()
    max_tokens_extraction = 700  # Allow more tokens for potentially detailed JSON

    # Default structure to return in case of errors
    default_result = {
        "extracted_data": {
            "mentioned_names": [],
            "mentioned_locations": [],
            "mentioned_dates": [],
            "potential_relationships": [],
            "key_facts": [],
        },
        "suggested_tasks": [],
    }

    # Step 4: Call the appropriate AI provider
    try:
        if ai_provider == "deepseek":
            # --- DeepSeek/OpenAI Compatible API Call ---
            if not openai_available or OpenAI is None:
                logger.error("extract_and_suggest_tasks: OpenAI library not available.")
                return default_result
            # Load config
            api_key = config_instance.DEEPSEEK_API_KEY
            model = config_instance.DEEPSEEK_AI_MODEL
            base_url = config_instance.DEEPSEEK_AI_BASE_URL
            if not all([api_key, model, base_url]):
                logger.error(
                    "extract_and_suggest_tasks: DeepSeek configuration incomplete."
                )
                return default_result
            # Create client and make request (requesting JSON object)
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.debug(f"Calling DeepSeek Extraction/Suggestion (Model: {model})...")
            # Get the extraction prompt from the JSON file if available
            extraction_prompt = EXTRACTION_TASK_SYSTEM_PROMPT
            if USE_JSON_PROMPTS:
                try:
                    json_prompt = get_prompt("extraction_task")
                    if json_prompt:
                        extraction_prompt = json_prompt
                        logger.debug("Using extraction prompt from ai_prompts.json")
                except Exception as e:
                    logger.error(f"Error getting extraction prompt from JSON: {e}")

            openai_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": extraction_prompt},
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
                if raw_response_content:
                    # Attempt to parse the JSON response
                    try:
                        parsed_json = json.loads(raw_response_content)
                        # Validate the structure of the parsed JSON
                        if isinstance(parsed_json, dict):
                            # If the response doesn't have the expected structure, add it
                            if "extracted_data" not in parsed_json:
                                logger.warning(
                                    "Adding missing 'extracted_data' field to AI response"
                                )
                                parsed_json["extracted_data"] = {
                                    "mentioned_names": [],
                                    "mentioned_locations": [],
                                    "mentioned_dates": [],
                                    "potential_relationships": [],
                                    "key_facts": [],
                                }
                            elif not isinstance(parsed_json["extracted_data"], dict):
                                logger.warning(
                                    "'extracted_data' is not a dictionary, replacing with default structure"
                                )
                                parsed_json["extracted_data"] = {
                                    "mentioned_names": [],
                                    "mentioned_locations": [],
                                    "mentioned_dates": [],
                                    "potential_relationships": [],
                                    "key_facts": [],
                                }

                            # Handle the extraction prompt format from ai_prompts.json which uses different field names
                            # Map the fields from the prompt format to the expected fields
                            if "mentioned_names" not in parsed_json["extracted_data"]:
                                # Check if this is using the format from ai_prompts.json with people, relationships, etc.
                                if "people" in parsed_json:
                                    logger.info(
                                        "Detected ai_prompts.json format with 'people' field, mapping to expected structure"
                                    )

                                    # Create a new structure with the expected fields
                                    new_structure = {
                                        "mentioned_names": [],
                                        "mentioned_locations": [],
                                        "mentioned_dates": [],
                                        "potential_relationships": [],
                                        "key_facts": [],
                                    }
                                # Check if this is using the format from the updated ai_prompts.json with mentioned_names, dates, locations, etc.
                                elif (
                                    "dates" in parsed_json["extracted_data"]
                                    or "locations" in parsed_json["extracted_data"]
                                    or "relationships" in parsed_json["extracted_data"]
                                ):
                                    logger.info(
                                        "Detected updated ai_prompts.json format with 'dates', 'locations', 'relationships' fields, mapping to expected structure"
                                    )

                                    # Create a new structure with the expected fields
                                    new_structure = {
                                        "mentioned_names": parsed_json[
                                            "extracted_data"
                                        ].get("mentioned_names", []),
                                        "mentioned_locations": parsed_json[
                                            "extracted_data"
                                        ].get("locations", []),
                                        "mentioned_dates": parsed_json[
                                            "extracted_data"
                                        ].get("dates", []),
                                        "potential_relationships": parsed_json[
                                            "extracted_data"
                                        ].get("relationships", []),
                                        "key_facts": [],
                                    }

                                    # Add occupations, events, and research_questions to key_facts
                                    if "occupations" in parsed_json["extracted_data"]:
                                        new_structure["key_facts"].extend(
                                            parsed_json["extracted_data"]["occupations"]
                                        )
                                    if "events" in parsed_json["extracted_data"]:
                                        new_structure["key_facts"].extend(
                                            parsed_json["extracted_data"]["events"]
                                        )
                                    if (
                                        "research_questions"
                                        in parsed_json["extracted_data"]
                                    ):
                                        parsed_json["suggested_tasks"] = parsed_json[
                                            "extracted_data"
                                        ]["research_questions"]

                                    # Update the extracted_data with the new structure
                                    parsed_json["extracted_data"] = new_structure
                                    logger.info(
                                        "Successfully mapped updated ai_prompts.json format to expected structure"
                                    )
                                    return parsed_json

                                    # Extract names from people array
                                    if "people" in parsed_json and isinstance(
                                        parsed_json["people"], list
                                    ):
                                        for person in parsed_json["people"]:
                                            name_parts = []
                                            if isinstance(person, dict):
                                                if (
                                                    "firstName" in person
                                                    and person["firstName"]
                                                ):
                                                    name_parts.append(
                                                        person["firstName"]
                                                    )
                                                if (
                                                    "middleName" in person
                                                    and person["middleName"]
                                                ):
                                                    name_parts.append(
                                                        person["middleName"]
                                                    )
                                                if (
                                                    "maidenName" in person
                                                    and person["maidenName"]
                                                ):
                                                    name_parts.append(
                                                        f"(née {person['maidenName']})"
                                                    )
                                                if (
                                                    "lastName" in person
                                                    and person["lastName"]
                                                ):
                                                    name_parts.append(
                                                        person["lastName"]
                                                    )

                                                if name_parts:
                                                    full_name = " ".join(name_parts)
                                                    new_structure[
                                                        "mentioned_names"
                                                    ].append(full_name)

                                                # Extract birth and death dates
                                                if (
                                                    "birthDate" in person
                                                    and isinstance(
                                                        person["birthDate"], dict
                                                    )
                                                ):
                                                    birth_date = ""
                                                    confidence = ""

                                                    if (
                                                        "full" in person["birthDate"]
                                                        and person["birthDate"]["full"]
                                                    ):
                                                        birth_date = person[
                                                            "birthDate"
                                                        ]["full"]
                                                    elif (
                                                        "year" in person["birthDate"]
                                                        and person["birthDate"]["year"]
                                                    ):
                                                        birth_date = person[
                                                            "birthDate"
                                                        ]["year"]

                                                    if (
                                                        "confidence"
                                                        in person["birthDate"]
                                                        and person["birthDate"][
                                                            "confidence"
                                                        ]
                                                        and person["birthDate"][
                                                            "confidence"
                                                        ]
                                                        != "certain"
                                                    ):
                                                        confidence = f" ({person['birthDate']['confidence']})"

                                                    if birth_date:
                                                        new_structure[
                                                            "mentioned_dates"
                                                        ].append(
                                                            f"Birth: {birth_date}{confidence}"
                                                        )

                                                if (
                                                    "deathDate" in person
                                                    and isinstance(
                                                        person["deathDate"], dict
                                                    )
                                                ):
                                                    death_date = ""
                                                    confidence = ""

                                                    if (
                                                        "full" in person["deathDate"]
                                                        and person["deathDate"]["full"]
                                                    ):
                                                        death_date = person[
                                                            "deathDate"
                                                        ]["full"]
                                                    elif (
                                                        "year" in person["deathDate"]
                                                        and person["deathDate"]["year"]
                                                    ):
                                                        death_date = person[
                                                            "deathDate"
                                                        ]["year"]

                                                    if (
                                                        "confidence"
                                                        in person["deathDate"]
                                                        and person["deathDate"][
                                                            "confidence"
                                                        ]
                                                        and person["deathDate"][
                                                            "confidence"
                                                        ]
                                                        != "certain"
                                                    ):
                                                        confidence = f" ({person['deathDate']['confidence']})"

                                                    if death_date:
                                                        new_structure[
                                                            "mentioned_dates"
                                                        ].append(
                                                            f"Death: {death_date}{confidence}"
                                                        )

                                                # Extract birth and death places
                                                if (
                                                    "birthPlace" in person
                                                    and isinstance(
                                                        person["birthPlace"], dict
                                                    )
                                                ):
                                                    place_description = ""
                                                    confidence = ""

                                                    if (
                                                        "description"
                                                        in person["birthPlace"]
                                                        and person["birthPlace"][
                                                            "description"
                                                        ]
                                                    ):
                                                        place_description = person[
                                                            "birthPlace"
                                                        ]["description"]
                                                    elif (
                                                        "country"
                                                        in person["birthPlace"]
                                                        and person["birthPlace"][
                                                            "country"
                                                        ]
                                                    ):
                                                        place_description = person[
                                                            "birthPlace"
                                                        ]["country"]

                                                    if (
                                                        "confidence"
                                                        in person["birthPlace"]
                                                        and person["birthPlace"][
                                                            "confidence"
                                                        ]
                                                        and person["birthPlace"][
                                                            "confidence"
                                                        ]
                                                        != "certain"
                                                    ):
                                                        confidence = f" ({person['birthPlace']['confidence']})"

                                                    if place_description:
                                                        new_structure[
                                                            "mentioned_locations"
                                                        ].append(
                                                            f"Birth: {place_description}{confidence}"
                                                        )

                                                if (
                                                    "deathPlace" in person
                                                    and isinstance(
                                                        person["deathPlace"], dict
                                                    )
                                                ):
                                                    place_description = ""
                                                    confidence = ""

                                                    if (
                                                        "description"
                                                        in person["deathPlace"]
                                                        and person["deathPlace"][
                                                            "description"
                                                        ]
                                                    ):
                                                        place_description = person[
                                                            "deathPlace"
                                                        ]["description"]
                                                    elif (
                                                        "country"
                                                        in person["deathPlace"]
                                                        and person["deathPlace"][
                                                            "country"
                                                        ]
                                                    ):
                                                        place_description = person[
                                                            "deathPlace"
                                                        ]["country"]

                                                    if (
                                                        "confidence"
                                                        in person["deathPlace"]
                                                        and person["deathPlace"][
                                                            "confidence"
                                                        ]
                                                        and person["deathPlace"][
                                                            "confidence"
                                                        ]
                                                        != "certain"
                                                    ):
                                                        confidence = f" ({person['deathPlace']['confidence']})"

                                                    if place_description:
                                                        new_structure[
                                                            "mentioned_locations"
                                                        ].append(
                                                            f"Death: {place_description}{confidence}"
                                                        )

                                                # Extract occupation as key fact
                                                if (
                                                    "occupation" in person
                                                    and person["occupation"]
                                                ):
                                                    new_structure["key_facts"].append(
                                                        f"Occupation: {person['occupation']}"
                                                    )

                                                # Extract notes as key fact
                                                if (
                                                    "notes" in person
                                                    and person["notes"]
                                                ):
                                                    new_structure["key_facts"].append(
                                                        person["notes"]
                                                    )

                                    # Extract relationships
                                    if "relationships" in parsed_json and isinstance(
                                        parsed_json["relationships"], list
                                    ):
                                        for relationship in parsed_json[
                                            "relationships"
                                        ]:
                                            if isinstance(relationship, dict):
                                                relation_description = ""
                                                confidence = ""

                                                # Get the specific relation description
                                                if "specificRelation" in relationship:
                                                    relation_description = relationship[
                                                        "specificRelation"
                                                    ]
                                                elif "relationshipType" in relationship:
                                                    relation_description = relationship[
                                                        "relationshipType"
                                                    ]

                                                    # Try to get the person names to make the relationship more specific
                                                    person1_id = relationship.get(
                                                        "person1Id"
                                                    )
                                                    person2_id = relationship.get(
                                                        "person2Id"
                                                    )

                                                    if person1_id and person2_id:
                                                        # Find the names of the people in the relationship
                                                        person1_name = ""
                                                        person2_name = ""

                                                        for person in parsed_json.get(
                                                            "people", []
                                                        ):
                                                            if (
                                                                isinstance(person, dict)
                                                                and "id" in person
                                                            ):
                                                                if (
                                                                    person["id"]
                                                                    == person1_id
                                                                ):
                                                                    name_parts = []
                                                                    if person.get(
                                                                        "firstName"
                                                                    ):
                                                                        name_parts.append(
                                                                            person[
                                                                                "firstName"
                                                                            ]
                                                                        )
                                                                    if person.get(
                                                                        "lastName"
                                                                    ):
                                                                        name_parts.append(
                                                                            person[
                                                                                "lastName"
                                                                            ]
                                                                        )
                                                                    if name_parts:
                                                                        person1_name = " ".join(
                                                                            name_parts
                                                                        )

                                                                if (
                                                                    person["id"]
                                                                    == person2_id
                                                                ):
                                                                    name_parts = []
                                                                    if person.get(
                                                                        "firstName"
                                                                    ):
                                                                        name_parts.append(
                                                                            person[
                                                                                "firstName"
                                                                            ]
                                                                        )
                                                                    if person.get(
                                                                        "lastName"
                                                                    ):
                                                                        name_parts.append(
                                                                            person[
                                                                                "lastName"
                                                                            ]
                                                                        )
                                                                    if name_parts:
                                                                        person2_name = " ".join(
                                                                            name_parts
                                                                        )

                                                        if (
                                                            person1_name
                                                            and person2_name
                                                        ):
                                                            relation_description = f"{person1_name} is {relation_description} of {person2_name}"

                                                # Add confidence if available and not "certain"
                                                if (
                                                    "confidence" in relationship
                                                    and relationship["confidence"]
                                                    and relationship["confidence"]
                                                    != "certain"
                                                ):
                                                    confidence = f" ({relationship['confidence']})"

                                                # Add notes if available
                                                notes = ""
                                                if (
                                                    "notes" in relationship
                                                    and relationship["notes"]
                                                ):
                                                    notes = (
                                                        f" - {relationship['notes']}"
                                                    )

                                                if relation_description:
                                                    new_structure[
                                                        "potential_relationships"
                                                    ].append(
                                                        f"{relation_description}{confidence}{notes}"
                                                    )

                                    # Extract research gaps as suggested tasks
                                    if "researchGaps" in parsed_json and isinstance(
                                        parsed_json["researchGaps"], list
                                    ):
                                        if "suggested_tasks" not in parsed_json:
                                            parsed_json["suggested_tasks"] = []

                                        for gap in parsed_json["researchGaps"]:
                                            if isinstance(gap, dict):
                                                gap_description = ""
                                                priority = ""
                                                sources = ""

                                                if "description" in gap:
                                                    gap_description = gap["description"]

                                                if (
                                                    "priority" in gap
                                                    and gap["priority"]
                                                    and gap["priority"] != "medium"
                                                ):
                                                    priority = (
                                                        f" ({gap['priority']} priority)"
                                                    )

                                                if (
                                                    "potentialSources" in gap
                                                    and isinstance(
                                                        gap["potentialSources"], list
                                                    )
                                                    and gap["potentialSources"]
                                                ):
                                                    sources = f" - Check: {', '.join(gap['potentialSources'])}"

                                                if gap_description:
                                                    task = f"{gap_description}{priority}{sources}"
                                                    parsed_json[
                                                        "suggested_tasks"
                                                    ].append(task)

                                    # Extract sources referenced as key facts
                                    if (
                                        "sourcesReferenced" in parsed_json
                                        and isinstance(
                                            parsed_json["sourcesReferenced"], list
                                        )
                                    ):
                                        for source in parsed_json["sourcesReferenced"]:
                                            if (
                                                isinstance(source, dict)
                                                and "description" in source
                                            ):
                                                source_type = ""
                                                credibility = ""

                                                if "type" in source and source["type"]:
                                                    source_type = f"{source['type'].capitalize()}: "

                                                if (
                                                    "credibility" in source
                                                    and source["credibility"]
                                                    and source["credibility"]
                                                    != "medium"
                                                ):
                                                    credibility = f" ({source['credibility']} credibility)"

                                                new_structure["key_facts"].append(
                                                    f"Source {source_type}{source['description']}{credibility}"
                                                )

                                    # Replace the extracted_data with the new structure
                                    parsed_json["extracted_data"] = new_structure

                                # Check if this is using the older improved prompt format with dates, locations, etc.
                                elif "dates" in parsed_json["extracted_data"]:
                                    logger.info(
                                        "Detected improved prompt format with 'dates' field, mapping fields to expected structure"
                                    )

                                    # Create a mapping of improved prompt fields to expected fields
                                    field_mapping = {
                                        "mentioned_names": "mentioned_names",
                                        "dates": "mentioned_dates",
                                        "locations": "mentioned_locations",
                                        "relationships": "potential_relationships",
                                        "occupations": "key_facts",
                                        "events": "key_facts",
                                        "research_questions": "suggested_tasks",
                                    }

                                    # Create a new structure with the expected fields
                                    new_structure = {
                                        "mentioned_names": [],
                                        "mentioned_locations": [],
                                        "mentioned_dates": [],
                                        "potential_relationships": [],
                                        "key_facts": [],
                                    }

                                    # Copy data from the improved prompt fields to the expected fields
                                    for (
                                        improved_field,
                                        expected_field,
                                    ) in field_mapping.items():
                                        if (
                                            improved_field
                                            in parsed_json["extracted_data"]
                                        ):
                                            if (
                                                expected_field == "key_facts"
                                                and improved_field
                                                in ["occupations", "events"]
                                            ):
                                                # Combine occupations and events into key_facts
                                                new_structure["key_facts"].extend(
                                                    parsed_json["extracted_data"][
                                                        improved_field
                                                    ]
                                                )
                                            elif (
                                                expected_field != "suggested_tasks"
                                            ):  # Handle suggested_tasks separately
                                                new_structure[expected_field] = (
                                                    parsed_json["extracted_data"][
                                                        improved_field
                                                    ]
                                                )

                                    # If research_questions exists, move it to suggested_tasks
                                    if (
                                        "research_questions"
                                        in parsed_json["extracted_data"]
                                    ):
                                        if "suggested_tasks" not in parsed_json:
                                            parsed_json["suggested_tasks"] = []
                                        parsed_json["suggested_tasks"].extend(
                                            parsed_json["extracted_data"][
                                                "research_questions"
                                            ]
                                        )

                                    # Replace the extracted_data with the new structure
                                    parsed_json["extracted_data"] = new_structure

                            # Ensure all required fields exist in extracted_data
                            for field in [
                                "mentioned_names",
                                "mentioned_locations",
                                "mentioned_dates",
                                "potential_relationships",
                                "key_facts",
                            ]:
                                if field not in parsed_json["extracted_data"]:
                                    logger.warning(
                                        f"Adding missing '{field}' field to extracted_data"
                                    )
                                    parsed_json["extracted_data"][field] = []
                                elif not isinstance(
                                    parsed_json["extracted_data"][field], list
                                ):
                                    logger.warning(
                                        f"'{field}' is not a list, replacing with empty list"
                                    )
                                    parsed_json["extracted_data"][field] = []

                            # If suggested_tasks is missing, add it
                            if "suggested_tasks" not in parsed_json:
                                logger.warning(
                                    "Adding missing 'suggested_tasks' field to AI response"
                                )
                                parsed_json["suggested_tasks"] = []
                            elif not isinstance(parsed_json["suggested_tasks"], list):
                                logger.warning(
                                    "'suggested_tasks' is not a list, replacing with empty list"
                                )
                                parsed_json["suggested_tasks"] = []

                            extraction_result = parsed_json  # Valid structure
                        else:
                            logger.warning(
                                f"DeepSeek extraction response is valid JSON but not a dictionary: {raw_response_content}"
                            )
                            extraction_result = default_result
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"DeepSeek extraction response was not valid JSON: {e}\nContent: {raw_response_content}"
                        )
                        extraction_result = default_result
                else:
                    logger.error("Empty content received from DeepSeek API")
                    extraction_result = default_result
            else:
                logger.error(
                    "Invalid response structure received from DeepSeek API (Extraction)."
                )
                extraction_result = default_result

        elif ai_provider == "gemini":
            # --- Google Gemini API Call ---
            logger.warning(
                "Gemini provider not fully implemented, using default result"
            )
            extraction_result = default_result
        else:
            # Handle unsupported provider
            logger.error(
                f"extract_and_suggest_tasks: Unsupported AI_PROVIDER configured: {ai_provider}"
            )
            extraction_result = default_result

    # Handle all exceptions
    except Exception as e:
        logger.error(
            f"Unexpected error during AI extraction ({ai_provider}): {type(e).__name__} - {e}",
            exc_info=True,
        )
        extraction_result = default_result

    # Step 6: Log duration and result
    duration = time.time() - start_time
    if extraction_result:
        logger.debug(f"AI extraction/suggestion successful. (Took {duration:.2f}s)")
    else:
        logger.error(
            f"AI extraction/suggestion failed for {ai_provider}. (Took {duration:.2f}s)"
        )
        # If we somehow still don't have a result, use the default
        extraction_result = default_result

    # Step 7: Return the parsed JSON dictionary
    return extraction_result


# End of extract_and_suggest_tasks


def generate_genealogical_reply(
    conversation_context: str,
    user_last_message: str,
    genealogical_data_str: str,
    session_manager: SessionManager,
) -> Optional[str]:
    """
    Calls the configured AI model to generate a personalized genealogical reply
    based on the conversation context, user's last message, and genealogical data.

    Args:
        conversation_context: The formatted conversation history string.
        user_last_message: The user's last message (for emphasis).
        genealogical_data_str: Structured string containing genealogical data about mentioned person(s).
        session_manager: The active SessionManager instance (for rate limiting).

    Returns:
        A string containing the generated reply, or None if generation failed.
    """
    # Step 1: Validate Inputs
    if not conversation_context or not user_last_message or not genealogical_data_str:
        logger.error("generate_genealogical_reply: Missing required input parameters.")
        return None

    # For test_ai_responses_menu.py, we'll allow a None session_manager
    if session_manager is not None and not session_manager.is_sess_valid():
        logger.warning(
            "generate_genealogical_reply: Session manager is not valid, but continuing for testing purposes."
        )
    # Skip rate limiting if session_manager is None

    # Get AI provider from config
    ai_provider = config_instance.AI_PROVIDER.lower()
    if not ai_provider:
        logger.error("generate_genealogical_reply: AI_PROVIDER not configured.")
        return None

    # Step 2: Apply Rate Limiting (skip if session_manager is None)
    if session_manager is not None:
        wait_time = session_manager.dynamic_rate_limiter.wait()
        # Optional: Log if wait time was significant
        # if wait_time > 0.1: logger.debug(f"AI Reply Generation API rate limit wait: {wait_time:.2f}s")

    # Step 3: Prepare Prompt
    # Get the genealogical reply prompt from the JSON file if available
    reply_prompt = GENERATE_GENEALOGICAL_REPLY_PROMPT
    if USE_JSON_PROMPTS:
        try:
            json_prompt = get_prompt("genealogical_reply")
            if json_prompt:
                reply_prompt = json_prompt
                logger.debug("Using genealogical reply prompt from ai_prompts.json")
        except Exception as e:
            logger.error(f"Error getting genealogical reply prompt from JSON: {e}")

    # Check if the prompt contains an {intent_classification} placeholder
    if "{intent_classification}" in reply_prompt:
        logger.info(
            "Detected {intent_classification} placeholder in prompt, getting intent classification"
        )

        # First try to get the intent classification from the classify_message_intent function
        try:
            from utils import SessionManager

            intent_classification = classify_message_intent(
                conversation_context, SessionManager()
            )
            logger.info(
                f"Got intent classification from classify_message_intent: {intent_classification}"
            )
        except Exception as e:
            logger.warning(
                f"Error getting intent classification from classify_message_intent: {e}"
            )
            intent_classification = None

        # If we couldn't get the intent classification from the function, infer it from the message content
        if not intent_classification:
            logger.info("Inferring intent classification from message content")
            # Try to infer intent from the message content
            # This is a simple heuristic and could be improved
            intent_classification = "ENTHUSIASTIC"  # Default to enthusiastic

            # Look for signals of different intents in the user's message
            lower_message = user_last_message.lower()

            # Check for uninterested signals
            uninterested_phrases = [
                "not interested",
                "stop messaging",
                "leave me alone",
                "don't contact",
            ]
            if any(phrase in lower_message for phrase in uninterested_phrases):
                intent_classification = "UNINTERESTED"

            # Check for confused signals
            confused_phrases = [
                "who are you",
                "why am i getting",
                "what is this about",
                "don't understand",
            ]
            if any(phrase in lower_message for phrase in confused_phrases):
                intent_classification = "CONFUSED"

            # Check for cautious signals
            cautious_phrases = [
                "not sure",
                "maybe",
                "possibly",
                "might be",
                "don't know if",
            ]
            if any(phrase in lower_message for phrase in cautious_phrases):
                intent_classification = "CAUTIOUSLY_INTERESTED"

            # If the message is very short, it's likely "OTHER"
            if len(user_last_message.split()) < 5:
                intent_classification = "OTHER"

            logger.info(f"Inferred intent classification: {intent_classification}")

        # Combine all inputs into a single prompt with the intent classification
        combined_prompt = f"""
{reply_prompt.replace("{intent_classification}", intent_classification)}

CONVERSATION HISTORY:
{conversation_context}

USER'S LAST MESSAGE:
{user_last_message}

GENEALOGICAL DATA:
{genealogical_data_str}
"""
    else:
        # Standard prompt without intent_classification
        combined_prompt = f"""
{reply_prompt}

CONVERSATION HISTORY:
{conversation_context}

USER'S LAST MESSAGE:
{user_last_message}

GENEALOGICAL DATA:
{genealogical_data_str}
"""

    # Step 4: Call the appropriate AI provider
    try:
        if ai_provider == "deepseek":
            # --- DeepSeek/OpenAI Compatible API Call ---
            if not openai_available or OpenAI is None:
                logger.error(
                    "generate_genealogical_reply: OpenAI library not available."
                )
                return None

            # Load config
            api_key = config_instance.DEEPSEEK_API_KEY
            model = config_instance.DEEPSEEK_AI_MODEL
            base_url = config_instance.DEEPSEEK_AI_BASE_URL

            if not all([api_key, model, base_url]):
                logger.error(
                    "generate_genealogical_reply: DeepSeek configuration incomplete."
                )
                return None

            # Initialize client
            try:
                client = OpenAI(api_key=api_key, base_url=base_url)
            except Exception as client_err:
                logger.error(f"Failed to initialize DeepSeek client: {client_err}")
                return None

            # Make API call
            openai_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GENERATE_GENEALOGICAL_REPLY_PROMPT},
                    {
                        "role": "user",
                        "content": f"CONVERSATION HISTORY:\n{conversation_context}\n\nUSER'S LAST MESSAGE:\n{user_last_message}\n\nGENEALOGICAL DATA:\n{genealogical_data_str}",
                    },
                ],
                temperature=0.7,  # Slightly creative but still focused
                max_tokens=1000,  # Reasonable length for a reply
            )

            # Process response
            if openai_response.choices and openai_response.choices[0].message:
                reply_text = openai_response.choices[0].message.content.strip()
                # Success - return the generated reply
                return reply_text
            else:
                logger.error(
                    "DeepSeek API returned empty response for genealogical reply."
                )
                return None

        elif ai_provider == "gemini":
            # --- Google Gemini API Call ---
            if not genai_available or genai is None:
                logger.error(
                    "generate_genealogical_reply: Google Generative AI library not available."
                )
                return None

            # Load config
            api_key = config_instance.GOOGLE_API_KEY
            model_name = config_instance.GOOGLE_AI_MODEL

            if not all([api_key, model_name]):
                logger.error(
                    "generate_genealogical_reply: Gemini configuration incomplete."
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

            # Set generation config
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1000,
            }

            # Prepare prompt content
            prompt_content = combined_prompt

            # Make API call
            response = model.generate_content(
                prompt_content, generation_config=generation_config
            )

            # Process response
            if not response.candidates:
                logger.error(
                    "Gemini API returned empty response for genealogical reply."
                )
                return None

            raw_response_content = response.text

            if raw_response_content:
                # Success - return the generated reply
                return raw_response_content.strip()
            else:
                logger.error("Gemini API returned empty text for genealogical reply.")
                return None
        else:
            logger.error(
                f"Unsupported AI provider '{ai_provider}' for genealogical reply generation."
            )
            return None

    # Step 5: Handle Specific API/Library Errors
    except Exception as e:
        logger.error(f"Error generating genealogical reply: {e}", exc_info=True)
        return None

    # If we get here, something went wrong
    return None


def self_check() -> bool:
    """
    Performs a self-check of the AI interface functionality using real API calls.
    Tests both intent classification and data extraction/task suggestion.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    from utils import SessionManager

    print("\n=== AI Interface Self-Check Starting ===")

    # Track overall status
    all_tests_passed = True

    # Step 1: Check configuration
    ai_provider = config_instance.AI_PROVIDER
    print(f"Using AI Provider: {ai_provider}")

    if ai_provider == "deepseek":
        if not openai_available:
            print("ERROR: OpenAI library not available. Cannot test DeepSeek.")
            return False
        api_key = config_instance.DEEPSEEK_API_KEY
        model = config_instance.DEEPSEEK_AI_MODEL
        base_url = config_instance.DEEPSEEK_AI_BASE_URL
        if not all([api_key, model, base_url]):
            print("ERROR: DeepSeek configuration incomplete.")
            return False
        print(f"DeepSeek Model: {model}")
    elif ai_provider == "gemini":
        if not genai_available:
            print(
                "ERROR: Google GenerativeAI library not available. Cannot test Gemini."
            )
            return False
        api_key = config_instance.GOOGLE_API_KEY
        model_name = config_instance.GOOGLE_AI_MODEL
        if not api_key or not model_name:
            print("ERROR: Gemini configuration incomplete.")
            return False
        print(f"Gemini Model: {model_name}")
    else:
        print(f"ERROR: Unsupported AI provider: {ai_provider}")
        return False

    # Step 2: Create a SessionManager for testing
    try:
        # Create SessionManager without starting a browser
        session_manager = SessionManager()
        print("Created SessionManager for testing (without browser)")
    except Exception as e:
        print(f"ERROR: Failed to create SessionManager: {e}")
        return False

    # Step 3: Test conversation histories
    # Test case 1: PRODUCTIVE intent
    productive_conversation = """
SCRIPT: Hello, I'm researching my family tree and we appear to be DNA matches. I'm trying to find our common ancestor. My tree includes the Simpson family from Aberdeen, Scotland in the 1800s. Does that connect with your research?

USER: Yes, I have Simpsons in my tree from Aberdeen! My great-grandmother was Margaret Simpson born around 1865. Her father was Alexander Simpson who was a fisherman. I'd be happy to share what I know about them.

SCRIPT: That's wonderful! Margaret Simpson sounds familiar. Do you know who Alexander's parents were or if he had any siblings?

USER: Alexander's parents were John Simpson and Elizabeth Cruickshank. He had a sister named Isobella who was born in 1840. I found their marriage certificate in the Scottish records. Would you like me to send you a copy?
"""

    # Test case 2: UNINTERESTED intent
    uninterested_conversation = """
SCRIPT: Hello, I'm researching my family tree and we appear to be DNA matches. I'm trying to find our common ancestor. My tree includes the Simpson family from Aberdeen, Scotland in the 1800s. Does that connect with your research?

USER: I don't really work on my family tree much anymore.

SCRIPT: I understand. If you happen to have any information about the Simpson family from Aberdeen, particularly from the 1800s, it could be very helpful. Even small details might help connect our trees.

USER: Sorry, I don't think I can help you. Good luck with your research.
"""

    # Test case 3: For data extraction and task suggestion
    extraction_conversation = """
SCRIPT: Hello, I'm researching my family tree and we appear to be DNA matches. I'm trying to find our common ancestor. My tree includes the Simpson family from Aberdeen, Scotland in the 1800s. Does that connect with your research?

USER: Yes, I have Simpsons in my tree from Aberdeen! My great-grandmother was Margaret Simpson born around 1865. Her father was Alexander Simpson who was a fisherman. He was born about 1835 and died in 1890. The family lived in a fishing village in Aberdeen.

SCRIPT: That's wonderful! Margaret Simpson sounds familiar. Do you know who Alexander's parents were or if he had any siblings?

USER: Alexander's parents were John Simpson and Elizabeth Cruickshank. They married in 1833 in Aberdeen. John was born around 1810 and worked as a fisherman like his son. Elizabeth was from Peterhead originally. Alexander had a sister named Isobella who was born in 1840. I found their marriage certificate in the Scottish records.
"""

    # Step 4: Test intent classification
    print("\n--- Testing Intent Classification ---")

    # Test PRODUCTIVE intent
    print("Testing PRODUCTIVE intent classification...")
    productive_result = classify_message_intent(
        productive_conversation, session_manager
    )
    if productive_result == "PRODUCTIVE":
        print("✓ PRODUCTIVE intent correctly classified")
    else:
        print(f"✗ PRODUCTIVE intent test failed. Got: {productive_result}")
        all_tests_passed = False

    # Test UNINTERESTED intent
    print("Testing UNINTERESTED intent classification...")
    uninterested_result = classify_message_intent(
        uninterested_conversation, session_manager
    )
    if uninterested_result == "UNINTERESTED":
        print("✓ UNINTERESTED intent correctly classified")
    else:
        print(f"✗ UNINTERESTED intent test failed. Got: {uninterested_result}")
        all_tests_passed = False

    # Step 5: Test data extraction and task suggestion
    print("\n--- Testing Data Extraction and Task Suggestion ---")
    extraction_result = extract_and_suggest_tasks(
        extraction_conversation, session_manager
    )

    if extraction_result and isinstance(extraction_result, dict):
        print("✓ Data extraction returned valid result structure")

        # Check for expected data in the result
        extracted_data = extraction_result.get("extracted_data", {})

        # Check for names
        if "mentioned_names" in extracted_data and isinstance(
            extracted_data["mentioned_names"], list
        ):
            names = extracted_data["mentioned_names"]
            if any("Margaret Simpson" in name for name in names) and any(
                "Alexander Simpson" in name for name in names
            ):
                print("✓ Expected names found in extraction result")
            else:
                print("WARNING: Expected names not found in extraction result")
                print(f"Found names: {names}")

        # Check for locations
        if "mentioned_locations" in extracted_data and isinstance(
            extracted_data["mentioned_locations"], list
        ):
            locations = extracted_data["mentioned_locations"]
            if any("Aberdeen" in loc for loc in locations):
                print("✓ Expected locations found in extraction result")
            else:
                print("WARNING: Expected locations not found in extraction result")
                print(f"Found locations: {locations}")

        # Check for dates
        if "mentioned_dates" in extracted_data and isinstance(
            extracted_data["mentioned_dates"], list
        ):
            dates = extracted_data["mentioned_dates"]
            if len(dates) > 0:
                print("✓ Dates found in extraction result")
                print(f"Found dates: {dates}")
            else:
                print("WARNING: No dates found in extraction result")

        # Check for suggested tasks
        if "suggested_tasks" in extraction_result and isinstance(
            extraction_result["suggested_tasks"], list
        ):
            tasks = extraction_result["suggested_tasks"]
            if len(tasks) > 0:
                print(f"✓ {len(tasks)} suggested tasks generated")
                for i, task in enumerate(tasks, 1):
                    print(f"  Task {i}: {task}")
            else:
                print("WARNING: No suggested tasks generated")

        # Print full extraction result for reference
        print("\nFull extraction result:")
        import json

        print(json.dumps(extraction_result, indent=2))

    else:
        print("✗ Data extraction test failed")
        all_tests_passed = False

    # Step 6: Collect test results for summary
    test_results = [
        {
            "name": "PRODUCTIVE Intent Classification",
            "result": productive_result == "PRODUCTIVE",
        },
        {
            "name": "UNINTERESTED Intent Classification",
            "result": uninterested_result == "UNINTERESTED",
        },
        {
            "name": "Data Extraction Valid Structure",
            "result": extraction_result is not None
            and isinstance(extraction_result, dict),
        },
    ]

    # Add extraction detail tests if extraction was successful
    if extraction_result and isinstance(extraction_result, dict):
        extracted_data = extraction_result.get("extracted_data", {})

        # Check for names
        names_found = False
        if "mentioned_names" in extracted_data and isinstance(
            extracted_data["mentioned_names"], list
        ):
            names = extracted_data["mentioned_names"]
            names_found = any("Margaret Simpson" in name for name in names) and any(
                "Alexander Simpson" in name for name in names
            )
        test_results.append(
            {"name": "Expected Names Extraction", "result": names_found}
        )

        # Check for locations
        locations_found = False
        if "mentioned_locations" in extracted_data and isinstance(
            extracted_data["mentioned_locations"], list
        ):
            locations = extracted_data["mentioned_locations"]
            locations_found = any("Aberdeen" in loc for loc in locations)
        test_results.append(
            {"name": "Expected Locations Extraction", "result": locations_found}
        )

        # Check for dates
        dates_found = False
        if "mentioned_dates" in extracted_data and isinstance(
            extracted_data["mentioned_dates"], list
        ):
            dates = extracted_data["mentioned_dates"]
            dates_found = len(dates) > 0
        test_results.append({"name": "Dates Extraction", "result": dates_found})

        # Check for tasks
        tasks_generated = False
        task_count = 0
        if "suggested_tasks" in extraction_result and isinstance(
            extraction_result["suggested_tasks"], list
        ):
            tasks = extraction_result["suggested_tasks"]
            task_count = len(tasks)
            tasks_generated = task_count > 0
        test_results.append(
            {"name": f"Task Generation ({task_count} tasks)", "result": tasks_generated}
        )

    # Step 7: Display test summary
    passed_tests = sum(1 for test in test_results if test["result"])
    total_tests = len(test_results)

    print("\n=== AI Interface Self-Test Summary ===")
    print(f"{'Test Name':<40} {'Result':<10}")
    print("-" * 50)

    for test in test_results:
        result_str = "✓ PASSED" if test["result"] else "✗ FAILED"
        print(f"{test['name']:<40} {result_str:<10}")

    print("-" * 50)
    print(f"{'Overall Statistics':<40}")
    print(f"{'Total Tests':<30}: {total_tests}")
    print(f"{'Passed':<30}: {passed_tests}")
    print(f"{'Failed':<30}: {total_tests - passed_tests}")
    print(f"{'Success Rate':<30}: {(passed_tests / total_tests) * 100:.0f}%")

    # Final result
    if all_tests_passed:
        print("\n=== All AI Interface tests PASSED ===")
    else:
        print("\n=== Some AI Interface tests FAILED ===")

    return all_tests_passed


# End of self_check


# Add main block to run self-test when script is executed directly
if __name__ == "__main__":
    import os

    print(f"\nRunning self-check for {os.path.basename(__file__)}")
    success = self_check()

    # Exit with appropriate code
    sys.exit(0 if success else 1)

# Functions for using custom prompts


def extract_with_custom_prompt(
    context_history: str, custom_prompt: str
) -> Optional[Dict[str, Any]]:
    """
    Calls the configured AI model with a custom prompt to extract genealogical entities.

    Args:
        context_history: The formatted conversation history string.
        custom_prompt: The custom system prompt to use.

    Returns:
        A dictionary containing 'extracted_data' if successful, otherwise None.
    """
    # Use the session manager from config
    from utils import SessionManager

    session_manager = SessionManager()

    # Step 1: Validate inputs and configuration
    ai_provider = config_instance.AI_PROVIDER
    if not ai_provider:
        logger.error("extract_with_custom_prompt: AI_PROVIDER not configured.")
        return None
    if not context_history:
        logger.warning(
            "extract_with_custom_prompt: Received empty context history. Cannot extract."
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
    if session_manager is not None and hasattr(session_manager, "dynamic_rate_limiter"):
        wait_time = session_manager.dynamic_rate_limiter.wait()

    # Step 3: Initialize result and timer
    extraction_result: Optional[Dict[str, Any]] = None
    start_time = time.time()
    max_tokens_extraction = 700  # Allow more tokens for potentially detailed JSON

    # Step 4: Call the appropriate AI provider
    try:
        if ai_provider == "deepseek":
            # --- DeepSeek/OpenAI Compatible API Call ---
            if not openai_available or OpenAI is None:
                logger.error(
                    "extract_with_custom_prompt: OpenAI library not available."
                )
                return None
            # Load config
            api_key = config_instance.DEEPSEEK_API_KEY
            model = config_instance.DEEPSEEK_AI_MODEL
            base_url = config_instance.DEEPSEEK_AI_BASE_URL
            if not all([api_key, model, base_url]):
                logger.error(
                    "extract_with_custom_prompt: DeepSeek configuration incomplete."
                )
                return None
            # Create client and make request (requesting JSON object)
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.debug(
                f"Calling DeepSeek Extraction with custom prompt (Model: {model})..."
            )
            openai_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": custom_prompt},
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
                # Log the raw response for debugging
                logger.info(f"Raw AI response: {raw_response_content[:500]}...")

                # Attempt to parse the JSON response
                try:
                    # Try to clean up the response if it's not valid JSON
                    cleaned_response = raw_response_content.strip()
                    # If response starts with ``` or ```json, remove it
                    if cleaned_response.startswith("```"):
                        # Find the end of the code block
                        end_marker = "```"
                        end_pos = cleaned_response.rfind(end_marker)
                        if end_pos > 3:  # Make sure we found the end marker
                            # Extract just the content between the markers
                            start_pos = cleaned_response.find(
                                "\n", 3
                            )  # Skip the ```json line
                            if (
                                start_pos == -1
                            ):  # If no newline after ```, use position 3
                                start_pos = 3
                            cleaned_response = cleaned_response[
                                start_pos:end_pos
                            ].strip()
                            logger.info(
                                f"Extracted JSON from code block: {cleaned_response[:200]}..."
                            )

                    # Try to find JSON object in the response if it's not a complete JSON
                    if not cleaned_response.startswith("{"):
                        start_pos = cleaned_response.find("{")
                        if start_pos != -1:
                            end_pos = cleaned_response.rfind("}")
                            if end_pos > start_pos:
                                cleaned_response = cleaned_response[
                                    start_pos : end_pos + 1
                                ]
                                logger.info(
                                    f"Extracted JSON object from response: {cleaned_response[:200]}..."
                                )

                    # Now try to parse the cleaned response
                    parsed_json = json.loads(cleaned_response)
                    logger.info(f"Successfully parsed JSON response")

                    # Validate the structure of the parsed JSON
                    if isinstance(parsed_json, dict):
                        # If the response doesn't have the expected structure, add it
                        if "extracted_data" not in parsed_json:
                            logger.warning(
                                "Adding missing 'extracted_data' field to AI response"
                            )
                            parsed_json["extracted_data"] = {
                                "mentioned_names": [],
                                "mentioned_locations": [],
                                "mentioned_dates": [],
                                "potential_relationships": [],
                                "key_facts": [],
                            }
                        elif not isinstance(parsed_json["extracted_data"], dict):
                            logger.warning(
                                "'extracted_data' is not a dictionary, replacing with default structure"
                            )
                            parsed_json["extracted_data"] = {
                                "mentioned_names": [],
                                "mentioned_locations": [],
                                "mentioned_dates": [],
                                "potential_relationships": [],
                                "key_facts": [],
                            }

                        # Ensure all required fields exist in extracted_data
                        for field in [
                            "mentioned_names",
                            "mentioned_locations",
                            "mentioned_dates",
                            "potential_relationships",
                            "key_facts",
                        ]:
                            if field not in parsed_json["extracted_data"]:
                                logger.warning(
                                    f"Adding missing '{field}' field to extracted_data"
                                )
                                parsed_json["extracted_data"][field] = []
                            elif not isinstance(
                                parsed_json["extracted_data"][field], list
                            ):
                                logger.warning(
                                    f"'{field}' is not a list, replacing with empty list"
                                )
                                parsed_json["extracted_data"][field] = []

                        # If suggested_tasks is missing, add it
                        if "suggested_tasks" not in parsed_json:
                            logger.warning(
                                "Adding missing 'suggested_tasks' field to AI response"
                            )
                            parsed_json["suggested_tasks"] = []
                        elif not isinstance(parsed_json["suggested_tasks"], list):
                            logger.warning(
                                "'suggested_tasks' is not a list, replacing with empty list"
                            )
                            parsed_json["suggested_tasks"] = []

                        extraction_result = parsed_json  # Valid structure
                    else:
                        logger.warning(
                            f"DeepSeek extraction response is valid JSON but not a dictionary: {raw_response_content}"
                        )
                        # Create a default structure
                        extraction_result = {
                            "extracted_data": {
                                "mentioned_names": [],
                                "mentioned_locations": [],
                                "mentioned_dates": [],
                                "potential_relationships": [],
                                "key_facts": [],
                            },
                            "suggested_tasks": [],
                        }
                except json.JSONDecodeError as e:
                    logger.error(
                        f"DeepSeek extraction response was not valid JSON: {e}\nContent: {raw_response_content}"
                    )
                    # Create a default structure
                    extraction_result = {
                        "extracted_data": {
                            "mentioned_names": [],
                            "mentioned_locations": [],
                            "mentioned_dates": [],
                            "potential_relationships": [],
                            "key_facts": [],
                        },
                        "suggested_tasks": [],
                    }
            else:
                logger.error(
                    "Invalid response structure received from DeepSeek API (Extraction)."
                )
                # Create a default structure
                extraction_result = {
                    "extracted_data": {
                        "mentioned_names": [],
                        "mentioned_locations": [],
                        "mentioned_dates": [],
                        "potential_relationships": [],
                        "key_facts": [],
                    },
                    "suggested_tasks": [],
                }

        # Add support for other providers as needed

    except Exception as e:
        logger.error(
            f"Unexpected error during AI extraction with custom prompt: {type(e).__name__} - {e}",
            exc_info=True,
        )
        # Create a default structure
        extraction_result = {
            "extracted_data": {
                "mentioned_names": [],
                "mentioned_locations": [],
                "mentioned_dates": [],
                "potential_relationships": [],
                "key_facts": [],
            },
            "suggested_tasks": [],
        }

    # Log duration and result
    duration = time.time() - start_time
    if extraction_result:
        logger.debug(
            f"AI extraction with custom prompt completed successfully (Took {duration:.2f}s)"
        )
    else:
        logger.error(f"AI extraction with custom prompt failed (Took {duration:.2f}s)")
        # Create a default structure if we somehow still don't have a result
        extraction_result = {
            "extracted_data": {
                "mentioned_names": [],
                "mentioned_locations": [],
                "mentioned_dates": [],
                "potential_relationships": [],
                "key_facts": [],
            },
            "suggested_tasks": [],
        }

    return extraction_result


def generate_with_custom_prompt(
    conversation_context: str,
    user_last_message: str,
    genealogical_data_str: str,
    custom_prompt: str,
) -> Optional[str]:
    """
    Generates a genealogical reply using a custom prompt.

    Args:
        conversation_context: The full conversation history.
        user_last_message: The user's last message.
        genealogical_data_str: Formatted genealogical data string.
        custom_prompt: The custom system prompt to use.

    Returns:
        A generated response string if successful, otherwise None.
    """
    # Use the session manager from config
    from utils import SessionManager

    session_manager = SessionManager()

    # Step 1: Validate inputs and configuration
    ai_provider = config_instance.AI_PROVIDER
    if not ai_provider:
        logger.error("generate_with_custom_prompt: AI_PROVIDER not configured.")
        return None
    if not conversation_context or not genealogical_data_str:
        logger.warning("generate_with_custom_prompt: Missing required inputs.")
        return None

    # If no custom prompt is provided, try to get it from the JSON file
    if not custom_prompt:
        logger.warning(
            "generate_with_custom_prompt: No custom prompt provided, using default."
        )
        # Try to get the prompt from the JSON file first
        if USE_JSON_PROMPTS:
            try:
                json_prompt = get_prompt("genealogical_reply")
                if json_prompt:
                    custom_prompt = json_prompt
                    logger.debug("Using genealogical reply prompt from ai_prompts.json")
                else:
                    custom_prompt = GENERATE_GENEALOGICAL_REPLY_PROMPT
            except Exception as e:
                logger.error(f"Error getting genealogical reply prompt from JSON: {e}")
                custom_prompt = GENERATE_GENEALOGICAL_REPLY_PROMPT
        else:
            custom_prompt = GENERATE_GENEALOGICAL_REPLY_PROMPT

    # Step 2: Apply Rate Limiting
    if session_manager is not None and hasattr(session_manager, "dynamic_rate_limiter"):
        wait_time = session_manager.dynamic_rate_limiter.wait()

    # Step 3: Initialize result and timer
    generation_result: Optional[str] = None
    start_time = time.time()
    max_tokens_generation = 1000  # Allow more tokens for detailed response

    # Step 4: Check if the custom prompt already contains placeholders
    has_placeholders = (
        "{conversation_context}" in custom_prompt
        and "{user_message}" in custom_prompt
        and "{genealogical_data}" in custom_prompt
    )

    # Check if the prompt contains an {intent_classification} placeholder
    if has_placeholders and "{intent_classification}" in custom_prompt:
        logger.info(
            "Detected {intent_classification} placeholder in prompt, getting intent classification"
        )

        # First try to get the intent classification from the classify_message_intent function
        try:
            from utils import SessionManager

            intent_classification = classify_message_intent(
                conversation_context, SessionManager()
            )
            logger.info(
                f"Got intent classification from classify_message_intent: {intent_classification}"
            )
        except Exception as e:
            logger.warning(
                f"Error getting intent classification from classify_message_intent: {e}"
            )
            intent_classification = None

        # If we couldn't get the intent classification from the function, infer it from the message content
        if not intent_classification:
            logger.info("Inferring intent classification from message content")
            # Try to infer intent from the message content
            # This is a simple heuristic and could be improved
            intent_classification = "ENTHUSIASTIC"  # Default to enthusiastic

            # Look for signals of different intents in the user's message
            lower_message = user_last_message.lower()

            # Check for uninterested signals
            uninterested_phrases = [
                "not interested",
                "stop messaging",
                "leave me alone",
                "don't contact",
            ]
            if any(phrase in lower_message for phrase in uninterested_phrases):
                intent_classification = "UNINTERESTED"

            # Check for confused signals
            confused_phrases = [
                "who are you",
                "why am i getting",
                "what is this about",
                "don't understand",
            ]
            if any(phrase in lower_message for phrase in confused_phrases):
                intent_classification = "CONFUSED"

            # Check for cautious signals
            cautious_phrases = [
                "not sure",
                "maybe",
                "possibly",
                "might be",
                "don't know if",
            ]
            if any(phrase in lower_message for phrase in cautious_phrases):
                intent_classification = "CAUTIOUSLY_INTERESTED"

            # If the message is very short, it's likely "OTHER"
            if len(user_last_message.split()) < 5:
                intent_classification = "OTHER"

            logger.info(f"Inferred intent classification: {intent_classification}")

        # Replace all placeholders including intent_classification
        full_prompt = custom_prompt.replace(
            "{conversation_context}", conversation_context
        )
        full_prompt = full_prompt.replace("{user_message}", user_last_message)
        full_prompt = full_prompt.replace("{genealogical_data}", genealogical_data_str)
        full_prompt = full_prompt.replace(
            "{intent_classification}", intent_classification
        )
    elif has_placeholders:
        # The prompt has standard placeholders, so we need to replace them
        logger.debug(
            "Using custom prompt with placeholders, replacing them with actual values"
        )
        full_prompt = custom_prompt.replace(
            "{conversation_context}", conversation_context
        )
        full_prompt = full_prompt.replace("{user_message}", user_last_message)
        full_prompt = full_prompt.replace("{genealogical_data}", genealogical_data_str)
    else:
        # The prompt doesn't have placeholders, so we'll add them
        logger.debug("Using custom prompt without placeholders, adding context")
        full_prompt = f"""
{custom_prompt}

CONVERSATION HISTORY:
{conversation_context}

USER'S LAST MESSAGE:
{user_last_message}

GENEALOGICAL DATA:
{genealogical_data_str}
"""

    # Step 5: Call the appropriate AI provider
    try:
        if ai_provider == "deepseek":
            # --- DeepSeek/OpenAI Compatible API Call ---
            if not openai_available or OpenAI is None:
                logger.error(
                    "generate_with_custom_prompt: OpenAI library not available."
                )
                return None
            # Load config
            api_key = config_instance.DEEPSEEK_API_KEY
            model = config_instance.DEEPSEEK_AI_MODEL
            base_url = config_instance.DEEPSEEK_AI_BASE_URL
            if not all([api_key, model, base_url]):
                logger.error(
                    "generate_with_custom_prompt: DeepSeek configuration incomplete."
                )
                return None
            # Create client and make request
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.debug(
                f"Calling DeepSeek Generation with custom prompt (Model: {model})..."
            )

            # If the custom prompt already has placeholders, we've already replaced them in full_prompt
            # So we should use that as the system message
            openai_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": full_prompt},
                    # No need for a separate user message since the full_prompt already contains everything
                ],
                stream=False,
                max_tokens=max_tokens_generation,
                temperature=0.7,  # Allow more creativity for natural responses
            )
            # Process response
            if openai_response.choices and openai_response.choices[0].message:
                generation_result = openai_response.choices[0].message.content.strip()
            else:
                logger.error(
                    "Invalid response structure received from DeepSeek API (Generation)."
                )

        # Add support for other providers as needed

    except Exception as e:
        logger.error(
            f"Unexpected error during AI generation with custom prompt: {type(e).__name__} - {e}",
            exc_info=True,
        )

    # Log duration and result
    duration = time.time() - start_time
    if generation_result:
        logger.debug(
            f"AI generation with custom prompt completed successfully (Took {duration:.2f}s)"
        )
    else:
        logger.error(f"AI generation with custom prompt failed (Took {duration:.2f}s)")

    return generation_result


# End of ai_interface.py
