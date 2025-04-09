#!/usr/bin/env python3

# ai_test.py

##########################################################################
# Imports
##########################################################################

import sys
import logging
import time

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
except ImportError:
    genai = None
    google_exceptions = None

# --- Local application imports ---
from config import config_instance
from logging_config import setup_logging

# Import SessionManager to access the rate limiter
from utils import SessionManager

##########################################################################
# Initialise
##########################################################################

# Setup logging
log_filename = "ai_test.log"
try:
    log_filename = config_instance.DATABASE_FILE.with_suffix(".log").name
except Exception as config_log_err:
    print(
        f"Warning: Could not get log filename from config: {config_log_err}",
        file=sys.stderr,
    )

logger = setup_logging(log_file=log_filename, log_level="DEBUG")

##########################################################################
# Constants and Configuration Loading
##########################################################################

ai_provider = config_instance.AI_PROVIDER

# --- Configuration Checks ---
if not ai_provider:
    logger.critical("AI_PROVIDER not configured in .env. Exiting test.")
    sys.exit(1)

logger.info(f"--- Starting AI Classification Test ---")
logger.info(f"Selected AI Provider: {ai_provider}")

# --- Load Provider Specific Config ---
ai_model_name = None
ai_api_key = None
ai_base_url = None  # Only used by DeepSeek

if ai_provider == "deepseek":
    ai_model_name = config_instance.DEEPSEEK_AI_MODEL
    ai_api_key = config_instance.DEEPSEEK_API_KEY
    ai_base_url = config_instance.DEEPSEEK_AI_BASE_URL
    if not ai_api_key:
        logger.critical("DEEPSEEK_API_KEY not set.")
        sys.exit(1)
    if not ai_model_name:
        logger.critical("DEEPSEEK_AI_MODEL not set.")
        sys.exit(1)
    if not ai_base_url:
        logger.critical("DEEPSEEK_AI_BASE_URL not set.")
        sys.exit(1)
    logger.info(f"Using DeepSeek Model: {ai_model_name}")
    logger.info(f"Using DeepSeek Base URL: {ai_base_url}")
elif ai_provider == "gemini":
    ai_model_name = config_instance.GOOGLE_AI_MODEL
    ai_api_key = config_instance.GOOGLE_API_KEY
    if not ai_api_key:
        logger.critical("GOOGLE_API_KEY not set.")
        sys.exit(1)
    if not ai_model_name:
        logger.critical("GOOGLE_AI_MODEL not set.")
        sys.exit(1)
    logger.info(f"Using Google AI Model: {ai_model_name}")
else:
    logger.critical(f"Unsupported AI_PROVIDER: {ai_provider}")
    sys.exit(1)

# --- System Prompt & Test Data ---
SYSTEM_PROMPT = """You are an AI assistant designed to classify messages from DNA matches on a genealogy website. Analyze the user's message and determine the primary intent. Respond ONLY with one of the following single-word categories:

DESIST: The user explicitly asks to stop receiving messages or indicates they are blocking the sender.
UNINTERESTED: The user politely declines further contact, states they cannot help, don't have time, are not knowledgeable about their family tree, or shows a clear lack of engagement.
PRODUCTIVE: The user provides helpful information, asks relevant questions about shared ancestors, confirms a relationship, expresses clear interest in collaborating, or shares tree information.
OTHER: The message does not clearly fall into the DESIST, UNINTERESTED, or PRODUCTIVE categories (e.g., simple greetings, unclear statements, off-topic).

Output only the single category word."""
EXPECTED_CATEGORIES = {"DESIST", "UNINTERESTED", "PRODUCTIVE", "OTHER"}
test_messages = [
    "Please stop contacting me.",
    "I'm afraid I don't have time to look into this right now.",
    "Thanks for reaching out, but I don't really manage my tree actively.",
    "I don't know much about my family history beyond my grandparents.",
    "Block.",
    "Hi! Thanks for the message. Yes, I recognise the Smith family connection you mentioned! My great-grandmother was Mary Smith.",
    "Hello there! Interesting match. Where did your McGregor ancestors live?",
    "Hey",
    "Good morning!",
    "Fuck off",
    "Sorry, I can't help you with this.",
    "I am not the person who manages this account.",
    "My tree is private.",
    "Can you tell me more about our shared connection?",
    "Why are you messaging me?",
    "OMG, i am so keen, i want to stalk you.",
    "Thank you for the information about the common ancestor! That helps clarify things.",
    "Remove me from your list.",
    "I have received your previous messages, but I won't be able to assist with your research. Best of luck.",
]


##########################################################################
# Setup AI Client & Rate Limiter
##########################################################################

deepseek_client = None
gemini_model_obj = None

# --- Initialize SessionManager to get the rate limiter ---
# Note: This doesn't start a browser session, just creates the object
test_session_manager = SessionManager()
rate_limiter = test_session_manager.dynamic_rate_limiter
logger.debug("DynamicRateLimiter obtained from SessionManager instance.")

# --- Initialize AI Client ---
if ai_provider == "deepseek":
    if OpenAI is None:
        logger.critical("OpenAI library not installed. Run 'pip install openai'.")
        sys.exit(1)
    try:
        deepseek_client = OpenAI(api_key=ai_api_key, base_url=ai_base_url)
        logger.debug("DeepSeek OpenAI client initialized.")
    except Exception as e:
        logger.critical(
            f"Failed to initialize DeepSeek OpenAI client: {e}", exc_info=True
        )
        sys.exit(1)

elif ai_provider == "gemini":
    # --- Gemini Initialization (Using GenerativeModel) ---
    # Perform diagnostic checks first
    genai_available = True
    if genai is None or google_exceptions is None:
        genai_available = False
        logger.critical("google-generativeai library not fully installed/imported.")
    else:
        if not hasattr(genai, "configure"):
            logging.error(
                "FATAL: google.generativeai module loaded, but does NOT have attribute 'configure'."
            )
            genai_available = False
        if not hasattr(genai, "GenerativeModel"):
            logging.error(
                "FATAL: google.generativeai module loaded, but does NOT have attribute 'GenerativeModel'."
            )
            genai_available = False

    if not genai_available:
        logger.critical(
            "Cannot initialize Gemini client due to import/attribute errors."
        )
        sys.exit(1)

    try:
        genai.configure(api_key=ai_api_key)
        gemini_model_obj = genai.GenerativeModel(ai_model_name)
        logger.debug("Google Gemini GenerativeModel initialized.")
    except AttributeError as ae:
        logger.critical(
            f"AttributeError during Gemini initialization: {ae}. Please check google-generativeai installation.",
            exc_info=True,
        )
        sys.exit(1)
    except Exception as e:
        logger.critical(
            f"Failed to initialize Google Gemini GenerativeModel: {e}", exc_info=True
        )
        sys.exit(1)


##########################################################################
# Test Classification
##########################################################################

logger.info("Starting classification tests...")
print("-" * 60)

successful_classifications = 0
error_classifications = 0

for i, message in enumerate(test_messages):
    logger.debug(f"Testing message {i+1}/{len(test_messages)}...")
    print(f'Message: "{message}"')

    classification = "ERROR (Provider Logic)"  # Default
    try:
        # --- Apply Rate Limiting ---
        wait_time = rate_limiter.wait()
        if wait_time > 0.1:  # Only log significant waits
            logger.debug(f"Rate limit wait: {wait_time:.2f}s")
        # --- End Rate Limiting ---

        # --- DeepSeek API Call ---
        if ai_provider == "deepseek" and deepseek_client:
            openai_response = deepseek_client.chat.completions.create(
                model=ai_model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
                stream=False,
                max_tokens=20,
                temperature=0.1,
            )
            if openai_response.choices and openai_response.choices[0].message:
                raw_classification = openai_response.choices[0].message.content.strip()
                if raw_classification in EXPECTED_CATEGORIES:
                    classification = raw_classification
                    rate_limiter.decrease_delay()  # Signal success to limiter
                else:
                    logger.warning(
                        f"AI returned unexpected classification: '{raw_classification}'"
                    )
                    classification = f"UNEXPECTED ({raw_classification})"
                    # Treat unexpected as success for rate limiter? Or error? Let's count as success for now.
                    rate_limiter.decrease_delay()
            else:
                logger.error("Invalid response structure from DeepSeek API.")
                classification = "ERROR (Invalid Response Structure)"
                rate_limiter.increase_delay()  # Treat error as potential rate limit trigger

        # --- Gemini API Call ---
        elif ai_provider == "gemini" and gemini_model_obj:
            generation_config = {
                "candidate_count": 1,
                "max_output_tokens": 20,
                "temperature": 0.1,
            }
            prompt_content = f"{SYSTEM_PROMPT}\n\nUser Message:\n{message}"

            response = gemini_model_obj.generate_content(
                prompt_content,
                generation_config=generation_config,
            )

            if not response.candidates:
                block_reason = "Unknown"
                try:
                    block_reason = response.prompt_feedback.block_reason.name
                except Exception:
                    pass
                logger.warning(
                    f"Gemini response blocked or empty. Reason: {block_reason}"
                )
                classification = "OTHER (Blocked/Empty)"
                rate_limiter.decrease_delay()  # Treat block as success for limiter
            else:
                raw_classification = response.text.strip()
                if raw_classification in EXPECTED_CATEGORIES:
                    classification = raw_classification
                    rate_limiter.decrease_delay()  # Signal success
                else:
                    logger.warning(
                        f"AI returned unexpected classification: '{raw_classification}'"
                    )
                    classification = f"UNEXPECTED ({raw_classification})"
                    rate_limiter.decrease_delay()  # Treat unexpected as success

    # --- Consolidated Error Handling ---
    except AuthenticationError as e:
        if AuthenticationError:
            logger.error(f"Authentication Error: Invalid API Key or Base URL? {e}")
        else:
            logger.error(
                f"AI API Error ({ai_provider}): Authentication Issue (Details: {e})"
            )
        classification = "ERROR (Authentication)"
        rate_limiter.increase_delay()  # Increase delay on errors too
        # break # Optional: Stop testing if auth fails
    except RateLimitError as e:
        if RateLimitError:
            logger.error(f"Rate Limit Exceeded: {e}")
        else:
            logger.error(
                f"AI API Error ({ai_provider}): Rate Limit Issue (Details: {e})"
            )
        classification = "ERROR (Rate Limit)"
        rate_limiter.increase_delay()  # Explicitly increase delay
        time.sleep(5)  # Add extra wait on rate limit error
    except APIConnectionError as e:
        if APIConnectionError:
            logger.error(f"Connection Error: Could not connect to API. {e}")
        else:
            logger.error(
                f"AI API Error ({ai_provider}): Connection Issue (Details: {e})"
            )
        classification = "ERROR (Connection)"
        rate_limiter.increase_delay()
    except APIError as e:
        if APIError:
            logger.error(f"API Error: {e.status_code} - {e.message}")
        else:
            logger.error(
                f"AI API Error ({ai_provider}): General API Issue (Details: {e})"
            )
        classification = f"ERROR (API - {e.status_code})"
        rate_limiter.increase_delay()
    except google_exceptions.PermissionDenied as e:
        if google_exceptions:
            logger.error(f"AI API Permission Denied (Gemini): Invalid API Key? {e}")
        else:
            logger.error(f"AI API Error (Gemini): Permission Issue (Details: {e})")
        classification = "ERROR (Authentication)"
        rate_limiter.increase_delay()
        # break
    except google_exceptions.ResourceExhausted as e:
        if google_exceptions:
            logger.error(f"AI API Rate Limit Exceeded (Gemini): {e}")
        else:
            logger.error(f"AI API Error (Gemini): Rate Limit Issue (Details: {e})")
        classification = "ERROR (Rate Limit)"
        rate_limiter.increase_delay()  # Explicitly increase delay
        time.sleep(5)  # Add extra wait
    except google_exceptions.GoogleAPIError as e:
        if google_exceptions:
            logger.error(f"Google API Error (Gemini): {e}")
        else:
            logger.error(f"AI API Error (Gemini): General API Issue (Details: {e})")
        classification = f"ERROR (API - Gemini)"
        rate_limiter.increase_delay()
    except AttributeError as ae:
        if ai_provider == "gemini":
            logger.critical(
                f"AttributeError using Gemini library: {ae}. Likely installation issue or incorrect usage.",
                exc_info=True,
            )
            classification = "ERROR (AttributeError)"
        else:
            logger.error(
                f"An unexpected AttributeError occurred ({ai_provider}): {ae}",
                exc_info=True,
            )
            classification = "ERROR (AttributeError)"
        rate_limiter.increase_delay()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        classification = "ERROR (Unexpected)"
        rate_limiter.increase_delay()  # Increase delay on unexpected errors

    # --- Tally Results ---
    if "ERROR" in classification:
        error_classifications += 1
    else:
        successful_classifications += 1

    print(f"AI Classification: ---> {classification} <---")
    print(
        f"Current Rate Limiter Delay: {rate_limiter.current_delay:.2f}s"
    )  # Show current delay
    print("-" * 60)
    # Removed explicit sleep(1) as rate_limiter.wait() handles delays

# --- Final Summary ---
logger.info("--- AI Classification Test Finished ---")
logger.info(f"Total Messages Tested: {len(test_messages)}")
logger.info(
    f"Successful Classifications (incl. UNEXPECTED/OTHER): {successful_classifications}"
)
logger.info(f"Failed Classifications (Errors): {error_classifications}")
logger.info(f"Final Rate Limiter Delay: {rate_limiter.current_delay:.2f}s")
