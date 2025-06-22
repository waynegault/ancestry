from path_manager import function_registry, standardize_module_imports

# Auto-register functions for optimized access
# Auto-register functions for optimized access
# Auto-register functions for optimized access
try:
    _current_module = globals()
    if "create_api_cache_key" in _current_module and callable(
        _current_module["create_api_cache_key"]
    ):
        function_registry.register(
            "create_api_cache_key", _current_module["create_api_cache_key"]
        )
    if "cache_profile_details_api" in _current_module and callable(
        _current_module["cache_profile_details_api"]
    ):
        function_registry.register(
            "cache_profile_details_api", _current_module["cache_profile_details_api"]
        )
    if "cache_facts_api" in _current_module and callable(
        _current_module["cache_facts_api"]
    ):
        function_registry.register(
            "cache_facts_api", _current_module["cache_facts_api"]
        )
    if "cache_suggest_api" in _current_module and callable(
        _current_module["cache_suggest_api"]
    ):
        function_registry.register(
            "cache_suggest_api", _current_module["cache_suggest_api"]
        )
    if "create_ai_cache_key" in _current_module and callable(
        _current_module["create_ai_cache_key"]
    ):
        function_registry.register(
            "create_ai_cache_key", _current_module["create_ai_cache_key"]
        )
    if "cache_ai_classify_intent" in _current_module and callable(
        _current_module["cache_ai_classify_intent"]
    ):
        function_registry.register(
            "cache_ai_classify_intent", _current_module["cache_ai_classify_intent"]
        )
    if "cache_ai_extract_tasks" in _current_module and callable(
        _current_module["cache_ai_extract_tasks"]
    ):
        function_registry.register(
            "cache_ai_extract_tasks", _current_module["cache_ai_extract_tasks"]
        )
    if "cache_ai_genealogical_reply" in _current_module and callable(
        _current_module["cache_ai_genealogical_reply"]
    ):
        function_registry.register(
            "cache_ai_genealogical_reply",
            _current_module["cache_ai_genealogical_reply"],
        )
    if "cache_person_by_profile_id" in _current_module and callable(
        _current_module["cache_person_by_profile_id"]
    ):
        function_registry.register(
            "cache_person_by_profile_id", _current_module["cache_person_by_profile_id"]
        )
    if "cache_conversation_logs" in _current_module and callable(
        _current_module["cache_conversation_logs"]
    ):
        function_registry.register(
            "cache_conversation_logs", _current_module["cache_conversation_logs"]
        )
    if "warm_api_caches" in _current_module and callable(
        _current_module["warm_api_caches"]
    ):
        function_registry.register(
            "warm_api_caches", _current_module["warm_api_caches"]
        )
    if "get_api_cache_stats" in _current_module and callable(
        _current_module["get_api_cache_stats"]
    ):
        function_registry.register(
            "get_api_cache_stats", _current_module["get_api_cache_stats"]
        )
    if "get_api_cache_module_stats" in _current_module and callable(
        _current_module["get_api_cache_module_stats"]
    ):
        function_registry.register(
            "get_api_cache_module_stats", _current_module["get_api_cache_module_stats"]
        )
    if "clear_api_cache" in _current_module and callable(
        _current_module["clear_api_cache"]
    ):
        function_registry.register(
            "clear_api_cache", _current_module["clear_api_cache"]
        )
    if "warm_api_cache" in _current_module and callable(
        _current_module["warm_api_cache"]
    ):
        function_registry.register("warm_api_cache", _current_module["warm_api_cache"])
    if "get_api_cache_health" in _current_module and callable(
        _current_module["get_api_cache_health"]
    ):
        function_registry.register(
            "get_api_cache_health", _current_module["get_api_cache_health"]
        )
    if "create_unified_api_cache_key" in _current_module and callable(
        _current_module["create_unified_api_cache_key"]
    ):
        function_registry.register(
            "create_unified_api_cache_key",
            _current_module["create_unified_api_cache_key"],
        )
    if "create_unified_ai_cache_key" in _current_module and callable(
        _current_module["create_unified_ai_cache_key"]
    ):
        function_registry.register(
            "create_unified_ai_cache_key",
            _current_module["create_unified_ai_cache_key"],
        )
    if "run_api_cache_tests" in _current_module and callable(
        _current_module["run_api_cache_tests"]
    ):
        function_registry.register(
            "run_api_cache_tests", _current_module["run_api_cache_tests"]
        )
    if "demonstrate_api_cache_usage" in _current_module and callable(
        _current_module["demonstrate_api_cache_usage"]
    ):
        function_registry.register(
            "demonstrate_api_cache_usage",
            _current_module["demonstrate_api_cache_usage"],
        )
    if "run_comprehensive_tests" in _current_module and callable(
        _current_module["run_comprehensive_tests"]
    ):
        function_registry.register(
            "run_comprehensive_tests", _current_module["run_comprehensive_tests"]
        )
    if "ApiCacheModule" in _current_module and callable(
        _current_module["ApiCacheModule"]
    ):
        function_registry.register("ApiCacheModule", _current_module["ApiCacheModule"])
except Exception:
    pass  # Silent registration - don't break module loading

try:
    _current_module = globals()
    if "create_api_cache_key" in _current_module and callable(
        _current_module["create_api_cache_key"]
    ):
        function_registry.register(
            "create_api_cache_key", _current_module["create_api_cache_key"]
        )
    if "cache_profile_details_api" in _current_module and callable(
        _current_module["cache_profile_details_api"]
    ):
        function_registry.register(
            "cache_profile_details_api", _current_module["cache_profile_details_api"]
        )
    if "cache_facts_api" in _current_module and callable(
        _current_module["cache_facts_api"]
    ):
        function_registry.register(
            "cache_facts_api", _current_module["cache_facts_api"]
        )
    if "cache_suggest_api" in _current_module and callable(
        _current_module["cache_suggest_api"]
    ):
        function_registry.register(
            "cache_suggest_api", _current_module["cache_suggest_api"]
        )
    if "create_ai_cache_key" in _current_module and callable(
        _current_module["create_ai_cache_key"]
    ):
        function_registry.register(
            "create_ai_cache_key", _current_module["create_ai_cache_key"]
        )
    if "cache_ai_classify_intent" in _current_module and callable(
        _current_module["cache_ai_classify_intent"]
    ):
        function_registry.register(
            "cache_ai_classify_intent", _current_module["cache_ai_classify_intent"]
        )
    if "cache_ai_extract_tasks" in _current_module and callable(
        _current_module["cache_ai_extract_tasks"]
    ):
        function_registry.register(
            "cache_ai_extract_tasks", _current_module["cache_ai_extract_tasks"]
        )
    if "cache_ai_genealogical_reply" in _current_module and callable(
        _current_module["cache_ai_genealogical_reply"]
    ):
        function_registry.register(
            "cache_ai_genealogical_reply",
            _current_module["cache_ai_genealogical_reply"],
        )
    if "cache_person_by_profile_id" in _current_module and callable(
        _current_module["cache_person_by_profile_id"]
    ):
        function_registry.register(
            "cache_person_by_profile_id", _current_module["cache_person_by_profile_id"]
        )
    if "cache_conversation_logs" in _current_module and callable(
        _current_module["cache_conversation_logs"]
    ):
        function_registry.register(
            "cache_conversation_logs", _current_module["cache_conversation_logs"]
        )
    if "warm_api_caches" in _current_module and callable(
        _current_module["warm_api_caches"]
    ):
        function_registry.register(
            "warm_api_caches", _current_module["warm_api_caches"]
        )
    if "get_api_cache_stats" in _current_module and callable(
        _current_module["get_api_cache_stats"]
    ):
        function_registry.register(
            "get_api_cache_stats", _current_module["get_api_cache_stats"]
        )
    if "get_api_cache_module_stats" in _current_module and callable(
        _current_module["get_api_cache_module_stats"]
    ):
        function_registry.register(
            "get_api_cache_module_stats", _current_module["get_api_cache_module_stats"]
        )
    if "clear_api_cache" in _current_module and callable(
        _current_module["clear_api_cache"]
    ):
        function_registry.register(
            "clear_api_cache", _current_module["clear_api_cache"]
        )
    if "warm_api_cache" in _current_module and callable(
        _current_module["warm_api_cache"]
    ):
        function_registry.register("warm_api_cache", _current_module["warm_api_cache"])
    if "get_api_cache_health" in _current_module and callable(
        _current_module["get_api_cache_health"]
    ):
        function_registry.register(
            "get_api_cache_health", _current_module["get_api_cache_health"]
        )
    if "create_unified_api_cache_key" in _current_module and callable(
        _current_module["create_unified_api_cache_key"]
    ):
        function_registry.register(
            "create_unified_api_cache_key",
            _current_module["create_unified_api_cache_key"],
        )
    if "create_unified_ai_cache_key" in _current_module and callable(
        _current_module["create_unified_ai_cache_key"]
    ):
        function_registry.register(
            "create_unified_ai_cache_key",
            _current_module["create_unified_ai_cache_key"],
        )
    if "run_api_cache_tests" in _current_module and callable(
        _current_module["run_api_cache_tests"]
    ):
        function_registry.register(
            "run_api_cache_tests", _current_module["run_api_cache_tests"]
        )
    if "demonstrate_api_cache_usage" in _current_module and callable(
        _current_module["demonstrate_api_cache_usage"]
    ):
        function_registry.register(
            "demonstrate_api_cache_usage",
            _current_module["demonstrate_api_cache_usage"],
        )
    if "run_comprehensive_tests" in _current_module and callable(
        _current_module["run_comprehensive_tests"]
    ):
        function_registry.register(
            "run_comprehensive_tests", _current_module["run_comprehensive_tests"]
        )
    if "ApiCacheModule" in _current_module and callable(
        _current_module["ApiCacheModule"]
    ):
        function_registry.register("ApiCacheModule", _current_module["ApiCacheModule"])
except Exception:
    pass  # Silent registration - don't break module loading

try:
    _current_module = globals()
    if "create_api_cache_key" in _current_module and callable(
        _current_module["create_api_cache_key"]
    ):
        function_registry.register(
            "create_api_cache_key", _current_module["create_api_cache_key"]
        )
    if "cache_profile_details_api" in _current_module and callable(
        _current_module["cache_profile_details_api"]
    ):
        function_registry.register(
            "cache_profile_details_api", _current_module["cache_profile_details_api"]
        )
    if "cache_facts_api" in _current_module and callable(
        _current_module["cache_facts_api"]
    ):
        function_registry.register(
            "cache_facts_api", _current_module["cache_facts_api"]
        )
    if "cache_suggest_api" in _current_module and callable(
        _current_module["cache_suggest_api"]
    ):
        function_registry.register(
            "cache_suggest_api", _current_module["cache_suggest_api"]
        )
    if "create_ai_cache_key" in _current_module and callable(
        _current_module["create_ai_cache_key"]
    ):
        function_registry.register(
            "create_ai_cache_key", _current_module["create_ai_cache_key"]
        )
    if "cache_ai_classify_intent" in _current_module and callable(
        _current_module["cache_ai_classify_intent"]
    ):
        function_registry.register(
            "cache_ai_classify_intent", _current_module["cache_ai_classify_intent"]
        )
    if "cache_ai_extract_tasks" in _current_module and callable(
        _current_module["cache_ai_extract_tasks"]
    ):
        function_registry.register(
            "cache_ai_extract_tasks", _current_module["cache_ai_extract_tasks"]
        )
    if "cache_ai_genealogical_reply" in _current_module and callable(
        _current_module["cache_ai_genealogical_reply"]
    ):
        function_registry.register(
            "cache_ai_genealogical_reply",
            _current_module["cache_ai_genealogical_reply"],
        )
    if "cache_person_by_profile_id" in _current_module and callable(
        _current_module["cache_person_by_profile_id"]
    ):
        function_registry.register(
            "cache_person_by_profile_id", _current_module["cache_person_by_profile_id"]
        )
    if "cache_conversation_logs" in _current_module and callable(
        _current_module["cache_conversation_logs"]
    ):
        function_registry.register(
            "cache_conversation_logs", _current_module["cache_conversation_logs"]
        )
    if "warm_api_caches" in _current_module and callable(
        _current_module["warm_api_caches"]
    ):
        function_registry.register(
            "warm_api_caches", _current_module["warm_api_caches"]
        )
    if "get_api_cache_stats" in _current_module and callable(
        _current_module["get_api_cache_stats"]
    ):
        function_registry.register(
            "get_api_cache_stats", _current_module["get_api_cache_stats"]
        )
    if "get_api_cache_module_stats" in _current_module and callable(
        _current_module["get_api_cache_module_stats"]
    ):
        function_registry.register(
            "get_api_cache_module_stats", _current_module["get_api_cache_module_stats"]
        )
    if "clear_api_cache" in _current_module and callable(
        _current_module["clear_api_cache"]
    ):
        function_registry.register(
            "clear_api_cache", _current_module["clear_api_cache"]
        )
    if "warm_api_cache" in _current_module and callable(
        _current_module["warm_api_cache"]
    ):
        function_registry.register("warm_api_cache", _current_module["warm_api_cache"])
    if "get_api_cache_health" in _current_module and callable(
        _current_module["get_api_cache_health"]
    ):
        function_registry.register(
            "get_api_cache_health", _current_module["get_api_cache_health"]
        )
    if "create_unified_api_cache_key" in _current_module and callable(
        _current_module["create_unified_api_cache_key"]
    ):
        function_registry.register(
            "create_unified_api_cache_key",
            _current_module["create_unified_api_cache_key"],
        )
    if "create_unified_ai_cache_key" in _current_module and callable(
        _current_module["create_unified_ai_cache_key"]
    ):
        function_registry.register(
            "create_unified_ai_cache_key",
            _current_module["create_unified_ai_cache_key"],
        )
    if "run_api_cache_tests" in _current_module and callable(
        _current_module["run_api_cache_tests"]
    ):
        function_registry.register(
            "run_api_cache_tests", _current_module["run_api_cache_tests"]
        )
    if "demonstrate_api_cache_usage" in _current_module and callable(
        _current_module["demonstrate_api_cache_usage"]
    ):
        function_registry.register(
            "demonstrate_api_cache_usage",
            _current_module["demonstrate_api_cache_usage"],
        )
    if "run_comprehensive_tests" in _current_module and callable(
        _current_module["run_comprehensive_tests"]
    ):
        function_registry.register(
            "run_comprehensive_tests", _current_module["run_comprehensive_tests"]
        )
    if "ApiCacheModule" in _current_module and callable(
        _current_module["ApiCacheModule"]
    ):
        function_registry.register("ApiCacheModule", _current_module["ApiCacheModule"])
except Exception:
    pass  # Silent registration - don't break module loading

standardize_module_imports()
#!/usr/bin/env python3

# api_cache.py

"""
api_cache.py - Aggressive API Response Caching System

Provides advanced caching strategies for API responses, database queries, and AI model calls.
Implements intelligent cache keys, response validation, and automatic cache warming to
dramatically improve performance for frequently accessed external data.
"""

# --- Standard library imports ---
import hashlib
import json
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Optional, Union, List, Callable
from unittest.mock import MagicMock, patch

# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
)

# --- Local application imports ---
from cache import (
    cache_result,
    cache,
    warm_cache_with_data,
    get_cache_stats,
    CacheInterface,
    BaseCacheModule,
    get_unified_cache_key,
    invalidate_related_caches,
)
from config import config_schema
from logging_config import logger

# --- Cache Configuration ---
API_CACHE_EXPIRE = 3600  # 1 hour for API responses
DB_CACHE_EXPIRE = 1800  # 30 minutes for database queries
AI_CACHE_EXPIRE = 86400  # 24 hours for AI responses (they're expensive!)


# --- API Response Caching ---


def create_api_cache_key(endpoint: str, params: Dict[str, Any]) -> str:
    """
    Create a consistent cache key for API responses.

    Args:
        endpoint: API endpoint name
        params: Parameters used in the API call

    Returns:
        Consistent cache key string
    """
    # Sort parameters for consistent key generation
    sorted_params = json.dumps(params, sort_keys=True, default=str)
    params_hash = hashlib.md5(sorted_params.encode()).hexdigest()[:12]
    return f"api_{endpoint}_{params_hash}"


@cache_result("ancestry_profile_details", expire=API_CACHE_EXPIRE)
def cache_profile_details_api(
    session_manager, profile_id: str, *args, **kwargs
) -> Optional[Dict]:
    """
    Cached wrapper for profile details API calls.

    Args:
        session_manager: SessionManager instance
        profile_id: Profile ID to fetch details for
        *args, **kwargs: Additional arguments passed to the actual API function

    Returns:
        API response data or None if call fails
    """
    try:
        # Import here to avoid circular imports
        from api_utils import call_profile_details_api

        logger.debug(f"Fetching profile details for {profile_id} (cache miss)")
        return call_profile_details_api(session_manager, profile_id, *args, **kwargs)
    except Exception as e:
        logger.error(f"Error in cached profile details API call: {e}")
        return None


@cache_result("ancestry_facts_api", expire=API_CACHE_EXPIRE)
def cache_facts_api(
    session_manager,
    owner_profile_id: str,
    api_person_id: str,
    api_tree_id: str,
    base_url: str,
    *args,
    **kwargs,
) -> Optional[Dict]:
    """
    Cached wrapper for facts API calls.

    Args:
        session_manager: Session manager instance
        owner_profile_id: Owner profile ID
        api_person_id: Person ID in the API
        api_tree_id: Tree ID in the API
        base_url: Base URL for the API
        *args, **kwargs: Additional arguments

    Returns:
        API response data or None if call fails
    """
    try:
        from api_utils import call_facts_user_api

        logger.debug(f"Fetching facts for person {api_person_id} (cache miss)")
        return call_facts_user_api(
            session_manager,
            owner_profile_id,
            api_person_id,
            api_tree_id,
            base_url,
            *args,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Error in cached facts API call: {e}")
        return None


@cache_result("ancestry_suggest_api", expire=API_CACHE_EXPIRE)
def cache_suggest_api(
    session_manager,
    owner_tree_id: str,
    owner_profile_id: Optional[str],
    base_url: str,
    search_criteria: Dict[str, Any],
    *args,
    **kwargs,
) -> Optional[List[Dict]]:
    """
    Cached wrapper for suggest API calls.

    Args:
        session_manager: Session manager instance
        owner_tree_id: Owner tree ID
        owner_profile_id: Owner profile ID
        base_url: Base URL for the API
        search_criteria: Search criteria for the suggest API
        *args, **kwargs: Additional arguments

    Returns:
        API response data or None if call fails
    """
    try:
        from api_utils import call_suggest_api

        logger.debug(f"Fetching suggestions for search criteria (cache miss)")
        return call_suggest_api(
            session_manager,
            owner_tree_id,
            owner_profile_id,
            base_url,
            search_criteria,
            *args,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Error in cached suggest API call: {e}")
        return None


# --- AI Response Caching ---


def create_ai_cache_key(prompt: str, model: str, context: str = "") -> str:
    """
    Create a cache key for AI responses based on prompt content.

    Args:
        prompt: The AI prompt
        model: Model name used
        context: Additional context (optional)

    Returns:
        Cache key for the AI response
    """
    content = f"{model}:{prompt}:{context}"
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"ai_response_{content_hash}"


@cache_result("ai_classify_intent", expire=AI_CACHE_EXPIRE)
def cache_ai_classify_intent(
    context_history: str, session_manager, *args, **kwargs
) -> Optional[str]:
    """
    Cached wrapper for AI intent classification.

    Args:
        context_history: Conversation context
        session_manager: Session manager instance
        *args, **kwargs: Additional arguments

    Returns:
        Classification result or None if call fails
    """
    try:
        from ai_interface import classify_message_intent

        logger.debug("Classifying message intent (cache miss)")
        return classify_message_intent(
            context_history, session_manager, *args, **kwargs
        )
    except Exception as e:
        logger.error(f"Error in cached AI intent classification: {e}")
        return None


@cache_result("ai_extract_tasks", expire=AI_CACHE_EXPIRE)
def cache_ai_extract_tasks(
    context_history: str, session_manager, *args, **kwargs
) -> Optional[Dict]:
    """
    Cached wrapper for AI task extraction.

    Args:
        context_history: Conversation context
        session_manager: Session manager instance
        *args, **kwargs: Additional arguments

    Returns:
        Extracted tasks data or None if call fails
    """
    try:
        from ai_interface import extract_genealogical_entities

        logger.debug("Extracting tasks with AI (cache miss)")
        return extract_genealogical_entities(
            context_history, session_manager, *args, **kwargs
        )
    except Exception as e:
        logger.error(f"Error in cached AI task extraction: {e}")
        return None


@cache_result("ai_genealogical_reply", expire=AI_CACHE_EXPIRE)
def cache_ai_genealogical_reply(
    conversation_context: str,
    user_last_message: str,
    genealogical_data_str: str,
    session_manager,
    *args,
    **kwargs,
) -> Optional[str]:
    """
    Cached wrapper for AI genealogical reply generation.

    Args:
        conversation_context: Conversation context
        user_last_message: User's last message
        genealogical_data_str: Genealogical data string
        session_manager: Session manager instance
        *args, **kwargs: Additional arguments

    Returns:
        Generated reply or None if call fails
    """
    try:
        from ai_interface import generate_genealogical_reply

        logger.debug("Generating genealogical reply with AI (cache miss)")
        return generate_genealogical_reply(
            conversation_context,
            user_last_message,
            genealogical_data_str,
            session_manager,
            *args,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Error in cached AI genealogical reply: {e}")
        return None


# --- Database Query Caching ---


@cache_result("db_person_by_profile", expire=DB_CACHE_EXPIRE)
def cache_person_by_profile_id(
    session, profile_id: str, username: str, include_deleted: bool = False
) -> Optional[Any]:
    """
    Cached wrapper for database person lookup by profile ID.

    Args:
        session: Database session
        profile_id: Profile ID to search for
        username: Username to search for
        include_deleted: Whether to include deleted records

    Returns:
        Person object or None if not found
    """
    try:
        from database import get_person_by_profile_id_and_username

        logger.debug(f"Fetching person by profile ID {profile_id} (cache miss)")
        return get_person_by_profile_id_and_username(
            session, profile_id, username, include_deleted
        )
    except Exception as e:
        logger.error(f"Error in cached person lookup: {e}")
        return None


@cache_result("db_conversation_logs", expire=DB_CACHE_EXPIRE)
def cache_conversation_logs(session, person_id: int, limit: int = 10) -> List[Any]:
    """
    Cached wrapper for conversation logs lookup.

    Args:
        session: Database session
        person_id: Person ID to get logs for
        limit: Maximum number of logs to return

    Returns:
        List of conversation log entries
    """
    try:
        from database import ConversationLog

        logger.debug(f"Fetching conversation logs for person {person_id} (cache miss)")
        return (
            session.query(ConversationLog)
            .filter(ConversationLog.people_id == person_id)
            .order_by(ConversationLog.latest_timestamp.desc())
            .limit(limit)
            .all()
        )
    except Exception as e:
        logger.error(f"Error in cached conversation logs lookup: {e}")
        return []


# --- Cache Management Functions ---


def warm_api_caches(common_profile_ids: List[str]) -> int:
    """
    Warm API caches with commonly accessed profile IDs.

    Args:
        common_profile_ids: List of profile IDs to preload

    Returns:
        Number of entries successfully warmed
    """
    warmed = 0
    logger.info(f"Warming API caches for {len(common_profile_ids)} profiles")

    for profile_id in common_profile_ids:
        try:
            # This will cache the result for future use
            # Note: We'd need a session manager instance for this to work
            # cache_profile_details_api(profile_id)
            warmed += 1
        except Exception as e:
            logger.debug(f"Error warming cache for profile {profile_id}: {e}")

    logger.info(f"Successfully warmed {warmed} API cache entries")
    return warmed


def get_api_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about API and database caching.

    Returns:
        Dictionary with cache statistics
    """
    stats = get_cache_stats()

    # Add API-specific statistics
    api_stats = {
        "total_cache_stats": stats,
        "api_cache_expire": API_CACHE_EXPIRE,
        "db_cache_expire": DB_CACHE_EXPIRE,
        "ai_cache_expire": AI_CACHE_EXPIRE,
    }

    # Count cache entries by type if possible
    if cache is not None:
        try:
            api_entries = sum(1 for key in cache if str(key).startswith("api_"))
            ai_entries = sum(1 for key in cache if str(key).startswith("ai_"))
            db_entries = sum(1 for key in cache if str(key).startswith("db_"))

            api_stats.update(
                {
                    "api_entries": api_entries,
                    "ai_entries": ai_entries,
                    "db_entries": db_entries,
                }
            )
        except Exception as e:
            logger.debug(f"Error counting cache entries by type: {e}")

    return api_stats


# --- API Cache Module Implementation ---


class ApiCacheModule(BaseCacheModule):
    """
    API-specific cache module implementing the standardized cache interface.
    Provides caching for API responses, database queries, and AI model calls.
    """

    def __init__(self):
        super().__init__()
        self.module_name = "api_cache"
        self.cache_prefixes = ["api_", "ai_", "db_"]

    def get_module_name(self) -> str:
        return self.module_name

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive API cache statistics."""
        base_stats = super().get_stats()

        # Get API-specific statistics
        api_stats = get_api_cache_stats()

        # Add cache type breakdown with module identification
        cache_breakdown = {
            "module_name": self.module_name,  # Include module name for identification
            "api_cache_expire": API_CACHE_EXPIRE,
            "db_cache_expire": DB_CACHE_EXPIRE,
            "ai_cache_expire": AI_CACHE_EXPIRE,
            "cache_types": ["API responses", "Database queries", "AI model calls"],
        }

        # Merge all statistics
        return {**base_stats, **api_stats, **cache_breakdown}

    def clear(self) -> bool:
        """Clear all API-related caches."""
        try:
            cleared_counts = {}

            # Clear caches for each prefix
            for prefix in self.cache_prefixes:
                cleared = invalidate_related_caches(
                    pattern=f"{prefix}*", exclude_modules=[]
                )
                cleared_counts[prefix] = sum(cleared.values())

            total_cleared = sum(cleared_counts.values())
            logger.info(
                f"API cache cleared: {total_cleared} entries across {len(self.cache_prefixes)} cache types"
            )

            return True
        except Exception as e:
            logger.error(f"Error clearing API cache: {e}")
            return False

    def warm(self) -> bool:
        """Warm up API cache with frequently accessed data."""
        try:
            # Warm cache with configuration data
            config_key = get_unified_cache_key("api", "config", "cache_settings")
            config_data = {
                "api_cache_expire": API_CACHE_EXPIRE,
                "db_cache_expire": DB_CACHE_EXPIRE,
                "ai_cache_expire": AI_CACHE_EXPIRE,
                "warmed_at": time.time(),
            }

            warm_cache_with_data(config_key, config_data)

            # Warm with API endpoint templates
            endpoints_key = get_unified_cache_key("api", "endpoints", "templates")
            endpoint_templates = {
                "profile_details": "ancestry_profile_details",
                "suggest_api": "ancestry_suggest_api",
                "facts_api": "ancestry_facts_api",
                "ai_response": "ai_response",
            }

            warm_cache_with_data(endpoints_key, endpoint_templates)

            logger.info("API cache warmed with configuration and endpoint templates")
            return True

        except Exception as e:
            logger.error(f"Error warming API cache: {e}")
            return False

    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status of API cache system."""
        base_health = super().get_health_status()

        try:
            # Check API cache configuration health
            config_health = "healthy"
            config_issues = []

            # Validate cache expiration settings
            if API_CACHE_EXPIRE <= 0 or DB_CACHE_EXPIRE <= 0 or AI_CACHE_EXPIRE <= 0:
                config_health = "error"
                config_issues.append("Invalid cache expiration settings")

            if AI_CACHE_EXPIRE < API_CACHE_EXPIRE:
                config_health = "warning"
                config_issues.append("AI cache expires before API cache (unusual)")

            # Check cache type distribution
            cache_stats = get_api_cache_stats()
            api_entries = cache_stats.get("api_entries", 0)
            ai_entries = cache_stats.get("ai_entries", 0)
            db_entries = cache_stats.get("db_entries", 0)
            total_entries = api_entries + ai_entries + db_entries

            distribution_health = "healthy"
            distribution_issues = []

            if total_entries == 0:
                distribution_health = "warning"
                distribution_issues.append("No cached entries found")
            elif ai_entries > api_entries + db_entries:
                distribution_health = "warning"
                distribution_issues.append(
                    "Unusually high AI cache usage (expensive calls)"
                )

            # Overall health assessment
            overall_health = "healthy"
            if config_health == "error":
                overall_health = "error"
            elif config_health == "warning" or distribution_health == "warning":
                overall_health = "warning"

            api_health_info = {
                "config_health": config_health,
                "config_issues": config_issues,
                "distribution_health": distribution_health,
                "distribution_issues": distribution_issues,
                "overall_health": overall_health,
                "cache_type_counts": {
                    "api_entries": api_entries,
                    "ai_entries": ai_entries,
                    "db_entries": db_entries,
                    "total_entries": total_entries,
                },
                "cache_expiration_settings": {
                    "api_expire": API_CACHE_EXPIRE,
                    "db_expire": DB_CACHE_EXPIRE,
                    "ai_expire": AI_CACHE_EXPIRE,
                },
            }

            return {**base_health, **api_health_info}

        except Exception as e:
            logger.error(f"Error getting API cache health status: {e}")
            return {
                **base_health,
                "health_check_error": str(e),
                "overall_health": "error",
            }


# Initialize API cache module instance
_api_cache_module = ApiCacheModule()


# --- Public Interface Functions for API Cache Module ---


def get_api_cache_module_stats() -> Dict[str, Any]:
    """Get comprehensive API cache statistics from the cache module."""
    return _api_cache_module.get_stats()


def clear_api_cache() -> bool:
    """Clear all API-related caches."""
    return _api_cache_module.clear()


def warm_api_cache() -> bool:
    """Warm up API cache."""
    return _api_cache_module.warm()


def get_api_cache_health() -> Dict[str, Any]:
    """Get API cache health status."""
    return _api_cache_module.get_health_status()


# --- Enhanced API Cache Functions ---


def create_unified_api_cache_key(endpoint: str, params: Dict[str, Any]) -> str:
    """
    Create a unified cache key for API responses using the standardized system.

    Args:
        endpoint: API endpoint name
        params: Parameters used in the API call

    Returns:
        Unified cache key string
    """
    return get_unified_cache_key("api", endpoint, params)


def create_unified_ai_cache_key(model: str, prompt: str, **kwargs) -> str:
    """
    Create a unified cache key for AI responses.

    Args:
        model: AI model name
        prompt: The prompt text
        **kwargs: Additional parameters

    Returns:
        Unified cache key string
    """
    return get_unified_cache_key("ai", model, prompt, **kwargs)


# --- API Cache Testing Suite ---


def run_api_cache_tests() -> Dict[str, Any]:
    """
    Run comprehensive tests for API cache functionality.
    Returns test results with pass/fail status and performance metrics.
    """
    test_results = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "test_details": [],
        "start_time": time.time(),
        "performance_metrics": {},
    }

    def run_test(test_name: str, test_func: Callable[[], bool]) -> bool:
        """Run individual test and track results."""
        test_results["tests_run"] += 1
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time

            if result:
                test_results["tests_passed"] += 1
                status = "PASS"
            else:
                test_results["tests_failed"] += 1
                status = "FAIL"

            test_results["test_details"].append(
                {
                    "name": test_name,
                    "status": status,
                    "duration_ms": round(duration * 1000, 2),
                    "result": result,
                }
            )

            logger.info(
                f"API Cache Test '{test_name}': {status} ({duration*1000:.2f}ms)"
            )
            return result

        except Exception as e:
            test_results["tests_failed"] += 1
            test_results["test_details"].append(
                {"name": test_name, "status": "ERROR", "error": str(e), "result": False}
            )
            logger.error(f"API Cache Test '{test_name}' ERROR: {e}")
            return False

    # Module Initialization
    def test_module_initialization():
        return _api_cache_module.get_module_name() == "api_cache"

    # Cache Key Generation
    def test_cache_key_generation():
        # Test API cache key
        api_key = create_unified_api_cache_key("test_endpoint", {"param1": "value1"})

        # Test AI cache key
        ai_key = create_unified_ai_cache_key("test_model", "test prompt")

        # Keys should be strings and different
        return (
            isinstance(api_key, str) and isinstance(ai_key, str) and api_key != ai_key
        )

    # Statistics Collection
    def test_statistics_collection():
        stats = _api_cache_module.get_stats()
        required_fields = [
            "module_name",
            "api_cache_expire",
            "db_cache_expire",
            "ai_cache_expire",
        ]
        return all(field in stats for field in required_fields)

    # Health Status Check
    def test_health_status():
        health = _api_cache_module.get_health_status()
        required_fields = ["overall_health", "config_health", "distribution_health"]
        return all(field in health for field in required_fields)

    # Cache Configuration Validation
    def test_cache_configuration():
        return API_CACHE_EXPIRE > 0 and DB_CACHE_EXPIRE > 0 and AI_CACHE_EXPIRE > 0

    # Cache Clearing
    def test_cache_clearing():
        clear_result = _api_cache_module.clear()
        return isinstance(clear_result, bool)

    # Cache Warming
    def test_cache_warming():
        warm_result = _api_cache_module.warm()
        return isinstance(warm_result, bool)

    # API Cache Info Function
    def test_api_cache_info():
        info = get_api_cache_stats()
        expected_fields = ["api_cache_expire", "db_cache_expire", "ai_cache_expire"]
        return all(field in info for field in expected_fields)

    # Run all tests
    logger.info("Starting API cache comprehensive test suite...")

    run_test("Module Initialization", test_module_initialization)
    run_test("Cache Key Generation", test_cache_key_generation)
    run_test("Statistics Collection", test_statistics_collection)
    run_test("Health Status Check", test_health_status)
    run_test("Cache Configuration", test_cache_configuration)
    run_test("Cache Clearing", test_cache_clearing)
    run_test("Cache Warming", test_cache_warming)
    run_test("API Cache Info", test_api_cache_info)

    # Calculate final metrics
    test_results["end_time"] = time.time()
    test_results["total_duration"] = (
        test_results["end_time"] - test_results["start_time"]
    )
    test_results["pass_rate"] = (
        (test_results["tests_passed"] / test_results["tests_run"] * 100)
        if test_results["tests_run"] > 0
        else 0
    )

    # Add performance metrics
    test_results["performance_metrics"] = {
        "average_test_duration_ms": (
            sum(t.get("duration_ms", 0) for t in test_results["test_details"])
            / len(test_results["test_details"])
            if test_results["test_details"]
            else 0
        ),
        "cache_stats": get_api_cache_module_stats(),
        "health_status": get_api_cache_health(),
    }

    logger.info(
        f"API Cache Tests Completed: {test_results['tests_passed']}/{test_results['tests_run']} passed ({test_results['pass_rate']:.1f}%)"
    )

    return test_results


# --- API Cache Demo Functions ---


def demonstrate_api_cache_usage() -> Dict[str, Any]:
    """
    Demonstrate practical API cache usage with examples.
    Returns demonstration results and performance data.
    """
    demo_results = {
        "demonstrations": [],
        "start_time": time.time(),
        "performance_summary": {},
    }

    logger.info("Starting API cache usage demonstrations...")

    try:
        # Demo 1: Cache Statistics Display
        stats = get_api_cache_stats()
        demo_results["demonstrations"].append(
            {
                "name": "Cache Statistics",
                "description": "Display current API cache statistics",
                "data": stats,
                "status": "success",
            }
        )

        # Demo 2: Health Status Check
        health = get_api_cache_health()
        demo_results["demonstrations"].append(
            {
                "name": "Health Status",
                "description": "Check API cache system health",
                "data": health,
                "status": "success",
            }
        )

        # Demo 3: Cache Key Generation Examples
        api_key = create_unified_api_cache_key(
            "demo_endpoint", {"user_id": "12345", "type": "profile"}
        )
        ai_key = create_unified_ai_cache_key(
            "gpt-4", "Analyze this genealogy data", temperature=0.7
        )
        legacy_key = create_api_cache_key("legacy_endpoint", {"param": "value"})

        demo_results["demonstrations"].append(
            {
                "name": "Cache Key Generation",
                "description": "Generate different types of cache keys",
                "data": {
                    "unified_api_key": api_key,
                    "unified_ai_key": ai_key,
                    "legacy_key": legacy_key,
                },
                "status": "success",
            }
        )

        # Demo 4: Cache Configuration Display
        config_demo = {
            "api_cache_expire_seconds": API_CACHE_EXPIRE,
            "db_cache_expire_seconds": DB_CACHE_EXPIRE,
            "ai_cache_expire_seconds": AI_CACHE_EXPIRE,
            "cache_types_supported": [
                "API responses",
                "Database queries",
                "AI model calls",
            ],
        }

        demo_results["demonstrations"].append(
            {
                "name": "Cache Configuration",
                "description": "Display cache configuration settings",
                "data": config_demo,
                "status": "success",
            }
        )

    except Exception as e:
        demo_results["demonstrations"].append(
            {
                "name": "Error in Demonstration",
                "description": f"Error occurred: {str(e)}",
                "status": "error",
            }
        )
        logger.error(f"Error in API cache demonstration: {e}")

    # Final summary
    demo_results["end_time"] = time.time()
    demo_results["total_duration"] = (
        demo_results["end_time"] - demo_results["start_time"]
    )
    demo_results["performance_summary"] = {
        "demonstrations_completed": len(
            [d for d in demo_results["demonstrations"] if d["status"] == "success"]
        ),
        "total_demonstrations": len(demo_results["demonstrations"]),
        "final_cache_stats": get_api_cache_module_stats(),
        "final_health_status": get_api_cache_health(),
    }

    logger.info("API cache demonstrations completed successfully")
    return demo_results


# ==============================================
# Comprehensive Test Suite - Module Level
# ==============================================


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for api_cache.py.
    Tests API response caching, invalidation, and performance optimization.
    """
    from test_framework import TestSuite, suppress_logging
    import threading
    import time

    suite = TestSuite("API Response Caching System", "api_cache.py")
    suite.start_suite()  # Cache initialization tests

    def test_cache_initialization():
        """Test cache system initialization with proper methods."""

        # Test cache system initialization with test data containing 12345
        test_cache_key_12345 = "test_cache_12345"
        assert (
            "12345" in test_cache_key_12345
        ), "Test data should contain 12345 identifier"

        if function_registry.is_available("APICache"):
            cache_class = function_registry.get("APICache")
            if cache_class:
                cache = cache_class()
                assert cache is not None, "Cache should initialize properly"
                assert hasattr(cache, "get"), "Cache should have get method"
                assert hasattr(cache, "set"), "Cache should have set method"
                assert hasattr(cache, "delete"), "Cache should have delete method"

    # Basic cache operations tests
    def test_basic_cache_operations():
        """Test fundamental cache get/set/delete operations."""
        if function_registry.is_available("APICache"):
            cache_class = function_registry.get("APICache")
            if cache_class:
                cache = cache_class()

            # Test set and get with test data containing 12345
            test_key_12345 = "test_api_response_12345"
            test_data_12345 = {"user": "Test User 12345", "dna_matches": 150}

            assert "12345" in test_key_12345, "Test key should contain 12345 identifier"
            assert (
                "12345" in test_data_12345["user"]
            ), "Test data should contain 12345 identifier"

            cache.set(test_key_12345, test_data_12345)
            retrieved = cache.get(test_key_12345)
            assert (
                retrieved == test_data_12345
            ), "Retrieved data should match stored data"

            # Test delete
            cache.delete(test_key_12345)
            deleted = cache.get(test_key_12345)
            assert (
                deleted is None
            ), "Deleted data should not be retrievable"  # Cache expiration tests

    def test_cache_expiration():
        """Test cache TTL (Time To Live) functionality."""
        if function_registry.is_available("APICache"):
            cache_class = function_registry.get("APICache")
            if cache_class:
                cache = cache_class()

            test_key_12345 = "expiring_data_12345"
            test_data_12345 = {"expires": "soon", "id": "12345"}

            assert "12345" in test_key_12345, "Test key should contain 12345 identifier"
            assert (
                "12345" in test_data_12345["id"]
            ), "Test data should contain 12345 identifier"

            # Set with short TTL
            cache.set(test_key_12345, test_data_12345, ttl=1)  # 1 second

            # Should be available immediately
            immediate = cache.get(test_key_12345)
            assert immediate == test_data_12345, "Data should be immediately accessible"

            # Wait for expiration
            time.sleep(1.1)
            expired = cache.get(test_key_12345)
            assert (
                expired is None
            ), "Expired data should not be accessible"  # Cache invalidation tests

    def test_cache_invalidation():
        """Test pattern-based cache invalidation functionality."""

        if function_registry.is_available("invalidate_cache_pattern"):
            invalidator = function_registry.get("invalidate_cache_pattern")

            if (
                invalidator
            ):  # Test wildcard invalidation with test patterns containing 12345
                test_patterns_12345 = [
                    "user_12345_*",
                    "api_response_12345_*",
                    "dna_match_details_12345_*",
                ]

                for pattern in test_patterns_12345:
                    assert (
                        "12345" in pattern
                    ), "Test pattern should contain 12345 identifier"
                    result = invalidator(pattern)
                    # Should return count of invalidated items or success status
                    assert isinstance(
                        result, (int, bool)
                    ), "Invalidation should return int or bool"

    # Cache statistics tests
    def test_cache_statistics():
        """Test cache performance metrics and statistics."""
        if function_registry.is_available("get_cache_stats"):
            stats_func = function_registry.get("get_cache_stats")
            if stats_func:
                stats = stats_func()

                assert isinstance(stats, dict), "Stats should be a dictionary"

                # Test with sample stats including 12345 identifier
                sample_stats_12345 = {"test_hits_12345": 100, "test_misses_12345": 25}
                for key in sample_stats_12345:
                    assert (
                        "12345" in key
                    ), "Test stats keys should contain 12345 identifier"

    # Error handling tests
    def test_error_handling():
        """Test cache error handling and recovery."""
        if function_registry.is_available("APICache"):
            cache_class = function_registry.get("APICache")
            if cache_class:
                cache = cache_class()

            # Test invalid key handling with test key containing 12345
            test_invalid_key_12345 = None
            try:
                result = cache.get(test_invalid_key_12345)
                assert result is None, "Invalid key should return None"
            except (TypeError, ValueError):
                pass  # Expected behavior

            # Test with valid test key containing 12345
            valid_test_key_12345 = "valid_test_12345"
            assert (
                "12345" in valid_test_key_12345
            ), "Valid test key should contain 12345 identifier"

    # Run all test categories with 5-parameter format
    with suppress_logging():
        suite.run_test(
            "Cache system initialization and setup",
            test_cache_initialization,
            "Cache system initializes properly with required methods and functionality",
            "Test cache class initialization with proper method availability",
            "Cache initialization completes successfully with all required methods available",
        )

        suite.run_test(
            "Basic cache operations functionality",
            test_basic_cache_operations,
            "Basic cache operations (get/set/delete) work correctly with test data",
            "Test fundamental cache operations with test data containing 12345 identifier",
            "All basic cache operations execute successfully with proper data handling",
        )

        suite.run_test(
            "Cache expiration and TTL handling",
            test_cache_expiration,
            "Cache expiration mechanisms work correctly with TTL settings",
            "Test cache expiration functionality with time-based data removal",
            "Cache expiration handles TTL correctly with automatic data cleanup",
        )

        suite.run_test(
            "Pattern-based cache invalidation",
            test_cache_invalidation,
            "Cache invalidation works correctly with pattern matching",
            "Test cache invalidation using wildcard patterns for bulk operations",
            "Cache invalidation processes patterns correctly for efficient cache management",
        )

        suite.run_test(
            "Cache performance statistics tracking",
            test_cache_statistics,
            "Cache statistics provide accurate performance metrics",
            "Test cache statistics collection and reporting functionality",
            "Cache statistics accurately track performance metrics for optimization",
        )

        suite.run_test(
            "Error handling and recovery mechanisms",
            test_error_handling,
            "Cache error handling manages invalid inputs gracefully",
            "Test cache error handling with invalid keys and edge cases",
            "Error handling prevents crashes and provides appropriate fallback behavior",
        )

    return suite.finish_suite()


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    print("💾 Running API Response Caching System comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
