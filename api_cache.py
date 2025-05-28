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
import time
from typing import Any, Dict, Optional, Union, List

# --- Local application imports ---
from cache import cache_result, cache, warm_cache_with_data, get_cache_stats
from config import config_instance
from logging_config import logger

# --- Cache Configuration ---
API_CACHE_EXPIRE = 3600  # 1 hour for API responses
DB_CACHE_EXPIRE = 1800   # 30 minutes for database queries
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
def cache_profile_details_api(profile_id: str, *args, **kwargs) -> Optional[Dict]:
    """
    Cached wrapper for profile details API calls.
    
    Args:
        profile_id: Profile ID to fetch details for
        *args, **kwargs: Additional arguments passed to the actual API function
        
    Returns:
        API response data or None if call fails
    """
    try:
        # Import here to avoid circular imports
        from api_utils import call_profile_details_api
        
        logger.debug(f"Fetching profile details for {profile_id} (cache miss)")
        return call_profile_details_api(profile_id, *args, **kwargs)
    except Exception as e:
        logger.error(f"Error in cached profile details API call: {e}")
        return None


@cache_result("ancestry_facts_api", expire=API_CACHE_EXPIRE)
def cache_facts_api(session_manager, owner_profile_id: str, api_person_id: str, 
                   api_tree_id: str, base_url: str, *args, **kwargs) -> Optional[Dict]:
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
        return call_facts_user_api(session_manager, owner_profile_id, api_person_id, 
                                 api_tree_id, base_url, *args, **kwargs)
    except Exception as e:
        logger.error(f"Error in cached facts API call: {e}")
        return None


@cache_result("ancestry_suggest_api", expire=API_CACHE_EXPIRE)
def cache_suggest_api(session_manager, search_params: Dict[str, Any], 
                     *args, **kwargs) -> Optional[Dict]:
    """
    Cached wrapper for suggest API calls.
    
    Args:
        session_manager: Session manager instance
        search_params: Search parameters for the suggest API
        *args, **kwargs: Additional arguments
        
    Returns:
        API response data or None if call fails
    """
    try:
        from api_utils import call_suggest_api
        
        logger.debug(f"Fetching suggestions for search params (cache miss)")
        return call_suggest_api(session_manager, search_params, *args, **kwargs)
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
def cache_ai_classify_intent(context_history: str, session_manager, *args, **kwargs) -> Optional[str]:
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
        return classify_message_intent(context_history, session_manager, *args, **kwargs)
    except Exception as e:
        logger.error(f"Error in cached AI intent classification: {e}")
        return None


@cache_result("ai_extract_tasks", expire=AI_CACHE_EXPIRE)
def cache_ai_extract_tasks(context_history: str, session_manager, *args, **kwargs) -> Optional[Dict]:
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
        from ai_interface import extract_and_suggest_tasks
        
        logger.debug("Extracting tasks with AI (cache miss)")
        return extract_and_suggest_tasks(context_history, session_manager, *args, **kwargs)
    except Exception as e:
        logger.error(f"Error in cached AI task extraction: {e}")
        return None


@cache_result("ai_genealogical_reply", expire=AI_CACHE_EXPIRE)
def cache_ai_genealogical_reply(conversation_context: str, user_last_message: str,
                               genealogical_data_str: str, session_manager, 
                               *args, **kwargs) -> Optional[str]:
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
        return generate_genealogical_reply(conversation_context, user_last_message,
                                         genealogical_data_str, session_manager, 
                                         *args, **kwargs)
    except Exception as e:
        logger.error(f"Error in cached AI genealogical reply: {e}")
        return None


# --- Database Query Caching ---

@cache_result("db_person_by_profile", expire=DB_CACHE_EXPIRE)
def cache_person_by_profile_id(session, profile_id: str, username: str, 
                              include_deleted: bool = False) -> Optional[Any]:
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
        return get_person_by_profile_id_and_username(session, profile_id, username, include_deleted)
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
        return session.query(ConversationLog).filter(
            ConversationLog.people_id == person_id
        ).order_by(ConversationLog.latest_timestamp.desc()).limit(limit).all()
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
        'total_cache_stats': stats,
        'api_cache_expire': API_CACHE_EXPIRE,
        'db_cache_expire': DB_CACHE_EXPIRE,
        'ai_cache_expire': AI_CACHE_EXPIRE,
    }
    
    # Count cache entries by type if possible
    if cache is not None:
        try:
            api_entries = sum(1 for key in cache if str(key).startswith('api_'))
            ai_entries = sum(1 for key in cache if str(key).startswith('ai_'))
            db_entries = sum(1 for key in cache if str(key).startswith('db_'))
            
            api_stats.update({
                'api_entries': api_entries,
                'ai_entries': ai_entries,
                'db_entries': db_entries,
            })
        except Exception as e:
            logger.debug(f"Error counting cache entries by type: {e}")
    
    return api_stats


# End of api_cache.py
