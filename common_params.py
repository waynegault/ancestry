#!/usr/bin/env python3
"""
common_params.py - Common Parameter Grouping Dataclasses

This module provides dataclasses for grouping commonly-used function parameters
to reduce parameter counts and improve code maintainability.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GraphContext:
    """
    Graph traversal context for genealogical relationship calculations.
    
    Used in gedcom_utils.py and relationship_utils.py for BFS/DFS operations.
    """
    id_to_parents: Dict[str, List[str]]
    id_to_children: Dict[str, List[str]]
    current_id: Optional[str] = None
    start_id: Optional[str] = None
    end_id: Optional[str] = None


@dataclass
class RetryContext:
    """
    Retry logic parameters for API calls and error handling.
    
    Used across multiple modules for exponential backoff retry logic.
    """
    attempt: int
    max_attempts: int
    max_delay: float
    backoff_factor: float = 2.0
    current_delay: float = 1.0


@dataclass
class MatchIdentifiers:
    """
    DNA match identification parameters.
    
    Used in action6_gather.py for processing DNA matches.
    """
    uuid: str
    username: str
    in_my_tree: bool
    log_ref_short: str
    profile_id: Optional[str] = None


@dataclass
class ConversationIdentifiers:
    """
    Conversation/messaging identification parameters.
    
    Used in action7_inbox.py and action8_messaging.py.
    """
    api_conv_id: str
    people_id: Optional[str] = None
    my_pid_lower: Optional[str] = None
    effective_conv_id: Optional[str] = None
    log_prefix: Optional[str] = None


@dataclass
class ApiIdentifiers:
    """
    API-related identification parameters.
    
    Used in api_utils.py and related modules.
    """
    owner_profile_id: str
    api_person_id: Optional[str] = None
    api_tree_id: Optional[str] = None
    owner_tree_id: Optional[str] = None


@dataclass
class BatchCounters:
    """
    Batch processing counters.
    
    Used in action6_gather.py and other batch processing modules.
    """
    new: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    total: int = 0


@dataclass
class SearchCriteria:
    """
    Search criteria for person/match searches.
    
    Used in api_search_utils.py and action10.py.
    """
    search_name: str
    field_name: Optional[str] = None
    test_name: Optional[str] = None
    candidate_data: Optional[Dict[str, Any]] = None
    name_flex: Optional[str] = None


# End of common_params.py

