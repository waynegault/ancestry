"""Backward-compatibility shim -- all logic lives in :mod:`ai.ai_prompt_utils`.

This module re-exports every public name so that any stale ``from ai.prompts
import ...`` statements continue to work.  New code should import directly from
:mod:`ai.ai_prompt_utils`.
"""

from ai.ai_prompt_utils import (
    get_prompt,
    get_prompt_version,
    get_prompt_with_experiment,
    get_prompts_summary,
    load_prompts,
    supports_json_prompts,
)

__all__ = [
    "get_prompt",
    "get_prompt_version",
    "get_prompt_with_experiment",
    "get_prompts_summary",
    "load_prompts",
    "supports_json_prompts",
]
