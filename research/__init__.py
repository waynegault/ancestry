"""Research Utilities Package.

Provides research-related utilities including:
- relationship_utils: Relationship calculations and pathfinding
- relationship_diagram: Relationship diagram generation
- research_suggestions: Research suggestion generation
- research_guidance_prompts: AI prompts for research guidance
- research_prioritization: Research task prioritization
"""

_SUBMODULES = frozenset(
    [
        "relationship_diagram",
        "relationship_utils",
        "research_guidance_prompts",
        "research_prioritization",
        "research_suggestions",
    ]
)


def __getattr__(name: str):
    """Lazy import submodules on attribute access."""
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available submodules."""
    return list(_SUBMODULES)
