"""
mcp_server - Ancestry MCP Server Package

Exposes the Ancestry research toolkit as MCP tools for AI assistants.
"""

from mcp_server.server import (
    AncestryMCPServer,
    run_mcp_server_action,
    start_mcp_server,
)

__all__ = [
    "AncestryMCPServer",
    "run_mcp_server_action",
    "start_mcp_server",
]
