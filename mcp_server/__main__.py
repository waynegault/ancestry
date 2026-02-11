#!/usr/bin/env python3
"""
Allow running the MCP server with: python -m mcp_server

This starts the Ancestry MCP Server over stdio transport.
"""

from mcp_server.server import start_mcp_server
import sys

if __name__ == "__main__":
    success = start_mcp_server()
    sys.exit(0 if success else 1)
