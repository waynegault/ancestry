#!/usr/bin/env python3
"""
Ancestry MCP Server - Exposes waynegault/ancestry toolkit as MCP tools.

This server wraps the existing ancestry automation system to allow AI assistants
to interact with DNA matching, GEDCOM parsing, inbox processing, and more.

Usage:
    # Set the path to your ancestry repo
    export ANCESTRY_ROOT=/path/to/waynegault/ancestry
    
    # Run the server
    python server.py
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Any, Optional
import logging

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ancestry-mcp")

# Get the ancestry repo root from environment
ANCESTRY_ROOT = os.environ.get("ANCESTRY_ROOT", "/path/to/waynegault/ancestry")

# Add ancestry repo to path so we can import its modules
if ANCESTRY_ROOT not in sys.path:
    sys.path.insert(0, ANCESTRY_ROOT)


class AncestryMCPServer:
    """MCP Server that wraps the waynegault/ancestry toolkit."""
    
    def __init__(self):
        self.server = Server("ancestry-mcp")
        self._setup_handlers()
        self._ancestry_modules = {}
        
    def _load_ancestry_module(self, module_name: str):
        """Lazy-load ancestry modules to avoid import errors at startup."""
        if module_name not in self._ancestry_modules:
            try:
                self._ancestry_modules[module_name] = __import__(module_name)
            except ImportError as e:
                logger.error(f"Failed to import {module_name}: {e}")
                return None
        return self._ancestry_modules[module_name]
    
    def _setup_handlers(self):
        """Set up MCP tool handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="gather_dna_matches",
                    description="Gather DNA matches from Ancestry.com. Supports checkpointing for large match lists. Returns match data including names, relationships, shared DNA, and ethnicity.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "max_matches": {
                                "type": "integer",
                                "description": "Maximum number of matches to gather (default: 100)",
                                "default": 100
                            },
                            "min_shared_cm": {
                                "type": "number",
                                "description": "Minimum shared cM threshold (default: 20)",
                                "default": 20
                            },
                            "resume_from_checkpoint": {
                                "type": "boolean",
                                "description": "Resume from last checkpoint if available",
                                "default": True
                            }
                        }
                    }
                ),
                Tool(
                    name="search_inbox",
                    description="Search and process Ancestry inbox messages. Uses AI to classify message intent, extract entities, and identify actionable items.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Optional search query to filter messages"
                            },
                            "unread_only": {
                                "type": "boolean",
                                "description": "Only return unread messages",
                                "default": False
                            },
                            "max_messages": {
                                "type": "integer",
                                "description": "Maximum messages to process",
                                "default": 50
                            }
                        }
                    }
                ),
                Tool(
                    name="search_gedcom",
                    description="Search GEDCOM files for individuals. Supports local file parsing and Ancestry API search with unified scoring.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name to search for (required)"
                            },
                            "birth_year": {
                                "type": "integer",
                                "description": "Approximate birth year"
                            },
                            "birth_place": {
                                "type": "string",
                                "description": "Birth location"
                            },
                            "gedcom_file": {
                                "type": "string",
                                "description": "Path to specific GEDCOM file (optional, searches all if not provided)"
                            },
                            "include_api_search": {
                                "type": "boolean",
                                "description": "Also search Ancestry API",
                                "default": True
                            }
                        },
                        "required": ["name"]
                    }
                ),
                Tool(
                    name="get_shared_matches",
                    description="Get shared DNA matches between you and a specific match. Useful for triangulation and identifying common ancestors.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "match_id": {
                                "type": "string",
                                "description": "The Ancestry match ID (GUID)"
                            },
                            "min_shared_cm": {
                                "type": "number",
                                "description": "Minimum shared cM for results",
                                "default": 20
                            }
                        },
                        "required": ["match_id"]
                    }
                ),
                Tool(
                    name="run_triangulation",
                    description="Run triangulation analysis on DNA matches to identify clusters and potential common ancestors.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "match_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of match IDs to analyze (optional, uses all matches if not provided)"
                            },
                            "min_cluster_size": {
                                "type": "integer",
                                "description": "Minimum matches per cluster",
                                "default": 3
                            }
                        }
                    }
                ),
                Tool(
                    name="get_match_details",
                    description="Get detailed information about a specific DNA match including shared DNA segments, trees, and relationship predictions.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "match_id": {
                                "type": "string",
                                "description": "The Ancestry match ID (GUID)"
                            }
                        },
                        "required": ["match_id"]
                    }
                ),
                Tool(
                    name="draft_message",
                    description="Draft a message to a DNA match using AI. Messages are saved to a review queue and NEVER sent automatically. You must approve before sending.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "match_id": {
                                "type": "string",
                                "description": "The match ID to message"
                            },
                            "context": {
                                "type": "string",
                                "description": "Context for the message (e.g., 'introduce myself', 'ask about common ancestor John Smith')"
                            },
                            "tone": {
                                "type": "string",
                                "enum": ["friendly", "formal", "brief"],
                                "description": "Message tone",
                                "default": "friendly"
                            }
                        },
                        "required": ["match_id", "context"]
                    }
                ),
                Tool(
                    name="list_pending_drafts",
                    description="List all pending message drafts awaiting review.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="query_local_database",
                    description="Run a read-only SQL query against the local ancestry database. Useful for analyzing collected match data.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL SELECT query to run"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_database_schema",
                    description="Get the schema of the local ancestry database to understand available tables and columns.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls by delegating to ancestry modules."""
            try:
                result = await self._execute_tool(name, arguments)
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
            except Exception as e:
                logger.exception(f"Error executing tool {name}")
                return [TextContent(type="text", text=json.dumps({
                    "error": str(e),
                    "tool": name,
                    "arguments": arguments
                }, indent=2))]
    
    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        """Execute a tool by calling the appropriate ancestry module."""
        
        if name == "gather_dna_matches":
            return await self._gather_dna_matches(**arguments)
        elif name == "search_inbox":
            return await self._search_inbox(**arguments)
        elif name == "search_gedcom":
            return await self._search_gedcom(**arguments)
        elif name == "get_shared_matches":
            return await self._get_shared_matches(**arguments)
        elif name == "run_triangulation":
            return await self._run_triangulation(**arguments)
        elif name == "get_match_details":
            return await self._get_match_details(**arguments)
        elif name == "draft_message":
            return await self._draft_message(**arguments)
        elif name == "list_pending_drafts":
            return await self._list_pending_drafts()
        elif name == "query_local_database":
            return await self._query_local_database(**arguments)
        elif name == "get_database_schema":
            return await self._get_database_schema()
        else:
            return {"error": f"Unknown tool: {name}"}
    
    async def _gather_dna_matches(
        self,
        max_matches: int = 100,
        min_shared_cm: float = 20,
        resume_from_checkpoint: bool = True
    ) -> dict:
        """Gather DNA matches from Ancestry."""
        # Import the action module
        try:
            from actions import action_06_gather_matches
            
            # Run in executor since the ancestry code is synchronous
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: action_06_gather_matches.gather_matches(
                    max_matches=max_matches,
                    min_shared_cm=min_shared_cm,
                    resume=resume_from_checkpoint
                )
            )
            return {"success": True, "matches": result}
        except ImportError:
            return self._not_configured_response("gather_matches")
        except Exception as e:
            return {"error": str(e)}
    
    async def _search_inbox(
        self,
        query: Optional[str] = None,
        unread_only: bool = False,
        max_messages: int = 50
    ) -> dict:
        """Search and process inbox messages."""
        try:
            from actions import action_07_process_inbox
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: action_07_process_inbox.search_messages(
                    query=query,
                    unread_only=unread_only,
                    max_messages=max_messages
                )
            )
            return {"success": True, "messages": result}
        except ImportError:
            return self._not_configured_response("inbox_search")
        except Exception as e:
            return {"error": str(e)}
    
    async def _search_gedcom(
        self,
        name: str,
        birth_year: Optional[int] = None,
        birth_place: Optional[str] = None,
        gedcom_file: Optional[str] = None,
        include_api_search: bool = True
    ) -> dict:
        """Search GEDCOM files for individuals."""
        try:
            from actions import action_10_gedcom_search
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: action_10_gedcom_search.search(
                    name=name,
                    birth_year=birth_year,
                    birth_place=birth_place,
                    gedcom_file=gedcom_file,
                    include_api=include_api_search
                )
            )
            return {"success": True, "results": result}
        except ImportError:
            return self._not_configured_response("gedcom_search")
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_shared_matches(
        self,
        match_id: str,
        min_shared_cm: float = 20
    ) -> dict:
        """Get shared matches with a specific match."""
        try:
            from actions import action_13_triangulation
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: action_13_triangulation.get_shared_matches(
                    match_id=match_id,
                    min_cm=min_shared_cm
                )
            )
            return {"success": True, "shared_matches": result}
        except ImportError:
            return self._not_configured_response("shared_matches")
        except Exception as e:
            return {"error": str(e)}
    
    async def _run_triangulation(
        self,
        match_ids: Optional[list[str]] = None,
        min_cluster_size: int = 3
    ) -> dict:
        """Run triangulation analysis."""
        try:
            from actions import action_13_triangulation
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: action_13_triangulation.run_triangulation(
                    match_ids=match_ids,
                    min_cluster_size=min_cluster_size
                )
            )
            return {"success": True, "clusters": result}
        except ImportError:
            return self._not_configured_response("triangulation")
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_match_details(self, match_id: str) -> dict:
        """Get detailed information about a match."""
        try:
            from core import database
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: database.get_match_by_id(match_id)
            )
            if result:
                return {"success": True, "match": result}
            else:
                return {"error": f"Match not found: {match_id}"}
        except ImportError:
            return self._not_configured_response("database")
        except Exception as e:
            return {"error": str(e)}
    
    async def _draft_message(
        self,
        match_id: str,
        context: str,
        tone: str = "friendly"
    ) -> dict:
        """Draft a message to a match (never auto-sends)."""
        try:
            from actions import action_08_auto_messaging
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: action_08_auto_messaging.draft_message(
                    match_id=match_id,
                    context=context,
                    tone=tone,
                    auto_send=False  # Always require human approval
                )
            )
            return {
                "success": True,
                "draft_id": result.get("draft_id"),
                "preview": result.get("message_preview"),
                "note": "Message saved to review queue. Use the main.py menu to review and send."
            }
        except ImportError:
            return self._not_configured_response("messaging")
        except Exception as e:
            return {"error": str(e)}
    
    async def _list_pending_drafts(self) -> dict:
        """List pending message drafts."""
        try:
            from actions import action_08_auto_messaging
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                action_08_auto_messaging.list_pending_drafts
            )
            return {"success": True, "drafts": result}
        except ImportError:
            return self._not_configured_response("messaging")
        except Exception as e:
            return {"error": str(e)}
    
    async def _query_local_database(self, query: str) -> dict:
        """Run a read-only SQL query."""
        # Security: only allow SELECT queries
        query_upper = query.strip().upper()
        if not query_upper.startswith("SELECT"):
            return {"error": "Only SELECT queries are allowed for safety"}
        
        try:
            from core import database
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: database.execute_query(query)
            )
            return {"success": True, "results": result}
        except ImportError:
            return self._not_configured_response("database")
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_database_schema(self) -> dict:
        """Get database schema information."""
        try:
            from core import database
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                database.get_schema
            )
            return {"success": True, "schema": result}
        except ImportError:
            return self._not_configured_response("database")
        except Exception as e:
            return {"error": str(e)}
    
    def _not_configured_response(self, feature: str) -> dict:
        """Return a helpful message when ancestry modules aren't available."""
        return {
            "error": f"The {feature} module is not available.",
            "hint": f"Make sure ANCESTRY_ROOT is set correctly. Current value: {ANCESTRY_ROOT}",
            "required": "The waynegault/ancestry repo must be properly configured with all dependencies."
        }
    
    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


def main():
    """Entry point for the MCP server."""
    # Verify ancestry root exists
    if not Path(ANCESTRY_ROOT).exists():
        logger.warning(f"ANCESTRY_ROOT does not exist: {ANCESTRY_ROOT}")
        logger.warning("Set the ANCESTRY_ROOT environment variable to your ancestry repo path")
    
    server = AncestryMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
