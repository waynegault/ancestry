#!/usr/bin/env python3
"""
mcp_server/server.py - Ancestry MCP Server

Exposes the Ancestry research toolkit as MCP (Model Context Protocol) tools,
enabling AI assistants to query DNA matches, search the database, review
pending drafts, and access genealogical data.

Safety features:
    - Messages are NEVER auto-sent; all drafts go to a review queue.
    - Database queries are read-only (SELECT only).
    - All operations use the existing DatabaseManager stack.

Usage:
    # From the project root
    python -m mcp_server

    # Or from main.py menu option 's'
"""

# === CORE INFRASTRUCTURE ===
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.database import (
    ConversationLog,
    DnaMatch,
    DraftReply,
    Person,
    PersonStatusEnum,
    SharedMatch,
)
from core.database_manager import DatabaseManager
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOL_DEFINITIONS: list[Any] = [
    Tool(
        name="query_database",
        description=(
            "Run a read-only SQL SELECT query against the local Ancestry SQLite "
            "database. Returns rows as a list of dicts. Only SELECT statements "
            "are permitted."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A SQL SELECT query (e.g. 'SELECT id, username FROM people LIMIT 10')",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="get_database_schema",
        description=(
            "Return the full schema of the local Ancestry SQLite database "
            "including all table names, column names, types, and indexes."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="get_match_summary",
        description=(
            "Get a statistical summary of DNA match data: total matches, average "
            "shared cM, relationship distribution, and top matches."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "min_cm": {
                    "type": "number",
                    "description": "Minimum shared cM to include (default: 0)",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max top matches to return (default: 20)",
                    "default": 20,
                },
            },
        },
    ),
    Tool(
        name="get_match_details",
        description=(
            "Get detailed information about a specific DNA match by person ID "
            "or UUID, including shared DNA, predicted relationship, tree info, "
            "and conversation history."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Person ID (integer) or UUID string to look up",
                },
            },
            "required": ["identifier"],
        },
    ),
    Tool(
        name="search_matches",
        description=(
            "Search DNA matches by name, minimum shared cM, relationship, or "
            "status. Returns matching Person records with DNA data."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Partial name match (case-insensitive)",
                },
                "min_cm": {
                    "type": "number",
                    "description": "Minimum shared cM",
                },
                "max_cm": {
                    "type": "number",
                    "description": "Maximum shared cM",
                },
                "status": {
                    "type": "string",
                    "description": "Person status filter (ACTIVE, DESIST, ARCHIVE, BLOCKED, DEAD)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 50)",
                    "default": 50,
                },
            },
        },
    ),
    Tool(
        name="list_pending_drafts",
        description=(
            "List all pending message drafts awaiting human review. Drafts are "
            "NEVER auto-sent. Shows draft ID, recipient, preview, and creation date."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum drafts to return (default: 50)",
                    "default": 50,
                },
            },
        },
    ),
    Tool(
        name="get_shared_matches",
        description=(
            "Get shared DNA matches for a given person. These are matches that "
            "share DNA with both you and the specified person."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "person_id": {
                    "type": "integer",
                    "description": "The person ID to find shared matches for",
                },
                "min_cm": {
                    "type": "number",
                    "description": "Minimum shared cM filter (default: 20)",
                    "default": 20,
                },
            },
            "required": ["person_id"],
        },
    ),
    Tool(
        name="get_conversation_history",
        description=(
            "Get the conversation history for a specific person. Returns "
            "messages exchanged, direction (IN/OUT), timestamps, and AI classification."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "person_id": {
                    "type": "integer",
                    "description": "The person ID to get conversation history for",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum messages to return (default: 50)",
                    "default": 50,
                },
            },
            "required": ["person_id"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _serialize_row(row: Any) -> dict[str, Any]:
    """Convert a SQLAlchemy row/mapping to a JSON-safe dict."""
    if hasattr(row, "_asdict"):
        return {k: _safe_value(v) for k, v in row._asdict().items()}
    if hasattr(row, "_mapping"):
        return {k: _safe_value(v) for k, v in row._mapping.items()}
    if isinstance(row, dict):
        return {k: _safe_value(v) for k, v in row.items()}
    return {"value": str(row)}


def _safe_value(v: Any) -> Any:
    """Make a value JSON-serializable."""
    if v is None or isinstance(v, (int, float, bool, str)):
        return v
    return str(v)


def _person_to_dict(person: Person) -> dict[str, Any]:
    """Convert a Person ORM object to a serializable dict."""
    result: dict[str, Any] = {
        "id": person.id,
        "uuid": person.uuid,
        "profile_id": person.profile_id,
        "username": person.username,
        "first_name": person.first_name,
        "status": str(person.status.value) if person.status else None,
        "in_my_tree": person.in_my_tree,
        "contactable": person.contactable,
    }
    if person.dna_match:
        dm = person.dna_match
        result["dna"] = {
            "cm_dna": dm.cm_dna,
            "predicted_relationship": dm.predicted_relationship,
            "shared_segments": dm.shared_segments,
            "longest_shared_segment": dm.longest_shared_segment,
            "has_public_tree": dm.has_public_tree,
            "tree_size": dm.tree_size,
        }
    return result


# ---------------------------------------------------------------------------
# Database helper
# ---------------------------------------------------------------------------


def _get_db_manager() -> DatabaseManager:
    """Obtain a ready DatabaseManager instance."""
    db = DatabaseManager()
    if not db.ensure_ready():
        raise RuntimeError("Failed to initialise database connection")
    return db


# ---------------------------------------------------------------------------
# Tool implementations (synchronous, run in executor by the async layer)
# ---------------------------------------------------------------------------


def _tool_query_database(arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a read-only SQL query."""
    from sqlalchemy import text

    query_str = arguments.get("query", "").strip()
    if not query_str:
        return {"error": "No query provided"}

    # Security: only allow SELECT/WITH statements
    normalised = query_str.lstrip("(").upper()
    if not normalised.startswith("SELECT") and not normalised.startswith("WITH"):
        return {"error": "Only SELECT/WITH queries are allowed for safety"}

    # Block dangerous keywords even within CTEs
    blocked = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "REPLACE"}
    tokens = set(normalised.split())
    if tokens & blocked:
        return {"error": f"Query contains disallowed keywords: {tokens & blocked}"}

    db = _get_db_manager()
    with db.get_session_context() as session:
        if session is None:
            return {"error": "Database session unavailable"}
        result = session.execute(text(query_str))
        rows = [_serialize_row(r) for r in result.fetchall()]
        return {"success": True, "row_count": len(rows), "rows": rows}


def _tool_get_database_schema(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the SQLite schema."""
    from sqlalchemy import text

    db = _get_db_manager()
    with db.get_session_context() as session:
        if session is None:
            return {"error": "Database session unavailable"}

        tables_result = session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        )
        tables: list[dict[str, Any]] = []
        for (table_name,) in tables_result.fetchall():
            cols_result = session.execute(text(f"PRAGMA table_info('{table_name}')"))
            columns: list[dict[str, Any]] = []
            for col in cols_result.fetchall():
                col_dict = _serialize_row(col)
                columns.append({
                    "name": col_dict.get("name", col_dict.get("value", "")),
                    "type": col_dict.get("type", ""),
                    "notnull": col_dict.get("notnull", 0),
                    "pk": col_dict.get("pk", 0),
                })

            idx_result = session.execute(text(f"PRAGMA index_list('{table_name}')"))
            indexes = [_serialize_row(idx) for idx in idx_result.fetchall()]

            tables.append({
                "table": table_name,
                "columns": columns,
                "indexes": indexes,
            })
        return {"success": True, "tables": tables}


def _tool_get_match_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    """Get DNA match statistics."""
    from sqlalchemy import func
    from sqlalchemy.orm import joinedload

    min_cm = arguments.get("min_cm", 0)
    limit = arguments.get("limit", 20)

    db = _get_db_manager()
    with db.get_session_context() as session:
        if session is None:
            return {"error": "Database session unavailable"}

        base_q = (
            session.query(Person)
            .join(DnaMatch)
            .filter(Person.deleted_at.is_(None))
        )
        if min_cm > 0:
            base_q = base_q.filter(DnaMatch.cm_dna >= min_cm)

        total = base_q.count()

        stats = (
            session.query(
                func.avg(DnaMatch.cm_dna).label("avg_cm"),
                func.min(DnaMatch.cm_dna).label("min_cm"),
                func.max(DnaMatch.cm_dna).label("max_cm"),
                func.sum(DnaMatch.cm_dna).label("total_cm"),
            )
            .join(Person)
            .filter(Person.deleted_at.is_(None))
        )
        if min_cm > 0:
            stats = stats.filter(DnaMatch.cm_dna >= min_cm)
        stat_row: Any = stats.first()

        # Relationship distribution
        rel_q = (
            session.query(
                DnaMatch.predicted_relationship,
                func.count().label("count"),
            )
            .join(Person)
            .filter(Person.deleted_at.is_(None))
            .group_by(DnaMatch.predicted_relationship)
            .order_by(func.count().desc())
        )
        if min_cm > 0:
            rel_q = rel_q.filter(DnaMatch.cm_dna >= min_cm)
        relationships = {r: c for r, c in rel_q.all() if r}

        # Top matches
        top = (
            base_q
            .options(joinedload(Person.dna_match))
            .order_by(DnaMatch.cm_dna.desc())
            .limit(limit)
            .all()
        )
        top_matches = [_person_to_dict(p) for p in top]

        return {
            "success": True,
            "total_matches": total,
            "statistics": {
                "avg_cm": round(float(stat_row.avg_cm or 0), 1),
                "min_cm": int(stat_row.min_cm or 0),
                "max_cm": int(stat_row.max_cm or 0),
                "total_cm": int(stat_row.total_cm or 0),
            },
            "relationship_distribution": relationships,
            "top_matches": top_matches,
        }


def _tool_get_match_details(arguments: dict[str, Any]) -> dict[str, Any]:
    """Look up a specific DNA match by ID or UUID."""
    from sqlalchemy.orm import joinedload

    identifier = arguments.get("identifier", "").strip()
    if not identifier:
        return {"error": "No identifier provided"}

    db = _get_db_manager()
    with db.get_session_context() as session:
        if session is None:
            return {"error": "Database session unavailable"}

        person = None
        # Try numeric ID first
        try:
            person_id = int(identifier)
            person = (
                session.query(Person)
                .options(
                    joinedload(Person.dna_match),
                    joinedload(Person.family_tree),
                    joinedload(Person.conversation_log_entries),
                )
                .filter(Person.id == person_id, Person.deleted_at.is_(None))
                .first()
            )
        except ValueError:
            pass

        # Fall back to UUID lookup (always uppercase)
        if person is None:
            person = (
                session.query(Person)
                .options(
                    joinedload(Person.dna_match),
                    joinedload(Person.family_tree),
                    joinedload(Person.conversation_log_entries),
                )
                .filter(Person.uuid == identifier.upper(), Person.deleted_at.is_(None))
                .first()
            )

        if person is None:
            return {"error": f"No match found for identifier: {identifier}"}

        result = _person_to_dict(person)

        # Add family tree info if available
        if person.family_tree:
            ft = person.family_tree
            result["family_tree"] = {
                "cfpid": getattr(ft, "cfpid", None),
                "actual_relationship": getattr(ft, "actual_relationship", None),
                "relationship_path": getattr(ft, "relationship_path", None),
            }

        # Add recent conversations
        if person.conversation_log_entries:
            convos = sorted(
                person.conversation_log_entries,
                key=lambda c: c.timestamp if c.timestamp else "",
                reverse=True,
            )[:10]
            result["recent_conversations"] = [
                {
                    "direction": str(c.direction.value) if c.direction else None,
                    "timestamp": str(c.timestamp) if c.timestamp else None,
                    "message_preview": (c.message_body[:200] + "...") if c.message_body and len(c.message_body) > 200 else c.message_body,
                    "classification": c.classification,
                }
                for c in convos
            ]

        return {"success": True, "match": result}


def _tool_search_matches(arguments: dict[str, Any]) -> dict[str, Any]:
    """Search DNA matches with filters."""
    from sqlalchemy.orm import joinedload

    db = _get_db_manager()
    with db.get_session_context() as session:
        if session is None:
            return {"error": "Database session unavailable"}

        q = (
            session.query(Person)
            .join(DnaMatch)
            .options(joinedload(Person.dna_match))
            .filter(Person.deleted_at.is_(None))
        )

        name = arguments.get("name")
        if name:
            q = q.filter(Person.username.ilike(f"%{name}%"))

        min_cm = arguments.get("min_cm")
        if min_cm is not None:
            q = q.filter(DnaMatch.cm_dna >= min_cm)

        max_cm = arguments.get("max_cm")
        if max_cm is not None:
            q = q.filter(DnaMatch.cm_dna <= max_cm)

        status = arguments.get("status")
        if status:
            try:
                status_enum = PersonStatusEnum(status.upper())
                q = q.filter(Person.status == status_enum)
            except ValueError:
                return {"error": f"Invalid status: {status}. Valid: ACTIVE, DESIST, ARCHIVE, BLOCKED, DEAD"}

        limit = arguments.get("limit", 50)
        q = q.order_by(DnaMatch.cm_dna.desc()).limit(limit)

        matches = [_person_to_dict(p) for p in q.all()]
        return {"success": True, "count": len(matches), "matches": matches}


def _tool_list_pending_drafts(arguments: dict[str, Any]) -> dict[str, Any]:
    """List pending message drafts."""
    db = _get_db_manager()
    limit = arguments.get("limit", 50)
    with db.get_session_context() as session:
        if session is None:
            return {"error": "Database session unavailable"}

        drafts = (
            session.query(DraftReply)
            .filter(DraftReply.status == "PENDING")
            .order_by(DraftReply.created_at.desc())
            .limit(limit)
            .all()
        )

        result: list[dict[str, Any]] = []
        for d in drafts:
            person = session.query(Person).filter(Person.id == d.people_id).first()
            result.append({
                "draft_id": d.id,
                "person_id": d.people_id,
                "recipient": person.username if person else "Unknown",
                "message_preview": (d.body[:200] + "...") if d.body and len(d.body) > 200 else d.body,
                "created_at": str(d.created_at) if d.created_at else None,
                "status": d.status,
            })

        return {
            "success": True,
            "count": len(result),
            "note": "Drafts are NEVER auto-sent. Use the main.py Review Queue to approve and send.",
            "drafts": result,
        }


def _tool_get_shared_matches(arguments: dict[str, Any]) -> dict[str, Any]:
    """Get shared matches for a person."""
    person_id = arguments.get("person_id")
    if person_id is None:
        return {"error": "person_id is required"}

    min_cm = arguments.get("min_cm", 20)
    db = _get_db_manager()
    with db.get_session_context() as session:
        if session is None:
            return {"error": "Database session unavailable"}

        person = session.query(Person).filter(Person.id == person_id, Person.deleted_at.is_(None)).first()
        if not person:
            return {"error": f"Person {person_id} not found"}

        shared = (
            session.query(SharedMatch)
            .filter(SharedMatch.people_id == person_id)
            .all()
        )

        results: list[dict[str, Any]] = []
        for sm in shared:
            shared_person = (
                session.query(Person)
                .join(DnaMatch)
                .filter(Person.id == sm.shared_people_id, Person.deleted_at.is_(None))
                .first()
            )
            if shared_person and shared_person.dna_match:
                if shared_person.dna_match.cm_dna and shared_person.dna_match.cm_dna >= min_cm:
                    results.append({
                        "person_id": shared_person.id,
                        "username": shared_person.username,
                        "shared_cm": shared_person.dna_match.cm_dna,
                        "predicted_relationship": shared_person.dna_match.predicted_relationship,
                    })

        results.sort(key=lambda x: x.get("shared_cm", 0), reverse=True)
        return {
            "success": True,
            "person": person.username,
            "count": len(results),
            "shared_matches": results,
        }


def _tool_get_conversation_history(arguments: dict[str, Any]) -> dict[str, Any]:
    """Get conversation history for a person."""
    person_id = arguments.get("person_id")
    if person_id is None:
        return {"error": "person_id is required"}

    limit = arguments.get("limit", 50)
    db = _get_db_manager()
    with db.get_session_context() as session:
        if session is None:
            return {"error": "Database session unavailable"}

        person = session.query(Person).filter(Person.id == person_id, Person.deleted_at.is_(None)).first()
        if not person:
            return {"error": f"Person {person_id} not found"}

        entries = (
            session.query(ConversationLog)
            .filter(ConversationLog.people_id == person_id)
            .order_by(ConversationLog.timestamp.desc())
            .limit(limit)
            .all()
        )

        messages: list[dict[str, Any]] = []
        for e in entries:
            messages.append({
                "direction": str(e.direction.value) if e.direction else None,
                "timestamp": str(e.timestamp) if e.timestamp else None,
                "message_body": e.message_body,
                "classification": e.classification,
                "sentiment": getattr(e, "sentiment", None),
            })

        messages.reverse()  # Chronological order
        return {
            "success": True,
            "person": person.username,
            "count": len(messages),
            "messages": messages,
        }


# ---------------------------------------------------------------------------
# Tool dispatch map
# ---------------------------------------------------------------------------

_TOOL_DISPATCH: dict[str, Any] = {
    "query_database": _tool_query_database,
    "get_database_schema": _tool_get_database_schema,
    "get_match_summary": _tool_get_match_summary,
    "get_match_details": _tool_get_match_details,
    "search_matches": _tool_search_matches,
    "list_pending_drafts": _tool_list_pending_drafts,
    "get_shared_matches": _tool_get_shared_matches,
    "get_conversation_history": _tool_get_conversation_history,
}


# ---------------------------------------------------------------------------
# MCP Server class
# ---------------------------------------------------------------------------


class AncestryMCPServer:
    """MCP Server that exposes the Ancestry toolkit as read-only tools."""

    def __init__(self) -> None:
        self.server: Any = Server("ancestry-mcp")
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Register MCP list_tools and call_tool handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Any]:
            return _TOOL_DEFINITIONS

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
            handler = _TOOL_DISPATCH.get(name)
            if handler is None:
                payload = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    loop = asyncio.get_event_loop()
                    payload = await loop.run_in_executor(None, handler, arguments)
                except Exception as exc:
                    logger.exception("Error executing tool %s", name)
                    payload = {"error": str(exc), "tool": name}
            return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]

    async def run(self) -> None:
        """Run the MCP server over stdio."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


# ---------------------------------------------------------------------------
# Entry-point helpers (called from main.py or __main__)
# ---------------------------------------------------------------------------


def start_mcp_server() -> bool:
    """Start the MCP server (blocking). Returns True on clean exit."""
    logger.info("Starting Ancestry MCP Server (stdio transport)")
    try:
        server = AncestryMCPServer()
        asyncio.run(server.run())
        return True
    except KeyboardInterrupt:
        logger.info("MCP server stopped by user")
        return True
    except Exception as exc:
        logger.error("MCP server error: %s", exc, exc_info=True)
        return False


def run_mcp_server_action() -> bool:
    """Menu-compatible action wrapper for main.py.

    This is a meta action — it runs the MCP server in the foreground
    (blocking until Ctrl+C). The server communicates over stdio.
    """
    print("\nStarting Ancestry MCP Server…")
    print("The server communicates over stdio (stdin/stdout).")
    print("Connect an MCP client (e.g. Claude Desktop) to use it.")
    print("Press Ctrl+C to stop.\n")
    return start_mcp_server()


# ---------------------------------------------------------------------------
# Module Tests
# ---------------------------------------------------------------------------


def _test_tool_definitions_valid() -> bool:
    """Verify all tool definitions have required fields."""
    assert len(_TOOL_DEFINITIONS) >= 8, f"Expected >=8 tools, got {len(_TOOL_DEFINITIONS)}"
    names: set[str] = set()
    for tool in _TOOL_DEFINITIONS:
        tool_name: str = tool.name
        tool_desc: str = tool.description
        tool_schema: dict[str, Any] = tool.inputSchema
        assert tool_name, "Tool must have a name"
        assert tool_desc, f"Tool '{tool_name}' must have a description"
        assert tool_schema, f"Tool '{tool_name}' must have an inputSchema"
        assert tool_name not in names, f"Duplicate tool name: {tool_name}"
        names.add(tool_name)
    return True


def _test_dispatch_map_covers_all_tools() -> bool:
    """Verify every defined tool has a dispatch handler."""
    tool_names: set[str] = {t.name for t in _TOOL_DEFINITIONS}
    handler_names = set(_TOOL_DISPATCH.keys())
    missing = tool_names - handler_names
    assert not missing, f"Tools without handlers: {missing}"
    extra = handler_names - tool_names
    assert not extra, f"Handlers without tool definitions: {extra}"
    return True


def _test_query_database_blocks_writes() -> bool:
    """Verify write queries are rejected."""
    for bad_query in [
        "INSERT INTO people VALUES (1)",
        "UPDATE people SET username='x'",
        "DELETE FROM people",
        "DROP TABLE people",
        "ALTER TABLE people ADD COLUMN x",
    ]:
        result = _tool_query_database({"query": bad_query})
        assert "error" in result, f"Should reject: {bad_query}"
    return True


def _test_query_database_allows_selects() -> bool:
    """Verify SELECT and WITH queries pass validation."""
    for good_query in ["SELECT 1", "WITH cte AS (SELECT 1) SELECT * FROM cte"]:
        normalised = good_query.lstrip("(").upper()
        is_valid = normalised.startswith("SELECT") or normalised.startswith("WITH")
        assert is_valid, f"Should accept: {good_query}"
    return True


def _test_query_database_empty_query() -> bool:
    """Verify empty query returns error."""
    result = _tool_query_database({"query": ""})
    assert "error" in result
    result2 = _tool_query_database({})
    assert "error" in result2
    return True


def _test_safe_value_serialization() -> bool:
    """Verify _safe_value handles all types."""
    assert _safe_value(None) is None
    assert _safe_value(42) == 42
    assert _safe_value(3.14) == 3.14
    assert _safe_value(True) is True
    assert _safe_value("hello") == "hello"
    assert isinstance(_safe_value(object()), str)
    return True


def _test_person_to_dict_structure() -> bool:
    """Verify _person_to_dict produces expected keys."""
    from unittest.mock import MagicMock

    mock_person = MagicMock(spec=Person)
    mock_person.id = 1
    mock_person.uuid = "TEST-UUID"
    mock_person.profile_id = "P123"
    mock_person.username = "Test User"
    mock_person.first_name = "Test"
    mock_person.status = PersonStatusEnum.ACTIVE
    mock_person.in_my_tree = False
    mock_person.contactable = True
    mock_person.dna_match = None

    result = _person_to_dict(mock_person)
    expected_keys = {"id", "uuid", "profile_id", "username", "first_name", "status", "in_my_tree", "contactable"}
    assert expected_keys.issubset(result.keys()), f"Missing keys: {expected_keys - result.keys()}"
    assert "dna" not in result, "Should not have dna key when dna_match is None"
    return True


def _test_person_to_dict_with_dna() -> bool:
    """Verify _person_to_dict includes DNA data when available."""
    from unittest.mock import MagicMock

    mock_person = MagicMock(spec=Person)
    mock_person.id = 2
    mock_person.uuid = "TEST-UUID-2"
    mock_person.profile_id = "P456"
    mock_person.username = "DNA User"
    mock_person.first_name = "DNA"
    mock_person.status = PersonStatusEnum.ACTIVE
    mock_person.in_my_tree = True
    mock_person.contactable = True

    mock_dna = MagicMock(spec=DnaMatch)
    mock_dna.cm_dna = 150
    mock_dna.predicted_relationship = "2nd Cousin"
    mock_dna.shared_segments = 5
    mock_dna.longest_shared_segment = 45.2
    mock_dna.has_public_tree = True
    mock_dna.tree_size = 200
    mock_person.dna_match = mock_dna

    result = _person_to_dict(mock_person)
    assert "dna" in result, "Should have dna key when dna_match exists"
    assert result["dna"]["cm_dna"] == 150
    assert result["dna"]["predicted_relationship"] == "2nd Cousin"
    return True


def _test_server_instantiation() -> bool:
    """Verify AncestryMCPServer can be instantiated."""
    server = AncestryMCPServer()
    assert server.server is not None
    assert server.server.name == "ancestry-mcp"
    return True


def _test_serialize_row_variants() -> bool:
    """Test _serialize_row with different row types."""
    d = _serialize_row({"a": 1, "b": "test"})
    assert d == {"a": 1, "b": "test"}

    d2 = _serialize_row(42)
    assert "value" in d2
    return True


def _test_tool_schema_types() -> bool:
    """Verify all tool input schemas have valid JSON Schema structure."""
    for tool in _TOOL_DEFINITIONS:
        schema: dict[str, Any] = tool.inputSchema
        tool_name: str = tool.name
        assert schema.get("type") == "object", f"Tool '{tool_name}' schema must be object type"
        assert "properties" in schema, f"Tool '{tool_name}' schema must have properties"
    return True


def _test_entry_point_functions_callable() -> bool:
    """Verify start_mcp_server and run_mcp_server_action are callable."""
    assert callable(start_mcp_server)
    assert callable(run_mcp_server_action)
    return True


def mcp_server_module_tests() -> bool:
    """Run the MCP server test suite."""
    from testing.test_framework import TestSuite

    suite = TestSuite("MCP Server", "mcp_server/server.py")
    suite.start_suite()

    suite.run_test(
        test_name="Tool definitions are valid and complete",
        test_func=_test_tool_definitions_valid,
        test_summary="All tool definitions have name, description, and inputSchema",
        method_description="Iterate _TOOL_DEFINITIONS checking required fields and uniqueness",
        expected_outcome="All tools pass validation with no duplicates",
    )
    suite.run_test(
        test_name="Dispatch map covers all tool definitions",
        test_func=_test_dispatch_map_covers_all_tools,
        test_summary="Every tool in definitions has a corresponding handler",
        method_description="Compare tool names in _TOOL_DEFINITIONS vs _TOOL_DISPATCH keys",
        expected_outcome="Both sets are identical with no missing or extra entries",
    )
    suite.run_test(
        test_name="SQL write queries are blocked",
        test_func=_test_query_database_blocks_writes,
        test_summary="INSERT, UPDATE, DELETE, DROP, ALTER are rejected",
        method_description="Pass various write queries to _tool_query_database and verify error",
        expected_outcome="All write queries return error response",
    )
    suite.run_test(
        test_name="SELECT and WITH queries pass validation",
        test_func=_test_query_database_allows_selects,
        test_summary="Valid read queries pass validation checks",
        method_description="Check normalised query detection logic for SELECT/WITH prefixes",
        expected_outcome="Both SELECT and WITH are detected as valid read queries",
    )
    suite.run_test(
        test_name="Empty query returns error",
        test_func=_test_query_database_empty_query,
        test_summary="Empty or missing query parameter is handled gracefully",
        method_description="Pass empty string and missing key to _tool_query_database",
        expected_outcome="Both cases return an error dict",
    )
    suite.run_test(
        test_name="_safe_value handles all types",
        test_func=_test_safe_value_serialization,
        test_summary="Serialisation helper handles None, int, float, bool, str, and objects",
        method_description="Call _safe_value with various types and verify output",
        expected_outcome="Primitives pass through; objects become strings",
    )
    suite.run_test(
        test_name="_person_to_dict structure without DNA",
        test_func=_test_person_to_dict_structure,
        test_summary="Person dict has all expected keys and no dna key when None",
        method_description="Create mock Person without dna_match and verify output keys",
        expected_outcome="All person fields present, no dna key",
    )
    suite.run_test(
        test_name="_person_to_dict includes DNA when available",
        test_func=_test_person_to_dict_with_dna,
        test_summary="Person dict includes nested dna data when dna_match exists",
        method_description="Create mock Person with dna_match and verify dna sub-dict",
        expected_outcome="dna key present with correct values",
    )
    suite.run_test(
        test_name="AncestryMCPServer instantiation",
        test_func=_test_server_instantiation,
        test_summary="Server object creates without errors",
        method_description="Instantiate AncestryMCPServer and check server.name",
        expected_outcome="Server created with name 'ancestry-mcp'",
    )
    suite.run_test(
        test_name="_serialize_row handles various row types",
        test_func=_test_serialize_row_variants,
        test_summary="Row serialisation handles dict and fallback cases",
        method_description="Pass dict and plain value to _serialize_row",
        expected_outcome="Dict passes through; plain value wrapped in 'value' key",
    )
    suite.run_test(
        test_name="Tool schemas have valid JSON Schema structure",
        test_func=_test_tool_schema_types,
        test_summary="All inputSchema values are object type with properties",
        method_description="Check each tool's inputSchema for type='object' and properties key",
        expected_outcome="All tool schemas pass validation",
    )
    suite.run_test(
        test_name="Entry-point functions are callable",
        test_func=_test_entry_point_functions_callable,
        test_summary="start_mcp_server and run_mcp_server_action exist and are callable",
        method_description="Check both functions are callable",
        expected_outcome="Both return True from callable() check",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(mcp_server_module_tests)


if __name__ == "__main__":
    # When run directly, start the MCP server (not tests).
    # Use `python -m mcp_server` or main.py 's' for the same behaviour.
    # Tests are discovered by run_all_tests.py via run_comprehensive_tests.
    success = start_mcp_server()
    sys.exit(0 if success else 1)
