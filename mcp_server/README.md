# Ancestry MCP Server

An MCP (Model Context Protocol) server that wraps the Ancestry research toolkit, enabling AI assistants to query your DNA match database, search matches, review pending drafts, and explore genealogical data.

## Prerequisites

1. The main Ancestry repository must be set up and working
2. Python 3.10+
3. MCP library (already in `requirements.txt`: `mcp>=1.0.0`)

## Running the Server

### From main.py menu
```
python main.py
# Select option 's' - Start MCP Server
```

### As a Python module
```bash
python -m mcp_server
```

### Directly
```bash
python mcp_server/server.py
```

## Claude Desktop Integration

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ancestry": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "C:\\path\\to\\ancestry"
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `query_database` | Run read-only SQL queries against the local SQLite database |
| `get_database_schema` | Explore database structure (tables, columns, indexes) |
| `get_match_summary` | Statistical summary: totals, averages, relationship distribution, top matches |
| `get_match_details` | Detailed info on a specific match by ID or UUID |
| `search_matches` | Search matches by name, cM range, status |
| `list_pending_drafts` | View pending message drafts awaiting review |
| `get_shared_matches` | Find shared matches between you and a specific person |
| `get_conversation_history` | View message history for a person |

## Safety Features

- **Messages are never auto-sent** — All drafted messages go to a review queue
- **Database queries are read-only** — Only SELECT/WITH statements allowed; INSERT/UPDATE/DELETE/DROP are blocked
- **No browser automation** — The MCP server only accesses the local SQLite database

## Testing

Tests follow the codebase convention and are discovered automatically by `run_all_tests.py`:

```bash
# Run MCP server tests only
python -m mcp_server.server

# Run all project tests (includes MCP server)
python run_all_tests.py
```
