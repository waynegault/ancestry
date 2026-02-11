# Ancestry MCP Server

An MCP (Model Context Protocol) server that wraps the waynegault/ancestry toolkit, enabling AI assistants to interact with your DNA matching, GEDCOM parsing, and messaging capabilities.

## Prerequisites

1. The main ancestry repository must be set up and working
2. Python 3.10+
3. MCP library (`pip install mcp`)

## Installation

```bash
# From the ancestry repo root
cd mcp_server
pip install -r requirements.txt
```

## Configuration

Set the `ANCESTRY_ROOT` environment variable to point to your ancestry repo:

```bash
export ANCESTRY_ROOT=/path/to/waynegault/ancestry
```

Or on Windows:
```cmd
set ANCESTRY_ROOT=C:\path\to\waynegault\ancestry
```

## Running the Server

```bash
python server.py
```

## Claude Desktop Integration

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ancestry": {
      "command": "python",
      "args": ["/path/to/ancestry/mcp_server/server.py"],
      "env": {
        "ANCESTRY_ROOT": "/path/to/ancestry"
      }
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `gather_dna_matches` | Fetch DNA matches with checkpoint support |
| `search_inbox` | Process inbox messages with AI classification |
| `search_gedcom` | Search GEDCOM files with API fallback |
| `get_shared_matches` | Find shared matches for triangulation |
| `run_triangulation` | Cluster analysis on DNA matches |
| `get_match_details` | Get detailed info on a specific match |
| `draft_message` | AI-generate messages (saved to review queue) |
| `list_pending_drafts` | View pending message drafts |
| `query_local_database` | Read-only SQL queries |
| `get_database_schema` | Explore database structure |

## Safety Features

- **Messages are never auto-sent** - All drafted messages go to a review queue
- **Database queries are read-only** - Only SELECT statements allowed
- **Checkpointing** - Long operations can be resumed if interrupted

## Troubleshooting

If you see "module not available" errors:
1. Verify `ANCESTRY_ROOT` is set correctly
2. Ensure the ancestry repo is fully configured
3. Check that all ancestry dependencies are installed
