# Filesystem MCP Server Setup

## Configuration

The filesystem MCP server has been successfully configured in `blackbox_mcp_settings.json` with the following settings:

```json
"github.com/modelcontextprotocol/servers/tree/main/src/filesystem": {
  "command": "npx",
  "args": [
    "-y",
    "@modelcontextprotocol/server-filesystem",
    "c:/Users/wayne/GitHub/Python/Projects/Ancestry"
  ]
}
```

## Server Details

- **Server Name**: `github.com/modelcontextprotocol/servers/tree/main/src/filesystem`
- **Installation Method**: NPX (Node Package Manager)
- **Package**: `@modelcontextprotocol/server-filesystem`
- **Allowed Directory**: `c:/Users/wayne/GitHub/Python/Projects/Ancestry`

## Available Tools

The filesystem server provides the following tools:

1. **read_text_file** - Read complete contents of a file as text
2. **read_media_file** - Read an image or audio file
3. **read_multiple_files** - Read multiple files simultaneously
4. **write_file** - Create new file or overwrite existing
5. **edit_file** - Make selective edits using pattern matching
6. **create_directory** - Create new directory or ensure it exists
7. **list_directory** - List directory contents
8. **list_directory_with_sizes** - List directory contents with file sizes
9. **move_file** - Move or rename files and directories
10. **search_files** - Recursively search for files/directories
11. **directory_tree** - Get recursive JSON tree structure
12. **get_file_info** - Get detailed file/directory metadata
13. **list_allowed_directories** - List all accessible directories

## Security

The server is configured with directory access control, restricting all filesystem operations to the specified allowed directory: `c:/Users/wayne/GitHub/Python/Projects/Ancestry`

## Next Steps

To use the server:
1. Restart BLACKBOX to load the new MCP server configuration
2. The filesystem server will be available for use with all its tools
3. All operations will be sandboxed to the allowed directory

## Demonstration

See below for a demonstration of the server's capabilities.
