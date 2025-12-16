# Microsoft Graph Integration Setup Guide

## Overview

The Ancestry platform integrates with Microsoft Graph API to create To-Do tasks from genealogical research activities. This enables automated task management in Microsoft 365 productivity tools.

## Prerequisites

1. **Microsoft 365 Account** (personal, work, or school)
2. **Azure AD App Registration** (optional for development)
3. **Environment Configuration**

## Quick Setup

### 1. Configure Environment Variables

Add the following to your `.env` file:

```env
# Microsoft Graph Configuration
MS_GRAPH_CLIENT_ID=your-client-id-here
MS_GRAPH_TENANT_ID=consumers    # Use 'consumers' for personal accounts

# To-Do List Configuration
MS_TODO_LIST_NAME=Ancestry Research Tasks
```

### 2. First-Time Authentication

The first time MS Graph is used, you'll see:

```
========================================
 MS GRAPH AUTHENTICATION REQUIRED
========================================
1. Open a web browser to: https://microsoft.com/devicelogin
2. Enter the code: XXXXXXXX
3. Sign in with your Microsoft account and grant permissions.
   (Waiting for authentication in browser...)
========================================
```

After authentication, the token is cached in `Data/ms_graph_cache.bin` and reused automatically.

## Architecture

### Token Flow

```
┌────────────────────────────────────────┐
│           Application Startup          │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│     Load cached token from cache.bin   │
└──────────────────┬─────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    Token Valid?        Token Expired?
         │                   │
         ▼                   ▼
    Use Token        Device Flow Auth
         │                   │
         └─────────┬─────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│         Graph API Calls Ready          │
└────────────────────────────────────────┘
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `acquire_token_device_flow()` | Main authentication entry point |
| `get_todo_list_id(token, list_name)` | Look up To-Do list by name |
| `create_todo_task(token, list_id, title, body)` | Create a new task |

### Cache Management

- **Cache Location**: `Data/ms_graph_cache.bin`
- **Auto-Save**: Cache is saved on script exit via `atexit` handler
- **Token Refresh**: Silent refresh attempted before device flow

## Usage Examples

### Action 9: Task Creation

```python
from integrations import ms_graph_utils

# Authentication (typically done at startup)
token = ms_graph_utils.acquire_token_device_flow()

# Get list ID
list_id = ms_graph_utils.get_todo_list_id(token, "Ancestry Research Tasks")

# Create task
result = ms_graph_utils.create_todo_task(
    token=token,
    list_id=list_id,
    title="Research John Smith birth record",
    body="Check Ohio vital records 1845-1855"
)
```

### Checking Task Creation

```python
if result and result.get("id"):
    logger.info(f"Task created: {result['id']}")
else:
    logger.warning("Task creation failed")
```

## Required Permissions (Scopes)

| Scope | Description |
|-------|-------------|
| `Tasks.ReadWrite` | Read and create To-Do tasks |
| `User.Read` | Read basic profile for logging |

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `MS_GRAPH_CLIENT_ID not found` | Missing .env config | Add CLIENT_ID to .env |
| Device flow timeout | Browser auth took too long | Retry, complete auth faster |
| `Failed to find list` | List name mismatch | Check exact list name in To-Do |
| Token refresh fails | Cache corrupted | Delete `Data/ms_graph_cache.bin` |

### Debug Commands

```powershell
# Check for MS Graph errors in logs
Select-String -Path Logs\app.log -Pattern "MS Graph|MSAL|device flow"

# Verify token cache exists
Test-Path Data\ms_graph_cache.bin

# Force re-authentication (delete cache)
Remove-Item Data\ms_graph_cache.bin
```

### Authentication Logs

Look for these log messages:

```
DEBUG: Loading MSAL cache from: Data\ms_graph_cache.bin
DEBUG: Account(s) found in cache. Attempting silent token acquisition...
INFO: Access token acquired silently from cache.
```

## Azure AD App Registration (Advanced)

For production or custom scopes:

1. Go to [Azure Portal](https://portal.azure.com) → Azure Active Directory → App registrations
2. Create new registration with:
   - Name: "Ancestry Research Automation"
   - Supported account types: "Personal Microsoft accounts only" (or as needed)
   - Redirect URI: `https://login.microsoftonline.com/common/oauth2/nativeclient`
3. Copy Application (client) ID to `MS_GRAPH_CLIENT_ID`
4. Under API permissions, add: `Tasks.ReadWrite`, `User.Read`

## Security Considerations

- **Token Storage**: Tokens cached locally in `Data/` directory
- **Scope Limitation**: Only task read/write permissions requested
- **No Secrets**: Uses device code flow (no client secret needed for public clients)
- **Audit**: All task creation logged with person ID and task description

## Integration Points

| Component | Integration |
|-----------|-------------|
| `main.py` | Authenticates at startup if enabled |
| `action9_process_productive.py` | Creates tasks from PRODUCTIVE messages |
| `TaskCreator` class | High-level task creation wrapper |

## Testing

Run MS Graph integration tests:

```powershell
python -m integrations.ms_graph_utils
```

Tests verify:
- Token acquisition mocking
- Task creation API calls
- Cache save/load functionality
