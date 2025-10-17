# Blackbox AI & MCP Agent Setup Guide

## ✅ Current Setup Status

### What's Working:
- ✅ **Supermemory MCP Server**: Installed and configured
- ✅ **API Key**: Configured with your Supermemory API key
- ✅ **Dependencies**: All npm packages installed
- ✅ **Configuration File**: `blackbox_mcp_settings.json` properly formatted

---

## 🚀 How to Use with Blackbox AI

### Option 1: Using Blackbox AI Extension in VSCode

1. **Install Blackbox AI Extension**:
   - Open VSCode Extensions (Ctrl+Shift+X)
   - Search for "Blackbox AI"
   - Install the official extension

2. **Configure MCP Server**:
   - The extension should automatically detect `blackbox_mcp_settings.json`
   - If not, you may need to point it to this file in settings

3. **Start the MCP Server**:
   ```bash
   cd supermemory-mcp
   npx wrangler dev --port 3000
   ```

4. **Use Blackbox AI**:
   - Open Blackbox AI chat in VSCode
   - The Supermemory MCP server will provide memory capabilities
   - Your conversations and context will be stored and retrieved automatically

---

### Option 2: Using with Claude Desktop (Recommended for MCP)

Claude Desktop has the best MCP support. Here's how to set it up:

1. **Install Claude Desktop**:
   - Download from: https://claude.ai/download
   - Install and sign in

2. **Configure Claude Desktop**:
   - Create/edit: `%APPDATA%\Claude\claude_desktop_config.json` (Windows)
   - Or: `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac)
   
   ```json
   {
     "mcpServers": {
       "supermemory": {
         "command": "npx",
         "args": ["wrangler", "dev", "--port", "3000"],
         "cwd": "C:/Users/wayne/GitHub/Python/Projects/Ancestry/supermemory-mcp",
         "env": {
           "SUPERMEMORY_API_KEY": "sm_oTX6DBkn8vBr5FRGhxvWbW_nawDmjWFwwqTZrIcVdSCnAMrAjuwddLWjBClggTxktuAnnKuzFXfoOjpzaEhawer"
         }
       }
     }
   }
   ```

3. **Restart Claude Desktop**:
   - Close and reopen Claude Desktop
   - The MCP server will start automatically
   - You'll see a 🔌 icon indicating MCP is connected

---

### Option 3: Using with Cline/Roo-Cline (VSCode Extension)

1. **Install Cline Extension**:
   - Open VSCode Extensions (Ctrl+Shift+X)
   - Search for "Cline" or "Roo-Cline"
   - Install the extension

2. **Configure MCP in Cline**:
   - Open Cline settings
   - Add MCP server configuration:
   ```json
   {
     "mcpServers": {
       "supermemory": {
         "command": "npx",
         "args": ["wrangler", "dev", "--port", "3000"],
         "cwd": "C:/Users/wayne/GitHub/Python/Projects/Ancestry/supermemory-mcp",
         "env": {
           "SUPERMEMORY_API_KEY": "sm_oTX6DBkn8vBr5FRGhxvWbW_nawDmjWFwwqTZrIcVdSCnAMrAjuwddLWjBClggTxktuAnnKuzFXfoOjpzaEhawer"
         }
       }
     }
   }
   ```

3. **Restart VSCode**:
   - Reload window (Ctrl+Shift+P → "Developer: Reload Window")
   - Cline will connect to the MCP server

---

## 🧪 Testing Your Setup

### Test 1: Verify MCP Server Starts

```bash
cd supermemory-mcp
npx wrangler dev --port 3000
```

**Expected Output**:
```
⛅️ wrangler 4.43.0
-------------------
⎔ Starting local server...
[wrangler:inf] Ready on http://localhost:3000
```

### Test 2: Test API Connection

Open a new terminal and run:
```bash
curl http://localhost:3000/health
```

**Expected**: Should return a health check response

### Test 3: Test with AI Client

1. Open your AI client (Blackbox, Claude, or Cline)
2. Ask: "Can you remember that my favorite color is blue?"
3. In a new conversation, ask: "What's my favorite color?"
4. The AI should retrieve the information from Supermemory

---

## 📁 File Structure

```
Ancestry/
├── blackbox_mcp_settings.json          # ✅ MCP configuration (updated)
├── supermemory-mcp/                    # ✅ MCP server directory
│   ├── package.json                    # ✅ Dependencies installed
│   ├── wrangler.jsonc                  # Cloudflare config
│   └── app/                            # Server code
└── BLACKBOX_SETUP_GUIDE.md            # 📖 This guide
```

---

## 🔧 Troubleshooting

### Issue 1: "Command not found: npx"
**Solution**: Install Node.js from https://nodejs.org/

### Issue 2: "Wrangler authentication required"
**Solution**: 
```bash
npx wrangler login
```

### Issue 3: "Port 3000 already in use"
**Solution**: Change port in configuration:
```json
"args": ["wrangler", "dev", "--port", "3001"]
```

### Issue 4: "API key invalid"
**Solution**: 
1. Visit https://console.supermemory.ai
2. Generate a new API key
3. Update in `blackbox_mcp_settings.json`

### Issue 5: MCP Server Not Connecting
**Solution**:
1. Check if server is running: `curl http://localhost:3000/health`
2. Check logs in terminal where wrangler is running
3. Restart your AI client
4. Verify configuration file path is correct

---

## 🎯 What Supermemory MCP Provides

### Memory Capabilities:
- **Store Information**: Save facts, preferences, and context
- **Retrieve Information**: Query stored memories across conversations
- **Persistent Memory**: Memories persist across sessions
- **Cross-Platform**: Use same memories in different AI clients

### Example Use Cases:
1. **Project Context**: "Remember that this is an Ancestry genealogy project"
2. **Preferences**: "I prefer Python over JavaScript"
3. **Facts**: "My DNA matches are stored in Data/ancestry.db"
4. **Relationships**: "Wayne Gault is the project owner"

---

## 📚 Additional Resources

- **Supermemory Documentation**: https://supermemory.ai/docs
- **MCP Protocol**: https://modelcontextprotocol.io/
- **Wrangler Docs**: https://developers.cloudflare.com/workers/wrangler/

---

## ⚠️ Important Notes

1. **API Key Security**: Your API key is stored in plain text. Keep this file secure.
2. **MCP v1 Deprecation**: Consider upgrading to latest version from app.supermemory.ai
3. **Port Conflicts**: Ensure port 3000 is available or change to another port
4. **Cloudflare Account**: Some features may require Cloudflare account setup

---

## 🎉 Next Steps

1. **Choose your AI client** (Blackbox, Claude Desktop, or Cline)
2. **Start the MCP server** (`cd supermemory-mcp && npx wrangler dev --port 3000`)
3. **Test the connection** with your AI client
4. **Start using memory features** in your conversations

---

**Setup Date**: January 2025
**Status**: ✅ Ready to Use
**API Key**: Configured
**Dependencies**: Installed
