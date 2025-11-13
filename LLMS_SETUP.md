# LLM Integration Guide: Adding Inception Mercury to Copilot & Cline

## Overview
You've successfully added Inception Mercury API to your project. This guide explains how to integrate it with:
1. **GitHub Copilot** (VS Code extension)
2. **Cline** (VS Code extension for autonomous coding)

---

## Part 1: Project Configuration (✅ DONE)

Your project is now configured with Inception Mercury:

### ai_api_test.py
- ✅ Added to `PROVIDERS` tuple
- ✅ Added to `PROVIDER_DISPLAY_NAMES` dictionary
- ✅ Implemented `_test_inception()` function
- ✅ Added to `PROVIDER_TESTERS` dictionary
- **Test it**: `python ai_api_test.py --provider inception`

### .env Configuration
```env
INCEPTION_AI_BASE_URL=https://api.inceptionlabs.ai/v1
INCEPTION_AI_MODEL=mercury
INCEPTION_API_KEY=mmmmm
```

---

## Part 2: GitHub Copilot Integration

### What is GitHub Copilot?
GitHub Copilot is VS Code's built-in AI code completion engine that suggests code as you type.

### Adding Inception Mercury to Copilot

**Note**: GitHub Copilot uses **fixed models** (GitHub Copilot and GPT-4 Turbo). You **cannot directly replace** Copilot's backend with Inception Mercury.

**However, you can configure it for use with Copilot Chat:**

1. **Open VS Code Settings**
   - Press `Ctrl+,` (or `Cmd+,` on Mac)
   - Search for "copilot"

2. **Look for "Copilot: Chat" Settings**
   - Find: `Copilot > Chat: Model`
   - Current options: Built-in (GitHub Copilot) or GPT-4 Turbo

3. **If you want to use Inception Mercury for Custom Chat Completions**
   - VS Code currently doesn't support custom OpenAI-compatible backends for Copilot natively
   - **Alternative**: Use an extension that proxies custom models (see **Cline** below)

### VS Code Extensions for Custom LLM Support

If you want Copilot-style features with Inception Mercury, use these extensions:

| Extension | Purpose | Custom LLM Support | Cost |
|-----------|---------|-------------------|------|
| **Cline** | Autonomous coding agent | ✅ Yes | Free |
| **Continue** | AI code assistant | ✅ Yes | Free |
| **Copilot** | GitHub's AI | ❌ No (fixed models) | $20/mo |

---

## Part 3: Cline Integration (Recommended)

### What is Cline?
Cline is an open-source VS Code extension for **autonomous AI coding**. It:
- ✅ Supports custom OpenAI-compatible APIs (like Inception Mercury)
- ✅ Can read/write files autonomously
- ✅ Integrates with your terminal
- ✅ Works with Claude, GPT-4, DeepSeek, and custom models

### Installation

1. **Install Cline**
   - Open VS Code Extensions (`Ctrl+Shift+X`)
   - Search: "cline"
   - Click "Install" on the official Cline extension

2. **Configure Cline for Inception Mercury**
   - Open VS Code Settings (`Ctrl+,`)
   - Search: "cline"
   - Look for: `Cline > Model Configuration`

### Configure Inception Mercury in Cline

#### Option A: Via VS Code UI (Easiest)

1. Open Cline Settings in VS Code (`Ctrl+,` → search "cline")
2. Find: **Cline: Custom Model**
3. Add configuration:

```json
{
  "models": [
    {
      "name": "inception-mercury",
      "provider": "openai",
      "baseUrl": "https://api.inceptionlabs.ai/v1",
      "apiKey": "mmmmm",
      "model": "mercury"
    }
  ]
}
```

#### Option B: Via `cline_config.json` (If available)

Create/edit `.vscode/cline_config.json` in your workspace:

```json
{
  "models": [
    {
      "name": "Inception Mercury",
      "id": "inception-mercury",
      "provider": "openai",
      "apiBaseUrl": "https://api.inceptionlabs.ai/v1",
      "apiKey": "${INCEPTION_API_KEY}",
      "modelId": "mercury",
      "maxTokens": 4096
    }
  ],
  "defaultModel": "inception-mercury",
  "environment": {
    "INCEPTION_API_KEY": "mmmmm"
  }
}
```

#### Option C: Via Environment Variable

1. Set environment variable:
   ```powershell
   $env:INCEPTION_API_KEY = "mmmmm"
   ```

2. Or in `.env` (already done):
   ```env
   INCEPTION_API_KEY=mmmmm
   ```

3. In Cline Settings, reference via `${INCEPTION_API_KEY}`

### Using Inception Mercury with Cline

1. **Open Cline Panel** in VS Code
   - Left sidebar → Cline icon
   - Or: Command Palette (`Ctrl+Shift+P`) → "Cline: Open"

2. **Select Model**
   - Top of Cline panel: Choose "inception-mercury" from dropdown

3. **Give it a task**
   ```
   Add comprehensive error logging to the metrics_collector.py file
   ```

4. **Cline will**
   - ✅ Read files from your workspace
   - ✅ Analyze code using Inception Mercury
   - ✅ Write/modify files
   - ✅ Run terminal commands
   - ✅ Show reasoning in chat

---

## Part 4: Continue.dev Integration (Alternative)

### What is Continue?
Similar to Cline, but focuses on **inline completions** + chat.

### Setup Continue for Inception Mercury

1. **Install Continue**
   - VS Code Extensions → Search "Continue"
   - Install official extension

2. **Configure**
   - Open `~/.continue/config.json` (macOS/Linux) or `%APPDATA%\Continue\config.json` (Windows)
   - Add Inception Mercury:

```json
{
  "models": [
    {
      "title": "Inception Mercury",
      "provider": "openai",
      "model": "mercury",
      "apiBase": "https://api.inceptionlabs.ai/v1",
      "apiKey": "mmmmm"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Inception Mercury",
    "provider": "openai",
    "model": "mercury",
    "apiBase": "https://api.inceptionlabs.ai/v1",
    "apiKey": "mmmmm"
  }
}
```

3. **Use in Continue**
   - Press `Ctrl+I` for inline completions
   - Or use Continue chat panel

---

## Part 5: Multi-Model Setup (Advanced)

You can configure **both** Copilot (built-in) **and** Inception Mercury (Cline/Continue) for different tasks:

### Recommended Configuration

```json
{
  "copilot": {
    "enabled": true,
    "purpose": "Fast inline completions, code suggestions"
  },
  "cline": {
    "defaultModel": "inception-mercury",
    "purpose": "Complex tasks, file generation, autonomy"
  },
  "models": [
    {
      "name": "Inception Mercury",
      "provider": "openai",
      "baseUrl": "https://api.inceptionlabs.ai/v1",
      "apiKey": "mmmmm",
      "model": "mercury",
      "maxTokens": 4096,
      "temperature": 0.7
    },
    {
      "name": "Local LLM (qwen3-4b)",
      "provider": "openai",
      "baseUrl": "http://localhost:1234/v1",
      "apiKey": "lm-studio",
      "model": "bullerwins/qwen3-4b-instruct-2507",
      "maxTokens": 4096
    }
  ]
}
```

### Use Cases

| Task | Best Model | Why |
|------|-----------|-----|
| Quick code suggestions | GitHub Copilot (built-in) | Fast, always available |
| Complex refactoring | Inception Mercury (Cline) | Better reasoning, file context |
| Privacy-sensitive code | Local LLM (qwen3-4b) | Runs locally, no API calls |
| Genealogy logic | Inception Mercury | Specialized for domain-specific tasks |

---

## Part 6: Testing Your Setup

### Test 1: Verify Inception Mercury API Works

```powershell
cd c:\Users\wayne\GitHub\Python\Projects\Ancestry
.venv\Scripts\python.exe ai_api_test.py --provider inception
```

Expected output:
```
✅ Inception Mercury returned a response
Endpoint: https://api.inceptionlabs.ai/v1
Model: mercury
Status: PASSED
```

### Test 2: Configure Cline (If Installed)

1. Open Cline panel in VS Code
2. Look for model selector dropdown
3. Select "inception-mercury"
4. Ask: "What models are configured?"

### Test 3: Use with Your Project

In Cline or Continue:
```
Analyze the core/metrics_collector.py file and suggest
performance optimizations for the MetricRegistry class.
```

---

## Part 7: Common Issues & Troubleshooting

### Issue 1: "API Key Invalid" Error
```
Error: 401 Unauthorized - Invalid API key
```

**Fix**:
- Verify `INCEPTION_API_KEY=mmmmm` is correct in `.env`
- Test with: `python ai_api_test.py --provider inception`
- Check if API key has changed on api.inceptionlabs.ai

### Issue 2: "Connection Refused" Error
```
Error: ConnectionError - Failed to connect to api.inceptionlabs.ai
```

**Fix**:
- Verify internet connection
- Check if Inception API is operational: `curl https://api.inceptionlabs.ai/v1/models`
- Verify base URL: `https://api.inceptionlabs.ai/v1` (note `/v1` suffix)

### Issue 3: Cline Not Using Inception Mercury
```
Cline is using default model instead of Inception Mercury
```

**Fix**:
- Restart VS Code
- Clear Cline cache: Command Palette → "Cline: Clear Cache"
- Verify model name in settings matches exactly: `inception-mercury`

### Issue 4: Timeout Errors
```
Error: Request timeout (30s+)
```

**Possible causes**:
- API is slow
- Model is overloaded
- Network latency to api.inceptionlabs.ai

**Fix**:
- Increase timeout in Cline settings
- Use Local LLM as fallback: `http://localhost:1234/v1`

---

## Part 8: Environment Variables Explained

| Variable | Purpose | Current Value |
|----------|---------|---|
| `INCEPTION_AI_BASE_URL` | API endpoint | `https://api.inceptionlabs.ai/v1` |
| `INCEPTION_AI_MODEL` | Model name | `mercury` |
| `INCEPTION_API_KEY` | Authentication | `mmmmm` (⚠️ Update with real key!) |

**⚠️ SECURITY WARNING**: The key `mmmmm` appears to be a placeholder. **Update it with your actual Inception Mercury API key** before production use.

---

## Part 9: Quick Reference

### Using Inception Mercury in Your Code

#### In Python (`ai_interface.py` pattern):
```python
from openai import OpenAI

client = OpenAI(
    api_key="mmmmm",
    base_url="https://api.inceptionlabs.ai/v1"
)

response = client.chat.completions.create(
    model="mercury",
    messages=[
        {"role": "user", "content": "Your prompt here"}
    ]
)
```

#### In VS Code Extensions:
- **Cline**: Command Palette → "Cline: New Task" → Select "inception-mercury"
- **Continue**: `Ctrl+I` → Choose Inception Mercury from model selector
- **Copilot**: Not directly supported (use Cline instead)

---

## Part 10: Next Steps

1. **Verify Configuration** ✅
   ```powershell
   python ai_api_test.py --provider inception
   ```

2. **Install Cline** (recommended for autonomous tasks)
   - VS Code Extensions → "Cline"

3. **Configure Cline for Inception Mercury** (see Part 3)

4. **Test with a Small Task**
   - Open Cline → Ask to "Add logging to metrics_integration.py"

5. **Update Real API Key**
   - Change `.env`: `INCEPTION_API_KEY=mmmmm` to your real key

---

## Summary

| Component | Status | Action |
|-----------|--------|--------|
| Project Configuration | ✅ DONE | Test with: `ai_api_test.py --provider inception` |
| GitHub Copilot | ⚠️ Limited | Use built-in for quick suggestions |
| Cline Setup | 📋 TODO | Install and configure (Part 3) |
| Continue.dev Setup | 📋 TODO | Optional alternative (Part 4) |
| API Key | ⚠️ PLACEHOLDER | Update `INCEPTION_API_KEY` with real key |

---

**Questions?** Check:
- Inception Mercury docs: https://api.inceptionlabs.ai/docs
- Cline GitHub: https://github.com/cline/cline
- Continue docs: https://docs.continue.dev
