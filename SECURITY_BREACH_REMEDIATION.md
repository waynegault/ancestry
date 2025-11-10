# 🚨 SECURITY BREACH REMEDIATION CHECKLIST
**Date**: November 10, 2025
**Incident**: `.env` file with credentials committed to git (Nov 7-10, 2025)
**Status**: Credentials were exposed on GitHub for ~3 days (Nov 7-10, 2025)
**Action Taken**: Git history completely rewritten to remove .env from all 2,399 commits
**Next Step**: FORCE PUSH to GitHub + Regenerate ALL API keys immediately

---

## ✅ COMPLETED ACTIONS

- [x] Remove `.env` from git tracking (commit d7ce75c → removed in history rewrite)
- [x] Sanitize `.env.example` file (commit dbf007f → commit d0a7daa after rewrite)
- [x] Verify `.gitignore` properly excludes `.env` files
- [x] ~~Confirm credentials NOT pushed to GitHub remote~~ **WERE PUSHED -History rewritten**
- [x] Install git-filter-repo tool
- [x] Create repository backup (Ancestry-backup-YYYYMMDD-HHMMSS)
- [x] Rewrite git history - remove .env from ALL commits
- [x] Install pre-commit hooks (8 security checks active)
- [x] Create security documentation (this file + GIT_HISTORY_CLEANUP_SUMMARY.md)

---

## 🔑 CRITICAL: API KEYS TO REGENERATE

### 1. DeepSeek AI API Key ⚠️ HIGH PRIORITY
- **Exposed Key**: `sk-b555b5de579548da9e1b25cdf646ecf8`
- **Action Required**:
  1. Go to https://platform.deepseek.com/api_keys
  2. Delete exposed key `sk-b555b5de579548da9e1b25cdf646ecf8`
  3. Generate new API key
  4. Update `.env` file with new key: `DEEPSEEK_API_KEY=sk-new-key-here`
  5. Test: `python -c "from ai_interface import call_ai; print(call_ai('intent_classification', {'message': 'test'}))"`
- **Status**: [x] ✅ Complete

---

### 2. Google Gemini API Key ⚠️ HIGH PRIORITY
- **Exposed Key**: `AIzaSyCZoT6x_5edlOf8DeVA6gy16wdVw8SJqCg`
- **Action Required**:
  1. Go to https://aistudio.google.com/apikey
  2. Delete/disable exposed key `AIzaSyCZoT6x_5edlOf8DeVA6gy16wdVw8SJqCg`
  3. Create new API key
  4. Update `.env` file with new key: `GOOGLE_API_KEY=AIza...new-key`
  5. Test: Change `AI_PROVIDER="gemini"` in .env and run AI test
- **Status**: [x] ✅ Complete

---

### 3. Microsoft Graph API Credentials ⚠️ MEDIUM PRIORITY
- **Exposed Client ID**: `ab07d95c-bb7f-4121-b365-0784fa7831e8`
- **Exposed Tenant ID**: `fa6506d2-27a3-47fb-8051-03562f4b7f49`
- **Action Required**:
  1. Go to https://portal.azure.com → Azure Active Directory → App Registrations
  2. Find app with Client ID `ab07d95c-bb7f-4121-b365-0784fa7831e8`
  3. Go to "Certificates & secrets" → Delete existing client secret
  4. Generate new client secret
  5. Update `.env` file with new secret: `MS_CLIENT_SECRET=new-secret-here`
  6. Note: Client ID and Tenant ID don't need rotation (not secret), but consider creating new app registration for best security
  7. Test: `python -c "from ms_graph_utils import get_todo_lists; print(get_todo_lists())"`
- **Status**: [ ] Not Started / [ ] In Progress / [ ] ✅ Complete

---

### 4. Ancestry.com Password ⚠️ HIGH PRIORITY
- **Exposed Email**: `waynegault@msn.com`
- **Exposed Password**: `WgAuLT69#!!?`
- **Also Exposed** (commented line): `francesmchardy@gmail.com` / `catherine!14`
- **Action Required**:
  1. Go to https://www.ancestry.com/account/security
  2. Log in with `waynegault@msn.com`
  3. Change password to new strong password
  4. Update `.env` file: `ANCESTRY_PASSWORD=new-password-here`
  5. **ALSO**: If `francesmchardy@gmail.com` is a valid account, change that password too
  6. Test: Run `python main.py` → Option 5: Check Login Status
- **Status**: [x] ✅ Complete

---

## 🧹 GIT HISTORY CLEANUP - ✅ COMPLETED

### History Rewrite Summary
**Status**: ✅ Complete - Ready for force push

```powershell
# ✅ COMPLETED - git-filter-repo installed
pip install git-filter-repo

# ✅ COMPLETED - Backup created
# Location: C:\Users\wayne\GitHub\Python\Projects\Ancestry-backup-YYYYMMDD-HHMMSS

# ✅ COMPLETED - .env removed from ALL commits
python -m git_filter_repo --path .env --invert-paths --force
# Result: Parsed 2,399 commits, removed .env, execution time 6.56 seconds

# ✅ VERIFIED - .env is gone from history
git log --all --oneline -- .env
# Returns empty (no commits contain .env)

# ⚠️ PENDING - Force push to update remote
git push origin main --force
```

**Verification**: `git log --all --oneline -- .env` returns empty ✅

---

## 📋 VERIFICATION CHECKLIST

After regenerating all keys:

- [ ] Test DeepSeek AI: `python ai_api_test.py` (if exists) or test prompt
- [ ] Test Google Gemini AI: Change provider in .env and test
- [ ] Test MS Graph: `python -c "from ms_graph_utils import get_todo_lists; print(get_todo_lists())"`
- [ ] Test Ancestry login: `python main.py` → Option 5
- [ ] Run full test suite: `python run_all_tests.py`
- [ ] Verify `.env` NOT in git: `git ls-files | Select-String "\.env"`
- [ ] Verify `.env` in .gitignore: `Get-Content .gitignore | Select-String "\.env"`
- [ ] Check git status clean: `git status`

---

## 🛡️ PREVENTION MEASURES

### Implement Pre-Commit Hooks
```powershell
# Install pre-commit framework
pip install pre-commit

# Create .pre-commit-config.yaml
@"
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-added-large-files
      - id: detect-private-key
      - id: check-yaml

  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
        exclude: package.lock.json
"@ | Out-File -Encoding UTF8 .pre-commit-config.yaml

# Install hooks
pre-commit install

# Scan existing files
pre-commit run --all-files
```

- **Status**: [ ] Not Started / [ ] In Progress / [ ] ✅ Complete

### Git Aliases for Safety
```powershell
# Never accidentally force-add ignored files
git config --global alias.safe-add "add --ignore-removal"

# Always show what would be added before adding
git config --global alias.add-dry "add --dry-run"
```

- **Status**: [ ] Not Started / [ ] In Progress / [ ] ✅ Complete

---

## 📊 INCIDENT TIMELINE

| Date/Time | Event | Commit |
|-----------|-------|--------|
| Nov 7, 2025 21:55 | `.env` force-added to git with 269 lines | b4a2658 |
| Nov 7-10, 2025 | Multiple commits with `.env` tracked | 6 commits |
| Nov 10, 2025 10:13 | `.env` removed from tracking | d7ce75c (pre-rewrite) |
| Nov 10, 2025 10:15 | `.env.example` sanitized | dbf007f → d0a7daa (post-rewrite) |
| Nov 10, 2025 10:30 | Git history rewritten - .env purged | 2,399 commits processed |
| Nov 10, 2025 10:35 | Pre-commit hooks installed | 8 security checks active |
| Nov 10, 2025 | **✅ COMPLETED - awaiting force push** | Ready for deployment |

---

## 🎯 PRIORITY ORDER

1. **IMMEDIATE** (Next 1 hour):
   - [ ] Regenerate DeepSeek API key
   - [ ] Regenerate Google Gemini API key
   - [ ] Change Ancestry.com password

2. **TODAY**:
   - [ ] Regenerate MS Graph client secret
   - [ ] Rewrite git history (Option A above)
   - [ ] Verify all tests pass with new credentials

3. **THIS WEEK**:
   - [ ] Implement pre-commit hooks
   - [ ] Set up git aliases for safety
   - [ ] Document incident in project notes
   - [ ] Review .gitignore completeness

---

## ❓ QUESTIONS TO CONSIDER

1. **Was `francesmchardy@gmail.com` account password also exposed?**
   - If yes, change that password too

2. **Do you have any other git remotes besides GitHub?**
   - Check: `git remote -v`
   - If yes, ensure .env not pushed there

3. **Are there any other sensitive files in the repository?**
   - Check: `git ls-files | Select-String "credentials|secrets|keys|tokens"`

4. **Should you create a new Azure app registration?**
   - Client ID/Tenant ID aren't secrets, but fresh start might be cleaner

---

## 📞 SUPPORT RESOURCES

- **DeepSeek Support**: https://platform.deepseek.com/docs
- **Google AI Support**: https://ai.google.dev/gemini-api/docs
- **Microsoft Graph**: https://learn.microsoft.com/en-us/graph/
- **Ancestry Developer**: (No public API - web scraping, no key rotation needed for browser automation)
- **Git Filter Repo Docs**: https://github.com/newren/git-filter-repo

---

**Last Updated**: November 10, 2025
**Remediation Owner**: Wayne Gault
**Next Review**: After all keys regenerated and history cleaned
