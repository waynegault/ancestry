# Quick Reference: GitHub Merge Button

A quick reference guide for using GitHub's merge button when reviewing and merging pull requests.

## TL;DR - Quick Steps

1. ✅ Ensure all CI checks pass (green checkmarks)
2. ✅ Get required approvals from reviewers
3. ✅ Resolve any merge conflicts
4. 🔽 Click dropdown next to "Merge pull request"
5. 🎯 Select merge strategy (usually "Squash and merge")
6. ✏️ Edit commit message if needed
7. ✔️ Click "Confirm squash and merge"
8. 🗑️ Delete branch (optional but recommended)

## Merge Strategy Cheat Sheet

| Strategy | Use When | Result |
|----------|----------|--------|
| **Merge commit** | Preserve full history | All commits + merge commit appear in main |
| **Squash and merge** ⭐ | Clean up messy commits | Single commit in main (recommended) |
| **Rebase and merge** | Linear history wanted | Commits replayed on main, no merge commit |

⭐ **Recommended default**: Squash and merge for cleanest history

## Visual Guide

### Finding the Merge Button

```
┌─────────────────────────────────────────────────────┐
│ Pull Request #42: Add awesome feature              │
├─────────────────────────────────────────────────────┤
│                                                     │
│ [Conversation] [Commits] [Files changed]           │
│                                                     │
│ ... PR discussion ...                              │
│                                                     │
│ ✅ All checks have passed                          │
│                                                     │
│ ┌────────────────────────────┐                     │
│ │ [▼ Merge pull request]     │ ← Click here!      │
│ └────────────────────────────┘                     │
│    └── Click dropdown for options                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Merge Options Dropdown

```
┌──────────────────────────────┐
│ Create a merge commit        │ ← Keeps all commits
├──────────────────────────────┤
│ Squash and merge             │ ← ⭐ Recommended
├──────────────────────────────┤
│ Rebase and merge             │ ← For clean commits
└──────────────────────────────┘
```

## Common Scenarios

### Scenario 1: Simple Feature (Recommended Flow)

```bash
# Your branch has 5 commits with review feedback fixes
Your PR: Fix typo → Add feature → Address review → Fix lint → Update tests

# Use: Squash and merge
Result on main: "Add awesome feature (#42)" ← Clean!
```

### Scenario 2: Complex Feature with Logical Steps

```bash
# Your branch has well-organized commits
Your PR: Database schema → API endpoints → UI components → Tests

# Use: Rebase and merge (or Merge commit if you want merge point preserved)
Result on main: Each commit appears individually with clear progression
```

### Scenario 3: Hotfix or Single Commit

```bash
# Your branch has 1 clean commit
Your PR: Fix critical security issue in authentication

# Use: Any strategy works, squash is still good for consistency
Result on main: Single clean commit
```

## Troubleshooting Decision Tree

```
Can't see merge button?
├─ Not a collaborator → Fork and create PR from your fork
├─ Wrong branch → Ensure PR targets correct base branch
└─ Missing permissions → Contact repository maintainer

Merge button disabled?
├─ CI checks failing? → Click check, view logs, fix issues
├─ Merge conflicts? → Update branch with main and resolve
├─ Needs approval? → Wait for reviewer or request review
└─ Branch out of date? → Update with: git pull origin main

Merge conflicts exist?
├─ Simple conflicts → Click "Resolve conflicts" on GitHub
├─ Complex conflicts → Resolve locally:
│   git checkout your-branch
│   git merge origin/main
│   # Fix conflicts
│   git push origin your-branch
└─ Not sure? → Ask for help in PR comments
```

## Best Practices

### ✅ DO

- **Squash and merge** for most PRs (keeps main clean)
- **Edit commit message** before squashing (make it descriptive)
- **Delete branch** after merging (keeps repo tidy)
- **Ensure tests pass** before merging
- **Get code review** for significant changes
- **Update branch** if main has moved ahead

### ❌ DON'T

- **Don't merge failing checks** (fix them first!)
- **Don't merge with unresolved conflicts**
- **Don't merge without review** (unless trivial)
- **Don't leave vague commit messages** when squashing
- **Don't force push** after someone has reviewed

## Commit Message Template (for Squash Merge)

```
Brief description of change (#PR-number)

Detailed explanation:
- What was changed
- Why it was needed
- Any breaking changes or migration notes

Fixes #issue-number (if applicable)
```

Example:
```
Add DNA match deduplication cache (#42)

Implemented caching layer for API calls to reduce redundant requests:
- Added APICallCache with 5-minute TTL
- Integrated into action6_gather workflow
- Achieved 14-20% cache hit rate in testing

Reduces processing time by 10-20 minutes for large batches.

Fixes #38
```

## Repository Configuration (For Maintainers)

To enable merge button options:

1. Go to **Settings** → **General**
2. Scroll to **Pull Requests** section
3. Check desired merge options:
   - ☑️ Allow merge commits
   - ☑️ Allow squash merging
   - ☑️ Allow rebase merging
4. Set **Default merge strategy** to "Squash and merge"
5. Enable **Automatically delete head branches**

### Recommended Settings

```yaml
Merge button:
  ✅ Allow squash merging (default)
  ✅ Allow rebase merging
  ✅ Allow merge commits
  ✅ Auto-delete branches
  ✅ Allow auto-merge
  ✅ Require PR for merge

Branch protection (main):
  ✅ Require pull request reviews (1 approval)
  ✅ Require status checks to pass
  ✅ Require branches to be up to date
  ✅ Include administrators
```

## FAQ

### Q: Which merge strategy should I use?

**A:** For this repository, **squash and merge** is recommended for most PRs. It keeps the main branch history clean while preserving full history in the PR.

### Q: What if I have important individual commits?

**A:** Use **rebase and merge** if each commit is well-crafted and meaningful. Otherwise, squash them and document the changes in the squash commit message.

### Q: Can I merge my own PR?

**A:** It depends on repository settings. Best practice is to wait for code review, but for minor changes (typos, docs), you may be allowed to merge after checks pass.

### Q: What happens to my branch after merge?

**A:** It remains in the repository unless deleted. Click "Delete branch" button after merge to keep things tidy. If auto-delete is enabled, it happens automatically.

### Q: How do I update my branch before merging?

**A:**
```bash
git checkout your-branch
git pull origin main
git push origin your-branch
```

Or click "Update branch" button on GitHub if available.

### Q: What if merge conflicts are too complex?

**A:** Ask for help! Comment on the PR mentioning the conflicts, and a maintainer can assist or merge from their local environment.

## Resources

- [GitHub Docs: Merging Pull Requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/merging-a-pull-request)
- [About Merge Methods](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/about-merge-methods-on-github)
- [Managing Auto-Merge](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/automatically-merging-a-pull-request)

---

**Need more help?** See [CONTRIBUTING.md](../CONTRIBUTING.md) for comprehensive contribution guidelines.
