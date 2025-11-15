# Contributing to Ancestry Research Automation

Thank you for your interest in contributing! This guide will help you understand how to contribute code changes and use GitHub's merge functionality.

## Table of Contents
- [Getting Started](#getting-started)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Using GitHub's Merge Button](#using-githubs-merge-button)
- [Code Quality Standards](#code-quality-standards)
- [Testing](#testing)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ancestry.git
   cd ancestry
   ```
3. **Set up the development environment**:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your configuration
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Making Changes

1. Make your changes in your feature branch
2. **Test your changes** thoroughly:
   ```bash
   # Run all tests
   python run_all_tests.py
   
   # Or run specific test modules
   python -m action6_gather
   ```
3. **Lint your code**:
   ```bash
   ruff check --fix .
   ```
4. **Commit your changes** with clear, descriptive commit messages:
   ```bash
   git add .
   git commit -m "Add feature: brief description of change"
   ```

## Pull Request Process

### Creating a Pull Request

1. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. **Navigate to the original repository** on GitHub
3. **Click "New Pull Request"**
4. **Select your fork and branch** as the source
5. **Fill out the PR template**:
   - Provide a clear title
   - Describe what changes you made and why
   - Reference any related issues
   - Include screenshots for UI changes
   - List any breaking changes

### PR Review Process

1. **Automated checks** will run:
   - Quality regression gate (checks AI extraction quality)
   - Code linting and formatting
2. **Maintainers will review** your PR:
   - They may request changes
   - Address feedback by pushing new commits to your branch
3. **Once approved**, your PR is ready to merge!

## Using GitHub's Merge Button

### Overview

GitHub provides several merge strategies when merging a pull request. Here's how to use the merge button and understand your options.

### Accessing the Merge Button

1. **Navigate to your Pull Request** on GitHub
2. **Scroll to the bottom** of the PR conversation tab
3. **Look for the green "Merge pull request" button**
   - If the button is grayed out, check for:
     - Failing CI/CD checks
     - Unresolved review comments
     - Merge conflicts
     - Branch protection rules requiring approvals

### Merge Strategies

Click the dropdown arrow next to the merge button to see three options:

#### 1. **Create a merge commit** (Default)
```
Preserves the complete history of commits from your branch
```
- **When to use**: 
  - When you want to preserve the full development history
  - For feature branches with multiple logical commits
  - When branch history provides valuable context
- **Result**: All commits from your branch appear in main branch history
- **Example**:
  ```
  * Merge pull request #42 from feature/awesome-feature
  |\
  | * Add comprehensive tests
  | * Implement feature logic
  | * Update documentation
  |/
  * Previous commit on main
  ```

#### 2. **Squash and merge**
```
Combines all commits into a single commit before merging
```
- **When to use**:
  - For cleanup commits (fixing typos, addressing review comments)
  - When you have many small commits that aren't individually meaningful
  - To keep main branch history clean and linear
  - **Recommended for most PRs in this project**
- **Result**: One new commit appears on main with all changes combined
- **Example**:
  ```
  * Add awesome feature (#42)
  * Previous commit on main
  ```
- **Best practice**: Edit the commit message to be descriptive before squashing

#### 3. **Rebase and merge**
```
Replays your commits on top of the base branch
```
- **When to use**:
  - When you want individual commits preserved but without merge commit
  - For small, focused PRs with clean commit history
  - When each commit is meaningful and well-crafted
- **Result**: Your commits appear in main branch as if developed directly there
- **Example**:
  ```
  * Update documentation
  * Implement feature logic
  * Add comprehensive tests
  * Previous commit on main
  ```

### Step-by-Step: Merging Your First PR

1. **Ensure all checks pass**:
   - Green checkmarks next to "All checks have passed"
   - If checks fail, review the logs and fix issues

2. **Get approval** (if required):
   - Wait for maintainer review
   - Address any requested changes

3. **Resolve conflicts** (if any):
   - Click "Resolve conflicts" button
   - Edit files to resolve conflicts
   - Mark as resolved and commit

4. **Choose merge strategy**:
   - Click dropdown arrow next to "Merge pull request"
   - Select your preferred strategy (usually "Squash and merge")

5. **Edit commit message** (optional):
   - Click "Squash and merge" button
   - Edit the commit message and description
   - Include PR number and relevant details

6. **Confirm merge**:
   - Click "Confirm squash and merge"
   - Your changes are now in the main branch!

7. **Delete branch** (optional):
   - Click "Delete branch" button after merge
   - Keeps repository tidy

### Repository Settings (For Maintainers)

To configure merge button options:

1. **Navigate to repository Settings**
2. **Go to "General" → "Pull Requests"**
3. **Configure merge button options**:
   - ☑️ Allow merge commits
   - ☑️ Allow squash merging (recommended default)
   - ☑️ Allow rebase merging
   - ☑️ Automatically delete head branches

4. **Set default merge strategy**:
   - Choose "Allow squash merging" for cleaner history

5. **Configure branch protection** (Settings → Branches):
   - Require pull request reviews
   - Require status checks to pass
   - Require branches to be up to date
   - Include administrators in restrictions

### Troubleshooting Common Issues

#### "Merge button is disabled"

**Possible causes:**
- ❌ Required CI checks are failing
  - **Solution**: Click on failed check, review logs, fix issues
- ❌ Merge conflicts exist
  - **Solution**: Click "Resolve conflicts" or update your branch:
    ```bash
    git checkout main
    git pull origin main
    git checkout your-branch
    git merge main
    # Resolve conflicts
    git push origin your-branch
    ```
- ❌ Required reviews not received
  - **Solution**: Wait for maintainer approval or request review
- ❌ Branch is out of date
  - **Solution**: Update your branch with latest main:
    ```bash
    git pull origin main
    git push origin your-branch
    ```

#### "Merge conflicts"

**To resolve:**
1. **Update your branch** with main:
   ```bash
   git fetch origin
   git checkout your-branch
   git merge origin/main
   ```
2. **Fix conflicts** in your editor:
   - Look for `<<<<<<<`, `=======`, `>>>>>>>` markers
   - Choose which changes to keep
   - Remove conflict markers
3. **Test your changes** after resolving
4. **Commit and push**:
   ```bash
   git add .
   git commit -m "Resolve merge conflicts with main"
   git push origin your-branch
   ```

#### "This branch is out of date"

**Solution:**
```bash
# Option 1: Merge main into your branch
git checkout your-branch
git merge origin/main
git push origin your-branch

# Option 2: Rebase your branch (cleaner history)
git checkout your-branch
git rebase origin/main
git push --force-with-lease origin your-branch
```

## Code Quality Standards

This project enforces quality standards through:

1. **Ruff linting**: Run `ruff check --fix .` before committing
2. **Type hints**: Use Pyright for type checking
3. **Quality regression gate**: AI extraction quality must not degrade
4. **Comprehensive testing**: 58 test modules must pass

### Pre-commit Checks

Consider installing pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

This will automatically:
- Format code with ruff
- Check for common issues
- Prevent commits with linting errors

## Testing

### Running Tests

```bash
# Run all tests sequentially
python run_all_tests.py

# Run tests in parallel (faster)
python run_all_tests.py --fast

# Run with performance analysis
python run_all_tests.py --analyze-logs

# Run specific module tests
python -m action6_gather
python -m action7_inbox
```

### Writing Tests

Follow the established test framework pattern:
```python
from test_framework import TestSuite, create_standard_test_runner

def module_tests() -> bool:
    """Test function for this module"""
    suite = TestSuite("Module Name", "module_file.py")
    
    # Add tests
    suite.add_test(lambda: assertion, "Test description")
    
    return suite.run_tests()

run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    import sys
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
```

## Questions?

- **Found a bug?** Open an issue with reproduction steps
- **Have a feature idea?** Open an issue to discuss before implementing
- **Need help?** Comment on your PR or open a discussion

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

**Thank you for contributing to make genealogical research automation better!** 🎉
