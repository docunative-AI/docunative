# UV Development Guide for Contributors

This guide covers everything you need to know about working with UV in the DocuNative project.

---

## Table of Contents

1. [Setting Up a New Branch](#setting-up-a-new-branch)
2. [Installing Additional Packages](#installing-additional-packages)
3. [Adding Dependencies to the Project](#adding-dependencies-to-the-project)
4. [Updating Existing Dependencies](#updating-existing-dependencies)
5. [Common UV Commands](#common-uv-commands)
6. [Troubleshooting UV](#troubleshooting-uv)
7. [Best Practices](#best-practices)
8. [Example: Complete Workflow](#example-complete-workflow)

---

## Setting Up a New Branch

When starting work on a new feature or bug fix:

```bash
# 1. Create and switch to your branch
git checkout -b feature/your-feature-name

# 2. Sync dependencies (creates .venv if needed)
uv sync

# 3. Activate the environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

**What `uv sync` does:**
- Reads `pyproject.toml` and `uv.lock`
- Installs exact versions specified in the lock file
- Ensures everyone has identical dependencies

---

## Installing Additional Packages

### Option 1: Using UV (Recommended)

For temporary testing or experimentation:

```bash
# Install a package temporarily (not added to pyproject.toml)
uv pip install package-name

# Example: Install ipython for debugging
uv pip install ipython
```

**Use cases:**
- Quick testing
- Debugging sessions
- Exploratory data analysis
- One-off scripts

### Option 2: Using pip (Fallback)

If UV has compatibility issues with a specific package:

```bash
# First, install pip into the UV environment
uv pip install pip

# Then use pip normally
pip install package-name

# Example:
pip install matplotlib
```

**Note:** This is rarely needed. UV handles 99% of packages perfectly.

---

## Adding Dependencies to the Project

If you need a package **permanently** for production code:

### Step 1: Edit `pyproject.toml`

Add your package to the `dependencies` array:

```toml
dependencies = [
    # ... existing deps ...
    "new-package>=1.0,<2.0",  # Add with version constraints
]
```

**Version constraint guidelines:**
- `>=1.0,<2.0` - Allow minor updates, block major breaking changes
- `==1.5.3` - Pin exact version (only if absolutely necessary)
- `>=1.0` - No upper bound (risky, not recommended)

### Step 2: Update the lock file

```bash
uv lock
```

This resolves all dependencies and regenerates `uv.lock`.

### Step 3: Install the new dependencies

```bash
uv sync
```

### Step 4: Commit both files

```bash
git add pyproject.toml uv.lock
git commit -m "Add new-package dependency for feature X"
```

**Important:** Always commit both files together!

### Complete Example

```bash
# You need pandas for data processing
vim pyproject.toml  # Add "pandas>=2.0,<3.0" to dependencies

uv lock             # Resolve dependencies
uv sync             # Install

# Test the import
python -c "import pandas; print(pandas.__version__)"

# Commit
git add pyproject.toml uv.lock
git commit -m "Add pandas for data processing pipeline"
```

---

## Updating Existing Dependencies

### Update All Dependencies

Upgrade all packages to their latest compatible versions:

```bash
# Update lock file with latest versions
uv lock --upgrade

# Install the updated versions
uv sync
```

**Use case:** Monthly maintenance updates

### Update Specific Package

Upgrade only one package:

```bash
# Upgrade only torch to latest compatible version
uv lock --upgrade-package torch

# Install the update
uv sync
```

**Use case:** Security patch for specific package, or testing new feature in one library

### Check What Would Be Updated

```bash
# See what uv lock --upgrade would change (dry run)
uv lock --upgrade --dry-run
```

---

## Common UV Commands

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `uv sync` | Install dependencies from `uv.lock` | After pulling changes, switching branches |
| `uv lock` | Regenerate `uv.lock` from `pyproject.toml` | After editing dependencies |
| `uv lock --upgrade` | Update all packages to latest versions | Monthly updates |
| `uv lock --upgrade-package pkg` | Update specific package | Security patches, feature updates |
| `uv pip install pkg` | Install package temporarily | Testing, debugging |
| `uv pip list` | List installed packages | Checking versions |
| `uv pip uninstall pkg` | Remove package | Cleanup |
| `uv run python script.py` | Run script without activating venv | Quick scripts, CI/CD |
| `uv venv` | Create virtual environment | Manual setup (rarely needed) |
| `uv venv --python 3.11` | Create venv with specific Python | Testing different versions |

---

## Troubleshooting UV

### Error: "Package not found"

**Symptom:** UV can't find a package on PyPI

**Solution:**
```bash
# Clear UV cache
rm -rf ~/.cache/uv

# Retry
uv sync
```

### Error: "Conflicting dependencies"

**Symptom:** UV shows dependency conflict error

**Example:**
```
Because package-a>=2.0 requires package-b<1.0
and package-c>=1.0 requires package-b>=1.5,
we can conclude that package-a>=2.0 and package-c>=1.0 are incompatible.
```

**Solution:** Adjust version constraints in `pyproject.toml` to find compatible ranges:

```toml
# Before (conflicting)
dependencies = [
    "package-a>=2.0",
    "package-c>=1.0",
]

# After (compatible)
dependencies = [
    "package-a>=1.5,<2.0",  # Use older version
    "package-c>=1.0",
]
```

### Error: "uv.lock is out of sync"

**Symptom:** Lock file doesn't match `pyproject.toml`

**Solution:**
```bash
# Regenerate lock file
uv lock

# Then sync
uv sync
```

### Error: "No binary distribution available"

**Symptom:** Package requires compilation but no pre-built wheel exists

**Solution:**
```bash
# Option 1: Install build tools
# macOS
brew install cmake

# Linux
sudo apt install build-essential

# Then retry
uv sync

# Option 2: Use pip fallback
uv pip install pip
pip install problematic-package
```

### Error: "Python version mismatch"

**Symptom:** UV complains about Python version

**Solution:**
```bash
# Check required version in pyproject.toml
# requires-python = ">=3.11,<3.12"

# Ensure Python 3.11 is available
python3.11 --version

# Recreate venv with correct version
rm -rf .venv
uv venv --python 3.11
uv sync
```

---

## Best Practices

### DO ✅

1. **Always run `uv sync` after pulling changes**
   ```bash
   git pull origin main
   uv sync  # Update dependencies
   ```

2. **Commit both `pyproject.toml` and `uv.lock` together**
   ```bash
   git add pyproject.toml uv.lock
   git commit -m "Update dependencies"
   ```

3. **Use version constraints for stability**
   ```toml
   dependencies = [
       "torch>=2.0,<3.0",      # Good: Allows patches, blocks breaking changes
       "gradio>=6.0,<7.0",     # Good: Semantic versioning
   ]
   ```

4. **Test changes with clean environment**
   ```bash
   rm -rf .venv
   make install
   make test
   ```

### DON'T ❌

1. **Don't manually edit `uv.lock`**
   - It's auto-generated
   - Changes will be overwritten by `uv lock`
   - Use `uv lock` to regenerate it

2. **Don't use `pip install -r requirements.txt`**
   - We don't use `requirements.txt` anymore
   - Use `uv sync` instead

3. **Don't commit `.venv/` directory**
   - It's in `.gitignore`
   - Everyone creates their own with `uv sync`
   - Contains machine-specific paths

4. **Don't use `pip freeze > requirements.txt`**
   - Use `pyproject.toml` for dependencies
   - Use `uv.lock` for exact versions
   - `requirements.txt` is deprecated in this project

---

## Example: Complete Workflow

Here's a complete workflow for adding a new feature that requires a new dependency:

```bash
# 1. Create feature branch
git checkout -b feature/add-pdf-compression

# 2. Install existing dependencies
uv sync

# 3. Test new package temporarily
uv pip install PyPDF2
python -c "import PyPDF2; print('Works!')"

# 4. Write your code
vim pipeline/compress.py
# (implement your feature)

# 5. Add package permanently to pyproject.toml
vim pyproject.toml
# Add: "PyPDF2>=3.0,<4.0" to dependencies

# 6. Update lock file
uv lock

# 7. Reinstall to verify everything works
uv sync

# 8. Test your feature
python pipeline/compress.py
# (verify it works)

# 9. Test with clean environment
rm -rf .venv
make install
python pipeline/compress.py
# (verify it still works)

# 10. Commit changes
git add pyproject.toml uv.lock pipeline/compress.py
git commit -m "Add PDF compression feature with PyPDF2"

# 11. Push and create PR
git push origin feature/add-pdf-compression
# (Create PR on GitHub)
```

---

## Quick Reference Card

```bash
# Daily Operations
uv sync                          # Install/update deps after pulling
uv run python script.py          # Run script without activating venv

# Adding Dependencies
vim pyproject.toml               # Add package to dependencies
uv lock                          # Resolve and update lock file
uv sync                          # Install new packages

# Testing Packages
uv pip install package-name      # Install temporarily
python -c "import package"       # Test it

# Updating
uv lock --upgrade                # Update all packages
uv lock --upgrade-package torch  # Update specific package
uv sync                          # Install updates

# Troubleshooting
rm -rf ~/.cache/uv               # Clear cache
uv lock                          # Regenerate lock file
uv sync                          # Reinstall
```

---

## Need Help?

- **UV Documentation:** https://docs.astral.sh/uv/
- **Project Issue Tracker:** GitHub Issues
- **Quick Questions:** Ask in project Discord/Slack

---

**Last Updated:** March 16, 2026
