# Makefile Update: Automatic Model Verification

**Date:** March 16, 2026  
**Feature:** Automatic model download during installation

---

## What Changed

The Makefile now includes **automatic model verification and download** during the installation process.

### New Behavior

```bash
make install
```

Now performs **4 steps** instead of 2:

1. ✅ Create Python 3.11 virtual environment
2. ✅ Install dependencies via UV
3. ✅ **Check for model files** (NEW)
4. ✅ **Download models if missing** (NEW)

---

## New Make Targets

### 1. `check-models` (internal)

Automatically called by `make install`. Checks if both model files exist:
- `models/tiny-aya-global-q4_k_m.gguf` (~2.1 GB)
- `models/tiny-aya-earth-q4_k_m.gguf` (~2.1 GB)

**If missing:** Downloads automatically via `uv run python models/pull_models.py`  
**If present:** Skips download, reports success

**Output examples:**

```bash
# Models missing
🔍 Checking for model files...
⚠️  Model files not found. Downloading (~4.2 GB, this may take 5-10 minutes)...
Downloading tiny-aya-global-q4_k_m.gguf...
Successfully downloaded to: models/tiny-aya-global-q4_k_m.gguf
...
✅ Models downloaded successfully.

# Models already present
🔍 Checking for model files...
✅ Model files already present.
```

### 2. `make models` (standalone)

Explicit command to download or re-download models:

```bash
make models
```

**Use cases:**
- Initial download failed during `make install`
- Models were deleted or corrupted
- Want to force re-download

---

## Updated Makefile

```makefile
.PHONY: install test server-global server-earth demo check-models models

install:
	uv venv --python 3.11
	uv sync
	@$(MAKE) check-models
	@echo "✅ Environment ready. Run 'source .venv/bin/activate' to activate."

check-models:
	@echo "🔍 Checking for model files..."
	@if [ ! -f models/tiny-aya-global-q4_k_m.gguf ] || [ ! -f models/tiny-aya-earth-q4_k_m.gguf ]; then \
		echo "⚠️  Model files not found. Downloading (~4.2 GB, this may take 5-10 minutes)..."; \
		uv run python models/pull_models.py; \
		if [ $$? -eq 0 ]; then \
			echo "✅ Models downloaded successfully."; \
		else \
			echo "❌ Model download failed. Please try manually:"; \
			echo "   source .venv/bin/activate && python models/pull_models.py"; \
			exit 1; \
		fi \
	else \
		echo "✅ Model files already present."; \
	fi

models:
	@echo "📥 Downloading model files (~4.2 GB total)..."
	@echo "This may take 5-10 minutes depending on your internet speed."
	uv run python models/pull_models.py

test:
	uv run pytest -q

server-global:
	bash models/start_server.sh global

server-earth:
	bash models/start_server.sh earth

demo:
	uv run python ui/app.py
```

---

## Benefits

### 1. **One-Command Setup**

**Before:**
```bash
make install                    # Step 1: Install dependencies
python models/pull_models.py    # Step 2: Download models (manual)
make server-global              # Step 3: Start server
```

**After:**
```bash
make install        # Does steps 1 + 2 automatically
make server-global  # Works immediately
```

### 2. **Idempotent**

Running `make install` multiple times is safe:
- First run: Downloads models
- Subsequent runs: Skips download (models present)

### 3. **Clear Feedback**

Uses emoji indicators for visual clarity:
- 🔍 Checking...
- ⚠️ Models missing
- ✅ Success
- ❌ Failure

### 4. **Graceful Failure Handling**

If automatic download fails:
```bash
❌ Model download failed. Please try manually:
   source .venv/bin/activate && python models/pull_models.py
```

Provides clear recovery instructions.

### 5. **Flexible Recovery**

Multiple ways to download models:
- `make install` - Automatic during setup
- `make models` - Standalone download command
- Manual script - `python models/pull_models.py`
- huggingface-cli - Direct download
- Browser download - Manual file download

---

## Timing Expectations

### First-Time Install (Models Missing)

```
make install
├─ uv venv --python 3.11        ~2 seconds
├─ uv sync                       ~30 seconds (fresh) or ~1 second (cached)
└─ check-models + download       ~5-10 minutes (depends on internet)
Total: ~6-11 minutes
```

### Subsequent Install (Models Present)

```
make install
├─ uv venv --python 3.11        ~1 second (already exists)
├─ uv sync                       ~1 second (cached)
└─ check-models                  ~1 second (files present, no download)
Total: ~3 seconds
```

---

## Error Handling

### Scenario 1: Model Download Fails

```bash
make install
# ... (deps install successfully) ...
🔍 Checking for model files...
⚠️  Model files not found. Downloading (~4.2 GB, this may take 5-10 minutes)...
❌ Model download failed. Please try manually:
   source .venv/bin/activate && python models/pull_models.py
make: *** [check-models] Error 1
```

**Recovery:**
```bash
# Try standalone download
make models

# Or manual
source .venv/bin/activate
python models/pull_models.py
```

### Scenario 2: Only One Model Downloaded

The check looks for **both** models. If only one is present, it attempts to re-download both.

**Example:**
```bash
$ ls models/
tiny-aya-global-q4_k_m.gguf  # Only global present

$ make install
# ... 
⚠️  Model files not found. Downloading (~4.2 GB, this may take 5-10 minutes)...
# Will download both (even though global exists)
```

**Optimization idea:** Could be improved to download only missing models in future.

---

## Testing the Feature

### Test 1: Fresh Install (No Models)

```bash
# Clean state
rm -rf models/*.gguf

# Run install
make install

# Expected:
# - Installs deps
# - Detects missing models
# - Downloads both models
# - Reports success
```

### Test 2: Partial Models

```bash
# Only one model present
rm models/tiny-aya-earth-q4_k_m.gguf

# Run install
make install

# Expected:
# - Installs deps
# - Detects incomplete models
# - Downloads both models
```

### Test 3: All Models Present

```bash
# Models already exist
ls models/*.gguf
# tiny-aya-global-q4_k_m.gguf
# tiny-aya-earth-q4_k_m.gguf

# Run install
make install

# Expected:
# - Installs deps
# - Detects models present
# - Skips download
# - Fast completion (~3 seconds)
```

### Test 4: Force Re-download

```bash
# Models exist but want fresh download
make models

# Expected:
# - Ignores existing files
# - Re-downloads both
# - Overwrites if already present
```

---

## Documentation Updates

Created comprehensive documentation:

1. **`docs/makefile-commands.md`** - Complete Makefile reference
   - All targets explained
   - Usage examples
   - Troubleshooting guide
   - Workflow examples

2. **This document** - Feature specification
   - Implementation details
   - Benefits and rationale
   - Testing scenarios

---

## Integration with Existing Workflow

### README.md

The README already references `make install` (line 79):

```bash
# 3. Create Python virtual environment and install all packages
make install
```

**No changes needed** - the feature is backward compatible. The README workflow still works, just now with automatic model download.

### CI/CD Implications

If using GitHub Actions or similar:

**Before:**
```yaml
- name: Install dependencies
  run: make install
  
- name: Download models
  run: python models/pull_models.py
  
- name: Test
  run: make test
```

**After:**
```yaml
- name: Install dependencies (includes models)
  run: make install
  
- name: Test
  run: make test
```

⚠️ **Note:** CI environments may have network/timeout constraints for 4.2 GB downloads. Consider caching the models directory.

---

## Future Improvements

### 1. Smart Partial Download

Current behavior downloads both models even if one is present.

**Improvement:**
```bash
check-models:
	@echo "🔍 Checking for model files..."
	@missing=0; \
	[ ! -f models/tiny-aya-global-q4_k_m.gguf ] && missing=1 && echo "⚠️ Missing: tiny-aya-global"; \
	[ ! -f models/tiny-aya-earth-q4_k_m.gguf ] && missing=1 && echo "⚠️ Missing: tiny-aya-earth"; \
	if [ $$missing -eq 1 ]; then \
		# Download only missing models
	fi
```

### 2. Model Verification

Add checksum verification:
```bash
# Verify file integrity
sha256sum -c models/checksums.txt
```

### 3. Progress Bar

Use `tqdm` or similar in `pull_models.py` for better download feedback.

### 4. Configurable Model Selection

Allow users to choose which models to download:
```bash
make models MODEL=global  # Download only global
make models MODEL=earth   # Download only earth
make models               # Download both (default)
```

---

## Comparison with Other Projects

### Similar Patterns

Many ML projects use similar automatic model download:

| Project | Approach |
|---------|----------|
| **Stable Diffusion WebUI** | Auto-downloads models on first run |
| **llama.cpp** | Requires manual model placement |
| **Ollama** | Auto-downloads on `ollama run model` |
| **LocalAI** | Auto-downloads via API call |
| **DocuNative (now)** | Auto-downloads during `make install` |

DocuNative now follows the "automatic setup" pattern of modern ML tools.

---

## Summary

✅ **Implemented:**
- Automatic model verification during `make install`
- Standalone `make models` command for manual download
- Clear visual feedback with emoji indicators
- Graceful failure handling with recovery instructions
- Comprehensive documentation

✅ **Benefits:**
- One-command setup experience
- Idempotent (safe to re-run)
- Prevents "model not found" errors when starting server
- Better developer experience

✅ **Backward Compatible:**
- Existing workflows still work
- No breaking changes to README instructions
- Optional manual download still available

---

**Implementation Complete!** 🎉

Users can now simply run:
```bash
make install
```

And everything (dependencies + models) will be ready to go.
