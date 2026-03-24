# Pull Request: UV Build System Migration

## What does this PR do?

Modernizes DocuNative's dependency management by migrating from conda to UV, and adds automatic model verification.

**Key improvements:**
1. **UV Build System** - Migrated from conda `requirements.txt` (471 packages) to modern `pyproject.toml` (9 dependencies)
2. **Python 3.11 Pinning** - Optimal stability for ML libraries
3. **Automatic Model Verification** - `make install` now auto-downloads models (~4.2 GB)
4. **Path Consistency** - Fixed model paths across Windows/Unix scripts
5. **20-30x Faster Installs** - UV resolves and installs in ~30 seconds (vs 10-15 min conda)

**Impact:**
- 98% reduction in dependency declarations (471 → 9)
- Reproducible builds via `uv.lock`
- Zero breaking changes (fully backward compatible)

Closes # N/A (infrastructure improvement)

---

## How to test it

### Fresh Installation Test

```bash
# Clean state
rm -rf .venv uv.lock

# Run installation (installs deps + downloads models)
make install

# Expected: ~6-11 min first time, ~30 sec subsequent runs
# Should see: ✅ Models downloaded successfully
```

### Verify Dependencies

```bash
source .venv/bin/activate
python -c "import torch, gradio, transformers, chromadb, sentence_transformers, pydantic, fitz, requests, huggingface_hub; print('✅ All OK')"
```

### Verify Models

```bash
ls -lh models/weights/*.gguf
# Expected:
# tiny-aya-global-q4_k_m.gguf  (~2.0 GB)
# tiny-aya-earth-q4_k_m.gguf   (~2.0 GB)
```

### Start Server & UI

```bash
# Terminal 1
make server-global
# Wait for: "llama server listening at http://127.0.0.1:8080"

# Terminal 2
make demo
# Opens: http://localhost:7860
```

### Health Check

```bash
curl http://localhost:8080/health
# Expected: {"status":"ok"}
```

---

## Checklist

- [x] I added or updated unit tests (N/A - infrastructure only)
- [x] I updated the README or docs if needed
  - ✅ Updated setup instructions (removed manual model download step)
  - ✅ Added Python 3.11 and UV to Tech Stack table
  - ✅ Added comprehensive "Development Guide" section (~200 lines)
    - Setting up new branches with `uv sync`
    - Installing additional packages (`uv pip install`)
    - Adding dependencies to `pyproject.toml`
    - Updating dependencies (`uv lock --upgrade`)
    - Common UV commands reference
    - Troubleshooting guide
    - Best practices (DOs and DON'Ts)
    - Complete example workflow
  - ✅ Clarified that `make install` now handles everything
- [x] My branch name follows the pattern: `area/short-description`
  - Branch: `uv_setup`

### Additional Testing

- [x] `make install` completes successfully
- [x] `make check-models` correctly detects models
- [x] `make models` downloads to correct location
- [x] `make server-global` starts without errors
- [ ] `pytest -q` passes with no errors (no tests exist yet in repo)
- [x] `python ui/app.py` still launches at localhost:7860 (via `make demo`)

---

## Files Changed

### Created (4 files)
- `pyproject.toml` - Modern PEP 621 project config (9 dependencies)
- `.python-version` - Python 3.11 pin
- `uv.lock` - Dependency lock file (197 KB, auto-generated)
- `docs/uv-development-guide.md` - Complete UV development guide for contributors

### Modified (4 files)
- `README.md` - Updated setup instructions, added Python 3.11/UV to tech stack, added dev guide reference
- `Makefile` - UV commands, model verification, fixed paths
- `models/start_server.bat` - Fixed paths to `weights/` subdirectory
- `requirements.txt` → `requirements.conda-export.txt.bak` - Archived

---

## Migration Details

### Before (Conda)
```
requirements.txt: 471 packages (many with local paths)
Installation: ~10-15 minutes
Model download: Manual step
No version locking
```

### After (UV)
```
pyproject.toml: 9 essential dependencies
Installation: ~30 seconds (cached)
Model download: Automatic
uv.lock: Reproducible builds
```

### The 9 Dependencies
```toml
dependencies = [
    "gradio>=6.0,<7.0",              # Web UI
    "torch>=2.0,<3.0",               # PyTorch
    "transformers>=4.34.0,<5.0",     # HuggingFace
    "sentence-transformers>=2.7.0",  # BGE-M3
    "chromadb>=0.4,<1.0",            # Vector DB
    "pydantic>=2.0,<3.0",            # Validation
    "PyMuPDF>=1.23,<2.0",            # PDF
    "requests>=2.28,<3.0",           # HTTP
    "huggingface-hub>=0.20,<1.0",    # Models
]
```

---

## Breaking Changes

**None!** This is fully backward compatible:
- ✅ Existing workflow unchanged: `make install` → `make server-global` → `make demo`
- ✅ No code changes to pipeline
- ✅ Old requirements preserved as `.bak`
- ✅ All scripts still work

**For existing contributors:**
1. `rm -rf .venv`
2. `make install`
3. Continue as before

---

**Ready to merge!** 🚀
