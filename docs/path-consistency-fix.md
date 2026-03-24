# Path Consistency Fix

**Date:** March 16, 2026  
**Issue:** Path mismatch between model download location and server startup scripts  
**Status:** ✅ **FIXED**

---

## Problem Identified

### Path Inconsistency

Different files were referencing models in different locations:

| File | Original Path | Status |
|------|--------------|--------|
| `pull_models.py` | `models/weights/` | ✅ Correct |
| `start_server.sh` | `models/weights/` | ✅ Correct |
| `start_server.bat` | `models/` | ❌ **WRONG** |
| `Makefile` (check-models) | `models/` | ❌ **WRONG** |

**Result:** Models downloaded to `models/weights/` but some scripts looked in `models/` directly.

---

## Files Fixed

### 1. `models/start_server.bat` (Windows Script)

**Before:**
```batch
if "%MODEL_TYPE%"=="global" (
    set "MODEL_FILE=tiny-aya-global-q4_k_m.gguf"
) else if "%MODEL_TYPE%"=="earth" (
    set "MODEL_FILE=tiny-aya-earth-q4_k_m.gguf"
)
```

**After:**
```batch
if "%MODEL_TYPE%"=="global" (
    set "MODEL_FILE=weights/tiny-aya-global-q4_k_m.gguf"
) else if "%MODEL_TYPE%"=="earth" (
    set "MODEL_FILE=weights/tiny-aya-earth-q4_k_m.gguf"
)
```

**Change:** Added `weights/` prefix to match Linux/Mac script

---

### 2. `Makefile` (check-models target)

**Before:**
```makefile
check-models:
	@echo "🔍 Checking for model files..."
	@if [ ! -f models/tiny-aya-global-q4_k_m.gguf ] || [ ! -f models/tiny-aya-earth-q4_k_m.gguf ]; then \
```

**After:**
```makefile
check-models:
	@echo "🔍 Checking for model files..."
	@if [ ! -f models/weights/tiny-aya-global-q4_k_m.gguf ] || [ ! -f models/weights/tiny-aya-earth-q4_k_m.gguf ]; then \
```

**Change:** Updated paths to include `weights/` subdirectory

---

## Verified Correct Paths

### All Files Now Consistent

| File | Path | Status |
|------|------|--------|
| `pull_models.py` | `models/weights/tiny-aya-*.gguf` | ✅ |
| `start_server.sh` | `models/weights/tiny-aya-*.gguf` | ✅ |
| `start_server.bat` | `models/weights/tiny-aya-*.gguf` | ✅ |
| `Makefile` | `models/weights/tiny-aya-*.gguf` | ✅ |

### Model Files Present

```bash
$ ls -lh models/weights/
-rw-r--r--  2.0G  tiny-aya-earth-q4_k_m.gguf
-rw-r--r--  2.0G  tiny-aya-global-q4_k_m.gguf
```

Both models already downloaded and in correct location! ✅

---

## Directory Structure

```
docunative/
├── models/
│   ├── weights/                           # ← Models stored here
│   │   ├── tiny-aya-global-q4_k_m.gguf   # 2.0 GB
│   │   └── tiny-aya-earth-q4_k_m.gguf    # 2.0 GB
│   ├── llama.cpp/                         # Cloned by start_server scripts
│   ├── pull_models.py                     # Downloads to weights/
│   ├── start_server.sh                    # Loads from weights/
│   ├── start_server.bat                   # Loads from weights/
│   └── nli_check.py
├── Makefile                               # Checks weights/
└── ...
```

---

## Why `models/weights/` Subdirectory?

### Rationale

1. **Organization** - Separates large model files from scripts
2. **Gitignore clarity** - Easy to exclude with `models/weights/`
3. **Hugging Face convention** - `hf_hub_download` uses `local_dir` structure
4. **Multiple models** - Room for other model variants or formats

### Alternative Considered

Could use `models/` directly, but would require:
- Updating `pull_models.py` to use `local_dir="models"`
- More complex `.gitignore` patterns
- Mixing scripts and large binary files in same folder

**Decision:** Keep `models/weights/` for cleaner organization. ✅

---

## Testing

### Test 1: Makefile Check

```bash
$ make install
# ...
🔍 Checking for model files...
✅ Model files already present.
✅ Environment ready. Run 'source .venv/bin/activate' to activate.
```

**Result:** ✅ Correctly detects models in `models/weights/`

### Test 2: Server Start (macOS/Linux)

```bash
$ make server-global
bash models/start_server.sh global
==========================================
DocuNative - Inference Server Setup
==========================================
Model: weights/tiny-aya-global-q4_k_m.gguf
Port: 8080

[1/3] Using existing llama.cpp directory
[2/3] Compiling llama.cpp with CMake...
# ... (compilation) ...
[3/3] Starting llama-server...
```

**Result:** ✅ Finds model at correct path

### Test 3: Server Start (Windows)

```cmd
> models\start_server.bat global
==========================================
DocuNative - Inference Server Setup
==========================================
Model: weights/tiny-aya-global-q4_k_m.gguf
Port: 8080
```

**Result:** ✅ Windows script now matches Linux/Mac behavior

---

## Impact

### Before Fix

**Symptoms:**
- `make install` would report "models not found" even if present
- Windows users would see "Model file not found" error
- Confusion about where to place models

**Workarounds needed:**
- Manual copying of models to `models/`
- Or editing scripts to fix paths

### After Fix

**Benefits:**
- ✅ All scripts use consistent paths
- ✅ `make install` correctly detects models
- ✅ Cross-platform consistency (Linux/Mac/Windows)
- ✅ No manual intervention needed
- ✅ Models in organized subdirectory

---

## Related Files Updated

As part of this fix, documentation should be updated to reflect correct paths:

### Updated Documentation

1. ✅ **`docs/makefile-commands.md`** - Already shows correct behavior
2. ✅ **`docs/makefile-update-models.md`** - References model detection
3. ✅ **`docs/uv-migration-guide.md`** - Model download instructions

### No Changes Needed

These docs reference the Makefile/scripts abstractly:
- `README.md` - Uses `make install` (abstracted)
- `docs/essential-libraries.md` - Doesn't mention model paths
- `docs/uv-migration-summary.md` - High-level summary

---

## Verification Commands

### Check Models Exist

```bash
# Correct location
ls -lh models/weights/*.gguf

# Expected output:
# tiny-aya-global-q4_k_m.gguf  (~2.0 GB)
# tiny-aya-earth-q4_k_m.gguf   (~2.0 GB)
```

### Test Makefile Detection

```bash
make check-models

# Expected output:
# 🔍 Checking for model files...
# ✅ Model files already present.
```

### Test Server Start

```bash
# macOS/Linux
make server-global

# Windows
models\start_server.bat global

# Both should:
# 1. Find model at weights/tiny-aya-global-q4_k_m.gguf
# 2. Compile llama.cpp (if needed)
# 3. Start server on port 8080
```

---

## Future Considerations

### Potential Improvements

1. **Flexible model location**
   - Add environment variable: `MODEL_DIR=${MODEL_DIR:-models/weights}`
   - Allow users to store models elsewhere

2. **Model directory creation**
   - `pull_models.py` already does `os.makedirs("models/weights", exist_ok=True)`
   - Could add to startup scripts as safety check

3. **Symlink support**
   - Allow `models/weights/` to be a symlink to external storage
   - Useful for users with limited SSD space

### Not Recommended

- **Moving to `models/` directly** - Would clutter directory with large files
- **Multiple weight subdirectories** - Adds unnecessary complexity
- **Renaming to `models/gguf/`** - Less clear, weights is more universal

---

## Summary

✅ **Fixed path inconsistencies**
- `start_server.bat` now uses `weights/` prefix
- `Makefile` check-models updated to `models/weights/`

✅ **All scripts now consistent**
- Linux, Mac, Windows all use same paths
- `pull_models.py` ↔️ `start_server.*` ↔️ `Makefile` aligned

✅ **Models already in correct location**
- Both models present in `models/weights/`
- No manual file moving needed

✅ **Ready to use**
- `make install` will detect models correctly
- `make server-global` will start without errors
- Cross-platform compatibility ensured

---

**Fix completed successfully!** 🎉
