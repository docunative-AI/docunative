# Makefile Commands Reference

## Available Commands

### `make install`
Complete first-time setup:
1. Creates Python 3.11 virtual environment
2. Installs all dependencies via UV
3. **Automatically checks for model files**
4. Downloads models if missing (~4.2 GB, 5-10 minutes)

```bash
make install
```

**What it does:**
- ✅ Creates `.venv/` with Python 3.11
- ✅ Installs 114 packages from `pyproject.toml`
- ✅ Checks for `tiny-aya-global-q4_k_m.gguf`
- ✅ Checks for `tiny-aya-earth-q4_k_m.gguf`
- ✅ Downloads models if missing
- ✅ Reports success or failure

**First-time install:** ~5-15 minutes (depending on internet speed)  
**Subsequent runs:** ~1-2 minutes (models already present)

---

### `make models`
Download or re-download model files:

```bash
make models
```

**When to use:**
- Models failed to download during `make install`
- Models were deleted or corrupted
- Want to force re-download

**Download size:** ~4.2 GB total
- `tiny-aya-global-q4_k_m.gguf` (~2.1 GB)
- `tiny-aya-earth-q4_k_m.gguf` (~2.1 GB)

---

### `make test`
Run the test suite:

```bash
make test
```

Uses `uv run pytest -q` to execute tests without manual venv activation.

---

### `make server-global`
Start the Tiny Aya Global model server:

```bash
make server-global
```

**What it does:**
- Loads `tiny-aya-global-q4_k_m.gguf` (~2.1 GB into RAM)
- Starts llama-server on port 8080
- Uses Metal (Mac), CUDA (Linux), or CPU

**Keep this terminal open** - the server runs in foreground.

---

### `make server-earth`
Start the Tiny Aya Earth model server:

```bash
make server-earth
```

**What it does:**
- Loads `tiny-aya-earth-q4_k_m.gguf` (~2.1 GB into RAM)
- Starts llama-server on port 8080
- Domain-specialist model (fine-tuned)

**Keep this terminal open** - the server runs in foreground.

---

### `make demo`
Launch the Gradio UI:

```bash
make demo
```

**Prerequisites:** 
- One of the model servers must be running in another terminal
- Run `make server-global` or `make server-earth` first

**Access:** Opens on http://localhost:7860

---

## Typical Workflow

### First-Time Setup

```bash
# Terminal 1: Setup everything
make install

# Wait for completion (~5-15 minutes on first run)
```

### Running the App

```bash
# Terminal 1: Start model server
make server-global

# Wait for: "llama server listening at http://127.0.0.1:8080"

# Terminal 2: Launch UI
make demo

# Open browser: http://localhost:7860
```

---

## Troubleshooting

### Models didn't download during install

```bash
# Try downloading manually
make models
```

### Model download fails with HuggingFace XET errors

**Option 1:** Use huggingface-cli directly:
```bash
source .venv/bin/activate
huggingface-cli download CohereLabs/tiny-aya-global-GGUF tiny-aya-global-q4_k_m.gguf --local-dir models
huggingface-cli download CohereLabs/tiny-aya-earth-GGUF tiny-aya-earth-q4_k_m.gguf --local-dir models
```

**Option 2:** Download from browser:
1. Visit https://huggingface.co/CohereLabs/tiny-aya-global-GGUF
2. Click "Files and versions" tab
3. Download `tiny-aya-global-q4_k_m.gguf`
4. Move to `models/` folder
5. Repeat for tiny-aya-earth-GGUF

### Server won't start - "Model file not found"

```bash
# Check if models exist
ls -lh models/*.gguf

# If missing, download:
make models
```

### "Python 3.11 not found"

```bash
# macOS
brew install python@3.11

# Linux
sudo apt install python3.11

# Then retry
make install
```

---

## Advanced Usage

### Clean install (reset everything)

```bash
rm -rf .venv uv.lock
make install
```

### Update dependencies

```bash
uv lock --upgrade
uv sync
```

### Check model files manually

```bash
ls -lh models/
# Should show:
# tiny-aya-global-q4_k_m.gguf (~2.1 GB)
# tiny-aya-earth-q4_k_m.gguf (~2.1 GB)
```

---

## Make Targets Summary

| Command | Purpose | Duration |
|---------|---------|----------|
| `make install` | Full setup + auto-download models | 5-15 min (first time) |
| `make models` | Download/re-download models | 5-10 min |
| `make test` | Run tests | ~10 sec |
| `make server-global` | Start Global model server | Stays running |
| `make server-earth` | Start Earth model server | Stays running |
| `make demo` | Launch Gradio UI | Stays running |

---

## What's New in This Makefile

### ✨ Automatic Model Verification

The `make install` command now **automatically checks** for model files and downloads them if missing.

**Before (old behavior):**
```bash
make install          # Install deps only
python models/pull_models.py  # Separate manual step
make server-global    # Would fail if models missing
```

**After (new behavior):**
```bash
make install          # Does everything:
                      # 1. Creates venv
                      # 2. Installs deps
                      # 3. Checks for models
                      # 4. Downloads if needed
make server-global    # Works immediately
```

### 🎯 Benefits

1. **One command setup** - `make install` does everything
2. **Idempotent** - Re-running `make install` skips model download if already present
3. **Clear feedback** - Reports model status with emoji indicators
4. **Graceful failure** - Shows manual download instructions if automatic fails
5. **Standalone target** - `make models` for manual re-download

---

## Example Session

```bash
$ make install
uv venv --python 3.11
Using CPython 3.11.13
Creating virtual environment at: .venv
uv sync
Resolved 132 packages in 6ms
Installed 114 packages in 392ms
🔍 Checking for model files...
⚠️  Model files not found. Downloading (~4.2 GB, this may take 5-10 minutes)...
Downloading tiny-aya-global-q4_k_m.gguf...
Successfully downloaded to: models/tiny-aya-global-q4_k_m.gguf
Downloading tiny-aya-earth-q4_k_m.gguf...
Successfully downloaded to: models/tiny-aya-earth-q4_k_m.gguf
All downloads complete! You are ready to run llama-server.
✅ Models downloaded successfully.
✅ Environment ready. Run 'source .venv/bin/activate' to activate.

$ make server-global
bash models/start_server.sh global
Loading model: models/tiny-aya-global-q4_k_m.gguf
llama server listening at http://127.0.0.1:8080
```

---

For more details, see:
- [UV Migration Guide](uv-migration-guide.md)
- [Essential Libraries Reference](essential-libraries.md)
- [Migration Summary](uv-migration-summary.md)
