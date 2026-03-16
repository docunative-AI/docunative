# 🌍 DocuNative

Privacy-first, fully offline cross-lingual document QA for migrants and newcomers.

Upload a foreign-language legal document (e.g. a German lease agreement) and ask
questions in your own language (e.g. Swahili). Get answers with source quotes and
a hallucination trust score: entirely on your device, nothing sent to the cloud.

Built for the Cohere AI Hackathon · March 10–24, 2026

---

## 🏗️ Tech Stack

| Component               | Technology                                                   |
| :---------------------- | :----------------------------------------------------------- |
| **Python**              | Python 3.11 (pinned for ML library stability)                |
| **Package Manager**     | UV (modern, fast dependency management)                      |
| **LLM**                 | Tiny Aya 3.35B GGUF via `llama-server` (local C++ inference) |
| **Embeddings**          | BAAI/bge-m3 (cross-lingual sentence transformers)            |
| **Vector DB**           | ChromaDB (local persistence)                                 |
| **Hallucination Check** | mDeBERTa-v3 (NLI Entailment)                                 |
| **UI**                  | Gradio                                                       |

---

## 🤔 New to the team? Two things to know first

**What is a Makefile?**  
You'll see commands like `make install` throughout this guide. A Makefile is just a shortcuts file — like the buttons on a washing machine. You don't need to know the exact spin speed; you just press "Quick Wash." When you type `make install`, it runs about a dozen setup commands behind the scenes so you don't have to.

**Why NOT ollama or llama-cpp-python?**  
Both Python wrappers currently crash on Tiny Aya's custom tokenizer (`unknown pre-tokenizer type: tiny_aya`). We compile and run `llama.cpp` directly as a C++ binary (`llama-server`). This is the only path that works reliably on all platforms.

---

## 🚀 Quick Start

### Step 1 — Prerequisites

You need a C++ build environment so we can compile the AI inference engine.

**Mac:**

```bash
brew install cmake
```

**Linux:**

```bash
sudo apt install cmake build-essential
```

**Windows:**  
Install [Visual Studio](https://visualstudio.microsoft.com/) with the **"Desktop development with C++"** workload, and [CMake](https://cmake.org/download/).

**HuggingFace (everyone):**  
You **must** have a HuggingFace account and agree to the model terms before downloading:  
→ [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global)  
→ [CohereLabs/tiny-aya-earth](https://huggingface.co/CohereLabs/tiny-aya-earth)

---

### Step 2 — First-Time Setup

Run these commands one by one in your terminal:

```bash
# 1. Clone the repo
git clone https://github.com/docunative-AI/docunative.git
cd docunative

# 2. Install uv — fast Python package manager (once per machine)
# macOS
brew install uv
# Linux / Windows (Git Bash or WSL)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create Python virtual environment and install all packages + models
make install
# This will:
# - Create a Python 3.11 virtual environment
# - Install all dependencies (~30 seconds)
# - Download Tiny Aya models if not present (~4.2 GB, 5-10 minutes)

# 4. Authenticate with HuggingFace (if models need downloading)
huggingface-cli login
```

Setup is done. You only ever need to run Step 2 once.

**Note:** The first `make install` takes 5-10 minutes (downloads models). Subsequent runs take ~30 seconds (models already present).

---

### Step 3 — Running the App (The Two-Terminal Rule)

DocuNative is 100% offline. The AI model runs as a background server on your machine — so you need **two terminal windows open at the same time**.

> ⚠️ If you close Terminal 1, the app in Terminal 2 will crash.

**🟢 Terminal 1 — Start the AI server (do this first)**

This loads the 2GB model into RAM and keeps it running on port 8080.

```bash
# Mac / Linux
make server-global

# Windows
models\start_server.bat global
```

Wait until you see:

```
llama server listening at http://127.0.0.1:8080
```

Leave this terminal open.

**🔵 Terminal 2 — Launch the UI**

Open a brand new terminal window:

```bash
cd docunative
source .venv/bin/activate        # Windows: .venv\Scripts\activate
make demo
```

🎉 Open **http://localhost:7860** in your browser. You're running DocuNative.

---

## 🔄 Switching Between Models

We have two model variants to test. To switch, stop Terminal 1 and restart it:

```bash
# The multilingual generalist (default)
make server-global

# The domain-specialist (fine-tuned on documents)
make server-earth
```

The UI model selector also reflects which model is currently loaded.

---

## 🩺 Health Check

At any point, verify the server is alive:

```bash
curl http://localhost:8080/health
# Expected: {"status":"ok"}
```

If the health check fails:

1. Check that Terminal 1 is still open and running
2. Check that the model finished loading (look for the `listening` line)
3. Try `make server-global` again from scratch

---

## 🏛️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Gradio UI (port 7860)                       │
│         PDF upload · Language selector · NLI trust badge      │
└────────────────────────────────┬───────────────────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │            pipeline/                │
              │  extract.py   (PDF → text)          │
              │  embed.py     (text → vecs)         │  ← BAAI/bge-m3
              │  retrieve.py  (vecs → top3)         │  ← ChromaDB (local)
              │  generate.py  (top3 → answer)       │  ← llama-server :8080
              │  validate.py  (answer → struct)     │
              │  nli.py       (hallucination check) │  ← mDeBERTa-v3
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │     llama-server (port 8080)        │
              │     Tiny Aya GGUF · C++ binary      │
              │     Metal / CUDA / CPU              │
              └─────────────────────────────────────┘
```

The setup script compiles `llama.cpp` automatically with the best backend for your hardware:

- **macOS ARM64 / x86_64** — Metal (Apple Silicon or Intel GPU)
- **Linux with NVIDIA GPU** — CUDA
- **Linux / Windows (CPU only)** — OpenBLAS

See [README_ROADMAP.md](README_ROADMAP.md) for the full dependency graph.

---

## 🔬 Research Hypotheses

We are building this pipeline to answer two specific questions for our Hackathon paper:

**H1 — The Specialist Advantage**  
Does Tiny Aya **Earth** (domain-specialist) outperform Tiny Aya **Global** (generalist) on non-English legal document QA?

**H2 — Resource-Level Degradation**  
Does accuracy degrade as the language resource level decreases?  
We test across: **German** (high-resource) → **Hindi** (medium-resource) → **Swahili** (low-resource)

To test H1 yourself: stop Terminal 1, restart with `make server-earth`, ask the same questions, compare.

---

## 🛠️ Troubleshooting

### Starting the inference server

The inference server runs a compiled C++ binary (llama-server) on port 8080. The setup script automatically:

1. Clones llama.cpp if not already present
2. Compiles with the appropriate backend:
   - **macOS ARM64**: Metal acceleration (Apple Silicon)
   - **macOS x86_64**: Metal acceleration
   - **Linux with CUDA**: GPU acceleration
   - **Linux without CUDA**: CPU-only
   - **Windows**: CPU-only (requires Visual Studio with C++ tools)
3. Starts the server on port 8080

### macOS/Linux

```bash
# Start with the global model
make server-global

# Or start with the earth model
make server-earth
```

### Windows

```bash
# Start with the global model
models\start_server.bat global

# Or start with the earth model
models\start_server.bat earth
```

### Health check

Verify the server is running:

```bash
curl http://localhost:8080/health
# Should return: {"status":"ok"}
```

> ⚠️ **Note:** We do NOT use ollama or llama-cpp-python. The model runs via
> llama-server (compiled C++ binary) on port 8080.

---

## 👩‍💻 Development Guide

### For Contributors: Working with UV

DocuNative uses **UV** for dependency management. This provides faster installs, reproducible builds, and better dependency resolution.

**📖 Complete UV Development Guide:** [`docs/uv-development-guide.md`](docs/uv-development-guide.md)

The guide covers:
- Setting up new branches with `uv sync`
- Installing packages (temporary vs. permanent)
- Adding dependencies to `pyproject.toml`
- Updating dependencies (`uv lock --upgrade`)
- Common commands reference
- Troubleshooting
- Best practices
- Complete example workflows

**Quick Start for Contributors:**

```bash
# 1. Create your branch
git checkout -b feature/your-feature

# 2. Install dependencies
uv sync

# 3. Start coding!
source .venv/bin/activate
```

> ⚠️ **Adding new packages?** Always use `make freeze` (not `pip freeze`) to update requirements.txt. This prevents Conda/Anaconda local paths from contaminating the file.
> ```bash
> uv pip install <package>
> make freeze   # safe — detects and rejects file:// contamination
> git add requirements.txt
> ```

For detailed workflows, see the full guide linked above.

---

## Rules of the Repo

- **Strictly offline.** The UI and RAG pipeline are forbidden from calling any cloud API.
- **No Python wrappers.** We compile raw `llama.cpp` via CMake. Ollama and `llama-cpp-python` both fail on Tiny Aya's tokenizer.
- **Two-terminal setup.** Terminal 1 = server. Terminal 2 = UI. Always.

---

## 📄 License

MIT License · Built during the Cohere AI Hackathon, March 2026
