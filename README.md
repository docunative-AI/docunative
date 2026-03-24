# 🌍 DocuNative

Privacy-first, fully offline cross-lingual document QA for migrants and newcomers.

Upload a foreign-language legal document (e.g. a Chinese lease agreement) and ask
questions in your own language (e.g. Polish). Get answers with source quotes and
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
→ [CohereLabs/tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire)

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
# The multilingual generalist (default) — GPU/Metal with prompt caching
make server-global

# The South Asian specialist (H1 — Hindi document QA)
make server-fire

# CPU users (Windows/Linux) — lower quantization for survivable latency
make server-global-q3    # ~30% faster than Q4 on CPU
make server-global-iq2   # ~60% faster, some quality loss
```

The UI model selector also reflects which model is currently loaded.

> **Mac users:** If generation feels slow, run `make check-metal` to verify Metal is active.

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

See [ROADMAP.md](docs/ROADMAP.md) for the full dependency graph.

---

## 🔬 Research Hypotheses

We are building this pipeline to answer two specific questions for our Hackathon paper:

**H1 — The Specialist Advantage**  
Does Tiny Aya **Fire** (South Asian specialist, trained with 5.8% South Asian data) outperform Tiny Aya **Global** (generalist) on Hindi legal document QA?

To test H1: stop Terminal 1, restart with `make server-fire`, ask the same questions in Hindi, compare results.

**H2 — Internal Training Proportion Gradient**  
Does accuracy degrade as the language's internal training proportion in Tiny Aya decreases?

We test across three languages with a clean step gradient in Tiny Aya's All Regions training mix (Appendix A, Tiny Aya technical report):

| Language | Internal % | External NLP Resources |
|---|---|---|
| Chinese (Simplified) | 1.9% | High — vast web presence, strong NLP ecosystem |
| Hindi | 1.7% | Medium — growing rapidly, good tooling |
| Polish | 1.4% | Medium-low — smaller NLP research community |

All three languages are natively supported by Aya Expanse 32B (used for document generation), eliminating any document quality confound. The 0.5% gradient from Chinese to Polish is the widest achievable while maintaining clean document generation.

**Why this framing matters:** Tiny Aya deliberately balances training data across languages to reduce the curse of multilinguality. If we still observe zh > hi > pl performance despite near-equal training proportions, it suggests structural and linguistic factors matter independently of training data quantity. If we observe no gradient, it confirms Tiny Aya's balancing technique achieves its design goal for document QA — itself a novel finding.

**External validation (Eval 3):** We validate findings on MKQA (Longpre et al., 2021 — ACL Anthology 2021.tacl-1.82), a real-world multilingual QA benchmark covering Chinese, Hindi, and Polish. This confirms results generalise beyond synthetic documents.

To test H2 yourself: run `make server-global`, then run `python -m eval.evaluate --qa dataset/output/qa_pairs.jsonl --docs dataset/output --model Global --run-name eval1-h2`.

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

# Or start with the fire model (H1 — South Asian specialist)
make server-fire
```

### Windows

```bash
# Start with the global model
models\start_server.bat global

# Or start with the fire model (H1 — South Asian specialist)
models\start_server.bat fire
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

## 🧪 Running the Evaluation

The full evaluation pipeline tests DocuNative against 3,600 synthetic QA pairs across Chinese, Hindi, and Polish documents.

**Three evaluation sets:**
- **Eval 1** — Template QA: English questions from deterministic seed facts
- **Eval 2** — LLM QA: Questions generated IN the document language by Aya Expanse 32B
- **Eval 3** — Real-world: MKQA benchmark (Chinese, Hindi, Polish) for external validation

**Step 1 — Generate documents (first time only):**
```bash
python -m dataset.builder.writer --language zh
python -m dataset.builder.writer --language hi
python -m dataset.builder.writer --language pl
```

**Step 2 — Generate QA pairs:**
```bash
# Eval 1 — template QA
python -m dataset.builder.qa_factory --full

# Eval 2 — LLM QA in document language
python -m dataset.builder.qa_factory_llm
```

**Step 3 — Pre-compute embeddings (run once, saves ~39 min per eval run):**
```bash
python -m eval.precompute_embeddings --docs dataset/output/
```

**Step 4 — Start the server (Terminal 1):**
```bash
make server-global
```

**Step 5 — Run Eval 1 — H2 (Terminal 2):**
```bash
# Full run — 3,600 pairs (~4-5 hours on Metal)
# Writes eval_results_eval1-h2.jsonl and eval_report_eval1-h2.txt
python -m eval.evaluate \
  --qa dataset/output/qa_pairs.jsonl \
  --docs dataset/output \
  --model Global \
  --run-name eval1-h2
```

**Step 6 — Run Eval 1 — H1 (Fire vs Global on Hindi):**
```bash
make server-fire   # Terminal 1
python -m eval.evaluate \
  --qa dataset/output/qa_pairs.jsonl \
  --docs dataset/output \
  --model Fire --language hi \
  --run-name eval1-h1-fire

make server-global  # Terminal 1
python -m eval.evaluate \
  --qa dataset/output/qa_pairs.jsonl \
  --docs dataset/output \
  --model Global --language hi \
  --run-name eval1-h1-global
```

**Step 7 — Run Eval 2 — LLM QA:**
```bash
make server-global  # Terminal 1
python -m eval.evaluate \
  --qa dataset/output/qa_pairs_llm.jsonl \
  --docs dataset/output \
  --model Global \
  --run-name eval2-llm
```

**Step 8 — Run Eval 3 — Real-world MKQA:**
```bash
make server-global  # Terminal 1
# Hindi is not in MKQA; default languages are zh and pl
python -m eval.eval_mkqa --model Global --limit 100
```

Eval 1 / Eval 2 outputs go to `eval/results/` as `eval_results_<run-name>.jsonl` and `eval_report_<run-name>.txt` (see `--run-name` above). Eval 3 writes `eval_mkqa_results.jsonl` and `eval_mkqa_report.txt`.

For the synthetic-vs-MKQA section in the Eval 3 report, `eval_mkqa` reads `eval/results/eval_results.jsonl` if it exists. After Step 5, copy or symlink your synthetic run there, e.g. `cp eval/results/eval_results_eval1-h2.jsonl eval/results/eval_results.jsonl`.

> **Note:** `eval/results/` and `dataset/output/*.jsonl` are gitignored. Share output files manually.

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
#  On Linux/macOS
source .venv/bin/activate

# On Windows
source .venv/Scripts/activate
```

> ⚠️ **Adding new packages?** Always use `uv add` — this updates both `pyproject.toml` and `uv.lock` atomically.
> ```bash
> uv add <package>
> git add pyproject.toml uv.lock
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
