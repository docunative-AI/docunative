# 🌍 DocuNative

Privacy-first, fully offline cross-lingual document QA for migrants and newcomers.

Upload a foreign-language legal document — a German lease, a French work contract, a Hindi tenancy agreement — and ask questions in **your own language**. Get answers with source quotes and a hallucination trust score: **entirely on your device, nothing sent to the cloud.**

Built for the Cohere AI Hackathon · March 10–24, 2026

---

## 🏗️ Tech Stack

| Component               | Technology                                                   |
| :---------------------- | :----------------------------------------------------------- |
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

# 2. Create Python virtual environment and install all packages
make install

# 3. Authenticate with HuggingFace (paste your token when prompted)
huggingface-cli login

# 4. Download the Tiny Aya models (~4.2 GB total — grab a coffee)
source venv/bin/activate          # Windows: venv\Scripts\activate
python models/pull_models.py
```

Setup is done. You only ever need to run Step 2 once.

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
source venv/bin/activate          # Windows: venv\Scripts\activate
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

| Symptom                             | Likely cause                          | Fix                                                                                                             |
| :---------------------------------- | :------------------------------------ | :-------------------------------------------------------------------------------------------------------------- |
| `curl /health` returns nothing      | Server not started                    | Open Terminal 1 and run `make server-global`                                                                    |
| UI loads but answers are empty      | Server crashed                        | Check Terminal 1 — look for error messages                                                                      |
| `unknown pre-tokenizer type` error  | Used ollama or llama-cpp-python       | Use `make server-global` — we do not use wrappers                                                               |
| `make: command not found` (Windows) | GNU Make not installed                | Install [Make for Windows](https://gnuwin32.sourceforge.net/packages/make.htm) or use the `.bat` files directly |
| Model download fails                | HF token not set / terms not accepted | Run `huggingface-cli login` and accept terms at both model pages                                                |
| Port 8080 already in use            | Another process using it              | Run `lsof -i :8080` (Mac/Linux) and kill the process                                                            |
| Zero chunks returned from PDF       | Scanned PDF (no text layer)           | Phase 2 supports text-layer PDFs only. OCR is Phase 3.                                                          |

---

## Rules of the Repo

- **Strictly offline.** The UI and RAG pipeline are forbidden from calling any cloud API.
- **No Python wrappers.** We compile raw `llama.cpp` via CMake. Ollama and `llama-cpp-python` both fail on Tiny Aya's tokenizer.
- **Two-terminal setup.** Terminal 1 = server. Terminal 2 = UI. Always.

---

## 📄 License

MIT License · Built during the Cohere AI Hackathon, March 2026
