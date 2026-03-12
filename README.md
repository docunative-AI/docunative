# DocuNative

Privacy-first, fully offline cross-lingual document QA for migrants and newcomers.

Upload a foreign-language legal document (e.g. a German lease agreement) and ask
questions in your own language (e.g. Swahili). Get answers with source quotes and
a hallucination trust score: entirely on your device, nothing sent to the cloud.

## Stack

| Component           | Technology                                           |
| ------------------- | ---------------------------------------------------- |
| LLM                 | Tiny Aya GGUF via llama-server (local C++ inference) |
| Embeddings          | BAAI/bge-m3 (cross-lingual)                          |
| Vector DB           | ChromaDB (local)                                     |
| Hallucination check | mDeBERTa-v3                                          |
| UI                  | Gradio                                               |

## Quick Start

```bash

> 🛑 **IMPORTANT PRE-REQUISITE:**
> You must have a HuggingFace account and agree to the model terms at [huggingface.co/CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global).

# 1. Clone the repo
git clone https://github.com/docunative-AI/docunative.git
cd docunative

# 2. Install dependencies
make install

# 3. Authenticate and download the models (~4GB total)
huggingface-cli login
python models/pull_models.py

# 4. Start the inference server (do this before running the app)
make server-global

# 5. In a new terminal, launch the UI
make demo

```

Then open http://localhost:7860 in your browser.

> ⚠️ **Note:** We do NOT use ollama or llama-cpp-python. The model runs via
> llama-server (compiled C++ binary) on port 8080. See Issue #26 for setup.

## Research Questions

- **H1:** Does Tiny Aya Earth outperform Global for non-English document QA?
- **H2:** Does accuracy degrade as language resource level decreases?
  (German → Hindi → Swahili)

## Architecture

See [README_ROADMAP.md](README_ROADMAP.md) for the full system diagram.

## Hackathon

Built during the Cohere AI Hackathon — March 10–24, 2026.
MIT License.
