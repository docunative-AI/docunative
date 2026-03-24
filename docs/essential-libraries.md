# DocuNative Essential Libraries Reference

**Project:** DocuNative  
**Python Version:** 3.11  
**Last Updated:** March 16, 2026

---

## Overview

This document lists all essential third-party libraries used in DocuNative, based on comprehensive analysis of actual imports across the entire codebase.

**Total Essential Dependencies:** 9 packages  
**Total Installed (with transitive deps):** 114 packages

---

## The 9 Essential Dependencies

### 1. gradio (UI Framework)

- **Version:** ≥6.0, <7.0
- **Installed:** 6.9.0
- **Purpose:** Web-based user interface
- **Used In:** `ui/app.py`
- **Import:** `import gradio as gr`
- **Description:** Creates the browser-based UI for document upload, question input, and answer display
- **Why this library:** 
  - Zero frontend code required
  - Built-in file upload and text components
  - Fast prototyping for ML demos
  - Works entirely offline (no external CDN dependencies)

---

### 2. torch (PyTorch)

- **Version:** ≥2.0, <3.0
- **Installed:** 2.10.0
- **Purpose:** Deep learning framework for ML model inference
- **Used In:** 
  - `pipeline/embed.py` - Hardware detection (CUDA/MPS/CPU)
  - `models/nli_check.py` - NLI model inference
- **Imports:**
  ```python
  import torch
  torch.cuda.is_available()
  torch.backends.mps.is_available()
  ```
- **Description:** Provides the runtime for transformer models (BGE-M3, mDeBERTa)
- **Why this library:**
  - Industry standard for ML in Python
  - Cross-platform (CPU, CUDA, Metal)
  - Required by sentence-transformers and transformers
  - Offline model inference support

---

### 3. transformers (Hugging Face Transformers)

- **Version:** ≥4.34.0, <5.0
- **Installed:** 4.57.6
- **Purpose:** Load and run pre-trained language models
- **Used In:** `models/nli_check.py`
- **Imports:**
  ```python
  from transformers import AutoModelForSequenceClassification, AutoTokenizer
  import transformers  # for logging config
  ```
- **Description:** Loads the mDeBERTa-v3-base-xnli model for hallucination detection
- **Why this library:**
  - Access to 100,000+ pre-trained models
  - Standardized model loading API
  - Required for NLI entailment checking
  - Excellent multilingual model support

---

### 4. sentence-transformers

- **Version:** ≥2.7.0, <3.0
- **Installed:** 2.7.0
- **Purpose:** Sentence and text embeddings
- **Used In:**
  - `pipeline/embed.py` - Document chunk embeddings
  - `pipeline/retrieve.py` - Query embeddings
- **Import:** `from sentence_transformers import SentenceTransformer`
- **Description:** Wraps the BAAI/bge-m3 model for cross-lingual semantic embeddings
- **Model Used:** `BAAI/bge-m3` (multilingual, 1024-dimensional vectors)
- **Why this library:**
  - High-quality sentence embeddings
  - Cross-lingual support (matches Swahili queries to German documents)
  - Simple API: `model.encode(texts)`
  - Built on top of transformers + torch

---

### 5. chromadb

- **Version:** ≥0.4, <1.0
- **Installed:** 0.6.3
- **Purpose:** Local vector database for semantic search
- **Used In:**
  - `pipeline/embed.py` - Store document chunk embeddings
  - `pipeline/retrieve.py` - Query and retrieve relevant chunks
- **Imports:**
  ```python
  import chromadb
  from chromadb.config import Settings
  ```
- **Description:** Persists embeddings to disk at `.chromadb/`, enables semantic similarity search
- **Why this library:**
  - 100% offline (no server, no cloud)
  - Persistent local storage
  - Fast approximate nearest neighbor search (HNSW)
  - Simple Python API
  - No separate database installation required

---

### 6. pydantic

- **Version:** ≥2.0, <3.0
- **Installed:** 2.12.5
- **Purpose:** Data validation and parsing
- **Used In:** `pipeline/validate.py`
- **Imports:**
  ```python
  from pydantic import BaseModel, Field
  ```
- **Description:** Defines the `ParsedOutput` schema for structured LLM responses
- **Schema Defined:**
  ```python
  class ParsedOutput(BaseModel):
      answer: str
      source_quote: str
      parse_success: bool
      raw_output: str
  ```
- **Why this library:**
  - Type-safe data structures
  - Automatic validation
  - Clear schema definition
  - Runtime type checking

---

### 7. PyMuPDF

- **Version:** ≥1.23, <2.0
- **Installed:** 1.27.2
- **Purpose:** PDF text extraction
- **Used In:** `pipeline/extract.py`
- **Import:** `import fitz  # PyMuPDF`
- **Description:** Extracts text from PDF files, handles multi-column layouts
- **Why this library:**
  - Fast C++ backend
  - Better layout detection than alternatives
  - Handles complex PDFs (legal documents, contracts)
  - No external dependencies (no Poppler, no Java)
  - Imported as `fitz` for historical reasons

---

### 8. requests

- **Version:** ≥2.28, <3.0
- **Installed:** 2.32.5
- **Purpose:** HTTP client
- **Used In:** `pipeline/generate.py`
- **Import:** `import requests`
- **Description:** Makes POST requests to the local llama-server (port 8080)
- **Usage:**
  ```python
  response = requests.post(
      "http://localhost:8080/completion",
      json={"prompt": prompt, ...},
      timeout=120
  )
  ```
- **Why this library:**
  - Simple, reliable HTTP API
  - Timeout support
  - JSON encoding/decoding
  - Standard for Python HTTP

---

### 9. huggingface-hub

- **Version:** ≥0.20, <1.0
- **Installed:** 0.36.2
- **Purpose:** Download models from Hugging Face Hub
- **Used In:** `models/pull_models.py`
- **Import:** `from huggingface_hub import hf_hub_download`
- **Description:** Downloads Tiny Aya GGUF models during setup
- **Models Downloaded:**
  - `CohereLabs/tiny-aya-global-GGUF` (~2.1 GB)
  - `CohereLabs/tiny-aya-earth-GGUF` (~2.1 GB)
- **Why this library:**
  - Official HF download client
  - Handles authentication
  - Resume interrupted downloads
  - Caching support

---

## Import Mapping Table

| File | Imports | Dependencies Used |
|------|---------|-------------------|
| `ui/app.py` | `import os`<br>`import gradio as gr` | gradio |
| `pipeline/extract.py` | `import re`<br>`import unicodedata`<br>`import fitz`<br>`from pathlib import Path`<br>`from typing import List` | PyMuPDF |
| `pipeline/embed.py` | `import hashlib`<br>`import logging`<br>`from typing import Optional`<br>`import torch`<br>`import chromadb`<br>`from chromadb.config import Settings`<br>`from sentence_transformers import SentenceTransformer` | torch, chromadb, sentence-transformers |
| `pipeline/retrieve.py` | `import logging`<br>`from dataclasses import dataclass`<br>`import chromadb`<br>`from sentence_transformers import SentenceTransformer` | chromadb, sentence-transformers |
| `pipeline/generate.py` | `import requests`<br>`from typing import List, Dict, Any` | requests |
| `pipeline/validate.py` | `import re`<br>`from pydantic import BaseModel, Field`<br>`from typing import Optional` | pydantic |
| `models/nli_check.py` | `import torch`<br>`from transformers import AutoModelForSequenceClassification, AutoTokenizer`<br>`import transformers` | torch, transformers |
| `models/pull_models.py` | `import os`<br>`from huggingface_hub import hf_hub_download` | huggingface-hub |

**Note:** The following imports are Python standard library (no installation required):
- `os`, `re`, `unicodedata`, `pathlib`, `typing`, `dataclasses`, `hashlib`, `logging`, `tempfile`, `sys`

---

## Dependency Graph

```
docunative (0.1.0)
├── gradio (6.9.0)
│   ├── fastapi
│   ├── uvicorn
│   ├── httpx
│   └── websockets
│
├── torch (2.10.0)
│   ├── numpy
│   ├── filelock
│   ├── typing-extensions
│   ├── sympy
│   └── networkx
│
├── transformers (4.57.6)
│   ├── tokenizers
│   ├── safetensors
│   ├── regex
│   ├── huggingface-hub
│   └── torch (shared)
│
├── sentence-transformers (2.7.0)
│   ├── transformers (shared)
│   ├── torch (shared)
│   ├── scikit-learn
│   └── scipy
│
├── chromadb (0.6.3)
│   ├── onnxruntime
│   ├── chroma-hnswlib
│   ├── pandas
│   └── numpy (shared)
│
├── pydantic (2.12.5)
│   └── pydantic-core
│
├── PyMuPDF (1.27.2)
│
├── requests (2.32.5)
│   ├── urllib3
│   ├── certifi
│   ├── charset-normalizer
│   └── idna
│
└── huggingface-hub (0.36.2)
    ├── requests (shared)
    └── pyyaml
```

---

## Not Included (Python Standard Library)

The following modules are used extensively but require **no installation**:

| Module | Used In | Purpose |
|--------|---------|---------|
| `os` | Multiple files | File paths, environment variables |
| `re` | `extract.py`, `validate.py` | Text cleaning, regex parsing |
| `unicodedata` | `extract.py` | Unicode category detection |
| `pathlib` | `extract.py` | Path manipulation |
| `typing` | Multiple files | Type hints |
| `dataclasses` | `retrieve.py` | Data classes |
| `hashlib` | `embed.py` | Chunk ID generation |
| `logging` | Multiple files | Logging |
| `tempfile` | `extract.py` (tests) | Temporary files |
| `sys` | `extract.py` (tests) | System arguments |

---

## Why Only 9 Dependencies?

### Design Philosophy

1. **Minimize attack surface** - Fewer dependencies = fewer security risks
2. **Reduce maintenance burden** - Each dependency needs monitoring for updates
3. **Faster installs** - Less to download and compile
4. **Clearer architecture** - Easy to see what the project actually needs
5. **Offline-first** - Only essential ML/UI libraries, no cloud SDKs

### What's NOT Included

The old `requirements.txt` (471 packages) included:

- ❌ **IDE tools** - Spyder, Anaconda Navigator, Jupyter
- ❌ **Build tools** - conda-build, wheel, setuptools (dev-only now)
- ❌ **Unused ML libs** - sklearn, xgboost, lightgbm, optuna (not imported)
- ❌ **Data viz** - matplotlib, seaborn, bokeh (not used in production)
- ❌ **Web scraping** - scrapy, beautifulsoup (not used)
- ❌ **Transitive deps** - 400+ packages pulled in by conda

All of these have been eliminated in favor of **only what the code actually imports**.

---

## Version Constraints Explained

| Dependency | Constraint | Reason |
|------------|-----------|--------|
| gradio | `>=6.0,<7.0` | Gradio 6.x stable, avoid breaking changes in 7.0 |
| torch | `>=2.0,<3.0` | PyTorch 2.x is current stable line |
| transformers | `>=4.34.0,<5.0` | **Must be <5.0** for sentence-transformers compatibility |
| sentence-transformers | `>=2.7.0,<3.0` | 2.7.0 is latest stable, requires transformers<5.0 |
| chromadb | `>=0.4,<1.0` | Chromadb 0.x API, 1.0 may have breaking changes |
| pydantic | `>=2.0,<3.0` | Pydantic 2.x is current (breaking from 1.x) |
| PyMuPDF | `>=1.23,<2.0` | 1.x stable, 2.0 may change API |
| requests | `>=2.28,<3.0` | requests 2.x very stable |
| huggingface-hub | `>=0.20,<1.0` | 0.x under active development |

### Key Constraint: transformers<5.0

⚠️ **Critical:** `sentence-transformers` requires `transformers<5.0`

Initial migration attempted `transformers>=5.0` but UV correctly rejected this:

```
Because sentence-transformers>=2.7.0 depends on transformers>=4.34.0,<5.0
and your project depends on transformers>=5.0,<6.0,
we can conclude that your project's requirements are unsatisfiable.
```

Solution: Use `transformers>=4.34.0,<5.0` to satisfy both constraints.

---

## Platform-Specific Notes

### macOS (Apple Silicon)

- **torch:** Automatically uses Metal Performance Shaders (MPS) for GPU acceleration
- **PyMuPDF:** Pre-compiled wheel available, no build required
- **chromadb:** Uses native ARM64 HNSW library

### Linux (CUDA)

- **torch:** CPU-only by default; for CUDA, install from PyTorch index:
  ```bash
  uv pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```
- **chromadb:** Works out-of-box with CPU or CUDA

### Windows

- **torch:** CPU-only by default
- **PyMuPDF:** Pre-compiled wheel available
- **chromadb:** May require Visual C++ redistributables

---

## Model Files (Not Python Dependencies)

These are **data files**, not Python packages:

| Model | Size | Downloaded By | Stored In |
|-------|------|---------------|-----------|
| BAAI/bge-m3 | ~2.3 GB | `sentence-transformers` (auto) | `~/.cache/huggingface/` |
| mDeBERTa-v3-base-xnli | ~1.5 GB | `transformers` (auto) | `~/.cache/huggingface/` |
| tiny-aya-global (GGUF) | ~2.1 GB | `models/pull_models.py` | `models/` |
| tiny-aya-earth (GGUF) | ~2.1 GB | `models/pull_models.py` | `models/` |

**Total model storage:** ~8 GB

---

## FAQ

### Why not include `numpy` explicitly?

It's a transitive dependency of torch, scipy, scikit-learn, pandas (via chromadb). UV resolves it automatically. No need to declare it explicitly.

### Why not include `pytest` in dependencies?

It's in `[tool.uv] dev-dependencies`. Only installed when developing, not for end users.

### Can I add new dependencies?

Yes! Edit `pyproject.toml`:

```toml
dependencies = [
    # ... existing ...
    "new-package>=1.0,<2.0",
]
```

Then run: `uv sync`

### How do I update a specific package?

```bash
uv lock --upgrade-package torch
uv sync
```

### Why is chromadb so large (many deps)?

ChromaDB includes:
- ONNX runtime for model inference
- FastAPI for server mode (not used by DocuNative)
- Pandas for data handling
- HNSW index library

We only use its embedded local mode, but can't avoid the full install.

---

## Maintenance Checklist

### Monthly

- [ ] Check for security updates: `uv lock --upgrade`
- [ ] Test that UI launches: `make demo`
- [ ] Verify all imports work:
  ```bash
  uv run python -c "import torch, gradio, transformers, chromadb, sentence_transformers, pydantic, fitz, requests, huggingface_hub; print('✅ All imports OK')"
  ```

### Before Major Releases

- [ ] Update Python version if 3.11 EOL approaching
- [ ] Review for breaking changes in ML libraries
- [ ] Test on fresh install (delete `.venv/` first)
- [ ] Update `uv-migration-guide.md` if dependencies change

---

## Additional Resources

- **UV Documentation:** https://docs.astral.sh/uv/
- **PyPI Package Index:** https://pypi.org/
- **Hugging Face Hub:** https://huggingface.co/
- **Gradio Docs:** https://gradio.app/docs/
- **Sentence Transformers:** https://www.sbert.net/
- **ChromaDB Docs:** https://docs.trychroma.com/

---

**Last verified:** March 16, 2026  
**Python version:** 3.11.13  
**UV version:** 0.8.13
