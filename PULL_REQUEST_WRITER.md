# Pull Request: Dataset writer.py — Cohere & Ollama document synthesis

## What does this PR do?

Implements **Step 2** of the dataset pipeline: take structured facts from `facts.py` (Issue #28) and generate realistic legal document text via **Cohere API** (production) or **Ollama** (local testing). Output is JSONL per language for downstream synthetic QA evaluation.

**Key features:**
1. **`dataset/builder/writer.py`** — Generates documents from facts using Cohere (Aya Expanse 32B) or local Ollama
2. **Dual backend** — Cohere for production (API credits); Ollama for testing pipeline without spending credits (e.g. `gemma3:4b`, `gpt-oss:20b`)
3. **Variable document length** — 1–5 pages per document (seed-based) for robust evaluation
4. **JSONL output** — One file per language: `dataset/output/de.jsonl`, `hi.jsonl`, `sw.jsonl` with `doc_id`, `language`, `domain`, `document_text`, `seed`, `facts`
5. **CLI** — `--test`, `--domain`, `--language`, `--ollama`, `--ollama-model` for flexible runs

**Scale:** 4 domains × 3 languages × 10 documents = **120 documents** (or 12 with `--test`)

Closes #17. Blocks #20, #23.

---

## How to test it

### Prerequisites

- **Cohere:** Set `COHERE_API_KEY` in `.env` (copy from `.env.example`)
- **Ollama (optional):** Install Ollama, then e.g. `ollama pull gemma3:4b`

### Local test (Ollama, no API cost)

```bash
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
uv run python dataset/builder/writer.py --test --ollama --ollama-model gemma3:4b
```

- Expected: 12 documents generated, progress bar, 3 JSONL files under `dataset/output/`
- Check: `head -1 dataset/output/de.jsonl | python -c "import sys,json; d=json.load(sys.stdin); print(d['doc_id'], d['language'], len(d['document_text']))"`

### Production run (Cohere)

```bash
# Ensure .env has COHERE_API_KEY and USE_OLLAMA=false
uv run python dataset/builder/writer.py
```

- Generates 120 documents; output in `dataset/output/de.jsonl`, `hi.jsonl`, `sw.jsonl`.

### CLI help

```bash
uv run python dataset/builder/writer.py --help
```

---

## Checklist

- [x] Implementation matches Issue #17 deliverables (writer.py, JSONL output, document length variability)
- [x] Dependencies added to `pyproject.toml` (cohere, ollama, python-dotenv, tqdm, pyyaml)
- [x] Config template (`.env.example`) and `.gitignore` already ignores `.env`
- [x] Ollama path tested with `--test --ollama` (pipeline runs; JSONL structure verified)
- [x] Error handling: failed documents are skipped and logged; run continues and prints success/failure summary

### Optional

- [ ] Merge or depend on `dataset/seed-facts` so `facts.py` and `dataset/seeds/*.yaml` are in tree (writer imports them; branch may have them copied in)

---

## Files changed

### Created
- **`dataset/builder/writer.py`** — Main implementation: `get_llm_client()`, `generate_document()`, `generate_all_documents()`, CLI
- **`dataset/output/.gitkeep`** — Ensures output directory exists in repo
- **`.env.example`** — Template for `COHERE_API_KEY`, `USE_OLLAMA`, `OLLAMA_MODEL`, `OLLAMA_BASE_URL`

### Modified
- **`pyproject.toml`** — Added: `cohere`, `ollama`, `python-dotenv`, `tqdm`, `pyyaml`

### Unchanged
- **`.gitignore`** — Already contains `.env`; no change needed

---

## Output format (JSONL)

Each line in `dataset/output/{de,hi,sw}.jsonl` is one JSON object:

```json
{
  "doc_id": "de_lease_0",
  "language": "de",
  "domain": "lease",
  "document_text": "**MIETVERTRAG**\n\nzwischen...",
  "seed": 2744148744117987467,
  "facts": { "monthly_rent": "1350 EUR", "_language": "de", ... }
}
```

Used later for synthetic QA: `document_text` is the RAG source; `facts` provide ground-truth answers for evaluation.

---

## Breaking changes

None. New code under `dataset/builder/` and new optional deps; existing pipeline and UI unchanged.

---

**Ready for review.**
