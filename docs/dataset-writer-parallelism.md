# Dataset writer parallelism

This document describes the optimisation applied to [`dataset/builder/writer.py`](../dataset/builder/writer.py) (synthetic legal document generation via Cohere or Ollama). The goal is to reduce wall-clock time when generating many documents without changing prompts, validation rules, or JSONL schema.

---

## Table of contents

1. [Summary](#summary)
2. [Problem](#problem)
3. [What changed](#what-changed)
4. [Architecture](#architecture)
5. [Configuration](#configuration)
6. [What stayed the same](#what-stayed-the-same)
7. [Operational notes](#operational-notes)
8. [Tests](#tests)
9. [Related files](#related-files)

---

## Summary

- **Before:** Each `(language, domain, document_index)` job ran **sequentially**; total time was roughly the sum of all LLM call latencies (plus retries).
- **After:** Jobs run in a **`ThreadPoolExecutor`** with **bounded concurrency**. Multiple documents are generated in parallel, subject to a configurable worker limit.
- **Defaults:** **8** concurrent workers for **Cohere**, **1** for **Ollama** (local queue / single-GPU setups).
- **Output:** JSONL files are still written to `dataset/output/{lang}.jsonl`, with rows **sorted** so domain and `doc_idx` order matches the legacy sequential writer (stable diffs and downstream assumptions).

---

## Problem

Document generation is dominated by **network I/O** to Cohere (or Ollama). A full Hindi run is **4 domains × 30 documents = 120** jobs; each call can take tens of seconds when the model emits long legal text. Running strictly one-after-another made full runs take on the order of **an hour**, even though the remote API can serve **multiple concurrent requests** (within account rate limits).

---

## What changed

| Area | Detail |
|------|--------|
| **Execution model** | All tasks are collected, then executed with `concurrent.futures.ThreadPoolExecutor` and `as_completed` for progress updates. |
| **Worker function** | `_generate_one_doc` encapsulates `generate_facts`, the validation/retry loop, and row assembly for one job. |
| **HTTP clients** | Each worker thread uses **`threading.local()`** and `_get_worker_client()` so each thread has its own `cohere.Client` or `ollama.Client`. Sharing a single sync client across threads is avoided. |
| **Ordering** | `_sort_rows_for_language` sorts by `SUPPORTED_DOMAINS` index, then `doc_idx` (from `doc_id`), before writing each language file. |
| **Progress** | `tqdm` is updated from the **main thread** as futures complete (no worker-thread updates to the bar). |
| **Tuning** | `WRITER_MAX_WORKERS` (environment) and `--workers` (CLI) override defaults; effective concurrency is `max(1, n)`. |

---

## Architecture

```text
generate_all_documents
  ├── get_llm_client(...)     # fail-fast config (API key / backend)
  ├── build task list         # (lang, domain, doc_idx) for all combos
  ├── _resolve_max_workers    # CLI > env > defaults
  └── ThreadPoolExecutor
        └── _generate_one_doc (per task)
              ├── _get_worker_client   # thread-local client
              ├── generate_document    # sync chat (unchanged API surface)
              └── validate_document + retries (unchanged logic)
  ├── merge results by language
  ├── _sort_rows_for_language
  └── write JSONL per language
```

---

## Configuration

### Environment (see also [`.env.example`](../.env.example))

| Variable | Role |
|----------|------|
| `WRITER_MAX_WORKERS` | If set to a positive integer, used as max parallel LLM calls when `--workers` is not passed. Empty or unset falls back to defaults below. |
| `COHERE_API_KEY` | Required for Cohere (unless using Ollama). |
| `USE_OLLAMA` | When truthy, backend is Ollama; default worker count is **1**. |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Ollama endpoint and model (unchanged from pre-parallel behaviour). |

### Command line

```bash
# Example: Hindi only, explicit parallelism
python -m dataset.builder.writer --language hi --workers 12
```

`--workers N` takes precedence over `WRITER_MAX_WORKERS`. Values less than `1` are clamped to `1`.

### Default worker counts (code constants)

- Cohere: `_DEFAULT_COHERE_WORKERS = 8`
- Ollama: `_DEFAULT_OLLAMA_WORKERS = 1`

---

## What stayed the same

- **Seeding and facts:** `generate_facts` and YAML seeds are unchanged; determinism per `(lang, domain, doc_idx)` is unchanged.
- **Prompts:** `_build_prompt` text and instructions are unchanged.
- **LLM call:** `generate_document` (Cohere `chat` / Ollama `chat`) and model selection (`COHERE_WRITER_MODEL`, `OLLAMA_MODEL`) are unchanged.
- **Validation:** `validate_document` rules and **up to three attempts** per document (`MAX_DOC_RETRIES = 2`) are unchanged.
- **Row schema:** Each JSONL object still has `doc_id`, `language`, `domain`, `document_text`, `seed`, `facts`.
- **Other pipeline stages:** `qa_factory`, `qa_factory_llm`, `eval`, and the main DocuNative RAG app are **not** modified by this optimisation; only the writer’s **scheduling** of document generation changed.

---

## Operational notes

- **Rate limits:** Raising `--workers` too high may trigger **HTTP 429** from Cohere (trial keys are often capped at **20 calls/minute**). The writer **retries after 429** with exponential backoff (see below). You can still lower `--workers` or `WRITER_MAX_WORKERS` to reduce how often you hit the limit.

### Cohere 429 backoff (`generate_document`)

On `cohere.errors.TooManyRequestsError`, **[Tenacity](https://github.com/jd/tenacity)** (`Retrying` + `stop_after_attempt` + custom `wait`) retries the same `client.chat` call until success or retries are exhausted. Sleep timing still uses the same env-driven exponential backoff, `Retry-After`, and jitter (implemented in `_wait_cohere_429` / `_cohere_rate_limit_delay_seconds`). Configuration (environment only):

| Variable | Default | Meaning |
|----------|---------|---------|
| `COHERE_429_MAX_RETRIES` | `10` | Maximum **retry** attempts after a 429 (not counting the first request). Total attempts = `1 + COHERE_429_MAX_RETRIES`. |
| `COHERE_429_INITIAL_DELAY_SEC` | `10` | Base delay in seconds before the first retry; doubled each retry (`×2`), capped below. |
| `COHERE_429_MAX_DELAY_SEC` | `120` | Upper cap on backoff delay per wait. |

If Cohere returns a **`Retry-After`** header (seconds), the wait is at least `max(backoff, Retry-After)`. A small random jitter is added so parallel workers do not all wake at once.
- **Cost:** Parallelism reduces **elapsed time**, not the **number** of API calls; billing is still per successful generation (retries add extra calls when validation fails).
- **Ollama:** Default **1** worker avoids overloading a single local GPU or Ollama’s internal queue; increase `--workers` only if your machine tolerates concurrent local inference.
- **Reproducibility:** Same seeds and temperature still apply per call; **non-deterministic** LLM sampling means regenerated text can differ run-to-run regardless of parallelism (same as before).

---

## Tests

Automated checks live in [`tests/test_writer_parallel.py`](../tests/test_writer_parallel.py):

- JSONL **domain order** matches `SUPPORTED_DOMAINS` after a parallel run (mocked LLM).
- **Peak concurrent** mocked `generate_document` calls respect `max_workers`.
- `_resolve_max_workers` respects env, CLI override, and `0 → 1` clamping.

Run:

```bash
pytest tests/test_writer_parallel.py -q
```

---

## Related files

| File | Purpose |
|------|---------|
| [`dataset/builder/writer.py`](../dataset/builder/writer.py) | Implementation (parallel pool + Tenacity for Cohere 429) |
| [`dataset/builder/facts.py`](../dataset/builder/facts.py) | Deterministic seed facts |
| [`tests/test_writer_parallel.py`](../tests/test_writer_parallel.py) | Unit tests (mocked) |
| [`.env.example`](../.env.example) | Documents `WRITER_MAX_WORKERS` |

For end-to-end dataset steps (QA generation, eval), see the main [README](../README.md) “Running the Evaluation” section.
