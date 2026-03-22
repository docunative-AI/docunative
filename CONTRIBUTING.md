# DocuNative — Extending the Research

This guide is for anyone who wants to add languages, add models, or extend the evaluation pipeline after the Cohere Expedition Hackathon Phase 2 (March 2026).

The architecture is designed to be extended. Adding a new language is a configuration change across 6 files. Adding a new model is 3 steps. Everything else follows automatically.

---

## Table of Contents

1. [Adding a New Language](#1-adding-a-new-language)
2. [Adding a New Model](#2-adding-a-new-model)
3. [Extending the Evaluation Pipeline](#3-extending-the-evaluation-pipeline)
4. [Recommended Next Languages](#4-recommended-next-languages)
5. [Known Limitations to Fix](#5-known-limitations-to-fix)

---

## 1. Adding a New Language

Adding a language requires changes to 6 files. Use this checklist. The example below adds **Swahili (sw)**.

### Step 1 — `dataset/builder/writer.py`

Add to `SUPPORTED_LANGUAGES` and `LANG_NAMES`:

```python
SUPPORTED_LANGUAGES = ["zh", "hi", "pl", "sw"]  # add your language code

LANG_NAMES = {
    "zh": "Chinese (Simplified)",
    "hi": "Hindi",
    "pl": "Polish",
    "sw": "Swahili",  # add this
}
```

Also update the default fallback at the bottom:
```python
facts.get("_language", "zh")  # keep zh as default
```

> ⚠️ **Important for Swahili and other low-resource languages:** Aya Expanse 32B (the document generator) does NOT natively support Swahili. You will need a different document generator. Options:
> - **Gemini 1.5 Pro** via Google AI API — supports Swahili natively
> - **GPT-4o** via OpenAI API — supports Swahili natively
> - Replace the `generate_document()` call in `writer.py` with your chosen API
>
> Chinese, Hindi, Polish, German, French, Spanish, Arabic, Japanese, Korean, Indonesian, and 12 other languages are natively supported by Aya Expanse 32B — no changes needed for those.

---

### Step 2 — `dataset/builder/qa_factory.py`

Add to `SUPPORTED_LANGUAGES`:

```python
SUPPORTED_LANGUAGES = ["zh", "hi", "pl", "sw"]
```

Question templates in Eval 1 are in English by design (cross-lingual evaluation). No translation needed here.

---

### Step 3 — `dataset/builder/qa_factory_llm.py`

Add to `SUPPORTED_LANGUAGES`, `LANG_NAMES`, and CLI choices:

```python
SUPPORTED_LANGUAGES = ["zh", "hi", "pl", "sw"]

LANG_NAMES = {
    "zh": "Chinese (Simplified)",
    "hi": "Hindi",
    "pl": "Polish",
    "sw": "Swahili",
}
```

And in the `argparse` section:
```python
parser.add_argument("--language", choices=["zh", "hi", "pl", "sw"], ...)
```

---

### Step 4 — `pipeline/validate.py`

Add "not found" phrases in the new language to `is_answer_missing()`. These are the phrases the model uses when it cannot find an answer:

```python
# Swahili
"haipatikani katika hati",
"hati haina taarifa hii",
"sijapata jibu",
"haijulikani",
```

Search for the `no_answer_phrases` list in `validate.py` and add your phrases there.

---

### Step 5 — `pipeline/generate.py`

Add stop tokens for the new language. These prevent the model from generating beyond the answer format:

```python
STOP_TOKENS = {
    "zh": ["问题", "回答", "[END]"],
    "hi": ["प्रश्न", "उत्तर", "[END]"],
    "pl": ["Pytanie", "Odpowiedź", "[END]"],
    "sw": ["Swali", "Jibu", "[END]"],  # add this
}
```

---

### Step 6 — `eval/evaluate.py`

Add resource label for the H2 report:

```python
RESOURCE_LABELS = {
    "zh": "High (1.9%)",
    "hi": "Medium (1.7%)",
    "pl": "Medium-low (1.4%)",
    "sw": "Low (X.X%)",  # check Tiny Aya Appendix A for the exact %
}
```

Also add to CLI choices:
```python
parser.add_argument("--language", choices=["zh", "hi", "pl", "sw", ...])
```

---

### Step 7 — `ui/app.py`

Add the language to the UI translation dictionary and the dropdown:

```python
UI_TRANSLATIONS = {
    ...
    "Swahili": {
        "upload_label": "📄 Pakia Hati ya Kisheria",
        "question_label": "Swali Lako",
        "question_ph": "mf. Amana ni kiasi gani?",
        "ask_btn": "Uliza DocuNative",
        "answer_label": "Jibu (kwa lugha yako)",
    },
}
```

And in the dropdown:
```python
ui_lang_dropdown = gr.Dropdown(
    choices=["English", "Chinese", "Hindi", "Polish", "Swahili"],
    ...
)
```

Also add a NOT_FOUND message:
```python
NOT_FOUND_MESSAGES = {
    ...
    "Swahili": "ℹ️ Taarifa hii haikupatikana katika hati iliyopakiwa.",
}
```

---

### Step 8 — Generate documents and run eval

```bash
# Generate documents (120 per language, 4 domains × 30 each)
python -m dataset.builder.writer --language sw

# Regenerate all QA pairs (includes new language)
python -m dataset.builder.qa_factory --full

# Regenerate LLM QA pairs
python -m dataset.builder.qa_factory_llm --language sw

# Precompute embeddings
python -m eval.precompute_embeddings --docs dataset/output/

# Run tests
pytest -q

# Run eval
make server-global
python -m eval.evaluate \
  --qa dataset/output/qa_pairs.jsonl \
  --docs dataset/output \
  --model Global
```

---

## 2. Adding a New Model

Adding a model requires 3 steps.

### Step 1 — Download the GGUF

Place the model file in `models/weights/`:

```bash
# Example: downloading a new Tiny Aya variant
huggingface-cli download CohereForAI/aya-expanse-8b-GGUF \
  aya-expanse-8b-q4_k_m.gguf \
  --local-dir models/weights/
```

### Step 2 — Add a Makefile target

Open `Makefile` and add a new server target:

```makefile
server-aya8b:
    $(LLAMA_SERVER) -m models/weights/aya-expanse-8b-q4_k_m.gguf \
        -ngl 99 --flash-attn auto --cache-prompt \
        -c 4096 -t $(CPU_THREADS) --port 8080
```

### Step 3 — Add to UI

In `ui/app.py`, add to the model radio button:

```python
model_radio = gr.Radio(
    choices=["Global", "Fire", "Aya8B"],
    ...
)
```

### Running H1-style evaluation with new model

```bash
# Terminal 1 — start the new model server
make server-aya8b

# Terminal 2 — run eval on all languages
python -m eval.evaluate \
  --qa dataset/output/qa_pairs.jsonl \
  --docs dataset/output \
  --model Aya8B

# Or compare against Global on a single language
python -m eval.evaluate \
  --qa dataset/output/qa_pairs.jsonl \
  --docs dataset/output \
  --model Aya8B --language hi
```

Back up results before switching models:
```bash
cp eval/results/eval_report.txt eval/results/eval_report_aya8b_all.txt
cp eval/results/eval_results.jsonl eval/results/eval_results_aya8b_all.jsonl
```

---

## 3. Extending the Evaluation Pipeline

### Adding more domains

Domains are defined in `dataset/builder/facts.py`. Each domain has a `facts` dictionary with seed values and a `template` string for document generation.

To add a new domain (e.g. `bank_account`):

1. Add a new entry to `DOMAIN_FACTS` in `facts.py`
2. Add the domain to `DOMAINS` list in `writer.py`
3. Add question templates in `qa_factory.py` for the new fields
4. Regenerate all documents and QA pairs

### Running Eval 3 (MKQA) properly

The MKQA evaluation (`eval/eval_mkqa.py`) was attempted but had low coverage (~4%) due to natural language queries not mapping well to Wikipedia article titles.

To improve coverage:
1. Use the Wikipedia **search** API instead of the **summary** API — it handles NL queries better
2. Or use a named entity extractor (spaCy, Flair) to extract the topic before lookup
3. Hindi is not in MKQA's 26 languages — use TyDi QA or IndicQA for Hindi external validation

```bash
# Current MKQA eval (works for zh and pl only)
python -m eval.eval_mkqa --limit 300

# Results saved to:
# eval/results/eval_mkqa_report.txt
# eval/results/eval_mkqa_results.jsonl
```

### Comparing multiple models in one report

Currently evaluate.py runs one model at a time. To compare two models:

```bash
# Run model 1
make server-global
python -m eval.evaluate --qa dataset/output/qa_pairs.jsonl \
  --docs dataset/output --model Global
cp eval/results/eval_results.jsonl eval/results/eval_results_global.jsonl

# Run model 2
make server-fire
python -m eval.evaluate --qa dataset/output/qa_pairs.jsonl \
  --docs dataset/output --model Fire --language hi
cp eval/results/eval_results.jsonl eval/results/eval_results_fire_hi.jsonl

# Both JSONL files can be visualised together in visualizations/docunative_results.html
```

---

## 4. Recommended Next Languages

These are the most impactful extensions for the research, in priority order:

| Language | Code | Tiny Aya % | Why |
|---|---|---|---|
| Swahili | sw | ~1.0% | True low-resource — would confirm the gradient Olena proposed. Needs Gemini/GPT-4 for doc generation. |
| Yoruba | yo | ~0.5% | Very low-resource African language. Strong research case. |
| French | fr | 1.7% | Same % as Hindi — tests script effect vs training % |
| German | de | 1.7% | Same % as Hindi — Latin script, different linguistic family |
| Arabic | ar | ~1.5% | Right-to-left script — tests chunking and tokenization assumptions |
| Japanese | ja | ~1.6% | CJK character set, similar to Chinese — good comparison |

**For the Swahili extension specifically:**
- Use Gemini 1.5 Pro for document generation (replace `writer.py` Cohere call)
- AfriQA benchmark exists for external validation (unlike MKQA which lacks hi/sw)
- Would extend the internal gradient from 3 to 4 points

---

## 5. Known Limitations to Fix

| Issue | Location | Fix |
|---|---|---|
| Hindi immigration Recall@3 = 2% | `pipeline/retrieve.py` | BGE-M3 struggles with Devanagari immigration vocabulary — try domain-specific chunking |
| Polish Refusal Rate 18.1% | `pipeline/generate.py` | Prompt engineering — add Polish few-shot examples to reduce over-refusal |
| MKQA 4% coverage | `eval/eval_mkqa.py` | Replace Wikipedia summary API with search API + entity extractor |
| No query translation | `pipeline/pipeline.py` | Add a local translation step (NLLB-200 via llama.cpp) before retrieval |
| Single document size bucket | `eval/evaluate.py` | Need longer documents to test size effect — current docs are all "small" |

---

## Quick Reference — File Map

```
dataset/
  builder/
    writer.py           ← Document generation. Add languages here first.
    qa_factory.py       ← Eval 1 template QA. Add SUPPORTED_LANGUAGES.
    qa_factory_llm.py   ← Eval 2 LLM QA. Add SUPPORTED_LANGUAGES + LANG_NAMES.
    facts.py            ← Domain facts and seed values. Add new domains here.

pipeline/
  extract.py            ← PDF/TXT extraction and chunking
  embed.py              ← BGE-M3 embedding
  retrieve.py           ← ChromaDB retrieval. DEFAULT_TOP_K = 2.
  generate.py           ← Tiny Aya generation via llama-server. Add stop tokens.
  validate.py           ← Output parsing. Add not-found phrases.
  nli.py                ← mDeBERTa NLI trust badge
  pipeline.py           ← Orchestrates all 6 steps

eval/
  evaluate.py           ← Main eval runner. Add language choices + resource labels.
  metrics.py            ← F1, EM, Recall@3, Refusal Rate. Add per-language tokenization.
  eval_mkqa.py          ← Eval 3 MKQA. Currently zh + pl only (Hindi not in MKQA).
  precompute_embeddings.py ← Pre-embed all documents. Run before eval.

ui/
  app.py                ← Gradio UI. Add language to dropdown + translations.

models/
  pull_models.py        ← Downloads GGUF files from HuggingFace
  weights/              ← GGUF files live here (gitignored)

Makefile                ← Server targets. Add new model target here.
```

---

## Citation

If you use or extend DocuNative in your research:

```
DocuNative: Privacy-First Multilingual Document QA at the Edge
Team DocuNative — Cohere Expedition Hackathon, March 2026
Olena Bugaiova, Vinod Anbalagan, Randy Christian Saputra,
Sudhanshu Mishra, Wahyu Dwi Nugraha, Paarth Sharma

External benchmark: Longpre, S., Lu, Y., & Daiber, J. (2021).
MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain
Question Answering. TACL, 9, 1389–1406.
https://aclanthology.org/2021.tacl-1.82/
```
