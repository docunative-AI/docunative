"""
eval/evaluate.py
----------------
Step 4 of the dataset pipeline: run the full DocuNative pipeline against
the pilot QA dataset and compute evaluation metrics.

Tests both research hypotheses:
  H1 — Does Fire model (South Asian specialist) outperform Global on Hindi?
  H2 — Does accuracy degrade Chinese (high) -> Hindi (medium) -> Polish (medium-low)?

Also includes:
  - Per-domain breakdown (Olena's suggestion)
  - Per-document-size breakdown (Olena's suggestion)
  - Command A LLM-as-judge layer for dataset validation

Usage:
    # Quick test on 10 pairs
    python -m eval.evaluate --qa dataset/output/qa_pairs.jsonl \\
                            --docs dataset/output \\
                            --model Global --limit 10

    # Full H2 run (Global model only)
    python -m eval.evaluate --qa dataset/output/qa_pairs.jsonl \\
                            --docs dataset/output \\
                            --model Global

    # Full H2 run (Global model, all languages)
    python -m eval.evaluate --qa dataset/output/qa_pairs.jsonl \\
                            --docs dataset/output \\
                            --model Global

    # H1 run (Fire vs Global on Hindi only)
    python -m eval.evaluate --qa dataset/output/qa_pairs.jsonl \\
                            --docs dataset/output \\
                            --model Fire --language hi

    # With Command A LLM judge
    python -m eval.evaluate --qa dataset/output/qa_pairs.jsonl \\
                            --docs dataset/output \\
                            --model both --llm-judge

    # Separate output files per step (eval_results_<LABEL>.jsonl, eval_report_<LABEL>.txt)
    python -m eval.evaluate --qa dataset/output/qa_pairs.jsonl \\
                            --docs dataset/output --model Global --run-name h2

    # Polish only for Eval 2 LLM — keep existing zh/hi in eval_results_eval2-llm.jsonl
    python -m eval.evaluate --qa dataset/output/qa_pairs_llm.jsonl \\
                            --docs dataset/output --model Global --run-name eval2-llm \\
                            --language pl --merge

Issue: #23
Author: Vinod Anbalagan
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from eval.metrics import (
    calculate_f1_score,
    calculate_exact_match,
    calculate_recall_3,
    per_language_breakdown,
)
from pipeline.pipeline import run, PipelineResult
from pipeline.validate import ParsedOutput

logger = logging.getLogger(__name__)


# Paths

RESULTS_DIR = Path("eval/results")


def sanitize_run_name(name: str) -> str:
    """Turn a user label into a safe filename fragment (letters, digits, - _)."""
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def resolve_eval_outputs(
    run_name: str | None, output_override: Path | None
) -> tuple[Path, Path]:
    """
    Default paths for JSONL report and text report.
    With --run-name, both files get a matching suffix so steps do not overwrite.
    --output overrides only the text report path.
    """
    if run_name:
        safe = sanitize_run_name(run_name)
        if not safe:
            raise SystemExit(
                "Error: --run-name must contain at least one letter, digit, -, or _."
            )
        raw_path = RESULTS_DIR / f"eval_results_{safe}.jsonl"
        report_path = (
            output_override
            if output_override is not None
            else RESULTS_DIR / f"eval_report_{safe}.txt"
        )
        return raw_path, report_path
    raw_path = RESULTS_DIR / "eval_results.jsonl"
    report_path = (
        output_override
        if output_override is not None
        else RESULTS_DIR / "eval_report.txt"
    )
    return raw_path, report_path


# Document helpers


def load_qa_pairs(qa_path: Path) -> list[dict]:
    """Load QA pairs from a JSONL file."""
    pairs = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    logger.info("Loaded %d QA pairs from %s", len(pairs), qa_path)
    return pairs


def load_eval_results_excluding_language(path: Path, exclude_lang: str) -> list[dict]:
    """Keep JSONL rows whose language is not ``exclude_lang`` (incremental eval for one language)."""
    if not path.exists():
        return []
    kept: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("language") != exclude_lang:
                kept.append(row)
    return kept


# ---------------------------------------------------------------------------
# Document cache — load all documents into memory once at startup
# instead of re-reading the JSONL file for every single QA pair.
# For 3,600 pairs across 360 documents this saves 3,600 file opens.

_doc_cache: dict[str, str] = {}  # doc_id -> document_text


def _load_doc_cache(docs_dir: Path) -> None:
    """Load all documents from JSONL files into memory once."""
    global _doc_cache
    if _doc_cache:
        return  # already loaded
    for lang in ["zh", "hi", "pl"]:
        jsonl_path = docs_dir / f"{lang}.jsonl"
        if not jsonl_path.exists():
            continue
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    _doc_cache[record["doc_id"]] = record["document_text"]
    logger.info("Loaded %d documents into memory cache", len(_doc_cache))


def find_document_path(docs_dir: Path, doc_id: str) -> Path | None:
    """
    Find the generated document file for a given doc_id.

    writer.py saves documents inside zh.jsonl / hi.jsonl / pl.jsonl.
    We extract the document text and write it to a temp .txt file so
    the pipeline can read it.

    doc_id format: "de_lease_0", "hi_employment_3", etc.

    Uses _doc_cache to avoid re-reading JSONL files on every call.
    """
    _load_doc_cache(docs_dir)  # no-op if already loaded

    text = _doc_cache.get(doc_id)
    if text is None:
        logger.warning("doc_id %s not found in document cache", doc_id)
        return None

    tmp_path = Path(f"/tmp/docunative_eval_{doc_id}.txt")
    tmp_path.write_text(text, encoding="utf-8")
    return tmp_path


def get_document_chunk_count(docs_dir: Path, doc_id: str) -> int:
    """
    Return approximate chunk count for a document based on its text length.
    Used for per-document-size breakdown.
    Buckets: small (<5 chunks), medium (5-10), large (>10)
    Uses _doc_cache — no JSONL file read needed.
    """
    _load_doc_cache(docs_dir)
    text = _doc_cache.get(doc_id, "")
    return max(1, len(text) // 1200) if text else 0


def size_bucket(chunk_count: int) -> str:
    """Bucket a chunk count into small / medium / large."""
    if chunk_count <= 4:
        return "small (1-4 chunks)"
    elif chunk_count <= 9:
        return "medium (5-9 chunks)"
    else:
        return "large (10+ chunks)"


# Command A LLM judge


def llm_judge_score(
    question: str,
    ground_truth: str,
    prediction: str,
    client,
) -> dict:
    """
    Use Command A as an LLM judge to score the model's answer.

    The judge returns:
      - score: 0 (wrong) / 1 (correct)
      - reasoning: short explanation

    This is the LLM-as-judge layer for research paper credibility.
    Runs only when --llm-judge flag is passed.

    Args:
        question:     The question asked
        ground_truth: The correct answer from seed facts
        prediction:   DocuNative's answer
        client:       Cohere client instance

    Returns:
        Dict with keys: score (int), reasoning (str)
    """
    prompt = f"""You are evaluating a document QA system. 
    
Question: {question}
Ground truth answer: {ground_truth}
System's answer: {prediction}

Is the system's answer correct or approximately correct given the ground truth?
Reply with ONLY a JSON object in this exact format:
{{"score": 1, "reasoning": "brief explanation"}}
or
{{"score": 0, "reasoning": "brief explanation"}}

score 1 = correct or approximately correct (minor phrasing differences are OK)
score 0 = wrong, missing, or hallucinated"""

    try:
        response = client.chat(
            model="command-a-03-2025",
            message=prompt,
            temperature=0.0,
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        logger.warning("LLM judge failed for question %r: %s", question[:40], e)
        return {"score": -1, "reasoning": f"judge error: {e}"}


def get_cohere_client():
    """Return a Cohere client using COHERE_API_KEY from environment."""
    try:
        import cohere
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY not set. Add it to .env or set it in your environment."
            )
        return cohere.Client(api_key=api_key)
    except ImportError:
        raise ImportError("cohere package not installed. Run: uv add cohere")


# Single eval runner


def run_single_eval(
    qa_pair: dict,
    docs_dir: Path,
    model_choice: str,
    cohere_client=None,
    *,
    save_retrieval: str | None = None,
) -> dict | None:
    """
    Run one QA pair through the full pipeline and score it.

    Returns a result dict with all scores, or None if the document
    couldn't be found or the pipeline failed.
    """
    doc_id   = qa_pair["doc_id"]
    question = qa_pair["question"]
    gt       = qa_pair["answer"]      # ground truth answer from seed facts
    language = qa_pair["language"]
    domain   = qa_pair["domain"]
    field    = qa_pair.get("field", "unknown")

    # Find the document
    doc_path = find_document_path(docs_dir, doc_id)
    if doc_path is None:
        logger.warning("Skipping %s — document not found", doc_id)
        return None

    # Get chunk count for document-size breakdown
    chunk_count = get_document_chunk_count(docs_dir, doc_id)
    bucket      = size_bucket(chunk_count)

    # Run the full pipeline
    t0 = time.time()
    result: PipelineResult = run(
        pdf_path=str(doc_path),
        question=question,
        model_choice=model_choice,
        force_reindex=True,  # always fresh for eval — no stale cache
    )
    elapsed = round(time.time() - t0, 2)

    if result.error:
        logger.warning("Pipeline error for %s: %s", doc_id, result.error)
        return None

    # Build ParsedOutput for metrics
    parsed = ParsedOutput(
        answer=result.answer,
        source_quote=result.source_quote,
        parse_success=result.parse_success,
        raw_output="",
    )

    # Language code for language-aware tokenization (critical for Chinese)
    lang = qa_pair.get("language", "")

    # Token F1 — language-aware (character-level for zh, word-level for others)
    f1 = calculate_f1_score(parsed, gt, lang=lang)

    # Exact Match — harsher factual extraction metric for academic reporting
    em = calculate_exact_match(parsed, gt, lang=lang)

    # Recall@3 — use retrieved_chunks (true retrieval metric)
    # NOT source_quote or context_text — those are model outputs, not retrieval results
    recall = calculate_recall_3(result.retrieved_chunks, gt, lang=lang)

    # NLI verdict from pipeline
    verdict = result.nli_verdict  # entailment / neutral / contradiction

    # Command A LLM judge (optional — only when --llm-judge flag is set)
    judge_score     = None
    judge_reasoning = None
    if cohere_client is not None:
        judge_result    = llm_judge_score(question, gt, result.answer, cohere_client)
        judge_score     = judge_result.get("score", -1)
        judge_reasoning = judge_result.get("reasoning", "")

    row = {
        # Identifiers
        "doc_id":        doc_id,
        "language":      language,
        "domain":        domain,
        "field":         field,
        "model":         model_choice,
        "size_bucket":   bucket,
        "chunk_count":   chunk_count,
        # Inputs
        "question":      question,
        "ground_truth":  gt,
        "prediction":    result.answer,
        "source_quote":  result.source_quote,
        # Scores
        "f1_score":      f1,
        "em_score":      em,
        "recall_3":      recall,
        "nli_label":     verdict,        # standardised key — matches metrics.py
        "parse_ok":      result.parse_success,
        # LLM judge (None if not requested)
        "judge_score":   judge_score,
        "judge_reasoning": judge_reasoning,
        # Timing
        "elapsed_s":     elapsed,
        "ttft_ms":       (result.timings or {}).get("ttft_ms", 0),
        "tpot_ms":       (result.timings or {}).get("tpot_ms", 0),
        "tokens_per_s":  (result.timings or {}).get("tokens_per_s", 0),
        "generate_s":    (result.timings or {}).get("generate_s", 0),
    }

    # Optional: persist top-k retrieved chunk texts for error analysis (large JSONL if mode=all)
    chunks = list(result.retrieved_chunks or [])
    if save_retrieval == "all":
        row["retrieved_chunks"] = chunks
    elif save_retrieval == "failure" and int(recall) == 0:
        row["retrieved_chunks"] = chunks

    return row


# Main eval loop


def run_evaluation(
    qa_path: Path,
    docs_dir: Path,
    model_choice: str = "Global",
    limit: int | None = None,
    language_filter: str | None = None,
    cohere_client=None,
    save_retrieval: str | None = None,
) -> list[dict]:
    """
    Run evaluation over all QA pairs for one model.

    Args:
        qa_path:         Path to qa_pairs.jsonl
        docs_dir:        Directory containing zh.jsonl, hi.jsonl, pl.jsonl
        model_choice:    "Global", "Earth", or "Fire"
        limit:           If set, only evaluate this many pairs (quick testing)
        language_filter: If set, only evaluate pairs for this language (e.g. "hi" for H1)
        cohere_client:   Cohere client for LLM judge (None = skip judge)

    Returns:
        List of result dicts, one per evaluated QA pair.

    ``save_retrieval``: ``None`` | ``\"failure\"`` | ``\"all\"`` — if set, rows may
    include ``retrieved_chunks`` (top-k chunk strings from the retriever). See ``--save-retrieval`` on the CLI.
    """
    qa_pairs = load_qa_pairs(qa_path)
    if language_filter:
        qa_pairs = [p for p in qa_pairs if p.get("language") == language_filter]
        logger.info("Filtered to %d pairs for language=%s", len(qa_pairs), language_filter)
    if limit:
        qa_pairs = qa_pairs[:limit]
        logger.info("Limiting to %d pairs for quick test", limit)

    # Load all documents into memory once — avoids 3,600 JSONL file opens
    _load_doc_cache(docs_dir)

    results = []
    total   = len(qa_pairs)

    # Progress bar — tqdm if available, simple counter fallback
    try:
        from tqdm import tqdm
        pair_iter = tqdm(
            enumerate(qa_pairs, 1),
            total=total,
            desc=f"Eval [{model_choice}]",
            unit="pair",
            dynamic_ncols=True,
        )
    except ImportError:
        pair_iter = enumerate(qa_pairs, 1)

    for i, pair in pair_iter:
        logger.debug(
            "[%d/%d] %s | %s | model=%s",
            i, total,
            pair["doc_id"],
            pair["question"][:50],
            model_choice,
        )
        result = run_single_eval(
            pair, docs_dir, model_choice, cohere_client, save_retrieval=save_retrieval
        )
        if result:
            results.append(result)

        # Update tqdm postfix with running F1 average
        try:
            if results and hasattr(pair_iter, 'set_postfix'):
                avg_f1 = round(sum(r['f1_score'] for r in results) / len(results), 3)
                lang_counts = {}
                for r in results:
                    lang_counts[r['language']] = lang_counts.get(r['language'], 0) + 1
                pair_iter.set_postfix(avg_f1=avg_f1, langs=lang_counts)
        except Exception:
            pass

    logger.info(
        "Completed %d/%d pairs for model=%s",
        len(results), total, model_choice,
    )
    return results


# Breakdown helpers (Olena's suggestions)


def per_domain_breakdown(results: list[dict]) -> dict:
    """
    Group results by domain and return avg F1, Recall@3, and NLI distribution.

    Returns dict keyed by domain name, same structure as per_language_breakdown.
    """
    if not results:
        return {}

    grouped: dict[str, list] = defaultdict(list)
    for r in results:
        grouped[r.get("domain", "unknown")].append(r)

    breakdown = {}
    for domain, items in grouped.items():
        total       = len(items)
        avg_f1      = round(sum(r.get("f1_score", 0.0) for r in items) / total, 3)
        recall_at_3 = round(sum(r.get("recall_3", 0) for r in items) / total, 3)
        entailment  = round(
            sum(1 for r in items if r.get("nli_label", "") == "entailment") / total, 3
        )
        breakdown[domain] = {
            "avg_f1":               avg_f1,
            "recall_at_3":          recall_at_3,
            "entailment_percentage": entailment,
            "total_questions":      total,
        }

    return breakdown


def per_size_breakdown(results: list[dict]) -> dict:
    """
    Group results by document size bucket and return avg F1 and Recall@3.

    Size buckets: small (1-4 chunks), medium (5-9 chunks), large (10+ chunks).
    Answers Olena's question: does accuracy drop for longer documents?
    """
    if not results:
        return {}

    grouped: dict[str, list] = defaultdict(list)
    for r in results:
        grouped[r.get("size_bucket", "unknown")].append(r)

    breakdown = {}
    for bucket, items in grouped.items():
        total       = len(items)
        avg_f1      = round(sum(r.get("f1_score", 0.0) for r in items) / total, 3)
        recall_at_3 = round(sum(r.get("recall_3", 0) for r in items) / total, 3)
        breakdown[bucket] = {
            "avg_f1":          avg_f1,
            "recall_at_3":     recall_at_3,
            "total_questions": total,
        }

    return breakdown


def per_field_breakdown(results: list[dict]) -> dict:
    """
    Group results by fact field and return avg F1.
    Shows which question types are hardest (e.g. monthly_rent vs pets_allowed).
    """
    if not results:
        return {}

    grouped: dict[str, list] = defaultdict(list)
    for r in results:
        grouped[r.get("field", "unknown")].append(r)

    return {
        field: {
            "avg_f1":          round(sum(r.get("f1_score", 0.0) for r in items) / len(items), 3),
            "total_questions": len(items),
        }
        for field, items in grouped.items()
    }


# Report generation


def generate_report(
    all_results: list[dict],
    output_path: Path,
) -> None:
    """
    Generate a human-readable evaluation report.

    Covers:
    - H1: Global vs Earth comparison
    - H2: Language resource degradation
    - Per-domain breakdown
    - Per-document-size breakdown
    - Command A LLM judge summary (if available)
    """
    lines = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    W = 85

    log("=" * W)
    log(" DOCUNATIVE — EVALUATION REPORT")
    log(f" Total results: {len(all_results)}")
    log("=" * W)

    models = sorted({r["model"] for r in all_results})

    # ── Per-model: H2 language breakdown ────────────────────────────────
    for model in models:
        model_results = [r for r in all_results if r["model"] == model]

        log()
        log(f"MODEL: {model}  ({len(model_results)} QA pairs)")
        log("-" * W)

        breakdown = per_language_breakdown(model_results)

        log(f"{'Language':<12} | {'Resource':>14} | {'Avg F1':>8} | {'Avg EM':>7} | {'Recall@3':>9} | {'Refusal%':>9} | {'Entail%':>8} | {'N':>5}")
        log("-" * W)

        resource = {"zh": "High (1.9%)", "hi": "Medium (1.7%)", "pl": "Medium-low (1.4%)"}
        for lang in ["zh", "hi", "pl"]:
            if lang not in breakdown:
                continue
            b = breakdown[lang]
            log(
                f"{lang:<12} | {resource.get(lang, '?'):>14} | {b['avg_f1']:>8.3f} | "
                f"{b.get('avg_em', 0):>7.3f} | {b['recall_at_3']:>9.3f} | "
                f"{b.get('refusal_rate', 0):>9.3f} | {b['entailment_percentage']:>8.3f} | "
                f"{b['total_questions']:>5}"
            )

        log()
        # H2 verdict
        if all(l in breakdown for l in ["zh", "hi", "pl"]):
            zh_f1 = breakdown["zh"]["avg_f1"]
            hi_f1 = breakdown["hi"]["avg_f1"]
            pl_f1 = breakdown["pl"]["avg_f1"]
            if zh_f1 >= hi_f1 >= pl_f1:
                log(f"  H2 ({model}): CONFIRMED — zh ({zh_f1}) ≥ hi ({hi_f1}) ≥ pl ({pl_f1})")
                log(f"  Internal training proportion gradient (1.9% / 1.7% / 1.4%) predicts performance")
            else:
                log(f"  H2 ({model}): NOT CONFIRMED — zh ({zh_f1}), hi ({hi_f1}), pl ({pl_f1})")
                log(f"  Tiny Aya's internal balancing appears effective for document QA")

    # ── H1: Global vs Fire on Hindi ──────────────────────────────────────
    if "Global" in models and "Fire" in models:
        log()
        log("=" * W)
        log(" H1: GLOBAL vs FIRE — South Asian Specialist Advantage on Hindi")
        log("=" * W)

        # H1 is evaluated on Hindi only — Fire is the South Asian specialist
        global_hi = [r for r in all_results if r["model"] == "Global" and r["language"] == "hi"]
        fire_hi   = [r for r in all_results if r["model"] == "Fire"   and r["language"] == "hi"]

        if global_hi and fire_hi:
            global_f1 = round(sum(r["f1_score"] for r in global_hi) / len(global_hi), 3)
            fire_f1   = round(sum(r["f1_score"] for r in fire_hi)   / len(fire_hi),   3)
            winner    = "Fire" if fire_f1 > global_f1 else ("Tie" if fire_f1 == global_f1 else "Global")

            log(f"{'Language':<12} | {'Global F1':>10} | {'Fire F1':>9} | {'Winner':>10}")
            log("-" * W)
            log(f"{'hi':<12} | {global_f1:>10.3f} | {fire_f1:>9.3f} | {winner:>10}")
            log()
            if fire_f1 > global_f1:
                log(f"  H1: CONFIRMED — Fire ({fire_f1}) outperforms Global ({global_f1}) on Hindi")
            elif fire_f1 == global_f1:
                log(f"  H1: TIE — Fire and Global perform equally on Hindi ({fire_f1})")
            else:
                log(f"  H1: NOT CONFIRMED — Global ({global_f1}) outperforms Fire ({fire_f1}) on Hindi")
        else:
            log("  H1: insufficient data — run with --model Fire --language hi")

    # ── Per-domain breakdown (Olena's suggestion) ────────────────────────
    log()
    log("=" * W)
    log(" PER-DOMAIN BREAKDOWN")
    log("=" * W)

    for model in models:
        model_results = [r for r in all_results if r["model"] == model]
        domain_bd     = per_domain_breakdown(model_results)

        log()
        log(f"Model: {model}")
        log(f"{'Domain':<25} | {'Avg F1':>8} | {'Recall@3':>9} | {'Entailment%':>12} | {'N':>5}")
        log("-" * W)

        for domain in sorted(domain_bd.keys()):
            b = domain_bd[domain]
            log(
                f"{domain:<25} | {b['avg_f1']:>8.3f} | {b['recall_at_3']:>9.3f} | "
                f"{b['entailment_percentage']:>12.3f} | {b['total_questions']:>5}"
            )

    # ── Per-document-size breakdown (Olena's suggestion) ─────────────────
    log()
    log("=" * W)
    log(" PER-DOCUMENT-SIZE BREAKDOWN")
    log("=" * W)
    log(" (Does accuracy drop for longer documents?)")

    for model in models:
        model_results = [r for r in all_results if r["model"] == model]
        size_bd       = per_size_breakdown(model_results)

        log()
        log(f"Model: {model}")
        log(f"{'Size Bucket':<25} | {'Avg F1':>8} | {'Recall@3':>9} | {'N':>5}")
        log("-" * W)

        for bucket in ["small (1-4 chunks)", "medium (5-9 chunks)", "large (10+ chunks)"]:
            if bucket not in size_bd:
                continue
            b = size_bd[bucket]
            log(f"{bucket:<25} | {b['avg_f1']:>8.3f} | {b['recall_at_3']:>9.3f} | {b['total_questions']:>5}")

    # ── Per-field breakdown ───────────────────────────────────────────────
    log()
    log("=" * W)
    log(" PER-FIELD BREAKDOWN (hardest question types)")
    log("=" * W)

    for model in models:
        model_results = [r for r in all_results if r["model"] == model]
        field_bd      = per_field_breakdown(model_results)

        log()
        log(f"Model: {model}")
        log(f"{'Field':<40} | {'Avg F1':>8} | {'N':>5}")
        log("-" * W)

        # Sort by avg_f1 ascending — show hardest fields first
        for field, b in sorted(field_bd.items(), key=lambda x: x[1]["avg_f1"]):
            log(f"{field:<40} | {b['avg_f1']:>8.3f} | {b['total_questions']:>5}")

    # ── Command A LLM judge summary ──────────────────────────────────────
    judge_results = [r for r in all_results if r.get("judge_score") is not None and r["judge_score"] >= 0]
    if judge_results:
        log()
        log("=" * W)
        log(" COMMAND A LLM JUDGE SUMMARY")
        log(" (Oracle validation — Command A as ground truth checker)")
        log("=" * W)

        for model in models:
            model_judge = [r for r in judge_results if r["model"] == model]
            if not model_judge:
                continue
            total      = len(model_judge)
            correct    = sum(1 for r in model_judge if r["judge_score"] == 1)
            judge_acc  = round(correct / total, 3)

            log()
            log(f"Model: {model}  |  Judge accuracy: {judge_acc:.1%}  ({correct}/{total} correct)")

            # Per-language judge accuracy
            lang_judge: dict[str, list] = defaultdict(list)
            for r in model_judge:
                lang_judge[r["language"]].append(r["judge_score"])

            log(f"{'Language':<12} | {'Judge Accuracy':>15} | {'N':>5}")
            log("-" * W)
            for lang in ["zh", "hi", "pl"]:
                if lang not in lang_judge:
                    continue
                scores = lang_judge[lang]
                acc    = round(sum(scores) / len(scores), 3)
                log(f"{lang:<12} | {acc:>15.1%} | {len(scores):>5}")

        log()
        log("  Note: Judge accuracy > Token F1 means the model is correct")
        log("  but uses different phrasing than the ground truth string.")
        log("  Judge accuracy close to Token F1 means the ground truth is reliable.")

    # ── TTFT / TPOT timing summary ────────────────────────────────────────
    timed = [r for r in all_results if r.get("ttft_ms", 0) > 0]
    if timed:
        log()
        log("=" * W)
        log(" TTFT / TPOT GENERATION TIMING (from llama-server timings object)")
        log(" Requested by Ali Edalati (Cohere mentor)")
        log("=" * W)
        log()

        ttft_vals = [r["ttft_ms"]     for r in timed]
        tpot_vals = [r["tpot_ms"]     for r in timed]
        tok_vals  = [r["tokens_per_s"] for r in timed]

        def _mean(vals): return round(sum(vals) / len(vals), 1) if vals else 0
        def _min(vals):  return round(min(vals), 1) if vals else 0
        def _max(vals):  return round(max(vals), 1) if vals else 0

        log(f"  Queries with timing data: {len(timed)}")
        log()
        log(f"  TTFT (Time to First Token — prompt prefill):")
        log(f"    Mean: {_mean(ttft_vals)}ms  |  Min: {_min(ttft_vals)}ms  |  Max: {_max(ttft_vals)}ms")
        log()
        log(f"  TPOT (Time Per Output Token — decode speed):")
        log(f"    Mean: {_mean(tpot_vals)}ms/token  |  Min: {_min(tpot_vals)}ms/token  |  Max: {_max(tpot_vals)}ms/token")
        log()
        log(f"  Throughput:")
        log(f"    Mean: {_mean(tok_vals)} tokens/sec  |  Peak: {_max(tok_vals)} tokens/sec")
        log()

        # Bottleneck analysis
        avg_ttft = _mean(ttft_vals)
        avg_gen  = _mean([r["generate_s"] * 1000 for r in timed])
        if avg_gen > 0:
            log(f"  Bottleneck: Prefill {avg_ttft:.0f}ms ({avg_ttft/avg_gen*100:.0f}%) | Decode {avg_gen - avg_ttft:.0f}ms ({(avg_gen-avg_ttft)/avg_gen*100:.0f}%)")
            if avg_ttft > (avg_gen - avg_ttft):
                log("  → Prefill-bound. Consider prompt caching or shorter context.")
            else:
                log("  → Decode-bound. Speculative decoding recommended for Phase 3.")

    log()
    log("=" * W)

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to {output_path}")


# CLI


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run DocuNative evaluation pipeline (evaluate.py, Issue #23)."
    )
    parser.add_argument(
        "--qa",
        type=Path,
        required=True,
        help="Path to qa_pairs.jsonl from qa_factory.py",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        required=True,
        help="Directory containing zh.jsonl, hi.jsonl, pl.jsonl from writer.py",
    )
    parser.add_argument(
        "--model",
        choices=["Global", "Earth", "Fire", "both"],
        default="Global",
        help="Which model to evaluate. 'both' runs Global + Fire. Fire requires --language hi.",
    )
    parser.add_argument(
        "--language",
        choices=["zh", "hi", "pl"],
        default=None,
        help="Filter evaluation to a specific language. Use with --model Fire for H1 (hi).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N QA pairs for quick testing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to save the human-readable report. "
            "Default: eval/results/eval_report.txt, or eval_report_<run-name>.txt if --run-name is set."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        metavar="LABEL",
        help=(
            "Label for this run — writes eval_results_<LABEL>.jsonl and eval_report_<LABEL>.txt "
            "(safe characters only) so different pipeline steps do not overwrite each other."
        ),
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help=(
            "Use Command A as an LLM judge to validate answers. "
            "Requires COHERE_API_KEY in .env. Adds latency per question."
        ),
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "With --language: load existing eval_results JSONL for this --run-name (or default path), "
            "drop rows for that language, append new results. Use after adding Polish (or any lang) "
            "without re-running zh/hi."
        ),
    )
    parser.add_argument(
        "--save-retrieval",
        choices=["off", "failure", "all"],
        default="off",
        metavar="MODE",
        help=(
            "Include top-k retrieved chunk strings in each JSONL row. "
            "'failure' = only when recall_3 is 0 (retrieval miss). "
            "'all' = every row (very large files). Default: off."
        ),
    )
    args = parser.parse_args()

    if args.merge and not args.language:
        print("Error: --merge requires --language (e.g. --language pl --merge --run-name eval2-llm)")
        return

    # Validate inputs
    if not args.qa.exists():
        print(f"Error: QA file not found: {args.qa}")
        print("Run: python -m dataset.builder.qa_factory --full")
        return
    if not args.docs.exists():
        print(f"Error: Docs directory not found: {args.docs}")
        print("Run: python -m dataset.builder.writer --test")
        return

    # Set up Command A judge if requested
    cohere_client = None
    if args.llm_judge:
        print("Setting up Command A LLM judge...")
        cohere_client = get_cohere_client()
        print("Command A judge ready.")

    raw_path, report_path = resolve_eval_outputs(args.run_name, args.output)

    save_retrieval = None if args.save_retrieval == "off" else args.save_retrieval

    merged_kept: list[dict] = []
    if args.merge:
        if raw_path.exists():
            merged_kept = load_eval_results_excluding_language(raw_path, args.language)
            print(
                f"Merge: keeping {len(merged_kept)} existing rows "
                f"(excluding language={args.language}) from {raw_path.name}"
            )
        else:
            print(f"Merge: no existing file at {raw_path} — writing only new language rows")

    # Run evaluation (only --language pairs when filter is set)
    new_results: list[dict] = []
    models = ["Global", "Fire"] if args.model == "both" else [args.model]

    for model in models:
        print(f"\nRunning evaluation with model: {model}")
        results = run_evaluation(
            qa_path=args.qa,
            docs_dir=args.docs,
            model_choice=model,
            limit=args.limit,
            language_filter=args.language,
            cohere_client=cohere_client,
            save_retrieval=save_retrieval,
        )
        new_results.extend(results)

    if not new_results and not merged_kept:
        print("No results — check that llama-server is running and documents exist.")
        print("Start server: make server-global")
        return

    if not new_results and merged_kept:
        print(
            "\nWARNING: No new eval rows (server or empty filter?) — rewriting file from kept rows only."
        )

    all_results = merged_kept + new_results

    # Save raw results (one line per QA pair)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nRaw results saved to {raw_path} ({len(all_results)} rows)")

    # Generate human-readable report
    generate_report(all_results, report_path)


if __name__ == "__main__":
    main()
