"""
eval/evaluate.py
----------------
Step 4 of the dataset pipeline: run the full DocuNative pipeline against
the pilot QA dataset and compute evaluation metrics.

Tests both research hypotheses:
  H1 — Does Earth model outperform Global on regional languages (hi, sw)?
  H2 — Does accuracy degrade German (high) -> Hindi (medium) -> Swahili (low)?

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

    # Full H1 + H2 run (both models)
    python -m eval.evaluate --qa dataset/output/qa_pairs.jsonl \\
                            --docs dataset/output \\
                            --model both

    # With Command A LLM judge
    python -m eval.evaluate --qa dataset/output/qa_pairs.jsonl \\
                            --docs dataset/output \\
                            --model both --llm-judge

Issue: #23
Author: Vinod Anbalagan
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

from eval.metrics import (
    calculate_f1_score,
    calculate_recall_3,
    per_language_breakdown,
)
from pipeline.pipeline import run, PipelineResult
from pipeline.validate import ParsedOutput

logger = logging.getLogger(__name__)


# Paths

RESULTS_DIR = Path("eval/results")


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


def find_document_path(docs_dir: Path, doc_id: str) -> Path | None:
    """
    Find the generated document file for a given doc_id.

    writer.py saves documents inside de.jsonl / hi.jsonl / sw.jsonl.
    We extract the document text and write it to a temp .txt file so
    the pipeline can read it.

    doc_id format: "de_lease_0", "hi_employment_3", etc.
    """
    lang = doc_id.split("_")[0]  # "de", "hi", "sw"
    jsonl_path = docs_dir / f"{lang}.jsonl"

    if not jsonl_path.exists():
        logger.warning("Document JSONL not found: %s", jsonl_path)
        return None

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            if record.get("doc_id") == doc_id:
                # Write document text to a temp .txt file for the pipeline
                tmp_path = Path(f"/tmp/docunative_eval_{doc_id}.txt")
                tmp_path.write_text(record["document_text"], encoding="utf-8")
                # Also return chunk count for document-size breakdown
                return tmp_path

    logger.warning("doc_id %s not found in %s", doc_id, jsonl_path)
    return None


def get_document_chunk_count(docs_dir: Path, doc_id: str) -> int:
    """
    Return approximate chunk count for a document based on its text length.
    Used for per-document-size breakdown.
    Buckets: small (<5 chunks), medium (5-10), large (>10)
    """
    lang = doc_id.split("_")[0]
    jsonl_path = docs_dir / f"{lang}.jsonl"

    if not jsonl_path.exists():
        return 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            if record.get("doc_id") == doc_id:
                text = record.get("document_text", "")
                # Approximate chunks: CHUNK_SIZE_CHARS = 1200
                return max(1, len(text) // 1200)
    return 0


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

    # Token F1
    f1 = calculate_f1_score(parsed, gt)

    # Recall@3 — use retrieved_chunks (true retrieval metric)
    # NOT source_quote or context_text — those are model outputs, not retrieval results
    recall = calculate_recall_3(result.retrieved_chunks, gt)

    # NLI verdict from pipeline
    verdict = result.nli_verdict  # entailment / neutral / contradiction

    # Command A LLM judge (optional — only when --llm-judge flag is set)
    judge_score     = None
    judge_reasoning = None
    if cohere_client is not None:
        judge_result    = llm_judge_score(question, gt, result.answer, cohere_client)
        judge_score     = judge_result.get("score", -1)
        judge_reasoning = judge_result.get("reasoning", "")

    return {
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
        "recall_3":      recall,
        "nli_label":     verdict,        # standardised key — matches metrics.py
        "parse_ok":      result.parse_success,
        # LLM judge (None if not requested)
        "judge_score":   judge_score,
        "judge_reasoning": judge_reasoning,
        # Timing
        "elapsed_s":     elapsed,
    }


# Main eval loop


def run_evaluation(
    qa_path: Path,
    docs_dir: Path,
    model_choice: str = "Global",
    limit: int | None = None,
    cohere_client=None,
) -> list[dict]:
    """
    Run evaluation over all QA pairs for one model.

    Args:
        qa_path:        Path to qa_pairs.jsonl
        docs_dir:       Directory containing de.jsonl, hi.jsonl, sw.jsonl
        model_choice:   "Global" or "Earth"
        limit:          If set, only evaluate this many pairs (quick testing)
        cohere_client:  Cohere client for LLM judge (None = skip judge)

    Returns:
        List of result dicts, one per evaluated QA pair.
    """
    qa_pairs = load_qa_pairs(qa_path)
    if limit:
        qa_pairs = qa_pairs[:limit]
        logger.info("Limiting to %d pairs for quick test", limit)

    results = []
    total   = len(qa_pairs)

    for i, pair in enumerate(qa_pairs, 1):
        logger.info(
            "[%d/%d] %s | %s | model=%s",
            i, total,
            pair["doc_id"],
            pair["question"][:50],
            model_choice,
        )
        result = run_single_eval(pair, docs_dir, model_choice, cohere_client)
        if result:
            results.append(result)

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

        log(f"{'Language':<12} | {'Resource':>10} | {'Avg F1':>8} | {'Recall@3':>9} | {'Entailment%':>12} | {'N':>5}")
        log("-" * W)

        resource = {"de": "High", "hi": "Medium", "sw": "Low"}
        for lang in ["de", "hi", "sw"]:
            if lang not in breakdown:
                continue
            b = breakdown[lang]
            log(
                f"{lang:<12} | {resource.get(lang, '?'):>10} | {b['avg_f1']:>8.3f} | "
                f"{b['recall_at_3']:>9.3f} | {b['entailment_percentage']:>12.3f} | "
                f"{b['total_questions']:>5}"
            )

        log()
        # H2 verdict
        if all(l in breakdown for l in ["de", "hi", "sw"]):
            de_f1 = breakdown["de"]["avg_f1"]
            hi_f1 = breakdown["hi"]["avg_f1"]
            sw_f1 = breakdown["sw"]["avg_f1"]
            if de_f1 >= hi_f1 >= sw_f1:
                log(f"  H2 ({model}): CONFIRMED — de ({de_f1}) ≥ hi ({hi_f1}) ≥ sw ({sw_f1})")
            else:
                log(f"  H2 ({model}): NOT CONFIRMED — de ({de_f1}), hi ({hi_f1}), sw ({sw_f1})")

    # ── H1: Global vs Earth ──────────────────────────────────────────────
    if "Global" in models and "Earth" in models:
        log()
        log("=" * W)
        log(" H1: GLOBAL vs EARTH — Regional Specialist Advantage")
        log("=" * W)

        global_bd = per_language_breakdown([r for r in all_results if r["model"] == "Global"])
        earth_bd  = per_language_breakdown([r for r in all_results if r["model"] == "Earth"])

        log(f"{'Language':<12} | {'Global F1':>10} | {'Earth F1':>9} | {'Winner':>10}")
        log("-" * W)

        for lang in ["de", "hi", "sw"]:
            if lang not in global_bd or lang not in earth_bd:
                continue
            gf1    = global_bd[lang]["avg_f1"]
            ef1    = earth_bd[lang]["avg_f1"]
            winner = "Earth" if ef1 > gf1 else ("Tie" if ef1 == gf1 else "Global")
            log(f"{lang:<12} | {gf1:>10.3f} | {ef1:>9.3f} | {winner:>10}")

        log()
        earth_wins = sum(
            1 for lang in ["hi", "sw"]
            if lang in global_bd and lang in earth_bd
            and earth_bd[lang]["avg_f1"] > global_bd[lang]["avg_f1"]
        )
        if earth_wins == 2:
            log("  H1: CONFIRMED — Earth outperforms Global on both hi and sw")
        elif earth_wins == 1:
            log("  H1: PARTIAL — Earth outperforms Global on one of hi/sw")
        else:
            log("  H1: NOT CONFIRMED — Global matches or beats Earth on hi and sw")

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
            for lang in ["de", "hi", "sw"]:
                if lang not in lang_judge:
                    continue
                scores = lang_judge[lang]
                acc    = round(sum(scores) / len(scores), 3)
                log(f"{lang:<12} | {acc:>15.1%} | {len(scores):>5}")

        log()
        log("  Note: Judge accuracy > Token F1 means the model is correct")
        log("  but uses different phrasing than the ground truth string.")
        log("  Judge accuracy close to Token F1 means the ground truth is reliable.")

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
        help="Directory containing de.jsonl, hi.jsonl, sw.jsonl from writer.py",
    )
    parser.add_argument(
        "--model",
        choices=["Global", "Earth", "both"],
        default="Global",
        help="Which model to evaluate. 'both' runs H1 + H2 comparison.",
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
        default=RESULTS_DIR / "eval_report.txt",
        help="Where to save the human-readable report.",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help=(
            "Use Command A as an LLM judge to validate answers. "
            "Requires COHERE_API_KEY in .env. Adds latency per question."
        ),
    )
    args = parser.parse_args()

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

    # Run evaluation
    all_results: list[dict] = []
    models = ["Global", "Earth"] if args.model == "both" else [args.model]

    for model in models:
        print(f"\nRunning evaluation with model: {model}")
        results = run_evaluation(
            qa_path=args.qa,
            docs_dir=args.docs,
            model_choice=model,
            limit=args.limit,
            cohere_client=cohere_client,
        )
        all_results.extend(results)

    if not all_results:
        print("No results — check that llama-server is running and documents exist.")
        print("Start server: make server-global")
        return

    # Save raw results (one line per QA pair)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "eval_results.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nRaw results saved to {raw_path}")

    # Generate human-readable report
    generate_report(all_results, args.output)


if __name__ == "__main__":
    main()
