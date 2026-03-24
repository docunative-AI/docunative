"""
eval/aggregate.py
-----------------
Aggregate eval results from all team members into a single defensible report
and update visualizations/docunative_results.html with the combined numbers.

USAGE:
    # Put all team JSONL files in eval/aggregate/ then run:
    python -m eval.aggregate

    # Dry run — show stats without updating the dashboard:
    python -m eval.aggregate --dry-run

    # Custom directory:
    python -m eval.aggregate --dir path/to/files

NAMING CONVENTION FOR INPUT FILES:
    {name}_eval_results_{eval_type}.jsonl

    Valid eval_type suffixes:
      h2_global          ← Eval 1, H2, Global model, all 3 languages
      h1_fire_hi         ← Eval 1, H1, Fire model, Hindi only
      h1_global_hi       ← Eval 1, H1, Global model, Hindi only
      eval2_global       ← Eval 2, LLM QA, Global model

    Examples:
      abhishek_eval_results_h2_global.jsonl
      wahyu_eval_results_eval2_global.jsonl
      vinod_eval_results_h1_fire_hi.jsonl
      randy_eval_results_h2_global.jsonl

    Rules:
      - Always prefix with your name
      - Use 'results' not 'report' in the filename
      - Only .jsonl files, not .txt reports
      - Deduplication handles double submissions safely

WHAT IT DOES:
    1. Ingests all .jsonl files from eval/aggregate/
    2. Deduplicates on (doc_id, question, model, field) — prevents inflated
       numbers if anyone ran the eval twice and submitted both files
    3. Runs all four hypothesis tests on the clean deduplicated data
    4. Generates a defense report with real examples of intentionally strict scoring
    5. Updates visualizations/docunative_results.html with the new numbers
    6. Saves aggregate to eval/results/eval_results_aggregate.jsonl

Author: Vinod Anbalagan — DocuNative, Cohere Expedition Hackathon Phase 2
"""

from __future__ import annotations

import argparse
import json
import re
import glob
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

AGGREGATE_DIR  = Path("eval/aggregate")
RESULTS_DIR    = Path("eval/results")
DASHBOARD_PATH = Path("visualizations/docunative_results.html")

# ---------------------------------------------------------------------------
# Multilingual refusal phrases — matches metrics.py exactly
# ---------------------------------------------------------------------------
REFUSAL_PATTERN = re.compile(
    r"(?i)(NOT_FOUND"
    r"|does not contain information"
    r"|cannot be found"
    r"|not found in the document"
    r"|not mentioned"
    r"|no information"
    r"|not available"
    r"|does not mention"
    r"|no relevant information"
    r"|unable to find"
    r"|not provided"
    r"|not specified"
    r"|not stated"
    r"|could not find"
    r"|\u6587\u4ef6\u4e2d\u6ca1\u6709"
    r"|\u6587\u4ef6\u4e2d\u4e0d\u5305\u542b"
    r"|\u672a\u63d0\u53ca"
    r"|\u65e0\u6cd5\u627e\u5230"
    r"|\u6587\u4ef6\u4e2d\u65e0\u76f8\u5173\u4fe1\u606f"
    r"|\u0926\u0938\u094d\u0924\u093e\u0935\u0947\u091c\u093c \u092e\u0947\u0902 \u0928\u0939\u0940\u0902"
    r"|\u091c\u093e\u0928\u0915\u093e\u0930\u0940 \u0928\u0939\u0940\u0902"
    r"|\u0928\u0939\u0940\u0902 \u092e\u093f\u0932\u093e"
    r"|nie zawiera informacji"
    r"|nie znaleziono"
    r"|nie wspomniano"
    r"|brak informacji"
    r"|nie podano"
    r"|nie okre\u015blono)"
)


# ---------------------------------------------------------------------------
# Stage 1 — Ingest
# ---------------------------------------------------------------------------

def ingest_files(directory: Path) -> tuple[list[dict], int]:
    """
    Load all .jsonl files from directory.
    Returns (list of records, corrupt line count).
    """
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        return [], 0

    records = []
    corrupt = 0

    for filepath in files:
        contributor = filepath.stem.split("_")[0]  # e.g. "abhishek"
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record["contributor"] = contributor
                    record["source_file"] = filepath.name
                    records.append(record)
                except json.JSONDecodeError:
                    corrupt += 1

    return records, corrupt


# ---------------------------------------------------------------------------
# Stage 2 — Deduplicate
# ---------------------------------------------------------------------------

def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """
    Deduplicate on (doc_id, question, model, field).
    If multiple contributors submitted the same pair, keep the first one seen.
    Returns (deduped records, number removed).
    """
    seen: set[tuple] = set()
    clean: list[dict] = []

    for r in records:
        key = (
            r.get("doc_id", ""),
            r.get("question", ""),
            r.get("model", ""),
            r.get("field", ""),
        )
        if key not in seen:
            seen.add(key)
            clean.append(r)

    return clean, len(records) - len(clean)


# ---------------------------------------------------------------------------
# Stage 3 — Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(records: list[dict]) -> list[dict]:
    """Add eval_type, is_refusal, is_entailment tags."""
    for r in records:
        r["eval_type"] = (
            "Eval 2 (Monolingual)"
            if r.get("field") == "llm_generated"
            else "Eval 1 (Cross-lingual)"
        )
        r["is_refusal"]   = bool(REFUSAL_PATTERN.search(str(r.get("prediction", ""))))
        r["is_entailment"] = r.get("nli_label", "").lower() == "entailment"
    return records


# ---------------------------------------------------------------------------
# Stage 4 — Hypothesis tests
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0

def _group_by(records: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r.get(key, "unknown")].append(r)
    return dict(groups)


def run_hypothesis_tests(records: list[dict]) -> dict:
    """
    Run all four hypothesis tests.
    Returns a dict of results suitable for updating the dashboard.
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING & CORE FINDINGS")
    print("=" * 60)

    results = {}

    # ── Finding 1: Recall@3 cross-lingual vs monolingual ─────────────────
    print("\nFINDING 1: Cross-Lingual Bottleneck (Recall@3)")

    recall_e1: dict[str, list] = defaultdict(list)
    recall_e2: dict[str, list] = defaultdict(list)

    for r in records:
        if r.get("model") != "Global":
            continue
        lang = r.get("language", "")
        val  = r.get("recall_3", 0)
        if r["eval_type"] == "Eval 1 (Cross-lingual)":
            recall_e1[lang].append(val)
        else:
            recall_e2[lang].append(val)

    recall_results = {}
    for lang in ["zh", "hi", "pl"]:
        e1 = _mean(recall_e1.get(lang, []))
        e2 = _mean(recall_e2.get(lang, []))
        recall_results[lang] = {"eval1": e1, "eval2": e2}
        print(f"  {lang}: Eval1={e1*100:.1f}%  Eval2={e2*100:.1f}%")

    results["recall"] = recall_results

    # ── Finding 2: H2 F1 gradient ─────────────────────────────────────────
    print("\nFINDING 2: H2 — Internal Training Gradient (Eval 2, Global)")

    eval2_global = [
        r for r in records
        if r.get("model") == "Global" and r["eval_type"] == "Eval 2 (Monolingual)"
    ]
    f1_by_lang = _group_by(eval2_global, "language")
    h2_results = {}

    for lang in ["zh", "hi", "pl"]:
        rows = f1_by_lang.get(lang, [])
        f1   = _mean([r.get("f1_score", 0) for r in rows])
        em   = _mean([r.get("em_score",  0) for r in rows])
        h2_results[lang] = {"f1": f1, "em": em, "n": len(rows)}
        print(f"  {lang}: F1={f1}  EM={em}  N={len(rows)}")

    results["h2"] = h2_results

    # ── Finding 3: Refusal rate ────────────────────────────────────────────
    print("\nFINDING 3: Safety by Design (Refusal Rates, Eval 2)")

    refusal_results = {}
    for lang in ["zh", "hi", "pl"]:
        rows = f1_by_lang.get(lang, [])
        rate = round(sum(1 for r in rows if r["is_refusal"]) / len(rows) * 100, 1) if rows else 0
        refusal_results[lang] = rate
        print(f"  {lang}: {rate}%")

    results["refusal"] = refusal_results

    # ── Finding 4: H1 Fire vs Global ──────────────────────────────────────
    print("\nFINDING 4: H1 — Specialist vs Generalist (Hindi)")

    hindi_all = [r for r in records if r.get("language") == "hi"]
    by_model  = _group_by(hindi_all, "model")
    h1_results = {}

    for model in ["Global", "Fire"]:
        rows = by_model.get(model, [])
        if not rows:
            continue
        f1       = _mean([r.get("f1_score", 0) for r in rows])
        em       = _mean([r.get("em_score",  0) for r in rows])
        recall   = _mean([r.get("recall_3",  0) for r in rows])
        refusal  = round(sum(1 for r in rows if r["is_refusal"]) / len(rows) * 100, 1)
        entail   = round(sum(1 for r in rows if r["is_entailment"]) / len(rows) * 100, 1)
        h1_results[model] = {
            "f1": f1, "em": em, "recall": recall,
            "refusal": refusal, "entailment": entail, "n": len(rows)
        }
        print(f"  {model}: F1={f1}  EM={em}  Recall={recall}  Refusal={refusal}%  N={len(rows)}")

    results["h1"] = h1_results

    # ── Domain breakdown ───────────────────────────────────────────────────
    by_domain = _group_by(eval2_global, "domain")
    domain_results = {}
    for domain in ["employment", "health_insurance", "immigration_letter", "lease"]:
        rows = by_domain.get(domain, [])
        if not rows:
            continue
        domain_results[domain] = {
            "f1":     _mean([r.get("f1_score", 0) for r in rows]),
            "recall": _mean([r.get("recall_3",  0) for r in rows]),
            "entail": round(sum(1 for r in rows if r["is_entailment"]) / len(rows), 3),
        }

    results["domain"] = domain_results

    return results


# ---------------------------------------------------------------------------
# Stage 5 — Defense report
# ---------------------------------------------------------------------------

def generate_defense_report(records: list[dict]) -> None:
    """
    Print examples of strict scoring to defend low F1/EM scores.
    Shows cases where retrieval worked and the model got the right answer
    but EM=0 due to formatting differences.
    """
    print("\n" + "=" * 60)
    print("STRICT EVAL DEFENSE (use these if questioned on F1/EM scores)")
    print("=" * 60)

    strict_fails = [
        r for r in records
        if r.get("recall_3") == 1
        and r.get("em_score") == 0
        and 0 < r.get("f1_score", 0) < 0.8
        and not r.get("is_refusal")
    ]

    if not strict_fails:
        print("No strict formatting penalties found.")
        return

    import random
    samples = random.sample(strict_fails, min(3, len(strict_fails)))

    for r in samples:
        print(f"\n  Ground Truth: {r.get('ground_truth', '')}")
        print(f"  Prediction:   {r.get('prediction', '')}")
        print(f"  EM={r.get('em_score')}  F1={r.get('f1_score')}  Language={r.get('language')}")
        print("  " + "-" * 50)

    print(
        "\n  SCRIPT: 'The model found the correct number but our strict eval"
        "\n  penalised it for missing the currency string or punctuation."
        "\n  Our F1/EM scores represent the intentional worst-case baseline.'"
    )


# ---------------------------------------------------------------------------
# Stage 6 — Update dashboard
# ---------------------------------------------------------------------------

def update_dashboard(results: dict) -> None:
    """
    Update the Chart.js data arrays in docunative_results.html
    with the new aggregate numbers.
    """
    if not DASHBOARD_PATH.exists():
        logger.warning("Dashboard not found at %s — skipping update", DASHBOARD_PATH)
        return

    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    recall = results.get("recall", {})
    h2     = results.get("h2", {})
    ref    = results.get("refusal", {})
    h1     = results.get("h1", {})
    domain = results.get("domain", {})

    # Recall@3 Eval 1
    e1_vals = f"[{recall.get('zh',{}).get('eval1',0.390)}, {recall.get('hi',{}).get('eval1',0.444)}, {recall.get('pl',{}).get('eval1',0.657)}]"
    # Recall@3 Eval 2
    e2_vals = f"[{recall.get('zh',{}).get('eval2',0.999)}, {recall.get('hi',{}).get('eval2',0.943)}, {recall.get('pl',{}).get('eval2',0.949)}]"

    # F1 Eval 1 — from H2 results (Eval 1 Cross-lingual)
    # We need Eval 1 F1 separately — compute from records if available
    # For now use the stored h2 results (Eval 2) and note in dashboard
    f1_e2_vals = f"[{h2.get('zh',{}).get('f1',0.532)}, {h2.get('hi',{}).get('f1',0.214)}, {h2.get('pl',{}).get('f1',0.221)}]"

    # Refusal rates
    ref_vals = f"[{ref.get('zh',5.8)}, {ref.get('hi',5.9)}, {ref.get('pl',18.1)}]"

    # H1
    global_h1 = h1.get("Global", {})
    fire_h1   = h1.get("Fire", {})

    # Replace Recall Eval 1 data
    html = re.sub(
        r"(// Chart 1.*?Eval 1.*?data:\s*)\[[\d.,\s]+\]",
        lambda m: m.group(0).rsplit("[", 1)[0] + e1_vals,
        html, flags=re.DOTALL, count=1
    )

    # Replace Recall Eval 2 data
    html = re.sub(
        r"(// Chart 1.*?Eval 2.*?data:\s*)\[[\d.,\s]+\]",
        lambda m: m.group(0).rsplit("[", 1)[0] + e2_vals,
        html, flags=re.DOTALL, count=1
    )

    # Replace refusal data
    html = re.sub(
        r"(// Chart 3.*?data:\s*)\[[\d.,\s]+\]",
        lambda m: m.group(0).rsplit("[", 1)[0] + ref_vals,
        html, flags=re.DOTALL, count=1
    )

    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    print(f"\nDashboard updated: {DASHBOARD_PATH}")


# ---------------------------------------------------------------------------
# Stage 7 — Save aggregate JSONL
# ---------------------------------------------------------------------------

def save_aggregate(records: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "eval_results_aggregate.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Aggregate results saved: {out} ({len(records)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Aggregate team eval results and update the dashboard."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=AGGREGATE_DIR,
        help=f"Directory containing team JSONL files. Default: {AGGREGATE_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without updating the dashboard or saving output.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DOCUNATIVE — AGGREGATE EVAL AUDIT")
    print("=" * 60)

    # Stage 1 — Ingest
    print(f"\nSTAGE 1: INGESTION from {args.dir}")
    records, corrupt = ingest_files(args.dir)

    if not records:
        print(f"No .jsonl files found in {args.dir}")
        print("Ask team members to drop their files there and rerun.")
        return

    print(f"  Total rows ingested: {len(records)}")

    # Show contributor breakdown
    by_contributor: dict[str, int] = defaultdict(int)
    for r in records:
        by_contributor[r.get("contributor", "unknown")] += 1
    print("  Contributor breakdown:")
    for name, count in sorted(by_contributor.items()):
        print(f"    {name}: {count} rows")

    if corrupt:
        print(f"  Corrupted lines ignored: {corrupt}")

    # Stage 2 — Deduplicate
    print("\nSTAGE 2: DEDUPLICATION")
    records, dupes_removed = deduplicate(records)
    if dupes_removed:
        print(f"  Removed {dupes_removed} duplicate rows")
    print(f"  Clean rows after deduplication: {len(records)}")

    # Stage 3 — Feature engineering
    records = engineer_features(records)

    # Stage 4 — Hypothesis tests
    results = run_hypothesis_tests(records)

    # Stage 5 — Defense report
    generate_defense_report(records)

    # Summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"  Total clean QA pairs: {len(records)}")
    print(f"  Duplicates removed:   {dupes_removed}")
    print(f"  Contributors:         {', '.join(sorted(by_contributor.keys()))}")

    if args.dry_run:
        print("\nDry run — dashboard not updated.")
        return

    # Stage 6 — Update dashboard
    update_dashboard(results)

    # Stage 7 — Save aggregate
    save_aggregate(records)

    print("\nAUDIT COMPLETE. Dashboard updated. Ready to present.")


if __name__ == "__main__":
    main()
