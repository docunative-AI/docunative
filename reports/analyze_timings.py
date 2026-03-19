"""
reports/analyze_timings.py
--------------------------
Reads logs/timings.jsonl and generates a clean hardware & ML model
latency report for cross-device performance comparison.
Built to answer the Cohere Mentor Ali's feedback regarding inference bottlenecks.

Usage:
    python -m reports.analyze_timings

Output:
    - Printed dashboard in terminal
    - Saved report at logs/inference_report.txt (share this with Ali)
"""

import json
import time
from collections import defaultdict
from pathlib import Path

LOG_FILE    = Path("logs/timings.jsonl")
REPORT_FILE = Path("logs/inference_report.txt")

def generate_report() -> None:
    if not LOG_FILE.exists():
        print(
            f"No log file found at {LOG_FILE}.\n"
            "Please run a few queries in the UI or terminal first!\n"
            "Every query automatically appends a record to that file."
        )
        return

    # ------
    # Data structures
    #   model_stats[hardware][component] = {"sum": float, "count": int}
    #   e2e_stats[hardware]              = {"sum": float, "count": int}
    # ------
    model_stats = defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "count": 0}))
    e2e_stats   = defaultdict(lambda: {"sum": 0.0, "count": 0})

    total_records = 0
    skipped       = 0

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            total_records += 1
            hw          = data.get("system", "Unknown Hardware")
            aya_variant = data.get("model", "Global")

            # 1. PyMuPDF — text extraction
            model_stats[hw]["1. PyMuPDF (Text Extraction)"]["sum"]   += data.get("extract_s", 0.0)
            model_stats[hw]["1. PyMuPDF (Text Extraction)"]["count"] += 1

            # 2. BGE-M3 — document indexing (only when embedding actually ran)
            # embed_s == 0.0 means the index was served from cache — skip those
            # so the average reflects real embedding latency, not cache hits.
            embed_time = data.get("embed_s", 0.0)
            if embed_time > 0.0:
                model_stats[hw]["2. BAAI/bge-m3 (Doc Indexing)"]["sum"]   += embed_time
                model_stats[hw]["2. BAAI/bge-m3 (Doc Indexing)"]["count"] += 1

            # 3. BGE-M3 — query embedding + ChromaDB retrieval
            model_stats[hw]["3. BAAI/bge-m3 (Query Retrieval)"]["sum"]   += data.get("retrieve_s", 0.0)
            model_stats[hw]["3. BAAI/bge-m3 (Query Retrieval)"]["count"] += 1

            # 4. Tiny Aya generation (labelled with Global/Earth variant)
            aya_label = f"4. Tiny Aya 3.35B ({aya_variant})"
            model_stats[hw][aya_label]["sum"]   += data.get("generate_s", 0.0)
            model_stats[hw][aya_label]["count"] += 1

            # 5. mDeBERTa-v3 hallucination check
            model_stats[hw]["5. mDeBERTa-v3 (NLI Validation)"]["sum"]   += data.get("nli_s", 0.0)
            model_stats[hw]["5. mDeBERTa-v3 (NLI Validation)"]["count"] += 1

            # End-to-end total
            e2e_stats[hw]["sum"]   += data.get("total_s", 0.0)
            e2e_stats[hw]["count"] += 1

    # Build the report
    lines = []

    def log(msg: str = "") -> None:
        """Print and collect output for file saving."""
        print(msg)
        lines.append(msg)

    W = 85  # table width

    log()
    log("=" * W)
    log(" DOCUNATIVE — MODEL INFERENCE & BOTTLENECK REPORT")
    log(f" Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  Records analysed: {total_records}")
    if skipped:
        log(f" ⚠️  Skipped {skipped} malformed line(s).")
    log("=" * W)

    if not model_stats:
        log("No valid timing data found in logs.")
        return

    for hw, components in sorted(model_stats.items()):
        log()
        log(f"HARDWARE PROFILE: {hw}")
        log("-" * W)
        log(f"{'Component / ML Model':<42} | {'Avg Latency':>13} | {'Runs'}")
        log("-" * W)

        for component_name, stats in sorted(components.items()):
            avg = stats["sum"] / stats["count"]
            log(f"{component_name:<42} | {avg:>9.3f}s    | {stats['count']:>5}")

        log("-" * W)

        # Cache hit rate for BGE-M3 indexing
        total_queries = e2e_stats[hw]["count"]
        index_runs    = model_stats[hw].get("2. BAAI/bge-m3 (Doc Indexing)", {}).get("count", 0)
        cache_hits    = total_queries - index_runs
        if total_queries > 0:
            log(
                f"  ℹ️  BGE-M3 Doc Indexing: {index_runs}/{total_queries} queries required "
                f"embedding ({cache_hits} served from cache)"
            )

        e2e_avg = e2e_stats[hw]["sum"] / e2e_stats[hw]["count"]
        log(f"{'⚡ END-TO-END PIPELINE WAIT TIME':<42} | {e2e_avg:>9.3f}s    | {total_queries:>5}")
        log("=" * W)

    # Cross-device speedup summary
    # Use Linux CPU Only as the baseline (worst case / most common cloud deployment)
    e2e_by_hw  = {hw: e2e_stats[hw]["sum"] / e2e_stats[hw]["count"] for hw in e2e_stats}
    linux_cpu_key = next((k for k in e2e_by_hw if "Linux" in k and "CPU Only" in k), None)
    baseline_key  = linux_cpu_key or next((k for k in e2e_by_hw if "CPU Only" in k), None)

    if baseline_key and len(e2e_by_hw) > 1:
        log()
        log("CROSS-DEVICE SPEEDUP (vs Linux CPU baseline — worst case):")
        log("-" * W)
        baseline_avg = e2e_by_hw[baseline_key]
        for hw, avg in sorted(e2e_by_hw.items()):
            if hw == baseline_key:
                log(f"  {hw[:72]:<72} → baseline ({avg:.1f}s)")
            else:
                speedup = baseline_avg / avg
                log(f"  {hw[:72]:<72} → {speedup:.1f}x faster ({avg:.1f}s)")
        log("-" * W)

        # Highlight Metal vs CUDA if both present
        metal_key = next((k for k in e2e_by_hw if "Metal" in k or "MPS" in k), None)
        cuda_key  = next((k for k in e2e_by_hw if "CUDA" in k), None)
        if metal_key and cuda_key:
            metal_avg = e2e_by_hw[metal_key]
            cuda_avg  = e2e_by_hw[cuda_key]
            if metal_avg < cuda_avg:
                ratio = cuda_avg / metal_avg
                log(f"  📌 Apple Silicon (Metal/MPS) outperforms NVIDIA CUDA by {ratio:.1f}x for this workload.")
                log(f"     Metal: {metal_avg:.1f}s  |  CUDA: {cuda_avg:.1f}s")
            else:
                ratio = metal_avg / cuda_avg
                log(f"  📌 NVIDIA CUDA outperforms Apple Silicon (Metal/MPS) by {ratio:.1f}x for this workload.")
                log(f"     CUDA: {cuda_avg:.1f}s  |  Metal: {metal_avg:.1f}s")
        log()

    # Summary note for Ali
    log()
    log("NOTES FOR RESEARCH PAPER:")
    log("  Embed time = 0.0s means ChromaDB index was reused (same doc, follow-up query).")
    log("  Generation time is roughly constant regardless of document length —")
    log("    only top-3 chunks are passed to Tiny Aya, not the full document.")
    log("  NLI (mDeBERTa-v3) now runs on source_quote vs best matching chunk only")
    log("    (not all 3 chunks). This is both faster and more precise — same-language")
    log("    comparison eliminates cross-lingual false contradictions.")
    if baseline_key:
        gen_key_cpu = next(
            (k for k in model_stats.get(baseline_key, {}) if "Tiny Aya" in k), None
        )
        if gen_key_cpu:
            gen_avg = model_stats[baseline_key][gen_key_cpu]["sum"] / model_stats[baseline_key][gen_key_cpu]["count"]
            gen_pct = (gen_avg / e2e_by_hw[baseline_key]) * 100
            log(f"  ⚠️  CPU BOTTLENECK: Generation accounts for {gen_pct:.0f}% of end-to-end latency on Linux CPU.")
            log("    Metal (Apple Silicon) or CUDA (NVIDIA) required for production use.")
    log("  To compare devices: collect logs/timings.jsonl from each teammate")
    log("    and concatenate them before running this script.")
    log()

   
    # Save report to file

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to {REPORT_FILE}")

if __name__ == "__main__":
    generate_report()