"""
eval/precompute_embeddings.py
------------------------------
Pre-compute BGE-M3 embeddings for all 360 synthetic documents ONCE
before running evaluate.py. Saves ~39 minutes of embedding time on
Metal and hours on CPU by avoiding re-embedding the same documents
on every eval query.

Without this:  360 docs × 6.5s embedding × 10 queries = 39 min wasted
With this:     360 docs embedded once = ~2 min total, then 0.05s cache loads

Usage:
    python -m eval.precompute_embeddings --docs dataset/output/
    python -m eval.precompute_embeddings --docs dataset/output/ --force

Then run evaluate.py as normal — it will auto-detect and use the cache.

Issue: perf/optimizations
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("eval/.embedding_cache")
LANGUAGES  = ["zh", "hi", "pl"]  # zh=Chinese, hi=Hindi, pl=Polish — internal gradient 1.9/1.7/1.4%


def precompute_all(docs_dir: Path, force: bool = False) -> None:
    """
    Embed all documents in docs_dir and save to eval/.embedding_cache/.
    Each document is saved as a .pkl file keyed by doc_id.

    Args:
        docs_dir: Path to dataset/output/ containing de.jsonl, hi.jsonl, sw.jsonl
        force:    If True, re-embed even if cache file already exists
    """
    # Import here so the module is importable without triggering model load
    from pipeline.embed import _get_model
    from pipeline.extract import chunk_text

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Cache directory: %s", CACHE_DIR.resolve())

    model = _get_model()  # loads BGE-M3 once

    total_docs = 0
    total_cached = 0
    total_skipped = 0
    t_start = time.time()

    for lang in LANGUAGES:
        jsonl_path = docs_dir / f"{lang}.jsonl"
        if not jsonl_path.exists():
            logger.warning("Missing: %s — skipping", jsonl_path)
            continue

        with open(jsonl_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        logger.info("Processing %d %s documents...", len(records), lang.upper())

        for record in records:
            doc_id     = record["doc_id"]
            cache_file = CACHE_DIR / f"{doc_id}.pkl"
            total_docs += 1

            if cache_file.exists() and not force:
                logger.debug("  skip %s (cached)", doc_id)
                total_skipped += 1
                continue

            # Extract chunks from document text
            doc_text = record.get("document_text", "")
            if not doc_text.strip():
                logger.warning("  skip %s (empty document text)", doc_id)
                continue

            chunks = chunk_text(doc_text)
            if not chunks:
                logger.warning("  skip %s (no chunks extracted)", doc_id)
                continue

            # Embed all chunks
            t0 = time.time()
            embeddings = model.encode(
                chunks,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            )
            elapsed = time.time() - t0

            # Save to cache
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "doc_id":     doc_id,
                    "chunks":     chunks,
                    "embeddings": embeddings,
                    "language":   lang,
                }, f)

            total_cached += 1
            logger.info(
                "  cached %s: %d chunks in %.2fs",
                doc_id, len(chunks), elapsed
            )

    elapsed_total = time.time() - t_start
    logger.info(
        "\nDone. %d/%d documents cached (%d skipped) in %.1fs",
        total_cached, total_docs, total_skipped, elapsed_total
    )
    logger.info(
        "Estimated eval embedding time saved: %.0f min → ~%ds",
        (total_docs * 6.5) / 60,
        total_docs * 0  # cache loads are ~0.05s each
    )


def load_cached_embeddings(doc_id: str) -> dict | None:
    """
    Load pre-computed embeddings for a document from cache.
    Returns None if not cached (evaluate.py falls back to live embedding).

    Returns:
        {
            "doc_id":     str,
            "chunks":     list[str],
            "embeddings": np.ndarray,   shape (n_chunks, 1024)
            "language":   str,
        }
        or None if not in cache.
    """
    cache_file = CACHE_DIR / f"{doc_id}.pkl"
    if not cache_file.exists():
        return None
    with open(cache_file, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute BGE-M3 embeddings for all synthetic eval documents."
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=Path("dataset/output/"),
        help="Path to dataset/output/ directory (default: dataset/output/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed even if cache already exists",
    )
    args = parser.parse_args()

    if not args.docs.exists():
        logger.error("docs directory not found: %s", args.docs)
        raise SystemExit(1)

    logger.info("Pre-computing embeddings from: %s", args.docs.resolve())
    logger.info("Force re-embed: %s", args.force)
    precompute_all(args.docs, force=args.force)
