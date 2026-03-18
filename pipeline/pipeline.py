"""
pipeline/pipeline.py
--------------------
Wires all six pipeline modules into a single run() function.

This is the only file that ui/app.py needs to import.
Everything else is an implementation detail.

Issue: #16
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from dataclasses import dataclass
from typing import Optional

import torch

import chromadb

from pipeline.extract import extract_chunks
from pipeline.embed import build_index
from pipeline.retrieve import retrieve
from pipeline.generate import generate_answer, check_server_health
from pipeline.validate import parse_output
from pipeline.nli import nli_validation, aggregate_verdict

logger = logging.getLogger(__name__)

# Return type

@dataclass
class PipelineResult:
    """
    Everything the UI needs to display a complete answer.

    Fields:
        answer:        The model's answer in the user's language.
        source_quote:  The exact clause from the document that supports the answer.
        nli_verdict:   "entailment" / "neutral" / "contradiction" — hallucination check.
        model:         Which model variant was used ("Global" or "Earth".
        parse_success: True if the model followed the Answer:/Source_Quote: format.
        error:         None on success, error message string on failure.
    """
    answer: str
    source_quote: str
    nli_verdict: str
    model: str
    parse_success: bool
    error: Optional[str] = None
    timings: Optional[dict] = None  # per-step latency in seconds
    context_text: str = ""  # concatenated retrieved chunks for UI display

    @property
    def nli_emoji(self) -> str:
        """Emoji badge for the Gradio UI NLI status field."""
        return {
            "entailment":    "✅ Grounded — answer is supported by the document",
            "neutral":       "⚠️ Unverified — could not confirm from source",
            "contradiction": "🚨 Hallucination detected — answer conflicts with document",
        }.get(self.nli_verdict, "❓ Unknown")


# ---------------------------------------------------------------------------
# Module-level collection cache
# We keep the ChromaDB collection in memory between queries on the same document.
# This means embed/index only runs once per upload, not once per question.

_current_collection: Optional[chromadb.Collection] = None
_current_doc_id: Optional[str] = None


# ---------------------------------------------------------------------------
# System diagnostics

def get_system_diagnostics() -> str:
    """
    Returns a human-readable string describing the current OS and compute backend.
    Used by Ali's performance tracking requirement — record what hardware each
    teammate is running on so we can compare latency across devices.
    """
    os_name = platform.system()
    arch = platform.machine()

    if torch.cuda.is_available():
        compute = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        compute = "Apple Silicon (Metal/MPS)"
    else:
        compute = "CPU Only"

    return f"OS: {os_name} {arch} | Compute: {compute}"


# ---------------------------------------------------------------------------
# Core function

def run(
    pdf_path: str,
    question: str,
    model_choice: str = "Global",
    top_k: int = 3,
    force_reindex: bool = False,
) -> PipelineResult:
    """
    Run the full DocuNative pipeline end-to-end.

    Steps:
        1. Extract text chunks from the PDF (extract.py)
        2. Embed chunks into ChromaDB — skipped if same doc already indexed (embed.py)
        3. Retrieve top-k most relevant chunks for the question (retrieve.py)
        4. Generate an answer via llama-server (generate.py)
        5. Parse the Answer:/Source_Quote: output (validate.py)
        6. Check hallucination via mDeBERTa NLI (nli.py)

    Args:
        pdf_path:      Path to the uploaded PDF file.
        question:      User's question in any language.
        model_choice:  "Global" or "Earth" — which Tiny Aya variant to use.
        top_k:         Number of chunks to retrieve. Default 3.
        force_reindex: If True, re-embed the document even if it's already cached.

    Returns:
        PipelineResult with answer, source quote, NLI verdict, and metadata.
    """
    global _current_collection, _current_doc_id

    # --- System diagnostics (for Ali's cross-device perf tracking) ------
    logger.info("💻 SYSTEM: %s", get_system_diagnostics())
    t_pipeline_start = time.time()
    timings = {}

    # --- Step 0: Health check -------------------------------------------
    if not check_server_health():
        return PipelineResult(
            answer="",
            source_quote="",
            nli_verdict="neutral",
            model=model_choice,
            parse_success=False,
            error=(
                "llama-server is not running. "
                "Open Terminal 1 and run: make server-global"
            ),
        )

    # --- Step 1: Extract ------------------------------------------------
    logger.info("Extracting chunks from: %s", pdf_path)
    t0 = time.time()
    try:
        chunks = extract_chunks(pdf_path)
    except Exception as e:
        return _error_result(model_choice, f"PDF extraction failed: {e}")
    timings["extract_s"] = round(time.time() - t0, 3)
    logger.info("⏱️  Extraction:  %.2fs | %d chunks", timings["extract_s"], len(chunks))

    if not chunks:
        return _error_result(model_choice, "No text extracted from PDF. Is it a scanned image?")

    # --- Step 2: Embed (with caching) ------------------------------------
    # Skip re-embedding if we already indexed this exact document.
    # force_reindex=True forces a fresh index (e.g. after a new upload).
    doc_id = pdf_path  # use file path as the document identifier

    if force_reindex or _current_doc_id != doc_id or _current_collection is None:
        logger.info("Building index for: %s (%d chunks)", doc_id, len(chunks))
        t0 = time.time()
        try:
            _current_collection = build_index(chunks, doc_id=doc_id)
            _current_doc_id = doc_id
        except Exception as e:
            return _error_result(model_choice, f"Embedding failed: {e}")
        timings["embed_s"] = round(time.time() - t0, 3)
        logger.info("⏱️  Embedding:   %.2fs", timings["embed_s"])
    else:
        timings["embed_s"] = 0.0  # cached — no re-embedding
        logger.info("Reusing cached index for: %s", doc_id)

    # --- Step 3: Retrieve -----------------------------------------------
    logger.info("Retrieving top-%d chunks for question: %r", top_k, question[:60])
    t0 = time.time()
    try:
        retrieved = retrieve(question, _current_collection, top_k=top_k)
    except Exception as e:
        return _error_result(model_choice, f"Retrieval failed: {e}")
    timings["retrieve_s"] = round(time.time() - t0, 3)
    logger.info("⏱️  Retrieval:   %.2fs | %d chunks returned", timings["retrieve_s"], len(retrieved))

    if not retrieved:
        return PipelineResult(
            answer="No relevant content found in the document for this question.",
            source_quote="N/A",
            nli_verdict="neutral",
            model=model_choice,
            parse_success=True,
            error=None,
        )

    # --- Step 4: Generate -----------------------------------------------
    # Pass chunk text strings directly — generate_answer owns the formatting.
    chunk_strings = [r.text for r in retrieved]

    logger.info("Generating answer with model: %s", model_choice)
    t0 = time.time()
    gen_result = generate_answer(
        question=question,
        chunks=chunk_strings,
        model_choice=model_choice,
    )
    timings["generate_s"] = round(time.time() - t0, 3)
    logger.info("⏱️  Generation:  %.2fs", timings["generate_s"])

    if not gen_result["success"]:
        return _error_result(model_choice, gen_result["error"])

    # --- Step 5: Validate (parse output) --------------------------------
    parsed = parse_output(gen_result["raw_output"])

    # --- Step 6: NLI hallucination check --------------------------------
    t0 = time.time()
    try:
        nli_results = nli_validation(
            list_premises=chunk_strings,
            llm_answer=parsed.answer,
        )
        nli_verdict = aggregate_verdict(nli_results)
    except Exception as e:
        logger.warning("NLI check failed (non-fatal): %s", e)
        nli_verdict = "neutral"  # fail open — don't block the answer
    timings["nli_s"] = round(time.time() - t0, 3)
    logger.info("⏱️  NLI check:   %.2fs | verdict: %s", timings["nli_s"], nli_verdict)

    # --- Total pipeline time --------------------------------------------
    timings["total_s"] = round(time.time() - t_pipeline_start, 3)
    timings["system"] = get_system_diagnostics()
    timings["model"] = model_choice
    timings["question"] = question[:80]
    logger.info("⏱️  TOTAL:        %.2fs | %s", timings["total_s"], timings["system"])

    # --- Save timings to log file ---------------------------------------
    # Each query appends one JSON line to logs/timings.jsonl
    # Share this file with Ali for cross-device performance comparison.
    _log_timings(timings)

    # Join retrieved chunks into a single context string for the UI
    context_text = "\n\n".join(chunk_strings)

    return PipelineResult(
        answer=parsed.answer,
        source_quote=parsed.source_quote,
        nli_verdict=nli_verdict,
        model=model_choice,
        parse_success=parsed.parse_success,
        error=None,
        timings=timings,
        context_text=context_text,
    )


# ---------------------------------------------------------------------------
# Helper

def _log_timings(timings: dict) -> None:
    """
    Append one timing record to logs/timings.jsonl.

    Each line is a JSON object with all step latencies and system info.
    Share logs/timings.jsonl with Ali for cross-device comparison.

    Format:
        {"timestamp": "...", "system": "OS: Darwin arm64 | Compute: MPS",
         "extract_s": 0.12, "embed_s": 3.21, "retrieve_s": 0.08,
         "generate_s": 4.55, "nli_s": 0.43, "total_s": 8.39,
         "model": "Global", "question": "What is the monthly rent?"}
    """
    os.makedirs("logs", exist_ok=True)
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **timings}
    try:
        with open("logs/timings.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning("Could not write timings log: %s", e)


def _error_result(model_choice: str, error_msg: str) -> PipelineResult:
    """Return a PipelineResult representing a clean pipeline failure."""
    logger.error("Pipeline error: %s", error_msg)
    return PipelineResult(
        answer="",
        source_quote="",
        nli_verdict="neutral",
        model=model_choice,
        parse_success=False,
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Standalone test


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Usage: python pipeline/pipeline.py path/to/doc.pdf "Your question here"
    if len(sys.argv) < 3:
        print("Usage: python pipeline/pipeline.py <pdf_path> <question>")
        print("Example: python pipeline/pipeline.py tests/sample_lease_de.pdf 'What is the monthly rent?'")
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "Global"

    print(f"\nRunning DocuNative pipeline...")
    print(f"  PDF:      {pdf_path}")
    print(f"  Question: {question}")
    print(f"  Model:    {model}\n")

    result = run(pdf_path, question, model_choice=model, force_reindex=True)

    if result.error:
        print(f"\n Pipeline failed: {result.error}")
        sys.exit(1)

    print(f"Answer:       {result.answer}")
    print(f"Source Quote: {result.source_quote}")
    print(f"NLI Verdict:  {result.nli_emoji}")
    print(f"Model:        {result.model}")
    print(f"Parse OK:     {result.parse_success}")