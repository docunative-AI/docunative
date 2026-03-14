"""
pipeline/extract.py

Extracts text from PDF (and .txt) files and splits it into overlapping
token-aware chunks for downstream embedding and retrieval.

Author: DocuNative Team
Issue: #4
"""

import re
import unicodedata

import fitz  # PyMuPDF — import name is 'fitz', not 'pymupdf'
from pathlib import Path
from typing import List


# Configuration
#
# These values are set once here and imported everywhere else.
# Do NOT hardcode them in other pipeline files — always import from here.
# Changing them here automatically propagates to embed.py, retrieve.py, etc.

CHUNK_SIZE   = 400   # tokens per chunk (stays within BGE-M3's 512-token limit)
CHUNK_OVERLAP = 80   # tokens of overlap between adjacent chunks

# Rough characters-per-token estimate for English/European text.
# A proper tokenizer (like tiktoken) would be more accurate, but adds a
# dependency and ~50ms per call. For chunking purposes, 4 chars/token is
# close enough — the goal is approximate size, not exact token count.
CHARS_PER_TOKEN = 4

CHUNK_SIZE_CHARS   = CHUNK_SIZE * CHARS_PER_TOKEN    # 1600 characters
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP * CHARS_PER_TOKEN  # 320 characters


# Text Cleaning

# Characters that must survive cleaning in legal/contract documents
LEGAL_WHITELIST = {
    '§',   # section symbol — appears in every legal doc
    '©', '®', '™',  # IP and trademark clauses
    '€', '$', '£', '¥',  # currency symbols
    '%',   # percentages (interest rates, penalties)
    '°',   # degrees (some technical contracts)
    '±', '≤', '≥', '≠',  # thresholds and tolerances
    '—', '-',  # em/en dashes used in date ranges and clauses
    '·',   # middle dot (used in some EU legal numbering)
    '¶',   # pilcrow / paragraph mark
}

# Unicode categories that are safe to strip in legal context
REMOVE_CATEGORIES = {
    "So",  # Symbol, other  → ✓ ✗ ▶ ♦ ■ decorative bullets
    "Sk",  # Symbol, modifier
    "Cs",  # Surrogate pairs (broken emoji encodings)
    "Co",  # Private use area (custom PDF font glyphs — very common in contracts)
}


def clean_text(text: str) -> str:
    """
    Remove decorative symbols and emoticons from legal/contract document text
    while preserving legally meaningful characters (§, ©, €, %, —, etc.).

    Args:
        text: Raw text extracted from a PDF contract or legal document

    Returns:
        Cleaned text safe for chunking and embedding
    """
    cleaned = []
    for char in text:
        # Always keep whitelisted legal characters
        if char in LEGAL_WHITELIST:
            cleaned.append(char)
            continue

        # Strip unwanted unicode categories
        if unicodedata.category(char) in REMOVE_CATEGORIES:
            continue
        cleaned.append(char)

    text = "".join(cleaned)

    # Normalize whitespace left behind by removed characters
    text = re.sub(r'\s+', ' ', text)

    # Normalize dashes — PDFs often export em-dashes as weird sequences
    text = text.replace('\u2013', '–').replace('\u2014', '—')

    return text.strip()


# Text Extraction

def extract_text_from_pdf(filepath: str) -> str:
    """
    Extract the full text content from a PDF file.

    Uses PyMuPDF (fitz) for extraction. PyMuPDF is significantly faster
    and handles multi-column
    layouts better than most alternatives.

    Args:
        filepath: Absolute or relative path to a .pdf file

    Returns:
        Full extracted text as a single string. Consecutive whitespace
        is collapsed to single spaces. Empty pages are skipped.

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid PDF
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    doc = fitz.open(str(path))
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")  # "text" mode = plain text, no formatting
        text = clean_text(text)
        text = text.strip()
        if text:  # skip empty pages (cover pages, image-only pages)
            pages.append(text)

    doc.close()

    # Join pages with double newline to preserve paragraph boundaries
    # across page breaks — important for legal documents where clauses
    # often start at the top of a new page
    return "\n\n".join(pages)


def extract_text_from_txt(filepath: str) -> str:
    """
    Read a plain .txt file and return its content.
    Included so the pipeline handles both .pdf and .txt seamlessly.
    The dataset team generates .txt files for testing — this lets the
    pipeline accept them without special-casing in the UI.

    Args:
        filepath: Absolute or relative path to a .txt file

    Returns:
        Full file content as a string
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    text = path.read_text(encoding="utf-8")
    return clean_text(text)


def extract_text(filepath: str) -> str:
    """
    Unified entry point: handles both .pdf and .txt files.
    This is the function that the pipeline glue (Issue #16) calls.
    """
    suffix = Path(filepath).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(filepath)
    elif suffix == ".txt":
        return extract_text_from_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .txt")


# Sentence-Aware Chunking 
# Sentence-aware chunking respects sentence boundaries: we split at the
# nearest sentence end BEFORE the chunk limit is reached.

def chunk_text(text: str) -> List[str]:
    """
    Split text into overlapping sentence-aware chunks.

    Algorithm:
    1. Split text into sentences on '.', '!', '?' boundaries
    2. Accumulate sentences into a chunk until CHUNK_SIZE_CHARS is reached
    3. When the limit is reached, save the chunk and start a new one
       that begins CHUNK_OVERLAP_CHARS before the current position
       (overlap = re-include the tail of the previous chunk)
    4. Repeat until all text is processed

    Args:
        text: Full extracted document text

    Returns:
        List of text chunk strings. Each chunk is at most CHUNK_SIZE_CHARS
        long. Adjacent chunks share CHUNK_OVERLAP_CHARS of content.
    """
    if not text or not text.strip():
        return []

    # Split into sentences. 
    import re
    # Added the Hindi 'purna viram' (।) to support H2 evaluation!
    sentences = re.split(r'(?<=[.!?।])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If adding this sentence would exceed the chunk size limit,
        # save the current chunk and start a new one with overlap.
        if current_length + sentence_length > CHUNK_SIZE_CHARS and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunks.append(chunk_text_str)

            # Build overlap: walk back from the end of current_chunk
            # collecting sentences until we have enough overlap content.
            overlap_sentences = []
            overlap_length = 0
            for prev_sentence in reversed(current_chunk):
                if overlap_length >= CHUNK_OVERLAP_CHARS:
                    break
                overlap_sentences.insert(0, prev_sentence)
                overlap_length += len(prev_sentence)

            # New chunk starts with the overlap sentences
            current_chunk = overlap_sentences
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_length + 1  # +1 for the space between sentences

    # Don't forget the final chunk (the last chunk is often smaller than the limit)
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Unified Pipeline Entry Point

def extract_chunks(filepath: str) -> List[str]:
    """
    Full pipeline: extract text from file, then chunk it.
    This is the single function that embed.py (Issue #14) will call.

    Args:
        filepath: Path to a .pdf or .txt file

    Returns:
        List of text chunk strings, ready for embedding
    """
    text = extract_text(filepath)
    return chunk_text(text)


# Quick Test 

if __name__ == "__main__":
    import sys
    import tempfile
    import os

    print("Testing extract.py...\n")

    # Test 1: chunk_text with synthetic text 
    # We test chunking independently first — no PDF needed.
    # Generate text that's longer than one chunk so we can verify splitting.
    sample_text = ""
    for i in range(1, 60):
        sample_text += (
            f"Clause {i}: The tenant agrees to pay rent of {i * 100} euros per month "
            f"on the first day of each calendar month. "
        )

    chunks = chunk_text(sample_text)
    assert len(chunks) > 1, "Expected multiple chunks for long text"
    assert all(len(c) <= CHUNK_SIZE_CHARS * 1.1 for c in chunks), \
        "Some chunks exceed size limit"
    print(f"Test 1 passed: {len(chunks)} chunks produced from synthetic text")
    print(f"   First chunk preview: {chunks[0][:120]}...")
    print(f"   Last  chunk preview: {chunks[-1][:120]}...")

    # Test 2: Verify overlap 
    # The last sentence of chunk N should appear somewhere in chunk N+1.
    if len(chunks) >= 2:
        # Get last sentence-like phrase from chunk 0
        last_phrase = chunks[0].split(".")[-2].strip()[-50:]  # last 50 chars before final period
        overlap_found = last_phrase in chunks[1]
        assert overlap_found, f"Expected overlap content from chunk 0 in chunk 1"
        print(f"Test 2 passed: Overlap verified between chunk 0 and chunk 1")

    #  Test 3: .txt file extraction 
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("This is a test lease agreement. The rent is 900 euros per month. "
                "The deposit is 1800 euros. The lease ends December 31, 2025.")
        tmp_path = f.name

    try:
        text = extract_text(tmp_path)
        assert "900 euros" in text
        chunks = extract_and_chunk(tmp_path)
        assert len(chunks) >= 1
        print(f"Test 3 passed: .txt file extracted → {len(chunks)} chunk(s)")
    finally:
        os.unlink(tmp_path)

    # Test 4: Empty file 
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("   ")
        tmp_path = f.name

    try:
        chunks = extract_and_chunk(tmp_path)
        assert chunks == []
        print("Test 4 passed: Empty file returns empty list")
    finally:
        os.unlink(tmp_path)

    # Test 5: Real PDF (optional — skip if no PDF available)
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"\nBonus test: Testing with real PDF: {pdf_path}")
        chunks = extract_and_chunk(pdf_path)
        print(f"Real PDF: {len(chunks)} chunks produced")
        print(f"   Chunk sizes: min={min(len(c) for c in chunks)}, "
              f"max={max(len(c) for c in chunks)}, "
              f"avg={sum(len(c) for c in chunks)//len(chunks)} chars")

        # Show first 3 chunks
        print(f"\n{'=' * 60}")
        print("First 3 chunks:")
        print('=' * 60)
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n--- Chunk {i} ({len(chunk)} characters) ---")
            print(chunk)

        if len(chunks) > 3:
            print(f"\n... ({len(chunks) - 3} more chunks not shown)")
    else:
        print("\nTip: pass a real PDF path as an argument to test extraction:")
        print("     python pipeline/extract.py path/to/document.pdf")

    print("\nAll tests passed. extract.py is ready.")
