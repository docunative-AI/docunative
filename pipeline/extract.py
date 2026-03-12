"""
DocuNative - RAG Pipeline: PDF Extraction and Chunking

This module handles reading PDF documents and splitting them into
overlapping text chunks for use in the RAG (Retrieval-Augmented Generation) pipeline.
"""

import re
from typing import List

try:
    import fitz
except ImportError:
    fitz = None


def extract_text(pdf_path: str) -> str:
    """
    Read all pages from a PDF file and return the full document text
    as a single string using PyMuPDF (fitz).

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Full text content from all pages
    """
    if fitz is None:
        raise ImportError(
            "PyMuPDF (fitz) is not installed. "
            "Install with: pip install PyMuPDF"
        )

    doc = fitz.open(pdf_path)
    text_parts = []

    for page in doc:
        text_parts.append(page.get_text())

    full_text = "\n".join(text_parts)
    doc.close()

    return full_text


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """
    Split text into overlapping chunks for RAG processing.

    Chunks are created by splitting text into segments of `chunk_size` tokens
    with an overlap of `overlap` tokens between chunks. The overlap ensures that
    context is preserved across chunk boundaries, helping maintain continuity.

    The function is sentence-aware - it will not cut a sentence in half.

    Args:
        text: Full text to chunk
        chunk_size: Target size for each chunk in tokens
        overlap: Number of tokens to overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    tokens = text.split()

    if not tokens:
        return chunks

    position = 0

    while position < len(tokens):
        # Calculate end position for this chunk (chunk_size + overlap)
        end = min(position + chunk_size + overlap, len(tokens))

        chunk_tokens = tokens[position:end]

        # Reconstruct chunk text
        chunk_text = " ".join(chunk_tokens)

        # Check if we would cut a sentence in half
        # Find the last sentence boundary before the cut point
        if end < len(tokens):
            # Look ahead for sentence-ending punctuation
            lookahead = tokens[end:end + 100]  # Look 100 tokens ahead for safety

            # Find the last sentence end before our cut
            last_sentence_end = -1
            for i, token in enumerate(lookahead):
                if token in ('.', '!', '?', ';'):
                    last_sentence_end = i + 1
                    break

            if last_sentence_end != -1:
                # Found a sentence boundary, adjust end to include full sentence
                end = min(end + last_sentence_end + 1, len(tokens))

        # Rebuild chunk with adjusted boundary
        chunk_tokens = tokens[position:end]
        chunk_text = " ".join(chunk_tokens)

        chunks.append(chunk_text)

        # Move position forward (chunk_size - overlap) to maintain overlap
        position += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    import sys
    full_text = extract_text("/Users/f/Documents/Projects/cohere/DocuNative_Phase_2__Developer_Onboarding_Guide.pdf")

    print(full_text)

    # if len(sys.argv) > 1:
    #     print("Usage: python pipeline/extract.py [pdf_path]")
    #     sys.exit(1)

    # pdf_path = sys.argv[1]

    # print(f"Extracting text from: {pdf_path}")

    # # Extract full text
    # full_text = extract_text(pdf_path)
    # print(f"Total text length: {len(full_text)} characters")

    # # Chunk into overlapping segments
    # chunks = chunk_text(full_text, chunk_size=400, overlap=80)
    # print(f"Created {len(chunks)} chunks")

    # # Display first 3 chunks
    # print("\nFirst 3 chunks:")
    # print("-" * 50)
    # for i, chunk in enumerate(chunks[:3]):
    #     print(f"\nChunk {i + 1}:")
    #     print(chunk[:150] + ("..." if len(chunk) > 150 else ""))
