"""
pipeline/embed.py
-----------------
BGE-M3 embeddings + ChromaDB local vector store.

Responsibilities:
  1. Load BGE-M3 once (lazy singleton - not on import, only on first use)
  2. Accept a list of text chunks from extract.py
  3. Embed each chunk with BGE-M3
  4. Store chunks + embeddings in a ChromaDB collection
  5. Reset the collection on each new document upload (no stale data)

Usage (standalone test):
    python pipeline/embed.py

Usage (from pipeline):
    from pipeline.embed import build_index
    collection = build_index(chunks, doc_id="upload_001")
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# -----------
# Constants

# BGE-M3 is cross-lingual — same model handles German, Hindi, Polish queries
# and maps them into the same embedding space as the document chunks.
MODEL_NAME = "BAAI/bge-m3"

# ChromaDB lives entirely on disk — no cloud, no server.
CHROMA_PATH = ".chromadb"

# The collection name is fixed. We reset it on each new document upload
# so stale chunks from a previous PDF never contaminate the current query.
COLLECTION_NAME = "docunative_chunks"


# Lazy singleton — model loads once, stays in RAM for the session

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """
    Return the BGE-M3 model, loading it on first call only.

    Why lazy loading?
    - The model is ~570MB and takes ~3s to load on first call.
    - If we loaded it at import time, every module that imports embed.py
      would pay that cost even if embeddings are not needed (e.g. unit tests).
    - With lazy loading, the cost is paid once, on the first real embed call,
      then the model stays in RAM for all subsequent queries in the session.
    """
    global _model
    if _model is None:
        # Auto-detect best available hardware
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"   # Apple Silicon Metal — ~5x faster than CPU
        else:
            device = "cpu"
        logger.info("Loading BGE-M3 model on %s (first call — ~3s)...", device.upper())
        _model = SentenceTransformer(MODEL_NAME, device=device)
        logger.info("BGE-M3 loaded on %s.", device.upper())
    return _model

# ChromaDB client — also a singleton

_chroma_client: Optional[chromadb.PersistentClient] = None


def _get_chroma_client() -> chromadb.PersistentClient:
    """
    Return the ChromaDB persistent client, creating it on first call only.

    PersistentClient stores all vectors on disk at CHROMA_PATH.
    This means embeddings survive process restarts — useful during development.
    We always reset the collection on new uploads so stale data never leaks.
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


# Core function


def build_index(chunks: list[str], doc_id: str = "default") -> chromadb.Collection:
    """
    Embed a list of text chunks and store them in ChromaDB.

    This function ALWAYS resets the collection before inserting new chunks.
    This is intentional — each call to build_index represents a new document
    upload, and we never want chunks from a previous PDF to appear in results.

    Args:
        chunks:  List of text strings from extract.py. Each string is one chunk.
        doc_id:  An identifier for this document (used in chunk metadata).
                 Defaults to "default". Use a filename or UUID in production.

    Returns:
        A ChromaDB Collection object, ready for querying via retrieve.py.

    Raises:
        ValueError: If chunks is empty.
    """
    if not chunks:
        raise ValueError(
            "chunks list is empty — nothing to embed. "
            "Check that extract.py returned at least one chunk."
        )

    model = _get_model()
    client = _get_chroma_client()

    # --- Reset collection ----
    # Delete the existing collection if it exists, then recreate it.
    # This guarantees: no stale chunks, no dimension mismatches, no duplicate IDs.
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'.", COLLECTION_NAME)
    except Exception:
        # Collection did not exist yet — fine on first run.
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        # cosine distance is best for sentence embeddings (normalised vectors)
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Created fresh collection '%s'.", COLLECTION_NAME)

    # --- Embed all chunks ----
    logger.info("Embedding %d chunks with BGE-M3...", len(chunks))

    # encode() returns a numpy array of shape (n_chunks, 1024)
    # Adaptive batch size: CPU can OOM with 32 chunks × ~1600 chars each.
    # MPS/CUDA have enough VRAM for 32. CPU is capped at 8 for safety.
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        batch_size = 32
    else:
        batch_size = 8  # CPU-only machines (Linux teammates)

    embeddings = model.encode(
        chunks,
        show_progress_bar=len(chunks) > 10,
        normalize_embeddings=True,  # cosine sim == dot product after normalisation
        batch_size=batch_size,
    )

    # --- Build ChromaDB document IDs -----
    # IDs must be unique strings. We use a hash of (doc_id + chunk_index)
    # so they are stable and reproducible across runs.
    ids = [_chunk_id(doc_id, i) for i in range(len(chunks))]

    # --- Metadata for each chunk ----
    metadatas = [
        {"doc_id": doc_id, "chunk_index": i, "chunk_length": len(chunk)}
        for i, chunk in enumerate(chunks)
    ]

    # --- Insert into ChromaDB ----
    # ChromaDB expects embeddings as a list of lists (not numpy arrays)
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=metadatas,
    )

    logger.info(
        "Indexed %d chunks into ChromaDB collection '%s'.",
        len(chunks),
        COLLECTION_NAME,
    )

    return collection

# Helper

def _chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Generate a stable, unique ID for a chunk.
    Format: sha256(doc_id:chunk_index)[:16]
    """
    raw = f"{doc_id}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

# Standalone test

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Simulate what extract.py would produce for a short German lease document
    test_chunks = [
        "Der monatliche Mietzins beträgt 1.350 EUR und ist jeweils zum ersten "
        "des Monats fällig. Die Kaution beläuft sich auf 2.700 EUR.",
        "Die Kündigungsfrist beträgt 60 Tage zum Monatsende. Eine Kündigung "
        "muss schriftlich erfolgen und dem Vermieter nachweislich zugehen.",
        "Haustiere sind in dieser Wohnung nicht gestattet. Ausnahmen bedürfen "
        "der ausdrücklichen schriftlichen Zustimmung des Vermieters.",
        "Der Mietvertrag beginnt am 1. April 2025 und läuft auf unbestimmte "
        "Zeit. Eine Mindestmietdauer ist nicht vereinbart.",
    ]

    """
    Translation: 
    The monthly rent is €1,350 and is due on the first of each month. The security deposit is €2,700.
    The notice period is 60 days to the end of the month. Notice of termination must be given in writing and demonstrably received by the landlord.
    Pets are not permitted in this apartment. Exceptions require the express written consent of the landlord.
    The lease begins on April 1, 2025, and is for an indefinite period.
    There is no minimum lease term.

    """

    print(f"\nBuilding index for {len(test_chunks)} test chunks...\n")
    collection = build_index(test_chunks, doc_id="test_lease_de")

    print(f"\n Collection built. Item count: {collection.count()}")
    print("\nRunning a sample query to verify retrieval works:")

    model = _get_model()
    q = "Wie hoch ist die Miete?" # How much is the rent?
    q_embedding = model.encode([q], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=2)
    print(f"\nQuery: '{q}'")
    for i, doc in enumerate(results["documents"][0]):
        dist = results["distances"][0][i]
        print(f"  [{i+1}] distance={dist:.4f} | {doc[:80]}...")
    print("\nIf distance is < 0.3 for the rent chunk, BGE-M3 cross-lingual retrieval is working.")
