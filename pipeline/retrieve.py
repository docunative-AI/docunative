"""
pipeline/retrieve.py
--------------------
Semantic retrieval — given a user query, return the top-3 most relevant
chunks from the ChromaDB collection built by embed.py.

This is the component that makes cross-lingual search possible:
  - User asks in Swahili: "Kodi ya kila mwezi ni ngapi?" (How much is the monthly rent?)
  - Document chunks are in German: "Der monatliche Mietzins beträgt 1.350 EUR"
  - BGE-M3 maps both into the same embedding space
  - Cosine similarity finds the match despite different languages

Usage (standalone test):
    python pipeline/retrieve.py

Usage (from pipeline):
    from pipeline.retrieve import retrieve
    chunks = retrieve(query="What is the monthly rent?", collection=collection)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

# Re-use the lazy singleton from embed.py — model loads only once per session
from pipeline.embed import _get_model

logger = logging.getLogger(__name__)

# Constants

# Top-3 gives the model enough context to answer most questions
# without overwhelming the prompt with the entire document.
DEFAULT_TOP_K = 2  # Reduced from 3 — 2 chunks = ~600 tokens context, ~25% faster generation.
                   # Legal questions are answered by a single specific clause.
                   # BGE-M3 retrieval quality is high enough to find it in top-2.

# Cosine distance threshold above which we consider a chunk "not relevant".
# Cosine distance is in [0, 2]; for normalised embeddings, practical range
# is [0, 1].
# Set to 1.0 (effectively disabled) so the model always receives context
# and can respond "not found" itself. This avoids silent empty-retrieval
# failures on short documents where all distances exceed a tighter threshold.
RELEVANCE_THRESHOLD = 1.0

# Return type



@dataclass
class RetrievedChunk:
    """
    A single retrieved chunk with its metadata and relevance score.

    Fields:
        text:        The raw chunk text (what gets passed to generate.py)
        chunk_index: Position of this chunk in the original document
        doc_id:      Identifier of the source document
        distance:    Cosine distance from query (lower = more similar)
        rank:        1-indexed rank in this result set (1 = most similar)
    """
    text: str
    chunk_index: int
    doc_id: str
    distance: float
    rank: int

    @property
    def similarity(self) -> float:
        """Cosine similarity = 1 - cosine distance. Higher is better."""
        return 1.0 - self.distance

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk(rank={self.rank}, "
            f"similarity={self.similarity:.3f}, "
            f"text={self.text[:60]!r}...)"
        )

# Core function

@lru_cache(maxsize=128)
def _cached_query_embedding(query_hash: str) -> list:
    """
    Cache query embeddings by content hash.
    Same question = same hash = skip re-embedding (free ~0.7s on Metal).
    maxsize=128: holds ~128 unique queries in RAM (~0.5MB total).
    query_hash is derived outside this function so lru_cache key is a plain string.
    """
    # The actual query text is retrieved via the cache miss path in retrieve()
    # We store it in the closure via _pending_query — see retrieve() below
    model = _get_model()
    embedding = model.encode(
        [_pending_query[0]],
        normalize_embeddings=True,
    )
    return embedding.tolist()


# Thread-local storage for the query text (needed because lru_cache only hashes args)
_pending_query: list = [""]


def retrieve(
    query: str,
    collection: chromadb.Collection,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """
    Embed the query and return the top-k most semantically similar chunks.

    The query can be in any language — BGE-M3 handles cross-lingual matching.
    A Swahili question will correctly retrieve German chunks about the same topic.

    Args:
        query:      The user's natural language question (any language).
        collection: The ChromaDB collection returned by embed.build_index().
        top_k:      Number of chunks to return. Default is 3.

    Returns:
        A list of RetrievedChunk objects, sorted by relevance (best first).
        May be shorter than top_k if the collection has fewer documents
        or if all remaining chunks exceed the relevance threshold.

    Raises:
        ValueError: If query is empty or top_k < 1.
        RuntimeError: If the collection is empty (nothing was indexed).
    """
    if not query or not query.strip():
        raise ValueError("query cannot be empty.")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")

    # Check collection is not empty before querying
    count = collection.count()
    if count == 0:
        raise RuntimeError(
            "ChromaDB collection is empty. "
            "Call embed.build_index() before retrieve()."
        )

    # Clamp top_k to collection size — ChromaDB raises if n_results > count
    effective_k = min(top_k, count)
    if effective_k < top_k:
        logger.warning(
            "Collection only has %d chunks; returning %d instead of %d.",
            count, effective_k, top_k,
        )

    # --- Embed the query (with LRU cache) ---
    # Same question = same hash = skip re-embedding entirely (~0.7s saved on Metal)
    query_clean = query.strip()
    query_hash = hashlib.md5(query_clean.encode()).hexdigest()
    _pending_query[0] = query_clean          # store for cache miss path
    query_embedding = _cached_query_embedding(query_hash)

    # --- Query ChromaDB ----
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=effective_k,
        include=["documents", "metadatas", "distances"],
    )

    # results["documents"] is a list of lists: [[chunk1, chunk2, chunk3]]
    # Unwrap the outer list (we only passed one query embedding)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # --- Build RetrievedChunk objects ---
    chunks: list[RetrievedChunk] = []
    for rank, (doc, meta, dist) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        # Filter out chunks that are clearly irrelevant
        if dist > RELEVANCE_THRESHOLD:
            logger.debug(
                "Chunk rank=%d skipped — distance %.3f exceeds threshold %.3f.",
                rank, dist, RELEVANCE_THRESHOLD,
            )
            continue

        chunks.append(
            RetrievedChunk(
                text=doc,
                chunk_index=meta.get("chunk_index", -1),
                doc_id=meta.get("doc_id", "unknown"),
                distance=dist,
                rank=rank,
            )
        )

    if not chunks:
        logger.warning(
            "No chunks passed the relevance threshold (%.2f) for query: %r",
            RELEVANCE_THRESHOLD, query,
        )

    logger.info(
        "Retrieved %d/%d chunks for query: %r",
        len(chunks), effective_k, query[:60],
    )

    return chunks


def format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a labelled context string.

    NOTE: This is a standalone debug/testing helper only.
    In the production pipeline (app.py / pipeline.py), do NOT use this.
    Instead pass chunk text strings directly to generate_answer():

        retrieved = retrieve(query, collection)
        chunk_strings = [c.text for c in retrieved]
        result = generate_answer(query, chunk_strings)

    generate_answer() owns its own prompt formatting internally.

    Args:
        chunks: List of RetrievedChunk objects from retrieve().

    Returns:
        A formatted string with [Context N] labels — for debugging only.
    """
    if not chunks:
        return "[No relevant context found in document]"

    parts = []
    for chunk in chunks:
        parts.append(f"[Context {chunk.rank}]\n{chunk.text}")

    return "\n\n".join(parts)


# ---------------
# Standalone test

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Import here to avoid circular import when run standalone
    from pipeline.embed import build_index

    # Build a small test index
    test_chunks = [
        "Der monatliche Mietzins beträgt 1.350 EUR und ist jeweils zum ersten "
        "des Monats fällig. Die Kaution beläuft sich auf 2.700 EUR.",
        "Die Kündigungsfrist beträgt 60 Tage zum Monatsende. Eine Kündigung "
        "muss schriftlich erfolgen und dem Vermieter nachweislich zugehen.",
        "Haustiere sind in dieser Wohnung nicht gestattet. Ausnahmen bedürfen "
        "der ausdrücklichen schriftlichen Zustimmung des Vermieters.",
        "Der Mietvertrag beginnt am 1. April 2025 und läuft auf unbestimmte "
        "Zeit. Eine Mindestmietdauer ist nicht vereinbart.",
        "Die Wohnung befindet sich im zweiten Obergeschoss und verfügt über "
        "drei Zimmer, Küche, Bad und einen Balkon mit Südausrichtung.",
    ]

    print("Building test index...")
    collection = build_index(test_chunks, doc_id="test_lease_de")
    print(f"Index built. {collection.count()} chunks stored.\n")

    # Test queries in multiple languages to verify cross-lingual retrieval
    test_queries = [
        ("German",  "Wie hoch ist die monatliche Miete?"),
        ("English", "What is the monthly rent?"),
        ("Hindi",   "मासिक किराया कितना है?"),
        ("Indonesian", "Berapa biaya sewa per bulan?"),
        ("German",  "Sind Haustiere erlaubt?"),
        ("English", "How many days notice is required?"),
    ]

    print("=" * 60)
    print("Cross-lingual retrieval test")
    print("=" * 60)

    for lang, query in test_queries:
        print(f"\n[{lang}] Query: {query}")
        results = retrieve(query, collection, top_k=2)
        if results:
            for r in results:
                print(f"  rank={r.rank} sim={r.similarity:.3f} | {r.text[:70]}...")
        else:
            print("  ⚠️  No results above relevance threshold.")

    print("\n" + "=" * 60)
    print("format_context() output (for generate.py):")
    print("=" * 60)
    sample_results = retrieve("What is the deposit amount?", collection)
    print(format_context(sample_results))
