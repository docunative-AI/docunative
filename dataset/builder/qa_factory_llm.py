"""
dataset/builder/qa_factory_llm.py
-----------------------------------
Eval 2 QA factory — generates natural language QA pairs using Aya Expanse 32B.

Unlike qa_factory.py (which uses hardcoded English templates derived from seed
facts), this module asks Aya Expanse 32B to READ each generated document and
produce questions IN THE DOCUMENT'S OWN LANGUAGE. Seed facts are used only as
a verification oracle to check that the LLM answer is numerically consistent.

This gives us a second evaluation set (Eval 2) that measures genuine reading
comprehension in the document's native language, not just English fact retrieval.

Pipeline position:
    writer.py → qa_factory_llm.py → evaluate.py (--qa qa_pairs_llm.jsonl)

Output format (identical schema to qa_factory.py so evaluate.py works on both):
    {
        "doc_id":   "de_lease_0",
        "language": "de",
        "domain":   "lease",
        "question": "Wie hoch ist die monatliche Miete?",   ← in document language
        "answer":   "1.350 EUR",                            ← from document text
        "field":    "llm_generated"                         ← marks Eval 2 pairs
    }

Key design decisions:
    - Questions are generated in the document's language (German → German questions)
    - Answers come from the document text, not seed facts directly
    - Seed facts act as oracle: if an answer is numerically inconsistent with
      known facts, the pair is rejected and retried (up to MAX_RETRIES)
    - Output is written to qa_pairs_llm.jsonl — separate from qa_pairs.jsonl
    - evaluate.py is unchanged — just pass --qa dataset/output/qa_pairs_llm.jsonl

Usage:
    # Full generation — all 360 documents, ~360 Cohere API calls
    python -m dataset.builder.qa_factory_llm

    # Quick test — 1 document per language/domain (12 total)
    python -m dataset.builder.qa_factory_llm --test

    # Single language
    python -m dataset.builder.qa_factory_llm --language de

    # Single domain
    python -m dataset.builder.qa_factory_llm --domain lease

Author: Vinod Anbalagan
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from dataset.builder.facts import SUPPORTED_DOMAINS, generate_facts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths

DATASET_ROOT   = Path(__file__).resolve().parent.parent
OUTPUT_DIR     = DATASET_ROOT / "output"
OUTPUT_FILE    = OUTPUT_DIR / "qa_pairs_llm.jsonl"

SUPPORTED_LANGUAGES = ["de", "hi", "id"]

# ---------------------------------------------------------------------------
# Constants

QA_PAIRS_PER_DOC = 10   # same as qa_factory.py
MAX_RETRIES      = 2    # retries if validation fails


# ---------------------------------------------------------------------------
# Language names for prompt

LANG_NAMES = {
    "de": "German",
    "hi": "Hindi",
    "id": "Indonesian",
}


# ---------------------------------------------------------------------------
# Cohere client

def _get_cohere_client():
    """Return a Cohere client. Requires COHERE_API_KEY in .env."""
    try:
        import cohere
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY not set. Add it to .env before running qa_factory_llm.py."
            )
        return cohere.Client(api_key=api_key)
    except ImportError:
        raise ImportError("cohere package not installed. Run: uv add cohere")


# ---------------------------------------------------------------------------
# Prompt builder

def _build_qa_prompt(
    document_text: str,
    language: str,
    domain: str,
    n_pairs: int,
) -> str:
    """
    Build the prompt for Aya Expanse 32B to generate QA pairs from a document.

    The model is asked to:
    1. Read the document carefully
    2. Generate n_pairs questions IN the document's language
    3. Answer each question using only information from the document
    4. Return a strict JSON array (no markdown fences)

    The force-JSON instruction is critical — we parse the response directly.
    """
    lang_name = LANG_NAMES.get(language, language)

    return f"""You are a multilingual document QA expert. Read the following {lang_name} document carefully.

Generate exactly {n_pairs} question-answer pairs IN {lang_name}. Each question must:
- Be written entirely in {lang_name}
- Be answerable from the document text alone
- Cover different facts (rent, dates, durations, permissions, fees, etc.)
- Be a natural question a reader would ask about this document

Return ONLY a valid JSON array. No markdown, no explanation, no preamble.
Each element must have exactly two keys: "question" and "answer".

Example format:
[
  {{"question": "question in {lang_name}", "answer": "answer in {lang_name}"}},
  {{"question": "question in {lang_name}", "answer": "answer in {lang_name}"}}
]

Document:
{document_text[:3000]}

Return the JSON array now:"""


# ---------------------------------------------------------------------------
# Response parser

def _parse_qa_response(response_text: str) -> list[dict[str, str]] | None:
    """
    Parse the LLM response into a list of QA dicts.

    Handles:
    - Clean JSON arrays
    - JSON arrays wrapped in markdown code fences
    - Partial JSON (truncated by token limit)

    Returns None if parsing fails completely.
    """
    text = response_text.strip()

    # Strip markdown fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [
                p for p in parsed
                if isinstance(p, dict)
                and "question" in p
                and "answer" in p
                and p["question"].strip()
                and p["answer"].strip()
            ]
    except json.JSONDecodeError:
        pass

    # Try to extract a JSON array using regex (handles truncated responses)
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [
                    p for p in parsed
                    if isinstance(p, dict)
                    and "question" in p
                    and "answer" in p
                    and p["question"].strip()
                    and p["answer"].strip()
                ]
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse QA response: %s", response_text[:200])
    return None


# ---------------------------------------------------------------------------
# Seed fact oracle validator

def _validate_against_facts(
    qa_pairs: list[dict[str, str]],
    facts: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Filter QA pairs whose answers are inconsistent with seed facts.

    Oracle validation: for each numeric seed fact (e.g. monthly_rent = 1350),
    if a QA pair's answer contains a number, check that it's not wildly
    inconsistent with a known fact value. This catches cases where the LLM
    hallucinated a number that doesn't appear in the document.

    This is a soft filter — we don't reject if we can't determine consistency.
    We only reject obvious contradictions (e.g. answer says 999 but fact is 1350
    and no fact is close to 999).
    """
    # Collect all numeric fact values
    numeric_values = set()
    for k, v in facts.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float)):
            numeric_values.add(str(int(v)))
        elif isinstance(v, str):
            # Extract numbers from money strings like "1350 EUR"
            nums = re.findall(r'\d+', v)
            numeric_values.update(nums)

    valid_pairs = []
    for pair in qa_pairs:
        answer = pair["answer"]
        # Extract numbers from the answer
        answer_nums = re.findall(r'\d{2,}', answer)  # 2+ digit numbers only

        if not answer_nums:
            # No numbers in answer — can't do numeric validation, accept it
            valid_pairs.append(pair)
            continue

        # Check if any answer number matches a known fact value
        # If none match and we have strong numeric seed facts, flag as suspicious
        if any(n in numeric_values for n in answer_nums):
            valid_pairs.append(pair)
        else:
            # Answer has numbers that don't match any seed fact
            # Only reject if we have many numeric seed facts (strong oracle signal)
            if len(numeric_values) >= 3:
                logger.debug(
                    "Rejecting pair — answer numbers %s not in seed facts %s: Q=%s",
                    answer_nums, numeric_values, pair["question"][:60]
                )
            else:
                # Weak oracle — accept anyway
                valid_pairs.append(pair)

    return valid_pairs


# ---------------------------------------------------------------------------
# Document loader

def _load_documents(docs_dir: Path, language: str) -> list[dict]:
    """
    Load all documents for a given language from the JSONL file.

    Returns list of dicts with keys: doc_id, language, domain, document_text, facts
    """
    jsonl_path = docs_dir / f"{language}.jsonl"
    if not jsonl_path.exists():
        logger.warning("No document file found: %s", jsonl_path)
        return []

    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    logger.info("Loaded %d documents for language=%s", len(docs), language)
    return docs


# ---------------------------------------------------------------------------
# Core generation function

def generate_llm_qa_pairs(
    doc: dict,
    client,
    n_pairs: int = QA_PAIRS_PER_DOC,
    model: str = "c4ai-aya-expanse-32b",
) -> list[dict[str, Any]]:
    """
    Generate LLM-based QA pairs for a single document.

    Args:
        doc:     Document dict from the JSONL file (has doc_id, language, domain,
                 document_text, facts)
        client:  Cohere client instance
        n_pairs: Number of QA pairs to generate
        model:   Cohere model to use for generation

    Returns:
        List of QA pair dicts with the same schema as qa_factory.py output.
        Returns empty list if generation or parsing fails after all retries.
    """
    doc_id        = doc["doc_id"]
    language      = doc["language"]
    domain        = doc["domain"]
    document_text = doc["document_text"]
    facts         = doc.get("facts", {})

    prompt = _build_qa_prompt(document_text, language, domain, n_pairs)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat(
                model=model,
                message=prompt,
                temperature=0.3,   # lower than writer.py — we want consistent QA not creative
            )
            raw_text = getattr(response, "text", "") or ""

            if not raw_text:
                logger.warning("Empty response for %s (attempt %d)", doc_id, attempt + 1)
                continue

            # Parse the JSON response
            pairs = _parse_qa_response(raw_text)
            if not pairs:
                logger.warning(
                    "Failed to parse response for %s (attempt %d)", doc_id, attempt + 1
                )
                continue

            # Validate against seed facts oracle
            pairs = _validate_against_facts(pairs, facts)
            if not pairs:
                logger.warning(
                    "All pairs rejected by oracle for %s (attempt %d)", doc_id, attempt + 1
                )
                continue

            # Cap to n_pairs and format to match qa_factory.py output schema
            result = []
            for pair in pairs[:n_pairs]:
                result.append({
                    "doc_id":   doc_id,
                    "language": language,
                    "domain":   domain,
                    "question": pair["question"].strip(),
                    "answer":   pair["answer"].strip(),
                    "field":    "llm_generated",  # marks this as Eval 2
                })

            logger.info(
                "Generated %d LLM QA pairs for %s (attempt %d)",
                len(result), doc_id, attempt + 1,
            )
            return result

        except Exception as e:
            logger.warning(
                "Generation error for %s (attempt %d): %s", doc_id, attempt + 1, e
            )

    logger.error("Failed to generate QA pairs for %s after %d attempts", doc_id, MAX_RETRIES + 1)
    return []


# ---------------------------------------------------------------------------
# Full pipeline

def generate_all_llm_qa_pairs(
    *,
    languages: list[str] | None = None,
    domains:   list[str] | None = None,
    test_mode: bool = False,
    docs_dir:  Path | None = None,
    output_path: Path | None = None,
    model: str = "c4ai-aya-expanse-32b",
) -> list[dict]:
    """
    Generate LLM QA pairs for all language × domain combinations.

    Args:
        languages:    List of language codes. Defaults to all 3 (de, hi, id).
        domains:      List of domain names. Defaults to all 4.
        test_mode:    If True, only process 1 document per language (quick test).
        docs_dir:     Directory containing de.jsonl, hi.jsonl, id.jsonl.
                      Defaults to dataset/output/.
        output_path:  Where to save qa_pairs_llm.jsonl.
                      Defaults to dataset/output/qa_pairs_llm.jsonl.
        model:        Cohere model for generation.

    Returns:
        Flat list of all generated QA pair dicts.
    """
    languages  = languages or SUPPORTED_LANGUAGES
    domains    = domains   or SUPPORTED_DOMAINS
    docs_dir   = docs_dir  or OUTPUT_DIR
    output_path = output_path or OUTPUT_FILE

    client = _get_cohere_client()

    all_pairs: list[dict] = []
    total_docs  = 0
    failed_docs = 0

    for language in languages:
        docs = _load_documents(docs_dir, language)
        if not docs:
            logger.warning("No documents found for language=%s — run writer.py first", language)
            continue

        # Filter by domain if specified
        if domains != SUPPORTED_DOMAINS:
            docs = [d for d in docs if d.get("domain") in domains]

        # Test mode: 1 doc per language
        if test_mode:
            docs = docs[:1]
            logger.info("Test mode: processing only 1 document for language=%s", language)

        for doc in docs:
            total_docs += 1
            pairs = generate_llm_qa_pairs(doc, client, model=model)
            if pairs:
                all_pairs.extend(pairs)
            else:
                failed_docs += 1

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  Documents processed: {total_docs}")
    print(f"  Documents failed:    {failed_docs}")
    print(f"  QA pairs generated:  {len(all_pairs)}")
    print(f"  Output saved to:     {output_path}")

    if failed_docs:
        print(f"\n  WARNING: {failed_docs} documents failed — check logs for details.")
        print(f"  Re-run with --language <code> to retry specific languages.")

    return all_pairs


# ---------------------------------------------------------------------------
# CLI

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Generate LLM-based QA pairs from existing documents (Eval 2). "
            "Requires COHERE_API_KEY in .env and de.jsonl/hi.jsonl/id.jsonl in dataset/output/."
        )
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test — only process 1 document per language (3 API calls total).",
    )
    parser.add_argument(
        "--language",
        choices=["de", "hi", "id"],
        default=None,
        help="Process only this language.",
    )
    parser.add_argument(
        "--domain",
        choices=SUPPORTED_DOMAINS,
        default=None,
        help="Process only this domain.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output path for qa_pairs_llm.jsonl. Default: {OUTPUT_FILE}",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory containing de.jsonl etc. Default: {OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model",
        default="c4ai-aya-expanse-32b",
        help="Cohere model for generation. Default: c4ai-aya-expanse-32b",
    )
    args = parser.parse_args()

    languages = [args.language] if args.language else None
    domains   = [args.domain]   if args.domain   else None

    if not args.docs.exists():
        print(f"Error: docs directory not found: {args.docs}")
        print("Run writer.py first: python -m dataset.builder.writer --language id")
        return

    generate_all_llm_qa_pairs(
        languages=languages,
        domains=domains,
        test_mode=args.test,
        docs_dir=args.docs,
        output_path=args.output,
        model=args.model,
    )


if __name__ == "__main__":
    main()
