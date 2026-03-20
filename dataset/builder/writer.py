"""
dataset/builder/writer.py

Step 2 of the dataset pipeline: take facts from facts.py and generate
realistic legal document text via Cohere API or local Ollama.

- Cohere: production (Aya Expanse 32B), uses API credits.
- Ollama: local testing (e.g. gemma2:9b, qwen2.5:14b), no API cost.

Output: JSONL files per language in dataset/output/ (de.jsonl, hi.jsonl, sw.jsonl).

Issue: #17
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Optional: facts.py may live in dataset/seed-facts branch until merged
try:
    from dataset.builder.facts import SUPPORTED_DOMAINS, generate_facts
except ImportError as e:
    raise ImportError(
        "dataset.builder.facts is required (Issue #28). "
        "Merge the dataset/seed-facts branch or add dataset/builder/facts.py and dataset/seeds/*.yaml."
    ) from e

logger = logging.getLogger(__name__)

# Paths
DATASET_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = DATASET_ROOT / "output"
SUPPORTED_LANGUAGES = ["de", "hi", "sw"]

# Domain -> human-readable document type for prompts
DOMAIN_LABELS = {
    "lease": "residential lease agreement (Mietvertrag / rental contract)",
    "employment": "employment contract (Arbeitsvertrag / work agreement)",
    "health_insurance": "health insurance policy document",
    "immigration_letter": "immigration / residence permit letter",
}

# Language code -> full name for prompts
LANG_NAMES = {
    "de": "German",
    "hi": "Hindi",
    "sw": "Swahili",
}


def _build_prompt(facts: dict[str, Any]) -> str:
    """Build the LLM prompt from a facts dict (from generate_facts)."""
    lang = facts.get("_language", "de")
    domain = facts.get("_domain", "lease")
    lang_name = LANG_NAMES.get(lang, lang)
    domain_label = DOMAIN_LABELS.get(domain, domain)

    # Deterministic page count 1-5 from seed
    seed = facts.get("_seed", 0)
    page_count = (seed % 5) + 1
    target_words = page_count * 250

    # Fact bullets (exclude metadata keys)
    fact_items = [
        f"- {k}: {v}"
        for k, v in sorted(facts.items())
        if not k.startswith("_")
    ]
    facts_block = "\n".join(fact_items)

    prompt = f"""Generate a realistic {lang_name} {domain_label} with the following mandatory facts:

{facts_block}

CRITICAL INSTRUCTIONS:
- Write the entire document in {lang_name} only. No English text unless it is a proper noun or legal term with no {lang_name} equivalent.
- Use proper legal/administrative wording, headings, and structure appropriate for {lang_name}-speaking countries and cultural context.
- IMPORTANT: The facts listed above are mandatory and must appear verbatim in the document. Do NOT use [Name], [Amount], [Date] or any square bracket placeholders anywhere.
- For all other fields not listed above (e.g. party names, addresses, dates, reference numbers, additional clauses): fill them in with realistic random values appropriate for the document type and language. Do NOT leave any field blank or use placeholders.
- Include all standard sections, clauses, and a signature block appropriate for this document type.
- Make the document feel realistic and complete — a real person should be able to read it and understand all the terms.
- Target length: approximately {target_words} words ({page_count} page(s)).
- Output only the document text, no preamble or explanation."""

    return prompt


# NOTE on model selection:
# c4ai-aya-expanse-32b covers 23 languages — does NOT include Swahili.
# For production use with Swahili and other low-resource languages, use Aya 101
# which covers all 70+ languages in the Tiny Aya family.
# To switch: set COHERE_WRITER_MODEL=aya-101 in your .env file.
# See: https://docs.cohere.com/docs/aya
DEFAULT_COHERE_MODEL = os.getenv("COHERE_WRITER_MODEL", "c4ai-aya-expanse-32b")


def get_llm_client(use_ollama: bool | None = None):
    """
    Return the appropriate LLM client based on config.
    use_ollama: True = Ollama, False = Cohere, None = read from env USE_OLLAMA.
    """
    if use_ollama is None:
        use_ollama = os.getenv("USE_OLLAMA", "").lower() in ("1", "true", "yes")

    if use_ollama:
        import ollama

        host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ("ollama", ollama.Client(host=host))
    else:
        import cohere

        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY is not set. Set it in .env or use --ollama for local testing."
            )
        return ("cohere", cohere.Client(api_key=api_key))


def generate_document(
    facts: dict[str, Any],
    client_type: str,
    client: Any,
    *,
    model_name: str | None = None,
) -> str | None:
    """
    Generate document text from facts using Cohere or Ollama.
    Returns document string or None on failure.
    """
    prompt = _build_prompt(facts)

    try:
        if client_type == "cohere":
            model = model_name or DEFAULT_COHERE_MODEL
            response = client.chat(
                model=model,
                message=prompt,
                temperature=0.7,
            )
            return getattr(response, "text", None)

        else:  # ollama
            model = model_name or os.getenv("OLLAMA_MODEL", "gemma2:9b")
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.7,
                    "num_predict": 2000,
                },
            )
            msg = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
            if msg is None:
                return None
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            return content if isinstance(content, str) else None

    except Exception as e:
        logger.exception("generate_document failed for %s/%s doc %s", facts.get("_language"), facts.get("_domain"), facts.get("_document_index"))
        return None


def validate_document(document_text: str, facts: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate that a generated document:
    1. Contains no square bracket placeholders like [Name] or [Amount]
    2. Contains at least one numeric seed fact value
    3. Is not suspiciously short (< 100 words)

    Returns (is_valid, list_of_issues).
    Used to decide whether to retry generation.
    """
    import re
    issues = []

    # Check 1: No square bracket placeholders
    placeholders = re.findall(r'\[[A-Za-z][^\]]{1,50}\]', document_text)
    if placeholders:
        issues.append(f"Contains {len(placeholders)} placeholder(s): {placeholders[:3]}")

    # Check 2: At least one numeric fact value appears in the document
    numeric_facts = [
        str(v) for k, v in facts.items()
        if not k.startswith("_") and isinstance(v, (int, float))
    ]
    fact_found = any(
        str(int(v)) in document_text or str(v) in document_text
        for v in numeric_facts
    ) if numeric_facts else True
    if not fact_found:
        issues.append("No numeric seed fact values found in document")

    # Check 3: Not too short
    word_count = len(document_text.split())
    if word_count < 100:
        issues.append(f"Document too short: {word_count} words (minimum 100)")

    # Check 4: No fact value appears more than 3 times (duplicate/repetition guard)
    # A fact like "1350" appearing 10 times suggests the document is repetitive
    for k, v in facts.items():
        if k.startswith("_"):
            continue
        val_str = str(int(v)) if isinstance(v, float) and v == int(v) else str(v)
        if len(val_str) < 3:  # skip very short values like "1" or "No"
            continue
        count = document_text.count(val_str)
        if count > 3:
            issues.append(f"Fact '{k}={val_str}' appears {count} times (possible repetition)")

    return len(issues) == 0, issues


def generate_all_documents(
    *,
    test_mode: bool = False,
    domain_filter: str | None = None,
    language_filter: str | None = None,
    use_ollama: bool | None = None,
    ollama_model: str | None = None,
) -> None:
    """
    Generate documents for all (or filtered) domain × language × index combos.
    Writes JSONL files to dataset/output/ (one file per language).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client_type, client = get_llm_client(use_ollama=use_ollama)
    model_name = ollama_model if client_type == "ollama" else None

    domains = [domain_filter] if domain_filter else SUPPORTED_DOMAINS
    languages = [language_filter] if language_filter else SUPPORTED_LANGUAGES
    n_per_combo = 1 if test_mode else 30

    # Collect rows per language for JSONL
    by_lang: dict[str, list[dict[str, Any]]] = {lang: [] for lang in languages}

    total = len(domains) * len(languages) * n_per_combo
    completed = 0
    failed = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(total=total, desc="Documents", unit="doc")
    except ImportError:
        iterator = None

    for domain in domains:
        if domain not in SUPPORTED_DOMAINS:
            logger.warning("Unknown domain %s, skipping", domain)
            continue
        for lang in languages:
            for doc_idx in range(n_per_combo):
                facts = generate_facts(lang, domain, doc_idx)
                doc_id = f"{lang}_{domain}_{doc_idx}"

                # Generate with validation + retry (max 2 retries)
                text = None
                MAX_RETRIES = 2
                for attempt in range(MAX_RETRIES + 1):
                    candidate = generate_document(facts, client_type, client, model_name=model_name)
                    if candidate is None:
                        continue
                    is_valid, issues = validate_document(candidate, facts)
                    if is_valid:
                        text = candidate
                        break
                    else:
                        logger.warning(
                            "Document %s failed validation (attempt %d/%d): %s",
                            doc_id, attempt + 1, MAX_RETRIES + 1, issues
                        )
                        if attempt == MAX_RETRIES:
                            # Accept the last attempt anyway rather than losing the document
                            logger.warning("Accepting %s despite issues after %d attempts", doc_id, MAX_RETRIES + 1)
                            text = candidate

                if iterator:
                    iterator.update(1)
                if text is None:
                    failed += 1
                    continue

                row = {
                    "doc_id": doc_id,
                    "language": lang,
                    "domain": domain,
                    "document_text": text,
                    "seed": facts.get("_seed"),
                    "facts": facts,
                }
                by_lang[lang].append(row)
                completed += 1

    if iterator:
        iterator.close()

    # Write JSONL per language
    for lang in languages:
        path = OUTPUT_DIR / f"{lang}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in by_lang[lang]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %s (%d documents)", path, len(by_lang[lang]))

    print(f"Done. Generated {completed} documents, {failed} failed.")
    if failed:
        print("Check logs for failed document indices.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate legal documents from facts via Cohere or Ollama (writer.py, Issue #17)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate only 1 document per language/domain (12 total).",
    )
    parser.add_argument(
        "--domain",
        metavar="NAME",
        help="Generate only this domain (e.g. lease, employment).",
    )
    parser.add_argument(
        "--language",
        metavar="CODE",
        help="Generate only this language (de, hi, sw).",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use local Ollama instead of Cohere (overrides USE_OLLAMA).",
    )
    parser.add_argument(
        "--ollama-model",
        metavar="NAME",
        default=None,
        help="Ollama model name (e.g. gemma2:9b). Default from OLLAMA_MODEL.",
    )
    args = parser.parse_args()

    generate_all_documents(
        test_mode=args.test,
        domain_filter=args.domain,
        language_filter=args.language,
        use_ollama=args.ollama or None,
        ollama_model=args.ollama_model,
    )


if __name__ == "__main__":
    main()
