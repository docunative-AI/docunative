"""
dataset/builder/writer.py

Step 2 of the dataset pipeline: take facts from facts.py and generate
realistic legal document text via Cohere API or local Ollama.

- Cohere: production (Aya Expanse 32B), uses API credits.
- Ollama: local testing (e.g. gemma2:9b, qwen2.5:14b), no API cost.

Output: JSONL files per language in dataset/output/ (de.jsonl, hi.jsonl, id.jsonl).

Issue: #17
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import ValidationError
from dataset.builder.facts import SUPPORTED_DOMAINS, generate_facts

load_dotenv()

from dataset.builder.cohere_retry import cohere_chat_with_429_retry
logger = logging.getLogger(__name__)

# Paths
DATASET_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = DATASET_ROOT / "output"
TARGET_LANGUAGES = ["zh", "hi", "pl"]

LLM_VALIDATION = True

# Per-thread LLM clients (avoid sharing sync HTTP clients across threads)
_thread_local = threading.local()

# Domain -> human-readable document type for prompts
DOMAIN_LABELS = {
    "lease": "residential lease agreement (Mietvertrag / rental contract)",
    "employment": "employment contract (Arbeitsvertrag / work agreement)",
    "health_insurance": "health insurance policy document",
    "immigration_letter": "immigration / residence permit letter",
}

# List of languges covered by Aya Expanse model:
# Language code -> full name for prompts
LANG_NAMES = {
    "zh": "Chinese (Simplified)",  # H2 high-resource: 1.9% internal training proportion
    "hi": "Hindi",                  # H1 + H2 medium-resource: 1.7% internal training proportion
    "pl": "Polish",                 # H2 medium-low resource: 1.4% internal training proportion
    "de": "German",
    "uk": "Ukrainian",
    "ar": "Arabic",
    "cs": "Czech",
    "nl": "Dutch",
    "en": "English",
    "fr": "French",
    "el": "Greek",
    "he": "Hebrew",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "fa": "Persian",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "tr": "Turkish",
    "vi": "Vietnamese"
}


class LLMValidationFeedback(BaseModel):
    is_valid: bool
    explanation: str
    

MAX_DOC_RETRIES = 2

_DEFAULT_COHERE_WORKERS = 8
_DEFAULT_OLLAMA_WORKERS = 1


def _resolve_max_workers(client_type: str, override: int | None) -> int:
    """Bounded concurrency: env WRITER_MAX_WORKERS, CLI override, then safe defaults."""
    if override is not None:
        n = override
    else:
        raw = os.getenv("WRITER_MAX_WORKERS", "").strip()
        if raw:
            n = int(raw)
        elif client_type == "ollama":
            n = _DEFAULT_OLLAMA_WORKERS
        else:
            n = _DEFAULT_COHERE_WORKERS
    return max(1, n)


def _get_worker_client(client_type: str) -> Any:
    """Return a dedicated client for the current worker thread."""
    if client_type == "cohere":
        if getattr(_thread_local, "cohere_client", None) is None:
            import cohere

            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError(
                    "COHERE_API_KEY is not set. Set it in .env or use --ollama for local testing."
                )
            _thread_local.cohere_client = cohere.Client(api_key=api_key)
        return _thread_local.cohere_client

    if getattr(_thread_local, "ollama_client", None) is None:
        import ollama

        host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        _thread_local.ollama_client = ollama.Client(host=host)
    return _thread_local.ollama_client


def _build_prompt(facts: dict[str, Any]) -> str:
    """Build the LLM prompt from a facts dict (from generate_facts)."""
    lang = facts.get("_language", "zh")
    domain = facts.get("_domain", "lease")
    lang_name = LANG_NAMES.get(lang, lang)
    domain_label = DOMAIN_LABELS.get(domain, domain)

    # Deterministic page count 1-5 from seed
    seed = facts.get("_seed", 0)
    page_count = (seed % 5) + 1

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
- Target length: approximately ({page_count} page(s)).
- Output only the document text, no preamble or explanation."""

    return prompt


# Documents validation
def _build_validation_prompt(facts: dict[str, Any], document_text: str) -> str:
    """Build the LLM validation prompt that verifies if the output is a valid document"""
    
    lang = facts.get("_language", "zh")
    domain = facts.get("_domain", "lease")
    
    lang_name = LANG_NAMES.get(lang, lang)
    domain_label = DOMAIN_LABELS.get(domain, domain)

    # Fact bullets (exclude metadata keys)
    fact_items = [
        f"- {k}: {v}"
        for k, v in sorted(facts.items())
        if not k.startswith("_")
    ]
    facts_block = "\n".join(fact_items)

    prompt = f"""Verify if the following text is a valid {domain_label} in {lang_name} containing the following mandatory facts:

        {facts_block}

        Present your result only as a JSON object that strongly follows this format:
        
        class DataFormat(BaseModel):
            is_valid: bool # Whether document is valid or not
            explanation: str # Explanation is text to clarify your answer
            
        Note: never write the word 'json'

        Given document text:
        {document_text}
    """

    return prompt


def _validate_document_with_LLM(
    facts: dict[str, Any],
    document_candidate,
    client_type: str,
    client: Any,
    *,
    model_name: str | None = None,
) -> dict[bool, str]:
    """
    Verifies whether document is valid, returns True or False
    """
    
    validation_prompt = _build_validation_prompt(facts, document_candidate)

    try:
        if client_type == "cohere":
            model = model_name or DEFAULT_COHERE_MODEL
            response = client.chat(
                model=model,
                message=validation_prompt,
                temperature=0.7,
            )
            response = getattr(response, "text", None)
            return response

        else:  # ollama
            model = model_name or os.getenv("OLLAMA_MODEL", "gemma2:9b")
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": validation_prompt}],
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
        logger.exception("llm_validation failed for %s/%s doc %s", facts.get("_language"), facts.get("_domain"), facts.get("_document_index"))
        return None

    
def _verify_llm_output(raw_output: str) -> LLMValidationFeedback | None:
    """
    Validates a raw JSON string from a validation LLM against the LLMValidationFeedback schema.
    """
    
    raw_output = raw_output.strip("```")
    raw_json_output = raw_output.lstrip("json")
    
    try:
        validated_response = LLMValidationFeedback.model_validate_json(raw_json_output)
        return validated_response
    
    except ValidationError as e:
        print("Error: Unable to validate validation LLM response against the LLMValidationFeedback schema.")
        print(f"Validation Errors: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: LLM validation output was not valid JSON.")
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
        if not k.startswith("_") and isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    fact_found = any(
        str(int(float(v))) in document_text or str(v) in document_text
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


# NOTE on model selection:
# c4ai-aya-expanse-32b covers 23 languages — includes Polish (pl) natively.
# Polish was chosen as the medium-low resource language for the zh → hi → pl gradient.
# Internal training proportions from Tiny Aya Appendix A: zh 1.9%, hi 1.7%, pl 1.4%.
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

    if client_type == "cohere":
        model = model_name or DEFAULT_COHERE_MODEL
        try:
            response = cohere_chat_with_429_retry(
                client,
                logger,
                model=model,
                message=prompt,
                temperature=0.7,
                exhausted_message=(
                    f"Cohere 429 rate limit for {facts.get('_language')}/{facts.get('_domain')} "
                    f"doc_index={facts.get('_document_index')}"
                ),
            )
            if response is None:
                return None
            return getattr(response, "text", None)
        except Exception:
            logger.exception(
                "generate_document failed for %s/%s doc %s",
                facts.get("_language"),
                facts.get("_domain"),
                facts.get("_document_index"),
            )
            return None

    try:
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

    except Exception:
        logger.exception(
            "generate_document failed for %s/%s doc %s",
            facts.get("_language"),
            facts.get("_domain"),
            facts.get("_document_index"),
        )
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
        if not k.startswith("_") and isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    fact_found = any(
        str(int(float(v))) in document_text or str(v) in document_text
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


def _generate_one_doc(
    lang: str,
    domain: str,
    doc_idx: int,
    client_type: str,
    model_name: str | None,
) -> tuple[str, int, int, dict[str, Any] | None, bool]:
    """
    Generate one document (facts + LLM + validation retries).

    Returns:
        (lang, domain_index, doc_idx, row_dict_or_none, hard_failed)
        hard_failed True means no row was produced (LLM returned nothing usable).
    """
    domain_index = SUPPORTED_DOMAINS.index(domain)
    facts = generate_facts(lang, domain, doc_idx)
    doc_id = f"{lang}_{domain}_{doc_idx}"
    client = _get_worker_client(client_type)

    text = None
    for attempt in range(MAX_DOC_RETRIES + 1):
        candidate = generate_document(facts, client_type, client, model_name=model_name)
        if candidate is None:
            continue
        is_valid, issues = validate_document(candidate, facts)
        if is_valid:
            text = candidate
            break
        logger.warning(
            "Document %s failed validation (attempt %d/%d): %s",
            doc_id, attempt + 1, MAX_DOC_RETRIES + 1, issues
        )
        if attempt == MAX_DOC_RETRIES:
            logger.warning("Accepting %s despite issues after %d attempts", doc_id, MAX_DOC_RETRIES + 1)
            text = candidate

    if text is None:
        return (lang, domain_index, doc_idx, None, True)

    row = {
        "doc_id": doc_id,
        "language": lang,
        "domain": domain,
        "document_text": text,
        "seed": facts.get("_seed"),
        "facts": facts,
    }
    return (lang, domain_index, doc_idx, row, False)


def _sort_rows_for_language(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable domain order (SUPPORTED_DOMAINS) then doc_idx, matching legacy sequential writer."""
    def key(r: dict[str, Any]) -> tuple[int, int]:
        return (SUPPORTED_DOMAINS.index(r["domain"]), int(r["doc_id"].rsplit("_", 1)[-1]))

    return sorted(rows, key=key)


def generate_all_documents(
    *,
    test_mode: bool = False,
    domain_filter: str | None = None,
    language_filter: str | None = None,
    use_ollama: bool | None = None,
    ollama_model: str | None = None,
    max_workers: int | None = None,
) -> None:
    """
    Generate documents for all (or filtered) domain × language × index combos.
    Writes JSONL files to dataset/output/ (one file per language).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client_type, _main_client = get_llm_client(use_ollama=use_ollama)
    model_name = ollama_model if client_type == "ollama" else None

    if domain_filter:
        if domain_filter not in SUPPORTED_DOMAINS:
            logger.warning("Unknown domain %s, skipping", domain_filter)
            domains = []
        else:
            domains = [domain_filter]
    else:
        domains = list(SUPPORTED_DOMAINS)

    languages = [language_filter] if language_filter else SUPPORTED_LANGUAGES
    n_per_combo = 1 if test_mode else 30

    tasks: list[tuple[str, str, int]] = []
    for domain in domains:
        for lang in languages:
            for doc_idx in range(n_per_combo):
                tasks.append((lang, domain, doc_idx))

    total = len(tasks)
    workers = _resolve_max_workers(client_type, max_workers)
    completed = 0
    failed = 0
    raw_results: list[tuple[str, int, int, dict[str, Any] | None, bool]] = []

    try:
        from tqdm import tqdm
        iterator = tqdm(total=total, desc="Documents", unit="doc")
    except ImportError:
        iterator = None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(_generate_one_doc, lang, domain, doc_idx, client_type, model_name): (lang, domain, doc_idx)
            for lang, domain, doc_idx in tasks
        }
        for fut in as_completed(future_to_task):
            raw_results.append(fut.result())
            if iterator:
                iterator.update(1)
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
                    
                    if LLM_VALIDATION:
                        
                        llm_validation_feedback_json = _validate_document_with_LLM(facts, candidate, client_type, client, model_name=model_name)
                        llm_validation_feedback = _verify_llm_output(llm_validation_feedback_json)
                        
                        if llm_validation_feedback is None:
                            is_valid = False
                            issues = "LLM validation provided not a valid JSON"
                        else:
                            is_valid = llm_validation_feedback.is_valid
                            issues = llm_validation_feedback.explanation
                    else: 
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

    by_lang: dict[str, list[dict[str, Any]]] = {lang: [] for lang in languages}
    for lang, _di, _dj, row, hard_failed in raw_results:
        if hard_failed or row is None:
            failed += 1
            continue
        by_lang[lang].append(row)
        completed += 1

    for lang in languages:
        rows = _sort_rows_for_language(by_lang[lang])
        by_lang[lang] = rows

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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Max parallel LLM requests (default: env WRITER_MAX_WORKERS or 8 for Cohere, 1 for Ollama).",
    )
    args = parser.parse_args()

    generate_all_documents(
        test_mode=args.test,
        domain_filter=args.domain,
        language_filter=args.language,
        use_ollama=args.ollama or None,
        ollama_model=args.ollama_model,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
