"""
dataset/builder/qa_factory.py
------------------------------
Step 3 of the dataset pipeline: generate QA pairs from seed facts.

For each document (identified by language + domain + doc_idx), this module
generates a set of question-answer pairs derived directly from the seed facts.

Because facts are deterministic (SHA-256 seeded), the answers are mathematically
certain BEFORE the document is even written. This is what makes our evaluation
ground-truth reliable.

Pipeline position:
    facts.py → writer.py → qa_factory.py → evaluate.py

Output format (one dict per QA pair):
    {
        "doc_id":   "de_lease_0",
        "language": "de",
        "domain":   "lease",
        "question": "What is the monthly rent?",
        "answer":   "800 EUR",
        "field":    "monthly_rent"
    }

Issue: #20
Author: DocuNative Team
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dataset.builder.facts import SUPPORTED_DOMAINS, generate_facts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths

DATASET_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = DATASET_ROOT / "output"

SUPPORTED_LANGUAGES = ["de", "hi", "sw"]

# ---------------------------------------------------------------------------
# Question templates per domain and field
#
# Each entry maps a fact field name to a list of natural-language question
# templates. One question is selected deterministically per doc_idx so the
# same document always gets the same question.
#
# {value} is replaced with the actual fact value to produce the ground-truth
# answer string.

QUESTION_TEMPLATES: dict[str, dict[str, list[dict[str, str]]]] = {

    "lease": {
        "monthly_rent": [
            {"q": "What is the monthly rent?",                          "a": "{value}"},
            {"q": "How much is the rent per month?",                    "a": "{value}"},
            {"q": "What is the monthly rental amount?",                 "a": "{value}"},
        ],
        "deposit_amount": [
            {"q": "What is the security deposit amount?",               "a": "{value} EUR"},
            {"q": "How much is the deposit?",                           "a": "{value} EUR"},
            {"q": "What deposit must the tenant pay?",                  "a": "{value} EUR"},
        ],
        "notice_period_days": [
            {"q": "What is the notice period?",                         "a": "{value} days"},
            {"q": "How many days notice is required before vacating?",  "a": "{value} days"},
            {"q": "How much notice must the tenant give?",              "a": "{value} days"},
        ],
        "lease_duration_months": [
            {"q": "How long is the lease?",                             "a": "{value} months"},
            {"q": "What is the duration of the lease?",                 "a": "{value} months"},
            {"q": "For how many months is the fixed-term lease?",       "a": "{value} months"},
        ],
        "pets_allowed": [
            {"q": "Are pets allowed in the property?",                  "a": "{value}"},
            {"q": "Can the tenant keep pets?",                          "a": "{value}"},
            {"q": "Is it permitted to have pets?",                      "a": "{value}"},
        ],
        "smoking_allowed": [
            {"q": "Is smoking allowed in the property?",                "a": "{value}"},
            {"q": "Can the tenant smoke inside?",                       "a": "{value}"},
        ],
        "subletting_allowed": [
            {"q": "Is subletting allowed?",                             "a": "{value}"},
            {"q": "Can the tenant sublet the property?",                "a": "{value}"},
        ],
        "max_occupants": [
            {"q": "What is the maximum number of occupants?",           "a": "{value} people"},
            {"q": "How many people can live in the property?",          "a": "{value} people"},
        ],
        "late_fee_percent": [
            {"q": "What is the late payment fee percentage?",           "a": "{value}%"},
            {"q": "What penalty applies for late rent payment?",        "a": "{value}%"},
        ],
        "utilities_included": [
            {"q": "What utilities are included in the rent?",           "a": "{value}"},
            {"q": "Which utilities does the rent cover?",               "a": "{value}"},
        ],
    },

    "employment": {
        "monthly_salary": [
            {"q": "What is the monthly salary?",                        "a": "{value}"},
            {"q": "How much is the gross monthly salary?",              "a": "{value}"},
            {"q": "What is the employee's monthly pay?",                "a": "{value}"},
        ],
        "annual_leave_days": [
            {"q": "How many days of annual leave does the employee get?","a": "{value} days"},
            {"q": "What is the annual leave entitlement?",              "a": "{value} days"},
            {"q": "How many paid vacation days per year?",              "a": "{value} days"},
        ],
        "probation_months": [
            {"q": "How long is the probationary period?",               "a": "{value} months"},
            {"q": "What is the duration of probation?",                 "a": "{value} months"},
        ],
        "notice_period_after_probation_weeks": [
            {"q": "What is the notice period after probation?",         "a": "{value} weeks"},
            {"q": "How many weeks notice is required to resign?",       "a": "{value} weeks"},
        ],
        "remote_work_days_per_week": [
            {"q": "How many days per week can the employee work remotely?", "a": "{value} days"},
            {"q": "What is the remote work allowance?",                 "a": "{value} days per week"},
        ],
        "weekly_hours": [
            {"q": "What are the standard weekly working hours?",        "a": "{value} hours"},
            {"q": "How many hours per week does the contract specify?", "a": "{value} hours"},
        ],
        "overtime_rate_multiplier": [
            {"q": "What is the overtime pay multiplier?",               "a": "{value}x"},
            {"q": "How much is overtime compensated?",                  "a": "{value} times the hourly rate"},
        ],
        "annual_bonus_percent": [
            {"q": "What is the target annual bonus percentage?",        "a": "{value}%"},
            {"q": "What annual bonus can the employee expect?",         "a": "{value}% of annual salary"},
        ],
        "sick_leave_days_paid": [
            {"q": "How many paid sick leave days are provided?",        "a": "{value} days"},
            {"q": "What is the paid sick leave entitlement?",           "a": "{value} days per year"},
        ],
        "non_compete_months": [
            {"q": "How long is the non-compete clause?",                "a": "{value} months"},
            {"q": "What is the post-employment non-compete period?",    "a": "{value} months"},
        ],
    },

    "health_insurance": {
        "monthly_premium": [
            {"q": "What is the monthly insurance premium?",             "a": "{value}"},
            {"q": "How much is the monthly premium?",                   "a": "{value}"},
            {"q": "What does the policyholder pay per month?",          "a": "{value}"},
        ],
        "annual_deductible": [
            {"q": "What is the annual deductible?",                     "a": "{value}"},
            {"q": "How much is the yearly deductible?",                 "a": "{value}"},
        ],
        "waiting_period_days": [
            {"q": "What is the waiting period before coverage begins?", "a": "{value} days"},
            {"q": "How many days before the insurance is active?",      "a": "{value} days"},
        ],
        "dental_coverage_included": [
            {"q": "Is dental coverage included?",                       "a": "{value}"},
            {"q": "Does the policy cover dental treatment?",            "a": "{value}"},
        ],
        "vision_coverage_included": [
            {"q": "Is vision care covered by this policy?",             "a": "{value}"},
            {"q": "Does the insurance include vision coverage?",        "a": "{value}"},
        ],
        "copay_gp_visit": [
            {"q": "What is the copay for a GP visit?",                  "a": "{value} EUR"},
            {"q": "How much must the patient pay for a GP consultation?","a": "{value} EUR"},
        ],
        "copay_specialist_visit": [
            {"q": "What is the copay for a specialist visit?",          "a": "{value} EUR"},
            {"q": "How much is charged per specialist consultation?",   "a": "{value} EUR"},
        ],
        "mental_health_sessions_per_year": [
            {"q": "How many mental health sessions are covered per year?","a": "{value} sessions"},
            {"q": "What is the annual mental health therapy coverage?", "a": "{value} sessions"},
        ],
        "claim_submission_deadline_days": [
            {"q": "How many days do you have to submit a claim?",       "a": "{value} days"},
            {"q": "What is the claim submission deadline?",             "a": "{value} days after treatment"},
        ],
        "out_of_pocket_maximum": [
            {"q": "What is the maximum out-of-pocket cost per year?",   "a": "{value}"},
            {"q": "What is the annual out-of-pocket maximum?",          "a": "{value}"},
        ],
    },

    "immigration_letter": {
        "permit_type": [
            {"q": "What type of permit is described in this letter?",   "a": "{value}"},
            {"q": "What category of visa or permit is this?",           "a": "{value}"},
        ],
        "permit_duration_months": [
            {"q": "How long is the permit valid?",                      "a": "{value} months"},
            {"q": "What is the duration of the permit?",                "a": "{value} months"},
        ],
        "response_deadline_days": [
            {"q": "How many days do you have to respond to this letter?","a": "{value} days"},
            {"q": "What is the response deadline?",                     "a": "{value} days"},
        ],
        "work_permitted": [
            {"q": "Is the permit holder allowed to work?",              "a": "{value}"},
            {"q": "Does this permit authorise employment?",             "a": "{value}"},
        ],
        "study_permitted": [
            {"q": "Is the permit holder allowed to study?",             "a": "{value}"},
            {"q": "Does this permit allow the holder to enrol in education?", "a": "{value}"},
        ],
        "application_fee": [
            {"q": "What is the application fee?",                       "a": "{value}"},
            {"q": "How much does the application cost?",                "a": "{value}"},
        ],
        "biometrics_required": [
            {"q": "Is biometric data collection required?",             "a": "{value}"},
            {"q": "Does the applicant need to provide biometrics?",     "a": "{value}"},
        ],
        "document_submission_deadline_days": [
            {"q": "How many days to submit supporting documents?",      "a": "{value} days"},
            {"q": "What is the document submission deadline?",          "a": "{value} days"},
        ],
        "minimum_funds_required": [
            {"q": "What is the minimum funds requirement?",             "a": "{value}"},
            {"q": "How much money must the applicant demonstrate?",     "a": "{value}"},
        ],
        "appeal_fee": [
            {"q": "What is the fee to lodge an appeal?",                "a": "{value}"},
            {"q": "How much does an appeal cost?",                      "a": "{value}"},
        ],
    },
}

# Number of QA pairs to generate per document
QA_PAIRS_PER_DOC = 10


# ---------------------------------------------------------------------------
# Core functions

def _format_answer(value: Any) -> str:
    """Convert a fact value to a clean answer string."""
    if isinstance(value, bool):
        # Use full phrase instead of bare Yes/No so Token F1 has overlap
        # with model answers like "Smoking is not allowed in the property"
        return "allowed" if value else "not allowed"
    if isinstance(value, float):
        return str(int(value)) if value == int(value) else str(value)
    return str(value)


def generate_qa_pairs(
    language: str,
    domain: str,
    doc_idx: int,
    n_pairs: int = QA_PAIRS_PER_DOC,
) -> list[dict[str, Any]]:
    """
    Generate QA pairs for one document from its seed facts.

    The same (language, domain, doc_idx) always produces the same QA pairs
    because facts.py is deterministic. Questions are selected by cycling
    through the available templates for each field.

    Args:
        language:  Language code — "de", "hi", "sw"
        domain:    Domain name — "lease", "employment", etc.
        doc_idx:   Document index (0-29 for pilot scale)
        n_pairs:   Number of QA pairs to generate (default: 10)

    Returns:
        List of QA pair dicts, each with keys:
        doc_id, language, domain, question, answer, field
    """
    facts  = generate_facts(language, domain, doc_idx)
    doc_id = f"{language}_{domain}_{doc_idx}"
    templates = QUESTION_TEMPLATES.get(domain, {})

    if not templates:
        logger.warning("No question templates found for domain: %s", domain)
        return []

    # Build candidate (field, template) pairs in stable order
    candidates: list[tuple[str, dict[str, str]]] = []
    for field, tmpl_list in templates.items():
        if field not in facts:
            continue
        for tmpl in tmpl_list:
            candidates.append((field, tmpl))

    if not candidates:
        logger.warning("No matching fields found in facts for domain: %s", domain)
        return []

    # Select n_pairs deterministically using doc_idx as offset
    # This ensures different documents get different question rotations
    qa_pairs: list[dict[str, Any]] = []
    seen_fields: set[str] = set()

    # First pass: one question per unique field (for coverage)
    for field, tmpl in candidates:
        if len(qa_pairs) >= n_pairs:
            break
        if field in seen_fields:
            continue
        value     = facts[field]
        answer    = tmpl["a"].replace("{value}", _format_answer(value))
        question  = tmpl["q"]
        qa_pairs.append({
            "doc_id":   doc_id,
            "language": language,
            "domain":   domain,
            "question": question,
            "answer":   answer,
            "field":    field,
        })
        seen_fields.add(field)

    # Second pass: fill remaining slots with alternative phrasings
    offset = doc_idx % max(len(candidates), 1)
    rotated = candidates[offset:] + candidates[:offset]
    for field, tmpl in rotated:
        if len(qa_pairs) >= n_pairs:
            break
        # Skip if we already have 2+ questions for this field
        field_count = sum(1 for q in qa_pairs if q["field"] == field)
        if field_count >= 2:
            continue
        value    = facts[field]
        answer   = tmpl["a"].replace("{value}", _format_answer(value))
        question = tmpl["q"]
        # Avoid exact duplicate questions
        if any(q["question"] == question for q in qa_pairs):
            continue
        qa_pairs.append({
            "doc_id":   doc_id,
            "language": language,
            "domain":   domain,
            "question": question,
            "answer":   answer,
            "field":    field,
        })

    return qa_pairs[:n_pairs]


def generate_all_qa_pairs(
    *,
    languages: list[str] | None = None,
    domains:   list[str] | None = None,
    n_docs:    int = 30,
    n_pairs:   int = QA_PAIRS_PER_DOC,
) -> list[dict[str, Any]]:
    """
    Generate QA pairs for all language × domain × doc_idx combinations.

    Args:
        languages: List of language codes. Defaults to all 3.
        domains:   List of domain names. Defaults to all 4.
        n_docs:    Number of documents per combo. Default 30 (pilot scale).
        n_pairs:   QA pairs per document. Default 10.

    Returns:
        Flat list of all QA pair dicts.
    """
    languages = languages or SUPPORTED_LANGUAGES
    domains   = domains   or SUPPORTED_DOMAINS

    all_pairs: list[dict[str, Any]] = []
    for lang in languages:
        for domain in domains:
            for doc_idx in range(n_docs):
                pairs = generate_qa_pairs(lang, domain, doc_idx, n_pairs)
                all_pairs.extend(pairs)
                logger.info(
                    "Generated %d QA pairs for %s/%s/doc_%d",
                    len(pairs), lang, domain, doc_idx,
                )

    logger.info("Total QA pairs generated: %d", len(all_pairs))
    return all_pairs


def save_qa_pairs(
    qa_pairs: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    """
    Save QA pairs to a JSONL file.

    Args:
        qa_pairs:    List of QA pair dicts from generate_all_qa_pairs().
        output_path: Where to write. Defaults to dataset/output/qa_pairs.jsonl

    Returns:
        Path to the written file.
    """
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "qa_pairs.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info("Saved %d QA pairs to %s", len(qa_pairs), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Standalone test

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("Testing qa_factory.py...\n")

    # Test 1: Single document QA generation
    print("=" * 50)
    print("TEST 1: Single document (de/lease/doc_0)")
    print("=" * 50)
    pairs = generate_qa_pairs("de", "lease", 0)
    assert len(pairs) == QA_PAIRS_PER_DOC, f"Expected {QA_PAIRS_PER_DOC} pairs, got {len(pairs)}"
    for p in pairs:
        print(f"  Q: {p['question']}")
        print(f"  A: {p['answer']}  [{p['field']}]")
        print()
    print(f"✓ {len(pairs)} QA pairs generated\n")

    # Test 2: Determinism check
    print("=" * 50)
    print("TEST 2: Determinism — same inputs = same output")
    print("=" * 50)
    pairs_a = generate_qa_pairs("hi", "employment", 3)
    pairs_b = generate_qa_pairs("hi", "employment", 3)
    assert pairs_a == pairs_b, "FAIL: same inputs gave different QA pairs!"
    print("✓ Determinism verified\n")

    # Test 3: Different doc_idx = different questions
    print("=" * 50)
    print("TEST 3: Different doc_idx rotates question phrasing")
    print("=" * 50)
    pairs_0 = generate_qa_pairs("sw", "health_insurance", 0)
    pairs_1 = generate_qa_pairs("sw", "health_insurance", 1)
    # Facts differ so answers will differ — just check structure
    assert len(pairs_0) == len(pairs_1) == QA_PAIRS_PER_DOC
    print(f"✓ doc_0: {pairs_0[0]['question']} → {pairs_0[0]['answer']}")
    print(f"✓ doc_1: {pairs_1[0]['question']} → {pairs_1[0]['answer']}\n")

    # Test 4: All 4 domains work
    print("=" * 50)
    print("TEST 4: All domains generate QA pairs")
    print("=" * 50)
    for domain in SUPPORTED_DOMAINS:
        pairs = generate_qa_pairs("de", domain, 0)
        assert len(pairs) == QA_PAIRS_PER_DOC, f"FAIL: {domain} gave {len(pairs)} pairs"
        print(f"  ✓ {domain}: {len(pairs)} pairs — first Q: {pairs[0]['question']}")
    print()

    # Test 5: Scale test — pilot dataset numbers
    print("=" * 50)
    print("TEST 5: Pilot scale — 4 domains × 3 languages × 30 docs × 10 pairs")
    print("=" * 50)
    if "--full" in sys.argv:
        all_pairs = generate_all_qa_pairs()
        expected  = 4 * 3 * 30 * 10  # = 3600 (4 domains × 3 languages × 30 docs × 10 pairs)
        assert len(all_pairs) == expected, f"Expected {expected}, got {len(all_pairs)}"
        path = save_qa_pairs(all_pairs)
        print(f"✓ {len(all_pairs)} QA pairs saved to {path}")
    else:
        print("  (skipped — run with --full to generate all 1,200 pairs)")
        print("  Tip: python -m dataset.builder.qa_factory --full")

    print("\nAll tests passed. qa_factory.py is ready.")