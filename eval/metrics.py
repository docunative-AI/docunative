"""
eval/metrics.py
---------------
Computes all evaluation metrics for DocuNative:
  - Token F1 (SQuAD-style)
  - Exact Match (EM) — added for academic rigour on factual extraction
  - Recall@3 (retrieval quality)
  - Refusal Rate — how often model says "not found"
  - NLI label distribution (entailment / neutral / contradiction)
  - Per-language breakdown (H1 + H2 analysis)

Language-aware tokenization:
  - Chinese (zh): character-level — Chinese has no word spaces, .split() breaks F1
  - Hindi (hi): word-level (space-separated Devanagari works correctly)
  - Polish (pl): word-level (standard Latin script)

Number normalisation:
  - German/Polish decimal suffix: 1.350,00 → 1350
  - Polish space thousands: 1 350 → 1350
  - Indian numbering: 1,50,000 → 150000
  - Western thousands: 1,350 / 1.350 → 1350
  - Currencies: EUR, PLN, CNY, INR, ZŁ all lowercased

Author: DocuNative Team
Updated: March 21, 2026 — zh/hi/pl language set, EM + Refusal Rate added
"""

import re
from collections import Counter
from pipeline.validate import ParsedOutput


# ---------------------------------------------------------------------------
# Language-aware tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str, lang: str = "") -> list[str]:
    """
    Language-aware tokenizer.

    Chinese (zh): character-level — Chinese sentences have no spaces.
    .split() treats an entire sentence as one token, making F1 and Recall@3
    near 0% for Chinese. Character-level tokenization is the standard approach
    for Chinese NLP evaluation (used in CMRC, DRCD, and MKQA papers).

    All other languages: word-level (.split()) — standard SQuAD approach.

    Args:
        text: Already normalised and lowercased text
        lang: Language code ("zh", "hi", "pl", etc.)

    Returns:
        List of tokens (characters for zh, words for others)
    """
    if lang == "zh":
        # Remove spaces then split into individual characters
        # Each Chinese character is semantically a token
        return list(text.replace(" ", ""))
    return text.split()


# ---------------------------------------------------------------------------
# Number and currency normalisation
# ---------------------------------------------------------------------------

# Not-found phrases used to detect refusals across all three languages
_NOT_FOUND_PHRASES = [
    # English
    "does not contain information",
    "cannot be found",
    "not found in the document",
    "n/a",
    "not mentioned",
    "no information",
    "not available",
    "does not mention",
    "no relevant information",
    "unable to find",
    "not provided",
    "not specified",
    "not stated",
    "could not find",
    # German (residual — kept for backward compat with old eval results)
    "nicht im dokument",
    "keine information",
    "nicht gefunden",
    # Hindi
    "दस्तावेज़ में नहीं",
    "जानकारी नहीं",
    "नहीं मिला",
    # Chinese
    "文件中没有",
    "文件中不包含",
    "未提及",
    "无法找到",
    "文件中无相关信息",
    # Polish
    "nie zawiera informacji",
    "nie znaleziono",
    "nie wspomniano",
    "brak informacji",
    "nie podano",
    "nie określono",
]


def _is_refusal(answer: str) -> bool:
    """
    Returns True if the model's answer is a refusal / not-found response.
    Used to compute Refusal Rate separately from wrong answers.
    """
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in _NOT_FOUND_PHRASES)


def normalise_for_eval(text: str) -> str:
    """
    Normalise numbers and currency strings for fair token comparison.

    Handles four number formatting conventions:
      1. German/Polish decimal suffix (,00): "1.350,00" → "1350"
      2. Polish space thousands:            "1 350"     → "1350"
      3. Indian numbering system:           "1,50,000"  → "150000"
      4. Western comma/period thousands:    "1,350"     → "1350"

    Also lowercases currency codes (EUR, PLN, CNY, INR, ZŁ, etc.)

    Why this matters:
      - Polish uses spaces for thousands (1 350 PLN) and commas for decimals
      - Indian numbering places commas at 2-digit intervals (1,50,000 INR)
      - Chinese documents may express amounts in Arabic numerals or Chinese chars
      - Without normalisation, "1 350 PLN" and "1350 PLN" score F1=0

    Args:
        text: Raw text string (any language)

    Returns:
        Normalised string ready for tokenisation and comparison
    """
    # Step 1: Strip German/Polish decimal suffix ,00 or .00 FIRST
    # "1.350,00 EUR" → "1.350 EUR"   "1 350,00 PLN" → "1 350 PLN"
    text = re.sub(r'(\d)[.,](00)\b', r'\1', text)

    # Step 2: Strip Polish space-based thousand separators
    # "1 350 PLN" → "1350 PLN"   "1 234 567" → "1234567"
    text = re.sub(r'(\d)\s(\d{3})\b', r'\1\2', text)

    # Step 3: Strip Western and Indian comma/period separators
    # Handles both 3-digit groups (Western: 1,350) and 2-digit groups (Indian: 1,50,000)
    # Use a loop to collapse repeated separators (1,50,000 needs two passes)
    for _ in range(3):
        text = re.sub(r'(\d)[.,](\d{2,3})\b', r'\1\2', text)

    # Step 4: Lowercase currency codes
    text = re.sub(
        r'\b(EUR|USD|GBP|INR|PLN|CNY|RMB|ZŁ|ZL|TSH|KSH|TShs|KShs)\b',
        lambda m: m.group().lower(),
        text,
        flags=re.IGNORECASE,
    )

    return text


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def calculate_f1_score(
    parsed_llm_output: ParsedOutput,
    ground_truth: str,
    lang: str = "",
) -> float:
    """
    Compute token-level F1 score (standard SQuAD definition).

    Uses language-aware tokenization:
    - Chinese (zh): character-level (no spaces in Chinese)
    - All others: word-level (.split())

    Formula:
        precision = overlap / predicted_tokens
        recall    = overlap / ground_truth_tokens
        F1        = 2 * precision * recall / (precision + recall)

    Args:
        parsed_llm_output: ParsedOutput from validate.py
        ground_truth:      Correct answer string from seed facts
        lang:              Language code ("zh", "hi", "pl") — affects tokenization

    Returns:
        Float in [0.0, 1.0]
    """
    llm_answer   = normalise_for_eval(parsed_llm_output.answer.lower())
    ground_truth = normalise_for_eval(ground_truth.lower())

    pred_tokens = Counter(_tokenize(llm_answer, lang))
    gt_tokens   = Counter(_tokenize(ground_truth, lang))

    overlap = sum((pred_tokens & gt_tokens).values())

    if overlap == 0:
        return 0.0

    precision = overlap / sum(pred_tokens.values())
    recall    = overlap / sum(gt_tokens.values())

    return round((2 * precision * recall) / (precision + recall), 3)


def calculate_exact_match(
    parsed_llm_output: ParsedOutput,
    ground_truth: str,
    lang: str = "",
) -> int:
    """
    Compute Exact Match (EM) score.

    EM = 1 if the normalised ground truth string appears anywhere in the
    normalised model answer, otherwise 0.

    Why substring matching instead of full equality?
    Our model answers in complete sentences ("The monthly rent is 1350 EUR")
    while ground truth is a bare fact ("1350 EUR"). Full string equality
    would always give EM=0. Substring match correctly captures whether the
    model included the correct fact in its answer.

    EM is a harsher metric than F1 and is standard for factual extraction tasks.
    It is particularly useful for Eval 1 where ground truth is a deterministic
    seed fact value (e.g. "1350", "allowed", "30 days").

    Args:
        parsed_llm_output: ParsedOutput from validate.py
        ground_truth:      Correct answer string from seed facts
        lang:              Language code — used for normalisation

    Returns:
        1 if exact match found, 0 otherwise
    """
    answer_norm = normalise_for_eval(parsed_llm_output.answer.lower())
    gt_norm     = normalise_for_eval(ground_truth.lower())

    # For Chinese: check character sequence containment
    # For others: check token-level containment (ground truth tokens all in answer)
    if lang == "zh":
        return 1 if gt_norm.replace(" ", "") in answer_norm.replace(" ", "") else 0

    # Word-level: all ground truth tokens must appear in the answer
    gt_tokens     = set(_tokenize(gt_norm, lang))
    answer_tokens = set(_tokenize(answer_norm, lang))
    return 1 if gt_tokens.issubset(answer_tokens) else 0


def calculate_recall_3(
    list_chunks: list,
    ground_truth: str,
    lang: str = "",
) -> int:
    """
    Compute Recall@3 — whether the retriever found the right chunk.

    Returns 1 if ≥50% of ground truth tokens appear in any of the top-3
    retrieved chunks, 0 otherwise.

    Uses language-aware tokenization:
    - Chinese (zh): character-level overlap check
    - All others: word-level (standard)

    Uses number normalisation before tokenising — so "1 350 PLN" in a Polish
    document chunk matches an English ground truth of "1350 PLN".

    Args:
        list_chunks:   Top-k retrieved chunks (strings) from retrieve.py
        ground_truth:  Correct answer string from seed facts
        lang:          Language code — affects tokenization

    Returns:
        1 if retriever found the right chunk, 0 if not
    """
    gt_tokens = Counter(_tokenize(
        normalise_for_eval(ground_truth.lower()), lang
    ))

    if not gt_tokens:
        return 0

    gt_total = sum(gt_tokens.values())

    for chunk in list_chunks[:3]:
        chunk_tokens = Counter(_tokenize(
            normalise_for_eval(chunk.lower()), lang
        ))
        overlap = sum((gt_tokens & chunk_tokens).values())
        if overlap / gt_total >= 0.5:
            return 1

    return 0


def nli_label_distribution(list_nli_checking_result: list) -> dict:
    """
    Compute NLI label distribution across a list of result dicts.

    Args:
        list_nli_checking_result: List of result dicts with "nli_label" key

    Returns:
        Dict with entailment_percentage, neutral_percentage,
        contradiction_percentage (all floats in [0, 1])
    """
    if not list_nli_checking_result:
        return {
            "entailment_percentage":    0.0,
            "neutral_percentage":       0.0,
            "contradiction_percentage": 0.0,
        }

    total = len(list_nli_checking_result)

    def _label(d):
        return d.get("nli_label", d.get("nli_result", "")).lower()

    entailment    = sum(1 for d in list_nli_checking_result if _label(d) == "entailment")
    neutral       = sum(1 for d in list_nli_checking_result if _label(d) == "neutral")
    contradiction = sum(1 for d in list_nli_checking_result if _label(d) == "contradiction")

    return {
        "entailment_percentage":    round(entailment    / total, 3),
        "neutral_percentage":       round(neutral       / total, 3),
        "contradiction_percentage": round(contradiction / total, 3),
    }


def per_language_breakdown(results: list) -> dict:
    """
    Group evaluation results by language and compute all metrics per language.

    Metrics returned per language:
      - avg_f1:                   Average Token F1
      - avg_em:                   Average Exact Match (NEW)
      - recall_at_3:              Proportion where retriever found right chunk
      - refusal_rate:             Proportion where model said "not found" (NEW)
      - entailment_percentage:    NLI entailment rate
      - neutral_percentage:       NLI neutral rate
      - contradiction_percentage: NLI contradiction rate
      - total_questions:          Total QA pairs evaluated

    The refusal_rate is separate from F1/EM — a refusal is neither correct
    nor wrong in the same way as a hallucinated answer. Knowing the refusal
    rate per language tells us whether the model loses confidence in lower-
    resource languages (more refusals) or hallucinates more (lower F1, same
    refusal rate).

    Args:
        results: List of result dicts from evaluate.py run_single_eval()

    Returns:
        Dict keyed by language code
    """
    if not results:
        return {}

    import logging
    logger = logging.getLogger(__name__)

    grouped: dict[str, list] = {}
    for item in results:
        lang = item.get("language", "unknown")
        grouped.setdefault(lang, []).append(item)

    breakdown = {}
    for lang, items in grouped.items():
        total = len(items)

        # Warn on missing fields
        missing_f1 = sum(1 for r in items if "f1_score" not in r)
        if missing_f1:
            logger.warning(
                "Language '%s': %d results missing 'f1_score', treated as 0.0",
                lang, missing_f1,
            )

        avg_f1      = round(sum(r.get("f1_score", 0.0) for r in items) / total, 3)
        avg_em      = round(sum(r.get("em_score",  0)   for r in items) / total, 3)
        recall_at_3 = round(sum(r.get("recall_3",  0)   for r in items) / total, 3)

        # Refusal rate — proportion where model said "not found"
        # Uses the prediction field (model's actual answer text)
        refusals    = sum(1 for r in items if _is_refusal(r.get("prediction", "")))
        refusal_rate = round(refusals / total, 3)

        nli_dist = nli_label_distribution(items)

        breakdown[lang] = {
            "avg_f1":                   avg_f1,
            "avg_em":                   avg_em,
            "recall_at_3":              recall_at_3,
            "refusal_rate":             refusal_rate,
            "entailment_percentage":    nli_dist["entailment_percentage"],
            "neutral_percentage":       nli_dist["neutral_percentage"],
            "contradiction_percentage": nli_dist["contradiction_percentage"],
            "total_questions":          total,
        }

    return breakdown


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing metrics.py — zh/hi/pl language set\n")

    # Test 1: Chinese character-level tokenization
    print("=" * 50)
    print("TEST 1: Chinese character-level F1")
    print("=" * 50)
    parsed = ParsedOutput(
        answer="月租金为一千三百五十欧元",
        source_quote="",
        parse_success=True,
        raw_output=""
    )
    gt = "一千三百五十欧元"
    f1 = calculate_f1_score(parsed, gt, lang="zh")
    print(f"  Answer: {parsed.answer}")
    print(f"  GT:     {gt}")
    print(f"  F1:     {f1}  (expected > 0 — character overlap)")
    assert f1 > 0, "Chinese F1 should be > 0 with character tokenization"
    print("  ✓ PASS\n")

    # Test 2: Polish number normalisation
    print("=" * 50)
    print("TEST 2: Polish space-thousands normalisation")
    print("=" * 50)
    parsed2 = ParsedOutput(
        answer="Czynsz wynosi 1 350 PLN miesięcznie",
        source_quote="",
        parse_success=True,
        raw_output=""
    )
    gt2 = "1350 PLN"
    f1_2 = calculate_f1_score(parsed2, gt2, lang="pl")
    em_2 = calculate_exact_match(parsed2, gt2, lang="pl")
    print(f"  Answer: {parsed2.answer}")
    print(f"  GT:     {gt2}")
    print(f"  F1:     {f1_2}  EM: {em_2}  (expected > 0 after normalisation)")
    assert f1_2 > 0, "Polish space-thousands should normalise correctly"
    print("  ✓ PASS\n")

    # Test 3: Exact Match
    print("=" * 50)
    print("TEST 3: Exact Match")
    print("=" * 50)
    parsed3 = ParsedOutput(
        answer="The monthly rent is 1350 EUR per month",
        source_quote="",
        parse_success=True,
        raw_output=""
    )
    em_hit  = calculate_exact_match(parsed3, "1350 EUR", lang="")
    em_miss = calculate_exact_match(parsed3, "2700 EUR", lang="")
    print(f"  EM hit  (1350 EUR in answer): {em_hit}   (expected 1)")
    print(f"  EM miss (2700 EUR in answer): {em_miss}  (expected 0)")
    assert em_hit == 1 and em_miss == 0
    print("  ✓ PASS\n")

    # Test 4: Refusal Rate detection
    print("=" * 50)
    print("TEST 4: Refusal Rate")
    print("=" * 50)
    refusal_results = [
        {"language": "zh", "prediction": "The document does not contain information to answer this question.", "f1_score": 0.0, "em_score": 0, "recall_3": 0, "nli_label": "neutral"},
        {"language": "zh", "prediction": "文件中没有相关信息", "f1_score": 0.0, "em_score": 0, "recall_3": 0, "nli_label": "neutral"},
        {"language": "zh", "prediction": "The monthly rent is 1350 CNY", "f1_score": 0.9, "em_score": 1, "recall_3": 1, "nli_label": "entailment"},
    ]
    bd = per_language_breakdown(refusal_results)
    refusal_rate = bd["zh"]["refusal_rate"]
    print(f"  Refusal rate for zh: {refusal_rate}  (expected ~0.667 — 2 of 3 refused)")
    assert refusal_rate > 0.5
    print("  ✓ PASS\n")

    # Test 5: Recall@3 with Chinese character overlap
    print("=" * 50)
    print("TEST 5: Recall@3 with Chinese characters")
    print("=" * 50)
    chunks_zh = [
        "本合同规定月租金为一千三百五十欧元",
        "租客需缴纳两个月租金作为押金",
        "合同期限为十二个月",
    ]
    gt_zh = "一千三百五十欧元"
    r3 = calculate_recall_3(chunks_zh, gt_zh, lang="zh")
    print(f"  GT: {gt_zh}")
    print(f"  Recall@3: {r3}  (expected 1 — characters overlap with chunk 1)")
    assert r3 == 1
    print("  ✓ PASS\n")

    print("All tests passed. metrics.py is ready for zh/hi/pl.")
