"""
tests/test_smoke.py
-------------------
Smoke tests for the DocuNative pipeline.

These tests verify that all modules import correctly, the core logic
works as expected, and nothing is silently broken before demo day.

They do NOT require llama-server to be running — generation is mocked.
They do NOT require BGE-M3 or mDeBERTa to be downloaded — those are
tested via integration, not unit tests.

Run with:
    pytest -q

Expected output on a healthy codebase:
    ........  8 passed in Xs
"""

import pytest
from pathlib import Path

# ── Path to the test fixture PDF ─────────────────────────────────────────────
SAMPLE_PDF = Path(__file__).parent / "sample_lease_de.pdf"


# ═════════════════════════════════════════════════════════════════════════════
# 1. Import smoke tests — catches circular imports and missing dependencies
# ═════════════════════════════════════════════════════════════════════════════

def test_imports_extract():
    """extract.py imports cleanly."""
    from pipeline.extract import extract_chunks, chunk_text, clean_text
    assert callable(extract_chunks)

def test_imports_validate():
    """validate.py imports cleanly and exports expected symbols."""
    from pipeline.validate import parse_output, is_answer_missing, ParsedOutput
    assert callable(parse_output)
    assert callable(is_answer_missing)

def test_imports_pipeline():
    """pipeline.py imports cleanly and PipelineResult is well-formed."""
    from pipeline.pipeline import PipelineResult
    result = PipelineResult(
        answer="test",
        source_quote="test",
        nli_verdict="neutral",
        model="Global",
        parse_success=True,
    )
    assert result.nli_emoji.startswith("⚠️")


# ═════════════════════════════════════════════════════════════════════════════
# 2. extract.py unit tests
# ═════════════════════════════════════════════════════════════════════════════

def test_extract_pdf_returns_chunks():
    """Real PDF extracts to at least one non-empty chunk."""
    from pipeline.extract import extract_chunks
    assert SAMPLE_PDF.exists(), f"Test fixture missing: {SAMPLE_PDF}"
    chunks = extract_chunks(str(SAMPLE_PDF))
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)

def test_extract_chunk_size_within_limit():
    """No chunk exceeds the configured size limit (with 10% tolerance)."""
    from pipeline.extract import extract_chunks, CHUNK_SIZE_CHARS
    chunks = extract_chunks(str(SAMPLE_PDF))
    for i, chunk in enumerate(chunks):
        assert len(chunk) <= CHUNK_SIZE_CHARS * 1.1, (
            f"Chunk {i} is {len(chunk)} chars — exceeds limit of {CHUNK_SIZE_CHARS}"
        )

def test_extract_unsupported_format_raises():
    """Unsupported file type raises ValueError."""
    from pipeline.extract import extract_chunks
    with pytest.raises((ValueError, FileNotFoundError)):
        extract_chunks("document.docx")


# ═════════════════════════════════════════════════════════════════════════════
# 3. validate.py unit tests — the most critical module for demo reliability
# ═════════════════════════════════════════════════════════════════════════════

def test_validate_perfect_output():
    """Happy path: both fields present with brackets."""
    from pipeline.validate import parse_output
    raw = "Answer: [The rent is 1,350 EUR per month]\nSource_Quote: [§ 2 Der Mietzins beträgt 1.350 EUR]"
    result = parse_output(raw)
    assert result.parse_success is True
    assert result.answer == "The rent is 1,350 EUR per month"
    assert result.source_quote == "§ 2 Der Mietzins beträgt 1.350 EUR"

def test_validate_lowercase_labels():
    """Model ignoring capitalisation is handled."""
    from pipeline.validate import parse_output
    raw = "answer: [The deposit is 2,700 EUR]\nsource_quote: [Die Kaution beträgt 2.700 EUR]"
    result = parse_output(raw)
    assert result.parse_success is True
    assert "2,700" in result.answer

def test_validate_missing_closing_bracket():
    """Missing closing bracket falls back gracefully."""
    from pipeline.validate import parse_output
    raw = "Answer: [The lease starts April 1, 2025\nSource_Quote: [Der Mietvertrag beginnt am 1. April 2025]"
    result = parse_output(raw)
    assert result.answer != ""
    assert "April" in result.answer

def test_validate_excerpt_hallucination_guard():
    """Model outputting 'Excerpt N' is caught and replaced."""
    from pipeline.validate import parse_output
    raw = "Answer: [Excerpt 2]\nSource_Quote: [N/A]"
    result = parse_output(raw)
    assert result.parse_success is True
    assert "does not contain" in result.answer.lower()

def test_validate_complete_failure_preserved():
    """Complete parse failure preserves raw_output and sets parse_success=False."""
    from pipeline.validate import parse_output
    raw = "Sorry I cannot help with that."
    result = parse_output(raw)
    assert result.parse_success is False
    assert result.raw_output == raw

def test_validate_empty_input():
    """Empty input handled without raising."""
    from pipeline.validate import parse_output
    result = parse_output("")
    assert result.parse_success is False

def test_is_answer_missing_detects_not_found():
    """is_answer_missing() returns True for all known not-found phrasings."""
    from pipeline.validate import parse_output, is_answer_missing
    not_found_phrases = [
        "The document does not contain information to answer this question.",
        "This cannot be found in the provided excerpts.",
        "No information is available.",
        "Not mentioned in the document.",
        "Not provided in the excerpts.",
    ]
    for phrase in not_found_phrases:
        raw = f"Answer: [{phrase}]\nSource_Quote: [N/A]"
        parsed = parse_output(raw)
        assert is_answer_missing(parsed), (
            f"is_answer_missing() failed to detect: {phrase!r}"
        )

def test_is_answer_missing_false_for_real_answer():
    """is_answer_missing() returns False when a real answer is present."""
    from pipeline.validate import parse_output, is_answer_missing
    raw = "Answer: [The monthly rent is 1,350 EUR]\nSource_Quote: [Der Mietzins beträgt 1.350 EUR]"
    parsed = parse_output(raw)
    assert is_answer_missing(parsed) is False


# ═════════════════════════════════════════════════════════════════════════════
# 4. metrics.py unit tests
# ═════════════════════════════════════════════════════════════════════════════

def test_metrics_f1_exact_match():
    """Identical answer and ground truth → F1 = 1.0."""
    from eval.metrics import calculate_f1_score
    from pipeline.validate import ParsedOutput
    parsed = ParsedOutput(
        answer="the rent is 1350 euros per month",
        source_quote="",
        parse_success=True,
        raw_output=""
    )
    f1 = calculate_f1_score(parsed, "the rent is 1350 euros per month")
    assert f1 == 1.0

def test_metrics_f1_no_overlap():
    """Completely different tokens → F1 = 0.0."""
    from eval.metrics import calculate_f1_score
    from pipeline.validate import ParsedOutput
    parsed = ParsedOutput(
        answer="cats and dogs",
        source_quote="",
        parse_success=True,
        raw_output=""
    )
    f1 = calculate_f1_score(parsed, "rent deposit lease")
    assert f1 == 0.0

def test_metrics_f1_uses_sets_not_lists():
    """Repeated tokens don't inflate F1 — set-based scoring."""
    from eval.metrics import calculate_f1_score
    from pipeline.validate import ParsedOutput
    # Answer repeats "the" 5 times — should not boost F1
    parsed = ParsedOutput(
        answer="the the the the the rent",
        source_quote="",
        parse_success=True,
        raw_output=""
    )
    f1_repeated = calculate_f1_score(parsed, "the rent is 1350")
    parsed_clean = ParsedOutput(
        answer="the rent",
        source_quote="",
        parse_success=True,
        raw_output=""
    )
    f1_clean = calculate_f1_score(parsed_clean, "the rent is 1350")
    # Both should give same F1 — repeated tokens don't help
    assert f1_repeated == f1_clean

def test_metrics_recall3_hit():
    """Ground truth tokens found in chunks → recall = 1."""
    from eval.metrics import calculate_recall_3
    chunks = [
        "The monthly rent is 1350 euros payable on the first of each month.",
        "The security deposit is 2700 euros.",
        "Pets are not permitted without written consent.",
    ]
    assert calculate_recall_3(chunks, "monthly rent 1350 euros") == 1

def test_metrics_recall3_miss():
    """Ground truth not in any chunk → recall = 0."""
    from eval.metrics import calculate_recall_3
    chunks = [
        "The security deposit is 2700 euros.",
        "Pets are not permitted.",
        "The lease begins April 2025.",
    ]
    assert calculate_recall_3(chunks, "swimming pool access hours morning") == 0

def test_metrics_per_language_breakdown():
    """per_language_breakdown returns correct structure for all languages."""
    from eval.metrics import per_language_breakdown
    results = [
        {"language": "de", "f1_score": 0.85, "recall_3": 1, "nli_result": "entailment"},
        {"language": "de", "f1_score": 0.70, "recall_3": 1, "nli_result": "neutral"},
        {"language": "hi", "f1_score": 0.55, "recall_3": 0, "nli_result": "contradiction"},
        {"language": "sw", "f1_score": 0.40, "recall_3": 1, "nli_result": "neutral"},
    ]
    breakdown = per_language_breakdown(results)
    assert set(breakdown.keys()) == {"de", "hi", "sw"}
    assert breakdown["de"]["avg_f1"] == 0.78
    assert breakdown["de"]["total_questions"] == 2
    assert breakdown["de"]["recall_at_3"] == 1.0
    assert breakdown["hi"]["contradiction_percentage"] == 1.0

def test_metrics_empty_results():
    """Empty results list returns empty dict without raising."""
    from eval.metrics import per_language_breakdown
    assert per_language_breakdown([]) == {}
