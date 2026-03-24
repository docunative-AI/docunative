import logging
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers

# Mute the harmless "UNEXPECTED position_ids" warning for a clean demo terminal
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Check for device availability once at module level (cheap operation)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ---------------------------------------------------------------------------
# Lazy singleton — model loads on first nli_validation() call, not at import.
# This mirrors the embed.py pattern and prevents blocking Gradio startup.

_tokenizer = None
_model = None
_label_mapping = None


def _get_nli_model():
    """Load and cache the mDeBERTa NLI model on first call only."""
    global _tokenizer, _model, _label_mapping
    if _model is None:
        logger.info("NLI [1/3] loading model and tokenizer on %s (first call)...", device)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        if device == "cpu":
            # INT8 dynamic quantization on CPU — 2x faster, ~0.5% quality loss
            # mDeBERTa doesn't support fp16, so INT8 is the right CPU optimisation
            import torch.quantization
            _model = torch.quantization.quantize_dynamic(
                base_model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            logger.info("NLI model quantized to INT8 for CPU inference.")
        else:
            _model = base_model
            _model.to(device)

        _model.eval()
        _label_mapping = _model.config.id2label
        logger.info("NLI model loaded on %s.", device)
    return _tokenizer, _model, _label_mapping


def _is_numerical_answer(text: str) -> bool:
    """
    Returns True if the text is primarily a number or currency amount.
    NLI struggles with numeric format differences (1,350 vs 1.350 vs 1350).
    Bypass NLI for these to avoid false contradiction badges.
    """
    cleaned = text.strip()
    return bool(re.match(
        r'^[\d.,]+\s*(%|EUR|USD|GBP|INR|TSH|KSH|days?|months?|weeks?|hours?|people)?$',
        cleaned, re.IGNORECASE
    ))


def nli_validation(
        list_premises: list,
        llm_answer: str
) -> list[dict]:
    """
    Check whether each retrieved source chunk supports the LLM's answer.

    NLI (Natural Language Inference) frames the problem as:
      Given a PREMISE (what the document says),
      does the HYPOTHESIS (what the model claimed) follow from it?

    Three possible relationships:
      entailment    = the premise logically supports the hypothesis
      neutral       = the premise is unrelated or can't confirm
      contradiction = the premise says the opposite of hypothesis

    Args:
        list_premises:  Retrieved source chunks from the document.
                        These are the text passages your retriever found.
                        Typically 1 chunk per query (source_quote vs best chunk).
        llm_answer:     The source_quote from the model output (document language).

    Returns:
        List of result dicts, one per chunk:
        [
          {
            "premise":    "The monthly rent is 1,200 EUR.",
            "llm_answer": "The rent is 1,200 EUR per month.",
            "nli_label":  "entailment",
            "confidence": 0.94
          },
          ...
        ]
    """
    tokenizer, model, label_mapping = _get_nli_model()
    logger.info("NLI [2/3] Start predicting input data")
    result = []
    for premise in list_premises:

        # Fix 3B: bypass NLI for short quotes — too short for reliable inference
        # For Chinese/Japanese/Korean, use character count (no spaces between words)
        # For other languages, use word count
        answer_str = str(llm_answer)
        is_cjk = any('\u4e00' <= c <= '\u9fff' for c in answer_str)
        too_short = (len(answer_str) < 8) if is_cjk else (len(answer_str.split()) < 4)
        if too_short:
            result.append({
                "premise": premise,
                "llm_answer": llm_answer,
                "nli_label": "neutral",
                "confidence": 0.0,
            })
            continue

        # Fix 3B: bypass NLI for pure numerical answers — format differences
        # cause false contradictions (1,350 vs 1.350 vs 1350 EUR)
        if _is_numerical_answer(str(llm_answer)):
            result.append({
                "premise": premise,
                "llm_answer": llm_answer,
                "nli_label": "neutral",
                "confidence": 0.0,
            })
            continue

        # Tokenization — Fix 3B: max_length=256 (was 512)
        # source_quote is always a short clause, never needs 512 tokens
        inputs = tokenizer(
            str(premise),       # premise  = what the document says
            str(llm_answer),    # hypothesis = what the LLM claimed
            return_tensors="pt",
            truncation=True,
            max_length=256,     # was 512 — halves NLI time, no quality loss for short quotes
        ).to(device)

        # Model prediction
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze()

        predicted_idx = probs.argmax().item()
        predicted_label = label_mapping[predicted_idx]
        confidence = round(probs[predicted_idx].item(), 4)

        # CONFIDENCE THRESHOLD GUARD:
        # If the LLM generates hallucinated or irrelevant text, mDeBERTa gets
        # confused and may randomly guess "entailment" with low confidence,
        # producing a false green badge. Force neutral (yellow) if confidence
        # is below 0.60 so we never show a green badge we can't trust.
        # Lower threshold for CJK — mDeBERTa is less confident on Chinese/Japanese/Korean
        # even when correct. 0.50 for CJK, 0.60 for everything else.
        threshold = 0.50 if any('\u4e00' <= c <= '\u9fff' for c in str(llm_answer)) else 0.60
        if confidence < threshold:
            predicted_label = "neutral"

        # Key is "nli_label" — standardised name used by eval/metrics.py
        # and evaluate.py. Do NOT change back to "nli_checking_result".
        result.append({
            "premise": premise,
            "llm_answer": llm_answer,
            "nli_label": predicted_label,
            "confidence": confidence,
        })

    return result


def aggregate_verdict(nli_results: list[dict]):
    """
    The aggregation rules in order of priority:

      Rule 1: ANY entailment = entailment
        The answer is supported by at least one chunk. That's enough.

      Rule 2: No entailment, but ANY contradiction = contradiction
        No chunk supports the answer AND at least one chunk says something
        that directly conflicts with it. This is a likely hallucination.

      Rule 3: Everything neutral = neutral
        No chunk confirms, no chunk contradicts.
    """
    logger.info("NLI [3/3] Aggregating the result")
    if not nli_results:
        return "neutral"

    labels = [r["nli_label"] for r in nli_results]

    if "entailment" in labels:
        return "entailment"

    if "contradiction" in labels:
        return "contradiction"

    return "neutral"
