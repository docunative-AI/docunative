"""
pipeline/validate.py

Parses the raw Markdown-delimited output from generate.py into a clean,
type-safe Pydantic object. Handles all known failure modes gracefully.

Author: Vinod Anbalagan
Issue: #18
"""

import re
from pydantic import BaseModel, Field
from typing import Optional


# Output Schema 
#
# This is the contract between generate.py and everything downstream.
# validate.py, nli.py, the Gradio UI, and the eval pipeline all import
# this class. If you change a field name here, you must update every caller.

class ParsedOutput(BaseModel):
    """
    The structured output of one model generation.

    Fields:
        answer:        The model's answer to the user's question.
        source_quote:  The exact quote from the document that supports the answer.
        parse_success: True if both fields were extracted cleanly. False means
                       the model's output was malformed — treat answer with caution.
        raw_output:    The original unmodified string from the model, always
                       preserved for debugging and for the eval pipeline.
    """
    answer: str = Field(default="", description="Extracted answer text")
    source_quote: str = Field(default="", description="Extracted source quote")
    parse_success: bool = Field(default=False, description="Whether parsing succeeded")
    raw_output: str = Field(default="", description="Original model output, unmodified")


# Regex Patterns 
#
# Why regex and not json.loads()?
# Small quantized models fail JSON schema reliably. A missing brace, a trailing
# comma, an unescaped quote inside the answer — all of these crash json.loads().
# Regex is more tolerant. It finds the pattern even in messy surrounding text.

ANSWER_PATTERN = re.compile(r'(?i)Answer:\s*\[([^\]\n]+)\]')
SOURCE_PATTERN = re.compile(r'(?i)Source_Quote:\s*\[([^\]\n]+)\]')

# Fallback: handles case where model forgot the closing bracket
# Captures everything after "Answer: [" to the end of that line
ANSWER_FALLBACK = re.compile(r'(?i)Answer:\s*\[(.+?)(?:\]|$)', re.MULTILINE)
SOURCE_FALLBACK = re.compile(r'(?i)Source_Quote:\s*\[(.+?)(?:\]|$)', re.MULTILINE)

# Last resort: model dropped brackets entirely — capture bare text after label.
# Lookahead (?=\s*Source_Quote:|$) stops capture before Source_Quote bleeds in,
# which happens when both fields land on the same line (common in 3B models).
ANSWER_BARE = re.compile(r'(?i)Answer:\s*([^\[\n][^\n]+?)(?=\s*Source_Quote:|$)', re.MULTILINE)
SOURCE_BARE = re.compile(r'(?i)Source_Quote:\s*([^\[\n][^\n]+?)(?=\s*\[END\]|$)', re.MULTILINE)


# Core Function 

def parse_output(raw_output: str) -> ParsedOutput:
    """
    Parse the raw model output string into a structured ParsedOutput.

    Tries strict patterns first (require closing bracket).
    Falls back to lenient patterns (accept missing closing bracket).
    If both fail, returns a ParsedOutput with parse_success=False and
    preserves the raw_output so the UI can display it as a fallback.

    Args:
        raw_output: The 'raw_output' string from generate_answer()

    Returns:
        ParsedOutput with extracted fields and parse_success flag
    """
    if not raw_output or not raw_output.strip():
        return ParsedOutput(
            answer="No output received from model.",
            source_quote="",
            parse_success=False,
            raw_output=raw_output,
        )

    # Step 1: Try strict patterns (both fields, both brackets present) 
    answer_match = ANSWER_PATTERN.search(raw_output)
    source_match = SOURCE_PATTERN.search(raw_output)

    if answer_match and source_match:
        return ParsedOutput(
            answer=answer_match.group(1).strip(),
            source_quote=source_match.group(1).strip(),
            parse_success=True,
            raw_output=raw_output,
        )

    # Step 2: Try fallback patterns (missing closing bracket)
    answer_match = answer_match or ANSWER_FALLBACK.search(raw_output)
    source_match = source_match or SOURCE_FALLBACK.search(raw_output)

    # Step 3: Last resort — model dropped brackets entirely
    answer_match = answer_match or ANSWER_BARE.search(raw_output)
    source_match = source_match or SOURCE_BARE.search(raw_output)

    if answer_match:
        extracted_answer = answer_match.group(1).strip()

        # HALLUCINATION GUARD 1: Small models sometimes fail to say "I don't know"
        # and instead lazily output the prompt label (e.g. "Excerpt 3" or "[Excerpt 2]").
        # Intercept this and force a safe "not found" response.
        if re.fullmatch(r'(?i)\[?excerpt\s*\d+\]?\.?', extracted_answer):
            return ParsedOutput(
                answer="The document does not contain information to answer this question.",
                source_quote="N/A",
                parse_success=True,
                raw_output=raw_output,
            )

        # HALLUCINATION GUARD 2: Template echo — 3B model copies format example
        # literally, returning "your answer here" or similar placeholder text.
        # parse_success=False signals the UI to show a formatting warning.
        _TEMPLATE_ECHO = [
            "your answer here",
            "the exact quote from the document that supports your answer",
            "the exact quote from the document",
        ]
        if any(p in extracted_answer.lower() for p in _TEMPLATE_ECHO):
            return ParsedOutput(
                answer="The document does not contain information to answer this question.",
                source_quote="N/A",
                parse_success=False,  # False — model failed to follow format
                raw_output=raw_output,
            )

        # Partial parse — got at least the answer
        return ParsedOutput(
            answer=extracted_answer,
            source_quote=source_match.group(1).strip() if source_match else "",
            parse_success=bool(answer_match and source_match),
            raw_output=raw_output,
        )

    # Step 3: Complete parse failure 
    # The model did not follow the format at all.
    # parse_success=False tells the UI to show a warning badge.
    # We preserve raw_output so the user still sees *something*.
    return ParsedOutput(
        answer="Could not parse model output. See raw output below.",
        source_quote="",
        parse_success=False,
        raw_output=raw_output,
    )


# Convenience Helper 

def is_answer_missing(parsed: ParsedOutput) -> bool:
    """
    Returns True if the model explicitly said it couldn't find the answer.
    This is different from parse_success=False (which means formatting failed).
    Useful for the eval pipeline to distinguish 'no answer' from 'wrong format'.
    """
    no_answer_phrases = [
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
        # German — Aya answers in query language for German questions
        "nicht im dokument",
        "keine information",
        "nicht gefunden",
        "nicht angegeben",
        "nicht erwähnt",
        "nicht enthalten",
        "nicht vorhanden",
        # Hindi
        "दस्तावेज़ में नहीं",
        "जानकारी नहीं",
        "उल्लेख नहीं",
        "नहीं मिला",
        # Swahili
        "haipo katika hati",
        "hakuna taarifa",
        "haipatikani",
        "haijatajwa",
        "haimo katika",
    ]
    return any(phrase in parsed.answer.lower() for phrase in no_answer_phrases)


#  Quick Test 

if __name__ == "__main__":
    print("Testing validate.py...\n")

    # Test 1: Perfect output (the happy path) 
    perfect = "Answer: [The deposit is 1,200 euros]\nSource_Quote: [A security deposit of 1,200 euros is required.]"
    result = parse_output(perfect)
    assert result.parse_success == True
    assert result.answer == "The deposit is 1,200 euros"
    assert result.source_quote == "A security deposit of 1,200 euros is required."
    print("Test 1 passed: Perfect output parsed correctly")

    # Test 2: Lowercase labels (model ignored capitalisation) 
    lowercase = "answer: [The notice period is 90 days]\nsource_quote: [The tenant must give 90 days notice.]"
    result = parse_output(lowercase)
    assert result.parse_success == True
    assert result.answer == "The notice period is 90 days"
    print("Test 2 passed: Lowercase labels handled")

    # Test 3: Missing closing bracket (common small-model failure)
    missing_bracket = "Answer: [The lease starts March 1, 2024\nSource_Quote: [The tenancy commences on March 1, 2024]"
    result = parse_output(missing_bracket)
    assert result.answer == "The lease starts March 1, 2024"
    print("Test 3 passed: Missing closing bracket handled")

    # Test 4: Model said answer not in document 
    no_answer = "Answer: [The document does not contain information to answer this question.]\nSource_Quote: [N/A]"
    result = parse_output(no_answer)
    assert result.parse_success == True
    assert is_answer_missing(result) == True
    print("Test 4 passed: Explicit no-answer detected")

    # Test 5: Complete garbage output 
    garbage = "Sorry, I am unable to process this request in the required format."
    result = parse_output(garbage)
    assert result.parse_success == False
    assert result.raw_output == garbage  # raw always preserved
    print("Test 5 passed: Parse failure handled gracefully")

    # Test 6: Empty input 
    result = parse_output("")
    assert result.parse_success == False
    print("Test 6 passed: Empty input handled")

    print("\nAll 6 tests passed. validate.py is ready.")