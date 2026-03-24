# Author : Vinod Anbalagan
# Project : DocuNative 
# Filename : generate.py  
# Start Date : 03-12-2026
# Modification Date : 03-12-2026
# Description : generate.py is the brain of the pipeline. It takes three inputs: 
# User question, top-3 retrieved chunks from ChromaDB (from retrieve.py), and model choice.
# Returns: The raw model response string in Markdown delimiter format.

"""
pipeline/generate.py

Sends a question + retrieved chunks to our local C++ llama-server and returns
the raw model response in Markdown delimiter format.

Issue: #15
"""

import requests
from typing import List, Dict, Any

# ---- Configuration ---

# We route everything to port 8080. The model choice (Global vs Earth) is handled 
# by whichever model the user spun up in Terminal 1 via `make server-global` or `make server-earth`.

# All models route to the same port 8080.
# The active model is determined by whichever GGUF was loaded when
# llama-server was started (make server-global / server-earth / server-fire).
SERVER_URLS = {
    "Global": "http://localhost:8080/completion",
    "Earth":  "http://localhost:8080/completion",
    "Fire":   "http://localhost:8080/completion",  # H1: South Asian specialist — use make server-fire
}

# Hyperparameters for generation

DEFAULT_MAX_TOKENS = 512    # Upper bound — overridden per-query by _estimate_max_tokens()
DEFAULT_TEMPERATURE = 0.1   # Low temperature = highly deterministic/robotic. Best for factual QA!


def _estimate_max_tokens(question: str) -> int:
    """
    Heuristic: cap generation tokens based on question type.
    Simple factual IE questions need ~15-30 tokens to answer.
    Reducing n_predict directly reduces generation time on all hardware.

    IE questions  (512 → 120): ~60% faster on Metal, ~60% faster on CPU
    Boolean questions (512 → 80): even faster
    Complex/inferential:  300   (still capped vs 512)
    """
    q = question.lower().strip()

    # Single-fact extraction — short answer guaranteed
    ie_signals = [
        "how much", "what is the", "when does", "what date",
        "how many", "what is my", "what are the",
        "wie viel", "was kostet", "wann ist", "wie lange",
        "कितना", "कब ", "क्या है",
        "berapa", "kapan", "apa itu",        # Indonesian IE signals
    ]
    if any(s in q for s in ie_signals):
        return 120

    # Boolean / permission questions
    bool_signals = ["is ", "are ", "can i", "am i", "darf ", "ist ", "क्या ", "je "]
    if any(q.startswith(s) for s in bool_signals):
        return 80

    # Inferential / open-ended — give more room but still cap
    return 300

# ---- Prompt Template ---
# Primary defense against hallucination - provides strict grounding. 
# Notice the CRITICAL language instruction: we need the answer in the user's native language.
# By ending the prompt mid-sentence with "Answer: [", we "force-feed" the model its first 
# characters, ensuring it starts exactly where we want it to, bypassing conversational filler.

PROMPT_TEMPLATE = """\
You are a precise, multilingual document assistant. Answer the question using ONLY the information \
in the provided document excerpts. Answer in a complete sentence, not just a number or single word. \
Do not use any outside knowledge.
CRITICAL: You must respond in the SAME LANGUAGE as the Question.
CRITICAL: Always answer in a complete sentence, not just a number or single word. \
For example, say 'The monthly rent is 1,350 EUR.' not just '1350'.

Format your response EXACTLY like this — no exceptions:
Answer: [your answer here]
Source_Quote: [the exact quote from the document that supports your answer]
[END]

If the answer is not found in the excerpts, respond exactly like this:
Answer: [NOT_FOUND]
Source_Quote: [N/A]
[END]

Document Excerpts:
{chunks}

Question: {question}

Answer: ["""

# --- Core Function ----

def generate_answer(
    question: str,
    chunks: List[str],
    model_choice: str = "Global",
    max_tokens: int = None,       # None = auto-estimate based on question type
    temperature: float = DEFAULT_TEMPERATURE 
) -> Dict[str, Any]: 
    """
    Sends the question + retrieved chunks to the local llama-server. 
    Returns a dictionary containing 'raw_output', 'model', 'success' boolean, and any 'error'.
    """
    
    # Format the incoming chunks into a neat, numbered list so the SLM can read them easily
    formatted_chunks = "\n\n".join(
        f"[Excerpt {i+1}]\n{chunk.strip()}"
        for i, chunk in enumerate(chunks)
    )

    # Auto-estimate token cap if not explicitly provided
    if max_tokens is None:
        max_tokens = _estimate_max_tokens(question)

    # Inject our chunks and question into the prompt template
    prompt = PROMPT_TEMPLATE.format(
        chunks=formatted_chunks,
        question=question,
    )
    
    # Grab the correct URL (defaults to Global if a weird string is passed)
    url = SERVER_URLS.get(model_choice, SERVER_URLS["Global"])

    try:
        # Make the HTTP POST request to our local C++ server
        response = requests.post(
            url,
            json={
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stop": [
                    "[END]",
                    "\n\nAnswer:",     # English re-loop guard
                    "\n\nQuestion:",   # English
                    "\n\nFrage:",      # German question marker
                    "\n\n\u092a\u094d\u0930\u0936\u094d\u0928:",      # Hindi प्रश्न: (question)
                    "\n\n\u95ee\u9898:",       # Chinese question marker 问题
                    "\n\n\u56de\u7b54:",       # Chinese answer re-loop guard 回答
                    "\n\nPytanie:",    # Polish question marker
                    "\n\nOdpowied\u017a:", # Polish answer re-loop guard
                    "\n\nSoru:",       # Turkish bleed — Tiny Aya occasionally cross-contaminates
                ],
                
            },
            timeout=120, # 2 min timeout in case the user's laptop is very slow (CPU-only)
        )
        # Raise an error if the server returned a 404, 500, etc.
        response.raise_for_status()
        
        # Extract the actual text generated by the model
        data    = response.json()
        content = data.get("content", "").strip()

        # Extract TTFT and TPOT from llama-server timings object
        # Requested by Ali Edalati (Cohere mentor) to pinpoint generation bottleneck
        # timings object: {prompt_n, prompt_ms, predicted_n, predicted_ms, predicted_per_token_ms}
        raw_timings = data.get("timings", {})
        ttft_ms  = round(raw_timings.get("prompt_ms", 0), 1)              # Time to First Token
        tpot_ms  = round(raw_timings.get("predicted_per_token_ms", 0), 1) # Time Per Output Token
        tokens_s = round(1000 / tpot_ms, 1) if tpot_ms > 0 else 0        # Tokens per second

        # Reconstruct the full answer. Because our prompt ended with "Answer: [", 
        # the model's output doesn't include that part. We add it back here so the Regex parser works later.
        raw_output = f"Answer: [{content}"

        return {
            "raw_output":    raw_output,
            "model":         model_choice,
            "success":       True,
            "error":         None,
            "ttft_ms":       ttft_ms,
            "tpot_ms":       tpot_ms,
            "tokens_per_s":  tokens_s,
            "prompt_tokens": raw_timings.get("prompt_n", 0),
            "output_tokens": raw_timings.get("predicted_n", 0),
        }

    # Catch specific connection errors to give the user helpful debugging advice
    except requests.exceptions.ConnectionError:
        return {
            "raw_output": "",
            "model": model_choice,
            "success": False,
            "error": "Cannot connect to llama-server. Is it running in Terminal 1? Run: make server-global",
        }    
    except requests.exceptions.Timeout:
        return {
            "raw_output": "",
            "model": model_choice,
            "success": False,
            "error": "llama-server timed out. The model may be overloaded.",
        } 
    except Exception as e:
        return {
            "raw_output": "",
            "model": model_choice,
            "success": False,
            "error": f"Unexpected error: {str(e)}",
        }  

# --- Health Check ---

def check_server_health(model_choice: str = "Global") -> bool:
    """Returns True if llama-server is reachable and healthy."""
    try:
        response = requests.get(
            "http://localhost:8080/health",
            timeout=5
        )
        return response.json().get("status") == "ok"
    except Exception:
        return False  

# --- Quick Test (Execution Block) ---

if __name__ == "__main__":
    print("Testing generate.py (Make sure you ran 'make server-global' in another terminal!)...\n")

    # 1. Health Check
    if not check_server_health():
        print(" llama-server is not running.")
        print("Please open Terminal 1 and run: make server-global")
        exit(1)
    print(" Server is healthy!\n")

    # ------------------------------
    # TEST 1: Standard English QA
    # ------------------------------
    print("--- TEST 1: Standard English Extraction ---")
    test_chunks = [
        """The monthly rent is 1,200 euros, payable on the first of each month.
        A security deposit of 2,400 euros (two months rent) is required before
        move-in. The lease term is 12 months, beginning March 1, 2024."""
    ]
    test_question = "What is the security deposit amount?"
    print(f"Question: {test_question}")
    
    result = generate_answer(question=test_question, chunks=test_chunks)
    if result["success"]:
        print(f"Raw output:\n{result['raw_output']}\n")
    else:
        print(f"Generation failed: {result['error']}\n")

    # ---------------------------------------------------------
    # TEST 2: Multilingual Support (German context, Indonesian query)
    # ---------------------------------------------------------
    print("--- TEST 2: Multilingual Support (H2 Testing) ---")
    german_chunk = """
    Die monatliche Miete beträgt 1.200 Euro und ist am ersten eines jeden Monats fällig.
    Die Kaution beträgt 2.400 Euro (zwei Monatsmieten) und ist vor dem Einzug zu zahlen.
    """
    indonesian_question = "Berapa jumlah uang jaminan yang harus dibayar?"  # "What is the deposit amount?"
    print(f"Question (Indonesian): {indonesian_question}")
    print(f"Context (German): {german_chunk.strip()}")

    result_multi = generate_answer(question=indonesian_question, chunks=[german_chunk])
    if result_multi["success"]:
        print(f"Raw output:\n{result_multi['raw_output']}\n")
    else:
        print(f"Generation failed: {result_multi['error']}\n")

    # ---------------------------------------------------------
    # TEST 3: Anti-Hallucination (Answer NOT in document)
    # ---------------------------------------------------------
    print("--- TEST 3: Anti-Hallucination (Missing Info) ---")
    trick_question = "Are pets allowed in the apartment?"
    print(f"Question: {trick_question}")
    
    result_trick = generate_answer(question=trick_question, chunks=test_chunks) # Using the English chunks from Test 1
    if result_trick["success"]:
        print(f"Raw output:\n{result_trick['raw_output']}\n")
    else:
        print(f"Generation failed: {result_trick['error']}\n")