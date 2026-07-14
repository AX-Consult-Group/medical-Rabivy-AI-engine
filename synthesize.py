# synthesize.py
# -------------------------------------------------------------------
# Phase 4.4 - Answer Synthesis (MVP).
# Turns ask.py's (route, data) output into an actual written answer,
# using a free LLM. Groq by default: no credit card, OpenAI-compatible
# endpoint, so this only needs the standard `openai` package pointed at
# a different base_url.
#
# This is deliberately the SIMPLEST version: one LLM call, no looping,
# no tool-calling/agent behaviour yet. Get this working first - the
# chained-query and Q20 multi-source ideas are a loop built on TOP of
# this, not a replacement for it.
#
# Setup:
#   1. pip install openai --break-system-packages
#   2. Get a free key at https://console.groq.com/keys (email signup,
#      no card needed)
#   3. Set it as an environment variable before running:
#        Windows PowerShell:  $env:GROQ_API_KEY = "your-key-here"
#        Mac/Linux:            export GROQ_API_KEY="your-key-here"
#   4. python synthesize.py "Who is the top GLP-1 prescriber in New York?"
# -------------------------------------------------------------------

import os
import sys
import json
from openai import OpenAI
import ask

GROQ_MODEL = "llama-3.3-70b-versatile"  # solid free-tier default; swap freely

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)

SYSTEM_PROMPT = """You are a commercial-intelligence assistant for Rabivy (maridebart cafraglutide) sales reps.

Rules you must follow:
- Only use the DATA block provided in the user message. Never invent numbers, NPIs, names, scores, or facts that are not present in it.
- If DATA is empty, marked "not found", or marked LOW CONFIDENCE, say so plainly rather than guessing or filling gaps.
- If DATA comes from retrieved document chunks, ground your answer in their text and mention which chunk/document it came from.
- If DATA is a structured table result (rows with fields like npi, propensity_score, tier), summarize it clearly - don't just restate raw JSON.
- Be concise and rep-friendly: a few sentences, not a report. No unnecessary preamble.
"""


def _describe_data(route, data):
    """Turn ask.py's raw (route, data) into a plain-text block the LLM
    can read. Handles every shape structured.py/search.py can return -
    see the different STRUCTURED functions' return shapes in
    structured.py and the RAG shape in ask.py's own Path 7."""
    lines = [f"ROUTE: {route}"]

    if route.startswith("STRUCTURED"):
        if isinstance(data, dict) and not data.get("found", True):
            lines.append(f"RESULT: not found - {data.get('error', 'no matching data')}")
        elif isinstance(data, dict) and "results" in data:
            lines.append(f"RESULT: {len(data['results'])} row(s) returned")
            lines.append(json.dumps(data["results"], indent=2, default=str))
        else:
            lines.append("RESULT (single row):")
            lines.append(json.dumps(data, indent=2, default=str))

    elif route == "RAG":
        if data.get("low_confidence"):
            lines.append("RESULT: LOW CONFIDENCE - no strongly relevant chunk found.")
        chunks = data.get("chunks", [])
        lines.append(f"RETRIEVED {len(chunks)} chunk(s):")
        for c in chunks:
            lines.append(f"--- {c['chunk_id']} ({c['doc_type']}, source: {c.get('source_doc')}) ---")
            lines.append(c["text"])

    else:
        lines.append(json.dumps(data, indent=2, default=str))

    return "\n".join(lines)


def synthesize(question, model=GROQ_MODEL):
    """Run the question through ask.py, then have the LLM write the
    final answer from whatever ask.py found. Returns a dict so callers
    can inspect the route/raw data too, not just the text."""
    route, data = ask.ask(question)
    data_block = _describe_data(route, data)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"QUESTION: {question}\n\nDATA:\n{data_block}\n\n"
                    f"Answer the question using only the DATA above."
                )},
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content
        error = None
    except Exception as e:
        # Network/auth/rate-limit errors all land here - fail with a
        # clear message rather than crashing, since this will eventually
        # sit behind something a rep is actually using.
        answer = None
        error = f"{type(e).__name__}: {e}"

    return {"question": question, "route": route, "answer": answer,
            "error": error, "raw_data": data}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Who is the top GLP-1 prescriber in New York?"
    result = synthesize(question)

    print(f"Q: {result['question']}")
    print(f"   (routed via: {result['route']})")
    if result["error"]:
        print(f"\n! LLM call failed: {result['error']}")
        print("  Check GROQ_API_KEY is set and you have network access.")
    else:
        print(f"\n{result['answer']}")