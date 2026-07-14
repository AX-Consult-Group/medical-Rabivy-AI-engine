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

SYSTEM_PROMPT = """You are a commercial-intelligence assistant for MariTide (maridebart cafraglutide) sales reps.

Rules you must follow:
- Only use the DATA block provided in the user message. Never invent numbers, NPIs, names, scores, or facts that are not present in it.
- If DATA is empty, marked "not found", or marked LOW CONFIDENCE, say so plainly rather than guessing or filling gaps.
- If DATA comes from retrieved document chunks, ground your answer in their text and mention which chunk/document it came from.
- If DATA is a structured table result (rows with fields like npi, propensity_score, tier), summarize it clearly - don't just restate raw JSON.
- If RESULT says "showing the top N of Y total", start your answer with something like "Here are the top N (out of Y total matches)" - a clean, natural lead-in, not an awkward afterthought sentence bolted onto the end.
- Be concise and rep-friendly: a few sentences, not a report. No unnecessary preamble.
"""


def _round_floats(obj, ndigits=3):
    """Round every float in a nested dict/list to a sane number of
    decimal places before it reaches the LLM. Spreadsheet floats come
    in with 15+ digits of precision (0.5317726198315235) - rounding
    here, once, deterministically, is more reliable than asking the
    LLM to do it correctly every time in its phrasing."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


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
            shown = len(data["results"])
            if "count" in data and data["count"] > shown:
                # filter_hcps() tracks the TRUE total separately from the
                # (possibly truncated) results list. Reporting only
                # len(results) silently undercounts - e.g. 11 real matches
                # but only the top 10 by propensity_score get returned,
                # and the LLM would otherwise say "there are 10" as if
                # that were the whole answer. Found via a real run where
                # the dashboard showed 11 and the answer said 10.
                lines.append(f"RESULT: showing the top {shown} of {data['count']} total "
                              f"matches (sorted by highest propensity_score first).")
            else:
                lines.append(f"RESULT: {shown} row(s) returned")
            lines.append(json.dumps(_round_floats(data["results"]), indent=2, default=str))
        else:
            lines.append("RESULT (single row):")
            lines.append(json.dumps(_round_floats(data), indent=2, default=str))

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

    # Print the ground-truth count directly from the data, not from the
    # LLM's prose. The LLM is told to mention truncation, but it won't
    # always comply perfectly (especially when summarizing a long list
    # instead of enumerating it) - this line is never wrong regardless
    # of what the LLM chooses to say.
    data = result["raw_data"]
    if result["route"].startswith("STRUCTURED") and isinstance(data, dict) and "results" in data:
        total = data.get("count", len(data["results"]))
        shown = len(data["results"])
        if total > shown:
            print(f"   [DATA CHECK: {total} total match(es), showing {shown}]")
        else:
            print(f"   [DATA CHECK: {total} total match(es), all shown]")

    if result["error"]:
        print(f"\n! LLM call failed: {result['error']}")
        print("  Check GROQ_API_KEY is set and you have network access.")
    else:
        print(f"\n{result['answer']}")