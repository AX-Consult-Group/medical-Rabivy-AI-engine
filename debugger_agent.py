# debugger_agent.py  (was: write_the_answer.py)
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# ask_a_question.py already does two jobs on its own: it finds the
# right DATA, and it turns that DATA into a correct, clean sentence via
# its own format_answer(). That second job is deterministic and safe -
# it never lets an LLM touch a row of numbers, so a rep can trust every
# NPI/score/count exactly.
#
# This file is ONLY the third job: making the small slice of answers
# that are genuinely narrative - a comparison, a "why isn't this doctor
# converting" synthesis, a semantic search over multiple documents -
# read like something a person wrote, instead of a raw dump of
# retrieved text. Everything else (counts, rankings, filtered lists,
# card lists) is left completely alone: ask_a_question.format_answer()
# already wrote it correctly, and this file just hands it back as-is.
#
# WHEN THIS RUNS:
#   python debugger_agent.py "Compare NPI 1344001929 to a typical endocrinologist"
#
# SETUP (one-time, unchanged from before):
#   1. pip install openai --break-system-packages
#   2. Free key at https://console.groq.com/keys (email signup, no card)
#   3. Windows PowerShell:  $env:GROQ_API_KEY = "your-key-here"
#      Mac/Linux:            export GROQ_API_KEY="your-key-here"
# =====================================================================

import os
import sys
import json
from openai import OpenAI
import ask_a_question

GROQ_MODEL = "llama-3.3-70b-versatile"

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)


# =====================================================================
# CHUNK 1 - WHICH "kind"s ACTUALLY GET SYNTHESIZED
# -----------------------------------------------------------------
# Plain English: ask_a_question.py tags every answer with a "kind"
# (see its Section 8 / format_answer). Only these three are genuinely
# narrative and benefit from an LLM rewording. Everything else - every
# list, every count, every single-value lookup - is deterministic
# prose already and is returned untouched. This set is the ONE place
# that decision gets made, so it's easy to check and easy to change.
# =====================================================================
SYNTHESIZE_KINDS = {"card_lookup", "comparison", "semantic_search"}

# Small, deterministic note - added by CODE, not asked of the LLM - for
# the one case (a moderate-confidence semantic match) where a rep
# should know to double check the source before relying on it.
MODERATE_CONFIDENCE_NOTE = (
    "\n\n(Note: confidence in this match is modest - worth checking the "
    "source document before relying on it heavily.)"
)


# =====================================================================
# CHUNK 2 - INSTRUCTIONS GIVEN TO THE AI
# -----------------------------------------------------------------
# Plain English: rules so the AI can only reword/organize what it's
# given, never invent or hide anything.
# =====================================================================
SYSTEM_PROMPT = """You are a commercial-intelligence assistant for Rabivy (maridebart cafraglutide) sales reps.

Rules you must follow:
- Only use the DATA block provided in the user message. Never invent numbers, NPIs, names, scores, or facts that are not present in it.
- Answer the specific QUESTION asked, using the relevant parts of DATA - you do not need to restate every sentence of a card word-for-word if only part of it answers the question.
- Do NOT quote or mention internal chunk_id/filename/document labels (e.g. do not say "as stated in the rabivy_product_benefits_brief chunk") - reps never see these and it reads as a technical glitch. Describe a source in plain English instead (e.g. "per our competitive positioning materials") if you need to.
- Always identify an HCP by their NPI. If DATA contains some other internal label (e.g. "HCP2302"), ignore it - NPI is the only identifier a rep should see.
- If DATA contains a CARD and a BENCHMARK (a comparison request), you must start with an intro sentence about the HCP (where they work, how long have they been working, what their specialty is, etc.)and you must explicitly compare them with numbers on both sides (e.g. "this HCP writes X/month vs. a typical Y specialist's Z/month") - do not just describe the card alone.
- If DATA contains multiple retrieved document chunks (a semantic search), synthesize them into one coherent narrative answer rather than listing them chunk by chunk - but do not blend facts from chunks that are actually unrelated to each other; if two chunks disagree or don't connect, say so plainly instead of forcing them together.
- If the question asks for something DATA doesn't actually contain (for example, specific talking points that aren't in any retrieved chunk), say plainly that this part isn't covered rather than inventing it.
- No unnecessary preamble or repeating the question back before answering.
- Be concise - a few sentences, no padding.
- If a question asks a question about an HCP but doesnt give a specific NPI, say that you need the NPI to answer it, rather than guessing or making up a number.
"""


# =====================================================================
# CHUNK 3 - TURNING THE RAW DATA INTO TEXT THE AI CAN READ
# -----------------------------------------------------------------
# Plain English: only handles the three SYNTHESIZE_KINDS shapes, since
# nothing else ever reaches this function (see synthesize() below).
# =====================================================================
def _describe_rag_data(kind, data):
    if kind == "card_lookup":
        chunk = data["chunk"]
        header = (f"HCP CARD - NPI {chunk.get('npi', '?')}, "
                  f"{chunk.get('specialty', '?')}, {chunk.get('state', '?')}:")
        return header + "\n" + chunk["text"]

    if kind == "comparison":
        lines = ["HCP CARD:", data["card"]["text"]]
        if data.get("benchmark"):
            lines.append("\nBENCHMARK DOCUMENT (compare the card against this):")
            lines.append(data["benchmark"][0]["chunk"]["text"])
        else:
            lines.append("\n(No benchmark document was found - say plainly that a "
                          "full comparison isn't possible, rather than comparing "
                          "against nothing.)")
        return "\n".join(lines)

    if kind == "semantic_search":
        results = data.get("results", [])
        lines = [f"RETRIEVED {len(results)} document chunk(s), most relevant first:"]
        for r in results:
            chunk = r["chunk"]
            lines.append(f"--- similarity {r['score']:.3f}, "
                          f"type: {chunk.get('doc_type', 'unknown')} ---")
            lines.append(chunk["text"])
        return "\n".join(lines)

    # Should never be reached - SYNTHESIZE_KINDS is the only gate that
    # routes anything here. Surfaced loudly rather than guessed at.
    return json.dumps(data, indent=2, default=str)


# =====================================================================
# CHUNK 3b - WHERE DID THIS ANSWER COME FROM
# -----------------------------------------------------------------
# Plain English: this is built directly from the chunk metadata that
# ask_a_question.py / search_documents.py already retrieved - it is
# NOT asked of the LLM. That matters: an LLM asked to "cite its
# source" can get that wrong or invent one, but code reading a field
# that's already sitting right there in the data can't. This is for
# YOUR terminal only, to sanity-check an answer - reps never see this
# (see SYSTEM_PROMPT's rule against exposing internal filenames to
# them in the actual wording).
# =====================================================================
def _describe_source(chunk):
    label = chunk.get("source_doc")
    if not label and chunk.get("npi"):
        # HCP card chunks may not carry a source_doc field the same
        # way reference/strategic docs do - fall back to the NPI,
        # since that's exactly how the card file is keyed.
        label = f"hcp_snapshots (NPI {chunk['npi']})"
    doc_type = chunk.get("doc_type")
    if label and doc_type:
        return f"{label} ({doc_type})"
    return label or "unknown source (no source_doc/npi field on this chunk)"


def _list_sources(kind, data):
    sources = []
    if kind == "card_lookup":
        sources.append(_describe_source(data["chunk"]))
    elif kind == "comparison":
        sources.append(_describe_source(data["card"]))
        if data.get("benchmark"):
            sources.append(_describe_source(data["benchmark"][0]["chunk"]))
    elif kind == "semantic_search":
        seen = set()
        for r in data.get("results", []):
            label = _describe_source(r["chunk"])
            if label not in seen:
                seen.add(label)
                score = r.get("score")
                sources.append(f"{label} [similarity {score:.3f}]" if score is not None else label)
    return sources


# =====================================================================
# CHUNK 4 - THE MAIN FUNCTION
# -----------------------------------------------------------------
# Plain English: ask ask_a_question.py what it found (route, data) and
# what it would say about it on its own (baseline). If this kind isn't
# one we synthesize, hand baseline straight back - no LLM call at all,
# so there's zero chance of the LLM touching a number it shouldn't.
# Only for card_lookup / comparison / semantic_search do we actually
# call the LLM, and even then baseline is kept as a safe fallback if
# the API call fails for any reason.
# =====================================================================
def synthesize(question, model=GROQ_MODEL):
    route, data = ask_a_question.ask(question)
    baseline = ask_a_question.format_answer(route, data)

    kind = data.get("kind")

    # Nothing found - ask_a_question already said so plainly. Nothing
    # for an LLM to add, and no risk it tries to guess anyway.
    if not data.get("found", True):
        return {"question": question, "route": route, "answer": baseline, "kind": kind,
                "error": None, "raw_data": data, "llm_used": False}

    # Not one of the narrative kinds -> the deterministic answer from
    # ask_a_question.py is already correct and complete. Return as-is.
    if kind not in SYNTHESIZE_KINDS:
        return {"question": question, "route": route, "answer": baseline, "kind": kind,
                "error": None, "raw_data": data, "llm_used": False}

    # A semantic search with nothing confident to go on has nothing
    # solid enough to synthesize either - same rule applies.
    if kind == "semantic_search" and (data.get("low_confidence") or not data.get("results")):
        return {"question": question, "route": route, "answer": baseline, "kind": kind,
                "error": None, "raw_data": data, "llm_used": False}

    data_block = _describe_rag_data(kind, data)
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
        answer = response.choices[0].message.content.strip()
        error = None
    except Exception as e:
        # Network/auth/rate-limit errors land here. Fall back to the
        # safe deterministic baseline rather than crashing or handing
        # back nothing - a raw comparison is still useful to a rep.
        answer = baseline
        error = f"{type(e).__name__}: {e}"

    if error is None:
        # These two additions are deterministic (code, not LLM), same
        # as ask_a_question.py's own rule for CLARIFICATIONS - a fact
        # this easy to miss shouldn't depend on the LLM remembering it.
        if kind == "semantic_search" and data.get("confidence") == "moderate":
            answer += MODERATE_CONFIDENCE_NOTE
        if route in ask_a_question.CLARIFICATIONS:
            answer += ask_a_question.CLARIFICATIONS[route]

    return {"question": question, "route": route, "answer": answer, "kind": kind,
            "error": error, "raw_data": data, "llm_used": True}


# =====================================================================
# CHUNK 5 - RUNNING ONE QUESTION AND PRINTING IT
# -----------------------------------------------------------------
# Plain English: pulled out of __main__ so both modes below (one-shot
# and interactive) print answers exactly the same way - one place to
# fix formatting instead of two copies drifting apart.
# =====================================================================
def _print_answer(question):
    result = synthesize(question)

    print(f"Q: {result['question']}")
    tag = "synthesized" if result["llm_used"] else "raw / deterministic"
    print(f"   (routed via: {result['route']}, {tag})")

    if result["error"]:
        print(f"\n! LLM call failed, showing raw fallback instead: {result['error']}")

    # Sources are only meaningful for the doc-retrieval kinds - a
    # count or a ranked list didn't come "from a document", it came
    # from the spreadsheet, so there's nothing to list there. Shown
    # FIRST and separately, so you can sanity-check the grounding
    # before reading the polished version - this is internal, for you
    # only, never shown to a rep.
    if result["kind"] in SYNTHESIZE_KINDS:
        sources = _list_sources(result["kind"], result["raw_data"])
        print("\nRetrieved from (internal - not shown to reps):")
        if sources:
            for s in sources:
                print(f"  - {s}")
        else:
            print("  (no source metadata found on the retrieved chunk(s))")

    print("\nWhat the rep sees:")
    print(result["answer"])
    print()


# =====================================================================
# CHUNK 6 - COMMAND LINE RUNNER
# -----------------------------------------------------------------
# Plain English: two ways to use this file from a terminal.
#   1) One-shot:      python debugger_agent.py "your question"
#      -> answers that one question and exits. Useful for scripting.
#   2) Interactive:   python debugger_agent.py
#      -> no question typed after the filename, so it drops into a
#         loop: type a question, hit enter, get an answer, repeat.
#         Type "quit", "exit", or press Ctrl+C to leave the loop.
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # A question was typed after the filename - answer that one
        # and exit, same as the old behavior.
        _print_answer(" ".join(sys.argv[1:]))
    else:
        print("Rabivy Q&A - type a question and press enter.")
        print("Type 'quit' or 'exit' to leave (or press Ctrl+C).\n")
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not question:
                continue
            if question.lower() in {"quit", "exit"}:
                print("Bye.")
                break
            _print_answer(question)