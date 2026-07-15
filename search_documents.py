# search_documents.py
# -------------------------------------------------------------------
# Routes each question to the right retrieval path:
#   0. Ranking/numeric intent  -> not yet built (needs a structured
#                                 query engine over the propensity/CRM
#                                 tables, not chunk retrieval - flagged
#                                 honestly rather than mis-answered)
#   1. NPI in question          -> exact card lookup
#   2. "list" intent + state/specialty (not a market question)
#                               -> card LIST filtered by state/specialty
#   3. State named (market)     -> semantic search within that state's docs
#   4. Otherwise                -> semantic search over narrative docs
# Cards are only returned via paths 1 and 2 (never fuzzy-matched).
#
# NOTE on warm state: model and embeddings load once, at import time
# (module level, below) - not inside search(). That's already correct
# for a long-lived process. It only becomes a problem if this module
# gets re-imported per request once it's behind an API - keep it a
# single warm process, don't re-import per call.
# -------------------------------------------------------------------

import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer

with open("output/chunks_tagged.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
embeddings = np.load("output/embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Helpers built once.
card_idx = [i for i, c in enumerate(chunks) if c["doc_type"] == "hcp_card"]
narrative_idx = [i for i, c in enumerate(chunks) if c["doc_type"] != "hcp_card"]
card_by_npi = {c["npi"]: i for i, c in enumerate(chunks) if c["doc_type"] == "hcp_card"}
states_all = sorted({c["state"] for c in chunks if c["state"]})
narrative_states = sorted({c["state"] for c in chunks if c["state"] and c["doc_type"] != "hcp_card"})

SPECIALTIES = {"endocrinolog": "Endocrinology", "primary care": "Primary Care",
               "obesity medicine": "Obesity Medicine", "obesity": "Obesity Medicine"}

# "top " removed from here - it was the cause of a real misrouting bug:
# "the access situation for the top prescriber in Missouri" contains
# "top " and a state, so it was tripping this list path and returning an
# unordered card list instead of the narrative answer the question
# actually wanted. Ranking language is handled separately below instead
# of being folded into "wants a list".
LIST_WORDS = ["show me", "list ", "find ", "which ", "who are", "give me",
              "prescribers", "doctors", "hcps", "physicians", "clinicians", "targets"]
MARKET_WORDS = ["market", "landscape", "summary", "overview"]

# Ranking/superlative/numeric-threshold language. Any question using this
# phrasing needs to sort or filter by a numeric field (propensity score,
# tier, switching score, targeting flag, sample-request date, etc.) that
# doesn't exist in chunks_tagged.json - only a structured query over the
# real HCP table can answer these correctly. Chunk-based retrieval (list
# or semantic) cannot rank, so rather than silently guess, this path
# says so explicitly.
RANKING_WORDS = ["top ", "top-", "highest", "lowest", "ranked", "rank ",
                 "sorted by", "top-tier", "best performing"]

# Word-boundary versions of the above. Plain substring checks (the list
# above) have a real bug: "top " matches inside "stop taking" ("s" +
# "top " + "taking"), which sent a genuine narrative question ("Why do
# patients stop taking GLP-1s...") into needs_structured_engine with
# ZERO results instead of running semantic search at all - found via a
# real eval run, not a hypothetical. This is the same substring-match
# class of bug as the original "top " / LIST_WORDS issue, reintroduced
# here in a different list. Use these compiled patterns, not the raw
# list above, for the actual check.
RANKING_PATTERNS = [re.compile(r"\btop\b"), re.compile(r"\bhighest\b"),
                     re.compile(r"\blowest\b"), re.compile(r"\branked\b"),
                     re.compile(r"\brank\b"), re.compile(r"\bsorted by\b"),
                     re.compile(r"\bbest performing\b")]

# Common state abbreviations reps might actually type, in addition to
# full state names already handled via states_all.
STATE_ALIASES = {
    "ny": "New York", "tx": "Texas", "fl": "Florida", "ca": "California",
    "il": "Illinois", "pa": "Pennsylvania", "oh": "Ohio", "ga": "Georgia",
    "nc": "North Carolina", "mi": "Michigan", "nj": "New Jersey",
    "va": "Virginia", "wa": "Washington", "az": "Arizona", "ma": "Massachusetts",
    "mo": "Missouri", "ar": "Arkansas",
}

# Below this cosine similarity, treat results as "no confident match"
# rather than silently handing back top_k chunks with no signal that
# none of them were actually relevant. Not a hard cutoff on what's
# returned (still useful to see the closest thing found) - just an
# honest flag in the route label.
MIN_SIMILARITY = 0.3


def _detect_specialty(ql):
    for key, val in SPECIALTIES.items():
        if key in ql:
            return val
    return None


def _detect_state(ql, candidates):
    """Full state name match first, then abbreviation aliases. Aliases
    are checked as whole words only, so 'ma' doesn't match inside other
    words like 'maintenance'."""
    hit = next((s for s in candidates if s.lower() in ql), None)
    if hit:
        return hit
    for abbr, full in STATE_ALIASES.items():
        if full in candidates and re.search(r"\b" + abbr + r"\b", ql):
            return full
    return None


def _semantic(question, pool, top_k):
    q = model.encode([question])[0]
    q = q / np.linalg.norm(q)
    scores = chunk_norms[pool] @ q
    order = np.argsort(scores)[::-1][:top_k]
    return [(chunks[pool[o]], float(scores[o])) for o in order]


def search(question, top_k=3):
    ql = question.lower()

    # PATH 1: exact NPI lookup (unambiguous, always takes priority)
    m = re.search(r"\b(\d{10})\b", question)
    if m and m.group(1) in card_by_npi:
        return "card lookup (NPI)", [(chunks[card_by_npi[m.group(1)]], 1.0)]

    # PATH 0: ranking/numeric intent - not answerable by chunk retrieval.
    # Checked before the list/market paths so ranking language ("top
    # prescriber", "highest switching score") can't be mistaken for
    # either a plain card-list request or a market-summary request.
    if any(p.search(ql) for p in RANKING_PATTERNS):
        return ("needs_structured_engine (ranking/numeric query - requires "
                "a structured query over the propensity/CRM table, not "
                "chunk retrieval)"), []

    # PATH 2: card LIST by state / specialty (only on explicit list intent, not market qs)
    wants_list = any(w in ql for w in LIST_WORDS)
    is_market = any(w in ql for w in MARKET_WORDS)
    spec = _detect_specialty(ql)
    st = _detect_state(ql, states_all)
    if wants_list and not is_market and (spec or st):
        pool = [i for i in card_idx
                if (not st or chunks[i]["state"] == st)
                and (not spec or chunks[i]["specialty"] == spec)]
        label = f"card list (state={st}, specialty={spec}) - {len(pool)} match"
        return label, [(chunks[i], None) for i in pool[:top_k]]

    # PATH 3: state market question -> narrative docs for that state
    picked = _detect_state(ql, narrative_states)
    if picked:
        pool = [i for i in narrative_idx if chunks[i]["state"] == picked]
        if pool:
            results = _semantic(question, pool, top_k)
            route = f"state filter ({picked})"
            if not results or results[0][1] < MIN_SIMILARITY:
                route += " - low confidence, no strong match"
            return route, results

    # PATH 4: general semantic search over narrative docs
    results = _semantic(question, narrative_idx, top_k)
    route = "semantic (narrative)"
    if not results or results[0][1] < MIN_SIMILARITY:
        route += " - low confidence, no strong match"
    return route, results


if __name__ == "__main__":
    questions = [
        "Tell me about the HCP with NPI 1000008396",
        "Show me endocrinologists in Arkansas",
        "What does the GLP-1 market look like in Missouri?",
        "How is Rabivy different from Zepbound?",
        "What's the access situation for the top prescriber in Missouri?",
        "Show me the top 10 by propensity score",
    ]
    for question in questions:
        route, results = search(question, top_k=3)
        print("=" * 72)
        print(f"Q: {question}")
        print(f"   route -> {route}")
        for chunk, score in results:
            s = f"[{score:.3f}]" if score is not None else "[card ]"
            print(f"   {s} {chunk['chunk_id']}  ({chunk['doc_type']})")
        print()