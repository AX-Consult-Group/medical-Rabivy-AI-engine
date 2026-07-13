# search.py
# -------------------------------------------------------------------
# Routes each question to the right retrieval path:
#   1. NPI in question          -> exact card lookup
#   2. "list" intent + state/specialty (not a market question)
#                               -> card LIST filtered by state/specialty
#   3. State named (market)     -> semantic search within that state's docs
#   4. Otherwise                -> semantic search over narrative docs
# Cards are only returned via paths 1 and 2 (never fuzzy-matched).
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
LIST_WORDS = ["show me", "list ", "find ", "which ", "who are", "give me",
              "prescribers", "doctors", "hcps", "physicians", "clinicians", "targets", "top "]
MARKET_WORDS = ["market", "landscape", "summary", "overview"]

def _detect_specialty(ql):
    for key, val in SPECIALTIES.items():
        if key in ql:
            return val
    return None

def _semantic(question, pool, top_k):
    q = model.encode([question])[0]
    q = q / np.linalg.norm(q)
    scores = chunk_norms[pool] @ q
    order = np.argsort(scores)[::-1][:top_k]
    return [(chunks[pool[o]], float(scores[o])) for o in order]


def search(question, top_k=3):
    ql = question.lower()

    # PATH 1: exact NPI lookup
    m = re.search(r"\b(\d{10})\b", question)
    if m and m.group(1) in card_by_npi:
        return "card lookup (NPI)", [(chunks[card_by_npi[m.group(1)]], 1.0)]

    # PATH 2: card LIST by state / specialty (only on explicit list intent, not market qs)
    wants_list = any(w in ql for w in LIST_WORDS)
    is_market = any(w in ql for w in MARKET_WORDS)
    spec = _detect_specialty(ql)
    st = next((s for s in states_all if s.lower() in ql), None)
    if wants_list and not is_market and (spec or st):
        pool = [i for i in card_idx
                if (not st or chunks[i]["state"] == st)
                and (not spec or chunks[i]["specialty"] == spec)]
        label = f"card list (state={st}, specialty={spec}) - {len(pool)} match"
        return label, [(chunks[i], None) for i in pool[:top_k]]

    # PATH 3: state market question -> narrative docs for that state
    picked = next((s for s in narrative_states if s.lower() in ql), None)
    if picked:
        pool = [i for i in narrative_idx if chunks[i]["state"] == picked]
        if pool:
            return f"state filter ({picked})", _semantic(question, pool, top_k)

    # PATH 4: general semantic search over narrative docs
    return "semantic (narrative)", _semantic(question, narrative_idx, top_k)


if __name__ == "__main__":
    questions = [
        "Tell me about the HCP with NPI 1000008396",
        "Show me endocrinologists in Arkansas",
        "What does the GLP-1 market look like in Missouri?",
        "How is Rabivy different from Zepbound?",
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