# search_documents.py
# =====================================================================
# WHAT THIS FILE IS FOR
# =====================================================================
# This file is the RAG data engine. It knows how to find things inside
# the document knowledge base - either by exact NPI, by listing HCP
# cards that match a state/specialty, or by meaning-based ("semantic")
# search over the narrative documents.
#
# IMPORTANT - what this file does NOT do:
# This file does not try to understand a rep's question. Every
# function below expects to be handed clean, already-decided values -
# a real NPI, a real state name, a search phrase to embed - and it
# just does the lookup. That understanding work lives in
# ask_a_question.py.
#
# CONFIDENCE TIERS (recalibrated from real evidence, not a guess):
# A semantic search result is labelled one of three ways:
#   - "high"     : score >= MODERATE_SIMILARITY - a solid, clear match.
#   - "moderate" : score >= MIN_SIMILARITY but below that - a real,
#                  correct match in many cases, but modest enough that
#                  it's shown WITH a caveat rather than as a confident
#                  answer, instead of being hidden entirely.
#   - "none"     : below MIN_SIMILARITY - genuinely nothing relevant.
# These two numbers were set by looking at real observed scores across
# many real test questions: scores below ~0.11 consistently turned out
# to be unrelated content, while scores from ~0.15 up were consistently
# the CORRECT document, just modestly scored (a known limitation of
# small/fast embedding models on short, informal phrasing). The old
# version of this file used one flat 0.3 cutoff for "found or not,"
# which was hiding many genuinely correct answers - see the "moderate"
# tier above for the fix.
# =====================================================================

import json
import numpy as np
from embedding_backend import get_query_backend

with open("output/chunks_tagged.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
embeddings = np.load("output/embeddings.npy")
# Load whichever backend embedding_meta.json says built the store
# (MiniLM normally; the offline TF-IDF/LSA fallback in sandboxes).
model = get_query_backend()

chunk_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

card_idx = [i for i, c in enumerate(chunks) if c["doc_type"] == "hcp_card"]
narrative_idx = [i for i, c in enumerate(chunks) if c["doc_type"] != "hcp_card"]
card_by_npi = {c["npi"]: i for i, c in enumerate(chunks) if c["doc_type"] == "hcp_card"}
states_all = sorted({c["state"] for c in chunks if c["state"]})
specialties_all = sorted({c["specialty"] for c in chunks if c["specialty"]})

MIN_SIMILARITY = 0.13
MODERATE_SIMILARITY = 0.30


def _check_state(state):
    if state is None:
        return None
    if state not in states_all:
        return f"'{state}' is not a recognized state in the document set. Valid states: {states_all}"
    return None


def _check_specialty(specialty):
    if specialty is None:
        return None
    if specialty not in specialties_all:
        return f"'{specialty}' is not a recognized specialty. Valid specialties: {specialties_all}"
    return None


# =====================================================================
# DATA FUNCTIONS
# =====================================================================

def lookup_card_by_npi(npi):
    npi_str = str(npi).strip()
    if npi_str in card_by_npi:
        return {"kind": "card_lookup", "found": True, "chunk": chunks[card_by_npi[npi_str]]}
    return {"kind": "card_lookup", "found": False,
            "error": f"No HCP card found for NPI {npi_str}."}


def list_cards(state=None, specialty=None, limit=None):
    state_err = _check_state(state)
    if state_err:
        return {"kind": "card_list", "found": False, "error": state_err}
    specialty_err = _check_specialty(specialty)
    if specialty_err:
        return {"kind": "card_list", "found": False, "error": specialty_err}

    pool = [i for i in card_idx
            if (not state or chunks[i]["state"] == state)
            and (not specialty or chunks[i]["specialty"] == specialty)]
    matched_chunks = [chunks[i] for i in pool]
    limited = matched_chunks if limit is None else matched_chunks[:limit]
    return {"kind": "card_list", "found": True, "count": len(matched_chunks), "chunks": limited}


def semantic_search(query_text, state=None, top_k=5):
    state_err = _check_state(state)
    if state_err:
        return {"kind": "semantic_search", "found": False, "error": state_err}

    pool = [i for i in narrative_idx if (not state or chunks[i]["state"] == state)]
    if not pool:
        return {"kind": "semantic_search", "found": False,
                "error": f"No narrative documents found for state={state}."}

    query_vector = model.encode([query_text])[0]
    query_vector = query_vector / np.linalg.norm(query_vector)
    scores = chunk_norms[pool] @ query_vector
    order = np.argsort(scores)[::-1][:top_k]

    results = [{"chunk": chunks[pool[o]], "score": float(scores[o])} for o in order]

    top_score = results[0]["score"] if results else 0.0
    if top_score >= MODERATE_SIMILARITY:
        confidence = "high"
    elif top_score >= MIN_SIMILARITY:
        confidence = "moderate"
    else:
        confidence = "none"
    low_confidence = confidence == "none"  # kept for backward-compatible found/not-found checks

    return {"kind": "semantic_search", "found": True,
            "confidence": confidence, "low_confidence": low_confidence, "results": results}


# =====================================================================
# Quick manual test - only runs if you execute this file directly.
# =====================================================================
if __name__ == "__main__":
    print("Known states:", states_all[:5], "...")
    print("Known specialties:", specialties_all)
    print()

    print("-- Exact NPI lookup --")
    result = lookup_card_by_npi("1000008396")
    print(result["found"], result.get("chunk", {}).get("chunk_id") if result["found"] else result.get("error"))
    print()

    print("-- List cards (Arkansas, Endocrinology) --")
    result = list_cards(state="Arkansas", specialty="Endocrinology", limit=3)
    print(f"found={result['found']}, count={result.get('count')}")
    print()

    print("-- Semantic search: 'What's our recommended messaging for competitive switchers?' --")
    result = semantic_search("What's our recommended messaging for competitive switchers?", top_k=5)
    print(f"confidence={result.get('confidence')}")
    for r in result.get("results", []):
        print(f"  [{r['score']:.3f}] {r['chunk']['chunk_id']} ({r['chunk']['doc_type']})")