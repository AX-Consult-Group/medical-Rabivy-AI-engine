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
# HYBRID SEARCH (added alongside the keyword-search branch work):
# semantic_search() now runs TWO rankings and fuses them, instead of
# meaning-based search alone:
#   1. The embedding ranking (as before - cosine similarity via
#      embedding_backend.py).
#   2. A BM25 keyword ranking (bm25_index.py) - exact word/number/
#      terminology overlap, which the embedding model can systematically
#      miss (see evals.html - "Retrieval ranking quality" / "Article
#      retrieval ranking": 4/10 and 4/12 questions respectively didn't
#      rank their correct chunk first under embeddings alone).
# The two rankings are combined via Reciprocal Rank Fusion (RRF): each
# chunk's RANK POSITION in each list is what matters, not its raw score
# - this sidesteps the fact that BM25 scores and cosine scores are on
# completely different, non-comparable scales. See _fuse_rankings()
# below.
# IMPORTANT: fusion only changes WHICH chunks surface and in WHAT ORDER.
# It does NOT change what a result's "score" means downstream - that's
# still the plain embedding cosine score, exactly as before, so the
# CONFIDENCE TIERS below (already calibrated against real embedding
# scores) don't need to be re-calibrated against a fused number nobody's
# validated thresholds for. Every result also carries "bm25_score" and
# "rrf_score" now, purely informational - callers that only look at
# "score"/"confidence" (ask_a_question.py, agent_tools.py) see no
# change in what those fields mean, only better chunks in a better order.
#
# semantic_search() takes an optional use_bm25=False to bypass fusion
# and return pure embedding-only results, for controlled before/after
# comparison - every existing caller keeps getting hybrid search by
# default without needing to change anything.
#
# STEP 6 - CROSS-ENCODER RERANKER (reranker_backend.py):
# On top of the RRF-fused shortlist, the top RERANK_POOL candidates are
# re-scored by a cross-encoder - a model that reads the query and a
# candidate chunk TOGETHER (unlike the embedding/BM25 scores, which each
# score the query and a chunk independently, which is what makes them
# cheap enough to run over the whole corpus). The cross-encoder is more
# accurate but too slow for that, which is exactly why it only re-scores
# the short RRF shortlist rather than the whole pool. See
# reranker_backend.py's header comment for the full explanation.
# Every result now also carries "rerank_score" - None if the reranker
# isn't available (same loud-never-silent degrade pattern as BM25: the
# pipeline keeps working off the plain RRF-fused order). Like bm25_score
# and rrf_score, rerank_score is purely informational for "score"/
# "confidence" - those still mean exactly what they meant before.
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
# tier above for the fix. Still computed off the top fused result's
# embedding score - see the hybrid-search note above for why.
# =====================================================================

import json
import numpy as np
from embedding_backend import get_query_backend
from bm25_index import load_bm25_index
from reranker_backend import get_reranker, RERANK_POOL

with open("output/chunks_tagged.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
embeddings = np.load("output/embeddings.npy")
# Load whichever backend embedding_meta.json says built the store
# (MiniLM normally; the offline TF-IDF/LSA fallback in sandboxes).
model = get_query_backend()

# Load the BM25 index the same loud-never-silent way embedding_backend.py
# handles a missing MiniLM model: if 4_create_bm25_index.py hasn't been
# run yet, don't crash every single caller of this file (the router, the
# agent, every eval script) just because hybrid search can't run yet -
# say so clearly and degrade to embedding-only search until it's built.
try:
    bm25_idx = load_bm25_index()
except FileNotFoundError as e:
    print(f"! BM25 index unavailable ({e})")
    print("! Falling back to embedding-only search until 4_create_bm25_index.py is run.")
    bm25_idx = None

# Load the cross-encoder reranker once here (not per-query - reloading
# a model on every search would be needlessly slow). get_reranker()
# never raises - it prints its own warning and returns None if the
# model can't be loaded, same loud-never-silent pattern as BM25 above.
reranker = get_reranker()

chunk_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

card_idx = [i for i, c in enumerate(chunks) if c["doc_type"] == "hcp_card"]
narrative_idx = [i for i, c in enumerate(chunks) if c["doc_type"] != "hcp_card"]
card_by_npi = {c["npi"]: i for i, c in enumerate(chunks) if c["doc_type"] == "hcp_card"}
states_all = sorted({c["state"] for c in chunks if c["state"]})
specialties_all = sorted({c["specialty"] for c in chunks if c["specialty"]})

MIN_SIMILARITY = 0.13
MODERATE_SIMILARITY = 0.30

# Standard default from the original RRF paper (Cormack, Clarke & Buettcher
# 2009) - not tuned against this project's own data, since RRF's whole
# appeal is that it's stable across a wide range of k without per-corpus
# tuning. Revisit only if fused rankings look obviously off in practice.
RRF_K = 60


def _fuse_rankings(pool, embedding_scores, bm25_scores_by_chunk_id):
    """Reciprocal Rank Fusion: combines the embedding ranking and the
    BM25 ranking over the same `pool` (a list of global chunk indices)
    using each chunk's RANK POSITION in each list, not its raw score -
    so a 0-1 cosine score and an unbounded BM25 score never need to be
    normalised against each other to be compared.

    Returns (fused_order, bm25_scores_for_pool) - fused_order is a list
    of positions into `pool`, best first; bm25_scores_for_pool is a
    plain array aligned to `pool`, for callers that want to report the
    BM25 score alongside the embedding score."""
    n = len(pool)
    bm25_scores = np.array([
        bm25_scores_by_chunk_id.get(chunks[pool[i]]["chunk_id"], 0.0)
        for i in range(n)
    ])

    embed_order = np.argsort(embedding_scores)[::-1]
    bm25_order = np.argsort(bm25_scores)[::-1]

    # rank[i] = 1-indexed position of pool-index i in that ranking.
    embed_rank = np.empty(n, dtype=int)
    embed_rank[embed_order] = np.arange(1, n + 1)
    bm25_rank = np.empty(n, dtype=int)
    bm25_rank[bm25_order] = np.arange(1, n + 1)

    rrf_scores = 1.0 / (RRF_K + embed_rank) + 1.0 / (RRF_K + bm25_rank)
    fused_order = np.argsort(rrf_scores)[::-1]
    return fused_order, bm25_scores, rrf_scores


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


def semantic_search(query_text, state=None, top_k=5, use_bm25=True):
    """use_bm25=True (default): fuse embedding + BM25 rankings (hybrid
    search). use_bm25=False: pure embedding-only search, the original
    behaviour - kept for controlled before/after comparison, not
    expected to be passed by production callers."""
    state_err = _check_state(state)
    if state_err:
        return {"kind": "semantic_search", "found": False, "error": state_err}

    pool = [i for i in narrative_idx if (not state or chunks[i]["state"] == state)]
    if not pool:
        return {"kind": "semantic_search", "found": False,
                "error": f"No narrative documents found for state={state}."}

    query_vector = model.encode([query_text])[0]
    query_vector = query_vector / np.linalg.norm(query_vector)
    embedding_scores = chunk_norms[pool] @ query_vector

    if use_bm25 and bm25_idx is not None:
        bm25_query_scores = bm25_idx.get_scores(query_text)
        fused_order, bm25_scores, rrf_scores = _fuse_rankings(
            pool, embedding_scores, bm25_query_scores)

        if reranker is not None:
            # Re-score only the top RERANK_POOL fused candidates (the
            # expensive step - see reranker_backend.py for why it's
            # kept short), then re-sort just that short list by the
            # cross-encoder's score before taking the final top_k.
            rerank_pool = fused_order[:RERANK_POOL]
            candidate_texts = [chunks[pool[o]]["text"] for o in rerank_pool]
            rerank_scores = reranker.score(query_text, candidate_texts)
            pool_order = np.argsort(rerank_scores)[::-1]
            final_order = [rerank_pool[i] for i in pool_order][:top_k]
            rerank_score_by_pos = {rerank_pool[i]: float(rerank_scores[i])
                                    for i in range(len(rerank_pool))}
            results = [{
                "chunk": chunks[pool[o]],
                "score": float(embedding_scores[o]),   # unchanged meaning - embedding cosine
                "bm25_score": float(bm25_scores[o]),    # informational
                "rrf_score": float(rrf_scores[o]),       # informational - fed the reranker its shortlist
                "rerank_score": rerank_score_by_pos[o],   # what actually ordered this final list
            } for o in final_order]
        else:
            # Reranker unavailable - serve the plain RRF-fused order,
            # exactly as hybrid search worked before step 6.
            order = fused_order[:top_k]
            results = [{
                "chunk": chunks[pool[o]],
                "score": float(embedding_scores[o]),
                "bm25_score": float(bm25_scores[o]),
                "rrf_score": float(rrf_scores[o]),
                "rerank_score": None,
            } for o in order]
    else:
        # Embedding-only path - identical to this function's pre-hybrid
        # behaviour, used when BM25 isn't built yet or a caller
        # explicitly asked for it (use_bm25=False). Deliberately never
        # reranked either, so it stays a clean, unmodified baseline for
        # before/after comparison.
        order = np.argsort(embedding_scores)[::-1][:top_k]
        results = [{"chunk": chunks[pool[o]], "score": float(embedding_scores[o])} for o in order]

    # Confidence is always read off the top result's embedding score,
    # never the RRF score - see the hybrid-search note at the top of
    # this file for why (RRF scores aren't on the calibrated scale
    # MIN_SIMILARITY/MODERATE_SIMILARITY were set against).
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
        extra = f" | bm25={r['bm25_score']:.2f} rrf={r['rrf_score']:.4f}" if "rrf_score" in r else ""
        if r.get("rerank_score") is not None:
            extra += f" rerank={r['rerank_score']:.3f}"
        print(f"  [{r['score']:.3f}] {r['chunk']['chunk_id']} ({r['chunk']['doc_type']}){extra}")

    print()
    print("-- Hybrid vs. embedding-only, same query (a case embeddings alone missed) --")
    q = "What did excluding exendin-4-backbone agents do to the kidney-outcome odds ratio, per Chen et al.?"
    hybrid = semantic_search(q, top_k=5, use_bm25=True)
    embed_only = semantic_search(q, top_k=5, use_bm25=False)
    print("Hybrid top 5:")
    for r in hybrid["results"]:
        print(f"  [{r['score']:.3f}] {r['chunk']['chunk_id']}")
    print("Embedding-only top 5:")
    for r in embed_only["results"]:
        print(f"  [{r['score']:.3f}] {r['chunk']['chunk_id']}")