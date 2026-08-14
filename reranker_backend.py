# reranker_backend.py
# -------------------------------------------------------------------
# Step 6 of hybrid search: a CROSS-ENCODER reranker, run on top of the
# RRF-fused shortlist that embedding_backend.py + bm25_index.py already
# produce (see search_documents.py's header comment for the full hybrid
# picture: embedding ranking + BM25 ranking, fused by rank).
#
# WHY THIS IS SEPARATE FROM THE OTHER TWO:
# Embeddings and BM25 are both "bi-encoders" - the query and a chunk are
# each turned into a number/vector INDEPENDENTLY, which is what makes
# them cheap enough to run against the whole corpus. A cross-encoder is
# the opposite: it reads the query and ONE candidate chunk TOGETHER, in
# the same pass, and outputs a single relevance number for that pair.
# That joint reading is much more accurate, but it's also far more
# expensive - you must run it once per candidate - so it is only ever
# used to re-score a SHORT shortlist an earlier, cheaper stage already
# narrowed down. This two-stage "retrieve cheap, then rerank precisely"
# design is standard practice; see Nogueira & Cho, "Passage Re-ranking
# with BERT" (2019) for the original description of this exact pattern.
#
# Model: cross-encoder/ms-marco-MiniLM-L-6-v2 - same MiniLM family
# already used for embeddings, via sentence-transformers' CrossEncoder
# class (no new dependency - sentence-transformers already ships this).
# Trained specifically for passage reranking on the MS MARCO dataset,
# a standard, well-established small/fast choice for this exact job.
#
# RERANK_POOL = how many of the RRF-fused candidates get sent through
# the (expensive) cross-encoder. Kept short on purpose - reranking the
# whole corpus would defeat the point of having a cheap first stage.
#
# Same "loud, never silent" pattern as embedding_backend.py and
# bm25_index.py: if the model can't be loaded (no internet, e.g. this
# project's cloud sandbox has no path to Hugging Face), print a clear
# warning and let search_documents.py fall back to serving the plain
# RRF-fused order - reranking is an ENHANCEMENT on top of hybrid
# search, never a requirement for hybrid search to keep working.
# -------------------------------------------------------------------

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = 20


class CrossEncoderReranker:
    """Reads (query, chunk_text) pairs together and scores relevance
    directly - see the module docstring above for why that's different
    from, and more accurate than, the embedding/BM25 scores."""

    name = RERANKER_NAME

    def __init__(self):
        from sentence_transformers import CrossEncoder  # deferred import, same pattern as embedding_backend.py
        self._model = CrossEncoder(RERANKER_NAME)

    def score(self, query_text, texts):
        """Returns one relevance score per text in `texts`, same order.
        Not comparable to embedding cosine or BM25 scores - a different
        scale entirely, same principle as those two already documented
        in search_documents.py."""
        pairs = [[query_text, t] for t in texts]
        return self._model.predict(pairs)


def get_reranker():
    """Loads the cross-encoder once at import time (see search_documents.py),
    not per-query - reloading a model on every search would be needlessly
    slow. Never raises: on failure, prints a clear warning and returns
    None, exactly like embedding_backend.py's get_build_backend() does
    for MiniLM - a missing reranker degrades hybrid search back to plain
    RRF-fused order, it never breaks it."""
    try:
        model = CrossEncoderReranker()
        print(f"Reranker in use: {model.name}")
        return model
    except Exception as e:
        print(f"! Cross-encoder reranker unavailable ({type(e).__name__}: {str(e)[:120]})")
        print("! Skipping the rerank step - results stay in RRF-fused order.")
        return None


# =====================================================================
# Quick manual test - only runs if you execute this file directly.
# =====================================================================
if __name__ == "__main__":
    reranker = get_reranker()
    if reranker is None:
        print("(Reranker unavailable in this environment - nothing more to test here.)")
    else:
        q = "Why do patients stop taking GLP-1s after a year?"
        texts = [
            "Real-world discontinuation is driven primarily by side effects and cost.",
            "Coverage, prior authorization, and step therapy determine whether the patient starts and stays on treatment.",
        ]
        scores = reranker.score(q, texts)
        for t, s in zip(texts, scores):
            print(f"  [{s:.3f}] {t[:70]}")
