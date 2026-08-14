# bm25_index.py
# -------------------------------------------------------------------
# Builds and loads the BM25 keyword-search index - the second half of
# hybrid search, alongside embedding_backend.py's meaning-based search.
# BM25 scores exact word/number/terminology overlap between a query and
# a chunk, which is exactly what the embedding model misses on cases
# like the Chen odds-ratio questions (see evals.html - "Retrieval
# ranking quality" / "Article retrieval ranking" - 4/10 and 4/12
# questions respectively didn't rank their correct chunk first under
# embeddings alone).
#
# Only covers the SAME narrative chunk pool search_documents.py's
# semantic_search() already searches over (HCP cards excluded). This
# mirrors the reasoning embedding_backend.py's FallbackBackend already
# documents for its own corpus choice: 15,000 near-identical HCP cards
# would swamp the corpus-wide term-rarity statistics BM25 depends on
# (its IDF weighting), making the handful of genuinely rare/important
# narrative terms look "common" by comparison just because some card
# template happens to repeat the same word 15,000 times over - the same
# failure mode already avoided there, for the same underlying reason.
#
# Scores are returned keyed by chunk_id, not by row position - so
# callers never have to trust that two arrays stayed aligned (the same
# principle embedding_ids.json already documents for the embedding
# store: "for verifying that alignment instead of trusting it").
#
# Build time (see 4_create_bm25_index.py):
#   build_bm25_index(chunk_ids, texts) -> saves output/bm25_index.pkl
# Query time (see search_documents.py):
#   load_bm25_index().get_scores(query_text) -> {chunk_id: score}
# -------------------------------------------------------------------

import os
import pickle
import re

_BM25_PATH = os.path.join("output", "bm25_index.pkl")

# Deliberately simple word-splitter: lowercase, then any run of letters
# or digits is one token. No stemming, no stopword removal. BM25's whole
# reason for being added here is to catch exact terms and numbers the
# embedding model glosses over (an odds ratio, a drug name, an NPI) - so
# quietly normalising or discarding tokens would undercut the point.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Wraps rank_bm25's BM25Okapi together with the chunk_ids it was
    built against, so every score can be looked up by chunk_id instead
    of trusted-by-position."""

    def __init__(self, chunk_ids, bm25):
        self.chunk_ids = chunk_ids
        self._bm25 = bm25

    def get_scores(self, query_text):
        """Returns {chunk_id: score} for every chunk in this index -
        same shape contract as looking up a cosine score per chunk_id,
        so search_documents.py can combine the two without caring how
        either one computed its number."""
        raw_scores = self._bm25.get_scores(_tokenize(query_text))
        return dict(zip(self.chunk_ids, raw_scores))


def build_bm25_index(chunk_ids, texts):
    """Build time: fit a BM25 index over the given chunk_ids/texts (same
    order, same length) and persist it to output/, so search_documents.py
    never has to refit it live on every query."""
    from rank_bm25 import BM25Okapi  # deferred import, same pattern as embedding_backend.py

    if len(chunk_ids) != len(texts):
        raise ValueError(
            f"chunk_ids ({len(chunk_ids)}) and texts ({len(texts)}) must be "
            f"the same length and in the same order."
        )

    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    index = BM25Index(list(chunk_ids), bm25)

    os.makedirs("output", exist_ok=True)
    with open(_BM25_PATH, "wb") as f:
        pickle.dump(index, f)
    return index


def load_bm25_index():
    """Query time: load the persisted BM25 index built above. Fails
    loudly (never silently returns an empty/stale index) if the build
    step hasn't been run yet - same principle as embedding_backend.py's
    get_query_backend() never guessing which model built the store."""
    if not os.path.exists(_BM25_PATH):
        raise FileNotFoundError(
            f"{_BM25_PATH} not found - run 4_create_bm25_index.py before "
            f"calling search."
        )
    with open(_BM25_PATH, "rb") as f:
        return pickle.load(f)


# =====================================================================
# Quick manual test - only runs if you execute this file directly.
# =====================================================================
if __name__ == "__main__":
    index = load_bm25_index()
    print(f"Loaded BM25 index: {len(index.chunk_ids)} chunks.")
    print()
    print("-- BM25 scores: 'pooled odds ratio composite kidney outcome' --")
    scores = index.get_scores("pooled odds ratio composite kidney outcome")
    top5 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for chunk_id, score in top5:
        print(f"  [{score:.3f}] {chunk_id}")
