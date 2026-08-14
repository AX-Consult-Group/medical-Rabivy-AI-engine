# embedding_backend.py
# -------------------------------------------------------------------
# One place that decides HOW text becomes a vector, so the build step
# (3_create_embeddings.py) and the search step (search_documents.py)
# can never drift apart on that decision.
#
# The desired model : sentence-transformers all-MiniLM-L6-v2 (unchanged
#                   from the original design). This is the real,
#                   preferred embedding model - used whenever it's
#                   available locally or downloadable. It is NOT itself
#                   a "backend" in this file's terminology below - it's
#                   the thing this whole pipeline actually wants to run.
# The fallback backend: TF-IDF + truncated SVD (LSA), fully offline. This
#                   IS "the backend" - an emergency stand-in, used
#                   automatically only when the desired model can't be
#                   loaded (e.g. sandboxed environments with no
#                   HuggingFace access). The fitted vectorizer is
#                   persisted to output/ so queries at search time are
#                   encoded in the SAME vector space as the chunks were.
#
# The desired model and the fallback backend expose the same two-method
# interface, so nothing downstream needs to know which one actually ran:
#   fit_encode(texts) -> np.ndarray   (build time: may fit, then encode)
#   encode(texts)     -> np.ndarray   (query time: encode only)
# embedding_meta.json records which one produced the vectors, and
# search_documents.py loads the matching one - a MiniLM index is never
# silently queried with TF-IDF vectors or vice versa.
# -------------------------------------------------------------------

import json
import os

import numpy as np

MINILM_NAME = "all-MiniLM-L6-v2"
FALLBACK_NAME = "tfidf-lsa-fallback"
_FALLBACK_PATH = os.path.join("output", "tfidf_backend.joblib")
_FALLBACK_DIM = 384  # same dimensionality as MiniLM, for a like-for-like store


class MiniLMModel:
    """The real, desired embedding model - not a "backend" or stand-in,
    this is the thing the pipeline is actually meant to run."""

    name = MINILM_NAME

    def __init__(self):
        from sentence_transformers import SentenceTransformer  # deferred import
        self._model = SentenceTransformer(MINILM_NAME)

    def fit_encode(self, texts):
        return self._model.encode(texts, show_progress_bar=True)

    def encode(self, texts):
        return self._model.encode(texts)


class FallbackBackend:
    """Offline stand-in: TF-IDF over word 1-2 grams, compressed to a
    dense vector by SPARSE RANDOM PROJECTION, L2-normalised.

    Random projection (not truncated SVD/LSA) is a deliberate choice:
    it approximately PRESERVES TF-IDF cosine similarity for every
    document - including ones whose distinctive terms are rare. SVD
    optimises for corpus-wide variance, so with 15,000 near-identical
    HCP cards dominating the corpus it crushed small novel documents
    (freshly ingested intel items) into generic directions and made
    them unretrievable; all pairwise similarities collapsed above 0.9.
    Johnson-Lindenstrauss projection has no such bias. min_df=1 for the
    same reason: a new item is often the only document containing its
    key terms, and those terms must stay in the vocabulary."""

    name = FALLBACK_NAME

    def __init__(self, fitted=None):
        self._pipeline = fitted  # (vectorizer, projector) once fitted

    def fit_encode(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.random_projection import SparseRandomProjection
        import joblib

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                     min_df=1, strip_accents="unicode")
        tfidf = _sk_normalize(vectorizer.fit_transform(texts))
        projector = SparseRandomProjection(n_components=_FALLBACK_DIM,
                                           dense_output=True, random_state=0)
        vecs = projector.fit_transform(tfidf)
        self._pipeline = (vectorizer, projector)
        os.makedirs("output", exist_ok=True)
        joblib.dump(self._pipeline, _FALLBACK_PATH)
        return self._l2(vecs)

    def encode(self, texts):
        if self._pipeline is None:
            import joblib
            self._pipeline = joblib.load(_FALLBACK_PATH)
        vectorizer, projector = self._pipeline
        return self._l2(projector.transform(_sk_normalize(vectorizer.transform(texts))))

    @staticmethod
    def _l2(vecs):
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def _sk_normalize(sparse_matrix):
    from sklearn.preprocessing import normalize
    return normalize(sparse_matrix)


def get_build_backend():
    """Build time: prefer the real MiniLM model, fall back to the
    TF-IDF/LSA backend with a loud, explicit message (never a silent
    substitution)."""
    try:
        model = MiniLMModel()
        print(f"Embedding model in use: {model.name}")
        return model
    except Exception as e:
        print(f"! MiniLM unavailable ({type(e).__name__}: {str(e)[:120]})")
        print(f"! Falling back to offline backend: {FALLBACK_NAME}")
        return FallbackBackend()


def get_query_backend():
    """Search time: load whichever one (real model or fallback backend)
    embedding_meta.json says actually built the vector store - never
    guess."""
    with open(os.path.join("output", "embedding_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    model_name = meta.get("model", MINILM_NAME)
    if model_name == FALLBACK_NAME:
        return FallbackBackend()
    return MiniLMModel()
