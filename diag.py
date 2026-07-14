# diag.py
# -------------------------------------------------------------------
# Diagnostic: for a given question, find where a KNOWN-correct chunk
# actually ranks in similarity search - and how far behind the top
# result it fell. Used to tell a chunking/wording miss (close rank,
# small gap) apart from a content gap (chunk barely appears at all,
# or the expected chunk_id isn't in the pool searched).
#
# Ground truth is an explicit chunk_id, not a keyword guess, so this
# is repeatable across questions rather than a one-off script edited
# each time. Add new (question, expected_chunk_id) pairs to
# TEST_CASES below to build this into a running regression list.
#
# NOTE: if eval.py already tracks a (question -> expected chunk) set
# for its hit-rate check, this probably belongs merged into that
# rather than living as a second, separate eval mechanism.
# -------------------------------------------------------------------

import json
import numpy as np
from sentence_transformers import SentenceTransformer

chunks = json.load(open("output/chunks_tagged.json"))
embeddings = np.load("output/embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Default pool is EVERYTHING (cards + narrative) - a diagnostic that
# silently restricts to one doc_type could mask the real bug being
# that the correct content lives somewhere else entirely.
ALL_IDX = list(range(len(chunks)))
narrative_idx = [i for i, c in enumerate(chunks) if c["doc_type"] != "hcp_card"]
card_idx = [i for i, c in enumerate(chunks) if c["doc_type"] == "hcp_card"]


def rank_check(question, expected_chunk_id, pool_idx=None, top_k=8):
    """Report where `expected_chunk_id` ranks for `question`, plus the
    similarity gap between it and the top-ranked result.

    pool_idx: restrict the search pool (e.g. narrative_idx) if you
    deliberately want to test routing within one doc_type. Defaults
    to ALL_IDX so nothing is silently excluded.
    """
    pool = pool_idx if pool_idx is not None else ALL_IDX

    q = model.encode([question])[0]
    q = q / np.linalg.norm(q)
    scores = chunk_norms[pool] @ q
    order = np.argsort(scores)[::-1]

    print(f"Q: {question}")
    print(f"Expected chunk_id: {expected_chunk_id}")
    print(f"Pool size: {len(pool)} chunks\n")

    print(f"Top {top_k} by score:")
    for rank, o in enumerate(order[:top_k], 1):
        c = chunks[pool[o]]
        flag = "  <-- expected" if c["chunk_id"] == expected_chunk_id else ""
        print(f"  {rank:2d}. [{scores[o]:.3f}] {c['chunk_id']}{flag}")

    top_score = float(scores[order[0]])

    expected_rank, expected_score = None, None
    for rank, o in enumerate(order, 1):
        if chunks[pool[o]]["chunk_id"] == expected_chunk_id:
            expected_rank, expected_score = rank, float(scores[o])
            break

    print()
    if expected_rank is None:
        print(f"! '{expected_chunk_id}' not found in this pool at all - "
              f"check the chunk_id is correct and actually in the pool searched.")
        return {"question": question, "expected_chunk_id": expected_chunk_id,
                 "found": False, "rank": None, "gap": None}

    gap = top_score - expected_score
    verdict = "close miss - likely chunking/wording" if gap < 0.1 else "large miss - check content/doc coverage"
    print(f"Expected chunk rank: {expected_rank} of {len(pool)}  [{expected_score:.3f}]")
    print(f"Gap vs top result:   {gap:.3f}  ({verdict})")
    print()

    return {"question": question, "expected_chunk_id": expected_chunk_id,
            "found": True, "rank": expected_rank, "gap": gap}


# Add cases here as you find misses - this is the seed of a repeatable
# regression list, not a one-off per-question edit. Fill in real
# chunk_ids from your chunk_manifest.json / chunks_tagged.json.
TEST_CASES = [
    ("Why do patients stop taking GLP-1s after a year?", "clinical__discontinuation_and_persistence"),
]

if __name__ == "__main__":
    results = [rank_check(q, eid) for q, eid in TEST_CASES]
    print("=" * 60)
    n_found = sum(1 for r in results if r["found"])
    print(f"{n_found}/{len(results)} expected chunks found in search pool.")
    ranks = [r["rank"] for r in results if r["found"]]
    if ranks:
        print(f"Ranks: {ranks}")