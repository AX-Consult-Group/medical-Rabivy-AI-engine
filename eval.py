# eval.py  (v2: state-aware search + fairer, honest ground truth)
# -------------------------------------------------------------------
# Improvements over v1:
#  1. If a question names a state, we FILTER to that state's chunks before
#     ranking (this is what the `state` tag from tag.py is for).
#  2. Ground truth allows a LIST of acceptable chunks per question, because
#     several chunks can legitimately answer one question.
# -------------------------------------------------------------------

import json
import numpy as np
from sentence_transformers import SentenceTransformer

with open("output/chunks_tagged.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
embeddings = np.load("output/embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# All state names we know about (taken from the tags themselves).
states = sorted({c["state"] for c in chunks if c["state"]})

def detect_state(question):
    ql = question.lower()
    for s in states:
        if s.lower() in ql:
            return s
    return None

def search(question, top_k=3):
    # If the question names a state, only consider that state's chunks.
    picked = detect_state(question)
    if picked:
        idxs = [i for i, c in enumerate(chunks) if c["state"] == picked]
    else:
        idxs = list(range(len(chunks)))

    q = model.encode([question])[0]
    q = q / np.linalg.norm(q)
    scores = chunk_norms[idxs] @ q            # rank only the allowed chunks
    order = np.argsort(scores)[::-1][:top_k]
    return [(chunks[idxs[o]]["chunk_id"], float(scores[o])) for o in order]


# Ground truth: each question -> LIST of chunk-id substrings that genuinely answer it.
ground_truth = {
    "What does the GLP-1 market look like in Missouri?":        ["state_market_summary__missouri"],
    "A doctor says they're happy with Ozempic. What do I say?": ["objection_handling_guide"],
    "How is Rabivy's mechanism different from tirzepatide?":    ["mechanism_comparison", "molecule_and_mechanism", "how_is_this_different_from_ti"],
    "Why do patients stop taking GLP-1s after a year?":         ["why_patients_discontinue", "persistence"],
    "What is Rabivy's main dosing advantage?":                  ["monthly_dosing", "where_rabivy_wins", "positioning_summary"],
    "What does a typical endocrinologist look like?":           ["endocrinology"],
    "How is prior authorization affecting access?":             ["prior_auth", "access"],
    "What are the talking points for competitive switchers?":   ["competitive_switchers"],
}

TOP_K = 3
hits = 0
print(f"Evaluating {len(ground_truth)} questions (checking top {TOP_K})\n")
for question, acceptable in ground_truth.items():
    returned = [cid for cid, _ in search(question, top_k=TOP_K)]
    found = any(any(a in cid for a in acceptable) for cid in returned)
    hits += found
    print(f"[{'PASS' if found else 'FAIL'}] {question}")
    print(f"        top {TOP_K}: {[c.split('__')[-1][:28] for c in returned]}")

print("\n" + "=" * 60)
print(f"HIT RATE: {hits}/{len(ground_truth)} in the top {TOP_K}.")