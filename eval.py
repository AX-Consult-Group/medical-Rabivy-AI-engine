# eval.py  (full system: routes through NPI lookup, state filter, semantic)
# -------------------------------------------------------------------
# Uses the SAME router as search.py, then checks each question's result
# against a ground-truth list of chunk-id substrings that should answer it.
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

narrative_idx = [i for i, c in enumerate(chunks) if c["doc_type"] != "hcp_card"]
card_by_npi = {c["npi"]: i for i, c in enumerate(chunks) if c["doc_type"] == "hcp_card"}
states = sorted({c["state"] for c in chunks if c["state"] and c["doc_type"] != "hcp_card"})

def _semantic(question, pool, top_k):
    q = model.encode([question])[0]
    q = q / np.linalg.norm(q)
    scores = chunk_norms[pool] @ q
    order = np.argsort(scores)[::-1][:top_k]
    return [chunks[pool[o]]["chunk_id"] for o in order]

def search_ids(question, top_k=3):
    m = re.search(r"\b(\d{10})\b", question)
    if m and m.group(1) in card_by_npi:
        return [chunks[card_by_npi[m.group(1)]]["chunk_id"]]
    ql = question.lower()
    picked = next((s for s in states if s.lower() in ql), None)
    if picked:
        pool = [i for i in narrative_idx if chunks[i]["state"] == picked]
        if pool:
            return _semantic(question, pool, top_k)
    return _semantic(question, narrative_idx, top_k)


# Ground truth: question -> acceptable chunk-id substrings.
ground_truth = {
    "Tell me about the HCP with NPI 1000008396":                ["card_1000008396"],
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
print(f"Evaluating {len(ground_truth)} questions (top {TOP_K})\n")
for question, acceptable in ground_truth.items():
    returned = search_ids(question, top_k=TOP_K)
    found = any(any(a in cid for a in acceptable) for cid in returned)
    hits += found
    print(f"[{'PASS' if found else 'FAIL'}] {question}")
    print(f"        top: {[c.split('__')[-1][:26] for c in returned]}")

print("\n" + "=" * 60)
print(f"HIT RATE: {hits}/{len(ground_truth)} in the top {TOP_K}.")