# search.py
# -------------------------------------------------------------------
# Answers a question by ROUTING it to the right retrieval path:
#   1. Question names an NPI  -> exact card lookup (no fuzzy matching).
#   2. Question names a state -> semantic search within that state's docs.
#   3. Otherwise              -> semantic search over the narrative docs.
# Cards are never returned by fuzzy search (they'd pollute results);
# they come back only via exact NPI lookup.
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

# Pre-build a few helpers.
narrative_idx = [i for i, c in enumerate(chunks) if c["doc_type"] != "hcp_card"]
card_by_npi = {c["npi"]: i for i, c in enumerate(chunks) if c["doc_type"] == "hcp_card"}
states = sorted({c["state"] for c in chunks if c["state"] and c["doc_type"] != "hcp_card"})

def _semantic(question, pool, top_k):
    q = model.encode([question])[0]
    q = q / np.linalg.norm(q)
    scores = chunk_norms[pool] @ q
    order = np.argsort(scores)[::-1][:top_k]
    return [(chunks[pool[o]], float(scores[o])) for o in order]

def search(question, top_k=3):
    # PATH 1: exact NPI lookup
    m = re.search(r"\b(\d{10})\b", question)
    if m and m.group(1) in card_by_npi:
        i = card_by_npi[m.group(1)]
        return "card lookup (NPI)", [(chunks[i], 1.0)]

    # PATH 2: state filter (over narrative docs)
    ql = question.lower()
    picked = next((s for s in states if s.lower() in ql), None)
    if picked:
        pool = [i for i in narrative_idx if chunks[i]["state"] == picked]
        if pool:
            return f"state filter ({picked})", _semantic(question, pool, top_k)

    # PATH 3: general semantic search over narrative docs
    return "semantic (narrative)", _semantic(question, narrative_idx, top_k)


# --- Try a few questions, showing which path each took ---
questions = [
    "Tell me about the HCP with NPI 1000008396",
    "What does the GLP-1 market look like in Missouri?",
    "How is Rabivy different from Zepbound?",
]
for question in questions:
    route, results = search(question, top_k=3)
    print("=" * 72)
    print(f"Q: {question}")
    print(f"   route -> {route}")
    for chunk, score in results:
        print(f"   [{score:.3f}] {chunk['chunk_id']}  ({chunk['doc_type']})")
    print()